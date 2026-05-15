from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TapConfig:
    """Layers are 1-indexed: layer=1 means output AFTER resblocks[0]."""
    vision_layers: List[int]
    text_layers: List[int]


@dataclass
class DistillOutputs:
    vision_layers: List[torch.Tensor]
    text_layers: List[torch.Tensor]
    image_features: torch.Tensor
    text_features: torch.Tensor
    similarity_logits: torch.Tensor


def infer_last_gene_layers(total_layers: int, gene_layers: int) -> List[int]:
    """Return 1-indexed layer ids for a stable last-N gene selection."""

    total_layers = int(total_layers)
    gene_layers = max(0, min(int(gene_layers), total_layers))
    if total_layers <= 0 or gene_layers <= 0:
        return []
    return list(range(total_layers - gene_layers + 1, total_layers + 1))


def compute_remaining_layer_ids(total_layers: int, gene_layer_ids: Sequence[int]) -> List[int]:
    """Return 1-indexed non-gene teacher layers before the first inherited gene layer."""

    total_layers = int(total_layers)
    gene_ids = sorted({int(x) for x in gene_layer_ids if int(x) > 0})
    if total_layers <= 0:
        return []
    if not gene_ids:
        return list(range(1, total_layers + 1))

    first_gene = min(gene_ids)
    boundary = max(1, min(first_gene, total_layers + 1))
    return [idx for idx in range(1, boundary) if idx not in gene_ids]


def compute_remaining_tap_layers(
    total_layers: int,
    gene_layer_ids: Sequence[int],
    num_taps: int = 3,
) -> List[int]:
    """Pick evenly spaced 1-indexed distillation taps from non-gene remaining layers."""

    remaining = compute_remaining_layer_ids(total_layers, gene_layer_ids)
    if not remaining or num_taps <= 0:
        return []

    max_remaining = max(remaining)
    raw_taps = [
        max(1, min(max_remaining, math.floor(max_remaining * k / num_taps)))
        for k in range(1, num_taps + 1)
    ]
    available = set(remaining)
    taps: List[int] = []
    for tap in raw_taps:
        if tap in available:
            taps.append(tap)
            continue
        lower = [idx for idx in remaining if idx <= tap]
        if lower:
            taps.append(lower[-1])
    return sorted(set(taps))


def _unique_sorted(xs: List[int]) -> List[int]:
    return sorted(list({int(x) for x in xs}))


class CLIPTeacherTaps(nn.Module):
    """OpenAI-CLIP teacher returning intermediate token sequences (frozen)."""

    def __init__(
        self,
        clip_model: nn.Module,
        tap_cfg: TapConfig,
        device: torch.device,
        teacher_fp16: bool = True,
    ):
        super().__init__()
        self.model = clip_model
        self.tap_cfg = TapConfig(
            vision_layers=_unique_sorted(tap_cfg.vision_layers),
            text_layers=_unique_sorted(tap_cfg.text_layers),
        )

        self.model.eval()
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        if teacher_fp16 and device.type in {"cuda", "npu"}:
            # Prefer fp16 to reduce overhead for taps; loss will be computed in fp32.
            try:
                self.model.half()
            except Exception:
                pass

    @torch.no_grad()
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return ([vision_tap_tokens...], [text_tap_tokens...]), each token seq is [seq,B,width]."""
        v = self._forward_visual_taps(images)
        t = self._forward_text_taps(tokens)
        return v, t

    def _teacher_dtype_device(self) -> Tuple[torch.dtype, torch.device]:
        p = next(self.model.parameters())
        return p.dtype, p.device

    def _forward_visual_taps(self, images: torch.Tensor) -> List[torch.Tensor]:
        m = self.model
        visual = m.visual

        dtype, device = self._teacher_dtype_device()
        images = images.to(device=device, dtype=dtype, non_blocking=True)

        # CLIP ViT: conv -> (B,grid^2,width) -> add CLS/pos -> ln_pre -> (seq,B,width)
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,grid^2,width]

        cls = visual.class_embedding.to(x.dtype)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # [seq,B,width]

        want = set(self.tap_cfg.vision_layers)
        taps: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(visual.transformer.resblocks):
            x = blk(x)
            layer = i + 1
            if layer in want:
                taps[layer] = x

        return [taps[k] for k in self.tap_cfg.vision_layers]

    def _forward_text_taps(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        m = self.model
        dtype, device = self._teacher_dtype_device()
        tokens = tokens.to(device=device, non_blocking=True)

        x = m.token_embedding(tokens).to(dtype)
        x = x + m.positional_embedding.to(dtype)
        x = x.permute(1, 0, 2)  # [L,B,width]

        want = set(self.tap_cfg.text_layers)
        taps: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(m.transformer.resblocks):
            x = blk(x)
            layer = i + 1
            if layer in want:
                taps[layer] = x

        return [taps[k] for k in self.tap_cfg.text_layers]


class CLIPTeacherDistiller(CLIPTeacherTaps):
    """Frozen CLIP teacher returning tap features plus final similarity signals."""

    def _forward_visual_taps_and_features(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        m = self.model
        visual = m.visual

        dtype, device = self._teacher_dtype_device()
        images = images.to(device=device, dtype=dtype, non_blocking=True)

        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        want = set(self.tap_cfg.vision_layers)
        taps: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(visual.transformer.resblocks):
            x = blk(x)
            layer = i + 1
            if layer in want:
                taps[layer] = x

        pooled = x.permute(1, 0, 2)[:, 0, :]
        pooled = visual.ln_post(pooled)
        if getattr(visual, "proj", None) is not None:
            pooled = pooled @ visual.proj
        return [taps[k] for k in self.tap_cfg.vision_layers], pooled

    def _forward_text_taps_and_features(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        m = self.model
        dtype, device = self._teacher_dtype_device()
        tokens = tokens.to(device=device, non_blocking=True)

        x = m.token_embedding(tokens).to(dtype)
        x = x + m.positional_embedding.to(dtype)
        x = x.permute(1, 0, 2)

        want = set(self.tap_cfg.text_layers)
        taps: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(m.transformer.resblocks):
            x = blk(x)
            layer = i + 1
            if layer in want:
                taps[layer] = x

        x = x.permute(1, 0, 2)
        x = m.ln_final(x).to(dtype)
        pooled = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        pooled = pooled @ m.text_projection
        return [taps[k] for k in self.tap_cfg.text_layers], pooled

    @torch.no_grad()
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> DistillOutputs:
        dtype, device = self._teacher_dtype_device()
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        tokens = tokens.to(device=device, non_blocking=True)

        vision_layers, image_features = self._forward_visual_taps_and_features(images)
        text_layers, text_features = self._forward_text_taps_and_features(tokens)
        image_features = image_features.float()
        text_features = text_features.float()
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = getattr(self.model, "logit_scale", None)
        if logit_scale is None:
            scale = image_features.new_tensor(1.0)
        else:
            scale = logit_scale.float().exp().clamp(max=100.0).to(image_features.device)
        similarity_logits = scale * image_features @ text_features.t()

        return DistillOutputs(
            vision_layers=vision_layers,
            text_layers=text_layers,
            image_features=image_features,
            text_features=text_features,
            similarity_logits=similarity_logits,
        )
