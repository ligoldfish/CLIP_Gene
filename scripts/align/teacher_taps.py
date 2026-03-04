from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class TapConfig:
    """Layers are 1-indexed: layer=1 means output AFTER resblocks[0]."""
    vision_layers: List[int]
    text_layers: List[int]


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

        if teacher_fp16 and device.type == "cuda":
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
