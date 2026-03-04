# scripts/finetune_common.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils.misc import unwrap_model, is_main_process
from scripts.data.transforms import build_clip_image_transform


def enable_tf32(enable: bool = True):
    """Enable TF32 on supported NVIDIA GPUs (Ampere+)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable


def get_model_image_dtype(model: nn.Module) -> torch.dtype:
    m = unwrap_model(model)
    # OpenAI CLIP has .dtype; StudentCLIP also has .dtype
    if hasattr(m, "dtype"):
        dt = getattr(m, "dtype")
        if isinstance(dt, torch.dtype):
            return dt
    # fallback: parameter dtype
    try:
        return next(m.parameters()).dtype
    except StopIteration:
        return torch.float32


def get_logit_scale(model: nn.Module) -> torch.Tensor:
    m = unwrap_model(model)
    if hasattr(m, "logit_scale"):
        ls = m.logit_scale
        # logit_scale stored in log-space in both OpenAI CLIP and StudentCLIP
        return ls.exp().clamp(max=100.0)
    # fallback constant
    device = next(m.parameters()).device
    return torch.tensor(1.0, device=device)


def encode_image(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Return L2-normalized image embedding [B, D].
    Supports:
      - OpenAI CLIP / TinyCLIP: encode_image
      - StudentCLIP: vision_stem + vision_tower
    """
    m = unwrap_model(model)
    img_dtype = get_model_image_dtype(m)
    images = images.to(device=next(m.parameters()).device, dtype=img_dtype, non_blocking=True)

    if hasattr(m, "encode_image"):
        z = m.encode_image(images)
    elif hasattr(m, "vision_stem") and hasattr(m, "vision_tower"):
        v_tokens = m.vision_stem(images)
        z = m.vision_tower(v_tokens)
    else:
        raise RuntimeError("Model does not support encode_image (need encode_image or StudentCLIP vision_stem/vision_tower).")

    return F.normalize(z, dim=-1)


def encode_text(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """
    Return L2-normalized text embedding [B, D].
    Supports:
      - OpenAI CLIP / TinyCLIP: encode_text
      - StudentCLIP: text_stem + text_tower
    """
    m = unwrap_model(model)
    tokens = tokens.to(device=next(m.parameters()).device, non_blocking=True)

    if hasattr(m, "encode_text"):
        z = m.encode_text(tokens)
    elif hasattr(m, "text_stem") and hasattr(m, "text_tower"):
        t_tokens = m.text_stem(tokens)
        z = m.text_tower(t_tokens, text=tokens)
    else:
        raise RuntimeError("Model does not support encode_text (need encode_text or StudentCLIP text_stem/text_tower).")

    return F.normalize(z, dim=-1)


def forward_features(model: nn.Module, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (img_emb, txt_emb), both normalized."""
    zi = encode_image(model, images)
    zt = encode_text(model, tokens)
    return zi, zt


def resolve_gene_variant_dirs(gene_dir: str) -> List[str]:
    """
    For ours(三种)：
      - If gene_dir itself contains learngene_visual.pt => treat as single variant.
      - Else scan subfolders under gene_dir and return those containing learngene_visual.pt.
    """
    if gene_dir is None or gene_dir == "":
        return []
    if os.path.isfile(os.path.join(gene_dir, "learngene_visual.pt")):
        return [gene_dir]
    outs = []
    if os.path.isdir(gene_dir):
        for name in sorted(os.listdir(gene_dir)):
            p = os.path.join(gene_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "learngene_visual.pt")):
                outs.append(p)
    return outs


def save_model_state(out_path: str, model: nn.Module, extra: Optional[Dict[str, Any]] = None):
    """
    Save a compact checkpoint: state_dict + optional extra.
    (避免你之前那种 1.2G：不要 torch.save(model)，也不要默认把 optimizer/scaler 全塞进去)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    state = {
        "state_dict": unwrap_model(model).state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, out_path)


def freeze_openai_clip_backbone(model: nn.Module, freeze_image: bool, freeze_text: bool):
    """
    For clip/tinyclip: optionally freeze visual or text parts.
    """
    m = unwrap_model(model)
    if freeze_image and hasattr(m, "visual"):
        for p in m.visual.parameters():
            p.requires_grad = False
    if freeze_text:
        # OpenAI CLIP uses token_embedding/positional_embedding/transformer/ln_final/text_projection
        for attr in ["token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection"]:
            if hasattr(m, attr):
                obj = getattr(m, attr)
                if isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad = False
                elif isinstance(obj, torch.Tensor) and obj.requires_grad:
                    obj.requires_grad = False


def pretty_trainable(model: nn.Module, max_lines: int = 50):
    m = unwrap_model(model)
    total = 0
    trainable = 0
    lines = []
    for n, p in m.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
            if len(lines) < max_lines:
                lines.append(f"  [T] {n:60s} {tuple(p.shape)}")
        else:
            if len(lines) < max_lines:
                lines.append(f"  [F] {n:60s} {tuple(p.shape)}")
    if is_main_process():
        print(f"[Params] total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
        for s in lines:
            print(s)
        if len(lines) == max_lines:
            print("  ...")

            from scripts.data.transforms import build_clip_image_transform

class FinetunedModelAdapter:
    """
    让 tasks/* 的评测器直接用你当前 fine-tune 的 model + tokenize。
    """
    def __init__(self, model, tokenize, image_size=224, device="cuda",
                 amp=False, amp_dtype="bf16"):
        self.model = model
        self._tokenize = tokenize
        self.device = torch.device(device)
        self._preprocess = build_clip_image_transform(image_size, is_train=False)
        self.amp = bool(amp)
        # accept both string ("bf16"/"fp16") and torch.dtype
        if isinstance(amp_dtype, torch.dtype):
            self.amp_dtype = amp_dtype
        else:
            s = str(amp_dtype).lower()
            self.amp_dtype = torch.bfloat16 if s in ["bf16", "bfloat16"] else torch.float16
    def preprocess_pil(self, pil):
        return self._preprocess(pil)

    def tokenize(self, texts):
        return self._tokenize(texts)

    @torch.no_grad()
    def encode_image(self, images):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
            # StudentCLIP/ours has forward(image, text) and may not implement .encode_image();
            # use helper encode_image(model, images) defined in this file.
            z = encode_image(self.model, images)
        z = z.float()
        return torch.nn.functional.normalize(z, dim=-1)

    @torch.no_grad()
    def encode_text(self, tokens):
        self.model.eval()
        tokens = tokens.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
            # StudentCLIP/ours may not implement .encode_text();
            # use helper encode_text(model, tokens) defined in this file.
            z = encode_text(self.model, tokens)
        z = z.float()
        return torch.nn.functional.normalize(z, dim=-1)
