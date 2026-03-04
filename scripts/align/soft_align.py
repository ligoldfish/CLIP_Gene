from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class SoftAlignWeights:
    w_cos: float = 1.0
    w_stat: float = 0.25
    w_delta: float = 0.25


def _token_cosine_loss(s: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - cosine(sim) on token embeddings, averaged over seq*batch."""
    s = F.normalize(s.float(), dim=-1, eps=eps)
    t = F.normalize(t.float(), dim=-1, eps=eps)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def _mean_std_loss(s: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match per-channel mean/std across (seq,B)."""
    s = s.float()
    t = t.float()
    s_mean = s.mean(dim=(0, 1))
    t_mean = t.mean(dim=(0, 1))
    s_std = s.var(dim=(0, 1), unbiased=False).add(eps).sqrt()
    t_std = t.var(dim=(0, 1), unbiased=False).add(eps).sqrt()
    return F.mse_loss(s_mean, t_mean) + F.mse_loss(s_std, t_std)


def soft_align_layers(
    student_layers: List[torch.Tensor],
    teacher_layers: List[torch.Tensor],
    weights: SoftAlignWeights,
    eps: float = 1e-6,
    drop_cls: bool = False,
) -> torch.Tensor:
    """Softly align student intermediate token sequences to teacher taps.

    Args:
        student_layers: list of [seq,B,width]
        teacher_layers: list of [seq,B,width]
        weights: weights for different soft constraints
        drop_cls: if True, ignore token 0 (useful for vision CLS)

    Returns:
        scalar loss
    """
    assert len(student_layers) == len(teacher_layers), (
        f"len(student_layers)={len(student_layers)} != len(teacher_layers)={len(teacher_layers)}"
    )

    def _maybe_drop(x: torch.Tensor) -> torch.Tensor:
        return x[1:] if drop_cls and x.shape[0] > 1 else x

    total = torch.zeros((), device=student_layers[0].device, dtype=torch.float32)
    n = float(max(1, len(student_layers)))

    # per-layer token constraints
    for s, t in zip(student_layers, teacher_layers):
        s = _maybe_drop(s)
        t = _maybe_drop(t)
        if weights.w_cos:
            total = total + float(weights.w_cos) * _token_cosine_loss(s, t, eps=eps)
        if weights.w_stat:
            total = total + float(weights.w_stat) * _mean_std_loss(s, t, eps=eps)

    # delta constraints (encourage similar *changes* across depth)
    if weights.w_delta and len(student_layers) >= 2:
        for i in range(1, len(student_layers)):
            ds = _maybe_drop(student_layers[i] - student_layers[i - 1])
            dt = _maybe_drop(teacher_layers[i] - teacher_layers[i - 1])
            total = total + float(weights.w_delta) * _token_cosine_loss(ds, dt, eps=eps)

    return total / n
