# scripts/utils/profile.py
from __future__ import annotations
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

@torch.no_grad()
def profile_clip_like(
    model: nn.Module,
    forward_fn: Callable[[], None],
    iters: int = 50,
    profile_speed: bool = False,
) -> Dict[str, Optional[float]]:
    """
    forward_fn: 一个无参闭包，内部跑一次模型 forward（比如 encode_image+encode_text）
    返回 flops_total (int) 以及可选 latency_ms / peak_mem_mb
    """
    out: Dict[str, Optional[float]] = {
        "flops_total": None,
        "latency_ms": None,
        "peak_mem_mb": None,
    }

    # -------- FLOPs (torch 原生 flop counter) --------
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(model, display=False) as fc:
            forward_fn()
        out["flops_total"] = int(fc.get_total_flops())
    except Exception:
        out["flops_total"] = None

    # -------- Latency / PeakMem (可选) --------
    if profile_speed and torch.cuda.is_available():
        device = next(model.parameters()).device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # warmup
        for _ in range(10):
            forward_fn()
        torch.cuda.synchronize(device)

        t0 = time.time()
        for _ in range(int(iters)):
            forward_fn()
        torch.cuda.synchronize(device)
        t1 = time.time()

        out["latency_ms"] = 1000.0 * (t1 - t0) / float(iters)
        out["peak_mem_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)

    return out
