# scripts/optim.py
import math
from typing import List


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    if total_steps <= 0:
        return base_lr
    t = step / total_steps
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_optimizer_lrs(optimizer, lrs: List[float]):
    assert len(optimizer.param_groups) == len(lrs)
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr
