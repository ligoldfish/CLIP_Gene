# scripts/utils/misc.py
import os
import random
import numpy as np
import torch

from scripts.utils.device import seed_accelerators


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_accelerators(seed)


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def is_main_process() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
