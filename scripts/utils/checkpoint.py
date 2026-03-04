# scripts/utils/checkpoint.py
import os
import torch
from typing import Any, Dict


def save_checkpoint(path: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
