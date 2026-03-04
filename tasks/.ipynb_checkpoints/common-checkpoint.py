# codes/tasks/common.py
import os
import json
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pil_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def default_clip_image_transform(image_size: int = 224):
    """
    A CLIP-like preprocessing (resize -> center crop -> to tensor -> normalize).
    If you use OpenAI CLIP adapter, you can directly use its preprocess instead.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def collate_keep_strings(batch):
    """
    Default collate but keeps strings as list.
    Batch elements can contain (tensor, string) etc.
    """
    # If batch is list of tuples
    if isinstance(batch[0], (tuple, list)):
        out = []
        for i in range(len(batch[0])):
            col = [b[i] for b in batch]
            if isinstance(col[0], str):
                out.append(col)
            else:
                out.append(default_collate(col))
        return tuple(out)
    return default_collate(batch)


def chunked(iterable: List[Any], n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
