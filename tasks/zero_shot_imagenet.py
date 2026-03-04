# codes/tasks/zero_shot_imagenet.py
from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .common import l2_normalize


DEFAULT_IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
]


def load_imagenet_classnames(classnames_txt: str) -> List[str]:
    """
    Expect a text file with 1000 lines, each line = class name.
    """
    with open(classnames_txt, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()]
    return names


@torch.no_grad()
def build_zeroshot_classifier(
    adapter,
    classnames: List[str],
    templates: List[str] = DEFAULT_IMAGENET_TEMPLATES,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Returns weights: [D, C] normalized
    """
    device = adapter.device
    all_weights = []

    for cname in classnames:
        texts = [t.format(cname) for t in templates]
        tok = adapter.tokenize(texts).to(device)
        z = adapter.encode_text(tok)  # [T, D]
        z = l2_normalize(z, dim=-1)
        class_emb = l2_normalize(z.mean(dim=0, keepdim=True), dim=-1)  # [1,D]
        all_weights.append(class_emb.cpu())

    W = torch.cat(all_weights, dim=0)  # [C,D]
    W = l2_normalize(W, dim=-1).t().contiguous()  # [D,C]
    return W


def build_imagenet_val_loader(adapter, imagenet_val_dir: str, batch_size: int = 128, num_workers: int = 8):
    """
    imagenet_val_dir should be ImageFolder structure:
      val/
        n01440764/
          *.JPEG
        n01443537/
          *.JPEG
        ...
    """
    ds = ImageFolder(imagenet_val_dir)

    def collate(batch):
        images = torch.stack([adapter.preprocess_pil(x[0]) for x in batch], 0)
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return images, labels

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return ds, loader


@torch.no_grad()
def zeroshot_eval_step(adapter, images: torch.Tensor, labels: torch.Tensor, classifier_W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    classifier_W: [D,C]
    Returns:
      logits: [B,C]
      labels: [B]
    """
    z = adapter.encode_image(images)  # [B,D]
    logits = (z @ classifier_W.to(z.device))  # cosine sim since normalized
    return logits, labels.to(logits.device)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> Dict[str, float]:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B,maxk]
    correct = pred.eq(labels.view(-1, 1))
    out = {}
    for k in ks:
        out[f"acc@{k}"] = float(correct[:, :k].any(dim=1).float().mean().item())
    return out
