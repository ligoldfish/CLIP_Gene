"""零样本分类（标准 CLIP）。本地保底：CIFAR-100。"""
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from eval.model_interface import EvalModel
from eval.datasets_local import CIFAR100Pickle
from eval.prompts import CIFAR100_TEMPLATES, cifar100_classnames


@torch.no_grad()
def build_zeroshot_classifier(model: EvalModel, class_names: List[str], templates: List[str]) -> torch.Tensor:
    """返回 [D, C] 分类器：每类对所有模板取均值再归一。"""
    weights = []
    for c in class_names:
        emb = model.encode_text([t.format(c) for t in templates])  # [T, D]
        emb = emb.mean(0)
        emb = emb / emb.norm()
        weights.append(emb)
    return torch.stack(weights, dim=1)  # [D, C]


@torch.no_grad()
def zero_shot_classification(model: EvalModel, loader: DataLoader, classifier: torch.Tensor) -> dict:
    top1 = top5 = n = 0
    for images, labels in loader:
        feats = model.encode_image(images)              # [B, D]
        logits = model.logit_scale_exp * (feats @ classifier)  # [B, C]
        labels = labels.to(logits.device)
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(labels.view(-1, 1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += labels.size(0)
    return {"top1": 100.0 * top1 / n, "top5": 100.0 * top5 / n, "n": n}


def evaluate_cifar100(model: EvalModel, preprocess, pickle_dir: str,
                      split: str = "test", templates: Optional[List[str]] = None,
                      batch_size: int = 256, num_workers: int = 4) -> dict:
    templates = templates or CIFAR100_TEMPLATES
    ds = CIFAR100Pickle(pickle_dir, split=split, transform=preprocess)
    classnames = cifar100_classnames(pickle_dir)
    classifier = build_zeroshot_classifier(model, classnames, templates)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    res = zero_shot_classification(model, loader, classifier)
    res["dataset"] = f"cifar100-{split}"
    return res
