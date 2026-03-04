# codes/tasks/builders.py
from __future__ import annotations
from typing import Dict, Tuple, List

from .retrieval import build_retrieval_datasets
from .matching import build_itm_loader
from .coco_multilabel import build_multilabel_loader
from .zero_shot_imagenet import build_imagenet_val_loader, load_imagenet_classnames, build_zeroshot_classifier


def build_task_retrieval(adapter, root: str, karpathy_json: str, split: str = "val"):
    """
    Returns:
      image_dataset, captions_list, img_to_caption_ids, caption_to_img
    """
    return build_retrieval_datasets(root=root, karpathy_json=karpathy_json, split=split)


def build_task_itm(adapter, coco_img_dir: str, coco_captions_json: str,
                   batch_size: int = 128, num_workers: int = 8,
                   pos_ratio: float = 0.5, seed: int = 42):
    return build_itm_loader(adapter, coco_img_dir, coco_captions_json, batch_size, num_workers, pos_ratio, seed)


def build_task_multilabel(adapter, coco_img_dir: str, coco_instances_json: str,
                          batch_size: int = 128, num_workers: int = 8, shuffle: bool = True):
    return build_multilabel_loader(adapter, coco_img_dir, coco_instances_json, batch_size, num_workers, shuffle)


def build_task_zeroshot_imagenet(adapter, imagenet_val_dir: str, classnames_txt: str,
                                batch_size: int = 128, num_workers: int = 8):
    classnames = load_imagenet_classnames(classnames_txt)
    ds, loader = build_imagenet_val_loader(adapter, imagenet_val_dir, batch_size, num_workers)
    W = build_zeroshot_classifier(adapter, classnames)
    return ds, loader, W
