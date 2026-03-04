# codes/tasks/coco_multilabel.py
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import read_json, pil_loader


class COCOMultiLabelDataset(Dataset):
    """
    Multi-label classification dataset from COCO instances file.
    Each image -> multi-hot vector over categories (default: 80).
    """
    def __init__(self, img_dir: str, instances_json: str):
        self.img_dir = img_dir
        ann = read_json(instances_json)

        # category_id -> index [0..C-1]
        cats = ann["categories"]
        cat_ids = [c["id"] for c in cats]
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(sorted(cat_ids))}
        self.num_classes = len(self.cat_id_to_idx)

        # image_id -> file_name
        self.id2file = {im["id"]: im["file_name"] for im in ann["images"]}

        # gather labels: image_id -> set(cat_idx)
        img2cats: Dict[int, set] = {}
        for a in ann["annotations"]:
            img_id = a["image_id"]
            cid = a["category_id"]
            if cid not in self.cat_id_to_idx:
                continue
            img2cats.setdefault(img_id, set()).add(self.cat_id_to_idx[cid])

        self.image_ids = sorted(list(img2cats.keys()))
        self.img2cats = img2cats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, self.id2file[img_id])
        image = pil_loader(img_path)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in self.img2cats[img_id]:
            y[c] = 1.0
        return image, y


def build_multilabel_loader(adapter, img_dir: str, instances_json: str,
                            batch_size: int = 128, num_workers: int = 8, shuffle: bool = True):
    ds = COCOMultiLabelDataset(img_dir=img_dir, instances_json=instances_json)

    def collate(batch):
        images = torch.stack([adapter.preprocess_pil(b[0]) for b in batch], 0)
        labels = torch.stack([b[1] for b in batch], 0)
        return images, labels

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate, drop_last=False)
    return ds, loader
