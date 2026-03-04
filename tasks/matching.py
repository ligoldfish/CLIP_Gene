# codes/tasks/matching.py
from __future__ import annotations
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import read_json, pil_loader, collate_keep_strings


class COCOMatchingDataset(Dataset):
    """
    Cross-modal matching (ITM) dataset built from COCO captions json.
    For each item:
      - with prob pos_ratio: return a positive (image, its caption, label=1)
      - else: return a negative (image, caption from another image, label=0)

    Expects official COCO captions file:
      captions_train2017.json / captions_val2017.json
    """
    def __init__(
        self,
        img_dir: str,
        captions_json: str,
        pos_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.img_dir = img_dir
        self.pos_ratio = pos_ratio
        self.rng = random.Random(seed)

        ann = read_json(captions_json)
        # image_id -> file_name
        id2file = {im["id"]: im["file_name"] for im in ann["images"]}
        # image_id -> captions list
        cap_map: Dict[int, List[str]] = {}
        for a in ann["annotations"]:
            cap_map.setdefault(a["image_id"], []).append(a["caption"])

        self.image_ids = list(cap_map.keys())
        self.id2file = id2file
        self.cap_map = cap_map

        # build a flat captions pool for negative sampling
        self.all_caps = []
        self.all_img_for_cap = []
        for img_id, caps in cap_map.items():
            for c in caps:
                self.all_caps.append(c)
                self.all_img_for_cap.append(img_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, self.id2file[img_id])
        image = pil_loader(img_path)

        if self.rng.random() < self.pos_ratio:
            caption = self.rng.choice(self.cap_map[img_id])
            label = 1
        else:
            # sample caption from different image
            while True:
                j = self.rng.randrange(len(self.all_caps))
                neg_img = self.all_img_for_cap[j]
                if neg_img != img_id:
                    caption = self.all_caps[j]
                    break
            label = 0

        return image, caption, label


def build_itm_loader(adapter, img_dir: str, captions_json: str,
                     batch_size: int = 128, num_workers: int = 8,
                     pos_ratio: float = 0.5, seed: int = 42):
    ds = COCOMatchingDataset(img_dir=img_dir, captions_json=captions_json, pos_ratio=pos_ratio, seed=seed)

    def collate(batch):
        images = torch.stack([adapter.preprocess_pil(b[0]) for b in batch], 0)
        texts = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        tokens = adapter.tokenize(texts)
        return images, tokens, labels

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=True)
    return ds, loader
