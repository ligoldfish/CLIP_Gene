"""COCO caption 数据集与 collate（统一替代 train_clip/train_geneclip/train_cnn 三处重复）。

提供：
  - COCOCaptionDataset：__getitem__ 返回 (PIL.Image, [captions...])
  - make_coco_collate(preprocess, tokenize_fn)：随机抽 1 条 caption，做 preprocess + tokenize
  - make_calibration_subset(...)：CM-BI / LSQ 用的确定性小子集（固定 seed、首条 caption）
"""
import os
import json
import random
from typing import List, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image


class COCOCaptionDataset(Dataset):
    """仅用 COCO captions（图像 + 文本注释）。

    __getitem__ 返回 (PIL.Image, [captions...])；preprocess/tokenize 留给 collate。
    """

    def __init__(self, img_dir: str, ann_file: str):
        self.img_dir = img_dir
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)
        self.images = {img["id"]: img for img in ann["images"]}
        caps = {}
        for a in ann["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"])
        # 只保留至少 1 条 caption 的图片，并按 id 排序保证确定性
        self.ids = sorted(i for i in self.images.keys() if i in caps)
        self.captions = caps

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[str]]:
        image_id = self.ids[idx]
        file_name = self.images[image_id]["file_name"]
        img = Image.open(os.path.join(self.img_dir, file_name)).convert("RGB")
        return img, self.captions[image_id]


def make_coco_collate(preprocess: Callable, tokenize_fn: Callable):
    """返回 collate_fn：随机抽 1 条 caption，做图像预处理与 tokenize。"""

    def collate(batch: List[Tuple[Image.Image, List[str]]]):
        images, caps = zip(*batch)
        txts = [random.choice(c) for c in caps]
        imgs = torch.stack([preprocess(im) for im in images])
        toks = tokenize_fn(txts)
        return imgs, toks

    return collate


def make_deterministic_collate(preprocess: Callable, tokenize_fn: Callable, caption_index: int = 0):
    """确定性 collate：固定取第 caption_index 条 caption（CM-BI / LSQ 复现用）。"""

    def collate(batch: List[Tuple[Image.Image, List[str]]]):
        images, caps = zip(*batch)
        txts = [c[min(caption_index, len(c) - 1)] for c in caps]
        imgs = torch.stack([preprocess(im) for im in images])
        toks = tokenize_fn(txts)
        return imgs, toks

    return collate


def make_calibration_subset(
    img_dir: str,
    ann_file: str,
    preprocess: Callable,
    tokenize_fn: Callable,
    n_pairs: int = 3000,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """CM-BI / LSQ 标定 loader：确定性子集（固定 seed 抽样、首条 caption、不 shuffle）。"""
    full = COCOCaptionDataset(img_dir, ann_file)
    n = min(n_pairs, len(full))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(full)), n))
    subset = Subset(full, indices)
    collate = make_deterministic_collate(preprocess, tokenize_fn, caption_index=0)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
