"""本地数据集加载：CIFAR-100（pickle）、COCO-val captions、Flickr30k（缺则 None）。

安全说明：CIFAR-100 pickle 是官方本地数据集（可信），非外部输入。
"""
import os
import json
import pickle
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR100Pickle(Dataset):
    """读官方 CIFAR-100 pickle（test/train），返回 (PIL.Image, fine_label)。"""

    def __init__(self, pickle_dir: str, split: str = "test", transform=None):
        path = os.path.join(pickle_dir, split)
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")  # trusted: official CIFAR-100
        data = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # [N,32,32,3]
        self.images = data
        self.labels = d["fine_labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx].astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_coco_val_pairs(val_img_dir: str, val_ann: str, max_images: Optional[int] = 5000
                        ) -> List[Tuple[str, List[str]]]:
    """返回 [(image_path, [captions...]), ...]（按 image_id 排序，确定性）。

    注：非 Karpathy 5k split。结果应标注 'COCO-val5k (non-Karpathy)'。
    """
    with open(val_ann, "r", encoding="utf-8") as f:
        ann = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"])
    ids = sorted(i for i in id2file if i in caps)
    if max_images:
        ids = ids[:max_images]
    pairs = [(os.path.join(val_img_dir, id2file[i]), caps[i]) for i in ids]
    return pairs


def load_flickr30k_pairs(img_dir: str, ann: str) -> Optional[List[Tuple[str, List[str]]]]:
    """Flickr30k（Karpathy split json）。缺数据返回 None。"""
    if not img_dir or not ann or not os.path.isfile(ann) or not os.path.isdir(img_dir):
        return None
    with open(ann, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    # 兼容 Karpathy dataset_flickr30k.json 结构（images: [{filename, sentences:[{raw}], split}]）
    images = data["images"] if isinstance(data, dict) and "images" in data else data
    for im in images:
        if im.get("split") not in (None, "test"):
            continue
        fn = im.get("filename") or im.get("file_name")
        sents = [s["raw"] if isinstance(s, dict) else s for s in im.get("sentences", im.get("captions", []))]
        if fn and sents:
            pairs.append((os.path.join(img_dir, fn), sents))
    return pairs or None
