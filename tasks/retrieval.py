# codes/tasks/retrieval.py
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import read_json, pil_loader, collate_keep_strings
from .model_adapters import ModelAdapter


class KarpathyRetrievalDataset(Dataset):
    """
    Works with Karpathy-style JSON (commonly used for COCO/Flickr retrieval):
    json format example:
      {
        "images": [
          {
            "filepath": "coco/train2014",
            "filename": "COCO_train2014_000000000009.jpg",
            "split": "train/val/test",
            "sentences": [{"raw": "a man ..."}, ...]
          }, ...
        ]
      }

    For COCO2017, you can still use a karpathy split json pointing to train2017/val2017 filenames.
    """
    def __init__(self, root: str, karpathy_json: str, split: str = "val"):
        self.root = root
        data = read_json(karpathy_json)
        self.images = []
        for item in data["images"]:
            if item.get("split") == split:
                self.images.append(item)

        # flatten captions list, while keeping mapping
        self.img_paths: List[str] = []
        self.captions: List[str] = []
        self.caption_to_img: List[int] = []
        self.img_to_caption_ids: Dict[int, List[int]] = {}

        for img_idx, item in enumerate(self.images):
            fp = item.get("filepath", "")
            fn = item["filename"]
            img_path = os.path.join(self.root, fp, fn) if fp else os.path.join(self.root, fn)
            self.img_paths.append(img_path)

            sents = item["sentences"]
            cap_ids = []
            for s in sents:
                cap_id = len(self.captions)
                self.captions.append(s["raw"])
                self.caption_to_img.append(img_idx)
                cap_ids.append(cap_id)
            self.img_to_caption_ids[img_idx] = cap_ids

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # for image loader (evaluation)
        return pil_loader(self.img_paths[idx]), idx


class RetrievalTextDataset(Dataset):
    """All captions as separate items (for evaluation)."""
    def __init__(self, captions: List[str]):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx], idx


@torch.no_grad()
def encode_all_images(adapter: ModelAdapter, image_ds: KarpathyRetrievalDataset, batch_size: int = 128, num_workers: int = 8):
    loader = DataLoader(
        image_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: (torch.stack([adapter.preprocess_pil(x[0]) for x in b], 0), torch.tensor([x[1] for x in b]))
    )
    feats = []
    order = []
    for images, idxs in loader:
        z = adapter.encode_image(images)
        feats.append(z.cpu())
        order.append(idxs.cpu())
    feats = torch.cat(feats, 0)
    order = torch.cat(order, 0)
    # ensure correct order
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    feats = feats[inv]
    return feats  # [N_img, D]


@torch.no_grad()
def encode_all_texts(adapter: ModelAdapter, captions: List[str], batch_size: int = 256, num_workers: int = 4):
    text_ds = RetrievalTextDataset(captions)
    loader = DataLoader(
        text_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_keep_strings
    )
    feats = []
    order = []
    for texts, idxs in loader:
        tok = adapter.tokenize(texts)
        z = adapter.encode_text(tok)
        feats.append(z.cpu())
        order.append(idxs.cpu())
    feats = torch.cat(feats, 0)
    order = torch.cat(order, 0)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    feats = feats[inv]
    return feats  # [N_txt, D]


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def compute_retrieval_metrics(
    img_feats: torch.Tensor,  # [N_img, D]
    txt_feats: torch.Tensor,  # [N_txt, D]
    img_to_caption_ids: Dict[int, List[int]],
    caption_to_img: List[int],
    ks=(1, 5, 10),
) -> Dict[str, float]:
    """
    Compute Recall@K for:
      - Image-to-Text (I2T): for each image, rank all texts, success if any GT caption in topK
      - Text-to-Image (T2I): for each text, rank all images, success if GT image in topK
    """
    img_feats = img_feats.float()
    txt_feats = txt_feats.float()
    sim = img_feats @ txt_feats.t()  # [N_img, N_txt]

    # I2T ranks
    i2t_ranks = np.zeros(sim.shape[0], dtype=np.int32)
    sim_np = sim.numpy()
    for i in range(sim.shape[0]):
        gt_caps = set(img_to_caption_ids[i])
        order = np.argsort(-sim_np[i])  # descending
        # find best rank among gt captions
        best = 10**9
        for cap_id in gt_caps:
            r = int(np.where(order == cap_id)[0][0])
            if r < best:
                best = r
        i2t_ranks[i] = best

    # T2I ranks
    # we can reuse sim: for each text, rank images by sim[:,t]
    t2i_ranks = np.zeros(sim.shape[1], dtype=np.int32)
    for t in range(sim.shape[1]):
        gt_img = caption_to_img[t]
        order = np.argsort(-sim_np[:, t])
        r = int(np.where(order == gt_img)[0][0])
        t2i_ranks[t] = r

    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = recall_at_k(i2t_ranks, k)
        out[f"t2i_R@{k}"] = recall_at_k(t2i_ranks, k)
    out["i2t_medR"] = float(np.median(i2t_ranks) + 1)
    out["t2i_medR"] = float(np.median(t2i_ranks) + 1)
    return out


def build_retrieval_datasets(root: str, karpathy_json: str, split: str = "val"):
    ds_img = KarpathyRetrievalDataset(root=root, karpathy_json=karpathy_json, split=split)
    return ds_img, ds_img.captions, ds_img.img_to_caption_ids, ds_img.caption_to_img
