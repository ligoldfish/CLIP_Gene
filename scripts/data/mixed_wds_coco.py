# scripts/data/mixed_wds_coco.py
"""Mix WebDataset (CC3M) stream with COCO2017 captions for CLIP-style pretraining.

Design goals
------------
1) Work with CC3M WebDataset shards (streaming, resampled) and COCO (local files).
2) Keep epoch length deterministic (important for LR schedule / logging).
3) Avoid COCO "5 captions per image" overweighting by *sampling captions per image*.
   We still count the *total number of captions* as COCO's pair count when sizing epochs,
   so you can set samples_per_epoch to CC3M_pairs + COCO_pairs to "maximize" the virtual
   sample budget without material memory bloat.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class CocoPoolStats:
    num_images: int
    num_pairs: int


class CocoCaptionPool:
    """COCO caption pool sampled by image, caption chosen randomly.

    This avoids flattening 5 captions/image into ~590k python tuples (memory heavy),
    while still exposing the full caption diversity over time.
    """

    def __init__(
        self,
        images_root: str,
        captions_json: str,
        transform: Optional[Callable] = None,
        max_images: int = -1,
    ):
        self.images_root = images_root
        self.transform = transform

        with open(captions_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in data.get("images", [])}

        # image_id -> list[captions]
        id2caps = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cap = ann.get("caption", "")
            if img_id is None:
                continue
            id2caps.setdefault(img_id, []).append(cap)

        paths: List[str] = []
        caps: List[List[str]] = []
        num_pairs = 0
        for img_id, c_list in id2caps.items():
            fn = id2file.get(img_id)
            if not fn:
                continue
            c_list = [c.strip() for c in c_list if isinstance(c, str) and c.strip()]
            if len(c_list) == 0:
                c_list = ["."]
            paths.append(os.path.join(images_root, fn))
            caps.append(c_list)
            num_pairs += len(c_list)

        if max_images > 0:
            paths = paths[:max_images]
            caps = caps[:max_images]
            num_pairs = sum(len(x) for x in caps)

        self._paths = paths
        self._caps = caps
        self.stats = CocoPoolStats(num_images=len(paths), num_pairs=int(num_pairs))

    def __len__(self) -> int:
        return self.stats.num_images

    def sample(self, rnd: random.Random) -> Tuple[torch.Tensor, str]:
        """Return (image_tensor, caption_str)."""
        if self.stats.num_images <= 0:
            raise RuntimeError("Empty COCO pool (num_images=0). Check paths / captions json.")

        k = rnd.randrange(self.stats.num_images)
        path = self._paths[k]
        cap_list = self._caps[k]
        cap = cap_list[rnd.randrange(len(cap_list))] if len(cap_list) > 0 else "."

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cap


class MixedWdsCocoPairs(IterableDataset):
    """Iterable that yields mixed (image_tensor, caption_str) pairs.

    - CC3M is an *iterable* (WebDataset pipeline) yielding tuples (img_tensor, caption_str)
    - COCO is a CocoCaptionPool sampled randomly.
    - The iterable yields exactly `samples_per_epoch` samples (per-rank), so training
      steps/epoch are deterministic.
    """

    def __init__(
        self,
        cc3m_iterable,
        coco_pool: CocoCaptionPool,
        p_coco: float,
        samples_per_epoch: int,
        seed: int = 42,
    ):
        super().__init__()
        assert 0.0 <= p_coco <= 1.0
        self.cc3m_iterable = cc3m_iterable
        self.coco_pool = coco_pool
        self.p_coco = float(p_coco)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        rnd = random.Random(self.seed + 100000 * rank + 1000 * wid)

        cc_it = iter(self.cc3m_iterable)
        for _ in range(self.samples_per_epoch):
            use_coco = (self.p_coco > 0.0) and (rnd.random() < self.p_coco)
            if use_coco:
                yield self.coco_pool.sample(rnd)
            else:
                try:
                    yield next(cc_it)
                except StopIteration:
                    cc_it = iter(self.cc3m_iterable)
                    yield next(cc_it)
