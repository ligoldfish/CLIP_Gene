# scripts/data/webdataset_pairs.py
"""WebDataset (tar shards) image-text pairs loader.

This module is used for large-scale web datasets (e.g., CC3M/CC12M) that are
stored as WebDataset shards:
  - each sample has an image (jpg/png/jpeg/webp) and a caption (txt).

We keep it minimal and robust:
  - distributed-safe splitting by node/worker
  - ignores broken samples (common with web-scale URL data)
  - supports resampled=True (endless stream) with a fixed epoch size via with_epoch

Note: This requires the `webdataset` package.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Callable, List, Union


@dataclass
class WdsPairConfig:
    # WebDataset shards. Can be a glob pattern string (e.g., "/path/train/*.tar")
    # or a concrete list of shard paths.
    shards: Union[str, List[str]]
    shuffle_buf: int = 20000
    resampled: bool = True
    samples_per_epoch: int = 0  # 0 = do not set epoch length
    handler: str = "warn"  # "warn" | "ignore"

def clean_cap(cap):
    if isinstance(cap, (bytes, bytearray)):
        cap = cap.decode("utf-8", "ignore")
    cap = (cap or "").strip()
    return cap if cap else "."   # 关键：空的变成一个

def _get_handler(name: str):
    import webdataset as wds
    if name == "ignore":
        return wds.handlers.ignore_and_continue
    return wds.handlers.warn_and_continue

def build_wds_pairs(cfg: WdsPairConfig, transform: Callable):
    import webdataset as wds
    from webdataset import shardlists

    ds = wds.WebDataset(
        cfg.shards,
        resampled=cfg.resampled,
        handler=_get_handler(cfg.handler),
        # ✅ 关键：显式指定按 node/worker 切 shards（单机也安全）
        nodesplitter=shardlists.split_by_node,
        workersplitter=shardlists.split_by_worker,
        # （可选）消掉 shardshuffle warning
        shardshuffle=False,
    )

    if cfg.shuffle_buf and cfg.shuffle_buf > 0:
        ds = ds.shuffle(cfg.shuffle_buf)

    ds = (
        ds.decode("pil")
        .to_tuple("jpg;png;jpeg;webp", "txt")
        .map_tuple(
            lambda im: transform(im.convert("RGB")),
            clean_cap,
        )
    )

    if cfg.samples_per_epoch and cfg.samples_per_epoch > 0:
        ds = ds.with_epoch(cfg.samples_per_epoch)

    return ds




def default_wds_collate(tokenize):
    """
    WebDataset .batched(..., collation_fn=...) 会调用 collation_fn(batch)
    其中 batch 是一个 list，元素通常是 (img_tensor, caption_str) 或 dict。
    """
    def _collate(batch):
        if len(batch) == 0:
            return None

        first = batch[0]

        # case1: 每个样本是 (img, cap)
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            images = torch.stack([b[0] for b in batch], dim=0)
            caps = [b[1] for b in batch]

        # case2: 每个样本是 dict（按你的 build_wds_pairs 实现调整 key）
        elif isinstance(first, dict):
            # 常见 key: "jpg"/"png" + "txt"
            # 也可能是你自己 build_wds_pairs 里产出的 "image"/"caption"
            img_key = "image" if "image" in first else ("jpg" if "jpg" in first else "png")
            cap_key = "caption" if "caption" in first else "txt"
            images = torch.stack([b[img_key] for b in batch], dim=0)
            caps = [b[cap_key] for b in batch]
        else:
            raise TypeError(f"Unexpected sample type in batch: {type(first)}")

        tokens = tokenize(caps)  # 一般 tokenize 支持 list[str] -> LongTensor
        return images, tokens

    return _collate