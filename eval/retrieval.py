"""图文检索 R@K。本地保底：COCO-val（非 Karpathy）。Flickr30k 缺数据跳过。"""
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
from PIL import Image

from eval.model_interface import EvalModel
from eval.datasets_local import load_coco_val_pairs, load_flickr30k_pairs


@torch.no_grad()
def _encode_images(model: EvalModel, paths: List[str], preprocess, batch_size=128) -> torch.Tensor:
    embs = []
    buf = []
    for p in paths:
        buf.append(preprocess(Image.open(p).convert("RGB")))
        if len(buf) == batch_size:
            embs.append(model.encode_image(torch.stack(buf)))
            buf = []
    if buf:
        embs.append(model.encode_image(torch.stack(buf)))
    return torch.cat(embs, dim=0)  # [Ni, D]


@torch.no_grad()
def _encode_texts(model: EvalModel, texts: List[str], batch_size=256) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        embs.append(model.encode_text(texts[i:i + batch_size]))
    return torch.cat(embs, dim=0)  # [Nt, D]


def _recall_at_k(ranks: List[int], ks=(1, 5, 10)) -> dict:
    n = len(ranks)
    out = {}
    r = torch.tensor(ranks, dtype=torch.float)
    for k in ks:
        out[f"R@{k}"] = 100.0 * (r < k).float().mean().item()
    out["medr"] = float(r.median().item()) + 1
    return out


@torch.no_grad()
def retrieval_eval(model: EvalModel, preprocess, pairs: List[Tuple[str, List[str]]],
                   batch_size=128, tag="") -> dict:
    paths = [p for p, _ in pairs]
    img_emb = _encode_images(model, paths, preprocess, batch_size)   # [Ni, D]

    cap_texts, cap2img = [], []
    img2caps = defaultdict(list)
    for i, (_, caps) in enumerate(pairs):
        for c in caps:
            img2caps[i].append(len(cap_texts))
            cap2img.append(i)
            cap_texts.append(c)
    txt_emb = _encode_texts(model, cap_texts)                        # [Nt, D]
    cap2img_t = torch.tensor(cap2img, device=img_emb.device)

    sim = img_emb @ txt_emb.t()                                      # [Ni, Nt]

    # image -> text：每图取其任一 caption 的最佳排名
    i2t_ranks = []
    order_i = sim.argsort(dim=1, descending=True)                    # [Ni, Nt]
    for i in range(sim.size(0)):
        gt = set(img2caps[i])
        row = order_i[i].tolist()
        rank = next(j for j, c in enumerate(row) if c in gt)
        i2t_ranks.append(rank)

    # text -> image
    t2i_ranks = []
    order_t = sim.t().argsort(dim=1, descending=True)                # [Nt, Ni]
    for c in range(sim.size(1)):
        gt_img = cap2img_t[c].item()
        row = order_t[c].tolist()
        rank = row.index(gt_img)
        t2i_ranks.append(rank)

    res = {}
    for k, v in _recall_at_k(i2t_ranks).items():
        res[f"i2t_{k}"] = v
    for k, v in _recall_at_k(t2i_ranks).items():
        res[f"t2i_{k}"] = v
    res["dataset"] = tag or "retrieval"
    res["n_images"] = len(paths)
    res["n_captions"] = len(cap_texts)
    return res


def evaluate_coco5k(model: EvalModel, preprocess, val_img_dir: str, val_ann: str,
                    max_images=5000, batch_size=128) -> dict:
    pairs = load_coco_val_pairs(val_img_dir, val_ann, max_images=max_images)
    return retrieval_eval(model, preprocess, pairs, batch_size, tag="COCO-val5k(non-Karpathy)")


def evaluate_flickr30k(model: EvalModel, preprocess, img_dir: str, ann: str,
                       batch_size=128) -> Optional[dict]:
    pairs = load_flickr30k_pairs(img_dir, ann)
    if pairs is None:
        print("[eval] Flickr30k 数据缺失，跳过")
        return None
    return retrieval_eval(model, preprocess, pairs, batch_size, tag="Flickr30k-1k")
