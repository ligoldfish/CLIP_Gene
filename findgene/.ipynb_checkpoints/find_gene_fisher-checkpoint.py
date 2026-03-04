#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learngene extraction from CLIP on COCO2017 using Fisher-like importance:
  importance(layer) = E_batch [ mean(grad^2) ]   (no optimizer.step needed)

Why this works when gradients are "flat" over training:
- You don't rely on "rise then fall" dynamics.
- You estimate how much each layer matters for the task loss (local sensitivity).

Outputs
- fisher_logs.npz: per-step per-layer fisher importance (visual/text)
- heatmaps: fisher_heatmap_visual.png / fisher_heatmap_text.png
- optional sensitivity bar plots: sensitivity_visual.png / sensitivity_text.png
- selected_layers.json
- learngene_visual.pt / learngene_text.pt / learngene_multimodal.pt

Run example:
python extract_learngene_clip_coco_fisher.py \
  --coco_img_dir /path/to/coco2017/train2017 \
  --coco_ann_file /path/to/coco2017/annotations/captions_train2017.json \
  --out_dir ./outputs/lg_clip_coco_fisher \
  --clip_model ViT-B/32 \
  --device cuda \
  --num_tasks 40 \
  --batches_per_task 30 \
  --batch_size 128 \
  --topk 3 \
  --prefer_last_ratio 0.5 \
  --do_sensitivity 1 \
  --sensitivity_batches 8 \
  --noise_alpha 1e-3
"""

import os
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip
from PIL import Image


# -------------------------
# Dataset
# -------------------------
class COCOCaptionDataset(Dataset):
    """Return (PIL_image, list_of_captions) for each image_id."""
    def __init__(self, img_dir: str, ann_file: str):
        self.img_dir = img_dir
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        self.images = {img["id"]: img["file_name"] for img in ann["images"]}
        self.captions: Dict[int, List[str]] = {}
        for a in ann["annotations"]:
            self.captions.setdefault(a["image_id"], []).append(a["caption"])
        self.ids = list(self.captions.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        file_name = self.images[img_id]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        return image, self.captions[img_id]


def make_collate_fn(preprocess):
    def _collate(batch):
        images, caps = zip(*batch)
        texts = [random.choice(c_list) for c_list in caps]
        image_inputs = torch.stack([preprocess(im) for im in images], dim=0)  # [B,3,H,W]
        text_inputs = clip.tokenize(texts, truncate=True)  # [B,77]
        return image_inputs, text_inputs
    return _collate


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_vit_clip(model):
    # OpenAI CLIP ViT uses: model.visual.transformer.resblocks
    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer"):
        raise RuntimeError("Expect ViT-based CLIP: model.visual.transformer.resblocks")
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "resblocks"):
        raise RuntimeError("Expect CLIP text transformer blocks: model.transformer.resblocks")


def build_layer_param_lists(model) -> Tuple[List[List[torch.nn.Parameter]], List[List[torch.nn.Parameter]]]:
    """
    Group parameters by transformer block index for:
      - visual.transformer.resblocks[i]
      - transformer.resblocks[i]
    """
    ensure_vit_clip(model)
    n_visual = len(model.visual.transformer.resblocks)
    n_text = len(model.transformer.resblocks)

    visual_params: List[List[torch.nn.Parameter]] = [[] for _ in range(n_visual)]
    text_params: List[List[torch.nn.Parameter]] = [[] for _ in range(n_text)]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("visual.transformer.resblocks."):
            idx = int(name.split("resblocks.")[1].split(".")[0])
            visual_params[idx].append(p)
        elif name.startswith("transformer.resblocks."):
            idx = int(name.split("resblocks.")[1].split(".")[0])
            text_params[idx].append(p)

    return visual_params, text_params


@torch.no_grad()
def layer_fisher_mean_g2(params: List[torch.nn.Parameter]) -> float:
    """mean(grad^2) averaged over params inside the layer (Fisher diag proxy)."""
    vals = []
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        vals.append((g * g).mean().item())
    if len(vals) == 0:
        return 0.0
    return float(sum(vals) / len(vals))


@torch.no_grad()
def layer_rel_update_proxy(params: List[torch.nn.Parameter], eps: float = 1e-12) -> float:
    """Optional: mean( |g| ) / mean( |w| )   (scale-free)."""
    gs, ws = [], []
    for p in params:
        if p.grad is None:
            continue
        gs.append(p.grad.detach().abs().mean().item())
        ws.append(p.detach().abs().mean().item())
    if len(gs) == 0:
        return 0.0
    g = float(sum(gs) / len(gs))
    w = float(sum(ws) / len(ws))
    return g / (w + eps)


def compute_layer_importance(
    visual_layer_params: List[List[torch.nn.Parameter]],
    text_layer_params: List[List[torch.nn.Parameter]],
    use_rel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-layer scalar importance:
      visual: [Lv], text: [Lt]
    Default: fisher-like mean(g^2)
    Optional: relative scale-free proxy mean(|g|)/mean(|w|)
    """
    if use_rel:
        iv = np.array([layer_rel_update_proxy(ps) for ps in visual_layer_params], dtype=np.float32)
        it = np.array([layer_rel_update_proxy(ps) for ps in text_layer_params], dtype=np.float32)
    else:
        iv = np.array([layer_fisher_mean_g2(ps) for ps in visual_layer_params], dtype=np.float32)
        it = np.array([layer_fisher_mean_g2(ps) for ps in text_layer_params], dtype=np.float32)
    return iv, it


def plot_heatmap(G: np.ndarray, title: str, out_path: str, log_scale: bool = True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X = G.copy()
    if log_scale:
        X = np.log10(X + 1e-12)
    plt.figure(figsize=(max(10, X.shape[1] * 0.5), 7))
    im = plt.imshow(X, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Layer index")
    plt.ylabel("Recorded step")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(values: List[float], title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    xs = np.arange(len(values))
    plt.figure(figsize=(max(10, len(values) * 0.6), 5))
    plt.bar(xs, values)
    plt.title(title)
    plt.xlabel("Layer index")
    plt.ylabel("Δ loss (perturbation sensitivity)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def select_topk_from_importance(
    I: np.ndarray,  # [T, L]
    topk: int,
    prefer_last_ratio: float = 0.5,
    use_cv: bool = True,
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """
    Score per layer by:
      score = mean(I) * (1 + CV(I))   if use_cv
      score = mean(I)                else
    Only consider last prefer_last_ratio portion of layers (default last half).
    """
    assert I.ndim == 2
    T, L = I.shape
    start = int(L * (1.0 - prefer_last_ratio))
    cand = list(range(start, L))

    mean = I.mean(axis=0)
    std = I.std(axis=0)
    cv = std / (mean + 1e-12)
    if use_cv:
        score = mean * (1.0 + cv)
    else:
        score = mean

    ranked = sorted(cand, key=lambda i: float(score[i]), reverse=True)
    top = ranked[:topk]

    stats = {}
    for i in top:
        stats[i] = {
            "mean": float(mean[i]),
            "std": float(std[i]),
            "cv": float(cv[i]),
            "score": float(score[i]),
        }
    return top, stats


def save_gene_layers_clip(model, visual_layers: List[int], text_layers: List[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()

    def _filter(prefix: str, layers: List[int]) -> Dict[str, torch.Tensor]:
        keep = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            for i in layers:
                if f"{prefix}{i}." in k:
                    keep[k] = v.detach().cpu()
                    break
        return keep

    visual_gene = _filter("visual.transformer.resblocks.", visual_layers)
    text_gene = _filter("transformer.resblocks.", text_layers)

    multimodal_gene = {}
    for key in ["logit_scale", "text_projection", "visual.proj"]:
        if key in sd:
            multimodal_gene[key] = sd[key].detach().cpu()

    torch.save({"layers": visual_layers, "state_dict": visual_gene}, os.path.join(out_dir, "learngene_visual.pt"))
    torch.save({"layers": text_layers, "state_dict": text_gene}, os.path.join(out_dir, "learngene_text.pt"))
    torch.save({"state_dict": multimodal_gene}, os.path.join(out_dir, "learngene_multimodal.pt"))


@torch.no_grad()
def eval_avg_loss(model, loader, device, max_batches: int) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            images, texts = next(it)
        except StopIteration:
            break
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        logits_per_image, logits_per_text = model(images, texts)
        labels = torch.arange(images.size(0), device=device)
        loss = 0.5 * (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def add_noise_to_params(params: List[torch.nn.Parameter], alpha: float):
    """
    In-place perturbation: p <- p + N(0, (alpha * std(p))^2)
    alpha=1e-3 is usually safe.
    """
    for p in params:
        if not p.requires_grad:
            continue
        if p.data is None:
            continue
        std = p.data.float().std().item()
        if std <= 0:
            continue
        noise = torch.randn_like(p.data) * (alpha * std)
        p.data.add_(noise)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--coco_ann_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_tasks", type=int, default=40)
    parser.add_argument("--batches_per_task", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--prefer_last_ratio", type=float, default=0.5)

    parser.add_argument("--use_rel_importance", type=int, default=0,
                        help="1: use mean(|g|)/mean(|w|) instead of fisher mean(g^2)")
    parser.add_argument("--use_cv", type=int, default=1,
                        help="1: score=mean*(1+cv), 0: score=mean")

    parser.add_argument("--do_sensitivity", type=int, default=0)
    parser.add_argument("--sensitivity_batches", type=int, default=8)
    parser.add_argument("--noise_alpha", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model = model.float()
    model.train()
    ensure_vit_clip(model)

    n_visual = len(model.visual.transformer.resblocks)
    n_text = len(model.transformer.resblocks)
    print(f"[INFO] CLIP={args.clip_model} | visual blocks={n_visual} | text blocks={n_text}")

    ds = COCOCaptionDataset(args.coco_img_dir, args.coco_ann_file)
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    splits = np.array_split(np.array(all_indices, dtype=np.int64), args.num_tasks)

    visual_layer_params, text_layer_params = build_layer_param_lists(model)

    Iv_records, It_records = [], []
    meta = []  # (task_id, batch_id, loss)
    collate_fn = make_collate_fn(preprocess)

    # ---- Fisher collection: forward+backward only (no optimizer step)
    for task_id, idxs in enumerate(splits):
        subset = Subset(ds, idxs.tolist())
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        it = iter(loader)
        losses = []
        for b in range(args.batches_per_task):
            try:
                images, texts = next(it)
            except StopIteration:
                it = iter(loader)
                images, texts = next(it)

            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            logits_per_image, logits_per_text = model(images, texts)
            labels = torch.arange(images.size(0), device=device)
            loss = 0.5 * (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))

            model.zero_grad(set_to_none=True)
            loss.backward()

            Iv, It = compute_layer_importance(
                visual_layer_params,
                text_layer_params,
                use_rel=bool(args.use_rel_importance),
            )
            Iv_records.append(Iv)
            It_records.append(It)
            meta.append((task_id, b, float(loss.item())))
            losses.append(float(loss.item()))

        print(f"[TASK {task_id+1:02d}/{args.num_tasks}] avg_loss={sum(losses)/max(1,len(losses)):.4f}")

    Iv = np.stack(Iv_records, axis=0)  # [T, Lv]
    It = np.stack(It_records, axis=0)  # [T, Lt]
    meta = np.array(meta, dtype=np.float32)

    # ---- Select topk
    top_v, stats_v = select_topk_from_importance(
        Iv, topk=args.topk, prefer_last_ratio=args.prefer_last_ratio, use_cv=bool(args.use_cv)
    )
    top_t, stats_t = select_topk_from_importance(
        It, topk=args.topk, prefer_last_ratio=args.prefer_last_ratio, use_cv=bool(args.use_cv)
    )

    print("\n[RESULT] Visual topk:", top_v)
    for i in top_v:
        s = stats_v[i]
        print(f"  - L{i}: mean={s['mean']:.3e} std={s['std']:.3e} cv={s['cv']:.3f} score={s['score']:.3e}")

    print("\n[RESULT] Text topk:", top_t)
    for i in top_t:
        s = stats_t[i]
        print(f"  - L{i}: mean={s['mean']:.3e} std={s['std']:.3e} cv={s['cv']:.3f} score={s['score']:.3e}")

    # ---- Save logs + plots
    np.savez_compressed(
        os.path.join(args.out_dir, "fisher_logs.npz"),
        Iv=Iv, It=It, meta=meta,
        topk_visual=np.array(top_v, dtype=np.int64),
        topk_text=np.array(top_t, dtype=np.int64),
    )

    title_suffix = "REL(|g|/|w|)" if args.use_rel_importance else "FISHER(mean(g^2))"
    plot_heatmap(Iv, f"CLIP Visual {title_suffix} Heatmap | {args.clip_model} | COCO2017", os.path.join(args.out_dir, "fisher_heatmap_visual.png"))
    plot_heatmap(It, f"CLIP Text {title_suffix} Heatmap | {args.clip_model} | COCO2017", os.path.join(args.out_dir, "fisher_heatmap_text.png"))

    # ---- Optional: sensitivity test (delta-loss per layer)
    sens_v, sens_t = None, None
    if args.do_sensitivity:
        # build a small eval loader from a fixed subset for stability
        eval_indices = all_indices[: min(4096, len(all_indices))]
        eval_subset = Subset(ds, eval_indices)
        eval_loader = DataLoader(
            eval_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        base_loss = eval_avg_loss(model, eval_loader, device, max_batches=args.sensitivity_batches)
        print(f"\n[SENS] baseline avg loss over {args.sensitivity_batches} batches = {base_loss:.4f}")

        # Visual sensitivity
        sens_v = []
        for li in range(n_visual):
            backup = [p.data.detach().clone() for p in visual_layer_params[li]]
            add_noise_to_params(visual_layer_params[li], alpha=args.noise_alpha)
            loss2 = eval_avg_loss(model, eval_loader, device, max_batches=args.sensitivity_batches)
            delta = max(0.0, loss2 - base_loss)
            sens_v.append(delta)
            # restore
            for p, b in zip(visual_layer_params[li], backup):
                p.data.copy_(b)
        plot_bar(sens_v, f"Visual layer sensitivity Δloss (alpha={args.noise_alpha})", os.path.join(args.out_dir, "sensitivity_visual.png"))

        # Text sensitivity
        sens_t = []
        for li in range(n_text):
            backup = [p.data.detach().clone() for p in text_layer_params[li]]
            add_noise_to_params(text_layer_params[li], alpha=args.noise_alpha)
            loss2 = eval_avg_loss(model, eval_loader, device, max_batches=args.sensitivity_batches)
            delta = max(0.0, loss2 - base_loss)
            sens_t.append(delta)
            for p, b in zip(text_layer_params[li], backup):
                p.data.copy_(b)
        plot_bar(sens_t, f"Text layer sensitivity Δloss (alpha={args.noise_alpha})", os.path.join(args.out_dir, "sensitivity_text.png"))

        print("[SENS] Saved sensitivity plots.")

    # ---- Save selection info + learngene weights
    with open(os.path.join(args.out_dir, "selected_layers.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_model": args.clip_model,
                "num_tasks": args.num_tasks,
                "batches_per_task": args.batches_per_task,
                "batch_size": args.batch_size,
                "importance": "rel(|g|/|w|)" if args.use_rel_importance else "fisher(mean(g^2))",
                "score": "mean*(1+cv)" if args.use_cv else "mean",
                "prefer_last_ratio": args.prefer_last_ratio,
                "topk": args.topk,
                "visual": {"selected": top_v, "stats": {str(k): v for k, v in stats_v.items()}},
                "text": {"selected": top_t, "stats": {str(k): v for k, v in stats_t.items()}},
                "sensitivity": {
                    "enabled": bool(args.do_sensitivity),
                    "noise_alpha": args.noise_alpha,
                    "baseline_batches": args.sensitivity_batches,
                    "visual_delta_loss": sens_v,
                    "text_delta_loss": sens_t,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_gene_layers_clip(model, top_v, top_t, args.out_dir)

    print(f"\n[DONE] Outputs saved to: {args.out_dir}")
    print("  - fisher_logs.npz")
    print("  - fisher_heatmap_visual.png / fisher_heatmap_text.png")
    if args.do_sensitivity:
        print("  - sensitivity_visual.png / sensitivity_text.png")
    print("  - selected_layers.json")
    print("  - learngene_visual.pt / learngene_text.pt / learngene_multimodal.pt")


if __name__ == "__main__":
    main()
