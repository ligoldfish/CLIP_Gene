#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction (Adapter Expansion) for CLIP on COCO2017:
CLIP + Adapter Expansion  -> use Adapter gradients to compute ρ(t) -> select top-k blocks.

ρ_i(t) = (#adapter params in block i with |grad| > σ) / (#adapter params in block i)
σ is FIXED after a short warmup, estimated from adapter |grad| distribution quantile on early steps.

Key idea:
- Freeze original CLIP weights.
- Insert lightweight Adapters into every Transformer block (visual + text).
- Train ONLY adapters (and optionally logit_scale).
- Now gradients live on adapters and produce clearer layer-wise dynamics.
- Select "rise then fall" blocks using ρ(t) curves.

Outputs in out_dir:
- rho_logs.npz                (Rv, Rt, meta, sigma_v/sigma_t, topk indices)
- rho_heatmap_visual.png      (Figure-3-like heatmap)
- rho_heatmap_text.png
- rho_trends_visual.png       (topk + refs)
- rho_trends_text.png
- selected_layers.json
- learngene_visual.pt         (original CLIP resblock weights for selected indices)
- learngene_text.pt
- learngene_multimodal.pt
- adapters_state.pt           (all adapter weights; optional for later analysis)

Install:
  pip install git+https://github.com/openai/CLIP.git
  pip install torchvision pillow matplotlib numpy

Run example:
python find_gene.py --coco_img_dir /root/autodl-tmp/train2017 --coco_ann_file /root/autodl-tmp/annotations/captions_train2017.json --out_dir ./outputs/lg_clip_adapter_rho --clip_model ViT-B/32 --device cuda --num_tasks 40 --steps_per_task 150 --batch_size 128 --adapter_bottleneck 64 --lr 1e-3 --sigma_warmup_steps 20 --sigma_quantile 0.90 --sigma_sample 200000 --record_every 1 --topk 3 --prefer_last_ratio 0.5
"""

import os
import re
import json
import math
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clip
from PIL import Image
import types

try:
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("Please install torchvision: pip install torchvision") from e


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


# -------------------------
# Stronger-difference task split (topic + length)
# -------------------------
_ANIMALS = {"dog","cat","bird","horse","cow","sheep","elephant","zebra","giraffe","bear","lion","tiger","monkey","duck","goose"}
_VEHICLES = {"car","bus","truck","train","motorcycle","bike","bicycle","boat","ship","airplane","plane","van","taxi","scooter"}
_INDOOR = {"kitchen","room","bed","bedroom","bathroom","table","desk","chair","couch","sofa","sink","refrigerator","stove","tv","computer","laptop"}
_SPORTS = {"tennis","baseball","soccer","basketball","ski","snowboard","skateboard","surf","football","golf","bat","ball","court","field"}
_FOOD = {"pizza","sandwich","burger","cake","hot dog","donut","apple","banana","salad","bread","pasta","rice","plate","bowl","cup"}
_PEOPLE = {"man","woman","boy","girl","person","people","child","children","kid","kids","adult"}

def _tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[a-z]+", s.lower())

def caption_topic_bucket(caption: str) -> str:
    toks = set(_tokenize_simple(caption))
    if toks & _ANIMALS:
        return "animals"
    if toks & _VEHICLES:
        return "vehicles"
    if toks & _INDOOR:
        return "indoor"
    if toks & _SPORTS:
        return "sports"
    if toks & _FOOD:
        return "food"
    if toks & _PEOPLE:
        return "people"
    return "others"

def caption_length_bucket(caption: str) -> str:
    n = len(_tokenize_simple(caption))
    if n <= 8:
        return "short"
    if n <= 16:
        return "medium"
    return "long"

def build_semantic_tasks(
    ds: COCOCaptionDataset,
    num_tasks: int,
    seed: int = 42,
    target_task_size: Optional[int] = None,
) -> List[List[int]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {}

    for idx in range(len(ds)):
        caps = ds.captions[ds.ids[idx]]
        cap = caps[0] if caps else ""
        key = f"{caption_topic_bucket(cap)}::{caption_length_bucket(cap)}"
        buckets.setdefault(key, []).append(idx)

    for k in buckets:
        rng.shuffle(buckets[k])

    N = len(ds)
    if target_task_size is None:
        target_task_size = max(256, N // num_tasks)

    tasks: List[List[int]] = []
    for k, idxs in buckets.items():
        for i in range(0, len(idxs), target_task_size):
            chunk = idxs[i:i + target_task_size]
            if len(chunk) >= max(64, target_task_size // 4):
                tasks.append(chunk)

    rng.shuffle(tasks)

    while len(tasks) > num_tasks:
        tasks.sort(key=len)
        a = tasks.pop(0)
        b = tasks.pop(0)
        tasks.append(a + b)

    while len(tasks) < num_tasks:
        tasks.sort(key=len, reverse=True)
        big = tasks.pop(0)
        if len(big) < 2 * max(64, target_task_size // 2):
            tasks.append(big)
            tasks.append(big[:])
        else:
            mid = len(big) // 2
            tasks.append(big[:mid])
            tasks.append(big[mid:])

    rng.shuffle(tasks)
    return tasks[:num_tasks]


# -------------------------
# Per-task prompt + augmentation + text corruption
# -------------------------
PROMPT_TEMPLATES = [
    "{cap}",
    "a photo of {cap}",
    "this image shows {cap}",
    "a close-up photo of {cap}",
    "a low quality photo of {cap}",
    "a photo taken at night of {cap}",
]

def build_augment(mode: str):
    # applied on PIL image before CLIP preprocess
    if mode == "none":
        return lambda x: x
    if mode == "weak":
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.80, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])
    if mode == "strong":
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.55, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.15),
        ])
    if mode == "style":
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.60, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.6),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
    raise ValueError(f"Unknown aug mode: {mode}")

def word_dropout(caption: str, drop_p: float) -> str:
    toks = caption.strip().split()
    if len(toks) <= 2 or drop_p <= 0:
        return caption
    kept = [w for w in toks if random.random() > drop_p]
    if len(kept) == 0:
        kept = [random.choice(toks)]
    return " ".join(kept)

def make_collate_fn(preprocess, prompt_mode: int, aug_mode: str, drop_p: float):
    aug = build_augment(aug_mode)

    def _collate(batch):
        images, caps = zip(*batch)
        texts = []
        for c_list in caps:
            cap = random.choice(c_list)
            cap = word_dropout(cap, drop_p=drop_p)
            tpl = PROMPT_TEMPLATES[prompt_mode % len(PROMPT_TEMPLATES)]
            texts.append(tpl.format(cap=cap))
        image_inputs = torch.stack([preprocess(aug(im)) for im in images], dim=0)
        text_inputs = clip.tokenize(texts, truncate=True)
        return image_inputs, text_inputs

    return _collate


# -------------------------
# Adapter expansion
# -------------------------
class Adapter(nn.Module):
    """
    Simple bottleneck adapter:
      Adapter(x) = W_up( act(W_down(x)) )
    Initialize W_up to zero => initially identity behavior for the whole block.
    """
    def __init__(self, d_model: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # x: [seq, batch, d]
        return self.up(self.drop(self.act(self.down(x))))


def ensure_vit_clip(model):
    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer"):
        raise RuntimeError("Expect ViT-based CLIP: model.visual.transformer.resblocks")
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "resblocks"):
        raise RuntimeError("Expect CLIP text transformer blocks: model.transformer.resblocks")


def inject_adapters_into_resblock(block: nn.Module, d_model: int, bottleneck: int, dropout: float):
    """
    Monkey-patch a CLIP ResidualAttentionBlock:
      original:
        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))
      patched:
        h1 = attn(ln1(x)); x = x + h1 + adapter_attn(h1)
        h2 = mlp(ln2(x)); x = x + h2 + adapter_mlp(h2)
    """
    if hasattr(block, "adapter_attn") or hasattr(block, "adapter_mlp"):
        return  # already patched

    block.adapter_attn = Adapter(d_model, bottleneck, dropout=dropout)
    block.adapter_mlp = Adapter(d_model, bottleneck, dropout=dropout)

    # Keep original methods
    if not hasattr(block, "attention"):
        raise RuntimeError("Block has no attention() method. Are you using OpenAI CLIP ViT?")

    def forward_with_adapters(self, x):
        h1 = self.attention(self.ln_1(x))
        x = x + h1 + self.adapter_attn(h1)
        h2 = self.mlp(self.ln_2(x))
        x = x + h2 + self.adapter_mlp(h2)
        return x

    block.forward = types.MethodType(forward_with_adapters, block)


def inject_adapters_clip(model, bottleneck: int = 64, dropout: float = 0.0):
    ensure_vit_clip(model)

    # infer hidden dims
    # For OpenAI CLIP ViT, visual transformer blocks use model.visual.transformer.width
    # Text transformer blocks use model.transformer.width
    d_visual = model.visual.transformer.width
    d_text = model.transformer.width

    for blk in model.visual.transformer.resblocks:
        inject_adapters_into_resblock(blk, d_model=d_visual, bottleneck=bottleneck, dropout=dropout)

    for blk in model.transformer.resblocks:
        inject_adapters_into_resblock(blk, d_model=d_text, bottleneck=bottleneck, dropout=dropout)

    return d_visual, d_text


def freeze_non_adapter_params(model, train_logit_scale: bool = True):
    for n, p in model.named_parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if ".adapter_attn." in n or ".adapter_mlp." in n:
            p.requires_grad = True
        if train_logit_scale and n == "logit_scale":
            p.requires_grad = True


def build_adapter_param_groups(model) -> Tuple[List[List[torch.nn.Parameter]], List[List[torch.nn.Parameter]]]:
    """
    Return per-block adapter params lists:
      visual_adapters[i] = params in block i's adapters (attn+mlp)
      text_adapters[i]   = params in block i's adapters (attn+mlp)
    """
    ensure_vit_clip(model)
    v_blocks = list(model.visual.transformer.resblocks)
    t_blocks = list(model.transformer.resblocks)

    visual = []
    for blk in v_blocks:
        ps = []
        for name in ["adapter_attn", "adapter_mlp"]:
            mod = getattr(blk, name)
            ps += [p for p in mod.parameters() if p.requires_grad]
        visual.append(ps)

    text = []
    for blk in t_blocks:
        ps = []
        for name in ["adapter_attn", "adapter_mlp"]:
            mod = getattr(blk, name)
            ps += [p for p in mod.parameters() if p.requires_grad]
        text.append(ps)

    return visual, text


# -------------------------
# rho / sigma
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def sample_abs_grads_from_paramlists(paramlists: List[List[torch.nn.Parameter]], sample_size: int) -> torch.Tensor:
    """
    Sample |grad| scalars from all params in `paramlists` (list of layers -> list of params).
    Used to estimate sigma quantile efficiently.
    """
    params_all = []
    for layer in paramlists:
        for p in layer:
            if p.grad is not None:
                params_all.append(p)
    if len(params_all) == 0 or sample_size <= 0:
        return torch.empty(0)

    # proportional allocation by numel
    numels = [p.grad.numel() for p in params_all]
    total = max(1, sum(numels))
    alloc = [int(sample_size * (n / total)) for n in numels]
    # fix rounding
    s = sum(alloc)
    if s < sample_size:
        order = np.argsort(numels)[::-1]
        for j in range(sample_size - s):
            alloc[int(order[j % len(order)])] += 1
    elif s > sample_size:
        order = np.argsort(numels)[::-1]
        for j in range(s - sample_size):
            i = int(order[j % len(order)])
            if alloc[i] > 0:
                alloc[i] -= 1

    chunks = []
    device = params_all[0].grad.device
    for p, k in zip(params_all, alloc):
        if k <= 0:
            continue
        g = p.grad.detach().abs().flatten()
        if g.numel() <= k:
            chunks.append(g)
        else:
            idx = torch.randint(low=0, high=g.numel(), size=(k,), device=device)
            chunks.append(g.index_select(0, idx))
    if len(chunks) == 0:
        return torch.empty(0, device=device)
    return torch.cat(chunks, dim=0)

@torch.no_grad()
def rho_per_layer(paramlists: List[List[torch.nn.Parameter]], sigma: float) -> np.ndarray:
    rhos = []
    thr = float(sigma)
    for layer in paramlists:
        num, den = 0, 0
        for p in layer:
            if p.grad is None:
                continue
            g = p.grad.detach().abs()
            num += int((g > thr).sum().item())
            den += int(g.numel())
        rhos.append(float(num) / float(max(1, den)))
    return np.array(rhos, dtype=np.float32)


def score_rise_then_fall_rho(
    R: np.ndarray,  # [T, L]
    prefer_last_ratio: float = 0.5,
    early_ratio: float = 0.25,
    late_ratio: float = 0.25,
    peak_not_too_late_ratio: float = 0.8,
    delta: float = 0.12,
    eps: float = 1e-12,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Score layers that look like: early higher -> late lower + has a peak earlier.
    """
    T, L = R.shape
    t_early = max(1, int(T * early_ratio))
    t_late = max(1, int(T * late_ratio))
    late_start = max(0, T - t_late)

    early = R[:t_early].mean(axis=0)
    late = R[late_start:].mean(axis=0)
    peak = R.max(axis=0)
    peak_step = R.argmax(axis=0)

    stats = []
    scores = np.zeros(L, dtype=np.float32)
    for i in range(L):
        cond1 = early[i] > late[i] * (1.0 + delta)
        cond2 = peak[i] > late[i] * (1.0 + delta)
        cond3 = peak_step[i] < int(T * peak_not_too_late_ratio)

        drop = float(early[i] - late[i])
        peak_gap = float(peak[i] - late[i])
        ratio = float((peak[i] + eps) / (late[i] + eps))
        score = (drop + peak_gap) * math.log1p(ratio)
        if not (cond1 and cond2 and cond3):
            score = 0.0

        scores[i] = float(score)
        stats.append({
            "early_mean": float(early[i]),
            "late_mean": float(late[i]),
            "peak": float(peak[i]),
            "peak_step": int(peak_step[i]),
            "score": float(score),
        })

    start_idx = int(L * (1.0 - prefer_last_ratio))
    cand = list(range(start_idx, L))
    ranked = sorted(cand, key=lambda i: float(scores[i]), reverse=True)
    return stats, ranked


# -------------------------
# Plotting
# -------------------------
def plot_heatmap_rho(R: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(max(10, R.shape[1] * 0.55), 7))
    im = plt.imshow(R, aspect="auto", interpolation="nearest",
                    vmin=0.0, vmax=min(1.0, float(np.max(R) + 1e-6)))
    plt.title(title)
    plt.xlabel("Block index")
    plt.ylabel("Recorded step")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="ρ = frac(|grad|>σ)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_trends(R: np.ndarray, layers: List[int], title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    for li in layers:
        plt.plot(R[:, li], label=f"B{li}", linewidth=1.6)
    plt.title(title)
    plt.xlabel("Recorded step")
    plt.ylabel("ρ")
    plt.ylim(0.0, min(1.0, float(np.max(R[:, layers]) + 0.05)))
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Save learngene weights (original CLIP blocks)
# -------------------------
def save_gene_layers_clip_original_blocks(model, visual_layers: List[int], text_layers: List[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()

    def _filter(prefix: str, layers: List[int]) -> Dict[str, torch.Tensor]:
        keep = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            for i in layers:
                if f"{prefix}{i}." in k:
                    # skip adapter params: we only want original CLIP block weights here
                    if ".adapter_attn." in k or ".adapter_mlp." in k:
                        continue
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


def save_all_adapters_state(model, out_dir: str):
    sd = model.state_dict()
    adapters = {k: v.detach().cpu() for k, v in sd.items() if (".adapter_attn." in k or ".adapter_mlp." in k)}
    torch.save(adapters, os.path.join(out_dir, "adapters_state.pt"))


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
    parser.add_argument("--steps_per_task", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_logit_scale", type=int, default=1)

    parser.add_argument("--sigma_warmup_steps", type=int, default=20,
                        help="number of early recorded steps used to estimate fixed sigma")
    parser.add_argument("--sigma_quantile", type=float, default=0.90,
                        help="sigma = quantile(|grad|, q) from warmup samples (fixed thereafter)")
    parser.add_argument("--sigma_sample", type=int, default=200000,
                        help="sample size of |grad| per recorded step for sigma estimation")

    parser.add_argument("--record_every", type=int, default=1)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--prefer_last_ratio", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model = model.float()
    model.train()
    ensure_vit_clip(model)

    # Inject adapters
    d_visual, d_text = inject_adapters_clip(
        model, bottleneck=args.adapter_bottleneck, dropout=args.adapter_dropout
    )
    model = model.to(device)
    model = model.float()
    freeze_non_adapter_params(model, train_logit_scale=bool(args.train_logit_scale))

    visual_adapter_params, text_adapter_params = build_adapter_param_groups(model)
    n_visual = len(visual_adapter_params)
    n_text = len(text_adapter_params)

    # Optimizer only on trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    print(f"[INFO] CLIP={args.clip_model} | device={device}")
    print(f"[INFO] Injected adapters: visual blocks={n_visual} (d={d_visual}), text blocks={n_text} (d={d_text})")
    print(f"[INFO] Trainable params={sum(p.numel() for p in trainable)/1e6:.3f}M "
          f"(adapters only{' + logit_scale' if args.train_logit_scale else ''})")
    print(f"[INFO] lr={args.lr} wd={args.wd} steps/task={args.steps_per_task} tasks={args.num_tasks}")

    # Dataset and tasks
    ds = COCOCaptionDataset(args.coco_img_dir, args.coco_ann_file)
    tasks = build_semantic_tasks(ds, num_tasks=args.num_tasks, seed=args.seed)
    print(f"[INFO] built {len(tasks)} tasks; sizes(first 5)={[len(t) for t in tasks[:5]]} ...")

    # Task modes to amplify differences
    aug_modes = ["weak", "strong", "style", "none"]
    # make text corruption vary with task
    drop_ps = [0.0, 0.10, 0.25, 0.0]

    # Logs
    rho_v_records: List[np.ndarray] = []
    rho_t_records: List[np.ndarray] = []
    meta: List[Tuple[int, int, int, float, int, str, float]] = []  # task, step, global, loss, prompt_mode, aug_mode, drop_p
    global_step = 0
    t0 = time.time()

    # Warmup sigma estimation buffers
    warm_abs_v = []
    warm_abs_t = []
    sigma_v = None
    sigma_t = None

    # Training / recording loop
    for task_id, idxs in enumerate(tasks):
        prompt_mode = task_id % len(PROMPT_TEMPLATES)
        aug_mode = aug_modes[task_id % len(aug_modes)]
        drop_p = drop_ps[task_id % len(drop_ps)]

        subset = Subset(ds, idxs)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(preprocess, prompt_mode=prompt_mode, aug_mode=aug_mode, drop_p=drop_p),
        )
        it = iter(loader)
        losses = []

        for step_in_task in range(args.steps_per_task):
            try:
                images, texts = next(it)
            except StopIteration:
                it = iter(loader)
                images, texts = next(it)

            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            logits_per_image, logits_per_text = model(images, texts)
            labels = torch.arange(images.size(0), device=device)

            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = 0.5 * (loss_i + loss_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.max_grad_norm)

            # record
            if (global_step % args.record_every) == 0:
                # estimate sigma in early steps (fixed later)
                if sigma_v is None or sigma_t is None:
                    sv = sample_abs_grads_from_paramlists(visual_adapter_params, sample_size=args.sigma_sample)
                    st = sample_abs_grads_from_paramlists(text_adapter_params, sample_size=args.sigma_sample)
                    if sv.numel() > 0:
                        warm_abs_v.append(sv.detach().float().cpu())
                    if st.numel() > 0:
                        warm_abs_t.append(st.detach().float().cpu())

                    # after enough warmup recorded steps -> fix sigma
                    if len(warm_abs_v) >= args.sigma_warmup_steps and len(warm_abs_t) >= args.sigma_warmup_steps:
                        allv = torch.cat(warm_abs_v, dim=0)
                        allt = torch.cat(warm_abs_t, dim=0)
                        sigma_v = float(torch.quantile(allv, args.sigma_quantile).item())
                        sigma_t = float(torch.quantile(allt, args.sigma_quantile).item())
                        # free buffers
                        warm_abs_v.clear()
                        warm_abs_t.clear()
                        print(f"[SIGMA] Fixed sigma_v={sigma_v:.3e}, sigma_t={sigma_t:.3e} "
                              f"(q={args.sigma_quantile}, warmup_steps={args.sigma_warmup_steps})")

                # If sigma not ready yet, use a provisional sigma from current samples
                if sigma_v is None:
                    sigma_v_now = float(torch.quantile(warm_abs_v[-1], args.sigma_quantile).item()) if warm_abs_v else 0.0
                else:
                    sigma_v_now = sigma_v

                if sigma_t is None:
                    sigma_t_now = float(torch.quantile(warm_abs_t[-1], args.sigma_quantile).item()) if warm_abs_t else 0.0
                else:
                    sigma_t_now = sigma_t

                rv = rho_per_layer(visual_adapter_params, sigma=sigma_v_now)
                rt = rho_per_layer(text_adapter_params, sigma=sigma_t_now)

                rho_v_records.append(rv)
                rho_t_records.append(rt)
                meta.append((task_id, step_in_task, global_step, float(loss.item()), prompt_mode, aug_mode, float(drop_p)))

            opt.step()

            losses.append(float(loss.item()))
            global_step += 1

        avg_loss = sum(losses) / max(1, len(losses))
        print(f"[TASK {task_id+1:02d}/{args.num_tasks}] prompt={prompt_mode} aug={aug_mode} drop_p={drop_p:.2f} "
              f"avg_loss={avg_loss:.4f} | global_step={global_step} | elapsed={(time.time()-t0)/60:.1f} min")

    Rv = np.stack(rho_v_records, axis=0)  # [T, Lv]
    Rt = np.stack(rho_t_records, axis=0)  # [T, Lt]
    meta_np = np.array(meta, dtype=object)

    # Selection by rise->fall on rho
    v_stats, v_ranked = score_rise_then_fall_rho(Rv, prefer_last_ratio=args.prefer_last_ratio)
    t_stats, t_ranked = score_rise_then_fall_rho(Rt, prefer_last_ratio=args.prefer_last_ratio)

    top_v = v_ranked[:args.topk]
    top_t = t_ranked[:args.topk]

    print("\n[RESULT] Selected learngene blocks (VISUAL):", top_v)
    for i in top_v:
        s = v_stats[i]
        print(f"  - B{i}: early={s['early_mean']:.4f} late={s['late_mean']:.4f} peak={s['peak']:.4f} "
              f"peak_step={s['peak_step']} score={s['score']:.6f}")

    print("\n[RESULT] Selected learngene blocks (TEXT):", top_t)
    for i in top_t:
        s = t_stats[i]
        print(f"  - B{i}: early={s['early_mean']:.4f} late={s['late_mean']:.4f} peak={s['peak']:.4f} "
              f"peak_step={s['peak_step']} score={s['score']:.6f}")

    # Save logs
    np.savez_compressed(
        os.path.join(args.out_dir, "rho_logs.npz"),
        Rv=Rv, Rt=Rt,
        meta=meta_np,
        sigma_v=np.array([0.0 if sigma_v is None else sigma_v], dtype=np.float32),
        sigma_t=np.array([0.0 if sigma_t is None else sigma_t], dtype=np.float32),
        topk_visual=np.array(top_v, dtype=np.int64),
        topk_text=np.array(top_t, dtype=np.int64),
    )

    # Plots
    plot_heatmap_rho(Rv, f"VISUAL Adapter ρ(t) heatmap | fixed σ | {args.clip_model}", os.path.join(args.out_dir, "rho_heatmap_visual.png"))
    plot_heatmap_rho(Rt, f"TEXT Adapter ρ(t) heatmap | fixed σ | {args.clip_model}", os.path.join(args.out_dir, "rho_heatmap_text.png"))

    # Trend curves: topk + refs
    ref_v = sorted(set(top_v + [max(0, n_visual//4), max(0, n_visual//2), max(0, n_visual-1)]))
    ref_t = sorted(set(top_t + [max(0, n_text//4), max(0, n_text//2), max(0, n_text-1)]))
    plot_trends(Rv, ref_v, "VISUAL Adapter ρ(t) trends (topk + refs)", os.path.join(args.out_dir, "rho_trends_visual.png"))
    plot_trends(Rt, ref_t, "TEXT Adapter ρ(t) trends (topk + refs)", os.path.join(args.out_dir, "rho_trends_text.png"))

    # Save selected info
    with open(os.path.join(args.out_dir, "selected_layers.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_model": args.clip_model,
                "num_tasks": args.num_tasks,
                "steps_per_task": args.steps_per_task,
                "batch_size": args.batch_size,
                "adapter_bottleneck": args.adapter_bottleneck,
                "adapter_dropout": args.adapter_dropout,
                "lr": args.lr,
                "wd": args.wd,
                "sigma_warmup_steps": args.sigma_warmup_steps,
                "sigma_quantile": args.sigma_quantile,
                "sigma_sample": args.sigma_sample,
                "sigma_v": None if sigma_v is None else float(sigma_v),
                "sigma_t": None if sigma_t is None else float(sigma_t),
                "prefer_last_ratio": args.prefer_last_ratio,
                "topk": args.topk,
                "task_split": "topic+length buckets",
                "task_shift": "per-task prompt template + augmentation mode + word dropout",
                "visual": {"selected": top_v, "stats": {str(i): v_stats[i] for i in top_v}},
                "text": {"selected": top_t, "stats": {str(i): t_stats[i] for i in top_t}},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save original CLIP block weights for selected indices (as your learngene layers)
    save_gene_layers_clip_original_blocks(model, top_v, top_t, args.out_dir)

    # Save all adapters (optional, useful for debugging)
    save_all_adapters_state(model, args.out_dir)

    print(f"\n[DONE] Saved to: {args.out_dir}")
    print("  - rho_logs.npz")
    print("  - rho_heatmap_visual.png / rho_heatmap_text.png")
    print("  - rho_trends_visual.png / rho_trends_text.png")
    print("  - selected_layers.json")
    print("  - learngene_visual.pt / learngene_text.pt / learngene_multimodal.pt")
    print("  - adapters_state.pt")


if __name__ == "__main__":
    main()
