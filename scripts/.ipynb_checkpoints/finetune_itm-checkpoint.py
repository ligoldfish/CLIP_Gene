# scripts/finetune_itm.py
# Finetune image-text matching (ITM) on COCO captions (pos/neg sampling)
#
# Updates:
#  - Add TEST split into finetune loop (evaluate test every val eval, or only when val improves)
#  - Save best checkpoint based on selected val metric
#  - Report Params / FLOPs (+ optional latency / peak memory)
#  - AMP dtype support (bf16 preferred), scaler only for fp16
#
# Example:
# torchrun --nproc_per_node=4 -m scripts.finetune_itm \
#   --distributed \
#   --model ours --gene_dir /root/gene_exports/last2 \
#   --init_ckpt outputs/pretrain_ours_last2/ckpt_last.pt \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --coco_val_images /root/autodl-tmp/val2017 \
#   --coco_val_captions /root/autodl-tmp/annotations/captions_val2017.json \
#   --coco_test_images /root/autodl-tmp/val2017 \
#   --coco_test_captions /root/autodl-tmp/annotations/captions_val2017.json \
#   --epochs 10 --batch_size 256 --amp --amp_dtype bf16 \
#   --val_every 1 --eval_test 1 --test_when_best \
#   --select_metric auc \
#   --out_dir outputs/ft_itm_ours_with_test
# torchrun --nproc_per_node=1 -m scripts.finetune_itm \
#   --model ours \
#   --gene_dir /root/gene_exports/last3 \
#   --shallow_layers 3 \
#   --use_tleg --tleg_target_depth 6 \
#   --freeze_gene \
#   --amp --amp_dtype bf16 \
#   --eval_only --eval_split test \
#   --init_ckpt outputs/ft_itm_ours_last3/model_best.pt \
#   --coco_val_images /root/autodl-tmp/val2017 \
#   --coco_val_captions /root/autodl-tmp/annotations/captions_val2017.json \
#   --val_batch_size 256 --num_workers 12 \
#   --flops_backend auto --flops_mode image_cached_text \
#   --out_dir outputs/eval_itm_ours_last3
# torchrun --nproc_per_node=1 -m scripts.finetune_itm \
#   --model ours \
#   --gene_dir /root/gene_exports/last2 \
#   --shallow_layers 3 \
#   --use_tleg --tleg_target_depth 4 --tleg_strict\
#   --freeze_gene \
#   --init_ckpt outputs/pretrain_ours_last2/ckpt_last.pt \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --coco_val_images /root/autodl-tmp/val2017 \
#   --coco_val_captions /root/autodl-tmp/annotations/captions_val2017.json \
#   --coco_test_images /root/autodl-tmp/val2017 \
#   --coco_test_captions /root/autodl-tmp/annotations/captions_val2017.json \
#   --epochs 10 --batch_size 256 \
#   --amp --amp_dtype bf16 \
#   --val_every 1 --eval_test 1 --test_when_best \
#   --select_metric auc \
#   --out_dir outputs/ft_itm_ours_last2

import os
import sys
import json
import time
import argparse
from typing import Tuple, Any, Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.data.transforms import build_clip_image_transform
from scripts.model_factory import (
    create_model_bundle,
    split_param_groups,
    set_requires_grad,
)
from scripts.optim import cosine_lr, set_optimizer_lrs
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint, load_checkpoint

# reuse dataset logic from tasks (already in your repo)
from tasks.matching import COCOMatchingDataset


# --- TF32 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _arg_was_set(flag: str) -> bool:
    """Return True if a CLI flag was explicitly provided (supports '--x y' and '--x=y')."""
    if flag in sys.argv:
        return True
    prefix = flag + "="
    return any(a.startswith(prefix) for a in sys.argv)


def _count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


@torch.no_grad()
def _profile_flops_latency_mem(
    model: nn.Module,
    forward_fn,
    profile_speed: bool,
    iters: int = 50,
    flops_backend: str = "auto",
) -> Dict[str, Optional[float]]:
    """
    forward_fn: a zero-arg closure that runs ONE forward on dummy inputs.

    Returns:
      - flops_total: total FLOPs counted by the selected backend (may be None if unavailable)
      - latency_ms: average latency for forward_fn (optional, CUDA only)
      - peak_mem_mb: peak CUDA memory allocated during repeated forward_fn (optional)
    """
    out: Dict[str, Optional[float]] = {"flops_total": None, "latency_ms": None, "peak_mem_mb": None}

    def _try_flop_counter() -> Optional[float]:
        try:
            from torch.utils.flop_counter import FlopCounterMode  # torch >= 2.1
            with FlopCounterMode(model, display=False) as fc:
                forward_fn()
            return float(fc.get_total_flops())
        except Exception:
            return None

    def _try_profiler() -> Optional[float]:
        try:
            from torch.profiler import profile, ProfilerActivity
            acts = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            with profile(activities=acts, record_shapes=False, with_flops=True) as prof:
                forward_fn()
            total = 0.0
            for e in prof.key_averages():
                f = getattr(e, "flops", None)
                if f is not None:
                    total += float(f)
            return total if total > 0 else None
        except Exception:
            return None

    backend = (flops_backend or "auto").lower()
    flops: Optional[float] = None
    if backend in ("auto", "flop_counter"):
        flops = _try_flop_counter()
        if flops is None and backend == "flop_counter":
            flops = None
    if flops is None and backend in ("auto", "profiler"):
        flops = _try_profiler()

    out["flops_total"] = flops

    # latency / peak memory (CUDA only)
    if profile_speed and torch.cuda.is_available():
        device = next(model.parameters()).device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # warmup
        for _ in range(10):
            forward_fn()
        torch.cuda.synchronize(device)

        t0 = time.time()
        for _ in range(int(iters)):
            forward_fn()
        torch.cuda.synchronize(device)
        t1 = time.time()

        out["latency_ms"] = 1000.0 * (t1 - t0) / float(iters)
        out["peak_mem_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)

    return out


def get_logit_scale(model: nn.Module) -> torch.Tensor:
    # Prefer CLIP-style logit_scale when present.
    if hasattr(model, "logit_scale"):
        try:
            return model.logit_scale.float().exp().clamp(max=50.0)
        except Exception:
            pass
    return torch.tensor(1.0 / 0.07, device=next(model.parameters()).device)


def forward_features(base_model: nn.Module, images: torch.Tensor, tokens: torch.Tensor):
    """Return (image_features, text_features)."""
    # OpenAI CLIP / OpenCLIP style
    if hasattr(base_model, "encode_image") and hasattr(base_model, "encode_text"):
        return base_model.encode_image(images), base_model.encode_text(tokens)

    # StudentCLIP style in this repo
    if all(hasattr(base_model, k) for k in ["vision_stem", "text_stem", "vision_tower", "text_tower"]):
        v_tokens = base_model.vision_stem(images)
        t_tokens = base_model.text_stem(tokens)
        z_img = base_model.vision_tower(v_tokens)
        z_txt = base_model.text_tower(t_tokens, text=tokens)
        return z_img, z_txt

    raise RuntimeError(
        "Model does not support feature extraction (need encode_image/encode_text or StudentCLIP stems/towers)."
    )


def _encode_image_only(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode image only (no text path)."""
    if hasattr(base_model, "encode_image"):
        return base_model.encode_image(images)
    if all(hasattr(base_model, k) for k in ["vision_stem", "vision_tower"]):
        v_tokens = base_model.vision_stem(images)
        return base_model.vision_tower(v_tokens)
    raise RuntimeError("Model does not support image encoding.")


def _encode_text_only(base_model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Encode text only (no image path)."""
    if hasattr(base_model, "encode_text"):
        return base_model.encode_text(tokens)
    if all(hasattr(base_model, k) for k in ["text_stem", "text_tower"]):
        t_tokens = base_model.text_stem(tokens)
        return base_model.text_tower(t_tokens, text=tokens)
    raise RuntimeError("Model does not support text encoding.")


@torch.no_grad()
def _infer_vision_token_count(base_model: nn.Module, image_size: int, device: str) -> Optional[int]:
    """Best-effort inference of vision token count N for logging/debugging."""
    try:
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        if hasattr(base_model, "vision_stem"):
            v = base_model.vision_stem(dummy)
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                return int(v.shape[1])
        # common CLIP/OpenCLIP variants
        if hasattr(base_model, "visual") and hasattr(base_model.visual, "positional_embedding"):
            pe = base_model.visual.positional_embedding
            if isinstance(pe, torch.Tensor) and pe.ndim >= 1:
                return int(pe.shape[0])
        if hasattr(base_model, "vision_model") and hasattr(base_model.vision_model, "embeddings"):
            emb = base_model.vision_model.embeddings
            if hasattr(emb, "position_embedding") and hasattr(emb.position_embedding, "weight"):
                return int(emb.position_embedding.weight.shape[0])
    except Exception:
        return None
    return None


@torch.no_grad()
def _run_profile_once(
    args,
    model: nn.Module,
    tokenize,
    amp_dtype: torch.dtype,
    tag: str,
) -> Dict[str, Any]:
    """Profile params/FLOPs (optionally latency/mem) for a chosen FLOPs mode."""
    base_model = unwrap_model(model)
    base_model.eval()

    # infer context length from tokenize
    dummy_tokens = tokenize(["a"])
    context_len = int(dummy_tokens.shape[-1])

    images = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
    toks = torch.zeros(1, context_len, dtype=torch.long, device=args.device)

    # optional cached text (excluded from FLOPs when using *_cached_text)
    cached_txt: Optional[torch.Tensor] = None
    if getattr(args, "flops_mode", "pair") in ("image_cached_text", "pair_cached_text"):
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            cached_txt = _encode_text_only(base_model, toks)
            cached_txt = F.normalize(cached_txt.float(), dim=-1)

    def _fw():
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            mode = getattr(args, "flops_mode", "pair")
            if mode == "image":
                _ = _encode_image_only(base_model, images)
                return
            if mode == "text":
                _ = _encode_text_only(base_model, toks)
                return

            # pair-like modes: by default count ONLY encoders (common reporting in papers).
            img_f = _encode_image_only(base_model, images)

            if mode in ("image_cached_text", "pair_cached_text"):
                assert cached_txt is not None
                txt_f = cached_txt
            else:
                txt_f = _encode_text_only(base_model, toks)

            if getattr(args, "flops_post", False):
                img_f = F.normalize(img_f.float(), dim=-1)
                txt_f = F.normalize(txt_f.float(), dim=-1)
                _ = get_logit_scale(base_model) * torch.sum(img_f * txt_f, dim=-1)
            else:
                # do nothing further; post ops excluded from FLOPs
                _ = None


    p = _count_params(base_model)
    vision_n = _infer_vision_token_count(base_model, args.image_size, args.device)
    prof = _profile_flops_latency_mem(
        base_model,
        _fw,
        profile_speed=args.profile_speed,
        iters=args.profile_iters,
        flops_backend=getattr(args, "flops_backend", "auto"),
    )

    flops = prof["flops_total"]
    out = {
        "tag": tag,
        "flops_mode": getattr(args, "flops_mode", "pair"),
        "flops_backend": getattr(args, "flops_backend", "auto"),
        "params_total": p["total"],
        "params_trainable": p["trainable"],
        "flops_total": (None if flops is None else int(flops)),
        "vision_tokens": vision_n,
        "text_context_len": int(context_len),
    }
    if prof.get("latency_ms") is not None:
        out["latency_ms"] = float(prof["latency_ms"])
        out["peak_mem_mb"] = float(prof["peak_mem_mb"])

    # optional MACs reporting
    if getattr(args, "flops_report_macs", False) and flops is not None:
        out["macs_total"] = int(flops // 2)

    if is_main_process():
        flops_str = "None" if out["flops_total"] is None else str(out["flops_total"])
        msg = (
            f"[PROFILE][{tag}] mode={out['flops_mode']} backend={out['flops_backend']} "
            f"params_total={out['params_total']} trainable={out['params_trainable']} flops={flops_str}"
        )
        if out["vision_tokens"] is not None:
            msg += f" vision_tokens={out['vision_tokens']}"
        msg += f" text_len={out['text_context_len']}"
        if "macs_total" in out:
            msg += f" macs={out['macs_total']}"
        print(msg)
        if "latency_ms" in out:
            print(f"[PROFILE][{tag}] latency_ms={out['latency_ms']:.3f} peak_mem_mb={out['peak_mem_mb']:.1f}")

    return out




def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "net", "module"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                sd = ckpt_obj[key]
                if any(k.startswith("module.") for k in sd.keys()):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                return sd
        if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(torch.is_tensor(v) for v in ckpt_obj.values()):
            sd = ckpt_obj
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            return sd
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def load_init_ckpt_if_any(base_model: nn.Module, init_ckpt: str):
    """Load a checkpoint into base_model with strict=False.

    NOTE: This must never change model behavior: it only loads matching keys and reports the rest.
    """
    if not init_ckpt:
        return
    obj = load_checkpoint(init_ckpt, map_location="cpu")
    sd = _extract_state_dict(obj)

    # load_state_dict returns an IncompatibleKeys object with missing_keys / unexpected_keys
    msg = base_model.load_state_dict(sd, strict=False)
    if is_main_process():
        print(f"[INIT] loaded: {init_ckpt}")
        print(f"[INIT] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        print("missing sample:", msg.missing_keys[:30])
        print("unexpected sample:", msg.unexpected_keys[:30])


def _is_no_decay_param(name: str, p: nn.Parameter) -> bool:
    n = name.lower()
    return (
        p.ndim == 1
        or n.endswith(".bias")
        or n.endswith("bias")
        or "ln" in n
        or "layernorm" in n
        or "logit_scale" in n
    )


def _build_adamw_groups(
    model: nn.Module,
    params: List[nn.Parameter],
    lr: float,
    wd: float,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Build AdamW groups with (decay, no_decay). Returns (groups, indices_of_these_groups)."""
    name_map = {id(p): n for n, p in model.named_parameters()}
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        n = name_map.get(id(p), "")
        if _is_no_decay_param(n, p):
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    ids: List[int] = []
    if len(decay) > 0:
        ids.append(len(groups))
        groups.append({"params": decay, "lr": lr, "weight_decay": wd})
    if len(no_decay) > 0:
        ids.append(len(groups))
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups, ids


def build_optimizer(args, base_model: nn.Module) -> Tuple[torch.optim.Optimizer, List[nn.Parameter], List[int], List[int]]:
    """Return (optimizer, gene_params, new_group_ids, gene_group_ids)."""

    if args.model == "ours":
        new_params, gene_params = split_param_groups(base_model, args.gene_keywords)
        if len(gene_params) == 0:
            if is_main_process():
                print("[WARN] gene_params empty; fallback to single (new) group.")
            new_params = [p for p in base_model.parameters()]

        if args.freeze_gene:
            set_requires_grad(gene_params, False)

        new_groups, _ = _build_adamw_groups(base_model, new_params, lr=args.lr, wd=args.wd)
        gene_groups, _ = _build_adamw_groups(base_model, gene_params, lr=0.0, wd=args.wd)

        param_groups = []
        new_group_ids: List[int] = []
        gene_group_ids: List[int] = []

        for g in new_groups:
            new_group_ids.append(len(param_groups))
            param_groups.append(g)

        for g in gene_groups:
            gene_group_ids.append(len(param_groups))
            param_groups.append(g)

        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
        return opt, list(gene_params), new_group_ids, gene_group_ids

    # clip / tinyclip baselines
    all_params = [p for p in base_model.parameters()]
    groups, _ = _build_adamw_groups(base_model, all_params, lr=args.lr, wd=args.wd)
    if len(groups) == 0:
        raise RuntimeError("No trainable parameters found.")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)
    return opt, [], list(range(len(groups))), []


def build_loader(args, tokenize, transform, split: str = "train"):
    if split == "train":
        img_dir = args.coco_images
        captions_json = args.coco_captions
        bs = args.batch_size
        shuffle = True
        drop_last = True
    elif split == "val":
        img_dir = args.coco_val_images
        captions_json = args.coco_val_captions
        bs = args.val_batch_size
        shuffle = False
        drop_last = False
    else:  # test
        img_dir = args.coco_test_images
        captions_json = args.coco_test_captions
        bs = args.val_batch_size
        shuffle = False
        drop_last = False

    ds = COCOMatchingDataset(
        img_dir=img_dir,
        captions_json=captions_json,
        pos_ratio=args.pos_ratio,
        seed=args.seed,
    )

    def collate(batch):
        # batch: (pil_image, caption, label)
        images = torch.stack([transform(b[0]) for b in batch], 0)
        texts = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        tokens = tokenize(texts)
        return images, tokens, labels

    if args.distributed and split == "train":
        sampler = DistributedSampler(
            ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,
        shuffle=(shuffle if sampler is None else False),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate,
    )
    return ds, loader, sampler


def _roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return float("nan")
    P = int(labels.sum())
    N = int(labels.size - P)
    if P == 0 or N == 0:
        return float("nan")

    order = np.argsort(scores)[::-1]
    s = scores[order]
    y = labels[order]

    tps = 0.0
    fps = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0
    i = 0
    n = y.size
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        y_group = y[i:j]
        tps += float(y_group.sum())
        fps += float((j - i) - y_group.sum())
        tpr = tps / P
        fpr = fps / N
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
        i = j
    return float(auc)


def _best_threshold_from_probs(probs: np.ndarray, labels: np.ndarray, grid: int = 201) -> Tuple[float, float]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.size == 0:
        return 0.5, float("nan")
    ths = np.linspace(0.0, 1.0, int(grid))
    best_thr = 0.5
    best_acc = -1.0
    for t in ths:
        pred = (probs > t).astype(np.int64)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return best_thr, best_acc


@torch.no_grad()
def evaluate(args, model, loader, amp_dtype: torch.dtype) -> Tuple[float, float, float, float, float]:
    """Return (loss, acc@0.5, auc, best_thr, best_acc)."""
    base_model = unwrap_model(model)
    base_model.eval()
    device = args.device

    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    n = 0
    correct05 = 0
    total = 0

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, tokens, labels in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            img_f, txt_f = forward_features(base_model, images, tokens)
            img_f = F.normalize(img_f.float(), dim=-1)
            txt_f = F.normalize(txt_f.float(), dim=-1)
            logit_scale = get_logit_scale(base_model)
            logits = logit_scale * torch.sum(img_f * txt_f, dim=-1)  # [B]
            loss = loss_fn(logits, labels)

        bs = int(labels.numel())
        loss_sum += float(loss.item()) * bs
        n += bs

        probs = torch.sigmoid(logits.float())
        pred05 = (probs > 0.5).long()
        correct05 += int((pred05 == labels.long()).sum().item())
        total += bs

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())

    avg_loss = loss_sum / max(1, n)
    acc05 = correct05 / max(1, total)

    logits_np = torch.cat(all_logits, dim=0).numpy() if len(all_logits) else np.zeros((0,), dtype=np.float32)
    labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.int64) if len(all_labels) else np.zeros((0,), dtype=np.int64)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    auc = _roc_auc_from_scores(probs_np, labels_np)
    best_thr, best_acc = _best_threshold_from_probs(probs_np, labels_np, grid=args.thr_grid)
    return float(avg_loss), float(acc05), float(auc), float(best_thr), float(best_acc)


def train_one_epoch(
    args,
    model,
    loader,
    sampler,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    total_steps: int,
    unfreeze_step: int,
    amp_dtype: torch.dtype,
):
    model.train()
    base_model = unwrap_model(model)
    device = args.device
    loss_fn = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    loss_count = 0
    correct = 0
    total = 0

    # staged unfreeze for ours
    if args.model == "ours":
        if args.freeze_gene:
            set_requires_grad(args._gene_params, False)
        else:
            if epoch < args.unfreeze_epoch:
                set_requires_grad(args._gene_params, False)
            else:
                set_requires_grad(args._gene_params, True)

    if sampler is not None:
        sampler.set_epoch(epoch)

    for it, (images, tokens, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        lr_new = cosine_lr(global_step, total_steps, args.lr, args.min_lr)

        gene_lr: Optional[float] = None
        if len(args._lr_group_ids_gene) > 0:
            if args.freeze_gene or global_step < unfreeze_step:
                gene_lr = 0.0
            else:
                target = lr_new / args.gene_lr_ratio
                if args.gene_warmup_steps <= 0:
                    gene_lr = target
                else:
                    w = min(1.0, (global_step - unfreeze_step) / float(args.gene_warmup_steps))
                    gene_lr = target * w

        # set lrs for each param group
        lrs: List[float] = []
        for gi in range(len(optimizer.param_groups)):
            if gi in args._lr_group_ids_gene:
                lrs.append(float(gene_lr if gene_lr is not None else 0.0))
            else:
                lrs.append(float(lr_new))
        set_optimizer_lrs(optimizer, lrs)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            img_f, txt_f = forward_features(base_model, images, tokens)
            img_f = F.normalize(img_f.float(), dim=-1)
            txt_f = F.normalize(txt_f.float(), dim=-1)
            logit_scale = get_logit_scale(base_model)
            logits = logit_scale * torch.sum(img_f * txt_f, dim=-1)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
            optimizer.step()

        loss_sum += float(loss.item())
        loss_count += 1
        global_step += 1

        probs = torch.sigmoid(logits.detach().float())
        preds = (probs > 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += int(labels.numel())

        # diagnostics
        if is_main_process() and (global_step % args.diag_every == 0):
            pos = probs[labels > 0.5]
            neg = probs[labels <= 0.5]
            pos_m = float(pos.mean().item()) if pos.numel() else float("nan")
            neg_m = float(neg.mean().item()) if neg.numel() else float("nan")
            ls = float(get_logit_scale(base_model).detach().float().mean().item())
            acc = correct / max(1, total)

            lr_new_show = float(np.mean([optimizer.param_groups[i]["lr"] for i in args._lr_group_ids_new]))
            if len(args._lr_group_ids_gene) > 0:
                lr_gene_show = float(np.mean([optimizer.param_groups[i]["lr"] for i in args._lr_group_ids_gene]))
                lr_info = f"lr_new={lr_new_show:.2e} lr_gene={lr_gene_show:.2e}"
            else:
                lr_info = f"lr={lr_new_show:.2e}"
            print(
                f"[E{epoch:03d}][{it:05d}] step={global_step} "
                f"loss={loss.item():.4f} acc@0.5={acc:.3f} "
                f"logit_scale={ls:.2f} p(pos)={pos_m:.3f} p(neg)={neg_m:.3f} {lr_info}"
            )

    avg_loss = loss_sum / max(1, loss_count)
    acc = correct / max(1, total)
    return global_step, float(avg_loss), float(acc)


def _select_score(select_metric: str, loss: float, auc: float, best_acc: float) -> float:
    if select_metric == "auc":
        return float(auc)
    if select_metric == "loss":
        return -float(loss)
    return float(best_acc)  # "acc"


def main():
    parser = argparse.ArgumentParser("Finetune ITM on COCO (ours/clip/tinyclip) + val/test + profile")

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)

    # ours knobs
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    # TLEG options kept for compatibility

    # TLEG is ON by default (recommended); use --no_tleg to disable.
    parser.add_argument("--use_tleg", action="store_true", default=True)
    parser.add_argument("--no_tleg", action="store_true", help="Disable TLEG even if enabled by default.")
    parser.add_argument("--tleg_target_depth", type=int, default=6)
    parser.add_argument("--tleg_strict", action="store_true", help="paper-faithful strict TLEG (only 2 endpoints trainable)")
    parser.add_argument("--use_multimodal_init", action="store_true")

    # CLIP stem init (ours): default ON unless disabled
    parser.add_argument(
        "--disable_stem_init_from_clip",
        action="store_true",
        help="Disable copying CLIP stem weights into the student (not recommended unless debugging).",
    )
    parser.add_argument("--stem_init_clip_name", type=str, default="ViT-B/32", help="Teacher CLIP name used for stem init.")
    parser.add_argument("--freeze_stem_after_init", action="store_true", help="Freeze stem params after copying from CLIP.")

    # baselines
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--tinyclip_ckpt", type=str, default="")

    # init ckpt
    parser.add_argument("--init_ckpt", type=str, default="")

    # evaluation-only
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (val/test) on a checkpoint (use --init_ckpt) and exit.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["val", "test", "both"],
        help="Which split(s) to evaluate in --eval_only mode.",
    )

    # data
    parser.add_argument("--coco_images", type=str, default="")
    parser.add_argument("--coco_captions", type=str, default="")

    # val
    parser.add_argument("--coco_val_images", type=str, default="")
    parser.add_argument("--coco_val_captions", type=str, default="")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--thr_grid", type=int, default=201)

    # test (融入微调流程)
    parser.add_argument("--coco_test_images", type=str, default="")
    parser.add_argument("--coco_test_captions", type=str, default="")
    parser.add_argument("--eval_test", type=int, default=1, choices=[0, 1],
                        help="1: run test evaluation along with val; 0: disable test")
    parser.add_argument("--test_when_best", action="store_true",
                        help="If set: only run test when val improves (recommended).")
    parser.add_argument("--select_metric", type=str, default="auc", choices=["auc", "acc", "loss"],
                        help="Which val metric defines 'best' checkpoint.")

    parser.add_argument("--pos_ratio", type=float, default=0.5)

    # train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.1)

    # AMP
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"],
                        help="bf16 is usually more stable; fp16 uses GradScaler.")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # freeze/unfreeze gene (ours)
    parser.add_argument("--freeze_gene", action="store_true")
    parser.add_argument("--unfreeze_epoch", type=int, default=1)
    parser.add_argument("--gene_lr_ratio", type=float, default=10.0)
    parser.add_argument("--gene_warmup_steps", type=int, default=200)
    parser.add_argument("--gene_keywords", type=str, nargs="+", default=["learngene", "gene"])

    # dist
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")

    # profile (lightweight advantage metrics)
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--profile_speed", action="store_true")
    parser.add_argument("--profile_iters", type=int, default=50)

    parser.add_argument("--flops_mode", type=str, default="pair",
                            choices=["pair", "image", "text", "image_cached_text", "pair_cached_text"],
                            help="FLOPs counting mode. 'pair' counts image+text+sim. "
                                 "'image'/'text' count a single encoder. "
                                 "'*_cached_text' precomputes text once and excludes it from per-forward FLOPs (deploy-style).")
    parser.add_argument("--flops_backend", type=str, default="auto",
                            choices=["auto", "flop_counter", "profiler"],
                            help="Backend for FLOPs counting. auto: try flop_counter then profiler.")
    parser.add_argument("--flops_report_macs", action="store_true",
                            help="Also report MACs (FLOPs/2).")
    parser.add_argument("--flops_post", action="store_true",
                        help="Also count post-processing FLOPs (normalize + dot + logit_scale). By default only encoders are counted.")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--diag_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/ft_itm_o")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_full_ckpt", action="store_true")

    args = parser.parse_args()

    # ours: default stem init is ON unless explicitly disabled
    args.stem_init_from_clip = (not getattr(args, 'disable_stem_init_from_clip', False))
    if getattr(args, 'no_tleg', False):
        args.use_tleg = False

    # AMP dtype resolve
    if args.amp:
        if args.amp_dtype == "auto":
            args.amp_dtype = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
        if args.amp_dtype == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            if is_main_process():
                print("[WARN] bf16 not supported; fallback to fp16")
            args.amp_dtype = "fp16"
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    else:
        amp_dtype = torch.float16  # unused when amp disabled

    # scaler only for fp16
    scaler = None
    if args.amp and args.amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # CLIP-safe defaults if user didn't specify
    if args.model in ("clip", "tinyclip"):
        if (not _arg_was_set("--lr")) and (args.lr == 1e-4):
            args.lr = 1e-5
        if (not _arg_was_set("--wd")) and (args.wd == 0.1):
            args.wd = 0.01

    # keep diag_every aligned with log_every unless explicitly set
    if not _arg_was_set("--diag_every"):
        args.diag_every = args.log_every

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend)

    # data arg validation
    if args.eval_only:
        if args.model == "ours" and (not args.init_ckpt):
            raise ValueError("--init_ckpt is required for --model ours when using --eval_only")
    else:
        if (not args.coco_images) or (not args.coco_captions):
            raise ValueError("--coco_images and --coco_captions are required for training (omit only with --eval_only)")

    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")

    # model
    bundle = create_model_bundle(args)
    model = bundle.model
    tokenize = bundle.tokenize

    if args.distributed and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    base_model = unwrap_model(model)
    base_model.float()  # good default for stability

    if args.init_ckpt:
        load_init_ckpt_if_any(base_model, args.init_ckpt)

    # -------------------------
    # Eval-only mode
    # -------------------------
    if args.eval_only:
        # build loaders only for requested splits (no training/optimizer)
        transform_eval = build_clip_image_transform(args.image_size, is_train=False)

        # profile in eval-only as well (so CLIP/TinyCLIP also prints params/FLOPs)
        if is_main_process() and (not args.skip_profile):
            try:
                _ = _run_profile_once(args, model, tokenize, amp_dtype=amp_dtype, tag="EVAL")
            except Exception as e:
                print(f"[PROFILE] failed: {type(e).__name__}: {e}")

        val_loader = None
        test_loader = None

        if args.eval_split in ("val", "both"):
            if args.coco_val_images and args.coco_val_captions:
                _, val_loader, _ = build_loader(args, tokenize, transform_eval, split="val")
            else:
                if is_main_process():
                    print("[EVAL] val requested but coco_val_* not provided; skip val.")

        if args.eval_split in ("test", "both"):
            # If user didn't provide test, fall back to val (common for COCO)
            if (not args.coco_test_images) or (not args.coco_test_captions):
                if args.coco_val_images and args.coco_val_captions:
                    args.coco_test_images = args.coco_val_images
                    args.coco_test_captions = args.coco_val_captions
                    if is_main_process():
                        print("[EVAL] coco_test_* not set; fallback to val split as test.")
                else:
                    if is_main_process():
                        print("[EVAL] test requested but coco_test_* not provided and no val to fallback; skip test.")
            if args.coco_test_images and args.coco_test_captions:
                _, test_loader, _ = build_loader(args, tokenize, transform_eval, split="test")

        results = {"ckpt": args.init_ckpt, "select_metric": args.select_metric}

        if is_main_process():
            print("==== ITM Eval-Only ====")
            print(f"ckpt: {args.init_ckpt}")
            print(f"eval_split: {args.eval_split}")
            print("================================")

        if val_loader is not None:
            v_loss, v_acc05, v_auc, v_thr, v_best_acc = evaluate(args, model, val_loader, amp_dtype=amp_dtype)
            if is_main_process():
                print(
                    f"[VAL] loss={v_loss:.4f} acc@0.5={v_acc05:.3f} auc={v_auc:.3f} best_thr={v_thr:.3f} best_acc={v_best_acc:.3f}"
                )
            results.update(
                {
                    "val_loss": v_loss,
                    "val_acc05": v_acc05,
                    "val_auc": v_auc,
                    "val_best_thr": v_thr,
                    "val_best_acc": v_best_acc,
                }
            )

        if test_loader is not None:
            t_loss, t_acc05, t_auc, t_thr, t_best_acc = evaluate(args, model, test_loader, amp_dtype=amp_dtype)
            if is_main_process():
                print(
                    f"[TEST] loss={t_loss:.4f} acc@0.5={t_acc05:.3f} auc={t_auc:.3f} best_thr={t_thr:.3f} best_acc={t_best_acc:.3f}"
                )
            results.update(
                {
                    "test_loss": t_loss,
                    "test_acc05": t_acc05,
                    "test_auc": t_auc,
                    "test_best_thr": t_thr,
                    "test_best_acc": t_best_acc,
                }
            )

        if is_main_process():
            mpath = os.path.join(args.out_dir, "eval_metrics.json")
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] {mpath}")
        return

    # data loaders
    transform_train = build_clip_image_transform(args.image_size, is_train=True)
    _, train_loader, train_sampler = build_loader(args, tokenize, transform_train, split="train")

    val_loader = None
    if args.coco_val_images and args.coco_val_captions:
        transform_val = build_clip_image_transform(args.image_size, is_train=False)
        _, val_loader, _ = build_loader(args, tokenize, transform_val, split="val")

    # test loader (if enabled)
    test_loader = None
    if args.eval_test == 1:
        # If user didn't provide test, fall back to val (common for COCO because official test captions aren't public)
        if (not args.coco_test_images) or (not args.coco_test_captions):
            if args.coco_val_images and args.coco_val_captions:
                args.coco_test_images = args.coco_val_images
                args.coco_test_captions = args.coco_val_captions
                if is_main_process():
                    print("[TEST] coco_test_* not set; fallback to val split as test.")
            else:
                if is_main_process():
                    print("[TEST] disabled: no test provided and no val to fallback.")
                args.eval_test = 0

        if args.eval_test == 1:
            transform_test = build_clip_image_transform(args.image_size, is_train=False)
            _, test_loader, _ = build_loader(args, tokenize, transform_test, split="test")

    # optimizer
    optimizer, gene_params, new_group_ids, gene_group_ids = build_optimizer(args, base_model)
    args._gene_params = gene_params
    args._lr_group_ids_new = new_group_ids
    args._lr_group_ids_gene = gene_group_ids

    total_steps = args.epochs * len(train_loader)
    unfreeze_step = args.unfreeze_epoch * len(train_loader)
    global_step = 0

    # profile (Params/FLOPs/latency/mem)
    if is_main_process() and (not args.skip_profile):
        try:
            args._profile_info = _run_profile_once(args, model, tokenize, amp_dtype=amp_dtype, tag="TRAIN")
        except Exception as e:
            print(f"[PROFILE] failed: {type(e).__name__}: {e}")

    if is_main_process():
        print("==== Finetune ITM Config ====")
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            print(f"{k}: {v}")
        print("================================")
        print(f"[INFO] steps/epoch={len(train_loader)} total_steps={total_steps} unfreeze_step={unfreeze_step}")

    # best tracking
    best_val_score = -1e9
    best_record: Dict[str, Any] = {}

    for epoch in range(args.epochs):
        global_step, train_loss, train_acc = train_one_epoch(
            args,
            model,
            train_loader,
            train_sampler,
            optimizer,
            scaler,
            epoch,
            global_step,
            total_steps,
            unfreeze_step,
            amp_dtype=amp_dtype,
        )

        if is_main_process():
            print(f"[E{epoch:03d}] train_loss={train_loss:.4f} train_acc@0.5={train_acc:.3f}")

        do_val = (val_loader is not None) and ((epoch + 1) % args.val_every == 0) and is_main_process()
        if do_val:
            val_loss, val_acc05, val_auc, best_thr, best_acc = evaluate(args, model, val_loader, amp_dtype=amp_dtype)
            print(
                f"[E{epoch:03d}] VAL loss={val_loss:.4f} acc@0.5={val_acc05:.3f} "
                f"auc={val_auc:.3f} best_thr={best_thr:.3f} best_acc={best_acc:.3f}"
            )

            score = _select_score(args.select_metric, val_loss, val_auc, best_acc)
            improved = score > best_val_score
            if improved:
                best_val_score = score

            # test integration
            run_test = (args.eval_test == 1 and test_loader is not None and
                        (improved if args.test_when_best else True))
            test_metrics = None
            if run_test:
                t_loss, t_acc05, t_auc, t_thr, t_best_acc = evaluate(args, model, test_loader, amp_dtype=amp_dtype)
                test_metrics = {
                    "test_loss": t_loss,
                    "test_acc05": t_acc05,
                    "test_auc": t_auc,
                    "test_best_thr": t_thr,
                    "test_best_acc": t_best_acc,
                }
                # NOTE: Test metrics are recorded; printed once at the end.

            # save best
            if improved:
                wbest = os.path.join(args.out_dir, "model_best.pt")
                save_checkpoint(wbest, {"model": unwrap_model(model).state_dict(), "epoch": epoch, "global_step": global_step})
                if is_main_process():
                    print(f"[SAVE] {wbest}")

                best_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "select_metric": args.select_metric,
                    "val_score": score,
                    "val_loss": val_loss,
                    "val_acc05": val_acc05,
                    "val_auc": val_auc,
                    "val_best_thr": best_thr,
                    "val_best_acc": best_acc,
                }
                if test_metrics is not None:
                    best_record.update(test_metrics)

                # also dump metrics json
                mpath = os.path.join(args.out_dir, "best_metrics.json")
                with open(mpath, "w", encoding="utf-8") as f:
                    json.dump(best_record, f, indent=2, ensure_ascii=False)
                if is_main_process():
                    print(f"[SAVE] {mpath}")

        # save every epoch
        if is_main_process() and args.save_every_epoch:
            wpath = os.path.join(args.out_dir, f"model_epoch{epoch:03d}.pt")
            save_checkpoint(wpath, {"model": unwrap_model(model).state_dict(), "epoch": epoch, "global_step": global_step})
            print(f"[SAVE] {wpath}")

            if args.save_full_ckpt:
                cpath = os.path.join(args.out_dir, f"ckpt_epoch{epoch:03d}.pt")
                save_checkpoint(
                    cpath,
                    {
                        "model": unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": (scaler.state_dict() if scaler is not None else None),
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    },
                )
                print(f"[SAVE] {cpath}")

    # If no validation was provided but test is enabled, evaluate test once at the end.
    if (val_loader is None) and (args.eval_test == 1) and (test_loader is not None) and is_main_process():
        t_loss, t_acc05, t_auc, t_thr, t_best_acc = evaluate(args, model, test_loader, amp_dtype=amp_dtype)
        print(
            f"[FINAL][TEST] loss={t_loss:.4f} acc@0.5={t_acc05:.3f} "
            f"auc={t_auc:.3f} best_thr={t_thr:.3f} best_acc={t_best_acc:.3f}"
        )

    if is_main_process():
        wpath = os.path.join(args.out_dir, "model_last.pt")
        save_checkpoint(wpath, {"model": unwrap_model(model).state_dict(), "epoch": args.epochs - 1, "global_step": global_step})
        print(f"[SAVE] {wpath}")

        if args.save_full_ckpt:
            cpath = os.path.join(args.out_dir, "ckpt_last.pt")
            save_checkpoint(
                cpath,
                {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": (scaler.state_dict() if scaler is not None else None),
                    "epoch": args.epochs - 1,
                    "global_step": global_step,
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                },
            )
            print(f"[SAVE] {cpath}")

        if best_record:
            # Print the best metrics ONCE (requested).
            e = best_record.get("epoch")
            gs = best_record.get("global_step")
            print(
                f"[BEST][VAL] epoch={e} step={gs} "
                f"loss={best_record.get('val_loss'):.4f} acc@0.5={best_record.get('val_acc05'):.3f} "
                f"auc={best_record.get('val_auc'):.3f} best_thr={best_record.get('val_best_thr'):.3f} "
                f"best_acc={best_record.get('val_best_acc'):.3f}"
            )
            if "test_loss" in best_record:
                print(
                    f"[BEST][TEST] epoch={e} step={gs} "
                    f"loss={best_record.get('test_loss'):.4f} acc@0.5={best_record.get('test_acc05'):.3f} "
                    f"auc={best_record.get('test_auc'):.3f} best_thr={best_record.get('test_best_thr'):.3f} "
                    f"best_acc={best_record.get('test_best_acc'):.3f}"
                )
            else:
                print("[BEST][TEST] not available (test was disabled or never evaluated).")


if __name__ == "__main__":
    main()
