# scripts/pretrain_cc3m.py

"""Pretrain (contrastive) on Conceptual Captions 3M (CC3M) using WebDataset.

This script is designed to be:
  - Scalable: supports CC3M-scale tar shards via WebDataset streaming.
  - Stable: warmup+cosine LR, no-WD on bias/LN/logit_scale, grad clipping.
  - Compatible: works for ours / clip / tinyclip with the existing model_factory.

Expected dataset layout (auto-detected):
  /root/autodl-tmp/cc3m/
      train/*.tar   (recommended)
      val/*.tar     (optional)
  or
      *.tar         (all in one folder)

Example:
  cd ~/codes
  MASTER_PORT=29501 torchrun --nproc_per_node=1 -m scripts.pretrain_cc3m \
    --distributed \
    --model ours --gene_dir /root/gene_exports/last3 \
    --cc3m_root /root/autodl-tmp/cc3m \
    --image_size 224 --batch_size 256 --epochs 3 --amp \
    --warmup_steps 2000 --lr 3e-4 --min_lr 1e-5 \
    --unfreeze_epoch 1 --gene_lr_ratio 10 --gene_warmup_steps 1000 \
    --use_tleg --tleg_target_depth 6 --tleg_last_epochs 1 \
    --out_dir outputs/pretrain_cc3m_ours
"""
# torchrun --nproc_per_node=1 -m scripts.pretrain_cc3m \
#   --distributed \
#   --model ours --gene_dir /root/gene_exports/last3 \
#   --init_ckpt outputs/pretrain_ours3/ckpt_last.pt \
#   --cc3m_root /root/autodl-tmp/cc3m \
#   --cc3m_train_shards "/root/autodl-tmp/cc3m/wds/train_part_*/*.tar" \
#   --image_size 224 --batch_size 256 --epochs 30 --amp \
#   --samples_per_epoch 800000 \
#   --warmup_steps 2000 --lr 3e-4 --min_lr 1e-5 --tleg_target_depth 6\
#   --unfreeze_epoch 5 --gene_lr_ratio 10 --gene_warmup_steps 1000 --tleg_last_epochs 2\
#   --out_dir outputs/pretrain_cc3m_ours
# torchrun --nproc_per_node=4 -m scripts.pretrain_cc3m \
#   --distributed \
#   --model ours \
#   --gene_dir /root/gene_exports/last3 \
#   --shallow_layers 5 \
#   --cc3m_root /root/autodl-tmp/cc3m/wds \
#   --image_size 224 \
#   --epochs 30 \
#   --batch_size 128 \
#   --num_workers 16 \
#   --lr 1e-4 --min_lr 1e-5 --warmup_steps 2000 \
#   --wd 0.1 \
#   --amp \
#   --grad_clip 0.5 \
#   --samples_per_epoch 812000 \
#   --use_tleg --tleg_target_depth 6 --tleg_last_epochs 3 \
#   --unfreeze_epoch 4 \
#   --gene_lr_ratio 10 --gene_warmup_steps 1000 \
#   --shuffle_buf 20000 --wds_handler warn \
#   --val_samples 0 \
#   --out_dir outputs/pretrain_ours_cc3m_last3

# torchrun --nproc_per_node=4 -m scripts.pretrain_cc3m   --distributed --shallow_layers 5  --model ours --gene_dir /root/gene_exports/last2_plus6   --cc3m_root /root/autodl-tmp/cc3m/wds   --image_size 224 --epochs 70 --batch_size 128 --num_workers 16   --lr 1e-4 --min_lr 1e-5 --warmup_steps 2000 --wd 0.1   --amp --amp_dtype bf16   --grad_clip 0.5   --samples_per_epoch 812000   --unfreeze_epoch 60 --gene_lr_ratio 25 --gene_warmup_steps 1000   --use_tleg --tleg_target_depth 6 --tleg_last_epochs 5   --shuffle_buf 20000 --wds_handler warn --val_samples 0   --out_dir outputs/pretrain_ours_cc3m_last26
# torchrun --nproc_per_node=4 -m scripts.pretrain_cc3m \
#   --distributed --shallow_layers 5\
#   --model ours --gene_dir /root/gene_exports/last2_plus6 \
#   --cc3m_root /root/autodl-tmp/cc3m/wds \
#   --image_size 224 --epochs 30 --batch_size 128 --num_workers 16 \
#   --lr 1e-4 --min_lr 1e-5 --warmup_steps 2000 --wd 0.1 \
#   --amp --amp_dtype bf16 \
#   --grad_clip 0.5 \
#   --samples_per_epoch 812000 \
#   --unfreeze_epoch 22 --gene_lr_ratio 20 --gene_warmup_steps 1000 \
#   --use_tleg --tleg_target_depth 6 --tleg_last_epochs 3 \
#   --shuffle_buf 20000 --wds_handler warn --val_samples 0 \
#   --out_dir outputs/pretrain_ours_cc3m_last3

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import asdict
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from scripts.data.transforms import build_clip_image_transform
from scripts.data.webdataset_pairs import WdsPairConfig, build_wds_pairs, default_wds_collate

from scripts.model_factory import create_model_bundle, split_param_groups, set_requires_grad
from scripts.losses import clip_contrastive_loss
from scripts.optim import set_optimizer_lrs
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint
from scripts.data.mixed_wds_coco import CocoCaptionPool, MixedWdsCocoPairs


# --- TF32 (Ampere/Ada/Hopper GPUs) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def set_all_learngene_tleg_active(model: nn.Module, flag: bool):
    """Enable/disable TLEG blocks inside Learngene modules (duck-typed)."""
    try:
        from models.learngene_loader import LearngeneModule
    except Exception:
        LearngeneModule = None

    for m in model.modules():
        if LearngeneModule is not None and isinstance(m, LearngeneModule):
            if getattr(m, "is_tleg", False):
                m.set_tleg_active(flag)
        else:
            if hasattr(m, "set_tleg_active") and hasattr(m, "is_tleg") and getattr(m, "is_tleg"):
                m.set_tleg_active(flag)


def forward_features(model: nn.Module, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (image_emb, text_emb) for contrastive loss."""
    if hasattr(model, "encode_image") and hasattr(model, "encode_text"):
        return model.encode_image(images), model.encode_text(tokens)

    # StudentCLIP-style fallback
    if all(hasattr(model, k) for k in ["vision_stem", "text_stem", "vision_tower", "text_tower"]):
        v_tokens = model.vision_stem(images)
        t_tokens = model.text_stem(tokens)
        z_img = model.vision_tower(v_tokens)
        z_txt = model.text_tower(t_tokens, text=tokens)
        return z_img, z_txt

    raise RuntimeError("Model does not support encode_image/encode_text and has no known StudentCLIP stems/towers")


def get_logit_scale(model: nn.Module, max_scale: float = 100.0) -> torch.Tensor:
    if hasattr(model, "logit_scale"):
        ls = model.logit_scale
        # 强制 fp32 计算，避免 autocast 下 exp/softmax 不稳
        ls = ls.float()
        out = ls.exp()
        out = torch.nan_to_num(out, nan=1.0, posinf=max_scale, neginf=1.0)
        return out.clamp(max=max_scale)
    return torch.tensor(1.0 / 0.07, device=next(model.parameters()).device)


def clamp_logit_scale_(model: nn.Module, max_scale: float = 100.0):
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.Tensor):
        with torch.no_grad():
            ls = model.logit_scale.data
            # 先修 NaN/Inf，再 clamp
            ls.nan_to_num_(nan=0.0, posinf=math.log(max_scale), neginf=0.0)
            ls.clamp_(0.0, math.log(max_scale))


def lr_warmup_cosine(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if total_steps <= 0:
        return base_lr

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))

    # cosine on the remaining steps
    t0 = warmup_steps
    t = min(1.0, float(step - t0) / float(max(1, total_steps - t0)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _should_decay(name: str, param: torch.Tensor) -> bool:
    """CLIP-style no-weight-decay rules."""
    n = name.lower()
    if n.endswith(".bias"):
        return False
    if "logit_scale" in n:
        return False
    # LayerNorm / BatchNorm / RMSNorm weights typically should NOT decay
    if any(k in n for k in ["ln", "layernorm", "rmsnorm", "bn", "norm"]):
        # guard: don't accidentally disable wd for e.g. 'linear_norm_proj' names
        # if it's a norm module param, it is usually 1D.
        if param.ndim <= 1:
            return False
    # 1D parameters (e.g. scale, bias) usually no decay
    if param.ndim <= 1:
        return False
    return True


def build_param_groups_named(
    model: nn.Module,
    params: List[nn.Parameter],
    lr: float,
    wd: float,
    ) -> List[dict]:
    """Split a parameter list into (decay, no_decay) groups using names."""
    idset = {id(p) for p in params}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if id(p) not in idset:
            continue
        if not p.requires_grad:
            continue
        (decay if _should_decay(name, p) else no_decay).append(p)

    groups = []
    if len(decay) > 0:
        groups.append({"params": decay, "lr": lr, "weight_decay": wd})
    if len(no_decay) > 0:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def guess_cc3m_shards(root: str, split: str):
    """Guess shards under root for a given split.

    Returns either a glob pattern string or a concrete list of shard paths.
    We prefer patterns for common layouts, and fall back to recursive search
    (returning an explicit list) to handle arbitrary nesting.
    """
    cand_dirs = []
    s = split.lower()
    if s in ["train", "training"]:
        cand_dirs = [
            os.path.join(root, "train"),
            os.path.join(root, "training"),
            os.path.join(root, "cc3m_train"),
            root,
        ]
    else:
        cand_dirs = [
            os.path.join(root, "val"),
            os.path.join(root, "valid"),
            os.path.join(root, "validation"),
            os.path.join(root, "dev"),
            root,
        ]

    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        if len(glob.glob(os.path.join(d, "*.tar"))) > 0:
            return os.path.join(d, "*.tar")

    # fallback: recursive search
    tars = sorted(glob.glob(os.path.join(root, "**", "*.tar"), recursive=True))
    if len(tars) > 0:
        s = split.lower()
        if s in ["train", "training"]:
            prefer = [p for p in tars if f"{os.sep}train{os.sep}" in p or f"{os.sep}training{os.sep}" in p]
        else:
            prefer = [p for p in tars if f"{os.sep}val{os.sep}" in p or f"{os.sep}valid{os.sep}" in p or f"{os.sep}validation{os.sep}" in p]
        return prefer if len(prefer) > 0 else tars

    raise FileNotFoundError(
        f"Cannot find any .tar shards under cc3m_root={root} for split={split}. "
        f"Expected {root}/train/*.tar and optionally {root}/val/*.tar (or .tar directly under root)."
    )


@torch.no_grad()
def evaluate(args, model, val_loader: Iterable, steps: int) -> float:
    model.eval()
    base_model = unwrap_model(model)
    device = args.device

    tot_loss = 0.0
    n = 0
    it = 0
    for batch in val_loader:
        images, tokens = batch
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=args._amp_autocast_dtype):
            img_f, txt_f = forward_features(base_model, images, tokens)
            loss = clip_contrastive_loss(img_f, txt_f, get_logit_scale(base_model, args.logit_scale_max))
        if loss is None:
            # skip bad batches in eval as well
            continue
        tot_loss += float(loss.item())
        n += 1
        it += 1
        if steps > 0 and it >= steps:
            break
    model.train()
    return tot_loss / max(1, n)


def train_one_epoch(
    args,
    model,
    train_loader: Iterable,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    total_steps: int,
    unfreeze_step: int,
    steps_this_epoch: int,
):
    """Train for exactly `steps_this_epoch` optimizer steps."""
    model.train()
    base_model = unwrap_model(model)
    device = args.device

    # staged freeze/unfreeze for ours
    if args.model == "ours":
        if global_step < unfreeze_step:
            set_requires_grad(args._gene_params, False)
        else:
            set_requires_grad(args._gene_params, True)

        # Enable TLEG only in the last N epochs (and not earlier than unfreeze).
        if args.use_tleg and args.tleg_last_epochs > 0:
            tleg_on_epoch = max(args.unfreeze_epoch, args.epochs - args.tleg_last_epochs)
            enable_tleg_now = epoch >= tleg_on_epoch
            set_all_learngene_tleg_active(base_model, enable_tleg_now)
            if is_main_process() and epoch == tleg_on_epoch:
                print(f"[TLEG] enabled at epoch={epoch} (last {args.tleg_last_epochs} epochs)")

    optimizer.zero_grad(set_to_none=True)

    # micro-batch accumulation
    accum = max(1, args.accum_steps)
    micro_step = 0
    opt_steps_done = 0

    for batch in train_loader:
        images, tokens = batch
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        # If unfreeze happens mid-epoch (e.g., --unfreeze_steps), ensure gene params are toggled per step.
        if args.model == "ours" and getattr(args, "_has_gene_groups", False):
            set_requires_grad(args._gene_params, global_step >= unfreeze_step)

        # LR schedule (by optimizer step)
        lr_new = lr_warmup_cosine(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)

        if args._has_gene_groups:
            # gene lr: 0 before unfreeze, then warmup to lr_new/gene_lr_ratio
            if global_step < unfreeze_step:
                gene_lr = 0.0
            else:
                target = lr_new / args.gene_lr_ratio
                if args.gene_warmup_steps <= 0:
                    gene_lr = target
                else:
                    w = min(1.0, (global_step - unfreeze_step) / float(args.gene_warmup_steps))
                    gene_lr = target * w

            # groups: new(decay/no_decay) + gene(decay/no_decay)
            set_optimizer_lrs(optimizer, [lr_new, lr_new, gene_lr, gene_lr])
        else:
            # groups: all(decay/no_decay)
            set_optimizer_lrs(optimizer, [lr_new, lr_new])

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=args._amp_autocast_dtype):
            img_f, txt_f = forward_features(base_model, images, tokens)
            loss = clip_contrastive_loss(img_f, txt_f, get_logit_scale(base_model, args.logit_scale_max))
            if loss is not None:
                loss = loss / float(accum)

        # ---- DDP-safe: if any rank sees bad features/logits/loss, ALL ranks skip this batch ----
        bad = (loss is None)
        if (not bad) and (not torch.isfinite(loss).all()):
            bad = True

        if args.distributed:
            import torch.distributed as dist
            bad_t = torch.tensor([1 if bad else 0], device=device, dtype=torch.int)
            dist.all_reduce(bad_t, op=dist.ReduceOp.SUM)
            bad = (bad_t.item() > 0)

        if bad:
            if is_main_process():
                with torch.no_grad():
                    img_ok = torch.isfinite(img_f).all().item()
                    txt_ok = torch.isfinite(txt_f).all().item()
                    img_absmax = torch.nan_to_num(img_f).abs().max().item()
                    txt_absmax = torch.nan_to_num(txt_f).abs().max().item()
                    ls = get_logit_scale(base_model, args.logit_scale_max).float().item()
                    sc = scaler.get_scale() if args._use_scaler else None
                    print(f"[BAD BATCH] step={global_step} img_ok={img_ok} txt_ok={txt_ok} "
                          f"img_absmax={img_absmax:.3e} txt_absmax={txt_absmax:.3e} "
                          f"logit_scale(exp)={ls:.3f} scaler_scale={sc}")

                # If parameters are already corrupted, abort (skipping won't help)
                bad_param = None
                for n, p in base_model.named_parameters():
                    if not torch.isfinite(p).all():
                        bad_param = n
                        break
                if bad_param is not None:
                    print("[MODEL CORRUPTED] first non-finite param:", bad_param)
                    raise RuntimeError("Model parameters became non-finite.")

            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            continue
        if is_main_process():
            # 5.545 ~ ln(256)，给点容差
            raw_loss = (loss.item() * float(accum))
            if raw_loss > 5.0:
                with torch.no_grad():
                    img_norm = img_f.float().norm(dim=-1)
                    txt_norm = txt_f.float().norm(dim=-1)
                    print("[COLLAPSE?] loss=", raw_loss,
                          "img_norm(mean/min)=", img_norm.mean().item(), img_norm.min().item(),
                          "txt_norm(mean/min)=", txt_norm.mean().item(), txt_norm.min().item())
                    ls = get_logit_scale(base_model, args.logit_scale_max).float()
                    print("[COLLAPSE?] logit_scale(exp)=", ls.item())    
        if args._use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        micro_step += 1

        if micro_step % accum != 0:
            continue
        if unfreeze_step <= global_step < unfreeze_step + 50:
            badg = None
            for n, p in base_model.named_parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    badg = n
                    break
            if badg and is_main_process():
                print("[BAD GRAD] first non-finite UN-SCALED grad param:", badg)
            # AMP/fp16 may overflow on the first few unfreeze steps; scaler will handle it.
            if badg and (not args._use_scaler):
                raise RuntimeError("Non-finite grad right after unfreeze")
        # optimizer step
        if args._use_scaler:
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)

        if args._use_scaler:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if is_main_process() and (scale_after < scale_before) and (unfreeze_step <= global_step < unfreeze_step + 50):
                print(f"[AMP OVERFLOW] step={global_step} scale {scale_before:g} -> {scale_after:g} (ok)")
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # keep temperature stable
        clamp_logit_scale_(base_model, args.logit_scale_max)

        if is_main_process() and (global_step % args.log_every == 0):
            if args._has_gene_groups:
                lr_info = f"lr_new={optimizer.param_groups[0]['lr']:.2e} lr_gene={optimizer.param_groups[2]['lr']:.2e}"
            else:
                lr_info = f"lr={optimizer.param_groups[0]['lr']:.2e}"
            print(f"[E{epoch:03d}] step={global_step} loss={(loss.item()*accum):.4f} {lr_info}")

        global_step += 1
        opt_steps_done += 1
        if opt_steps_done >= steps_this_epoch:
            break

    return global_step


def main():
    parser = argparse.ArgumentParser("Pretrain on CC3M (ours/clip/tinyclip) via WebDataset")

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)

    # ours
    parser.add_argument("--gene_dir", type=str, default="", help="folder containing learngene_visual.pt / learngene_text.pt ...")
    parser.add_argument("--shallow_layers", type=int, default=5)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--use_tleg", action="store_true")
    parser.add_argument("--tleg_target_depth", type=int, default=4)
    parser.add_argument("--use_multimodal_init", action="store_true")
    parser.add_argument(
        "--tleg_last_epochs",
        type=int,
        default=1,
        help="enable TLEG only for last N epochs (default 1). set 0 to never enable.",
    )

    # clip baseline
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    # tinyclip baseline (wkcn)
    parser.add_argument("--tinyclip_ckpt", type=str, default="", help="TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt")

    # data
    parser.add_argument("--cc3m_root", type=str, default="/root/autodl-tmp/cc3m")
    parser.add_argument("--cc3m_train_shards", type=str, default="", help="override train shards pattern (e.g. /path/train/*.tar)")
    parser.add_argument("--cc3m_val_shards", type=str, default="", help="override val shards pattern (e.g. /path/val/*.tar)")
    parser.add_argument("--shuffle_buf", type=int, default=20000)
    parser.add_argument("--wds_handler", type=str, default="warn", choices=["warn", "ignore"])
    parser.add_argument(
        "--coco_images",
        type=str,
        default="",
        help="COCO train2017 image folder. If set together with --coco_captions, mixes COCO into CC3M.",
    )
    parser.add_argument(
        "--coco_captions",
        type=str,
        default="",
        help="COCO captions json (e.g. captions_train2017.json).",
    )
    parser.add_argument(
        "--coco_max_images",
        type=int,
        default=-1,
        help="Optional cap on number of COCO images used (-1 = all).",
    )
    parser.add_argument(
        "--coco_mix_prob",
        type=float,
        default=-1.0,
        help="Probability of sampling COCO per sample (0~1). -1 uses auto by size with cap.",
    )
    parser.add_argument(
        "--coco_mix_prob_cap",
        type=float,
        default=0.30,
        help="Cap for auto coco_mix_prob (keeps training stable; 0.2~0.4 are common).",
    )

    # epoch sizing (global)
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=-1,
        help="Global samples per epoch (across all ranks). -1 uses cc3m_samples_est.",
    )
    parser.add_argument(
        "--cc3m_samples_est",
        type=int,
        default=3300000,
        help="Used only when samples_per_epoch=-1. CC3M train size is ~3.3M pairs.",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=20000,
        help="Global validation samples per epoch (across all ranks). 0 disables validation.",
    )

    # train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="AMP autocast dtype. 'auto' picks bf16 if supported else fp16.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--logit_scale_max", type=float, default=100.0)

    # progressive unfreeze (ours)
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=1,
        help="unfreeze learngene after this epoch (approx). step-based unfreeze uses --unfreeze_steps",
    )
    parser.add_argument("--unfreeze_steps", type=int, default=-1, help="if >0, unfreeze by optimizer-step index")
    parser.add_argument("--gene_lr_ratio", type=float, default=10.0)
    parser.add_argument("--gene_warmup_steps", type=int, default=1000)
    parser.add_argument("--gene_keywords", type=str, nargs="+", default=["learngene", "gene"])

    # dist
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/pretrain_cc3m")
    parser.add_argument("--save_every_epoch", action="store_true")

    args = parser.parse_args()

    # ---- AMP configuration ----
    # autocast dtype: bf16 is typically more stable than fp16.
    if args.amp:
        if args.amp_dtype == "auto":
            args.amp_dtype = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        if args.amp_dtype == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            if is_main_process():
                print("[WARN] --amp_dtype bf16 requested but not supported; falling back to fp16")
            args.amp_dtype = "fp16"
        args._amp_autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    else:
        # value won't matter when autocast is disabled, but keep it defined.
        args._amp_autocast_dtype = torch.float16

    # GradScaler is only needed for fp16. For bf16, disable scaler for stability/simplicity.
    args._use_scaler = bool(args.amp and args._amp_autocast_dtype == torch.float16)

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend)

    world_size = get_world_size()

    # dataset patterns
    train_shards = args.cc3m_train_shards or guess_cc3m_shards(args.cc3m_root, "train")
    val_shards = args.cc3m_val_shards
    if not val_shards:
        try:
            val_shards = guess_cc3m_shards(args.cc3m_root, "val")
        except Exception:
            val_shards = ""  # optional

    # epoch sizing
    global_samples_per_epoch = args.samples_per_epoch
    if global_samples_per_epoch < 0:
        global_samples_per_epoch = int(args.cc3m_samples_est)

    # per-rank samples (drop_last in batching makes steps deterministic)
    samples_per_rank = int(math.ceil(global_samples_per_epoch / float(world_size)))
    val_samples_per_rank = int(math.ceil(args.val_samples / float(world_size))) if args.val_samples > 0 else 0

    # Data pipeline: (image_tensor, caption_str)
    transform = build_clip_image_transform(args.image_size, is_train=True)
    train_cfg = WdsPairConfig(
        shards=train_shards,
        shuffle_buf=args.shuffle_buf,
        resampled=True,
        samples_per_epoch=samples_per_rank,
        handler=args.wds_handler,
    )
    val_cfg = WdsPairConfig(
        shards=val_shards,
        shuffle_buf=0,
        resampled=False,
        samples_per_epoch=val_samples_per_rank,
        handler=args.wds_handler,
    )

    if is_main_process():
        print("==== CC3M WebDataset ====")
        print(f"train_shards: {train_shards}")
        print(f"val_shards:   {val_shards or '[none]'}")
        print(f"world_size:   {world_size}")
        print(f"samples/epoch(global): {global_samples_per_epoch}")
        print(f"samples/epoch(per-rank): {samples_per_rank}")
        if args.val_samples > 0:
            print(f"val_samples(global): {args.val_samples} (per-rank {val_samples_per_rank})")
        print("=========================")

    # model
    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")
    if args.model == "tinyclip" and not args.tinyclip_ckpt:
        raise ValueError("--tinyclip_ckpt is required for --model tinyclip")

    bundle = create_model_bundle(args)
    model = bundle.model
    tokenize = bundle.tokenize

    # DDP
    if args.distributed and torch.cuda.is_available() and args.device.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    base_model = unwrap_model(model)
    base_model.float()
    if args.use_tleg and args.tleg_last_epochs > 0:
        set_all_learngene_tleg_active(base_model, False)

    # Build dataloaders
    train_ds = build_wds_pairs(train_cfg, transform=transform)
    # defer tokenization to collation for speed
    train_collate = default_wds_collate(tokenize)
    try:
        import webdataset as wds
        train_ds = train_ds.batched(args.batch_size, collation_fn=train_collate, partial=False)
        train_loader = wds.WebLoader(
            train_ds,
            batch_size=None,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    except Exception as e:
        raise RuntimeError("WebDataset is required for CC3M training. `pip install webdataset`.") from e

    val_loader = None
    if val_shards and args.val_samples > 0:
        val_transform = build_clip_image_transform(args.image_size, is_train=False)
        val_ds = build_wds_pairs(val_cfg, transform=val_transform)
        val_collate = default_wds_collate(tokenize)
        import webdataset as wds
        val_ds = val_ds.batched(args.batch_size, collation_fn=val_collate, partial=False)
        val_loader = wds.WebLoader(
            val_ds,
            batch_size=None,
            num_workers=max(2, args.num_workers // 2),
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

    # Optimizer: CLIP-style no-wd + (ours) separate gene groups
    if args.model == "ours":
        for p in base_model.parameters():
            p.requires_grad = True

        new_params, gene_params = split_param_groups(base_model, args.gene_keywords)
        args._gene_params = gene_params
        args._has_gene_groups = len(gene_params) > 0

        if not args._has_gene_groups:
            if is_main_process():
                print("[WARN] gene_params empty. Check gene_keywords / naming.")
            all_params = [p for p in base_model.parameters() if p.requires_grad]
            param_groups = build_param_groups_named(base_model, all_params, args.lr, args.wd)
            # normalize to 2 groups (decay/no_decay)
            if len(param_groups) == 1:
                param_groups.append({"params": [], "lr": args.lr, "weight_decay": 0.0})
        else:
            # start frozen; unfreeze by step
            set_requires_grad(gene_params, False)
            new_groups = build_param_groups_named(base_model, new_params, args.lr, args.wd)
            gene_groups = build_param_groups_named(base_model, gene_params, 0.0, args.wd)

            # ensure fixed layout: new_decay,new_no,gene_decay,gene_no
            def _pad_to_two(gs, lr):
                if len(gs) == 0:
                    return [{"params": [], "lr": lr, "weight_decay": args.wd}, {"params": [], "lr": lr, "weight_decay": 0.0}]
                if len(gs) == 1:
                    # only one group (either decay or no_decay)
                    wd0 = gs[0]["weight_decay"]
                    if wd0 > 0:
                        return [gs[0], {"params": [], "lr": lr, "weight_decay": 0.0}]
                    return [{"params": [], "lr": lr, "weight_decay": args.wd}, gs[0]]
                return gs

            new_groups = _pad_to_two(new_groups, args.lr)
            gene_groups = _pad_to_two(gene_groups, 0.0)
            param_groups = [new_groups[0], new_groups[1], gene_groups[0], gene_groups[1]]

    else:
        args._gene_params = []
        args._has_gene_groups = False
        all_params = [p for p in base_model.parameters() if p.requires_grad]
        param_groups = build_param_groups_named(base_model, all_params, args.lr, args.wd)
        if len(param_groups) == 1:
            param_groups.append({"params": [], "lr": args.lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    # GradScaler only for fp16; bf16 does not require scaling.
    # init_scale a bit smaller for stability when unfreezing.
    scaler = torch.cuda.amp.GradScaler(enabled=args._use_scaler, init_scale=2.0**12, growth_interval=2000)

    # steps/epoch (optimizer steps)
    micro_batches_per_rank = (samples_per_rank // args.batch_size)
    steps_per_epoch = max(1, micro_batches_per_rank // max(1, args.accum_steps))
    total_steps = args.epochs * steps_per_epoch

    # unfreeze step
    if args.unfreeze_steps and args.unfreeze_steps > 0:
        unfreeze_step = int(args.unfreeze_steps)
    else:
        unfreeze_step = int(args.unfreeze_epoch * steps_per_epoch)

    global_step = 0

    if is_main_process():
        print("==== Pretrain Config ====")
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            print(f"{k}: {v}")
        print("=========================")
        print(f"[INFO] steps/epoch={steps_per_epoch} total_steps={total_steps} unfreeze_step={unfreeze_step}")

    for epoch in range(args.epochs):
        global_step = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            global_step,
            total_steps,
            unfreeze_step,
            steps_this_epoch=steps_per_epoch,
        )

        # optional validation
        if is_main_process() and val_loader is not None:
            val_batches = max(1, (val_samples_per_rank // args.batch_size))
            val_loss = evaluate(args, model, val_loader, steps=val_batches)
            print(f"[VAL] epoch={epoch} loss={val_loss:.4f}")

        if is_main_process() and args.save_every_epoch:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch:03d}.pt")
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                },
            )
            print(f"[SAVE] {ckpt_path}")

    if is_main_process():
        ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
        save_checkpoint(
            ckpt_path,
            {
                "epoch": args.epochs - 1,
                "global_step": global_step,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            },
        )
        print(f"[SAVE] {ckpt_path}")


if __name__ == "__main__":
    main()
