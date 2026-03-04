# scripts/finetune_imagenet1k_cls.py
# ImageNet-1K classification for ours/clip/tinyclip:
#  - linear probe (default): freeze backbone, train linear head
#  - finetune: --train_backbone (train backbone + head)
#
# Expected ImageNet folder:
#   IMAGENET_ROOT/train/<class_name>/*.JPEG
#   IMAGENET_ROOT/val/<class_name>/*.JPEG
#
# Example (linear probe):
# torchrun --nproc_per_node=1 -m scripts.finetune_imagenet1k_cls \
#   --model ours --gene_dir /root/gene_exports/last3 --init_ckpt outputs/pretrain_ours3/ckpt_last.pt \
#   --imagenet_root /root/autodl-tmp/imagenet \
#   --epochs 10 --batch_size 256 --val_batch_size 512 --amp \
#   --out_dir outputs/imagenet_lp_ours_last3
#
# Example (finetune):
# torchrun --nproc_per_node=1 -m scripts.finetune_imagenet1k_cls \
#   --model ours --gene_dir /root/gene_exports/last3 --init_ckpt outputs/pretrain_ours3/ckpt_last.pt \
#   --imagenet_root /root/autodl-tmp/imagenet \
#   --train_backbone --lr 2e-4 --head_lr_ratio 10 \
#   --epochs 30 --batch_size 128 --val_batch_size 256 --amp \
#   --out_dir outputs/imagenet_ft_ours_last3

import os
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision.datasets import ImageFolder

from scripts.model_factory import create_model_bundle
from scripts.optim import cosine_lr, set_optimizer_lrs
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.checkpoint import load_checkpoint, save_checkpoint

from scripts.data.transforms import build_clip_image_transform


# -------------------------
# helpers
# -------------------------
def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "net", "module"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                if any(kk.startswith("module.") for kk in sd.keys()):
                    sd = {kk[len("module."):]: vv for kk, vv in sd.items()}
                return sd
        if all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
            sd = obj
            if any(kk.startswith("module.") for kk in sd.keys()):
                sd = {kk[len("module."):]: vv for kk, vv in sd.items()}
            return sd
    raise ValueError(f"Unrecognized ckpt format: {type(obj)}")


def load_init_ckpt(backbone: nn.Module, init_ckpt: str):
    if not init_ckpt:
        return
    obj = load_checkpoint(init_ckpt, map_location="cpu")
    sd = _extract_state_dict(obj)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if is_main_process():
        print(f"[INIT] loaded init_ckpt={init_ckpt}")
        print(f"[INIT] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[INIT] missing sample:", missing[:20])


def forward_image_features(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return image embedding [B, D] for clip / tinyclip / ours."""
    if hasattr(base_model, "encode_image"):
        return base_model.encode_image(images)

    if hasattr(base_model, "vision_stem") and hasattr(base_model, "vision_tower"):
        v_tokens = base_model.vision_stem(images)
        z_img = base_model.vision_tower(v_tokens)
        return z_img

    raise RuntimeError("Backbone does not support encode_image or (vision_stem + vision_tower).")


def reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    """All-reduce sum across processes (if distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (top1_count, top5_count) as tensors on the same device."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [maxk, B]

    top1 = correct[:1].reshape(-1).float().sum()
    top5 = correct[:5].reshape(-1).float().sum() if maxk >= 5 else correct.reshape(-1).float().sum()
    return top1, top5


class ImageNetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int = 1000,
                 normalize_features: bool = True, head_dropout: float = 0.0,
                 train_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.normalize_features = normalize_features
        self.train_backbone = train_backbone
        self.dropout = nn.Dropout(p=head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, num_classes, bias=True)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.train_backbone:
            feats = forward_image_features(self.backbone, images)
        else:
            self.backbone.eval()
            with torch.no_grad():
                feats = forward_image_features(self.backbone, images)

        feats = feats.float()
        if self.normalize_features:
            feats = F.normalize(feats, dim=-1)
        feats = self.dropout(feats)
        return self.head(feats)


def build_imagenet_loaders(args):
    if args.imagenet_root:
        train_dir = os.path.join(args.imagenet_root, "train")
        val_dir = os.path.join(args.imagenet_root, "val")
    else:
        train_dir, val_dir = args.imagenet_train_dir, args.imagenet_val_dir

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val dir not found: {val_dir}")

    transform_train = build_clip_image_transform(args.image_size, is_train=True)
    transform_val = build_clip_image_transform(args.image_size, is_train=False)

    train_ds = ImageFolder(train_dir, transform=transform_train)
    val_ds = ImageFolder(val_dir, transform=transform_val)

    if args.distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=False
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, train_sampler, val_loader


@torch.no_grad()
def evaluate(args, model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    device = torch.device(args.device)

    total = torch.tensor(0.0, device=device)
    top1 = torch.tensor(0.0, device=device)
    top5 = torch.tensor(0.0, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)

        t1, t5 = accuracy_topk(logits.float(), targets, topk=(1, 5))
        top1 += t1
        top5 += t5
        total += torch.tensor(float(images.size(0)), device=device)

    # distributed aggregate
    total = reduce_tensor(total)
    top1 = reduce_tensor(top1)
    top5 = reduce_tensor(top5)

    top1_acc = (top1 / total).item() * 100.0
    top5_acc = (top5 / total).item() * 100.0
    return top1_acc, top5_acc


def main():
    p = argparse.ArgumentParser("ImageNet-1K Linear Probe / Finetune Classification")

    p.add_argument("--model", type=str, required=True, choices=["ours", "clip", "tinyclip"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tf32", action="store_true")

    # ours args (must match your model_factory)
    p.add_argument("--gene_dir", type=str, default="")
    p.add_argument("--shallow_layers", type=int, default=4)
    p.add_argument("--bottleneck_dim", type=int, default=-1)
    p.add_argument("--bottleneck_dropout", type=float, default=0.0)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--proj_head", type=str, default="mlp", choices=["mlp", "linear"])
    p.add_argument("--proj_hidden_dim", type=int, default=-1)
    p.add_argument("--proj_dropout", type=float, default=0.0)
    p.add_argument("--use_tleg", action="store_true")
    p.add_argument("--tleg_target_depth", type=int, default=2)
    p.add_argument("--use_multimodal_init", action="store_true")

    # baselines
    p.add_argument("--clip_name", type=str, default="ViT-B/32")
    p.add_argument("--tinyclip_ckpt", type=str, default="")

    # init backbone (mainly ours)
    p.add_argument("--init_ckpt", type=str, default="")

    # ImageNet data
    p.add_argument("--imagenet_root", type=str, default="", help="root with train/ and val/ subfolders")
    p.add_argument("--imagenet_train_dir", type=str, default="")
    p.add_argument("--imagenet_val_dir", type=str, default="")
    p.add_argument("--image_size", type=int, default=224)

    # classifier config
    p.add_argument("--normalize_features", action="store_true")
    p.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    p.set_defaults(normalize_features=True)
    p.add_argument("--head_dropout", type=float, default=0.0)

    # mode: linear probe vs finetune
    p.add_argument("--train_backbone", action="store_true",
                   help="if set, finetune backbone+head; otherwise linear probe (freeze backbone)")

    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.0)

    # lr: base lr for backbone; head lr = lr * head_lr_ratio (commonly 10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--head_lr_ratio", type=float, default=10.0)
    p.add_argument("--wd", type=float, default=0.0)

    # io
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", type=str, default="", help="resume full classifier (backbone+head) ckpt")
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--save_head_only", action="store_true")
    p.add_argument("--out_dir", type=str, required=True)

    args = p.parse_args()

    seed_everything(args.seed)
    mkdir(args.out_dir)
    if args.tf32:
        enable_tf32()

    if args.distributed:
        setup_distributed(args.backend)

    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")
    if args.model == "tinyclip" and not args.tinyclip_ckpt:
        raise ValueError("--tinyclip_ckpt is required for --model tinyclip")
    if (not args.imagenet_root) and (not args.imagenet_train_dir or not args.imagenet_val_dir):
        raise ValueError("Provide --imagenet_root or both --imagenet_train_dir/--imagenet_val_dir")

    # backbone
    bundle = create_model_bundle(args)
    backbone = bundle.model.to(args.device)
    backbone.float()

    if args.init_ckpt:
        load_init_ckpt(backbone, args.init_ckpt)

    # data
    train_loader, train_sampler, val_loader = build_imagenet_loaders(args)

    # infer feature dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=torch.device(args.device))
        with torch.cuda.amp.autocast(enabled=args.amp):
            z = forward_image_features(backbone, dummy)
        feat_dim = int(z.shape[-1])

    model = ImageNetClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=1000,
        normalize_features=args.normalize_features,
        head_dropout=args.head_dropout,
        train_backbone=args.train_backbone,
    ).to(args.device)

    # freeze backbone for linear probe
    if not args.train_backbone:
        for pp in model.backbone.parameters():
            pp.requires_grad = False
    for pp in model.head.parameters():
        pp.requires_grad = True

    # resume
    if args.resume:
        obj = load_checkpoint(args.resume, map_location="cpu")
        sd = _extract_state_dict(obj)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main_process():
            print(f"[RESUME] {args.resume} missing={len(missing)} unexpected={len(unexpected)}")

    # DDP
    if args.distributed and torch.cuda.is_available() and args.device.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    # loss
    criterion = nn.CrossEntropyLoss()

    # eval only
    if args.eval_only:
        top1, top5 = evaluate(args, model, val_loader)
        if is_main_process():
            print(f"[EVAL] ImageNet-1K Top-1={top1:.2f} Top-5={top5:.2f}")
        return

    # optimizer:
    # - linear probe: only head params trainable
    # - finetune: backbone + head, with head lr higher
    backbone_params = [p for p in unwrap_model(model).backbone.parameters() if p.requires_grad]
    head_params = [p for p in unwrap_model(model).head.parameters() if p.requires_grad]

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": args.lr, "weight_decay": args.wd})
    param_groups.append({"params": head_params, "lr": args.lr * args.head_lr_ratio, "weight_decay": args.wd})

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.epochs * len(train_loader)
    global_step = 0

    best_top1 = -1.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        unwrap_model(model).train()
        loss_sum, n = 0.0, 0

        for it, (images, targets) in enumerate(train_loader):
            images = images.to(torch.device(args.device), non_blocking=True)
            targets = targets.to(torch.device(args.device), non_blocking=True)

            # cosine lr schedule for backbone base lr; head lr follows ratio
            lr_now = cosine_lr(global_step, total_steps, args.lr, args.min_lr)
            lrs = []
            for g in opt.param_groups:
                if g["params"] is backbone_params and len(backbone_params) > 0:
                    lrs.append(lr_now)
                else:
                    lrs.append(lr_now * args.head_lr_ratio)
            set_optimizer_lrs(opt, lrs)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits.float(), targets)

            opt.zero_grad(set_to_none=True)
            if args.amp:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
                opt.step()

            loss_sum += float(loss.item())
            n += 1
            global_step += 1

            if is_main_process() and (global_step % args.log_every == 0):
                # show first group lr as base lr
                print(f"[E{epoch:03d}][{it:05d}] step={global_step} loss={loss.item():.4f} lr_base={opt.param_groups[0]['lr']:.2e}")

        # eval
        top1, top5 = evaluate(args, model, val_loader)
        if is_main_process():
            print(f"[E{epoch:03d}] train_loss={loss_sum / max(1, n):.4f} Top-1={top1:.2f} Top-5={top5:.2f}")

        # save
        if is_main_process() and args.save_every_epoch:
            if args.save_head_only:
                wpath = os.path.join(args.out_dir, f"head_epoch{epoch:03d}.pt")
                save_checkpoint(wpath, {
                    "head": unwrap_model(model).head.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "feat_dim": feat_dim,
                })
            else:
                wpath = os.path.join(args.out_dir, f"model_epoch{epoch:03d}.pt")
                save_checkpoint(wpath, {
                    "model": unwrap_model(model).state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                })
            print(f"[SAVE] {wpath}")

        # track best
        if is_main_process() and top1 > best_top1:
            best_top1 = top1
            wpath = os.path.join(args.out_dir, "model_best.pt")
            save_checkpoint(wpath, {
                "model": unwrap_model(model).state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_top1": best_top1,
            })
            print(f"[SAVE] best -> {wpath} (Top-1={best_top1:.2f})")

    if is_main_process():
        wpath = os.path.join(args.out_dir, "model_last.pt")
        save_checkpoint(wpath, {
            "model": unwrap_model(model).state_dict(),
            "epoch": args.epochs - 1,
            "global_step": global_step,
            "best_top1": best_top1,
        })
        print(f"[SAVE] {wpath}")


if __name__ == "__main__":
    main()
