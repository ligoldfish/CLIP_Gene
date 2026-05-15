from __future__ import annotations

import os
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.data.transforms import build_clip_image_transform
from scripts.model_factory import create_model_bundle, split_param_groups, set_requires_grad
from scripts.optim import cosine_lr, set_optimizer_lrs
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.checkpoint import load_checkpoint, save_checkpoint
from scripts.utils.device import (
    autocast_context,
    enable_tf32 as _enable_tf32,
    get_default_device,
    is_accelerator_device,
    make_grad_scaler,
    normalize_backend,
    resolve_amp_dtype as _resolve_amp_dtype,
    resolve_device,
    scaler_is_enabled,
)
from scripts.metrics import mean_average_precision
from tasks.coco_multilabel import COCOMultiLabelDataset
from tasks.dataset_registry import MultiLabelDatasetPaths, is_placeholder_path


def enable_tf32(enable: bool = True):
    _enable_tf32(enable)
    try:
        torch.set_float32_matmul_precision("high" if enable else "highest")
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
    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")


def load_init_ckpt(backbone: nn.Module, init_ckpt: str):
    if not init_ckpt:
        return
    obj = load_checkpoint(init_ckpt, map_location="cpu")
    sd = _extract_state_dict(obj)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if is_main_process():
        print(f"[INIT] loaded {init_ckpt}")
        print(f"[INIT] missing={len(missing)} unexpected={len(unexpected)}")


def forward_image_features(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(base_model, "encode_image"):
        return base_model.encode_image(images)

    if hasattr(base_model, "vision_stem") and hasattr(base_model, "vision_tower"):
        v_tokens = base_model.vision_stem(images)
        return base_model.vision_tower(v_tokens)

    raise RuntimeError("Backbone does not support image encoding.")


def reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def macro_f1_score(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (1.0 / (1.0 + np.exp(-y_score)) >= threshold).astype(np.float32)
    tp = (y_pred * y_true).sum(axis=0)
    fp = (y_pred * (1.0 - y_true)).sum(axis=0)
    fn = ((1.0 - y_pred) * y_true).sum(axis=0)
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)
    valid = (y_true.sum(axis=0) > 0)
    if not np.any(valid):
        return 0.0
    return float(np.mean(f1[valid]))


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        normalize_features: bool = True,
        head_dropout: float = 0.0,
        train_backbone: bool = False,
    ):
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


def build_multilabel_loader(
    img_dir: str,
    instances_json: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    distributed: bool = False,
):
    ds = COCOMultiLabelDataset(img_dir=img_dir, instances_json=instances_json)
    transform = build_clip_image_transform(image_size, is_train=is_train)

    def collate(batch):
        images = torch.stack([transform(x[0]) for x in batch], dim=0)
        labels = torch.stack([x[1] for x in batch], dim=0)
        return images, labels

    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=is_train, drop_last=False)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )
    return ds, loader, sampler


def resolve_amp_dtype(args) -> Tuple[torch.dtype, bool]:
    if not args.amp:
        return torch.float16, False

    torch_dtype = _resolve_amp_dtype(args.device, args.amp, args.amp_dtype)
    args.amp_dtype = "bf16" if torch_dtype == torch.bfloat16 else "fp16"
    use_scaler = bool(args.amp and torch_dtype == torch.float16 and is_accelerator_device(args.device))
    return torch_dtype, use_scaler


@torch.no_grad()
def evaluate(args, model: nn.Module, loader: DataLoader, criterion, amp_dtype: torch.dtype) -> Dict[str, float]:
    model.eval()
    device = torch.device(args.device)

    loss_sum = torch.tensor(0.0, device=device)
    sample_sum = torch.tensor(0.0, device=device)
    logits_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(device, enabled=args.amp, dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits.float(), labels)

        bs = float(images.size(0))
        loss_sum += loss.detach() * bs
        sample_sum += torch.tensor(bs, device=device)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.float().cpu())

    loss_sum = reduce_tensor(loss_sum)
    sample_sum = reduce_tensor(sample_sum)

    logits_np = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, 0), dtype=np.float32)
    labels_np = torch.cat(labels_all, dim=0).numpy() if labels_all else np.zeros((0, 0), dtype=np.float32)
    if dist.is_available() and dist.is_initialized():
        gathered_logits = [None for _ in range(get_world_size())]
        gathered_labels = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered_logits, logits_np)
        dist.all_gather_object(gathered_labels, labels_np)
        valid_logits = [x for x in gathered_logits if x is not None and x.size > 0]
        valid_labels = [x for x in gathered_labels if x is not None and x.size > 0]
        if len(valid_logits) > 0:
            logits_np = np.concatenate(valid_logits, axis=0)
        if len(valid_labels) > 0:
            labels_np = np.concatenate(valid_labels, axis=0)
    metrics = {
        "loss": float((loss_sum / torch.clamp(sample_sum, min=1.0)).item()),
        "mAP": mean_average_precision(labels_np, logits_np) if logits_np.size > 0 else 0.0,
        "macro_f1@0.5": macro_f1_score(labels_np, logits_np, threshold=0.5) if logits_np.size > 0 else 0.0,
    }
    return metrics


def main():
    p = argparse.ArgumentParser("Finetune COCO Multi-Label Classification")
    defaults = MultiLabelDatasetPaths()

    p.add_argument("--model", type=str, required=True, choices=["ours", "clip", "tinyclip"])
    p.add_argument("--device", type=str, default=get_default_device(), help="cpu/cuda/npu/auto")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--backend", type=str, default="auto", help="Distributed backend: auto/nccl/hccl/gloo.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tf32", action="store_true")

    p.add_argument("--gene_dir", type=str, default="")
    p.add_argument("--shallow_layers", type=int, default=3)
    p.add_argument("--shallow_type", type=str, default="transformer", choices=["transformer", "cnn"])
    p.add_argument("--shallow_kernel_size", type=int, default=3)
    p.add_argument("--bottleneck_dim", type=int, default=-1)
    p.add_argument("--bottleneck_dropout", type=float, default=0.0)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    p.add_argument("--proj_hidden_dim", type=int, default=-1)
    p.add_argument("--proj_dropout", type=float, default=0.0)
    p.add_argument("--use_tleg", action="store_true")
    p.add_argument("--no_tleg", action="store_true")
    p.add_argument("--tleg_target_depth", type=int, default=4)
    p.add_argument("--tleg_strict", action="store_true")
    p.add_argument("--use_multimodal_init", action="store_true")
    p.add_argument("--clip_init", action="store_true")
    p.add_argument("--no_clip_init", action="store_true")
    p.add_argument("--disable_stem_init_from_clip", action="store_true")
    p.add_argument("--stem_init_clip_name", type=str, default="ViT-B/32")
    p.add_argument("--freeze_stem_after_init", action="store_true")
    p.add_argument("--freeze_gene", action="store_true")
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--no_frozen", action="store_true")

    p.add_argument("--clip_name", type=str, default="ViT-B/32")
    p.add_argument("--tinyclip_ckpt", type=str, default="")
    p.add_argument("--init_ckpt", type=str, default="")

    p.add_argument("--coco_train_img_dir", type=str, default=defaults.coco_train_images)
    p.add_argument("--coco_train_instances_json", type=str, default=defaults.coco_train_instances_json)
    p.add_argument("--coco_val_img_dir", type=str, default=defaults.coco_val_images)
    p.add_argument("--coco_val_instances_json", type=str, default=defaults.coco_val_instances_json)
    p.add_argument("--image_size", type=int, default=224)

    p.add_argument("--normalize_features", action="store_true")
    p.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    p.set_defaults(normalize_features=True)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--train_backbone", action="store_true")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--head_lr_ratio", type=float, default=10.0)
    p.add_argument("--wd", type=float, default=0.0)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--out_dir", type=str, required=True)

    args = p.parse_args()

    args.device = str(resolve_device(args.device, allow_cpu_fallback=False))
    args.backend = normalize_backend(args.backend, args.device)
    args.clip_init = False
    args.stem_init_from_clip = False
    if getattr(args, "no_tleg", False):
        args.use_tleg = False
    args.freeze_gene = bool(getattr(args, "freeze_gene", False) or getattr(args, "frozen", False))
    if getattr(args, "no_frozen", False):
        args.freeze_gene = False

    if args.tf32:
        enable_tf32(True)

    amp_dtype, use_scaler = resolve_amp_dtype(args)

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend, args.device)

    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")
    if args.model == "tinyclip" and not args.tinyclip_ckpt:
        raise ValueError("--tinyclip_ckpt is required for --model tinyclip")
    if (
        is_placeholder_path(args.coco_train_img_dir)
        or is_placeholder_path(args.coco_train_instances_json)
        or is_placeholder_path(args.coco_val_img_dir)
        or is_placeholder_path(args.coco_val_instances_json)
    ):
        raise ValueError("Please replace the default placeholder dataset paths before running multilabel experiments.")

    bundle = create_model_bundle(args)
    backbone = bundle.model.to(args.device)
    backbone.float()

    if args.init_ckpt:
        load_init_ckpt(backbone, args.init_ckpt)

    train_ds, train_loader, train_sampler = build_multilabel_loader(
        img_dir=args.coco_train_img_dir,
        instances_json=args.coco_train_instances_json,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        distributed=args.distributed,
    )
    val_ds, val_loader, _ = build_multilabel_loader(
        img_dir=args.coco_val_img_dir,
        instances_json=args.coco_val_instances_json,
        image_size=args.image_size,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        is_train=False,
        distributed=args.distributed,
    )

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=torch.device(args.device))
        with autocast_context(args.device, enabled=args.amp, dtype=amp_dtype):
            feat_dim = int(forward_image_features(backbone, dummy).shape[-1])

    model = MultiLabelClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=train_ds.num_classes,
        normalize_features=args.normalize_features,
        head_dropout=args.head_dropout,
        train_backbone=args.train_backbone,
    ).to(args.device)

    if not args.train_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif args.model == "ours" and args.freeze_gene:
        _, gene_params = split_param_groups(model.backbone, ["learngene", "gene"])
        set_requires_grad(gene_params, False)

    for p in model.head.parameters():
        p.requires_grad = True

    if args.resume:
        obj = load_checkpoint(args.resume, map_location="cpu")
        sd = _extract_state_dict(obj)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main_process():
            print(f"[RESUME] {args.resume} missing={len(missing)} unexpected={len(unexpected)}")

    if args.distributed and is_accelerator_device(args.device):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = nn.BCEWithLogitsLoss()

    if args.eval_only:
        metrics = evaluate(args, model, val_loader, criterion, amp_dtype)
        if is_main_process():
            print(f"[EVAL] loss={metrics['loss']:.4f} mAP={metrics['mAP']:.4f} macro_f1@0.5={metrics['macro_f1@0.5']:.4f}")
        return

    backbone_params = [p for p in unwrap_model(model).backbone.parameters() if p.requires_grad]
    head_params = [p for p in unwrap_model(model).head.parameters() if p.requires_grad]
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": args.lr, "weight_decay": args.wd})
    param_groups.append({"params": head_params, "lr": args.lr * args.head_lr_ratio, "weight_decay": args.wd})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
    scaler = make_grad_scaler(args.device, enabled=use_scaler)
    use_scaler = scaler_is_enabled(scaler)

    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    best_map = -1.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        unwrap_model(model).train()
        loss_sum = 0.0
        n = 0

        for it, (images, labels) in enumerate(train_loader):
            images = images.to(torch.device(args.device), non_blocking=True)
            labels = labels.to(torch.device(args.device), non_blocking=True)

            lr_now = cosine_lr(global_step, total_steps, args.lr, args.min_lr)
            if len(param_groups) == 2:
                set_optimizer_lrs(optimizer, [lr_now, lr_now * args.head_lr_ratio])
            else:
                set_optimizer_lrs(optimizer, [lr_now * args.head_lr_ratio])

            with autocast_context(args.device, enabled=args.amp, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
                optimizer.step()

            loss_sum += float(loss.item())
            n += 1
            global_step += 1

            if is_main_process() and (global_step % args.log_every == 0):
                print(f"[E{epoch:03d}][{it:05d}] step={global_step} loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        metrics = evaluate(args, model, val_loader, criterion, amp_dtype)
        if is_main_process():
            print(
                f"[E{epoch:03d}] train_loss={loss_sum / max(1, n):.4f} "
                f"val_loss={metrics['loss']:.4f} mAP={metrics['mAP']:.4f} macro_f1@0.5={metrics['macro_f1@0.5']:.4f}"
            )

        if is_main_process() and args.save_every_epoch:
            out_path = os.path.join(args.out_dir, f"model_epoch{epoch:03d}.pt")
            save_checkpoint(
                out_path,
                {
                    "model": unwrap_model(model).state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "metrics": metrics,
                },
            )
            print(f"[SAVE] {out_path}")

        if is_main_process() and metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            out_path = os.path.join(args.out_dir, "model_best.pt")
            save_checkpoint(
                out_path,
                {
                    "model": unwrap_model(model).state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_mAP": best_map,
                    "metrics": metrics,
                },
            )
            print(f"[SAVE] best -> {out_path} (mAP={best_map:.4f})")

    if is_main_process():
        out_path = os.path.join(args.out_dir, "model_last.pt")
        save_checkpoint(
            out_path,
            {
                "model": unwrap_model(model).state_dict(),
                "epoch": args.epochs - 1,
                "global_step": global_step,
                "best_mAP": best_map,
            },
        )
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
