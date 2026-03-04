# scripts/finetune_itm.py
# Finetune image-text matching (ITM) on COCO captions (pos/neg sampling)
#
# Score is CLIP-style similarity (dot product) * logit_scale -> BCEWithLogits
#
# Example:
# torchrun --nproc_per_node=1 -m scripts.finetune_itm \
#   --model ours \
#   --gene_dir /root/gene_exports/last2_plus6 \
#   --init_ckpt outputs/pretrain_ours26/ckpt_last.pt \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --batch_size 256 --epochs 10 --amp \
#   --freeze_gene \
#   --out_dir outputs/ft_itm_ours26
# torchrun --nproc_per_node=1 -m scripts.finetune_itm \
#   --model clip \
#   --clip_name "ViT-B/32" \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --batch_size 256 --epochs 5 --amp \
#   --out_dir outputs/ft_itm_clip
# torchrun --nproc_per_node=1 -m scripts.finetune_itm \
#   --model tinyclip \
#   --tinyclip_ckpt /root/autodl-tmp/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --batch_size 256 --epochs 5 --amp \
#   --out_dir outputs/ft_itm_tinyclip


import os
import argparse
from typing import Optional, Tuple, Any, Dict, List

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


def get_logit_scale(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "logit_scale"):
        try:
            return model.logit_scale.exp().clamp(max=100.0)
        except Exception:
            pass
    return torch.tensor(1.0 / 0.07, device=next(model.parameters()).device)


def forward_features(base_model: nn.Module, images: torch.Tensor, tokens: torch.Tensor):
    if hasattr(base_model, "encode_image") and hasattr(base_model, "encode_text"):
        return base_model.encode_image(images), base_model.encode_text(tokens)

    if all(hasattr(base_model, k) for k in ["vision_stem", "text_stem", "vision_tower", "text_tower"]):
        v_tokens = base_model.vision_stem(images)
        t_tokens = base_model.text_stem(tokens)
        z_img = base_model.vision_tower(v_tokens)
        z_txt = base_model.text_tower(t_tokens, text=tokens)
        return z_img, z_txt

    raise RuntimeError("Model does not support feature extraction (need encode_image/encode_text or StudentCLIP stems/towers).")


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "net", "module"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                sd = ckpt_obj[key]
                if any(k.startswith("module.") for k in sd.keys()):
                    sd = {k[len("module."):]: v for k, v in sd.items()}
                return sd
        if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(torch.is_tensor(v) for v in ckpt_obj.values()):
            sd = ckpt_obj
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            return sd
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def load_init_ckpt_if_any(base_model: nn.Module, init_ckpt: str):
    if not init_ckpt:
        return
    obj = load_checkpoint(init_ckpt, map_location="cpu")
    sd = _extract_state_dict(obj)
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if is_main_process():
        print(f"[INIT] loaded: {init_ckpt}")
        print(f"[INIT] missing={len(missing)} unexpected={len(unexpected)}")


def build_optimizer(args, base_model: nn.Module) -> Tuple[torch.optim.Optimizer, List[nn.Parameter]]:
    if args.model == "ours":
        new_params, gene_params = split_param_groups(base_model, args.gene_keywords)
        if len(gene_params) == 0:
            if is_main_process():
                print("[WARN] gene_params empty; fallback to single group.")
            param_groups = [{"params": [p for p in base_model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}]
            gene_params = []
        else:
            if args.freeze_gene:
                set_requires_grad(gene_params, False)
            param_groups = [
                {"params": [p for p in new_params if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
                {"params": list(gene_params), "lr": 0.0, "weight_decay": args.wd},
            ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
        return optimizer, list(gene_params)

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in base_model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6
    )
    return optimizer, []


def build_loader(args, tokenize, transform, split: str = "train"):
    ds = COCOMatchingDataset(
        img_dir=args.coco_images if split == "train" else args.coco_val_images,
        captions_json=args.coco_captions if split == "train" else args.coco_val_captions,
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
        sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=args.batch_size if split == "train" else args.val_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate,
    )
    return ds, loader, sampler


@torch.no_grad()
def evaluate(args, model, loader) -> Tuple[float, float]:
    base_model = unwrap_model(model)
    base_model.eval()
    device = args.device

    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    n = 0
    correct = 0
    total = 0

    for images, tokens, labels in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        img_f, txt_f = forward_features(base_model, images, tokens)
        img_f = F.normalize(img_f.float(), dim=-1)
        txt_f = F.normalize(txt_f.float(), dim=-1)
        logit_scale = get_logit_scale(base_model)

        logits = logit_scale * torch.sum(img_f * txt_f, dim=-1)  # [B]
        loss = loss_fn(logits, labels)

        loss_sum += float(loss.item()) * labels.numel()
        n += labels.numel()

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += int(labels.numel())

    avg_loss = loss_sum / max(1, n)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch(args, model, loader, sampler, optimizer, scaler, epoch: int, global_step: int, total_steps: int, unfreeze_step: int):
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

        if len(optimizer.param_groups) == 1:
            set_optimizer_lrs(optimizer, [lr_new])
            lr_gene = None
        else:
            if args.freeze_gene or global_step < unfreeze_step:
                gene_lr = 0.0
            else:
                target = lr_new / args.gene_lr_ratio
                if args.gene_warmup_steps <= 0:
                    gene_lr = target
                else:
                    w = min(1.0, (global_step - unfreeze_step) / float(args.gene_warmup_steps))
                    gene_lr = target * w
            set_optimizer_lrs(optimizer, [lr_new, gene_lr])
            lr_gene = gene_lr

        with torch.cuda.amp.autocast(enabled=args.amp):
            img_f, txt_f = forward_features(base_model, images, tokens)
            img_f = F.normalize(img_f.float(), dim=-1)
            txt_f = F.normalize(txt_f.float(), dim=-1)
            logit_scale = get_logit_scale(base_model)
            logits = logit_scale * torch.sum(img_f * txt_f, dim=-1)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if args.amp:
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

        preds = (torch.sigmoid(logits.detach()) > 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += int(labels.numel())

        if is_main_process() and (global_step % args.log_every == 0):
            if lr_gene is None:
                lr_info = f"lr={optimizer.param_groups[0]['lr']:.2e}"
            else:
                lr_info = f"lr_new={optimizer.param_groups[0]['lr']:.2e} lr_gene={optimizer.param_groups[1]['lr']:.2e}"
            acc = correct / max(1, total)
            print(f"[E{epoch:03d}][{it:05d}] step={global_step} loss={loss.item():.4f} acc={acc:.3f} {lr_info}")

    avg_loss = loss_sum / max(1, loss_count)
    acc = correct / max(1, total)
    return global_step, avg_loss, acc


def main():
    parser = argparse.ArgumentParser("Finetune ITM on COCO (ours/clip/tinyclip)")

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)

    # ours knobs
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--shallow_layers", type=int, default=2)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    # TLEG options kept for compatibility
    parser.add_argument("--use_tleg", action="store_true")
    parser.add_argument("--tleg_target_depth", type=int, default=2)
    parser.add_argument("--use_multimodal_init", action="store_true")

    # baselines
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--tinyclip_ckpt", type=str, default="")

    # init ckpt
    parser.add_argument("--init_ckpt", type=str, default="")

    # data
    parser.add_argument("--coco_images", type=str, required=True)
    parser.add_argument("--coco_captions", type=str, required=True)

    # optional val
    parser.add_argument("--coco_val_images", type=str, default="")
    parser.add_argument("--coco_val_captions", type=str, default="")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=256)

    parser.add_argument("--pos_ratio", type=float, default=0.5)

    # train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
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

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/finetune_itm")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_full_ckpt", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend)

    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")

    # model
    bundle = create_model_bundle(args)
    model = bundle.model
    tokenize = bundle.tokenize

    if args.distributed and torch.cuda.is_available() and args.device.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    base_model = unwrap_model(model)
    base_model.float()  # IMPORTANT for AMP stability

    if args.init_ckpt:
        load_init_ckpt_if_any(base_model, args.init_ckpt)

    # data loaders
    transform_train = build_clip_image_transform(args.image_size, is_train=True)
    train_ds, train_loader, train_sampler = build_loader(args, tokenize, transform_train, split="train")

    val_loader = None
    if args.coco_val_images and args.coco_val_captions:
        transform_val = build_clip_image_transform(args.image_size, is_train=False)
        _, val_loader, _ = build_loader(args, tokenize, transform_val, split="val")

    # optimizer
    optimizer, gene_params = build_optimizer(args, base_model)
    args._gene_params = gene_params

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.epochs * len(train_loader)
    unfreeze_step = args.unfreeze_epoch * len(train_loader)
    global_step = 0

    if is_main_process():
        print("==== Finetune ITM Config ====")
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            print(f"{k}: {v}")
        print("============================")
        print(f"[INFO] total_steps={total_steps} unfreeze_step={unfreeze_step} steps/epoch={len(train_loader)}")

    for epoch in range(args.epochs):
        global_step, train_loss, train_acc = train_one_epoch(
            args, model, train_loader, train_sampler, optimizer, scaler, epoch, global_step, total_steps, unfreeze_step
        )

        if is_main_process():
            print(f"[E{epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f}")

        if val_loader is not None and ((epoch + 1) % args.val_every == 0) and is_main_process():
            val_loss, val_acc = evaluate(args, model, val_loader)
            print(f"[E{epoch:03d}] val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

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
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    },
                )
                print(f"[SAVE] {cpath}")

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
                    "scaler": scaler.state_dict(),
                    "epoch": args.epochs - 1,
                    "global_step": global_step,
                    "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                },
            )
            print(f"[SAVE] {cpath}")


if __name__ == "__main__":
    main()
