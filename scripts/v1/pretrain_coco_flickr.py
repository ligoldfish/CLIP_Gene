# scripts/pretrain_coco_flickr.py

# torchrun --nproc_per_node=1 -m scripts.pretrain_coco_flickr \
#   --distributed \
#   --model ours \
#   --gene_dir /root/gene_exports/last3 \
#   --use_tleg \
#   --tleg_target_depth 6 \
#   --tleg_last_epochs 2 \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --flickr_images /root/autodl-tmp/flickr30k/images \
#   --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
#   --batch_size 256 --epochs 15 --amp \
#   --unfreeze_epoch 10 --gene_lr_ratio 10 --gene_warmup_steps 200 \
#   --out_dir outputs/pretrain_ours3
# torchrun --nproc_per_node=1 -m scripts.pretrain_coco_flickr \
#   --distributed \
#   --model ours \
#   --gene_dir /root/gene_exports/last2_plus6 \
#   --use_tleg \
#   --tleg_target_depth 6 \
#   --tleg_last_epochs 2 \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --flickr_images /root/autodl-tmp/flickr30k/images \
#   --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
#   --batch_size 256 --epochs 15 --amp \
#   --unfreeze_epoch 10 --gene_lr_ratio 10 --gene_warmup_steps 200 \
#   --out_dir outputs/pretrain_ours26
# torchrun --nproc_per_node=1 -m scripts.pretrain_coco_flickr \
#   --distributed \
#   --model clip \
#   --clip_name "ViT-B/32" \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --flickr_images /root/autodl-tmp/flickr30k/images \
#   --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
#   --flickr_split train \
#   --batch_size 256 --epochs 15 --amp \
#   --unfreeze_epoch 8 --gene_lr_ratio 10 --gene_warmup_steps 200 \
#   --out_dir outputs/pretrain_clip

import os
import argparse
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.data.transforms import build_clip_image_transform
from scripts.data.coco_captions import CocoCaptionsPairs
from scripts.data.karpathy_pairs import KarpathyPairs
from scripts.data.mixed import MixedDataset

from scripts.model_factory import (
    create_model_bundle,
    split_param_groups,
    set_requires_grad,
)
from scripts.losses import clip_contrastive_loss
from scripts.optim import cosine_lr, set_optimizer_lrs
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size, is_dist_avail_and_initialized
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint

# --- TF32 (Ampere/Ada/Hopper GPUs) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch>=2.0 推荐
try:
    torch.set_float32_matmul_precision("high")  # "highest"/"high"/"medium"
except Exception:
    pass
def set_all_learngene_tleg_active(model: nn.Module, flag: bool):
    # import here to avoid import issues
    try:
        from models.learngene_loader import LearngeneModule
    except Exception:
        LearngeneModule = None

    for m in model.modules():
        if LearngeneModule is not None and isinstance(m, LearngeneModule):
            if getattr(m, "is_tleg", False):
                m.set_tleg_active(flag)
        else:
            # fallback: duck typing
            if hasattr(m, "set_tleg_active") and hasattr(m, "is_tleg") and getattr(m, "is_tleg"):
                m.set_tleg_active(flag)

def collate_fn(batch: List[Dict[str, Any]], tokenize):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    tokens = tokenize(texts)
    return images, tokens


def get_logit_scale(model: nn.Module) -> torch.Tensor:
    m = model
    if hasattr(m, "logit_scale"):
        # openai clip uses logit_scale as log-space parameter
        try:
            return m.logit_scale.exp().clamp(max=100.0)
        except Exception:
            pass
    return torch.tensor(1.0 / 0.07, device=next(m.parameters()).device)


def forward_features(model: nn.Module, images: torch.Tensor, tokens: torch.Tensor):
    """
    Prefer encode_image/text; otherwise try StudentCLIP internal modules.
    """
    if hasattr(model, "encode_image") and hasattr(model, "encode_text"):
        return model.encode_image(images), model.encode_text(tokens)

    # common StudentCLIP pattern (if you implemented like we discussed earlier)
    if hasattr(model, "vision_stem") and hasattr(model, "text_stem") and hasattr(model, "vision_tower") and hasattr(model, "text_tower"):
        v_tokens = model.vision_stem(images)
        t_tokens = model.text_stem(tokens)
        # towers should output normalized embeddings
        z_img = model.vision_tower(v_tokens)
        z_txt = model.text_tower(t_tokens, text=tokens)
        return z_img, z_txt

    # fallback: try forward returning logits is not supported here (we need embeddings for DDP gather)
    raise RuntimeError("Model does not support feature extraction (need encode_image/encode_text or StudentCLIP stems/towers).")


def train_one_epoch(args, model, loader, optimizer, scaler, epoch: int, global_step: int, total_steps: int, unfreeze_step: int):
    model.train()
    device = args.device

    base_model = unwrap_model(model)

    # staged freeze/unfreeze for ours
    if args.model == "ours":
        if epoch < args.unfreeze_epoch:
            set_requires_grad(args._gene_params, False)
        else:
            set_requires_grad(args._gene_params, True)
    if args.model == "ours":
        # enable TLEG for last N epochs, but also建议不要早于 unfreeze_epoch（更稳）
        tleg_on_epoch = max(args.unfreeze_epoch, args.epochs - args.tleg_last_epochs) if args.tleg_last_epochs > 0 else 0
        enable_tleg_now = args.use_tleg and (args.tleg_last_epochs > 0) and (epoch >= tleg_on_epoch)

        if args.use_tleg and args.tleg_last_epochs > 0:
            set_all_learngene_tleg_active(base_model, enable_tleg_now)
            if is_main_process() and epoch == tleg_on_epoch:
                print(f"[TLEG] enabled at epoch={epoch} (last {args.tleg_last_epochs} epochs)")


    for it, (images, tokens) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        # cosine LR for new params
        lr_new = cosine_lr(global_step, total_steps, args.lr, args.min_lr)

        if len(optimizer.param_groups) == 1:
            set_optimizer_lrs(optimizer, [lr_new])
        else:
            # gene lr: smaller + warmup after unfreeze
            if global_step < unfreeze_step:
                gene_lr = 0.0
            else:
                # warmup gene lr from 0 -> lr_new/gene_lr_ratio
                target = lr_new / args.gene_lr_ratio
                if args.gene_warmup_steps <= 0:
                    gene_lr = target
                else:
                    w = min(1.0, (global_step - unfreeze_step) / float(args.gene_warmup_steps))
                    gene_lr = target * w
            set_optimizer_lrs(optimizer, [lr_new, gene_lr])

        with torch.cuda.amp.autocast(enabled=args.amp):
            img_f, txt_f = forward_features(base_model, images, tokens)
            logit_scale = get_logit_scale(base_model)
            loss = clip_contrastive_loss(img_f, txt_f, logit_scale)

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

        if is_main_process() and (global_step % args.log_every == 0):
            if len(optimizer.param_groups) == 1:
                lr_info = f"lr={optimizer.param_groups[0]['lr']:.2e}"
            else:
                lr_info = f"lr_new={optimizer.param_groups[0]['lr']:.2e} lr_gene={optimizer.param_groups[1]['lr']:.2e}"
            print(f"[E{epoch:03d}][{it:05d}] step={global_step} loss={loss.item():.4f} {lr_info}")

        global_step += 1

    return global_step


def main():
    parser = argparse.ArgumentParser("Pretrain on COCO Captions + Flickr30k (ours/clip/tinyclip)")

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)

    # ours: NO student_cfg anymore
    parser.add_argument("--gene_dir", type=str, default="", help="folder containing learngene_visual.pt / learngene_text.pt ...")
    parser.add_argument("--shallow_layers", type=int, default=4)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--use_tleg", action="store_true")
    parser.add_argument("--tleg_target_depth", type=int, default=2)
    parser.add_argument("--use_multimodal_init", action="store_true")
    parser.add_argument("--tleg_last_epochs", type=int, default=2,
                    help="enable TLEG only for last N epochs (default 2). set 0 to always use base depth.")


    # clip baseline
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")

    # tinyclip baseline (wkcn)
    parser.add_argument("--tinyclip_ckpt", type=str, default="", help="TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt")

    # data
    parser.add_argument("--coco_images", type=str, required=True)
    parser.add_argument("--coco_captions", type=str, required=True)  # captions_train2017.json
    parser.add_argument("--flickr_images", type=str, required=True)
    parser.add_argument("--flickr_ann", type=str, required=True)  # Karpathy json
    parser.add_argument("--flickr_split", type=str, default="train")

    parser.add_argument("--coco_max", type=int, default=-1)
    parser.add_argument("--flickr_max", type=int, default=-1)
    parser.add_argument("--mix_probs", type=float, nargs=2, default=[0.7, 0.3])
    parser.add_argument("--mix_length", type=int, default=-1)

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # progressive unfreeze (ours only)
    parser.add_argument("--unfreeze_epoch", type=int, default=8)
    parser.add_argument("--gene_lr_ratio", type=float, default=10.0)
    parser.add_argument("--gene_warmup_steps", type=int, default=200)
    parser.add_argument("--gene_keywords", type=str, nargs="+", default=["learngene", "gene"])

    # dist
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/pretrain")
    parser.add_argument("--save_every_epoch", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend)

    # datasets
    transform = build_clip_image_transform(args.image_size, is_train=True)

    coco_ds = CocoCaptionsPairs(args.coco_images, args.coco_captions, transform=transform, max_samples=args.coco_max)
    flickr_ds = KarpathyPairs(args.flickr_images, args.flickr_ann, split=args.flickr_split, transform=transform, max_samples=args.flickr_max)
    train_ds = MixedDataset([coco_ds, flickr_ds], probs=args.mix_probs, length=args.mix_length)

    # model
    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours (point to outputs/lg_clip_adapter_rho)")

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
        # start with base depth (TLEG OFF)
        set_all_learngene_tleg_active(base_model, False)
    # sampler/loader
    if args.distributed:
        sampler = DistributedSampler(train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, tokenize),
    )

    # optimizer with param groups
    if args.model == "ours":
        for p in base_model.parameters():
            p.requires_grad = True

        new_params, gene_params = split_param_groups(base_model, args.gene_keywords)
        args._gene_params = gene_params  # store for freezing control

        if len(gene_params) == 0:
            print("[WARN] gene_params empty. Check naming; fallback to single group.")
            param_groups = [{"params": [p for p in base_model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}]
        else:
            # start with gene requires_grad=False, lr=0 (scheduler will keep 0 before unfreeze)
            set_requires_grad(gene_params, False)
            param_groups = [
                {"params": [p for p in new_params if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
                {"params": [p for p in gene_params], "lr": 0.0, "weight_decay": args.wd},
            ]
    else:
        args._gene_params = []
        param_groups = [{"params": [p for p in base_model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.epochs * len(loader)
    unfreeze_step = args.unfreeze_epoch * len(loader)
    global_step = 0

    if is_main_process():
        print("==== Pretrain Config ====")
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            print(f"{k}: {v}")
        print("=========================")
        print(f"[INFO] total_steps={total_steps} unfreeze_step={unfreeze_step}")

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        global_step = train_one_epoch(args, model, loader, optimizer, scaler, epoch, global_step, total_steps, unfreeze_step)

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
