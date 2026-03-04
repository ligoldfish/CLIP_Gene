# scripts/ft_cifar100.py
# Finetune on CIFAR-100 (transfer classification)
#
# Recommended strategy for small CLIP-like students:
#   Stage-1: Linear probe (train classifier head only)
#   Stage-2: Light finetune (unfreeze non-gene backbone with small LR; keep gene frozen)
#
# Example:
# torchrun --nproc_per_node=4 -m scripts.ft_cifar100 \
#   --model ours --distributed \
#   --gene_dir /root/gene_exports/last3 --shallow_layers 3 \
#   --init_ckpt outputs/pretrain_ours_cc3m_last3/ckpt_last.pt \
#   --data_root /root/autodl-tmp/cifar100 --download \
#   --out_dir outputs/ft_cifar100_ours_last3_lpft \
#   --epochs 30 --probe_epochs 5 --finetune_backbone \
#   --batch_size 256 --num_workers 16 --amp --amp_dtype bf16
#
# Notes:
# - Default uses CLIP normalization (mean/std). You can switch to ImageNet norm via --norm imagenet
# - CIFAR images are 32x32; we resize to 224 to match CLIP-like encoders.
# torchrun --nproc_per_node=4 -m scripts.ft_cifar100 \
#   --model ours --distributed \
#   --gene_dir /root/gene_exports/last3 --shallow_layers 3 \
#   --init_ckpt outputs/pretrain_ours_cc3m_last3/ckpt_last.pt \
#   --data_root /root/autodl-tmp/cifar100 --download \
#   --out_dir outputs/ft_cifar100_ours_last3_lpft \
#   --epochs 30 --probe_epochs 5 --finetune_backbone \
#   --batch_size 256 --num_workers 16 \
#   --freeze_gene --amp --amp_dtype bf16

import os
import math
import time
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import sys
import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as T
from scripts.data.transforms import build_clip_image_transform
from scripts.data.webdataset_pairs import WdsPairConfig, build_wds_pairs, default_wds_collate

from scripts.model_factory import create_model_bundle, split_param_groups, set_requires_grad
from scripts.losses import clip_contrastive_loss
from scripts.optim import set_optimizer_lrs
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint

# -------------------------
# Utils
# -------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_for_distributed(is_master: bool):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # torchrun environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        dist.barrier()

    setup_for_distributed(is_main_process())

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[torch.Tensor, ...]:
    """Compute top-k accuracy for classification."""
    maxk = max(topk)
    B = target.size(0)
    _, pred = logits.topk(maxk, 1, True, True)  # (B, maxk)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / B))
    return tuple(res)

def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_checkpoint(state: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)

def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    return ckpt

def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


# -------------------------
# Model builder (adapt to your repo)
# -------------------------
def create_base_model(args) -> nn.Module:
    """
    Robustly locate and call your repo's create_model (same as other scripts).
    Priority:
      1) Ensure repo root is on sys.path
      2) Parse existing scripts (pretrain/finetune) for 'from X import create_model'
      3) Try common module paths
      4) Fallback: scan repo for a module that defines create_model
    """
    from pathlib import Path
    import re

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent  # assuming scripts/ is directly under repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    def _try_get_create_model(mod_name: str):
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, "create_model"):
                return getattr(m, "create_model")
        except Exception:
            return None
        return None

    def _call_create_model(create_model_fn):
        # be tolerant to signature differences
        kw = dict(gene_dir=args.gene_dir, shallow_layers=args.shallow_layers)
        # some repos support image_size; some don't
        try:
            return create_model_fn(args.model, image_size=getattr(args, "image_size", 224), **kw)
        except TypeError:
            return create_model_fn(args.model, **kw)

    # 1) Parse your existing scripts to discover the exact import path you already use
    probe_files = [
        "pretrain_cc3m.py",
        "finetune_retrieval.py",
        "finetune_itm.py",
        "ft_multilabel.py",
        "imagenet_zs.py",
    ]
    import_re = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+create_model\b")

    for fname in probe_files:
        fpath = scripts_dir / fname
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in text[:120]:  # only need the header region
            m = import_re.match(line)
            if m:
                mod_name = m.group(1)
                fn = _try_get_create_model(mod_name)
                if fn is not None:
                    if is_main_process():
                        print(f"[IMPORT] create_model found via {fname}: from {mod_name} import create_model")
                    return _call_create_model(fn)

    # 2) Try a few common module paths (keep short)
    for mod_name in ["models", "models.factory", "model", "model.factory"]:
        fn = _try_get_create_model(mod_name)
        if fn is not None:
            if is_main_process():
                print(f"[IMPORT] create_model found via module: {mod_name}")
            return _call_create_model(fn)

    # 3) Fallback: scan repo for a module defining 'def create_model'
    #    and import it by dotted module path (so relative imports keep working).
    def _iter_py_modules(root: Path):
        skip = {"outputs", "data", "datasets", "wandb", ".git", "__pycache__"}
        for p in root.rglob("*.py"):
            if any(s in p.parts for s in skip):
                continue
            yield p

    for py in _iter_py_modules(repo_root):
        try:
            s = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "def create_model" not in s:
            continue
        rel = py.relative_to(repo_root).with_suffix("")  # e.g. models/factory
        mod_name = ".".join(rel.parts)
        fn = _try_get_create_model(mod_name)
        if fn is not None:
            if is_main_process():
                print(f"[IMPORT] create_model found via scan: {mod_name}")
            return _call_create_model(fn)

    raise ImportError(
        "Cannot locate create_model in this repo. "
        "Please open one of your existing scripts (e.g., scripts/pretrain_cc3m.py) "
        "and copy its 'from XXX import create_model' line into ft_cifar100.py."
    )


def get_image_features(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Prefer CLIP-style encode_image if available; fallback to visual or forward.
    """
    if hasattr(base_model, "encode_image"):
        feat = base_model.encode_image(images)
    elif hasattr(base_model, "visual"):
        feat = base_model.visual(images)
    else:
        feat = base_model(images)

    # Unwrap common return types
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    if isinstance(feat, dict):
        # try common keys
        for k in ["image_embeds", "image_features", "feat", "features", "embedding"]:
            if k in feat:
                feat = feat[k]
                break
        else:
            raise RuntimeError(f"Unknown dict output keys: {list(feat.keys())}")

    # Ensure 2D (B, D)
    if feat.dim() > 2:
        feat = feat.flatten(1)
    return feat

class CIFAR100Classifier(nn.Module):
    def __init__(self, base_model: nn.Module, feat_dim: int, num_classes: int = 100):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = get_image_features(self.base, images)
        # Many CLIP-like encoders already output normalized embedding; no harm keeping it.
        logits = self.head(feat)
        return logits

def freeze_by_keywords(model: nn.Module, keywords, freeze: bool = True):
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in keywords):
            p.requires_grad = not freeze

def freeze_all(model: nn.Module, freeze: bool = True):
    for p in model.parameters():
        p.requires_grad = not freeze


# -------------------------
# FLOPs / Throughput helpers
# -------------------------
@torch.no_grad()
def measure_throughput(model: nn.Module, device: torch.device, image_size: int, batch_size: int,
                       iters: int = 100, warmup: int = 30) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    imgs = batch_size * iters
    return imgs / max(t1 - t0, 1e-9)

@torch.no_grad()
def try_compute_flops(model: nn.Module, device: torch.device, image_size: int) -> Optional[float]:
    """
    Returns GFLOPs for 1 forward pass (1 image) if fvcore or thop is available.
    """
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()

    # fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy).total()
        return flops / 1e9
    except Exception:
        pass

    # thop
    try:
        from thop import profile
        flops, _params = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        pass

    return None


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.cuda.amp.GradScaler], device: torch.device,
                    epoch: int, args) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_n = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=args._amp_dtype):
            logits = model(images)
            loss = ce(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        bsz = targets.size(0)
        top1, top5 = accuracy_topk(logits.detach(), targets, topk=(1, 5))
        total_loss += float(loss.detach()) * bsz
        total_top1 += float(top1) * bsz
        total_top5 += float(top5) * bsz
        total_n += bsz

        if is_main_process() and (step % args.log_every == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(f"[E{epoch:03d}] step={step:05d} loss={float(loss):.4f} "
                  f"top1={float(top1):.2f} top5={float(top5):.2f} lr={lr:.3e}")

    # reduce across processes
    stats = torch.tensor([total_loss, total_top1, total_top5, total_n], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss, total_top1, total_top5, total_n = stats.tolist()

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "loss": total_loss / max(total_n, 1),
        "top1": total_top1 / max(total_n, 1),
        "top5": total_top5 / max(total_n, 1),
        "peak_mem_mb": peak_mem,
    }

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=args._amp_dtype):
            logits = model(images)
            loss = ce(logits, targets)

        bsz = targets.size(0)
        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        total_loss += float(loss) * bsz
        total_top1 += float(top1) * bsz
        total_top5 += float(top5) * bsz
        total_n += bsz

    stats = torch.tensor([total_loss, total_top1, total_top5, total_n], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss, total_top1, total_top5, total_n = stats.tolist()

    return {
        "loss": total_loss / max(total_n, 1),
        "top1": total_top1 / max(total_n, 1),
        "top5": total_top5 / max(total_n, 1),
    }


# -------------------------
# Main
# -------------------------
def build_transforms(image_size: int, norm: str):
    # CLIP norm (OpenAI)
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    # ImageNet norm
    imn_mean = (0.485, 0.456, 0.406)
    imn_std = (0.229, 0.224, 0.225)

    if norm == "clip":
        mean, std = clip_mean, clip_std
    else:
        mean, std = imn_mean, imn_std

    train_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        # light augment; keep it stable for small models
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf

def main():
    parser = argparse.ArgumentParser("CIFAR-100 finetune (transfer)")

    # model args (keep consistent with your other scripts)
    parser.add_argument("--model", type=str, default="ours")
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--init_ckpt", type=str, default="", help="pretrained checkpoint to init from")
    parser.add_argument("--freeze_gene", action="store_true", help="freeze gene params (recommended)")
    parser.add_argument("--gene_keywords", type=str, nargs="+", default=["learngene", "gene"])

    # data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--norm", type=str, default="clip", choices=["clip", "imagenet"])

    # train
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--probe_epochs", type=int, default=5, help="linear probe epochs (head only)")
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="after probe, unfreeze non-gene backbone with small lr")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="head lr")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="backbone lr (stage-2)")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # amp / dist
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--distributed", action="store_true")  # for compatibility
    parser.add_argument("--log_every", type=int, default=50)

    # output
    parser.add_argument("--out_dir", type=str, required=True)

    # speed metrics
    parser.add_argument("--throughput_bs", type=int, default=256)
    parser.add_argument("--throughput_iters", type=int, default=100)

    args = parser.parse_args()
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed + get_rank())

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    args._amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    os.makedirs(args.out_dir, exist_ok=True)

    # data
    train_tf, val_tf = build_transforms(args.image_size, args.norm)
    train_set = torchvision.datasets.CIFAR100(root=args.data_root, train=True, transform=train_tf, download=args.download)
    val_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, transform=val_tf, download=args.download)

    if args.distributed and is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # model
    base = create_base_model(args)
    base.to(device)

    # init from ckpt
    if args.init_ckpt:
        ckpt = load_checkpoint(args.init_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = base.load_state_dict(state, strict=False)
        if is_main_process():
            print(f"[CKPT] loaded init_ckpt={args.init_ckpt}")
            print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    # infer feature dim with a dummy forward
    base.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
        feat = get_image_features(base, dummy)
        feat_dim = feat.shape[-1]

    model = CIFAR100Classifier(base, feat_dim=feat_dim, num_classes=100).to(device)

    # freeze policy:
    # - Always freeze gene (recommended)
    # - Stage-1: freeze full backbone (linear probe)
    if args.freeze_gene:
        freeze_by_keywords(model.base, args.gene_keywords, freeze=True)

    # stage-1: freeze all backbone
    freeze_all(model.base, freeze=True)
    # keep head trainable
    for p in model.head.parameters():
        p.requires_grad = True
    # ensure gene stays frozen if requested
    if args.freeze_gene:
        freeze_by_keywords(model.base, args.gene_keywords, freeze=True)

    # DDP
    if args.distributed and is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # optimizer (will be rebuilt at stage boundary)
    def make_optimizer(stage2: bool):
        m = unwrap_model(model)
        head_params = [p for p in m.head.parameters() if p.requires_grad]

        if stage2 and args.finetune_backbone:
            # unfreeze non-gene backbone
            freeze_all(m.base, freeze=False)
            if args.freeze_gene:
                freeze_by_keywords(m.base, args.gene_keywords, freeze=True)

            backbone_params = [p for p in m.base.parameters() if p.requires_grad]
            param_groups = [
                {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
                {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
            ]
        else:
            # head only
            param_groups = [{"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay}]

        return torch.optim.AdamW(param_groups)

    optimizer = make_optimizer(stage2=False)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # cosine lr schedule (simple)
    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        # warmup 1 epoch
        warmup = len(train_loader)
        if step < warmup:
            return float(step) / float(max(1, warmup))
        # cosine
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # stats
    total_params, _ = count_params(unwrap_model(model))
    if is_main_process():
        print(f"[MODEL] feat_dim={feat_dim} total_params={total_params/1e6:.3f}M")
        print(f"[CFG] epochs={args.epochs} probe_epochs={args.probe_epochs} finetune_backbone={args.finetune_backbone}")
        print(f"[CFG] lr_head={args.lr} lr_backbone={args.lr_backbone} wd={args.weight_decay} amp={args.amp}({args.amp_dtype})")

    best_top1 = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # stage transition
        if epoch == args.probe_epochs:
            if is_main_process():
                print("[STAGE] Switch to stage-2 (light finetune): unfreeze non-gene backbone with small LR.")
            optimizer = make_optimizer(stage2=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)  # re-init
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
            global_step = epoch * len(train_loader)

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)

        # step scheduler per-iter (we used lambda step index)
        # Here we approximate by stepping len(train_loader) times
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1

        val_stats = evaluate(model, val_loader, device, args)

        if is_main_process():
            total_params, trainable_params = count_params(unwrap_model(model))
            print(f"[E{epoch:03d}] "
                  f"train_loss={train_stats['loss']:.4f} train_top1={train_stats['top1']:.2f} train_top5={train_stats['top5']:.2f} "
                  f"val_loss={val_stats['loss']:.4f} val_top1={val_stats['top1']:.2f} val_top5={val_stats['top5']:.2f} "
                  f"trainable={trainable_params/1e6:.3f}M peak_mem={train_stats['peak_mem_mb']:.1f}MB")

        # save best
        if val_stats["top1"] > best_top1:
            best_top1 = val_stats["top1"]
            if is_main_process():
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val": val_stats,
                        "args": vars(args),
                    },
                    os.path.join(args.out_dir, "ckpt_best.pt"),
                )

        if is_main_process() and (epoch % 5 == 0 or epoch == args.epochs - 1):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val": val_stats,
                    "args": vars(args),
                },
                os.path.join(args.out_dir, "ckpt_last.pt"),
            )

    # final metrics: throughput / flops
    if is_main_process():
        total_params, trainable_params = count_params(unwrap_model(model))
        th = measure_throughput(model, device, args.image_size, args.throughput_bs,
                                iters=args.throughput_iters, warmup=30)
        gflops = try_compute_flops(model, device, args.image_size)
        msg = {
            "best_val_top1": best_top1,
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "throughput_img_s": th,
            "gflops_per_image": gflops,
        }
        print("[METRICS]", json.dumps(msg, indent=2))
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2)

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
