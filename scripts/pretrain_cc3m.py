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
# torchrun --nproc_per_node=4 -m scripts.pretrain_cc3m \
#   --model ours \
#   --device cuda \
#   --image_size 224 \
#   --gene_dir /root/gene_exports/last2_plus6 \
#   --shallow_layers 3 \
#   --bottleneck_dim -1 \
#   --bottleneck_dropout 0.0 \
#   --proj_head linear \
#   --proj_dim 512 \
#   --proj_hidden_dim -1 \
#   --proj_dropout 0.0 \
#   --use_tleg \
#   --tleg_target_depth 6 \
#   --tleg_strict \
#   --use_multimodal_init \
#   --tleg_last_epochs -1 \
#   --stem_init_clip_name ViT-B/32 \
#   --freeze_stem_after_init \
#   --cc3m_root /root/autodl-tmp/cc3m/wds \
#   --cc3m_train_shards "" \
#   --cc3m_val_shards "" \
#   --shuffle_buf 5000 \
#   --wds_handler warn \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --coco_max_images 118000 \
#   --coco_mix_prob 0.2 \
#   --coco_mix_prob_cap 0.5 \
#   --samples_per_epoch -1 \
#   --cc3m_samples_est 3300000 \
#   --val_samples 50000 \
#   --epochs 40 \
#   --batch_size 256 \
#   --num_workers 16 \
#   --accum_steps 1 \
#   --lr 3e-4 \
#   --min_lr 1e-5 \
#   --warmup_steps 5000 \
#   --weight_decay 0.2 \
#   --amp \
#   --amp_dtype bf16 \
#   --grad_clip 1.0 \
#   --logit_scale_max 100.0 \
#   --unfreeze_epoch 5 \
#   --unfreeze_steps 0 \
#   --gene_lr_ratio 20.0 \
#   --gene_warmup_steps 1000 \
#   --gene_keywords learngene gene \
#   --distributed \
#   --backend nccl \
#   --seed 42 \
#   --log_every 100 \
#   --out_dir outputs/pretrain_ours_last26

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
import numpy as np

import torch
import torch.nn as nn

import json
import random
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from scripts.data.transforms import build_clip_image_transform
from scripts.data.webdataset_pairs import WdsPairConfig, build_wds_pairs, default_wds_collate

from scripts.model_factory import create_model_bundle, split_param_groups, set_requires_grad
from scripts.losses import clip_contrastive_loss
from scripts.optim import set_optimizer_lrs
from scripts.align.teacher_taps import (
    CLIPTeacherDistiller,
    TapConfig,
    compute_remaining_tap_layers,
    infer_last_gene_layers,
)
from scripts.align.soft_align import (
    SoftAlignWeights,
    embedding_distillation_loss,
    similarity_distillation_loss,
    soft_align_layers,
)
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint
from scripts.utils.device import (
    autocast_context,
    enable_tf32,
    get_default_device,
    is_accelerator_device,
    make_grad_scaler,
    normalize_backend,
    resolve_amp_dtype,
    resolve_device,
    scaler_is_enabled,
)
from tasks.dataset_registry import PretrainDatasetPaths
# --- TF32 (Ampere/Ada/Hopper GPUs) ---
enable_tf32(True)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



# ---------------------------
# CC3M + COCO mixing utilities
# ---------------------------

@dataclass
class CocoPoolStats:
    num_images: int
    num_pairs: int


class CocoCaptionPool:
    """COCO caption pool sampled by image, caption chosen randomly.

    We sample by *image* to avoid overweighting COCO (5 captions per image) and to
    keep memory reasonable. Over time, random caption sampling still exposes the
    full caption diversity.
    """

    def __init__(
        self,
        images_root: str,
        captions_json: str,
        transform: Optional[Callable] = None,
        max_images: int = -1,
    ):
        self.images_root = images_root
        self.transform = transform

        with open(captions_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in data.get("images", [])}

        id2caps = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cap = ann.get("caption", "")
            if img_id is None:
                continue
            if not isinstance(cap, str):
                continue
            cap = cap.strip()
            if not cap:
                continue
            id2caps.setdefault(img_id, []).append(cap)

        paths = []
        caps = []
        num_pairs = 0
        for img_id, c_list in id2caps.items():
            fn = id2file.get(img_id)
            if not fn:
                continue
            paths.append(os.path.join(images_root, fn))
            caps.append(c_list)
            num_pairs += len(c_list)

        if max_images and max_images > 0:
            paths = paths[:max_images]
            caps = caps[:max_images]
            num_pairs = sum(len(x) for x in caps)

        self._paths = paths
        self._caps = caps
        self.stats = CocoPoolStats(num_images=len(paths), num_pairs=int(num_pairs))

        if self.stats.num_images <= 0:
            raise RuntimeError(
                f"Empty COCO pool. Check coco_images={images_root} and coco_captions={captions_json}."
            )

    def sample(self, rnd: random.Random):
        k = rnd.randrange(self.stats.num_images)
        path = self._paths[k]
        c_list = self._caps[k]
        cap = c_list[rnd.randrange(len(c_list))] if c_list else "."

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cap


class MixedWdsCocoPairs(IterableDataset):
    """Mix a CC3M WebDataset stream with COCO captions.

    Yields exactly `samples_per_epoch` samples *per rank* per epoch.
    With num_workers>0, we split this budget across workers to keep the total
    sample count deterministic (important for LR schedule / steps_per_epoch).
    """

    def __init__(
        self,
        cc3m_iterable,
        coco_pool: CocoCaptionPool,
        p_coco: float,
        samples_per_epoch: int,
        seed: int = 42,
    ):
        super().__init__()
        assert 0.0 <= p_coco <= 1.0
        self.cc3m_iterable = cc3m_iterable
        self.coco_pool = coco_pool
        self.p_coco = float(p_coco)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)

    def __iter__(self) -> Iterator:
        worker = get_worker_info()
        if worker is None:
            wid, nworkers = 0, 1
        else:
            wid, nworkers = worker.id, worker.num_workers

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        # Split global per-rank sample budget across workers
        base = self.samples_per_epoch // nworkers
        rem = self.samples_per_epoch % nworkers
        this_n = base + (1 if wid < rem else 0)

        rnd = random.Random(self.seed + 100000 * rank + 1000 * wid)

        cc_it = iter(self.cc3m_iterable)
        for _ in range(this_n):
            use_coco = (self.p_coco > 0.0) and (rnd.random() < self.p_coco)
            if use_coco:
                yield self.coco_pool.sample(rnd)
            else:
                try:
                    yield next(cc_it)
                except StopIteration:
                    cc_it = iter(self.cc3m_iterable)
                    yield next(cc_it)

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


def forward_features_with_aux(
    model: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
):
    if hasattr(model, "encode_image_with_aux") and hasattr(model, "encode_text_with_aux"):
        z_img, v_aux = model.encode_image_with_aux(images)
        z_txt, t_aux = model.encode_text_with_aux(tokens)
        return z_img, z_txt, v_aux, t_aux

    z_img, z_txt = forward_features(model, images, tokens)
    return z_img, z_txt, None, None


def _parse_tap_layers(spec: str, total_layers: int, target_count: int) -> List[int]:
    if spec:
        vals = []
        for chunk in str(spec).replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            vals.append(int(chunk))
        vals = sorted({min(total_layers, max(1, v)) for v in vals})
        if len(vals) == 0:
            raise ValueError(f"Invalid tap layer spec: {spec!r}")
        return vals

    count = max(1, min(int(target_count), int(total_layers)))
    if count == 1:
        return [int(total_layers)]
    pos = np.linspace(1, total_layers, num=count)
    return sorted({int(round(x)) for x in pos})


def build_teacher_distiller_module(args) -> Optional[CLIPTeacherDistiller]:
    if getattr(args, "distill_mode", "none") == "none" or args.model != "ours":
        args.teacher_total_visual_layers = 0
        args.teacher_total_text_layers = 0
        args.gene_visual_layers = []
        args.gene_text_layers = []
        args.distill_vision_layers = []
        args.distill_text_layers = []
        return None

    try:
        import clip
    except Exception as e:
        raise RuntimeError("distillation requires OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`)") from e

    teacher_name = (
        getattr(args, "distill_clip_name", "")
        or getattr(args, "teacher_tap_clip_name", "")
        or getattr(args, "clip_name", "ViT-B/32")
    )
    teacher, _ = clip.load(teacher_name, device="cpu", jit=False)
    teacher.eval()

    n_visual = len(teacher.visual.transformer.resblocks)
    n_text = len(teacher.transformer.resblocks)
    gene_layers = int(getattr(args, "gene_layers", 3))
    gene_visual_layers = infer_last_gene_layers(n_visual, gene_layers)
    gene_text_layers = infer_last_gene_layers(n_text, gene_layers)

    vision_spec = getattr(args, "teacher_tap_vision_layers", "")
    text_spec = getattr(args, "teacher_tap_text_layers", "")
    vision_taps = (
        _parse_tap_layers(vision_spec, n_visual, 3)
        if vision_spec
        else compute_remaining_tap_layers(n_visual, gene_visual_layers, num_taps=3)
    )
    text_taps = (
        _parse_tap_layers(text_spec, n_text, 3)
        if text_spec
        else compute_remaining_tap_layers(n_text, gene_text_layers, num_taps=3)
    )

    args.teacher_total_visual_layers = n_visual
    args.teacher_total_text_layers = n_text
    args.gene_visual_layers = gene_visual_layers
    args.gene_text_layers = gene_text_layers
    args.distill_vision_layers = vision_taps
    args.distill_text_layers = text_taps

    tap_cfg = TapConfig(
        vision_layers=vision_taps,
        text_layers=text_taps,
    )

    if is_main_process():
        print(
            f"[Distill] mode={getattr(args, 'distill_mode', 'none')} teacher={teacher_name} "
            f"gene_layers={gene_layers} gene_visual={gene_visual_layers} gene_text={gene_text_layers} "
            f"vision_taps={tap_cfg.vision_layers} text_taps={tap_cfg.text_layers} "
            f"weights(feature={getattr(args, 'distill_feature_weight', 0.0):.4f}, "
            f"logit={getattr(args, 'distill_logit_weight', 0.0):.4f}, "
            f"embed={getattr(args, 'distill_embed_weight', 0.0):.4f})"
        )

    return CLIPTeacherDistiller(
        clip_model=teacher,
        tap_cfg=tap_cfg,
        device=torch.device(args.device),
        teacher_fp16=bool(getattr(args, "teacher_tap_fp16", True)),
    )


def _select_student_layers_for_distill(layers: List[torch.Tensor], target_count: int) -> List[torch.Tensor]:
    """Map arbitrary shallow depth to the three teacher taps without changing model depth."""

    if len(layers) <= target_count:
        return layers
    if target_count <= 1:
        return [layers[-1]]
    positions = np.linspace(0, len(layers) - 1, num=target_count)
    idxs = sorted({int(round(x)) for x in positions})
    return [layers[i] for i in idxs]


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
    """Distributed-safe evaluation.

    IMPORTANT: clip_contrastive_loss performs collective ops (all_gather) when
    args.distributed=True. Therefore, ALL ranks must enter evaluate() together.
    We also make 'bad batch' skipping synchronous across ranks to avoid
    desynchronization (which would deadlock NCCL collectives).
    """
    model.eval()
    base_model = unwrap_model(model)
    device = args.device

    tot_loss = 0.0
    n = 0
    it = 0

    for batch in val_loader:
        if batch is None:
            bad = True
            loss = None
        else:
            images, tokens = batch
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            with autocast_context(device, enabled=args.amp, dtype=args._amp_autocast_dtype):
                img_f, txt_f = forward_features(base_model, images, tokens)
                loss = clip_contrastive_loss(img_f, txt_f, get_logit_scale(base_model, args.logit_scale_max))

            bad = (loss is None)
            if (not bad) and (not torch.isfinite(loss).all()):
                bad = True

        # ---- DDP-safe: if ANY rank sees a bad batch, ALL ranks skip it ----
        if args.distributed:
            import torch.distributed as dist
            bad_t = torch.tensor([1 if bad else 0], device=device, dtype=torch.int)
            dist.all_reduce(bad_t, op=dist.ReduceOp.SUM)
            bad = (bad_t.item() > 0)

        if not bad:
            tot_loss += float(loss.item())
            n += 1

        it += 1
        if steps > 0 and it >= steps:
            break

    # Reduce (sum_loss, count) across ranks to get a global-average val loss
    if args.distributed:
        import torch.distributed as dist
        t = torch.tensor([tot_loss, float(n)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tot_loss = float(t[0].item())
        n = int(t[1].item())

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
        if getattr(args, "frozen", False):
            set_requires_grad(args._gene_params, False)
        elif global_step < unfreeze_step:
            set_requires_grad(args._gene_params, False)
        else:
            set_requires_grad(args._gene_params, True)

        # TLEG scheduling:
        # - strict (paper-faithful): keep TLEG ON for the whole training (linear constraint always enforced)
        # - non-strict: optionally enable only in last N epochs (your previous behavior)
        if args.use_tleg:
            if getattr(args, "tleg_strict", False):
                set_all_learngene_tleg_active(base_model, True)
                if is_main_process() and epoch == 0:
                    print(f"[TLEGStrict] enabled from epoch=0 (target_depth={args.tleg_target_depth})")
            elif args.tleg_last_epochs > 0:
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
            if getattr(args, "frozen", False):
                set_requires_grad(args._gene_params, False)
            else:
                set_requires_grad(args._gene_params, global_step >= unfreeze_step)

        # LR schedule (by optimizer step)
        lr_new = lr_warmup_cosine(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)

        if args._has_gene_groups:
            # gene lr: 0 before unfreeze, then warmup to lr_new/gene_lr_ratio
            if getattr(args, "frozen", False):
                gene_lr = 0.0
            elif global_step < unfreeze_step:
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

        contrastive_loss = None
        tap_feature_loss = None
        logit_distill_loss = None
        embed_distill_loss = None
        with autocast_context(device, enabled=args.amp, dtype=args._amp_autocast_dtype):
            img_f, txt_f, v_aux, t_aux = forward_features_with_aux(base_model, images, tokens)
            contrastive_loss = clip_contrastive_loss(img_f, txt_f, get_logit_scale(base_model, args.logit_scale_max))
            loss = contrastive_loss
            if (
                loss is not None
                and getattr(args, "_teacher_distiller", None) is not None
                and v_aux is not None
                and t_aux is not None
            ):
                teacher_out = args._teacher_distiller(images, tokens)
                mode = getattr(args, "distill_mode", "none")
                if (
                    mode in ("tap", "tap_logit")
                    and float(getattr(args, "distill_feature_weight", 0.0)) > 0
                    and teacher_out.vision_layers
                    and teacher_out.text_layers
                ):
                    student_v = _select_student_layers_for_distill(v_aux.get("shallow", []), len(teacher_out.vision_layers))
                    student_t = _select_student_layers_for_distill(t_aux.get("shallow", []), len(teacher_out.text_layers))
                    if len(student_v) == len(teacher_out.vision_layers) and len(student_t) == len(teacher_out.text_layers):
                        tap_feature_loss = (
                            soft_align_layers(
                                student_v,
                                teacher_out.vision_layers,
                                weights=args._teacher_tap_weights,
                                drop_cls=bool(getattr(args, "teacher_tap_drop_cls", True)),
                            )
                            + soft_align_layers(
                                student_t,
                                teacher_out.text_layers,
                                weights=args._teacher_tap_weights,
                                drop_cls=False,
                            )
                        )
                        loss = loss + float(args.distill_feature_weight) * tap_feature_loss

                if mode == "tap_logit":
                    if float(getattr(args, "distill_logit_weight", 0.0)) > 0:
                        logit_distill_loss = similarity_distillation_loss(
                            img_f,
                            txt_f,
                            teacher_out.similarity_logits,
                            student_logit_scale=get_logit_scale(base_model, args.logit_scale_max),
                            temperature=float(getattr(args, "distill_temperature", 1.0)),
                        )
                        loss = loss + float(args.distill_logit_weight) * logit_distill_loss
                    if float(getattr(args, "distill_embed_weight", 0.0)) > 0:
                        embed_distill_loss = embedding_distillation_loss(
                            img_f,
                            txt_f,
                            teacher_out.image_features,
                            teacher_out.text_features,
                        )
                        loss = loss + float(args.distill_embed_weight) * embed_distill_loss
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
            extra_parts = []
            if contrastive_loss is not None:
                extra_parts.append(f"contrastive={float(contrastive_loss.item()):.4f}")
            if tap_feature_loss is not None:
                extra_parts.append(f"tap_feature={float(tap_feature_loss.item()):.4f}")
            if logit_distill_loss is not None:
                extra_parts.append(f"logit_distill={float(logit_distill_loss.item()):.4f}")
            if embed_distill_loss is not None:
                extra_parts.append(f"embed_distill={float(embed_distill_loss.item()):.4f}")
            extra = (" " + " ".join(extra_parts)) if extra_parts else ""
            print(f"[E{epoch:03d}] step={global_step} loss={(loss.item()*accum):.4f}{extra} {lr_info}")

        global_step += 1
        opt_steps_done += 1
        if opt_steps_done >= steps_this_epoch:
            break

    return global_step


def main():
    parser = argparse.ArgumentParser("Pretrain on CC3M (ours/clip/tinyclip) via WebDataset")
    data_defaults = PretrainDatasetPaths()

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default=get_default_device(), help="cpu/cuda/npu/auto")
    parser.add_argument("--image_size", type=int, default=224)

    # ours
    parser.add_argument("--gene_dir", type=str, default="", help="folder containing learngene_visual.pt / learngene_text.pt ...")
    parser.add_argument("--gene_layers", type=int, default=3, choices=[2, 3], help="Stable last-N gene depth inherited from CLIP.")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--shallow_type", type=str, default="transformer", choices=["transformer", "cnn"])
    parser.add_argument("--shallow_kernel_size", type=int, default=3)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--use_tleg", action="store_true")
    parser.add_argument("--no_tleg", action="store_true", help="Disable TLEG even if the base config enables it.")
    parser.add_argument("--tleg_target_depth", type=int, default=4)
    parser.add_argument("--tleg_strict", action="store_true", help="paper-faithful strict TLEG (only 2 endpoints trainable)")
    parser.add_argument("--use_multimodal_init", action="store_true")
    parser.add_argument("--frozen", action="store_true", help="Keep learngene frozen for the whole run.")
    parser.add_argument("--no_frozen", action="store_true", help="Alias for enabling learngene unfreezing.")

    # CLIP stem init (ours): deprecated; main experiments keep it disabled.
    parser.add_argument("--clip_init", action="store_true", help="[Deprecated] Ignored by the revised main experiments.")
    parser.add_argument("--no_clip_init", action="store_true", help="[Deprecated] Kept for backward-compatible configs.")
    parser.add_argument(
        "--disable_stem_init_from_clip",
        action="store_true",
        help="[Deprecated] CLIP stem initialization is disabled by default in the revised setup.",
    )
    parser.add_argument(
        "--stem_init_clip_name",
        type=str,
        default="ViT-B/32",
        help="Teacher CLIP name used for stem initialization.",
    )
    parser.add_argument(
        "--freeze_stem_after_init",
        action="store_true",
        help="Freeze stem params after copying from CLIP.",
    )

    parser.add_argument(
        "--tleg_last_epochs",
        type=int,
        default=5,
        help="enable TLEG only for last N epochs (default 1). set 0 to never enable.",
    )
    parser.add_argument("--distill_mode", type=str, default="tap_logit", choices=["none", "tap", "tap_logit"])
    parser.add_argument("--distill_clip_name", type=str, default="ViT-B/32", help="Teacher CLIP used only during training distillation.")
    parser.add_argument("--distill_feature_weight", type=float, default=0.1)
    parser.add_argument("--distill_logit_weight", type=float, default=0.5)
    parser.add_argument("--distill_embed_weight", type=float, default=0.05)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--teacher_tap", action="store_true", help="[Deprecated] Use --distill_mode tap_logit instead.")
    parser.add_argument("--no_teacher_tap", action="store_true", help="[Deprecated] Alias for --distill_mode none.")
    parser.add_argument("--teacher_tap_clip_name", type=str, default="", help="[Deprecated] Use --distill_clip_name.")
    parser.add_argument("--teacher_tap_weight", type=float, default=0.0, help="[Deprecated] Use --distill_feature_weight.")
    parser.add_argument("--teacher_tap_vision_layers", type=str, default="", help="Comma-separated teacher vision tap layers. Empty = auto.")
    parser.add_argument("--teacher_tap_text_layers", type=str, default="", help="Comma-separated teacher text tap layers. Empty = auto.")
    parser.add_argument("--teacher_tap_fp16", action="store_true", help="Run the teacher tap model in fp16 on CUDA.")
    parser.add_argument("--teacher_tap_drop_cls", action="store_true", help="Drop vision CLS token when aligning teacher taps.")
    parser.add_argument("--teacher_tap_w_cos", type=float, default=1.0)
    parser.add_argument("--teacher_tap_w_stat", type=float, default=0.25)
    parser.add_argument("--teacher_tap_w_delta", type=float, default=0.25)

    # clip baseline
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    # tinyclip baseline (wkcn)
    parser.add_argument("--tinyclip_ckpt", type=str, default="", help="TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt")

    # data
    parser.add_argument("--cc3m_root", type=str, default=data_defaults.cc3m_root)
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
    parser.add_argument("--weight_decay", type=float, default=None, help="Alias for --wd (weight decay).")
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
    parser.add_argument("--backend", type=str, default="auto", help="Distributed backend: auto/nccl/hccl/gloo.")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/pretrain_cc3m")
    parser.add_argument("--save_every_epoch", action="store_true")

    args = parser.parse_args()


    # ---- argument aliases / derived flags ----
    args.device = str(resolve_device(args.device, allow_cpu_fallback=False))
    args.backend = normalize_backend(args.backend, args.device)

    # allow --weight_decay as an alias of --wd for compatibility with other codebases
    if getattr(args, "weight_decay", None) is not None:
        args.wd = float(args.weight_decay)

    # Revised main setup: CLIPInit is kept only as a deprecated CLI surface and is forced off.
    if getattr(args, "clip_init", False) and is_main_process():
        print("[WARN] --clip_init is deprecated and ignored; revised ClipGene disables CLIPInit.")
    args.clip_init = False
    args.stem_init_from_clip = False

    if getattr(args, "no_tleg", False):
        args.use_tleg = False

    args.frozen = bool(getattr(args, "frozen", False))
    if getattr(args, "no_frozen", False):
        args.frozen = False

    if getattr(args, "no_teacher_tap", False):
        args.distill_mode = "none"
    elif getattr(args, "teacher_tap", False) and getattr(args, "distill_mode", "none") == "none":
        args.distill_mode = "tap_logit"
    args.teacher_tap = args.distill_mode != "none"
    if getattr(args, "teacher_tap_clip_name", "") and not getattr(args, "distill_clip_name", ""):
        args.distill_clip_name = args.teacher_tap_clip_name
    if float(getattr(args, "teacher_tap_weight", 0.0)) > 0:
        args.distill_feature_weight = float(args.teacher_tap_weight)
    args._teacher_tap_weights = SoftAlignWeights(
        w_cos=float(getattr(args, "teacher_tap_w_cos", 1.0)),
        w_stat=float(getattr(args, "teacher_tap_w_stat", 0.25)),
        w_delta=float(getattr(args, "teacher_tap_w_delta", 0.25)),
    )
    args.distill_train_only = args.distill_mode != "none"

    args._amp_autocast_dtype = resolve_amp_dtype(args.device, args.amp, args.amp_dtype)
    args.amp_dtype = "bf16" if args._amp_autocast_dtype == torch.bfloat16 else "fp16"
    args._use_scaler = bool(
        args.amp
        and args._amp_autocast_dtype == torch.float16
        and is_accelerator_device(args.device)
    )

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend, args.device)

    world_size = get_world_size()


    # dataset patterns
    train_shards = args.cc3m_train_shards or guess_cc3m_shards(args.cc3m_root, "train")
    val_shards = args.cc3m_val_shards
    if not val_shards:
        try:
            val_shards = guess_cc3m_shards(args.cc3m_root, "val")
        except Exception:
            val_shards = ""  # optional

    # optional COCO pool (for mixing + sizing)
    use_coco = bool(args.coco_images and args.coco_captions)
    if (args.coco_images and not args.coco_captions) or (args.coco_captions and not args.coco_images):
        raise ValueError("To mix COCO, you must set BOTH --coco_images and --coco_captions.")

    coco_pool = None
    coco_pairs = 0
    coco_images = 0
    if use_coco:
        if not os.path.isdir(args.coco_images):
            raise FileNotFoundError(f"--coco_images is not a directory: {args.coco_images}")
        if not os.path.isfile(args.coco_captions):
            raise FileNotFoundError(f"--coco_captions is not a file: {args.coco_captions}")
        # transform is attached later (after we build it)
        coco_pool = CocoCaptionPool(
            images_root=args.coco_images,
            captions_json=args.coco_captions,
            transform=None,
            max_images=args.coco_max_images,
        )
        coco_pairs = int(coco_pool.stats.num_pairs)
        coco_images = int(coco_pool.stats.num_images)

    # epoch sizing (global)
    # If samples_per_epoch=-1, maximize virtual pretrain sample budget by using CC3M_est + COCO_pairs.
    global_samples_per_epoch = args.samples_per_epoch
    if global_samples_per_epoch < 0:
        global_samples_per_epoch = int(args.cc3m_samples_est + (coco_pairs if use_coco else 0))

    # per-rank samples (drop_last in batching makes steps deterministic)
    samples_per_rank = int(math.ceil(global_samples_per_epoch / float(world_size)))
    val_samples_per_rank = int(math.ceil(args.val_samples / float(world_size))) if args.val_samples > 0 else 0


    # Data pipeline: (image_tensor, caption_str)
    transform = build_clip_image_transform(args.image_size, is_train=True)
    if use_coco:
        assert coco_pool is not None
        coco_pool.transform = transform

    train_cfg = WdsPairConfig(
        shards=train_shards,
        shuffle_buf=args.shuffle_buf,
        resampled=True,
        samples_per_epoch=samples_per_rank,  # per-rank CC3M budget (mixed dataset controls final epoch size)
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
        if use_coco:
            print("==== COCO Mix ====")
            print(f"coco_images: {coco_images}")
            print(f"coco_pairs:  {coco_pairs}")
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
    args._teacher_distiller = build_teacher_distiller_module(args)

    # DDP
    if args.distributed and is_accelerator_device(args.device):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    base_model = unwrap_model(model)
    base_model.float()
    if args.use_tleg and (not getattr(args, "tleg_strict", False)) and args.tleg_last_epochs > 0:
        set_all_learngene_tleg_active(base_model, False)


    # Build dataloaders
    train_collate = default_wds_collate(tokenize)  # collate expects list[(img_tensor, caption_str)]

    if use_coco:
        assert coco_pool is not None
        # CC3M stream yields single samples
        cc3m_stream = build_wds_pairs(train_cfg, transform=transform)

        # Auto mixing probability by size, capped for stability
        denom = float(max(1, args.cc3m_samples_est + coco_pairs))
        auto_p = float(coco_pairs) / denom
        p_coco = float(args.coco_mix_prob) if args.coco_mix_prob >= 0 else min(float(args.coco_mix_prob_cap), auto_p)
        p_coco = max(0.0, min(1.0, p_coco))

        if is_main_process():
            print(f"mix_prob(coco)={p_coco:.3f} (auto={auto_p:.3f}, cap={args.coco_mix_prob_cap})")

        train_ds = MixedWdsCocoPairs(
            cc3m_iterable=cc3m_stream,
            coco_pool=coco_pool,
            p_coco=p_coco,
            samples_per_epoch=samples_per_rank,
            seed=args.seed,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_collate,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        # CC3M-only training (original fast WebDataset pipeline)
        train_ds = build_wds_pairs(train_cfg, transform=transform)
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
    scaler = make_grad_scaler(args.device, enabled=args._use_scaler, init_scale=2.0**12, growth_interval=2000)
    args._use_scaler = scaler_is_enabled(scaler)

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
        if val_loader is not None:
            val_batches = max(1, (val_samples_per_rank // args.batch_size))
            val_loss = evaluate(args, model, val_loader, steps=val_batches)
            if is_main_process():
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
