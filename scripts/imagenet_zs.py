# scripts/eval_zeroshot_imagenet1k.py

# torchrun --nproc_per_node=4 -m scripts.imagenet_zs \
#   --model ours --distributed --gene_dir /root/gene_exports/last3 --shallow_layers 3 --init_ckpt outputs/pretrain_ours_last3/ckpt_last.pt\
#   --imagenet_val_dir /root/autodl-tmp/imagenet/val \
#   --imagenet_val_labels /root/autodl-tmp/imagenet/ImageNet_val_label.txt \
#   --class_index_json /root/autodl-tmp/imagenet/ImageNet_class_index.json \
#   --image_size 224 --batch_size 128 --num_workers 16 \
#   --amp --amp_dtype bf16 --use_tleg --tleg_target_depth 6\
#   --template_set clip --max_synonyms 3 --text_batch_size 256 \
#   --cache_classifier outputs/cache/zs_W_ours_last3.pt \
#   --out_dir outputs/eval_imagenet_zs_ours
# torchrun --nproc_per_node=4 -m scripts.imagenet_zs \
#     --model clip --clip_name ViT-B/32 --distributed\
#   --imagenet_val_dir /root/autodl-tmp/imagenet/val \
#   --imagenet_val_labels /root/autodl-tmp/imagenet/ImageNet_val_label.txt \
#   --class_index_json /root/autodl-tmp/imagenet/ImageNet_class_index.json \
#   --image_size 224 --batch_size 128 --num_workers 16 \
#   --amp --amp_dtype bf16\
#   --template_set clip --max_synonyms 3 --text_batch_size 256 \
#   --cache_classifier outputs/cache/zs_W_clip.pt \
#   --out_dir outputs/eval_imagenet_zs_clip
# torchrun --nproc_per_node=4 -m scripts.imagenet_zs \
#     --model tinyclip --distributed --tinyclip_ckpt /root/autodl-tmp/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt\
#   --imagenet_val_dir /root/autodl-tmp/imagenet/val \
#   --imagenet_val_labels /root/autodl-tmp/imagenet/ImageNet_val_label.txt \
#   --class_index_json /root/autodl-tmp/imagenet/ImageNet_class_index.json \
#   --image_size 224 --batch_size 128 --num_workers 16 \
#   --amp --amp_dtype bf16\
#   --template_set clip --max_synonyms 3 --text_batch_size 256 \
#   --cache_classifier outputs/cache/zs_W_clip.pt \
#   --out_dir outputs/eval_imagenet_zs_clip
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.data.transforms import build_clip_image_transform
from scripts.model_factory import create_model_bundle
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model

# torch 2.x flop counter
# ---- FLOPs backends (torch flop_counter / fvcore / thop) ----
FlopCounterMode = None
FlopCountAnalysis = None
thop_profile = None

try:
    # torch 2.x (not always available)
    from torch.utils.flop_counter import FlopCounterMode  # type: ignore
except Exception:
    FlopCounterMode = None

try:
    # works on torch 1.x/2.x
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except Exception:
    FlopCountAnalysis = None

try:
    from thop import profile as thop_profile  # type: ignore
except Exception:
    thop_profile = None


# -----------------------
# Prompt templates (good for zero-shot)
# -----------------------
# A slightly larger template set generally helps small text towers.
# If你觉得太慢，可用 --template_set simple
IMAGENET_TEMPLATES_CLIP = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of my {}.",
    "a photo of many {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a photo of a {} in the wild.",
    "a photo of the {} in the wild.",
    "a photo of a {} in a room.",
    "a photo of a {} outdoors.",
    "a photo of a {} indoors.",
    "a photo of a {} on a table.",
    "a photo of a {} on the ground.",
    "a photo of a {} in the water.",
    "a photo of a {} in the sky.",
    "a photo of a {} on the beach.",
    "a photo of a {} in the snow.",
    "a photo of a {} in the forest.",
    "a photo of a {} in a field.",
    "a photo of a {} on the street.",
    "a photo of a {} in a park.",
    "a photo of a {} at night.",
    "a photo of a {} during the day.",
    "a photo of a {} in sunlight.",
    "a photo of a {} in shadow.",
    "a photo of a {} from above.",
    "a photo of a {} from below.",
    "a photo of a {} from the side.",
    "a photo of a {} in front view.",
    "a photo of a {} in profile.",
    "a photo of a {} on a shelf.",
    "a photo of a {} in a box.",
    "a photo of a {} in a zoo.",
    "a photo of a {} in a museum.",
    "a photo of a {} in a kitchen.",
    "a photo of a {} in a bathroom.",
    "a photo of a {} in a garden.",
    "a photo of a {} in a stadium.",
    "a photo of a {} in a store.",
    "a photo of a {} in a supermarket.",
    "a photo of a {} in a factory.",
    "a photo of a {} in a lab.",
    "a photo of a {} with a person.",
    "a photo of a {} with people.",
    "a photo of a {} with other objects.",
    "a photo of a {} on a road.",
    "a photo of a {} on a highway.",
    "a photo of a {} in the desert.",
]

IMAGENET_TEMPLATES_SIMPLE = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a good photo of a {}.",
    "a photo of many {}.",
]


# -----------------------
# ImageNet val dataset (flat images + label file)
# -----------------------
class ImageNetValFlat(Dataset):
    """
    val_dir: a folder containing only images, e.g. ILSVRC2012_val_00000001.JPEG ...
    label_file: text file lines:  <filename> <synset>
      e.g. ILSVRC2012_val_00000001.JPEG n01751748
    """
    def __init__(self, val_dir: str, label_file: str, transform, synset_to_idx: Dict[str, int]):
        self.val_dir = val_dir
        self.transform = transform
        self.synset_to_idx = synset_to_idx

        items: List[Tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fn, syn = parts[0], parts[1]
                if syn not in synset_to_idx:
                    # unseen synset (shouldn't happen if mapping is correct)
                    continue
                path = os.path.join(val_dir, fn)
                items.append((path, synset_to_idx[syn]))

        if len(items) == 0:
            raise RuntimeError(
                f"Empty ImageNet val set. Check:\n"
                f"  val_dir={val_dir}\n"
                f"  label_file={label_file}\n"
                f"  mapping size={len(synset_to_idx)}"
            )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


# -----------------------
# Utilities
# -----------------------
def count_params(model: nn.Module) -> Tuple[int, int, float]:
    total = 0
    trainable = 0
    nbytes = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        nbytes += n * p.element_size()
    size_mb = float(nbytes) / (1024.0 * 1024.0)
    return total, trainable, size_mb


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def try_find_class_index_json(user_path: str = "") -> str:
    """
    Try to locate imagenet_class_index.json automatically.
    """
    cands = []
    if user_path:
        cands.append(user_path)

    # common relative paths
    cands += [
        "imagenet_class_index.json",
        "data/imagenet_class_index.json",
        "assets/imagenet_class_index.json",
        "scripts/imagenet_class_index.json",
    ]

    # env var
    envp = os.environ.get("IMAGENET_CLASS_INDEX_JSON", "")
    if envp:
        cands.append(envp)

    # home cache
    home = os.path.expanduser("~")
    cands += [
        os.path.join(home, ".cache", "imagenet_class_index.json"),
        os.path.join(home, ".cache", "clip", "imagenet_class_index.json"),
    ]

    for p in cands:
        if p and os.path.isfile(p):
            return p
    return ""


def load_imagenet_class_index(class_index_json: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Supports Keras-like format:
      {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
    Returns:
      synsets[idx], classnames[idx], synset_to_idx
    """
    with open(class_index_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # normalize to dict[str, [synset, name]]
    if isinstance(data, list):
        # list of [synset, name] by index
        dd = {str(i): v for i, v in enumerate(data)}
        data = dd

    synsets = [""] * 1000
    names = [""] * 1000
    for k, v in data.items():
        idx = int(k)
        if idx < 0 or idx >= 1000:
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            synsets[idx] = str(v[0])
            names[idx] = str(v[1])
        else:
            # unknown format
            synsets[idx] = ""
            names[idx] = str(v)

    if any(s == "" for s in synsets):
        # still usable, but mapping might be incomplete
        missing = sum(1 for s in synsets if s == "")
        print(f"[WARN] class_index_json seems incomplete: missing_synsets={missing}/1000")

    synset_to_idx = {s: i for i, s in enumerate(synsets) if s}
    return synsets, names, synset_to_idx


def split_synonyms(name: str, max_synonyms: int = 3) -> List[str]:
    """
    ImageNet names often look like: "tench, Tinca tinca"
    We'll split by comma and keep up to max_synonyms.
    """
    parts = [p.strip() for p in name.split(",") if p.strip()]
    if not parts:
        return [name.strip()] if name.strip() else []
    return parts[:max_synonyms]


def forward_encode_image(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized image embedding [B,D].
    Works for:
      - OpenAI CLIP / TinyCLIP: model.encode_image
      - ours (StudentCLIP): vision_stem + vision_tower
    """
    if hasattr(base_model, "encode_image"):
        z = base_model.encode_image(images)
        return l2norm(z.float())
    # ours
    if all(hasattr(base_model, k) for k in ["vision_stem", "vision_tower"]):
        v = base_model.vision_stem(images)
        z = base_model.vision_tower(v)
        return l2norm(z.float())
    raise RuntimeError("Model has no encode_image and is not StudentCLIP-style (vision_stem/vision_tower)")


def forward_encode_text(base_model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized text embedding [B,D].
    """
    if hasattr(base_model, "encode_text"):
        z = base_model.encode_text(tokens)
        return l2norm(z.float())
    if all(hasattr(base_model, k) for k in ["text_stem", "text_tower"]):
        t = base_model.text_stem(tokens)
        z = base_model.text_tower(t, text=tokens)
        return l2norm(z.float())
    raise RuntimeError("Model has no encode_text and is not StudentCLIP-style (text_stem/text_tower)")


@dataclass
class ZSConfig:
    template_set: str = "clip"
    max_synonyms: int = 3
    text_batch_size: int = 256


@torch.no_grad()
def build_zeroshot_classifier(
    base_model: nn.Module,
    tokenize,
    classnames: List[str],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    cfg: ZSConfig,
    cache_path: str = "",
) -> torch.Tensor:
    """
    Build zeroshot classifier weights W: [D, C] (normalized).
    - prompt ensemble + synonym split
    - returns W on CPU (float32), move to GPU later.
    """
    if cache_path and os.path.isfile(cache_path):
        W = torch.load(cache_path, map_location="cpu")
        if isinstance(W, torch.Tensor) and W.ndim == 2:
            if is_main_process():
                print(f"[CACHE] loaded classifier_W from {cache_path}  shape={tuple(W.shape)}")
            return W

    if cfg.template_set == "simple":
        templates = IMAGENET_TEMPLATES_SIMPLE
    else:
        templates = IMAGENET_TEMPLATES_CLIP

    all_w = []
    for cname in classnames:
        syns = split_synonyms(cname, max_synonyms=cfg.max_synonyms)
        if len(syns) == 0:
            syns = [cname]

        # prompts
        texts = []
        for s in syns:
            for t in templates:
                texts.append(t.format(s))

        toks = tokenize(texts)  # [T, ctx]
        # encode in chunks
        zs = []
        for i in range(0, toks.shape[0], cfg.text_batch_size):
            bt = toks[i:i + cfg.text_batch_size].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                z = forward_encode_text(base_model, bt)  # [b,D] normalized
            zs.append(z.float().cpu())
        z_all = torch.cat(zs, dim=0)  # [T,D]
        w = l2norm(z_all.mean(dim=0, keepdim=True))  # [1,D]
        all_w.append(w)

    W = torch.cat(all_w, dim=0)  # [C,D]
    W = l2norm(W).t().contiguous()  # [D,C]

    if cache_path:
        mkdir(os.path.dirname(cache_path) or ".")
        torch.save(W, cache_path)
        if is_main_process():
            print(f"[CACHE] saved classifier_W to {cache_path}")

    return W


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    return pred.eq(y.view(-1, 1)).any(dim=1).float().sum()


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed as dist
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# -----------------------
# FLOPs counting wrapper
# -----------------------
class EncodeImageWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor):
        return forward_encode_image(self.base_model, x)


def estimate_flops_per_image(base_model: nn.Module, device: torch.device, image_size: int,
                             amp: bool, amp_dtype: torch.dtype) -> Tuple[int, str]:
    """
    Count FLOPs for encode_image(1,3,H,W).
    Returns: (flops_int, backend_name)
    """
    wrapper = EncodeImageWrapper(base_model).to(device).eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    # 1) torch flop_counter (if available)
    if FlopCounterMode is not None:
        try:
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                with FlopCounterMode(wrapper, display=False) as fc:
                    _ = wrapper(x)
            return int(fc.get_total_flops()), "torch.flop_counter"
        except Exception as e:
            if is_main_process():
                print(f"[WARN] torch flop_counter failed -> fallback. err={e.__class__.__name__}: {e}")

    # 2) fvcore
    if FlopCountAnalysis is not None:
        try:
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                _ = wrapper(x)  # warm
            # fvcore一般在fp32统计更稳
            wrapper_fp32 = wrapper.float()
            x_fp32 = x.float()
            flops = FlopCountAnalysis(wrapper_fp32, x_fp32).total()
            return int(flops), "fvcore"
        except Exception as e:
            if is_main_process():
                print(f"[WARN] fvcore failed -> fallback. err={e.__class__.__name__}: {e}")

    # 3) thop (approx)
    if thop_profile is not None:
        try:
            wrapper_fp32 = wrapper.float()
            x_fp32 = x.float()
            flops, _params = thop_profile(wrapper_fp32, inputs=(x_fp32,), verbose=False)
            return int(flops), "thop"
        except Exception as e:
            if is_main_process():
                print(f"[WARN] thop failed. err={e.__class__.__name__}: {e}")

    return -1, "none"



def main():
    parser = argparse.ArgumentParser("ImageNet-1K zero-shot eval (top1/top5) with prompt ensemble & metrics")

    # model selection (match your repo)
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ours
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

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

    parser.add_argument("--init_ckpt", type=str, default="")

    # clip / tinyclip
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--tinyclip_ckpt", type=str, default="")

    # data: flat val + label file
    parser.add_argument("--imagenet_val_dir", type=str, required=True, help="val folder containing only images")
    parser.add_argument("--imagenet_val_labels", type=str, required=True, help="label file: '<filename> <synset>' per line")

    # class mapping (for best zero-shot prompts)
    parser.add_argument("--class_index_json", type=str, default="", help="imagenet_class_index.json (Keras-style)")
    parser.add_argument("--use_synset_as_classname", action="store_true",
                        help="If no class_index_json, use synset id as prompt text (worse accuracy).")

    # eval settings
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    # prompt ensemble knobs
    parser.add_argument("--template_set", type=str, default="clip", choices=["clip", "simple"])
    parser.add_argument("--max_synonyms", type=int, default=3)
    parser.add_argument("--text_batch_size", type=int, default=256)

    # efficiency / debug
    parser.add_argument("--limit", type=int, default=-1, help="only evaluate first N samples (debug)")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")

    # cache
    parser.add_argument("--cache_classifier", type=str, default="",
                        help="path to cache classifier_W (model-specific). e.g. outputs/zs_W_ours.pt")

    # output
    parser.add_argument("--out_dir", type=str, default="outputs/eval_imagenet_zs")

    args = parser.parse_args()

    # ours: default stem init is ON unless explicitly disabled
    args.stem_init_from_clip = (not getattr(args, 'disable_stem_init_from_clip', False))
    if getattr(args, 'no_tleg', False):
        args.use_tleg = False

    # AMP dtype
    if args.amp:
        if args.amp_dtype == "auto":
            args.amp_dtype = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        if args.amp_dtype == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            if is_main_process():
                print("[WARN] bf16 requested but not supported; fallback to fp16")
            args.amp_dtype = "fp16"
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    else:
        amp_dtype = torch.float16  # unused when amp disabled

    # distributed
    if args.distributed:
        setup_distributed(args.backend)

    seed_everything(args.seed)
    mkdir(args.out_dir)

    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")

    # build model bundle
    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")
    if args.model == "tinyclip" and not args.tinyclip_ckpt:
        raise ValueError("--tinyclip_ckpt is required for --model tinyclip")

    bundle = create_model_bundle(args)
    model = bundle.model.to(device).eval()
    tokenize = bundle.tokenize
    base_model = unwrap_model(model)
    # ---- load pretrained checkpoint (ours / clip / tinyclip all supported) ----
    if args.init_ckpt:
        if not os.path.isfile(args.init_ckpt):
            raise FileNotFoundError(f"--init_ckpt not found: {args.init_ckpt}")

        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        # your pretrain_cc3m.py saves {"model": state_dict, ...}
        state = ckpt.get("model", ckpt)

        # handle DDP-saved keys
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = base_model.load_state_dict(state, strict=False)

        if is_main_process():
            print(f"[CKPT] loaded: {args.init_ckpt}")
            if len(missing) > 0:
                print(f"[CKPT] missing keys (show first 20): {missing[:20]}")
            if len(unexpected) > 0:
                print(f"[CKPT] unexpected keys (show first 20): {unexpected[:20]}")

    # mapping synset -> idx and classnames
    class_index_path = try_find_class_index_json(args.class_index_json)
    synset_to_idx: Dict[str, int] = {}
    classnames: List[str] = []
    synsets: List[str] = []

    if class_index_path:
        synsets, classnames, synset_to_idx = load_imagenet_class_index(class_index_path)
        if is_main_process():
            print(f"[CLASSMAP] loaded {len(synset_to_idx)} synsets from: {class_index_path}")
    else:
        # fallback: build mapping from label file unique synsets
        if not args.use_synset_as_classname:
            if is_main_process():
                print("[WARN] No imagenet_class_index.json found. "
                      "For best accuracy, provide --class_index_json. "
                      "Falling back to synset-as-classname mode.")
        synset_set = set()
        with open(args.imagenet_val_labels, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    synset_set.add(parts[1])
        synsets = sorted(list(synset_set))
        synset_to_idx = {s: i for i, s in enumerate(synsets)}
        classnames = synsets[:]  # use synset id as name
        if is_main_process():
            print(f"[CLASSMAP] fallback mapping from labels: {len(classnames)} classes (synset-as-text)")

    # Build dataset/loader
    transform = build_clip_image_transform(args.image_size, is_train=False)
    ds = ImageNetValFlat(args.imagenet_val_dir, args.imagenet_val_labels, transform, synset_to_idx)

    if args.limit > 0:
        # cheap limit by slicing items
        ds.items = ds.items[:args.limit]

    sampler = None
    if args.distributed:
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Metrics: params, flops, memory
    total_p, trainable_p, size_mb = count_params(base_model)
    if is_main_process():
        print("==== Model ====")
        print(f"model: {args.model}")
        if args.model == "clip":
            print(f"clip_name: {args.clip_name}")
        if args.model == "tinyclip":
            print(f"tinyclip_ckpt: {args.tinyclip_ckpt}")
        if args.model == "ours":
            print(f"gene_dir: {args.gene_dir}")
            print(f"shallow_layers: {args.shallow_layers}")
        print(f"params_total: {total_p:,}")
        print(f"params_trainable: {trainable_p:,}")
        print(f"model_size_mb (by param dtype): {size_mb:.2f} MB")

    # FLOPs per image (encode_image)
    try:
        flops_img, flops_backend = estimate_flops_per_image(base_model, device, args.image_size, args.amp, amp_dtype)
    except Exception as e:
        flops_img = -1
        if is_main_process():
            print(f"[WARN] FLOPs counting failed: {e.__class__.__name__}: {e}")

    # build classifier W
    zs_cfg = ZSConfig(
        template_set=args.template_set,
        max_synonyms=args.max_synonyms,
        text_batch_size=args.text_batch_size,
    )
    if is_main_process():
        print("==== Zero-shot classifier ====")
        print(f"classes: {len(classnames)}")
        print(f"template_set: {args.template_set}")
        print(f"max_synonyms: {args.max_synonyms}")
        print(f"text_batch_size: {args.text_batch_size}")
        print(f"cache_classifier: {args.cache_classifier or '[none]'}")

    # reset cuda mem stats
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    W = build_zeroshot_classifier(
        base_model=base_model,
        tokenize=tokenize,
        classnames=classnames,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype,
        cfg=zs_cfg,
        cache_path=args.cache_classifier,
    )
    t_build = time.time() - t0

    # eval loop
    base_model.eval()
    W = W.to(device, non_blocking=True)

    top1 = torch.tensor(0.0, device=device)
    top5 = torch.tensor(0.0, device=device)
    n = torch.tensor(0.0, device=device)

    # timing
    eval_start = time.time()
    # warmup a bit for stable throughput
    warmup_iters = 5
    seen_iters = 0
    compute_time = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        torch.cuda.synchronize() if (device.type == "cuda") else None
        t1 = time.time()
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            z = forward_encode_image(base_model, images)  # [B,D] normalized
            logits = z @ W  # cosine sim
        torch.cuda.synchronize() if (device.type == "cuda") else None
        t2 = time.time()

        if seen_iters >= warmup_iters:
            compute_time += (t2 - t1)
        seen_iters += 1

        bsz = labels.numel()
        n += float(bsz)
        top1 += topk_correct(logits, labels, k=1)
        top5 += topk_correct(logits, labels, k=5)

    # reduce across ranks
    top1 = reduce_sum(top1)
    top5 = reduce_sum(top5)
    n = reduce_sum(n)

    eval_total = time.time() - eval_start
    top1_acc = (top1 / n).item() if n.item() > 0 else 0.0
    top5_acc = (top5 / n).item() if n.item() > 0 else 0.0

    # throughput / latency (exclude warmup)
    # compute_time is only the forward time sum after warmup iters
    # number of samples counted for compute_time:
    if seen_iters > warmup_iters:
        # approximate samples after warmup: total samples * (iters_after / total_iters)
        # better: just compute from batch sizes; keep simple + robust:
        # use overall throughput too.
        pass

    # overall throughput (including data)
    imgs_per_sec_total = (n.item() / max(eval_total, 1e-9))
    # forward-only throughput (rough)
    imgs_per_sec_fwd = None
    avg_ms_per_img_fwd = None
    if compute_time > 0 and n.item() > 0:
        # very rough: compute_time excludes warmup but n includes all;
        # still gives a decent ballpark if dataset is large.
        imgs_per_sec_fwd = (n.item() / max(compute_time, 1e-9))
        avg_ms_per_img_fwd = 1000.0 / max(imgs_per_sec_fwd, 1e-9)

    # peak memory
    peak_mem_mb = None
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    # FLOPs to GFLOPs
    gflops_img = (flops_img / 1e9) if flops_img and flops_img > 0 else None

    if is_main_process():
        print("==== Result ====")
        print(f"num_samples: {int(n.item())}")
        print(f"top1: {top1_acc:.4f}")
        print(f"top5: {top5_acc:.4f}")
        print(f"classifier_build_s: {t_build:.2f}s")
        print(f"eval_total_s: {eval_total:.2f}s")
        print(f"throughput_total_img_s: {imgs_per_sec_total:.2f}")
        print(f"flops_backend: {flops_backend}")
        if imgs_per_sec_fwd is not None:
            print(f"throughput_fwd_img_s (rough): {imgs_per_sec_fwd:.2f}")
            print(f"latency_fwd_ms_per_img (rough): {avg_ms_per_img_fwd:.3f} ms")
        if peak_mem_mb is not None:
            print(f"peak_mem_allocated_mb: {peak_mem_mb:.1f}")
        if gflops_img is not None:
            print(f"flops_per_image: {flops_img:,}  ({gflops_img:.3f} GFLOPs)")
        print("================")

        # save json
        out = {
            "model": args.model,
            "clip_name": args.clip_name if args.model == "clip" else "",
            "tinyclip_ckpt": args.tinyclip_ckpt if args.model == "tinyclip" else "",
            "gene_dir": args.gene_dir if args.model == "ours" else "",
            "shallow_layers": args.shallow_layers if args.model == "ours" else -1,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "amp": bool(args.amp),
            "amp_dtype": args.amp_dtype,
            "template_set": args.template_set,
            "max_synonyms": args.max_synonyms,
            "num_samples": int(n.item()),
            "top1": float(top1_acc),
            "top5": float(top5_acc),
            "params_total": int(total_p),
            "params_trainable": int(trainable_p),
            "model_size_mb": float(size_mb),
            "flops_per_image": int(flops_img) if flops_img and flops_img > 0 else -1,
            "gflops_per_image": float(gflops_img) if gflops_img is not None else None,
            "peak_mem_allocated_mb": float(peak_mem_mb) if peak_mem_mb is not None else None,
            "classifier_build_s": float(t_build),
            "eval_total_s": float(eval_total),
            "throughput_total_img_s": float(imgs_per_sec_total),
            "throughput_fwd_img_s_rough": float(imgs_per_sec_fwd) if imgs_per_sec_fwd is not None else None,
            "latency_fwd_ms_per_img_rough": float(avg_ms_per_img_fwd) if avg_ms_per_img_fwd is not None else None,
            "class_index_json": class_index_path,
            "flops_backend": flops_backend,
        }
        with open(os.path.join(args.out_dir, "imagenet1k_zeroshot_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[SAVE] {os.path.join(args.out_dir, 'imagenet1k_zeroshot_metrics.json')}")


if __name__ == "__main__":
    main()
