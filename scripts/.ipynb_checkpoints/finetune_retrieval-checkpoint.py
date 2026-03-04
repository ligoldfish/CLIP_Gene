# scripts/finetune_retrieval.py
# Finetune retrieval (contrastive) on COCO captions + Flickr30k (Karpathy json)
#
# Patched:
#  - Add --eval_only mode (no training required; just run retrieval evaluation)
#  - Add --amp_dtype (auto/fp16/bf16) and use GradScaler only for fp16
#  - Run params/FLOPs profiling in eval_only too (same forward path)
#  - Fix best_by_dataset scope (track best across epochs, not reset each epoch)
#
# NOTE: This patch does NOT change the model architecture or training math when training.
#       It only adds a proper evaluation-only entry point + fixes logging/profiling + best tracking.
# torchrun --nproc_per_node=1 -m scripts.finetune_retrieval \
#   --model ours \
#   --gene_dir /root/gene_exports/last2 \
#   --shallow_layers 3 \
#   --init_ckpt outputs/pretrain_ours_last2/ckpt_last.pt \
#   --coco_images /root/autodl-tmp/train2017 \
#   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
#   --flickr_images /root/autodl-tmp/flickr30k/images \
#   --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
#   --batch_size 256 --epochs 20 \
#   --amp --amp_dtype fp16 \
#   --freeze_gene \
#   --use_tleg --tleg_target_depth 4 --tleg_strict\
#   --out_dir outputs/ft_retrieval_ours_last2
# torchrun --nproc_per_node=4 -m scripts.finetune_retrieval   --model clip  --distributed --coco_images /root/autodl-tmp/train2017   --coco_captions /root/autodl-tmp/annotations/captions_train2017.json   --flickr_images /root/autodl-tmp/flickr30k/images   --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json   --batch_size 256 --epochs 10   --amp --amp_dtype fp16   --out_dir outputs/ft_retrieval_clip

# torchrun --nproc_per_node=4 -m scripts.finetune_retrieval \
#   --model tinyclip --distributed --tinyclip_ckpt /root/autodl-tmp/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt\
#   --eval_only \
#   --eval_flickr_images /root/autodl-tmp/flickr30k/images \
#   --eval_flickr_karpathy_json /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
#   --eval_splits val test \
#   --eval_batch_size 256 --eval_num_workers 8 \
#   --amp --amp_dtype bf16 \
#   --out_dir outputs/eval_retrieval_clip

import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

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
from scripts.utils.distributed import setup_distributed, get_rank, get_world_size
from scripts.utils.misc import seed_everything, mkdir, is_main_process, unwrap_model
from scripts.utils.checkpoint import save_checkpoint, load_checkpoint
# NOTE: tasks/retrieval.py in this repo exposes a *feature-level* API:
#   - build_retrieval_datasets(img_root, karpathy_json, split)
#   - encode_all_images / encode_all_texts (adapter -> features)
#   - compute_retrieval_metrics(img_feats, txt_feats, img_to_caption_ids, caption_to_img)
from tasks.retrieval import (
    build_retrieval_datasets,
    encode_all_images,
    encode_all_texts,
    compute_retrieval_metrics,
)
from scripts.finetune_common import FinetunedModelAdapter
from scripts.utils.profile import count_params, profile_clip_like


# --- TF32 (Ampere/Ada/Hopper GPUs) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def collate_fn(batch: List[Dict[str, Any]], tokenize):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    tokens = tokenize(texts)  # CPU tokens
    return images, tokens


def get_logit_scale(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "logit_scale"):
        try:
            return model.logit_scale.exp().clamp(max=100.0)
        except Exception:
            pass
    return torch.tensor(1.0 / 0.07, device=next(model.parameters()).device)


def forward_features(base_model: nn.Module, images: torch.Tensor, tokens: torch.Tensor):
    """
    Feature extraction for:
      - OpenAI CLIP / OpenCLIP (encode_image / encode_text)
      - TinyCLIP (wkcn) built by clip.build_model (same API)
      - StudentCLIP (ours): vision_stem/text_stem + towers
    """
    if hasattr(base_model, "encode_image") and hasattr(base_model, "encode_text"):
        return base_model.encode_image(images), base_model.encode_text(tokens)

    if all(hasattr(base_model, k) for k in ["vision_stem", "text_stem", "vision_tower", "text_tower"]):
        v_tokens = base_model.vision_stem(images)
        t_tokens = base_model.text_stem(tokens)
        z_img = base_model.vision_tower(v_tokens)
        z_txt = base_model.text_tower(t_tokens, text=tokens)
        return z_img, z_txt

    raise RuntimeError(
        "Model does not support feature extraction (need encode_image/encode_text or StudentCLIP stems/towers)."
    )


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Robustly extract a model state_dict from various checkpoint formats."""
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
    """Returns optimizer and gene_params list (maybe empty)."""
    if args.model == "ours":
        new_params, gene_params = split_param_groups(base_model, args.gene_keywords)

        if len(gene_params) == 0:
            if is_main_process():
                print("[WARN] gene_params empty; falling back to single param group.")
            param_groups = [{"params": [p for p in base_model.parameters() if p.requires_grad],
                             "lr": args.lr, "weight_decay": args.wd}]
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


def _resolve_amp_dtype(args) -> Tuple[torch.dtype, Optional[torch.cuda.amp.GradScaler]]:
    if not args.amp:
        return torch.float16, None

    if args.amp_dtype == "auto":
        args.amp_dtype = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"

    if args.amp_dtype == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        if is_main_process():
            print("[WARN] bf16 not supported; fallback to fp16")
        args.amp_dtype = "fp16"

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.amp_dtype == "fp16"))
    if not scaler.is_enabled():
        scaler = None
    return amp_dtype, scaler


def train_one_epoch(
    args,
    model,
    loader,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    total_steps: int,
    unfreeze_step: int,
    amp_dtype: torch.dtype,
):
    model.train()
    device = args.device
    base_model = unwrap_model(model)

    loss_sum = 0.0
    loss_count = 0

    # staged unfreeze for ours
    if args.model == "ours":
        if args.freeze_gene:
            set_requires_grad(args._gene_params, False)
        else:
            if epoch < args.unfreeze_epoch:
                set_requires_grad(args._gene_params, False)
            else:
                set_requires_grad(args._gene_params, True)

    for it, (images, tokens) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

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

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            img_f, txt_f = forward_features(base_model, images, tokens)
            logit_scale = get_logit_scale(base_model)
            loss = clip_contrastive_loss(img_f, txt_f, logit_scale)

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

        if is_main_process() and (global_step % args.log_every == 0):
            if lr_gene is None:
                lr_info = f"lr={optimizer.param_groups[0]['lr']:.2e}"
            else:
                lr_info = f"lr_new={optimizer.param_groups[0]['lr']:.2e} lr_gene={optimizer.param_groups[1]['lr']:.2e}"
            print(f"[E{epoch:03d}][{it:05d}] step={global_step} loss={loss.item():.4f} {lr_info}")

    avg_loss = loss_sum / max(1, loss_count)
    return global_step, avg_loss


def _mean_recall_from_retrieval_metrics(metrics: dict) -> float:
    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]
    return sum(vals) / float(len(vals))


@torch.no_grad()
def run_retrieval_eval(args, model, tokenize) -> Dict[str, Any]:
    """Run retrieval evaluation on specified datasets/splits. Returns a flat dict of metrics."""
    eval_specs = []
    if args.eval_coco_images and args.eval_coco_karpathy_json:
        eval_specs.append(("coco", args.eval_coco_images, args.eval_coco_karpathy_json))
    if args.eval_flickr_images and args.eval_flickr_karpathy_json:
        eval_specs.append(("flickr30k", args.eval_flickr_images, args.eval_flickr_karpathy_json))

    if len(eval_specs) == 0:
        if is_main_process():
            print("[EVAL] No eval specs provided (need --eval_*). Skip.")
        return {}

    amp_dtype = getattr(args, "_amp_torch_dtype", torch.float16)

    adapter = FinetunedModelAdapter(
        model=unwrap_model(model),
        tokenize=tokenize,
        image_size=args.image_size,
        device=args.device,
        amp=args.amp,
        amp_dtype=amp_dtype,
    )

    unwrap_model(model).eval()

    out: Dict[str, Any] = {}
    for dname, img_root, kjson in eval_specs:
        for split in args.eval_splits:
            # Build retrieval eval data from Karpathy JSON.
            ds_img, captions, img_to_caption_ids, caption_to_img = build_retrieval_datasets(
                img_root, kjson, split=split
            )

            # Optional speed hack (not recommended for reporting). If you use it, metrics are NOT comparable.
            if args.eval_max_texts is not None and int(args.eval_max_texts) > 0:
                max_t = int(args.eval_max_texts)
                captions = captions[:max_t]
                caption_to_img = caption_to_img[:max_t]
                img_to_caption_ids = [[cid for cid in cids if cid < max_t] for cids in img_to_caption_ids]

            # Encode all images/texts -> features.
            img_feats = encode_all_images(
                adapter,
                ds_img,
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
            )
            txt_feats = encode_all_texts(
                adapter,
                captions,
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
            )

            # Compute retrieval metrics from features.
            m = compute_retrieval_metrics(img_feats, txt_feats, img_to_caption_ids, caption_to_img)
            mr = _mean_recall_from_retrieval_metrics(m)
            if is_main_process():
                print(
                    f"[EVAL][{dname}][{split}] "
                    f"i2t R@1/5/10={m['i2t_R@1']:.1f}/{m['i2t_R@5']:.1f}/{m['i2t_R@10']:.1f}  "
                    f"t2i R@1/5/10={m['t2i_R@1']:.1f}/{m['t2i_R@5']:.1f}/{m['t2i_R@10']:.1f}  "
                    f"mR={mr:.2f}"
                )

            prefix = f"{dname}_{split}_"
            out[prefix + "mR"] = float(mr)
            for k, v in m.items():
                out[prefix + k] = float(v)
    return out


def main():
    parser = argparse.ArgumentParser("Finetune Retrieval (contrastive) on COCO+Flickr (ours/clip/tinyclip)")

    # model
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=224)

    # eval-only
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run retrieval evaluation (requires --eval_* args). No training.")

    # ours knobs
    parser.add_argument("--gene_dir", type=str, default="", help="folder containing learngene_visual.pt / learngene_text.pt ...")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--proj_head", type=str, default="linear", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    # TLEG
    parser.add_argument("--use_tleg", action="store_true", default=True)
    parser.add_argument("--no_tleg", action="store_true", help="Disable TLEG even if enabled by default.")
    parser.add_argument("--tleg_target_depth", type=int, default=6)
    parser.add_argument("--tleg_strict", action="store_true", help="paper-faithful strict TLEG (only 2 endpoints trainable)")
    parser.add_argument("--use_multimodal_init", action="store_true")

    # CLIP stem init (ours)
    parser.add_argument("--disable_stem_init_from_clip", action="store_true")
    parser.add_argument("--stem_init_clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--freeze_stem_after_init", action="store_true")

    # baseline clip/tinyclip
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--tinyclip_ckpt", type=str, default="", help="TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt")

    # init checkpoint (optional)
    parser.add_argument("--init_ckpt", type=str, default="", help="optional init checkpoint (ckpt_last.pt or model_last.pt)")

    # data (training)
    parser.add_argument("--coco_images", type=str, default="")
    parser.add_argument("--coco_captions", type=str, default="")
    parser.add_argument("--flickr_images", type=str, default="")
    parser.add_argument("--flickr_ann", type=str, default="")
    parser.add_argument("--flickr_split", type=str, default="train")

    parser.add_argument("--coco_max", type=int, default=-1)
    parser.add_argument("--flickr_max", type=int, default=-1)
    parser.add_argument("--mix_probs", type=float, nargs=2, default=[0.7, 0.3])
    parser.add_argument("--mix_length", type=int, default=-1)

    # eval (retrieval)
    parser.add_argument("--eval_coco_images", type=str, default="")
    parser.add_argument("--eval_coco_karpathy_json", type=str, default="")
    parser.add_argument("--eval_flickr_images", type=str, default="")
    parser.add_argument("--eval_flickr_karpathy_json", type=str, default="")
    parser.add_argument("--eval_splits", type=str, nargs="+", default=["val", "test"])
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--eval_max_texts", type=int, default=-1)

    # profile
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--profile_speed", action="store_true")
    parser.add_argument("--profile_iters", type=int, default=50)

    # train
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.1)

    # AMP
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # freeze/unfreeze gene (ours)
    parser.add_argument("--freeze_gene", action="store_true", help="freeze learngene for entire finetune (recommended)")
    parser.add_argument("--unfreeze_epoch", type=int, default=2)
    parser.add_argument("--gene_lr_ratio", type=float, default=10.0)
    parser.add_argument("--gene_warmup_steps", type=int, default=200)
    parser.add_argument("--gene_keywords", type=str, nargs="+", default=["learngene", "gene"])

    # dist
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/finetune_retrieval")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_full_ckpt", action="store_true", help="also save optimizer/scaler for resume")

    args = parser.parse_args()

    args.stem_init_from_clip = (not getattr(args, 'disable_stem_init_from_clip', False))
    if getattr(args, 'no_tleg', False):
        args.use_tleg = False

    amp_dtype, scaler = _resolve_amp_dtype(args)
    # pass resolved torch.dtype to evaluation adapter (avoid string dtype bugs)
    args._amp_torch_dtype = amp_dtype

    seed_everything(args.seed)
    mkdir(args.out_dir)

    if args.distributed:
        setup_distributed(args.backend)

    if args.model == "ours" and not args.gene_dir:
        raise ValueError("--gene_dir is required for --model ours")

    if args.eval_only:
        if not ((args.eval_coco_images and args.eval_coco_karpathy_json) or (args.eval_flickr_images and args.eval_flickr_karpathy_json)):
            raise ValueError("--eval_only requires at least one eval spec: "
                             "(--eval_coco_images & --eval_coco_karpathy_json) or "
                             "(--eval_flickr_images & --eval_flickr_karpathy_json)")
        if args.model == "tinyclip" and (not args.init_ckpt) and (not args.tinyclip_ckpt):
            raise ValueError("--model tinyclip requires --tinyclip_ckpt or --init_ckpt for --eval_only")
    else:
        if not (args.coco_images and args.coco_captions and args.flickr_images and args.flickr_ann):
            raise ValueError("Training requires --coco_images --coco_captions --flickr_images --flickr_ann")

    bundle = create_model_bundle(args)
    model = bundle.model
    tokenize = bundle.tokenize

    if args.distributed and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    base_model = unwrap_model(model)
    base_model.float()

    if args.init_ckpt:
        load_init_ckpt_if_any(base_model, args.init_ckpt)

    # profile (also in eval_only)
    if is_main_process() and not args.skip_profile:
        try:
            dummy_tokens = tokenize(["a"])
            context_len = int(dummy_tokens.shape[-1])

            images = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
            toks = torch.zeros(1, context_len, dtype=torch.long, device=args.device)

            def _fw():
                with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
                    forward_features(base_model, images, toks)

            p = count_params(base_model)
            prof = profile_clip_like(base_model, _fw, iters=args.profile_iters, profile_speed=args.profile_speed)
            flops = prof.get("flops_total", None)
            tag = "[PROFILE][EVAL]" if args.eval_only else "[PROFILE]"
            print(f"{tag} params_total={p['total']} trainable={p['trainable']} flops={flops}")
            if prof.get("latency_ms") is not None:
                print(f"{tag} latency_ms={prof['latency_ms']:.3f} peak_mem_mb={prof['peak_mem_mb']:.1f}")
        except Exception as e:
            print(f"[PROFILE] failed: {type(e).__name__}: {e}")

    if args.eval_only:
        metrics = run_retrieval_eval(args, model, tokenize)
        if is_main_process():
            out_path = os.path.join(args.out_dir, "eval_metrics.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": args.model,
                    "clip_name": args.clip_name,
                    "tinyclip_ckpt": args.tinyclip_ckpt,
                    "init_ckpt": args.init_ckpt,
                    **metrics
                }, f, indent=2, ensure_ascii=False)
            print(f"[SAVE] {out_path}")
        return

    # datasets
    transform = build_clip_image_transform(args.image_size, is_train=True)
    coco_ds = CocoCaptionsPairs(args.coco_images, args.coco_captions, transform=transform, max_samples=args.coco_max)
    flickr_ds = KarpathyPairs(args.flickr_images, args.flickr_ann, split=args.flickr_split,
                              transform=transform, max_samples=args.flickr_max)
    train_ds = MixedDataset([coco_ds, flickr_ds], probs=args.mix_probs, length=args.mix_length)

    if args.distributed:
        sampler = DistributedSampler(
            train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True
        )
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

    optimizer, gene_params = build_optimizer(args, base_model)
    args._gene_params = gene_params

    total_steps = args.epochs * len(loader)
    unfreeze_step = args.unfreeze_epoch * len(loader)
    global_step = 0

    if is_main_process():
        print("==== Finetune Retrieval Config ====")
        for k, v in sorted(vars(args).items()):
            if k.startswith("_"):
                continue
            print(f"{k}: {v}")
        print("==============================")
        print(f"[INFO] total_steps={total_steps} unfreeze_step={unfreeze_step} steps/epoch={len(loader)}")

    has_eval = bool((args.eval_coco_images and args.eval_coco_karpathy_json) or (args.eval_flickr_images and args.eval_flickr_karpathy_json))
    best_by_dataset: Dict[str, float] = {}

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        global_step, avg_loss = train_one_epoch(
            args, model, loader, optimizer, scaler, epoch, global_step, total_steps, unfreeze_step, amp_dtype=amp_dtype
        )

        if is_main_process():
            print(f"[E{epoch:03d}] avg_loss={avg_loss:.4f}")

        if is_main_process() and has_eval and args.eval_every > 0:
            do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
            if do_eval:
                metrics = run_retrieval_eval(args, model, tokenize)

                # save best per dataset based on val mR
                for key, val in metrics.items():
                    if not key.endswith("_mR"):
                        continue
                    parts = key.split("_")
                    if len(parts) < 3:
                        continue
                    dname = parts[0]
                    split = parts[1]
                    if split != "val":
                        continue
                    mr = float(val)
                    if mr > float(best_by_dataset.get(dname, -1.0)):
                        best_by_dataset[dname] = mr
                        bpath = os.path.join(args.out_dir, f"model_best_{dname}.pt")
                        save_checkpoint(bpath, {"model": unwrap_model(model).state_dict(),
                                                "epoch": epoch, "global_step": global_step})
                        print(f"[SAVE] {bpath} (best {dname} val mR={mr:.2f})")

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


if __name__ == "__main__":
    main()
