"""学生训练（替代 train_geneclip.py / train_cnn_geneclip.py）。

要点：基因全程冻结；可训 = adapter + 缝合层 + logit_scale(+ 可选 pos/proj)；
训练前 LSQ 初始化缝合；同源蒸馏（InfoNCE + 特征MSE + affinity-KL）。

用法（cwd = learngene_clip）：
  python -m scripts.train_student --K 6 --init gene
  python -m scripts.train_student --K 6 --init random   # 消融 baseline
"""
import os
import sys
import math
import time
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip  # tokenize（离线安全）

from configs.base_config import config
from utils.teacher_loader import load_teacher_offline
from utils.coco_data import COCOCaptionDataset, make_coco_collate, make_calibration_subset
from utils.build_student import build_student_from_gene
from utils.losses_distill import compute_distill_loss
from utils.teacher_forward import teacher_embeddings
from models.blocks import ResidualAdapter, StitchLayer
from models.stitch_init import lsq_init_stitches
from utils.device import autocast, make_grad_scaler, manual_seed_all


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    manual_seed_all(s)


def cosine_with_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def reinit_gene_blocks(student):
    """随机重置基因块权重（仍冻结），用于 random/scratch 消融 baseline。"""
    @torch.no_grad()
    def _reset(block):
        for p in block.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    for enc in [student.vision, student.text]:
        for w in enc.trunk.blocks:
            _reset(w.gene_block)


def _is_norm_name(n: str) -> bool:
    return n.startswith("ln_") or ".ln_" in n or "norm" in n.lower()


def unfreeze_gene_norms(student):
    """B: 解冻基因块内 LayerNorm（仅 ln_1/ln_2），其余基因权重仍冻结。"""
    cnt = 0
    for enc in [student.vision, student.text]:
        for w in enc.trunk.blocks:
            for n, p in w.gene_block.named_parameters():
                if _is_norm_name(n):
                    p.requires_grad_(True)
                    cnt += p.numel()
    print(f"[train] LN-tuning: 解冻基因 LayerNorm {cnt/1e3:.1f}K 参数")


def get_param_groups(student, cfg):
    adapters, stitches = [], []
    for m in student.modules():
        if isinstance(m, ResidualAdapter):
            adapters += [p for p in m.parameters() if p.requires_grad]
        elif isinstance(m, StitchLayer):
            stitches += [p for p in m.parameters() if p.requires_grad]
    # B: 基因块内被解冻的 LayerNorm 参数
    gene_norms = []
    for enc in [student.vision, student.text]:
        for w in enc.trunk.blocks:
            for n, p in w.gene_block.named_parameters():
                if p.requires_grad and _is_norm_name(n):
                    gene_norms.append(p)
    extra = []
    if cfg.TRAIN_POS_PROJ:
        for enc, attrs in [(student.vision, ["positional_embedding", "proj"]),
                           (student.text, ["positional_embedding", "text_projection"])]:
            for a in attrs:
                p = getattr(enc, a, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad_(True)
                    extra.append(p)
    groups = [
        {"params": adapters, "lr": cfg.LR_MAIN, "weight_decay": cfg.WEIGHT_DECAY, "_base": cfg.LR_MAIN},
        {"params": stitches, "lr": cfg.LR_STITCH, "weight_decay": cfg.WEIGHT_DECAY, "_base": cfg.LR_STITCH},
        {"params": gene_norms, "lr": cfg.LR_MAIN, "weight_decay": 0.0, "_base": cfg.LR_MAIN},
        {"params": [student.logit_scale], "lr": cfg.LR_MAIN, "weight_decay": 0.0, "_base": cfg.LR_MAIN},
        {"params": extra, "lr": cfg.LR_MAIN * 0.1, "weight_decay": cfg.WEIGHT_DECAY, "_base": cfg.LR_MAIN * 0.1},
    ]
    return [g for g in groups if len(g["params"]) > 0]


def assert_genes_frozen(student, allow_norms=False):
    for enc in [student.vision, student.text]:
        for w in enc.trunk.blocks:
            for n, p in w.gene_block.named_parameters():
                if allow_norms and _is_norm_name(n):
                    continue
                assert p.requires_grad is False, f"基因非norm参数 {n} 未冻结！"


def setup_cost_eval(student, cfg, args, device):
    """成本曲线：固定 CIFAR-100 子集 + 学生 zero-shot 分类器。返回 (logger, em, classifier, loader) 或 None。"""
    if args.eval_every <= 0:
        return None
    from torch.utils.data import Subset, DataLoader as _DL
    from eval.model_interface import EvalModel
    from eval.zero_shot import build_zeroshot_classifier
    from eval.datasets_local import CIFAR100Pickle
    from eval.prompts import cifar100_classnames, CIFAR100_TEMPLATES
    from eval.cost_logger import CostLogger
    from utils.teacher_loader import load_teacher_offline  # for preprocess only (already have)
    _, preprocess = load_teacher_offline(cfg, device)
    em = EvalModel(student, clip.tokenize, device, logit_scale_max=cfg.LOGIT_SCALE_MAX)
    names = cifar100_classnames(cfg.CIFAR100_PICKLE_DIR)
    ds = CIFAR100Pickle(cfg.CIFAR100_PICKLE_DIR, split="test", transform=preprocess)
    sub = Subset(ds, list(range(min(2000, len(ds)))))   # 子集，快
    loader = _DL(sub, batch_size=256, shuffle=False, num_workers=cfg.num_workers)
    csv = args.cost_csv or os.path.join("results", f"cost_K{args.K}_{args.init}.csv")
    logger = CostLogger(f"K{args.K}", args.init, "cifar100_top1", args.cost_target, csv)
    return logger, em, names, CIFAR100_TEMPLATES, loader


@torch.no_grad()
def cost_eval_top1(em, names, templates, loader):
    from eval.zero_shot import build_zeroshot_classifier
    clf = build_zeroshot_classifier(em, names, templates)
    top1 = n = 0
    for images, labels in loader:
        logits = em.logit_scale_exp * (em.encode_image(images) @ clf)
        labels = labels.to(logits.device)
        top1 += (logits.argmax(1) == labels).sum().item(); n += labels.size(0)
    return 100.0 * top1 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=config.BUDGET_K)
    ap.add_argument("--init", choices=["gene", "random", "scratch"], default="gene")
    ap.add_argument("--epochs", type=int, default=config.INDIVIDUAL_EPOCHS)
    ap.add_argument("--no_lsq", action="store_true", help="跳过缝合 LSQ 初始化")
    ap.add_argument("--eval_every", type=int, default=0,
                    help="每 N step 在 CIFAR-100 子集上测 top1 并记成本曲线（0=关）")
    ap.add_argument("--cost_target", type=float, default=30.0, help="成本曲线目标 top1(%)")
    ap.add_argument("--cost_csv", default="", help="成本曲线 CSV（默认 results/cost_K{K}_{init}.csv）")
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="混合精度：bf16(默认,NPU 推荐,无 loss scaler) / fp16(用 GradScaler) / fp32(关 AMP)")
    ap.add_argument("--contiguous", action="store_true",
                    help="A: 选连续段基因（覆盖 config.CONTIGUOUS_SPAN）")
    ap.add_argument("--train_gene_norms", action="store_true",
                    help="B: 解冻基因块内 LayerNorm（LN-tuning）")
    ap.add_argument("--adapter_bottleneck", type=int, default=config.ADAPTER_BOTTLENECK,
                    help="D: adapter 瓶颈维（默认 config 值；调大如 128 增容量）")
    args = ap.parse_args()

    cfg = config
    # CLI 覆盖配置（A/B/D），并存进 ckpt 以便评估重建对齐
    cfg.CONTIGUOUS_SPAN = bool(args.contiguous) or cfg.CONTIGUOUS_SPAN
    cfg.ADAPTER_BOTTLENECK = args.adapter_bottleneck
    cfg.TRAIN_GENE_NORMS = bool(args.train_gene_norms) or cfg.TRAIN_GENE_NORMS
    run_tag = args.init
    if cfg.CONTIGUOUS_SPAN:
        run_tag += "_contig"
    if cfg.TRAIN_GENE_NORMS:
        run_tag += "_ln"
    if cfg.ADAPTER_BOTTLENECK != 64:
        run_tag += f"_ab{cfg.ADAPTER_BOTTLENECK}"
    if not args.cost_csv:
        args.cost_csv = os.path.join("results", f"cost_K{args.K}_{run_tag}.csv")
    print(f"[train] tag={run_tag} contiguous={cfg.CONTIGUOUS_SPAN} "
          f"adapter_bottleneck={cfg.ADAPTER_BOTTLENECK} train_gene_norms={cfg.TRAIN_GENE_NORMS}")

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.CALIB_SEED)
    device = cfg.DEVICE

    teacher, preprocess = load_teacher_offline(cfg, device)
    student, meta, preprocess, teacher = build_student_from_gene(
        cfg, K=args.K, device=device, teacher=teacher, preprocess=preprocess)
    print("[train] gene indices V/T:", meta["vision_sel"], meta["text_sel"],
          "| gaps V/T:", meta["vision_gaps"], meta["text_gaps"])

    if args.init in ("random", "scratch"):
        reinit_gene_blocks(student)
        print(f"[train] init={args.init}: 基因块已随机重置（baseline）")

    if cfg.TRAIN_GENE_NORMS:
        unfreeze_gene_norms(student)
    assert_genes_frozen(student, allow_norms=cfg.TRAIN_GENE_NORMS)

    # LSQ 初始化缝合（用确定性标定子集）
    if not args.no_lsq and (meta["vision_gaps"] + meta["text_gaps"] > 0):
        calib = make_calibration_subset(
            cfg.train_img_dir, cfg.train_ann_file, preprocess, clip.tokenize,
            n_pairs=min(cfg.CALIB_PAIRS, 2000), batch_size=cfg.CALIB_BATCH_SIZE,
            seed=cfg.CALIB_SEED, num_workers=cfg.num_workers)
        lsq_init_stitches(student, teacher, calib, device)

    # 训练数据
    ds = COCOCaptionDataset(cfg.train_img_dir, cfg.train_ann_file)
    dl = DataLoader(ds, batch_size=cfg.INDIVIDUAL_BATCH_SIZE, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True,
                    collate_fn=make_coco_collate(preprocess, clip.tokenize), drop_last=True)

    groups = get_param_groups(student, cfg)
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"[train] 可训参数: {n_train/1e6:.3f} M（基因冻结）")

    total_steps = args.epochs * len(dl)
    warmup_steps = int(cfg.WARMUP_RATIO * total_steps)
    # AMP 策略：bf16/fp32 不需要 loss scaler（避免 fp16 溢出导致 scale→0 的死亡螺旋）
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp]
    use_scaler = (args.amp == "fp16")
    scaler = make_grad_scaler() if use_scaler else None
    print(f"[train] AMP={args.amp} (scaler={'on' if use_scaler else 'off'})")
    teacher_scale = teacher.logit_scale.exp().clamp(max=cfg.LOGIT_SCALE_MAX).detach()

    cost = setup_cost_eval(student, cfg, args, device)

    global_step = 0
    best_loss = float("inf")
    student.train()
    for epoch in range(1, args.epochs + 1):
        run_sum, run_n, t0 = 0.0, 0, time.time()
        for step, (imgs, toks) in enumerate(dl, 1):
            imgs = imgs.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            with torch.no_grad():
                zi_t, zt_t = teacher_embeddings(teacher, imgs, toks)
            with autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                zi_s = student.encode_image(imgs)
                zt_s = student.encode_text(toks)
            zi_s, zt_s = zi_s.float(), zt_s.float()
            scale_s = student.logit_scale.exp().clamp(max=cfg.LOGIT_SCALE_MAX)
            loss, parts = compute_distill_loss(zi_s, zt_s, zi_t, zt_t, scale_s, teacher_scale, cfg)

            if not torch.isfinite(loss):
                print(f"  [warn] step {global_step} loss 非有限({loss.item()})，跳过"); global_step += 1
                continue

            lr_scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
            for g in optim.param_groups:
                g["lr"] = g["_base"] * lr_scale

            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), cfg.MAX_GRAD_NORM)
                optim.step()

            run_sum += loss.item(); run_n += 1; global_step += 1
            if step % cfg.LOG_INTERVAL == 0:
                print(f"E{epoch}/{args.epochs} S{step}/{len(dl)} loss={run_sum/run_n:.4f} "
                      f"clip={parts['clip']:.3f} feat={parts['feat']:.3f} aff={parts['aff']:.3f} "
                      f"lr={optim.param_groups[0]['lr']:.2e}")
            if cost is not None and global_step % args.eval_every == 0:
                logger, em, names, templates, loader = cost
                student.eval()
                top1 = cost_eval_top1(em, names, templates, loader)
                logger.log(epoch, global_step, top1)
                student.train()
                print(f"  [cost] step {global_step} cifar100_sub_top1={top1:.2f}")

        epoch_loss = run_sum / max(1, run_n)
        print(f"[E{epoch}] mean_loss={epoch_loss:.4f} time={(time.time()-t0)/60:.1f}min")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(cfg.SAVE_DIR, f"student_K{args.K}_{run_tag}.pth")
            torch.save({"state_dict": student.state_dict(), "meta": meta,
                        "K": args.K, "init": args.init, "tag": run_tag,
                        "contiguous": cfg.CONTIGUOUS_SPAN,
                        "adapter_bottleneck": cfg.ADAPTER_BOTTLENECK,
                        "train_gene_norms": cfg.TRAIN_GENE_NORMS,
                        "epoch": epoch, "loss": epoch_loss}, save_path)
            print(f"[train] saved best -> {save_path} (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
