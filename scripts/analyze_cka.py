"""CKA 分析：学生基因块隐状态 vs teacher 各层隐状态，出热力图。

证明"基因块 + 缝合"保住了 teacher 表示（学生第 k 基因块应与其来源 teacher 层最相似）。
线性 CKA（Kornblith 2019）。本地保底数据：COCO 标定子集。

用法：
  python -m scripts.analyze_cka --ckpt saved_models/student_K6_gene.pth --pairs 500
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import clip

from configs.base_config import config
from utils.teacher_loader import load_teacher_offline
from utils.coco_data import make_calibration_subset
from utils.build_student import build_student_from_gene


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """X:[N,Dx] Y:[N,Dy] -> 线性 CKA in [0,1]。"""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    nx = np.linalg.norm(X.T @ X, ord="fro")
    ny = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (nx * ny + 1e-12))


@torch.no_grad()
def collect(teacher, student, loader, device, branch="vision", max_batches=None):
    """返回 student_feats[list K of [N,D]] 与 teacher_feats[list 12 of [N,D]]（mean-pool token）。"""
    resblocks = teacher.visual.transformer.resblocks if branch == "vision" else teacher.transformer.resblocks
    t_out = {}
    handles = []
    for i, blk in enumerate(resblocks):
        def hook(m, inp, out, idx=i):
            t_out[idx] = out.detach()           # [L,B,D]
        handles.append(blk.register_forward_hook(hook))

    s_feats, t_feats = None, None
    nb = 0
    for imgs, toks in loader:
        imgs = imgs.to(device); toks = toks.to(device)
        t_out.clear()
        if branch == "vision":
            _, s_hidden = student.vision(imgs, return_hidden=True)
            teacher.encode_image(imgs)
        else:
            _, s_hidden = student.text(toks, return_hidden=True)
            teacher.encode_text(toks)
        # mean-pool over tokens -> [B,D]
        s_b = [h.mean(0).float().cpu().numpy() for h in s_hidden]
        t_b = [t_out[i].mean(0).float().cpu().numpy() for i in range(len(resblocks))]
        if s_feats is None:
            s_feats = [[] for _ in s_b]; t_feats = [[] for _ in t_b]
        for k, f in enumerate(s_b):
            s_feats[k].append(f)
        for k, f in enumerate(t_b):
            t_feats[k].append(f)
        nb += 1
        if max_batches and nb >= max_batches:
            break
    for h in handles:
        h.remove()
    s_feats = [np.concatenate(x, 0) for x in s_feats]
    t_feats = [np.concatenate(x, 0) for x in t_feats]
    return s_feats, t_feats


def heatmap(mat, xt, yt, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.set_xlabel("teacher layer"); ax.set_ylabel("student gene block")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[cka] saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="student .pth；空则用当前 gene_spec 直接 build")
    ap.add_argument("--K", type=int, default=config.BUDGET_K)
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--out_dir", default="results/cka")
    args = ap.parse_args()

    device = config.DEVICE
    teacher, preprocess = load_teacher_offline(config, device)
    ckpt = None
    K = args.K
    if args.ckpt:
        # 自存可信 ckpt；先读架构旋钮再 build，确保与权重对齐
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        K = ckpt.get("K", args.K)
        config.CONTIGUOUS_SPAN = ckpt.get("contiguous", config.CONTIGUOUS_SPAN)
        config.ADAPTER_BOTTLENECK = ckpt.get("adapter_bottleneck", config.ADAPTER_BOTTLENECK)
        config.FRONT_INIT = ckpt.get("front_init", config.FRONT_INIT)
        config.FRONT_BLOCKS_V = ckpt.get("front_blocks_v", config.FRONT_BLOCKS_V)
        config.FRONT_BLOCKS_T = ckpt.get("front_blocks_t", config.FRONT_BLOCKS_T)
        config.USE_LORA = ckpt.get("use_lora", config.USE_LORA)
        config.LORA_RANK = ckpt.get("lora_rank", config.LORA_RANK)
        config.LORA_TARGETS = ckpt.get("lora_targets", config.LORA_TARGETS)
    student, meta, preprocess, teacher = build_student_from_gene(
        config, K=K, device=device, teacher=teacher, preprocess=preprocess)
    if ckpt is not None:
        student.load_state_dict(ckpt["state_dict"])
    student.eval()

    loader = make_calibration_subset(config.train_img_dir, config.train_ann_file, preprocess,
                                     clip.tokenize, n_pairs=args.pairs,
                                     batch_size=config.CALIB_BATCH_SIZE, seed=config.CALIB_SEED,
                                     num_workers=config.num_workers)

    for branch, sel in [("vision", meta["vision_sel"]), ("text", meta["text_sel"])]:
        s_feats, t_feats = collect(teacher, student, loader, device, branch=branch)
        mat = np.zeros((len(s_feats), len(t_feats)))
        for i in range(len(s_feats)):
            for j in range(len(t_feats)):
                mat[i, j] = linear_cka(s_feats[i], t_feats[j])
        yt = [f"g{sel[i]}" for i in range(len(s_feats))]
        xt = [str(j) for j in range(len(t_feats))]
        heatmap(mat, xt, yt, f"CKA {branch} (student gene vs teacher layers)",
                os.path.join(args.out_dir, f"cka_{branch}.png"))


if __name__ == "__main__":
    main()
