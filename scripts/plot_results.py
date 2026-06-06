"""结果可视化：成本曲线（命门图）、消融柱状、模态间隔。

用法：
  # 命门图：gene vs random 收敛
  python -m scripts.plot_results cost --csvs results/cost_K6_gene.csv results/cost_K6_random.csv --out results/fig_cost.png
  # 消融柱状：多个 eval.json 的 CIFAR top1 / COCO R@1
  python -m scripts.plot_results bars --evals results/student_K6_gene.json results/student_K6_random.json --labels gene random --out results/fig_bars.png
  # 模态间隔
  python -m scripts.plot_results gap --evals results/teacher.json results/student_K6_gene.json --labels teacher gene --out results/fig_gap.png
"""
import os
import sys
import csv
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] saved {out}")


def plot_cost(csvs, out, x="step", target=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path in csvs:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        init = rows[0]["init"]
        xs = [float(r[x] if x != "step" else r["step"]) for r in rows]
        ys = [float(r["metric"]) for r in rows]
        ax.plot(xs, ys, marker="o", ms=3, label=init)
    if target:
        ax.axhline(target, ls="--", c="gray", label=f"target={target}")
    ax.set_xlabel(x); ax.set_ylabel("CIFAR-100 sub top1 (%)")
    ax.set_title("Convergence: gene warm-start vs random")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out)


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def plot_bars(evals, labels, out):
    metrics = {"cifar100_top1": [], "coco_i2t_R@1": [], "coco_t2i_R@1": []}
    for path in evals:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        ds = r.get("datasets", {})
        metrics["cifar100_top1"].append(_get(ds, "cifar100", "top1", default=0))
        metrics["coco_i2t_R@1"].append(_get(ds, "coco_ret", "i2t_R@1", default=0))
        metrics["coco_t2i_R@1"].append(_get(ds, "coco_ret", "t2i_R@1", default=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    import numpy as np
    x = np.arange(len(labels)); w = 0.25
    for i, (k, v) in enumerate(metrics.items()):
        ax.bar(x + (i - 1) * w, v, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("metric (%)"); ax.set_title("Ablation: zero-shot / retrieval")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def plot_gap(evals, labels, out):
    gaps = []
    for path in evals:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        gaps.append(_get(r, "datasets", "modality_gap", "gap_l2", default=0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, gaps, color="steelblue")
    for i, g in enumerate(gaps):
        ax.text(i, g, f"{g:.3f}", ha="center", va="bottom")
    ax.set_ylabel("modality gap ‖Δcentroid‖₂"); ax.set_title("Modality gap (Mind-the-Gap)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost"); c.add_argument("--csvs", nargs="+", required=True)
    c.add_argument("--x", default="step"); c.add_argument("--target", type=float, default=30.0)
    c.add_argument("--out", default="results/fig_cost.png")
    b = sub.add_parser("bars"); b.add_argument("--evals", nargs="+", required=True)
    b.add_argument("--labels", nargs="+", required=True); b.add_argument("--out", default="results/fig_bars.png")
    g = sub.add_parser("gap"); g.add_argument("--evals", nargs="+", required=True)
    g.add_argument("--labels", nargs="+", required=True); g.add_argument("--out", default="results/fig_gap.png")
    args = ap.parse_args()

    if args.cmd == "cost":
        plot_cost(args.csvs, args.out, x=args.x, target=args.target)
    elif args.cmd == "bars":
        plot_bars(args.evals, args.labels, args.out)
    elif args.cmd == "gap":
        plot_gap(args.evals, args.labels, args.out)


if __name__ == "__main__":
    main()
