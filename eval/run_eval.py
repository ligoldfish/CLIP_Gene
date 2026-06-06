"""评估入口 CLI。

用法（cwd = learngene_clip）：
  python -m eval.run_eval --ckpt teacher                       # 评 teacher（sanity）
  python -m eval.run_eval --ckpt saved_models/student_K6_gene.pth
  python -m eval.run_eval --ckpt teacher --datasets cifar100,coco_ret,modality_gap
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import clip

from configs.base_config import config
from utils.teacher_loader import load_teacher_offline
from eval.model_interface import EvalModel
from eval.registry import run_dataset, LOCAL_DEFAULT


def _load_model(ckpt_arg, device):
    teacher, preprocess = load_teacher_offline(config, device)
    if ckpt_arg == "teacher":
        return teacher, preprocess
    from utils.build_student import build_student_from_gene
    # 安全说明：该 ckpt 由本项目 train_student.py 自存（含 state_dict + meta dict），可信来源，
    # 故用 weights_only=False 以读回 meta；勿用于加载外部不可信权重。
    ckpt = torch.load(ckpt_arg, map_location=device, weights_only=False)
    K = ckpt.get("K", config.BUDGET_K)
    student, _, preprocess, _ = build_student_from_gene(
        config, K=K, device=device, teacher=teacher, preprocess=preprocess)
    student.load_state_dict(ckpt["state_dict"])
    return student, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="teacher", help="'teacher' 或 student .pth 路径")
    ap.add_argument("--datasets", default=",".join(LOCAL_DEFAULT))
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    device = config.DEVICE
    model, preprocess = _load_model(args.ckpt, device)
    em = EvalModel(model, clip.tokenize, device, logit_scale_max=config.LOGIT_SCALE_MAX)

    results = {"ckpt": args.ckpt, "datasets": {}}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"[eval] running {name} ...")
        res = run_dataset(name, em, preprocess, config)
        results["datasets"][name] = res
        print(f"[eval] {name}: {res}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eval] 结果写入 {args.out}")


if __name__ == "__main__":
    main()
