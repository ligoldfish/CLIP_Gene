"""基因抽取入口：冻结 teacher 上跑 CM-BI（或 grad-rho baseline）选层，存 gene_spec.json。

用法（cwd = learngene_clip）：
  python -m scripts.extract_genes                      # CM-BI, cos 分数
  python -m scripts.extract_genes --criterion grad_rho # baseline
  python -m scripts.extract_genes --score infonce --pairs 5000
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip  # noqa: E402  (仅用 tokenize，离线安全)

from configs.base_config import config  # noqa: E402
from utils.teacher_loader import load_teacher_offline  # noqa: E402
from utils.coco_data import make_calibration_subset  # noqa: E402
from models.cmbi import (  # noqa: E402
    compute_cmbi_ranking,
    compute_grad_rho_ranking,
    save_gene_spec,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", choices=["cmbi", "grad_rho"], default=config.SELECTION_CRITERION)
    ap.add_argument("--score", choices=["cos", "infonce", "acc"], default=config.CMBI_SCORE)
    ap.add_argument("--pairs", type=int, default=config.CALIB_PAIRS)
    ap.add_argument("--out", default=config.GENE_SPEC_PATH)
    args = ap.parse_args()

    device = config.DEVICE
    print(f"[extract] 离线加载 teacher（{config.LOCAL_CLIP_DIR}）...")
    teacher, preprocess = load_teacher_offline(config, device)

    print(f"[extract] 构建标定集：{args.pairs} 对 COCO（seed={config.CALIB_SEED}）...")
    calib = make_calibration_subset(
        config.train_img_dir, config.train_ann_file, preprocess, clip.tokenize,
        n_pairs=args.pairs, batch_size=config.CALIB_BATCH_SIZE, seed=config.CALIB_SEED,
        num_workers=config.num_workers,
    )

    if args.criterion == "cmbi":
        print(f"[extract] CM-BI 打分（score={args.score}）...")
        rank = compute_cmbi_ranking(teacher, calib, device, score=args.score)
    else:
        print("[extract] grad-rho baseline 打分...")
        rank = compute_grad_rho_ranking(teacher, calib, device, threshold=config.GRADIENT_THRESHOLD)

    spec = {
        "criterion": args.criterion,
        "score": args.score if args.criterion == "cmbi" else "grad_rho",
        "clip_model": config.CLIP_MODEL,
        "calib_pairs": args.pairs,
        "calib_seed": config.CALIB_SEED,
        **rank,
    }
    save_gene_spec(spec, args.out)
    print("[extract] vision ranking:", spec["vision"]["ranking"])
    print("[extract] text   ranking:", spec["text"]["ranking"])
    # 排名非平凡检查
    vs = list(spec["vision"]["scores"].values())
    assert max(vs) - min(vs) > 1e-9, "视觉分数全平 —— 选层信号无效，检查标定/跳块"


if __name__ == "__main__":
    main()
