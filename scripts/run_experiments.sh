#!/usr/bin/env bash
# CLIPgene 实验矩阵（cwd = learngene_clip）。NPU/GPU 同脚本。
# 全量矩阵很大；下面是代表性子集 + 注释里的完整网格。按需取消注释。
set -e
mkdir -p results saved_models

# ---- 0. 离线 teacher 验收 + sanity ----
python -c "from configs.base_config import config; from utils.teacher_loader import verify_conversion; print(verify_conversion(config, device=config.DEVICE))"
python -m eval.run_eval --ckpt teacher --datasets cifar100,coco_ret,modality_gap --out results/teacher.json

# ---- 1. 选层（判据消融）----
python -m scripts.extract_genes --criterion cmbi --score cos                  # 主
cp saved_models/gene_spec.json saved_models/gene_spec_cmbi.json
# python -m scripts.extract_genes --criterion cmbi --score infonce            # 消融
# python -m scripts.extract_genes --criterion grad_rho                        # baseline；cp 到 gene_spec_gradrho.json

# ---- 2. 训练（预算-K 族 + init 消融 + 成本曲线）----
for K in 4 6 8; do
  python -m scripts.train_student --K $K --init gene   --eval_every 200 --cost_target 30
done
python -m scripts.train_student --K 6 --init random   --eval_every 200 --cost_target 30   # 对照
# 散点 vs 连续（改 config.CONTIGUOUS_SPAN=True 重跑，或加 --no_lsq 测无缝合）

# ---- 3. 评估每个学生 ----
for K in 4 6 8; do
  python -m eval.run_eval --ckpt saved_models/student_K${K}_gene.pth \
    --datasets cifar100,coco_ret,modality_gap --out results/student_K${K}_gene.json
done
python -m eval.run_eval --ckpt saved_models/student_K6_random.pth \
  --datasets cifar100,coco_ret,modality_gap --out results/student_K6_random.json

# ---- 4. 可视化 ----
python -m scripts.plot_results cost --csvs results/cost_K6_gene.csv results/cost_K6_random.csv --out results/fig_cost.png
python -m scripts.plot_results bars --evals results/student_K4_gene.json results/student_K6_gene.json results/student_K8_gene.json results/student_K6_random.json --labels K4 K6 K8 K6-rand --out results/fig_bars.png
python -m scripts.plot_results gap  --evals results/teacher.json results/student_K6_gene.json results/student_K6_random.json --labels teacher gene random --out results/fig_gap.png
python -m scripts.analyze_cka --ckpt saved_models/student_K6_gene.pth --pairs 500

echo "实验完成；图见 results/"
