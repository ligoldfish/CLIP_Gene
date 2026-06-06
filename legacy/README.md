# legacy/ — 归档代码（不参与新管线）

新管线（Cross-modal Learngene）请见上层：
- 选层：`scripts/extract_genes.py` + `models/cmbi.py`
- 学生：`models/student_clip.py` + `utils/build_student.py`（含缝合层）
- 训练：`scripts/train_student.py` + `utils/losses_distill.py`（同源蒸馏）
- 离线 teacher：`utils/teacher_loader.py`
- 评估：`eval/`

## 归档内容与原因

**Gen-A（CIFAR100 分类分支，已废弃，多 bug）**
- `models/collective_model.py`, `models/individual_model.py`, `models/vision_model.py`
- `scripts/train_collective.py`（`scaler.scale(loss).backward` 漏括号→开放世界阶段无反传）,
  `scripts/train_vision.py`, `scripts/train_text.py`, `scripts/train_individual.py`
- `utils/gradient_utils.py`（deprecated register_backward_hook）, `utils/losses.py`（compute_fisher 坏）

**旧 Gen-B（被 CM-BI 管线取代）**
- `models/geneclip.py`（attn_mask kwarg 对 stock openai-clip 损坏）, `models/cnn_geneclip.py`, `models/test.py`
- `utils/build_geneclip.py`, `utils/build_cnn_geneclip.py`（致命 bug：读全量 model_ckpt → 零压缩）
- `scripts/train_clip.py`（微调-then-梯度ρ 选层；CM-BI 在冻结 teacher 上取代）,
  `scripts/train_geneclip.py`, `scripts/train_cnn_geneclip.py`, `scripts/save_vision-3.py`, `scripts/test.py`

仅作参考与对照实验留存，勿在新管线中 import。
