# CLIPgene — Cross-modal Learngene 管线

冻结 teacher 上 CM-BI 选层 → 视觉文本联合块级基因 → 缝合层修残差流 →
同源蒸馏轻训 → 一基因预算-K 实例化可变规模族。**完全离线**（本地 HF 权重）。

## 依赖（AutoDL 上 pip 安装，不下载模型权重）
```bash
pip install torch torchvision transformers safetensors ftfy regex matplotlib
pip install git+https://github.com/openai/CLIP.git    # 提供 clip.model.build_model + clip.tokenize（离线安全）
# Ascend NPU 额外：pip install torch_npu（匹配 CANN 版本）；utils/device.py 自动探测
```

## 设备后端（CUDA / Ascend-NPU / CPU）
`utils/device.py` 自动探测：装了 torch_npu 且 `torch.npu.is_available()` → 走 npu，
AMP 用 `torch.npu.amp`；否则 cuda；否则 cpu。业务代码统一用 `device.autocast()/make_grad_scaler()`，
无需改。检查：`python -c "from utils.device import backend; print(backend())"`。

## 数据集
| 数据集 | 角色 | 本地? | 获取 |
|---|---|---|---|
| HF CLIP ViT-B/32 | teacher | ✅ clip-vit-base-patch32/ | — |
| COCO2017 | 训练/检索/标定 | ✅ coco2017/（train 92k 子集 + val 5k + captions） | cocodataset.org |
| CIFAR-100 | 零样本 headline | ✅ CIFAR-100/cifar-100-python | cs.toronto.edu/~kriz |
| Flickr30k | 检索（可选） | ❌ | kaggle + Karpathy split；填 FLICKR30K_* |
| ImageNet-1k | 零样本（可选） | ❌ | image-net.org（无免费镜像，用 CIFAR-100 代理） |
| ELEVATER 小集 | 零样本扩展 | ❌ | `python -m scripts.download_data --elevater`（torchvision） |
| CC3M | 规模(S6) | ❌ | Google Conceptual Captions TSV，需自爬；填 CC3M_ROOT |

`python -m scripts.download_data --list` 打印全部来源。

## 必需可视化输出
| 图 | 脚本 | 论文作用 |
|---|---|---|
| 收敛成本曲线（命门图） | `train_student --eval_every` 产 CSV → `plot_results cost` → results/fig_cost.png | 证 gene 暖启动比 random 快收敛 |
| 消融柱状（ZS/检索） | `plot_results bars` → results/fig_bars.png | K×init×判据 对比 |
| 模态间隔 | `plot_results gap` → results/fig_gap.png | 证保住对齐几何(Mind-the-Gap) |
| CKA 热力图 | `analyze_cka` → results/cka_{vision,text}.png | 证基因+缝合保 teacher 表示 |

## 数据/权重路径（configs/base_config.py）
- `LOCAL_CLIP_DIR` → 本地 HF CLIP 目录（含 model.safetensors）
- `COCO_ROOT` → coco2017（train2017 + annotations_trainval2017）
- `CIFAR100_PICKLE_DIR` → CIFAR-100/cifar-100-python
AutoDL 上按实际上传位置覆盖这些字段。

## 流程（cwd = learngene_clip）
```bash
# 0) 验收离线 teacher（图像 embedding 余弦须 > 0.999）
python -c "from configs.base_config import config; from utils.teacher_loader import verify_conversion; print(verify_conversion(config, device=config.DEVICE))"

# 0b) teacher 零样本 sanity（CIFAR-100 top-1 应 ≈60-65%）
python -m eval.run_eval --ckpt teacher --datasets cifar100

# 1) CM-BI 选层 → saved_models/gene_spec.json
python -m scripts.extract_genes                       # CM-BI(cos)
python -m scripts.extract_genes --criterion grad_rho  # 消融 baseline

# 2) 训练学生（预算-K：4/6/8 = S/M/L）
python -m scripts.train_student --K 6 --init gene
python -m scripts.train_student --K 6 --init random   # 消融对照

# 3) 评估
python -m eval.run_eval --ckpt saved_models/student_K6_gene.pth \
    --datasets cifar100,coco_ret,modality_gap --out results/student_K6.json
```

## 主要模块
| 模块 | 作用 |
|---|---|
| `utils/teacher_loader.py` | 离线 HF→openai teacher + `verify_conversion` |
| `utils/teacher_forward.py` | 恒等跳块前向（CM-BI/蒸馏共用） |
| `models/cmbi.py` | CM-BI 选层（+ grad-rho baseline）+ gene-spec I/O |
| `models/blocks.py` | robust GeneBlockWrapper / ResidualAdapter / **StitchLayer** / shallow-CNN |
| `models/student_clip.py` | 非连续基因 + 缝合 + clamp 的统一学生 |
| `models/stitch_init.py` | 缝合层 LSQ 初始化 |
| `utils/build_student.py` | 预算-K 构建 + 权重一致性断言 |
| `utils/losses_distill.py` | InfoNCE + 特征MSE + affinity-KL（同源） |
| `scripts/train_student.py` | 冻基因 + 分组LR + LSQ + 蒸馏 |
| `eval/` | 零样本/检索/模态间隔/成本曲线 + run_eval CLI |

旧 Gen-A/Gen-B 代码已归档至 `legacy/`（见 legacy/README.md）。

## 实验矩阵（S5）
判据 {grad_rho, CM-BI} × 连续/散点+stitch × init {gene, random} → CIFAR-100 ZS / COCO R@K /
epochs-to-target（cost_logger）/ modality_gap。CC3M（S6）需 provision `CC3M_ROOT`。
```
