# CLIP_Gene

本仓库围绕 **CLIP 的 “Learngene” 层提取** 与一个轻量学生模型 **StudentCLIP（ours）** 展开：

* 从 OpenAI CLIP（如 ViT-B/32）中**选择并导出若干 Transformer Block**，作为 `learngene_visual.pt / learngene_text.pt`（可选导出 `learngene_multimodal.pt`）。
* 用这些 gene 权重构建 StudentCLIP（浅层 backbone + gene + 可选 TLEG 插值扩展），并提供在 **CC3M 预训练、COCO/Flickr30k 检索微调、COCO ITM 微调、ImageNet 线性探测/微调、ImageNet zero-shot** 等任务上的脚本。

> 代码入口主要在：`findgene/`（提取/导出）与 `scripts/`（训练/评测）；根目录有一个 `test.py` 用于快速加载与跑通前向。

---

## 目录结构

```
codes/
  test.py                       # 快速加载 StudentCLIP 并跑一张图+一句文本相似度

  findgene/
    get_gene.py                 # 直接按规则导出 CLIP 的最后2层/最后3层/last2+mid
    find_gene.py                # Adapter Expansion + ρ(t) 动态选层（COCO2017）
    find_gene_fisher.py         # Fisher-like (E[grad^2]) 选层（COCO2017）

  models/
    student_clip.py             # StudentCLIP（vision/text stem + towers + gene + tleg）
    learngene_loader.py         # 读取 learngene_*.pt 并装载到 StudentCLIP
    tleg.py                     # TLEG：对 gene 层权重做线性插值扩展到指定深度
    blocks.py                   # 基础模块

  scripts/
    pretrain_cc3m.py            # CC3M（WebDataset shards）对比学习预训练（可混 COCO）
    pretrain_coco_flickr.py     # COCO + Flickr30k 对比学习预训练
    finetune_retrieval.py       # COCO+Flickr 检索微调/评测（含 eval_only）
    finetune_itm.py             # COCO ITM 微调/评测（含 test split / eval_only）
    finetune_coco_multilabel.py # 实际是 ImageNet-1K 线性探测/微调分类（文件名有点历史遗留）
    imagenet_zs.py              # ImageNet-1K zero-shot 评测
    model_factory.py            # ours/clip/tinyclip 的统一构建入口
    losses.py / optim.py / ...  # 损失、优化器、分布式、profile 等工具

  tasks/
    retrieval.py / matching.py / zero_shot_imagenet.py / ...  # 任务数据与评测实现
```

---

## 环境依赖

建议：Python 3.8+，PyTorch 1.13+/2.x。

最小依赖：

```bash
pip install -U torch torchvision
pip install -U pillow numpy matplotlib
pip install -U webdataset
pip install -U git+https://github.com/openai/CLIP.git
pip install -U fvcore thop
```

---

## Learngene 导出

### A) `get_gene.py`：

**用途**：不做任何“选层算法”，直接把 CLIP 指定层拷贝出来保存为 gene 文件。
输出文件：

* `learngene_visual.pt`（包含：`{"layers": [...], "state_dict": ...}`）
* `learngene_text.pt`
* `learngene_multimodal.pt`（如有：`logit_scale / text_projection / visual.proj`）

示例：

```bash
# last2：导出最后 2 层
python findgene/get_gene.py --variant last2 --out_dir /root/gene_exports/last2 --device cuda

# last3：导出最后 3 层
python findgene/get_gene.py --variant last3 --out_dir /root/gene_exports/last3 --device cuda

# last2_plus6：导出 {第6层 + 最后2层}（mid 默认为 6，可改）
python findgene/get_gene.py --variant last2_plus6 --mid 6 --out_dir /root/gene_exports/last2_plus6 --device cuda
```

---

### B) `find_gene.py`：Adapter Expansion + ρ(t) 动态选层（COCO2017）

**核心思路**：

* 冻结原始 CLIP 权重
* 给每个 Transformer block 插入轻量 adapter，只训练 adapter（可选 logit_scale）
* 通过 adapter 梯度统计得到每层的 `ρ_i(t)`（梯度“显著比例”），用“先升后降”的动态趋势做选层

输出文件（在 `out_dir`）：

* `rho_logs.npz`（Rv/Rt、sigma、topk 等）
* `rho_heatmap_visual.png / rho_heatmap_text.png`
* `rho_trends_visual.png / rho_trends_text.png`
* `selected_layers.json`
* `learngene_visual.pt / learngene_text.pt / learngene_multimodal.pt`
* `adapters_state.pt`（可选分析用）

示例：

```bash
python findgene/find_gene.py \
  --coco_img_dir /root/autodl-tmp/train2017 \
  --coco_ann_file /root/autodl-tmp/annotations/captions_train2017.json \
  --out_dir ./outputs/lg_clip_adapter_rho \
  --clip_model ViT-B/32 --device cuda \
  --num_tasks 40 --steps_per_task 150 --batch_size 128 \
  --adapter_bottleneck 64 --lr 1e-3 \
  --sigma_warmup_steps 20 --sigma_quantile 0.90 --sigma_sample 200000 \
  --record_every 1 --topk 3 --prefer_last_ratio 0.5
```

---

### C) `find_gene_fisher.py`：Fisher-like（E[grad²]）选层（COCO2017）

**核心思路**：不依赖“动态趋势”，而是用局部敏感度：
[
\text{importance(layer)} = \mathbb{E}_{batch}[\text{mean}(grad^2)]
]
输出文件（在 `out_dir`）：

* `fisher_logs.npz`、热力图 png、`selected_layers.json`
* `learngene_visual.pt / learngene_text.pt / learngene_multimodal.pt`
* 可选的敏感度分析图（`--do_sensitivity`）

示例：

```bash
python findgene/find_gene_fisher.py \
  --coco_img_dir /root/autodl-tmp/train2017 \
  --coco_ann_file /root/autodl-tmp/annotations/captions_train2017.json \
  --out_dir ./outputs/lg_clip_coco_fisher \
  --clip_model ViT-B/32 --device cuda \
  --num_tasks 40 --batches_per_task 30 --batch_size 128 \
  --topk 3 --prefer_last_ratio 0.5 \
  --do_sensitivity 1 --sensitivity_batches 8 --noise_alpha 1e-3
```

---

## StudentCLIP（ours）快速跑通：`test.py`

`test.py` 做了最小化验证：

* 从 `gene_variant_dir` 读取 `learngene_visual.pt / learngene_text.pt`
* 构建 `StudentCLIPConfig`（`shallow_layers / bottleneck_dim / proj_dim / use_tleg / ...`）
* 对单张图 + 单句文本做 embedding，相似度输出

示例：

```bash
cd codes

python test.py \
  --gene_variant_dir /root/gene_exports/last2 \
  --shallow_layers 3 \
  --use_tleg \
  --tleg_target_depth 6 \
  --image_path /root/autodl-tmp/val2017/000000000139.jpg \
  --text "a dog with a frisbee"
```

常用参数说明：

* `--gene_variant_dir`：gene 文件夹（内含 `learngene_visual.pt / learngene_text.pt`）
* `--shallow_layers`：学生模型浅层 transformer 层数（默认 3）
* `--bottleneck_dim`：gene 前的 bottleneck（-1 表示关闭）
* `--use_tleg --tleg_target_depth`：对 gene 层做插值扩展到指定深度

---

## 训练与评测脚本（scripts/）

> 统一入口在 `scripts/model_factory.py`：
>
> * `--model ours`：StudentCLIP
> * `--model clip`：OpenAI CLIP baseline（`--clip_name ViT-B/32`）
> * `--model tinyclip`：wkcn/TinyCLIP checkpoint（`--tinyclip_ckpt ...pt`）

### 1) CC3M 预训练：`scripts/pretrain_cc3m.py`

特点：WebDataset 流式读取、warmup+cosine、grad clip、可混入 COCO caption 池。

示例（单卡）：

```bash
cd codes
MASTER_PORT=29501 torchrun --nproc_per_node=1 -m scripts.pretrain_cc3m \
  --distributed \
  --model ours --gene_dir /root/gene_exports/last3 \
  --cc3m_root /root/autodl-tmp/cc3m \
  --image_size 224 --batch_size 256 --epochs 3 \
  --amp --amp_dtype bf16 \
  --warmup_steps 2000 --lr 3e-4 --min_lr 1e-5 \
  --unfreeze_epoch 1 --gene_lr_ratio 10 --gene_warmup_steps 1000 \
  --use_tleg --tleg_target_depth 6 --tleg_last_epochs 1 \
  --out_dir outputs/pretrain_cc3m_ours
```

输出（`out_dir`）：

* `ckpt_last.pt`（包含 model/optimizer/scaler/args）
* 若 `--save_every_epoch`：`ckpt_epochXXX.pt`

---

### 2) 检索微调/评测：`scripts/finetune_retrieval.py`

支持：

* 训练：COCO captions + Flickr30k（Karpathy JSON）
* `--eval_only`：只跑评测（也会做 params/FLOPs profile，除非 `--skip_profile`）

示例（训练）：

```bash
torchrun --nproc_per_node=1 -m scripts.finetune_retrieval \
  --model ours \
  --gene_dir /root/gene_exports/last2 \
  --shallow_layers 3 \
  --init_ckpt outputs/pretrain_cc3m_ours/ckpt_last.pt \
  --coco_images /root/autodl-tmp/train2017 \
  --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
  --flickr_images /root/autodl-tmp/flickr30k/images \
  --flickr_ann /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
  --batch_size 256 --epochs 10 \
  --amp --amp_dtype bf16 \
  --freeze_gene \
  --use_tleg --tleg_target_depth 6 \
  --out_dir outputs/ft_retrieval_ours_last2
```

示例（只评测）：

```bash
torchrun --nproc_per_node=1 -m scripts.finetune_retrieval \
  --model ours \
  --gene_dir /root/gene_exports/last2 \
  --init_ckpt outputs/ft_retrieval_ours_last2/model_last.pt \
  --eval_only \
  --eval_flickr_images /root/autodl-tmp/flickr30k/images \
  --eval_flickr_karpathy_json /root/autodl-tmp/flickr30k/annotations/dataset_flickr30k.json \
  --eval_splits val test \
  --eval_batch_size 256 --eval_num_workers 8 \
  --amp --amp_dtype bf16 \
  --out_dir outputs/eval_retrieval_ours_last2
```

输出（`out_dir`）：

* `model_last.pt`
* 每数据集最优：`model_best_coco.pt / model_best_flickr.pt`（按 val mR）
* 若 `--save_every_epoch`：`model_epochXXX.pt`
* 若 `--save_full_ckpt`：额外 `ckpt_last.pt / ckpt_epochXXX.pt`

---

### 3) ITM 微调/评测：`scripts/finetune_itm.py`

支持：

* COCO ITM（正负采样）
* val/test 评测（可 `--test_when_best`）
* `--eval_only`：加载 checkpoint 直接评测

示例（训练 + val/test）：

```bash
torchrun --nproc_per_node=1 -m scripts.finetune_itm \
  --model ours --gene_dir /root/gene_exports/last2 \
  --init_ckpt outputs/pretrain_cc3m_ours/ckpt_last.pt \
  --coco_images /root/autodl-tmp/train2017 \
  --coco_captions /root/autodl-tmp/annotations/captions_train2017.json \
  --coco_val_images /root/autodl-tmp/val2017 \
  --coco_val_captions /root/autodl-tmp/annotations/captions_val2017.json \
  --coco_test_images /root/autodl-tmp/val2017 \
  --coco_test_captions /root/autodl-tmp/annotations/captions_val2017.json \
  --epochs 10 --batch_size 256 \
  --amp --amp_dtype bf16 \
  --val_every 1 --eval_test 1 --test_when_best \
  --select_metric auc \
  --freeze_gene \
  --use_tleg --tleg_target_depth 6 \
  --out_dir outputs/ft_itm_ours_last2
```

输出（`out_dir`）：

* `model_best.pt`（按 `--select_metric`）
* `best_metrics.json`（记录 best 的 val/test 指标）
* 若 `--save_every_epoch`：`model_epochXXX.pt`
* 结束会保存 `model_last.pt`（脚本中同风格）

---

### 4) ImageNet-1K 线性探测/微调：`scripts/finetune_coco_multilabel.py`

> **注意**：文件名是 `finetune_coco_multilabel.py`，但脚本实际内容是 **ImageNet-1K 分类**（线性 probe / finetune）。

示例（线性探测）：

```bash
torchrun --nproc_per_node=1 -m scripts.finetune_coco_multilabel \
  --model ours \
  --gene_dir /root/gene_exports/last3 \
  --init_ckpt outputs/pretrain_cc3m_ours/ckpt_last.pt \
  --imagenet_root /root/autodl-tmp/imagenet \
  --epochs 10 --batch_size 256 --val_batch_size 512 \
  --amp \
  --out_dir outputs/imagenet_lp_ours_last3
```

示例（微调 backbone）：

```bash
torchrun --nproc_per_node=1 -m scripts.finetune_coco_multilabel \
  --model ours \
  --gene_dir /root/gene_exports/last3 \
  --init_ckpt outputs/pretrain_cc3m_ours/ckpt_last.pt \
  --imagenet_root /root/autodl-tmp/imagenet \
  --train_backbone --lr 2e-4 --head_lr_ratio 10 \
  --epochs 30 --batch_size 128 --val_batch_size 256 \
  --amp \
  --out_dir outputs/imagenet_ft_ours_last3
```

---

### 5) ImageNet-1K Zero-shot：`scripts/imagenet_zs.py`

示例：

```bash
torchrun --nproc_per_node=1 -m scripts.imagenet_zs \
  --model ours \
  --gene_dir /root/gene_exports/last3 \
  --init_ckpt outputs/pretrain_cc3m_ours/ckpt_last.pt \
  --imagenet_val_dir /root/autodl-tmp/imagenet/val \
  --imagenet_val_labels /root/autodl-tmp/imagenet/ImageNet_val_label.txt \
  --class_index_json /root/autodl-tmp/imagenet/ImageNet_class_index.json \
  --image_size 224 --batch_size 128 --num_workers 16 \
  --amp --amp_dtype bf16 \
  --template_set clip --max_synonyms 3 --text_batch_size 256 \
  --cache_classifier outputs/cache/zs_W_ours_last3.pt \
  --out_dir outputs/eval_imagenet_zs_ours
```

---

## Checkpoint / Gene 文件格式约定

### Gene 文件（`learngene_visual.pt / learngene_text.pt`）

* `get_gene.py` 与 `find_gene*.py` 导出的 gene 都是 `torch.save(dict)`
* 常见结构：

  * `{"layers": [layer_ids...], "state_dict": {...}}`
  * multimodal：`{"state_dict": {"logit_scale": ..., "text_projection": ..., "visual.proj": ...}}`

### 训练 checkpoint

* `pretrain_cc3m.py`：`ckpt_last.pt`（含 optimizer/scaler/args）
* `finetune_retrieval.py / finetune_itm.py`：默认 `model_last.pt`（仅 model state_dict），可选 `--save_full_ckpt` 保存完整状态

---

## 致谢

* OpenAI CLIP（tokenize / model structure / baseline 权重加载）
* WebDataset（用于 CC3M 等 tar shard 数据流式训练）
