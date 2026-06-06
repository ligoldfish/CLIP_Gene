# 从零部署到 NPU 远端服务器

代码与数据分离：**代码走 git（小），数据走 scp/重下（大）**。

---
## A. 本地：把 learngene_clip 单独推成 git 仓库（仅代码）

`.gitignore` 已排除数据/权重/产物。在本地 `learngene_clip/` 下：
```bash
cd d:/Download/CLIPgene/learngene_clip
git init
git add .
git status            # 确认无 coco2017/ CIFAR-100/ *.safetensors 等大文件
git commit -m "CLIPgene cross-modal learngene pipeline"
# 在 GitHub 新建空仓库 clipgene（不要勾 README），然后：
git branch -M main
git remote add origin git@github.com:<你的账号>/clipgene.git
git push -u origin main
```
> 若不便用 GitHub，改用打包：`tar --exclude-from=.gitignore -czf clipgene_code.tgz .`，再 scp 到远端解压。

---
## B. 远端：连接 + NPU 基础环境

```bash
ssh <user>@<npu-host>
# 确认 NPU 与 CANN
npu-smi info                      # 看到昇腾卡即正常
echo $ASCEND_HOME_PATH            # CANN 安装路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 每个新 shell 都要 source（或写进 ~/.bashrc）

# conda 环境
conda create -n clipgene python=3.9 -y
conda activate clipgene
```

### 装 torch + torch_npu（版本必须与 CANN 匹配）
```bash
# 示例：CANN 8.0 对应 torch 2.1.0。具体版本查 https://gitee.com/ascend/pytorch 对照表
pip install torch==2.1.0            # CPU 版 torch（torch_npu 提供 NPU 后端）
pip install torch_npu==2.1.0.post* # 与 torch 主版本对齐
pip install torchvision==0.16.0
# 验证
python -c "import torch, torch_npu; print(torch.npu.is_available())"   # True
```

---
## C. 远端：拉代码 + 装依赖

```bash
git clone git@github.com:<你的账号>/clipgene.git
cd clipgene
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git    # 若远端无外网，见 D 的离线办法
python -c "from utils.device import backend; print(backend())"   # 应打印 npu
```
> 远端无外网装 openai clip：本地 `pip download git+https://github.com/openai/CLIP.git -d clip_pkg` 后 scp，再 `pip install --no-index --find-links clip_pkg clip`。或直接把本地 `site-packages/clip/` scp 过去。

---
## D. 远端：数据上传 / 重下

代码默认路径（configs/base_config.py，相对 learngene_clip 根；按远端实际改）：
- `LOCAL_CLIP_DIR=./clip-vit-base-patch32`
- `COCO_ROOT=./coco2017`（train2017 + val2017 + annotations_trainval2017）
- `CIFAR100_PICKLE_DIR=../CIFAR-100/cifar-100-python`

### 1) CLIP teacher 权重（~340MB，必需）
```bash
# 本地 scp（推荐，已有）：
scp -r d:/Download/CLIPgene/learngene_clip/clip-vit-base-patch32 <user>@<host>:~/clipgene/
# 或远端有外网时重下：
huggingface-cli download openai/clip-vit-base-patch32 --local-dir clip-vit-base-patch32
```

### 2) CIFAR-100（~170MB，零样本评估必需）
```bash
# 远端有外网：
mkdir -p ../CIFAR-100 && cd ../CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz && cd -
# 或本地 scp 现成的 CIFAR-100/cifar-100-python
```

### 3) COCO2017（大，训练/检索/标定必需）
```bash
# 远端有外网重下（比从家里上传快）：
mkdir -p coco2017 && cd coco2017
wget http://images.cocodataset.org/zips/train2017.zip      # ~18GB
wget http://images.cocodataset.org/zips/val2017.zip        # ~1GB
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip; unzip -q val2017.zip; unzip -q annotations_trainval2017.zip
# 注意：代码默认 annotations 目录名为 annotations_trainval2017/，官方解压成 annotations/
mv annotations annotations_trainval2017 2>/dev/null || true
cd -
# 或仅传本地已有子集：scp -r d:/Download/CLIPgene/learngene_clip/coco2017 ...
```
> 若只想先跑通，可只用 val2017 + COCO 子集做标定/快训；改小 `CALIB_PAIRS`、`INDIVIDUAL_EPOCHS`。

### 4)（可选）ELEVATER 小集
```bash
python -m scripts.download_data --elevater
```

---
## E. 远端：验收 + 跑实验

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 确保 CANN env
conda activate clipgene
cd ~/clipgene

# 0) 离线 teacher 验收（图像 embedding 余弦 > 0.999 才算转换正确）
python -c "from configs.base_config import config; from utils.teacher_loader import verify_conversion; print(verify_conversion(config, device=config.DEVICE))"

# 0b) teacher 零样本 sanity（CIFAR-100 ≈60-65%）
python -m eval.run_eval --ckpt teacher --datasets cifar100

# 1-4) 全流程
bash scripts/run_experiments.sh
```

---
## F. 常见坑
- `torch.npu.is_available()` False → 没 `source set_env.sh`，或 torch_npu 与 CANN 版本不匹配。
- `verify_conversion` 余弦 < 0.999 → HF 权重目录不对，或 transformers 版本差异导致 key 名变化（检查 `pre_layrnorm` typo）。
- COCO 路径报错 → 解压后 annotations 目录名与 `COCO_VAL_ANN` 不符，按 D-3 改名或改 config。
- OOM → 调小 `INDIVIDUAL_BATCH_SIZE` / `CALIB_BATCH_SIZE`；NPU 显存与 GPU 不同，先小批验证。
- 多卡：当前脚本单卡。需要分布式再加 `torch_npu` + `torchrun`（后续工作）。
