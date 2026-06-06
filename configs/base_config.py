import torch
import os

class Config:
    # =========================================================================
    # 数据集路径
    # =========================================================================
    # CIFAR100 split (Gen-A legacy 用)
    BASE_CLASSES_DATA = "./CIFAR100_splited/CIFAR100_splited/base/train"
    OPEN_WORLD_DATA = "./CIFAR100_splited/CIFAR100_splited/novel/train"
    NOVEL_CLASSES_DATA = "./CIFAR100_splited/CIFAR100_splited/open_world/train"

    # CIFAR-100 (zero-shot 评估用 pickle)。本地树在 CLIPgene/CIFAR-100，
    # 脚本 cwd 通常是 learngene_clip，故默认 ../CIFAR-100。AutoDL 上按需覆盖。
    CIFAR_ROOT = "./CIFAR-100"
    CIFAR100_PICKLE_DIR = os.path.join("..", "CIFAR-100", "cifar-100-python")

    # COCO2017（本地实际目录；原 "autodl-tmp" 路径已修正）
    COCO_ROOT = "./coco2017"
    train_img_dir = os.path.join(COCO_ROOT, "train2017")
    train_ann_file = os.path.join(COCO_ROOT, "annotations_trainval2017", "captions_train2017.json")
    COCO_VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2017")
    COCO_VAL_ANN = os.path.join(COCO_ROOT, "annotations_trainval2017", "captions_val2017.json")

    # Flickr30k（非本地；缺则评估跳过）
    FLICKR30K_IMG_DIR = ""        # e.g. "./flickr30k/images"
    FLICKR30K_ANN = ""            # Karpathy flickr test split json

    # CC3M（非本地；规模阶段 provision 后填）
    CC3M_ROOT = ""

    # =========================================================================
    # 模型 / 离线权重
    # =========================================================================
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CLIP_MODEL = "ViT-B/32"
    # 离线 teacher：本地 HF safetensors 目录（含 config.json + model.safetensors）
    LOCAL_CLIP_DIR = "./clip-vit-base-patch32"

    SAVE_PATH = "./saved_models"
    SAVE_DIR = "./saved_models"
    SAVE_NAME = "gene_clip_model.pth"
    # CM-BI 选层产物（取代旧的 model_ckpt / extracted_layers_ckpt 选层依赖）
    GENE_SPEC_PATH = os.path.join(SAVE_PATH, "gene_spec.json")

    # ---- 以下为旧 Gen-B / Gen-A / grad-rho baseline 路径，保留以兼容 legacy ----
    MODEL_SAVE_PATH = "./saved_models/vision_model_layer.pth"
    WHOLE_MODEL_SAVE = "./saved_models/vision_model.pth"
    FEATURE_MODEL_SAVE_PATH = "./savedmodels/vision_feature_model.pth"
    model_ckpt = os.path.join(SAVE_PATH, "clip_cross_modal.pt")          # DEPRECATED: builder 不再读
    extracted_layers_ckpt = os.path.join(SAVE_PATH, "clip_topk_layers.pt")  # 仅 grad-rho baseline
    TEXT_MODEL_PATH = os.path.join(SAVE_PATH, "text_model.pt")
    TEXT_MODEL_LAYER_PATH = os.path.join(SAVE_PATH, "text_model_layer.pt")
    CROSS_MODEL_PT = "./cross_model.pt"

    # =========================================================================
    # CM-BI 基因抽取
    # =========================================================================
    SELECTION_CRITERION = "cmbi"   # "cmbi" | "grad_rho"
    CMBI_SCORE = "cos"             # "cos" | "infonce" | "acc"
    CALIB_PAIRS = 3000             # 标定集图文对数
    CALIB_BATCH_SIZE = 64
    CALIB_SEED = 42

    # =========================================================================
    # 学生：预算-K / 缝合 / 模块
    # =========================================================================
    BUDGET_K = 6                   # S/M/L = 4/6/8
    BUDGET_K_V = None              # 视觉侧单独覆盖（None=用 BUDGET_K）
    BUDGET_K_T = None              # 文本侧单独覆盖
    CONTIGUOUS_SPAN = False        # True = 连续段选层消融

    ADAPTER_BOTTLENECK = 64
    USE_SHALLOW_CNN = False        # gate VisionShallowCNN / TextShallowCNN
    SHALLOW_CNN_LAYERS_V = 2
    SHALLOW_CNN_LAYERS_T = 1
    SHALLOW_CNN_BOTTLENECK = 64

    USE_FULL_STITCH = True         # True => DxD 全缝合；False => 低秩
    STITCH_RANK = 32               # 低秩时的秩
    STITCH_AT_HEAD = False         # 末块后是否插缝合

    LOGIT_SCALE_MAX = 100.0        # clamp exp(logit_scale)

    # =========================================================================
    # 训练（学生）
    # =========================================================================
    INDIVIDUAL_BATCH_SIZE = 32
    INDIVIDUAL_EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    WARMUP_RATIO = 0.03
    LR_MAIN = 5e-4                 # adapter / 投影 学习率
    LR_STITCH = 1e-4               # 缝合层（LSQ 初始化后）低学习率
    TRAIN_POS_PROJ = False         # 是否训练 pos/proj（默认否，保持轻量）
    MAX_GRAD_NORM = 1.0
    LOG_INTERVAL = 50
    TEMPERATURE = 0.07
    num_workers = 8

    # 蒸馏损失权重（同源）
    DISTILL_ALPHA = 1.0            # InfoNCE
    DISTILL_BETA = 1.0             # 特征 MSE
    DISTILL_GAMMA = 1.0            # affinity-KL
    DISTILL_TAU = 4.0             # affinity-KL 温度

    # =========================================================================
    # DEPRECATED（新管线忽略；保留供 legacy import 不报错）
    # =========================================================================
    LR_GENE = 5e-5                 # DEPRECATED: 基因全程冻结
    UNFREEZE_AFTER_EPOCH = 6       # DEPRECATED
    UNFREEZE_LAST_V = 1            # DEPRECATED
    UNFREEZE_LAST_T = 1            # DEPRECATED
    PROX_LAMBDA = 1e-4             # DEPRECATED: 基因冻结后无需 prox

    # ---- Gen-A / baseline 旧字段 ----
    INITIAL_CLASSES = 60
    COLLECTIVE_BATCH_SIZE = 64
    COLLECTIVE_LR = 1e-6
    INDIVIDUAL_LR = 1e-3
    EWC_LAMBDA = 1e3
    OPEN_WORLD_THRESHOLD = 0.5
    EMBEDDING_DIM = 512
    ADOPTOR_DIM = 256
    NOVEL_CLASSES = 20
    MIN_EPOCHS_FOR_STABILITY = 10
    NUM_CLASSES = 100
    num_epochs = 30
    clip_epochs = 5
    GRADIENT_THRESHOLD = 1e-3      # grad-rho baseline
    GRADIENT_WINDOW_SIZE = 5
    NUM_TASKS = 16
    top_k = 5                      # grad-rho baseline 选层数

    # 设备后端：CUDA / Ascend-NPU / CPU 自动探测（见 utils/device.py）
    try:
        from utils.device import DEVICE as _DEV
        DEVICE = _DEV
    except Exception:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
