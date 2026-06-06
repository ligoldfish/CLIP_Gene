"""零样本提示模板 + CIFAR-100 类名。模板源自 OpenAI CLIP / ELEVATER。"""
import os
import pickle

# CIFAR-100 标准 7 模板子集（ELEVATER）
CIFAR100_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low contrast photo of a {}.",
    "a photo of the {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a good photo of a {}.",
]

# 通用零样本模板（OpenAI 子集）
ZEROSHOT_TEMPLATES = CIFAR100_TEMPLATES + [
    "a photo of many {}.",
    "a close-up photo of a {}.",
    "a photo of one {}.",
    "a pixelated photo of a {}.",
    "itap of a {}.",
    "a bad photo of the {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
]


def cifar100_classnames(pickle_dir: str):
    """从 CIFAR-100 meta pickle 读 fine_label_names，'_'→空格。

    安全说明：pickle 来源是官方 CIFAR-100 本地数据集（可信），非外部输入。
    """
    meta_path = os.path.join(pickle_dir, "meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")  # trusted: official CIFAR-100
    names = meta["fine_label_names"]
    return [n.replace("_", " ") for n in names]
