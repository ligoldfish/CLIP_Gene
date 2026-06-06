"""数据集获取助手。

本地已有（无需下载）：HF CLIP 权重、COCO2017、CIFAR-100。
可选扩展（需联网）：ELEVATER 小数据集经 torchvision 下载；其余给手动链接。

用法：
  python -m scripts.download_data --elevater         # 下 cifar10/dtd/eurosat/pets/food101
  python -m scripts.download_data --list             # 仅打印各数据集来源
"""
import argparse
import os

DATA_SOURCES = {
    "HF CLIP ViT-B/32": "已本地：clip-vit-base-patch32/（HF openai/clip-vit-base-patch32）",
    "COCO2017":        "已本地：coco2017/（官方 cocodataset.org，train/val2017 + captions）",
    "CIFAR-100":       "已本地：CIFAR-100/cifar-100-python（cs.toronto.edu/~kriz）",
    "Flickr30k":       "手动：images(kaggle: hsankesara/flickr-image-dataset) + Karpathy split dataset_flickr30k.json",
    "ImageNet-1k val": "手动：image-net.org（需注册）；无免费镜像，零样本可用 CIFAR-100 代理",
    "CC3M":            "手动：Google Conceptual Captions TSV（ai.google.com/research/ConceptualCaptions），需自爬图",
    "ELEVATER 小集":   "torchvision 下载：cifar10/dtd/eurosat/oxford-pets/food101（本脚本 --elevater）",
}


def download_elevater(root="./elevater_data"):
    os.makedirs(root, exist_ok=True)
    import torchvision
    jobs = [
        ("CIFAR10", lambda: torchvision.datasets.CIFAR10(root, train=False, download=True)),
        ("DTD", lambda: torchvision.datasets.DTD(root, split="test", download=True)),
        ("EuroSAT", lambda: torchvision.datasets.EuroSAT(root, download=True)),
        ("OxfordIIITPet", lambda: torchvision.datasets.OxfordIIITPet(root, split="test", download=True)),
        ("Food101", lambda: torchvision.datasets.Food101(root, split="test", download=True)),
    ]
    for name, fn in jobs:
        try:
            fn()
            print(f"[download] {name} OK -> {root}")
        except Exception as e:  # noqa: BLE001
            print(f"[download] {name} 失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--elevater", action="store_true")
    ap.add_argument("--root", default="./elevater_data")
    args = ap.parse_args()
    if args.list or not args.elevater:
        for k, v in DATA_SOURCES.items():
            print(f"- {k}: {v}")
    if args.elevater:
        download_elevater(args.root)


if __name__ == "__main__":
    main()
