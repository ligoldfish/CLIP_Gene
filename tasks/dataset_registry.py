from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict


PLACEHOLDER_ROOT = "<DATASET_ROOT>"
DEFAULT_DATA_ROOT = os.environ.get("CLIPGENE_DATA_ROOT", "/root/autodl-tmp/clipgene_data")


def get_data_root(root: str = "") -> str:
    """Return the shared local dataset root used by download and training scripts."""
    return root or DEFAULT_DATA_ROOT


@dataclass(frozen=True)
class PretrainDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    cc3m_root: str = f"{DEFAULT_DATA_ROOT}/cc3m/wds"
    coco_train_images: str = f"{DEFAULT_DATA_ROOT}/coco/train2017"
    coco_train_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_train2017.json"


@dataclass(frozen=True)
class FindGeneDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    coco_images: str = f"{DEFAULT_DATA_ROOT}/coco/train2017"
    coco_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_train2017.json"


@dataclass(frozen=True)
class ITMDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    coco_train_images: str = f"{DEFAULT_DATA_ROOT}/coco/train2017"
    coco_train_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_train2017.json"
    coco_val_images: str = f"{DEFAULT_DATA_ROOT}/coco/val2017"
    coco_val_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_val2017.json"
    coco_test_images: str = f"{DEFAULT_DATA_ROOT}/coco/val2017"
    coco_test_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_val2017.json"
    flickr_images: str = f"{DEFAULT_DATA_ROOT}/flickr30k/images"
    flickr_karpathy_json: str = f"{DEFAULT_DATA_ROOT}/flickr30k/annotations/dataset_flickr30k.json"


@dataclass(frozen=True)
class RetrievalDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    coco_images: str = f"{DEFAULT_DATA_ROOT}/coco/train2017"
    coco_captions_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/captions_train2017.json"
    coco_karpathy_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/dataset_coco.json"
    flickr_images: str = f"{DEFAULT_DATA_ROOT}/flickr30k/images"
    flickr_karpathy_json: str = f"{DEFAULT_DATA_ROOT}/flickr30k/annotations/dataset_flickr30k.json"


@dataclass(frozen=True)
class MultiLabelDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    coco_train_images: str = f"{DEFAULT_DATA_ROOT}/coco/train2017"
    coco_train_instances_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/instances_train2017.json"
    coco_val_images: str = f"{DEFAULT_DATA_ROOT}/coco/val2017"
    coco_val_instances_json: str = f"{DEFAULT_DATA_ROOT}/coco/annotations/instances_val2017.json"


@dataclass(frozen=True)
class ImageNetDatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    imagenet_val_dir: str = f"{DEFAULT_DATA_ROOT}/imagenet/val"
    imagenet_val_imagefolder_dir: str = f"{DEFAULT_DATA_ROOT}/imagenet/val_imagefolder"
    imagenet_val_labels: str = f"{DEFAULT_DATA_ROOT}/imagenet/ImageNet_val_label.txt"
    class_index_json: str = f"{DEFAULT_DATA_ROOT}/imagenet/imagenet_class_index.json"
    classnames_txt: str = f"{DEFAULT_DATA_ROOT}/imagenet/imagenet_classnames.txt"


@dataclass(frozen=True)
class CIFAR100DatasetPaths:
    data_root: str = DEFAULT_DATA_ROOT
    cifar100_root: str = f"{DEFAULT_DATA_ROOT}/cifar100"


def get_task_dataset_placeholders() -> Dict[str, Dict[str, str]]:
    return {
        "pretrain": asdict(PretrainDatasetPaths()),
        "findgene": asdict(FindGeneDatasetPaths()),
        "itm": asdict(ITMDatasetPaths()),
        "retrieval": asdict(RetrievalDatasetPaths()),
        "multilabel": asdict(MultiLabelDatasetPaths()),
        "imagenet": asdict(ImageNetDatasetPaths()),
        "cifar100": asdict(CIFAR100DatasetPaths()),
    }


def is_placeholder_path(path: str) -> bool:
    return isinstance(path, str) and PLACEHOLDER_ROOT in path
