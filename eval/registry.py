"""数据集注册表：配置驱动，本地缺数据优雅跳过；ELEVATER 扩展挂载位。"""
from eval.zero_shot import evaluate_cifar100
from eval.retrieval import evaluate_coco5k, evaluate_flickr30k
from eval.modality_gap import modality_gap


def _run_cifar100(model, preprocess, cfg):
    return evaluate_cifar100(model, preprocess, cfg.CIFAR100_PICKLE_DIR,
                             num_workers=cfg.num_workers)


def _run_coco_ret(model, preprocess, cfg):
    return evaluate_coco5k(model, preprocess, cfg.COCO_VAL_IMG_DIR, cfg.COCO_VAL_ANN)


def _run_flickr30k(model, preprocess, cfg):
    return evaluate_flickr30k(model, preprocess, cfg.FLICKR30K_IMG_DIR, cfg.FLICKR30K_ANN)


def _run_modality_gap(model, preprocess, cfg):
    return modality_gap(model, preprocess, cfg.COCO_VAL_IMG_DIR, cfg.COCO_VAL_ANN)


# name -> {task, local, run}
DATASET_REGISTRY = {
    "cifar100":     {"task": "zeroshot",  "local": True,  "run": _run_cifar100},
    "coco_ret":     {"task": "retrieval", "local": True,  "run": _run_coco_ret},
    "modality_gap": {"task": "diagnostic", "local": True, "run": _run_modality_gap},
    "flickr30k":    {"task": "retrieval", "local": False, "run": _run_flickr30k},
    # --- ELEVATER 扩展（需下载；加 loader + cfg 路径后填 run）---
    "imagenet1k":   {"task": "zeroshot", "local": False, "run": None,
                     "hint": "ImageNet val ~6.7GB 未本地；用 CIFAR-100 作零样本代理"},
    "cifar10":      {"task": "zeroshot", "local": False, "run": None,
                     "hint": "torchvision 可下载"},
    "dtd":          {"task": "zeroshot", "local": False, "run": None, "hint": "ELEVATER 小数据集，可下载"},
    "eurosat":      {"task": "zeroshot", "local": False, "run": None, "hint": "ELEVATER 小数据集，可下载"},
    "oxford_pets":  {"task": "zeroshot", "local": False, "run": None, "hint": "ELEVATER 小数据集，可下载"},
    "food101":      {"task": "zeroshot", "local": False, "run": None, "hint": "ELEVATER 小数据集，可下载"},
}

LOCAL_DEFAULT = ["cifar100", "coco_ret", "modality_gap"]


def run_dataset(name, model, preprocess, cfg):
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        return {"dataset": name, "skipped": "未知数据集"}
    if spec["run"] is None:
        return {"dataset": name, "skipped": spec.get("hint", "未实现/需下载")}
    res = spec["run"](model, preprocess, cfg)
    if res is None:
        return {"dataset": name, "skipped": "数据缺失"}
    return res
