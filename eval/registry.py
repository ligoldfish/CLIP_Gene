"""数据集注册表：配置驱动，缺数据/报错优雅跳过。本地保底 + ELEVATER/Flickr/ImageNet 扩展。"""
from eval.zero_shot import evaluate_cifar100
from eval.retrieval import evaluate_coco5k, evaluate_flickr30k
from eval.modality_gap import modality_gap
from eval.elevater import evaluate_torchvision, ELEVATER_BUILDERS, _imagenet


def _run_cifar100(model, preprocess, cfg):
    return evaluate_cifar100(model, preprocess, cfg.CIFAR100_PICKLE_DIR, num_workers=cfg.num_workers)


def _run_coco_ret(model, preprocess, cfg):
    return evaluate_coco5k(model, preprocess, cfg.COCO_VAL_IMG_DIR, cfg.COCO_VAL_ANN)


def _run_flickr30k(model, preprocess, cfg):
    return evaluate_flickr30k(model, preprocess, cfg.FLICKR30K_IMG_DIR, cfg.FLICKR30K_ANN)


def _run_modality_gap(model, preprocess, cfg):
    return modality_gap(model, preprocess, cfg.COCO_VAL_IMG_DIR, cfg.COCO_VAL_ANN)


def _make_elevater_run(name):
    def run(model, preprocess, cfg):
        builder = ELEVATER_BUILDERS[name](cfg.ELEVATER_ROOT)
        return evaluate_torchvision(model, preprocess, builder, name, num_workers=cfg.num_workers)
    return run


def _run_imagenet(model, preprocess, cfg):
    builder = _imagenet(cfg.IMAGENET_ROOT)
    return evaluate_torchvision(model, preprocess, builder, "imagenet1k", num_workers=cfg.num_workers)


# name -> {task, local, run}
DATASET_REGISTRY = {
    "cifar100":     {"task": "zeroshot",   "local": True,  "run": _run_cifar100},
    "coco_ret":     {"task": "retrieval",  "local": True,  "run": _run_coco_ret},
    "modality_gap": {"task": "diagnostic", "local": True,  "run": _run_modality_gap},
    "flickr30k":    {"task": "retrieval",  "local": False, "run": _run_flickr30k},
    "imagenet1k":   {"task": "zeroshot",   "local": False, "run": _run_imagenet},
    "cifar10":      {"task": "zeroshot",   "local": False, "run": _make_elevater_run("cifar10")},
    "dtd":          {"task": "zeroshot",   "local": False, "run": _make_elevater_run("dtd")},
    "eurosat":      {"task": "zeroshot",   "local": False, "run": _make_elevater_run("eurosat")},
    "oxford_pets":  {"task": "zeroshot",   "local": False, "run": _make_elevater_run("oxford_pets")},
    "food101":      {"task": "zeroshot",   "local": False, "run": _make_elevater_run("food101")},
}

LOCAL_DEFAULT = ["cifar100", "coco_ret", "modality_gap"]
ALL_DATASETS = list(DATASET_REGISTRY.keys())


def run_dataset(name, model, preprocess, cfg):
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        return {"dataset": name, "skipped": "未知数据集"}
    if spec["run"] is None:
        return {"dataset": name, "skipped": spec.get("hint", "未实现/需下载")}
    try:
        res = spec["run"](model, preprocess, cfg)
    except Exception as e:  # noqa: BLE001  数据缺失/未下载 -> 优雅跳过
        return {"dataset": name, "skipped": f"{type(e).__name__}: {e}"}
    if res is None:
        return {"dataset": name, "skipped": "数据缺失"}
    return res
