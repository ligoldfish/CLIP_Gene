"""ELEVATER 小集 + ImageNet 零样本（torchvision 数据集，类名取自 dataset.classes）。

数据缺失时调用方 try/except 跳过（见 registry.run_dataset）。
"""
from torch.utils.data import DataLoader

from eval.zero_shot import build_zeroshot_classifier, zero_shot_classification
from eval.prompts import ZEROSHOT_TEMPLATES


def _names(ds):
    cls = getattr(ds, "classes", None)
    assert cls is not None and len(cls) > 0, "dataset 无 .classes"
    out = []
    for c in cls:
        if isinstance(c, (list, tuple)):   # ImageNet: 同义词元组取首个
            c = c[0]
        out.append(str(c).replace("_", " ").replace("/", " ").strip().lower())
    return out


def evaluate_torchvision(model, preprocess, builder, name, templates=None,
                         batch_size=256, num_workers=4) -> dict:
    ds = builder(preprocess)
    names = _names(ds)
    classifier = build_zeroshot_classifier(model, names, templates or ZEROSHOT_TEMPLATES)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    res = zero_shot_classification(model, loader, classifier)
    res["dataset"] = name
    res["n_classes"] = len(names)
    return res


# ---- 各数据集 builder（download=False；数据须已下好）----
def _cifar10(root):
    import torchvision as tv
    return lambda pp: tv.datasets.CIFAR10(root, train=False, transform=pp, download=False)


def _dtd(root):
    import torchvision as tv
    return lambda pp: tv.datasets.DTD(root, split="test", transform=pp, download=False)


def _eurosat(root):
    import torchvision as tv
    return lambda pp: tv.datasets.EuroSAT(root, transform=pp, download=False)


def _oxford_pets(root):
    import torchvision as tv
    return lambda pp: tv.datasets.OxfordIIITPet(root, split="test", transform=pp, download=False)


def _food101(root):
    import torchvision as tv
    return lambda pp: tv.datasets.Food101(root, split="test", transform=pp, download=False)


def _imagenet(root):
    import torchvision as tv
    return lambda pp: tv.datasets.ImageNet(root, split="val", transform=pp)


ELEVATER_BUILDERS = {
    "cifar10": _cifar10,
    "dtd": _dtd,
    "eurosat": _eurosat,
    "oxford_pets": _oxford_pets,
    "food101": _food101,
}
