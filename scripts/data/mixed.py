# scripts/data/mixed.py
import random
from typing import Dict, Any
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    """
    Mix datasets by sampling probabilities, ignore idx.
    """
    def __init__(self, datasets, probs=None, length: int = -1):
        assert len(datasets) >= 1
        self.datasets = datasets
        if probs is None:
            probs = [1.0 / len(datasets)] * len(datasets)
        s = sum(probs)
        self.probs = [p / s for p in probs]
        self.length = length if length > 0 else sum(len(d) for d in datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds = random.choices(self.datasets, weights=self.probs, k=1)[0]
        j = random.randint(0, len(ds) - 1)
        return ds[j]
