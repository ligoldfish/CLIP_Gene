# scripts/data/karpathy_pairs.py
import os
import json
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from torch.utils.data import Dataset


class KarpathyPairs(Dataset):
    """
    Karpathy json format (COCO/Flickr retrieval):
      {"images":[{"filename":..., "filepath":..., "split":"train", "sentences":[{"raw":...}, ...]}, ...]}
    Output flattened (img_path, caption) over specified split.
    """
    def __init__(self, images_root: str, karpathy_json: str, split: str = "train", transform=None, max_samples: int = -1):
        self.images_root = images_root
        self.transform = transform
        self.split = split

        with open(karpathy_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "images" in data, "Karpathy json must contain 'images'"

        pairs: List[Tuple[str, str]] = []
        for item in data["images"]:
            if item.get("split") != split:
                continue

            filename = item.get("filename") or item.get("file_name")
            filepath = item.get("filepath", "") or ""
            if filename is None:
                continue

            # Try path join
            p1 = os.path.join(images_root, filename)
            p2 = os.path.join(images_root, filepath, filename) if filepath else p1
            img_path = p1 if os.path.exists(p1) else p2

            sents = item.get("sentences", [])
            for s in sents:
                cap = s.get("raw") if isinstance(s, dict) else str(s)
                if cap:
                    pairs.append((img_path, cap))

        if max_samples > 0:
            pairs = pairs[:max_samples]

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, cap = self.pairs[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "text": cap, "source": "karpathy"}
