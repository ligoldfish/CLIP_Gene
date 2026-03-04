# scripts/data/coco_captions.py
import os
import json
from typing import List, Tuple, Dict, Any
from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionsPairs(Dataset):
    """
    captions_train2017.json:
      images: [{id, file_name, ...}]
      annotations: [{image_id, caption, ...}]
    flatten -> (img_path, caption)
    """
    def __init__(self, images_root: str, captions_json: str, transform=None, max_samples: int = -1):
        self.images_root = images_root
        self.transform = transform

        with open(captions_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in data["images"]}

        pairs: List[Tuple[str, str]] = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cap = ann["caption"]
            fn = id2file.get(img_id, None)
            if fn is None:
                continue
            pairs.append((os.path.join(images_root, fn), cap))

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
        return {"image": img, "text": cap, "source": "coco"}
