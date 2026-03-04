# scripts/data/flickr30k.py
import os
import json
from typing import List, Tuple, Dict, Any
from PIL import Image
from torch.utils.data import Dataset


class Flickr30kPairs(Dataset):
    """
    支持常见的 json 标注格式（你如果格式不同，改这里最方便）
    期望结构之一：
      [{"image": "xxx.jpg", "caption": ["a ...", "b ...", ...]}, ...]
    或：
      {"images": [{"file_name":..., "sentences":[{"raw":...}, ...]}, ...]}
    """

    def __init__(self, images_root: str, ann_path: str, transform=None, max_samples: int = -1):
        self.images_root = images_root
        self.transform = transform

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs: List[Tuple[str, str]] = []

        if isinstance(data, list):
            # list of {image, caption/list}
            for item in data:
                img_fn = item.get("image") or item.get("file_name")
                caps = item.get("caption") or item.get("captions") or item.get("sentences")
                if img_fn is None or caps is None:
                    continue
                if isinstance(caps, str):
                    caps = [caps]
                for c in caps:
                    if isinstance(c, dict):
                        c = c.get("raw") or c.get("caption") or ""
                    pairs.append((os.path.join(images_root, img_fn), c))
        elif isinstance(data, dict) and "images" in data:
            for img_item in data["images"]:
                img_fn = img_item.get("file_name") or img_item.get("filename") or img_item.get("img")
                sents = img_item.get("sentences") or img_item.get("caption") or img_item.get("captions")
                if img_fn is None or sents is None:
                    continue
                if isinstance(sents, str):
                    sents = [sents]
                for s in sents:
                    if isinstance(s, dict):
                        s = s.get("raw") or s.get("caption") or ""
                    pairs.append((os.path.join(images_root, img_fn), s))
        else:
            raise ValueError(f"Unknown Flickr30k annotation format: {type(data)}")

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
        return {"image": img, "text": cap, "source": "flickr30k"}
