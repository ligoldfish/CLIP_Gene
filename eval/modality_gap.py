"""模态间隔诊断（Mind the Gap, Liang et al. 2022）。

gap = ‖mean(img_emb) − mean(txt_emb)‖₂，固定 COCO-val 子集上算，
检验学生是否保住 teacher 的对齐几何（gene 暖启动 vs 随机初始化）。
"""
import torch
from PIL import Image

from eval.model_interface import EvalModel
from eval.datasets_local import load_coco_val_pairs


@torch.no_grad()
def modality_gap(model: EvalModel, preprocess, val_img_dir: str, val_ann: str,
                 n_pairs: int = 1000, batch_size: int = 128) -> dict:
    pairs = load_coco_val_pairs(val_img_dir, val_ann, max_images=n_pairs)
    # 图像
    img_embs, buf = [], []
    for p, _ in pairs:
        buf.append(preprocess(Image.open(p).convert("RGB")))
        if len(buf) == batch_size:
            img_embs.append(model.encode_image(torch.stack(buf))); buf = []
    if buf:
        img_embs.append(model.encode_image(torch.stack(buf)))
    img_emb = torch.cat(img_embs, dim=0)
    # 文本（每图首条 caption）
    texts = [caps[0] for _, caps in pairs]
    txt_emb = torch.cat([model.encode_text(texts[i:i + batch_size])
                         for i in range(0, len(texts), batch_size)], dim=0)

    img_c = img_emb.mean(0)
    txt_c = txt_emb.mean(0)
    gap = (img_c - txt_c).norm().item()
    paired_cos = torch.cosine_similarity(img_emb, txt_emb, dim=-1).mean().item()
    return {
        "gap_l2": gap,
        "img_centroid_norm": img_c.norm().item(),
        "txt_centroid_norm": txt_c.norm().item(),
        "mean_cos_paired": paired_cos,
        "n": len(pairs),
    }
