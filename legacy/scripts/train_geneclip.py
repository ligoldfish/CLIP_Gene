import math
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset

# 导入上面的模块
from configs.base_config import Config
# from datasets.coco_caption_dataset import COCOCaptionDataset, make_coco_collate
from utils.build_geneclip import build_student_from_gene


class COCOCaptionDataset(Dataset):
    """仅用 COCO captions（训练图像 + 文本注释）。
    __getitem__ 返回 (PIL.Image, [captions...])，在 collate 里再做 preprocess/tokenize。
    """
    def __init__(self, img_dir: str, ann_file: str):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        self.images = {img['id']: img for img in ann['images']}
        # 收集 image_id -> captions
        caps = {}
        for a in ann['annotations']:
            caps.setdefault(a['image_id'], []).append(a['caption'])
        # 只保留至少 1 条 caption 的图片
        self.ids = [i for i in self.images.keys() if i in caps]
        self.captions = caps

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        file_name = self.images[image_id]['file_name']
        img = Image.open(os.path.join(self.img_dir, file_name)).convert('RGB')
        return img, self.captions[image_id]


def make_coco_collate(preprocess, tokenize_fn):
    """返回一个 collate_fn：随机抽1条 caption，做图像预处理与 tokenize。"""
    def collate(batch: List[Tuple[Image.Image, List[str]]]):
        images, caps = zip(*batch)
        txts = [random.choice(c) for c in caps]
        imgs = torch.stack([preprocess(im) for im in images])
        toks = tokenize_fn(txts)
        return imgs, toks
    return collate

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_with_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def get_param_groups(student, lr_main, lr_gene, weight_decay):
    # 收集基因参数（wrapper 内部 gene_block）
    gene_params = []
    for enc in [student.vision, student.text]:
        for blk in enc.blocks:
            for p in blk.gene_block.parameters():
                gene_params.append(p)
    gene_set = set(gene_params)

    main_params = [p for p in student.parameters() if p not in gene_set]

    return [
        {"params": main_params, "lr": lr_main, "weight_decay": weight_decay},
        {"params": gene_params, "lr": lr_gene, "weight_decay": weight_decay},
    ]


def train():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    device = cfg.DEVICE
    set_seed(42)

    # 构建学生 + 预处理
    student, meta, preprocess = build_student_from_gene(cfg, device)
    print("Gene indices (vision/text):", meta["visual_gene_indices"], meta["text_gene_indices"])

    # Data
    ds = COCOCaptionDataset(cfg.train_img_dir, cfg.train_ann_file)
    import clip
    collate = make_coco_collate(preprocess, tokenize_fn=clip.tokenize)
    dl = DataLoader(ds, batch_size=cfg.INDIVIDUAL_BATCH_SIZE, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=collate, drop_last=True)

    # Optimizer & Scheduler（两组LR：main / gene）
    param_groups = get_param_groups(student, cfg.LR_MAIN, cfg.LR_GENE, cfg.WEIGHT_DECAY)
    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)

    total_steps = cfg.INDIVIDUAL_EPOCHS * len(dl)
    warmup_steps = int(cfg.WARMUP_RATIO * total_steps)
    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    student.train()
    best_loss = float('inf')

    for epoch in range(1, cfg.INDIVIDUAL_EPOCHS + 1):
        # 阶段切换：到达阈值后解冻少量基因层（仍使用较小 lr 的 param-group）
        if epoch == cfg.UNFREEZE_AFTER_EPOCH:
            student.unfreeze_gene_layers(cfg.UNFREEZE_LAST_V, cfg.UNFREEZE_LAST_T)
            print(f"[Epoch {epoch}] Unfreeze last {cfg.UNFREEZE_LAST_V} (vision) & {cfg.UNFREEZE_LAST_T} (text) gene layers.")

        running = 0.0
        t0 = time.time()
        for step, (imgs, toks) in enumerate(dl, 1):
            imgs = imgs.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits_i, logits_t = student(imgs, toks)
                labels = torch.arange(logits_i.size(0), device=device)
                loss_i = F.cross_entropy(logits_i, labels)
                loss_t = F.cross_entropy(logits_t, labels)
                loss_clip = 0.5 * (loss_i + loss_t)
                loss_prox = cfg.PROX_LAMBDA * student.prox_loss()
                loss = loss_clip + loss_prox

            # lr schedule（逐步更新 param-group 的 lr）
            lr_scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
            for i, g in enumerate(optim.param_groups):
                base_lr = cfg.LR_MAIN if i == 0 else cfg.LR_GENE
                g["lr"] = base_lr * lr_scale

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            global_step += 1

            if step % cfg.LOG_INTERVAL == 0:
                avg = running / cfg.LOG_INTERVAL
                print(f"Epoch {epoch}/{cfg.INDIVIDUAL_EPOCHS} | Step {step}/{len(dl)} | loss={avg:.4f} | clip={loss_clip.item():.4f} | prox={loss_prox.item():.4f} | lr={optim.param_groups[0]['lr']:.2e}")
                running = 0.0

        dt = time.time() - t0
        print(f"[Epoch {epoch}] time={dt/60:.1f} min")

        # 简单保存（按 loss 判断）
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_path = os.path.join(cfg.SAVE_DIR, cfg.SAVE_NAME)
            torch.save(student.state_dict(), save_path)
            print(f"Saved best student to {save_path} (loss={best_loss:.4f})")


if __name__ == "__main__":
    train()
