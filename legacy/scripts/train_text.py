# train_text_and_select.py
import os
import json
import random
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from configs.base_config import Config

class COCOTextDataset(Dataset):
    """加载 COCO Captions，只返回每图的 caption 列表"""
    def __init__(self, ann_file):
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        self.caption_dict = {}
        for img in ann['images']:
            self.caption_dict[img['id']] = []
        for a in ann['annotations']:
            self.caption_dict[a['image_id']].append(a['caption'])
        self.ids = list(self.caption_dict.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 返回该图像所有 captions
        return self.caption_dict[self.ids[idx]]

# 随机选取 caption 对并 tokenize
def collate_fn(batch):
    texts1, texts2 = [], []
    for caps in batch:
        t1, t2 = random.sample(caps, 2)
        texts1.append(t1)
        texts2.append(t2)
    return clip.tokenize(texts1), clip.tokenize(texts2)

# 统计文本分支每层梯度活跃度
def compute_layer_rhos(model, threshold):
    rho = {}
    for name, param in model.named_parameters():
        if name.startswith('transformer.resblocks.') and param.grad is not None:
            layer = int(name.split('resblocks.')[1].split('.')[0])
            active = param.grad.abs().mean().item() > threshold
            rho.setdefault(layer, []).append(active)
    return {layer: sum(flags) / len(flags) for layer, flags in rho.items()}

# 从 state_dict 提取指定层参数并保存
def save_gene_layers(state_dict, layers, save_path):
    gene_state = {k: v for k, v in state_dict.items()
                  if any(f'transformer.resblocks.{i}.' in k for i in layers)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gene_state, save_path)
    print(f"Saved TEXT-only gene layers {layers} to {save_path}")

# Text-only 训练并提取基因层

def train_and_select_text():
    os.makedirs(Config.SAVE_PATH, exist_ok=True)
    # 数据加载
    ds = COCOTextDataset(Config.train_ann_file)
    loader = DataLoader(
        ds,
        batch_size=Config.COLLECTIVE_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 加载 CLIP 模型，仅训练文本分支
    model, _ = clip.load(Config.CLIP_MODEL, device=Config.DEVICE, jit=False)
    model = model.float().train()
    # 冻结视觉分支参数
    for name, p in model.named_parameters():
        if name.startswith('visual.'):
            p.requires_grad = False

    # 优化器和损失
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=Config.COLLECTIVE_LR)
    scaler = torch.cuda.amp.GradScaler()

    # 梯度活跃度累积
    num_blocks = len(model.transformer.resblocks)
    accum = {i: [] for i in range(num_blocks)}

    for epoch in range(Config.num_epochs):
        for texts1, texts2 in loader:
            texts1 = texts1.to(Config.DEVICE)
            texts2 = texts2.to(Config.DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z1 = model.encode_text(texts1)
                z2 = model.encode_text(texts2)
                z1 = z1 / z1.norm(dim=-1, keepdim=True)
                z2 = z2 / z2.norm(dim=-1, keepdim=True)
                logits = z1 @ z2.t() / Config.TEMPERATURE
                labels = torch.arange(logits.size(0), device=Config.DEVICE)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            # 记录活跃度
            step_rho = compute_layer_rhos(model, Config.GRADIENT_THRESHOLD)
            for L, active in step_rho.items():
                accum[L].append(active)
            scaler.step(optimizer)
            scaler.update()

    torch.save(model.state_dict(), Config.TEXT_MODEL_PATH)
    print(f"Saved trained CLIP to {Config.TEXT_MODEL_PATH}")
    # 选 Top-K 基因层
    mean_rho = {L: sum(flags)/len(flags) for L, flags in accum.items() if flags}
    top_layers = sorted(mean_rho, key=lambda x: mean_rho[x], reverse=True)[:Config.top_k]
    print(f"TEXT-only selected gene layers: {top_layers}")
    save_gene_layers(model.state_dict(), top_layers, Config.TEXT_MODEL_LAYER_PATH)

if __name__ == '__main__':
    train_and_select_text()
