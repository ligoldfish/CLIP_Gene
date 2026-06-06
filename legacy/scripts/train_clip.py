import os
import json
import random
import torch
import clip
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from configs.base_config import Config

class COCOCaptionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        images = {img['id']: img['file_name'] for img in ann['images']}
        self.captions = {}
        for a in ann['annotations']:
            self.captions.setdefault(a['image_id'], []).append(a['caption'])
        self.ids = list(self.captions.keys())
        self.images = images

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_name = self.images[img_id]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.captions[img_id]
    
def collate_fn(batch, preprocess):
    images, caps = zip(*batch)
    texts = [random.choice(c_list) for c_list in caps]
    image_inputs = torch.stack([preprocess(img) for img in images])
    text_inputs = clip.tokenize(texts)
    return image_inputs, text_inputs

# 收集每层梯度超过阈值比例
def compute_layer_rhos(model):
    rho = {}
    for name, param in model.named_parameters():
        if ((name.startswith('visual.transformer.resblocks.') or name.startswith('transformer.resblocks.'))
                and param.grad is not None):
            layer = name.split('resblocks.')[1].split('.')[0]
            grad_norm = param.grad.abs().mean().item()
            rho.setdefault(layer, []).append(grad_norm)
    # 统计每层大于阈值的比率
    layer_rho = {int(l): sum(1 for g in grads if g>Config.GRADIENT_THRESHOLD)/len(grads)
                 for l, grads in rho.items()}
    return layer_rho

def save_gene_layers(state_dict, layers, save_path):
    gene_state = {k: v for k, v in state_dict.items()
                  if any(f'resblocks.{i}.' in k for i in layers)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gene_state, save_path)
    print(f"Saved gene layers {layers} to {save_path}")

def main():
    os.makedirs(Config.SAVE_PATH, exist_ok=True)
    model, preprocess = clip.load(Config.CLIP_MODEL, device=Config.DEVICE)
    model = model.float()           # 把所有权重强制转回 float32
    model.train()
    
    def collate_fn_pre(batch):
        return collate_fn(batch, preprocess)
    # 数据变换与加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("start data loading\n")
    train_ds = COCOCaptionDataset(Config.train_img_dir, Config.train_ann_file, transform = None)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.COLLECTIVE_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn_pre
    )
    # 加在 DataLoader 构造之后，训练循环之前
    # model.eval()  
    # with torch.no_grad():
    #     images, texts = next(iter(train_loader))
    #     images = images.to(Config.DEVICE)
    #     texts  = texts.to(Config.DEVICE)
    #     logits_i, logits_t = model(images, texts)
    #     print("Forward on one batch →",
    #           "logits_i NaN?", torch.isnan(logits_i).any(),
    #           " Inf?", torch.isinf(logits_i).any(),
    #           "| logits_t NaN?", torch.isnan(logits_t).any(),
    #           " Inf?", torch.isinf(logits_t).any())

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.COLLECTIVE_LR)
    total = len(model.visual.transformer.resblocks)
    accum = {i: [] for i in range(total)}

    for epoch in range(Config.clip_epochs):
        running_loss = 0.0
        for step, (images, texts) in enumerate(train_loader):
            images = images.to(Config.DEVICE)
            texts = texts.to(Config.DEVICE)
            # if torch.isnan(images).any():
            #     print("NaN in images!")
            # if torch.isnan(texts.float()).any():  # token 本身不会 NaN，但做 embedding 后可检查
            #     print("NaN in texts/token embeddings!")

            logits_per_image, logits_per_text = model(images, texts)
            # 检查 logits
            # if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
            #     print("🚨 bad logits_per_image:", 
            #       "NaN?", torch.isnan(logits_per_image).any(), 
            #       "Inf?", torch.isinf(logits_per_image).any())
            # if torch.isnan(logits_per_text).any() or torch.isinf(logits_per_text).any():
            #     print("🚨 bad logits_per_text:", 
            #       "NaN?", torch.isnan(logits_per_text).any(), 
            #       "Inf?", torch.isinf(logits_per_text).any())
            labels = torch.arange(images.size(0), device=Config.DEVICE)
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = 0.5 * (loss_i + loss_t)

            optimizer.zero_grad()
            loss.backward()
            step_rho = compute_layer_rhos(model)
            for layer, rate in step_rho.items():
                accum[layer].append(rate)
            # for name, p in model.named_parameters():
            #     if p.grad is not None and torch.isnan(p.grad).any():
            #         print(f"NaN grad in {name}")
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} Step {step+1} Loss {running_loss/100:.4f}")
                running_loss = 0.0

    # 保存模型权重
    torch.save(model.state_dict(), Config.model_ckpt)
    print(f"Saved trained CLIP to {Config.model_ckpt}")

    mean_rho = {layer: sum(rates) / len(rates) for layer, rates in accum.items() if rates}
    top_layers = sorted(mean_rho, key=lambda x: mean_rho[x], reverse=True)[:Config.top_k]
    print(f"Selected gene layers: {top_layers}")

    # 保存基因层参数
    save_gene_layers(model.state_dict(), top_layers, Config.extracted_layers_ckpt)


if __name__ == "__main__":
    main()
