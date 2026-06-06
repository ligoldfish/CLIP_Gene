import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gradient_utils import GradientMonitor
from models.collective_model import CollectiveModel
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
#from utils.data_loader import get_loaders
from utils.gradient_utils import GradientMonitor
from utils.data_loader import get_base_loader, load_open_world_class
from configs.base_config import config
from torch.utils.data import TensorDataset, DataLoader
#from transformers import CLIPVisionModel
import torch.nn.functional as F
import tqdm as tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

from configs.base_config import Config
from models.vision_model import CLIPVisionModel

#训练CLIP的vision encoder
#print(f"GPU可用: {torch.cuda.is_available()}")

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=Config.CIFAR_ROOT, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root=Config.CIFAR_ROOT, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.COLLECTIVE_BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.COLLECTIVE_BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    scaler = GradScaler()
    
    for images, labels in tqdm.tqdm(loader, desc="Training"):
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            logits, feature = model(images)
            loss = criterion(logits, labels)
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc="Testing"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            with autocast():
                logits, feature = model(images)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

def main():
    # 初始化
    train_loader, test_loader = get_dataloaders()
    model = CLIPVisionModel().to(Config.DEVICE)
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    
    # 优化所有参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.COLLECTIVE_LR, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scaler = GradScaler()
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs * len(train_loader))
    
    best_acc = 0.0
    print(f"Starting training on {Config.DEVICE}...")
    
    for epoch in range(Config.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Saved best model with accuracy {best_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()


