import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.individual_model import IndividualModel
from utils.data_loader import get_base_loader
from utils.losses import compute_fisher, ewc_loss
from utils.gradient_utils import GradientMonitor
from configs.base_config import config
from models.collective_model import CollectiveModel
from utils.data_loader import get_novel_loader
from transformers import CLIPVisionModel

import tqdm

def train_individual():
    # -------------------- 加载集体模型与Learngene信息 --------------------
    checkpoint = torch.load("learngene_info.pth")
    print("检查点结构:")
    print(f"- learngene_layers类型: {type(checkpoint['learngene_layers'])}")
    print(f"- 层数: {len(checkpoint['learngene_layers'])}")
    print(f"- 第一层结构: {checkpoint['learngene_layers'][0]}")
    collective_model = CollectiveModel(config).to(config.DEVICE)
    #clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    #collective_model = CollectiveModel(config, clip_model=clip_model)
    # print("Model keys:", collective_model.state_dict().keys())
    # print("Checkpoint keys:", checkpoint['collective_state_dict'].keys())
    # 过滤掉分类器新增层的参数
    base_params = {
    k: v for k, v in checkpoint['collective_state_dict'].items()
    if not k.startswith('classifier.layers.')
    }
    collective_model.load_state_dict(base_params, strict=False)
    #collective_model.load_state_dict(checkpoint['collective_state_dict'])
    
    # 动态加载梯度监控选择的层
    learngene_layers = checkpoint['learngene_layers']
    
    # -------------------- 初始化个体模型 --------------------
    individual_model = IndividualModel(learngene_layers).to(config.DEVICE)
    
    # 冻结Learngene参数（仅训练适配层和分类头）
    for param in individual_model.learngene.parameters():
        param.requires_grad = False
    
    # -------------------- 数据加载 --------------------
    novel_loader = get_novel_loader(config.NOVEL_CLASSES_DATA, n_shot=5)
    
    # -------------------- 计算Fisher信息矩阵（用于EWC正则化） --------------------
    # 使用集体模型的分类头数据计算Fisher（模拟元知识保留）
    base_loader = get_base_loader(config.BASE_CLASSES_DATA, config.COLLECTIVE_BATCH_SIZE)
    fisher_matrix = compute_fisher(collective_model.classifier, base_loader)
    
    # -------------------- 优化器与损失函数 --------------------
    optimizer = torch.optim.Adam([
    {'params': individual_model.adaptor.parameters(), 'lr': config.INDIVIDUAL_LR},  
    {'params': individual_model.classifier.parameters(), 'lr': config.INDIVIDUAL_LR* 10} 
    ])
    criterion = nn.CrossEntropyLoss()
    
    # -------------------- 梯度监控（可选） --------------------
    gradient_monitor = GradientMonitor(individual_model)
    
    # -------------------- 训练循环 --------------------
    for epoch in tqdm.tqdm(range(30), desc="epoch"):  # 训练30个epoch
        individual_model.train()
        total_loss = 0.0
        correct = 0
        
        for x, y in novel_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            # 前向传播
            logits = individual_model(x)
            loss_ce = criterion(logits, y)
            
            # EWC正则化损失（仅作用于Learngene参数）
            loss_ewc = ewc_loss(individual_model.learngene, fisher_matrix, config.EWC_LAMBDA)
            
            # 总损失
            loss = loss_ce + loss_ewc
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
        
        # 计算epoch精度
        accuracy = 100 * correct / len(novel_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(novel_loader):.4f}, Acc={accuracy:.2f}%")
        
        # 保存模型（可选）
        if (epoch+1) % 10 == 0:
            torch.save(individual_model.state_dict(), f"individual_model_epoch{epoch+1}.pth")
    
    # -------------------- 验证开放世界识别（可选） --------------------
    # individual_model.eval()
    # open_world_loader = get_base_loader(config.OPEN_WORLD_DATA, config.INDIVIDUAL_BATCH_SIZE)
    # ...（实现开放世界检测逻辑，如公式2-3）

train_individual()