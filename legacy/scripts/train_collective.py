import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gradient_utils import GradientMonitor
from models.collective_model import CollectiveModel
import torch
import torch.optim as optim
#from utils.data_loader import get_loaders
from utils.gradient_utils import GradientMonitor
from utils.data_loader import get_base_loader, load_open_world_class
from configs.base_config import config
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm as tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler 

print(f"GPU可用: {torch.cuda.is_available()}")

def train_collective():
    model = CollectiveModel(config).to(config.DEVICE)
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.COLLECTIVE_LR)
    
    # 初始化梯度监控器
    gradient_monitor = GradientMonitor(model.clip.vision_model.encoder)  # 仅监控CLIP的Transformer块
    
    # 初始任务训练
    # 替换原有的数据加载逻辑
    base_loader = get_base_loader(config.BASE_CLASSES_DATA, config.COLLECTIVE_BATCH_SIZE)
    scaler = GradScaler()
    
    
    #open_world_images = load_open_world_class(class_id=60)  # 假设检测到新类60
    for epoch in tqdm.tqdm(range(50), desc="Epochs"):
        print("start training")
        model.train()
        for x, y in base_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            with autocast():
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            #optimizer.step()
            
            # 记录梯度（每个batch更新）
            #gradient_monitor.get_stable_layers(threshold=config.GRADIENT_THRESHOLD)
            gradient_monitor.get_stable_layers()
            if epoch >= config.MIN_EPOCHS_FOR_STABILITY:
                stable_layers = gradient_monitor.get_stable_layers()
                if stable_layers:
                    print(f"Epoch {epoch}: 检测到稳定层 {len(stable_layers)}个")
                else: 
                    print(f"Epoce{epoch}:未检测到稳定层")
    
    # 持续学习阶段：检测并加载开放世界类
    for task_id in tqdm.tqdm(range(config.NUM_TASKS), desc="Tasks"):
        open_world_class_id = 60 + task_id  
        open_world_images = load_open_world_class(open_world_class_id)
        if open_world_images is None:
            print(f"Open world class {open_world_class_id} not found.")

        #print(open_world_images)
        if isinstance(open_world_images, torch.Tensor):
            open_world_images = [open_world_images]
        model.expand_classifier(new_classes=1)  
        #dataset = TensorDataset(open_world_images, torch.full((len(open_world_images), task_id + 1), fill_value=open_world_class_id))
        image_tensor = torch.stack(open_world_images)
        labels = torch.full((image_tensor.shape[0],), fill_value=open_world_class_id)
        dataset = TensorDataset(image_tensor, labels)
        open_loader = DataLoader(dataset, batch_size=config.COLLECTIVE_BATCH_SIZE, shuffle=True)
        
        for x, y in open_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            #print("x.shape:", x)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward
            scaler.step(optimizer)
            scaler.update()

    stable_layers = gradient_monitor.get_stable_layers()
    assert len(stable_layers) > 0, "未检测到任何稳定层！"
    
    learngene_layers = []
    for name, module in model.clip.vision_model.encoder.named_children():
        if name in stable_layers:
            learngene_layers.append(module)

    torch.save({
        'learngene_layers': learngene_layers,
        'collective_state_dict': model.state_dict()
    }, "learngene_info.pth")

    
    # plt.plot(gradient_monitor.grad_history['resblock_11'])
    # plt.xlabel('Iteration')
    # plt.ylabel('Gradient Norm')
    # plt.title('Gradient Stability of Layer resblock_11')
    # plt.show()


if __name__ == '__main__':
    # Windows多进程必须的代码
    torch.multiprocessing.freeze_support()
    train_collective()