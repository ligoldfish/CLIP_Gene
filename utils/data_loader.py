import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from configs.base_config import config
# -------------------- 预处理定义 --------------------
def get_clip_preprocess():
    """CLIP标准预处理（用于基类和开放世界类）"""
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

def get_novel_augmentation():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        get_clip_preprocess()
    ])

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        
        # 遍历每个类别的子目录
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue  # 跳过非目录文件
            # 收集所有图片路径和标签
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        """返回数据集总样本数"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class NovelDataset(Dataset):
    def __init__(self, root_dir, n_shot=5, transform=None):
        self.root_dir = root_dir
        self.n_shot = n_shot
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        
        # 遍历每个类别的子目录
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue  # 跳过非目录文件
            # 随机选择n_shot张图片
            all_images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            selected_images = random.sample(all_images, min(self.n_shot, len(all_images)))
            # 收集路径和标签
            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        """返回数据集总样本数"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_base_loader(data_root, batch_size=config.COLLECTIVE_BATCH_SIZE):
    transform = get_clip_preprocess()
    dataset = BaseDataset(data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

def get_novel_loader(data_root, n_shot=5, batch_size=config.INDIVIDUAL_BATCH_SIZE):
    transform = get_novel_augmentation()
    dataset = NovelDataset(data_root, n_shot=n_shot, transform=transform)
    print(f"成功加载样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("No images found!")
        return None  # 如果没有图片，返回空
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

def load_open_world_class(class_id, data_root="./CIFAR100_splited/CIFAR100_splited/open_world/train"):
    class_dir = os.path.join(data_root, f"class_{class_id}")
    if not os.path.exists(class_dir):
        print("Class not found!")
        return None  # 类别不存在时返回空
    preprocess = get_clip_preprocess()
    images = []
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        images.append(preprocess(img))
        if not images:
            print("No images found!")
            return None
    return images