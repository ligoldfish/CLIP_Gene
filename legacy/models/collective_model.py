import torch
import torch.nn as nn
from transformers import CLIPVisionModel
#需要挂代理

class DynamicClassifier(nn.Module):
    """动态扩展的分类头"""
    def __init__(self, embed_dim, initial_classes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(embed_dim, initial_classes)])
    
    def expand(self, new_classes):
        # 获取最后一层的输入维度（例如768）
        in_features = self.layers[-1].in_features
        # 新层：输入维度不变，输出维度增加new_classes
        new_layer = nn.Linear(in_features, self.layers[-1].out_features + new_classes)
        self.layers.append(new_layer)

class CollectiveModel(nn.Module):
    """集体模型：CLIP主干 + 动态分类头"""
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(config.CLIP_MODEL_NAME)
        self.freeze_clip()  # 冻结CLIP主干
        embed_dim = self.clip.config.hidden_size
        self.classifier = DynamicClassifier(embed_dim, config.INITIAL_CLASSES)
    
    def freeze_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.clip(x).pooler_output
        return self.classifier.layers[-1](features)
    
    def expand_classifier(self, new_classes):
        self.classifier.expand(new_classes)