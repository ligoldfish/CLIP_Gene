# clip_cifar100/model.py
import clip
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from configs.base_config import Config

#print(clip.__file__)

class CLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练CLIP模型
        clip_model, _ = clip.load(Config.CLIP_MODEL, device=Config.DEVICE, jit=False)
        self.vision_model = clip_model.visual.float()
        
        # 替换分类头
        self.classifier = nn.Linear(
            self.vision_model.output_dim, 
            Config.NUM_CLASSES
        ).to(Config.DEVICE).float()
    
    def forward(self, x):
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(next(self.parameters()).dtype)
        # 输入处理
        x = self.vision_model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 添加分类token
        class_token = self.vision_model.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        
        # 位置编码
        x = x + self.vision_model.positional_embedding.to(x.dtype)
        x = self.vision_model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 存储最后三层的输出
        features = []
        for i, block in enumerate(self.vision_model.transformer.resblocks):
            x = block(x)
            if i >= len(self.vision_model.transformer.resblocks) - 3:
                features.append(x)
        
        # 后处理
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.vision_model.ln_post(x[:, 0, :])
        
        if self.vision_model.proj is not None:
            x = x @ self.vision_model.proj
        
        # 分类
        logits = self.classifier(x)
        
        # 返回最后三层的分类token特征
        last_layer_features = []
        for feat in features:
            # 提取分类token特征 [batch_size, feature_dim]
            class_token_feat = feat.permute(1, 0, 2)[:, 0, :]
            last_layer_features.append(class_token_feat)
        
        # 将特征从列表转换为张量
        # last_layer_features[0] 是倒数第三层
        # last_layer_features[1] 是倒数第二层
        # last_layer_features[2] 是最后一层
        return logits, last_layer_features

class FeatureExtractor(nn.Module):
    """用于下游任务的特征提取器"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # 冻结基础模型的所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 只返回特征，不返回分类logits
        _, features = self.base_model(x)
        return features