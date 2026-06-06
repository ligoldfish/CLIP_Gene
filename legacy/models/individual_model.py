import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 改进后的IndividualModel实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.base_config import config

class IndividualModel(nn.Module):
    """增强版个体模型结构"""
    def __init__(self, learngene_layers):
        super().__init__()
        if not isinstance(learngene_layers, list) or len(learngene_layers) == 0:
            raise ValueError(
                f"learngene_layers必须是非空列表，实际得到: {type(learngene_layers)} "
                f"(长度: {len(learngene_layers) if hasattr(learngene_layers, '__len__') else 'N/A'})"
            )
        # Part 1: 学习基因层 (固定不变)
        self.learngene = nn.Sequential(*learngene_layers)
        
        # Part 2: 自适应结构 (可调整)
        self.embed_dim = learngene_layers[-1].output_dim  # 自动获取最后一层输出维度
        self.adaptor = nn.Sequential(
            ResidualBlock(self.embed_dim, 256),
            ResidualBlock(256, 256)
        )
        
        # Part 3: 分类头
        self.classifier = nn.Linear(256, config.NOVEL_CLASSES)  # 动态类别数
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化适配层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 步骤1: 通过学习基因层
        with torch.no_grad():  # 保持学习基因不变
            x = self.learngene(x)  # [B, embed_dim]

        x = F.normalize(x, p=2, dim=-1)
        # 步骤2: 自适应转换
        x = self.adaptor(x)  # [B, 256]
        
        # 步骤3: 分类
        return self.classifier(x)  # [B, num_classes]
    
class ResidualBlock(nn.Module):
    """技巧2：残差块实现"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.linear(x) + self.shortcut(x)