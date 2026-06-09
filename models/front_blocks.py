"""前端可训 transformer 块（基因块之前）。

= deepcopy 的 openai ResidualAttentionBlock，全可训。文本沿用 openai 的
self.attn_mask 属性约定（block.forward(x) 内部用 self.attn_mask）。
"""
import torch.nn as nn


class FrontBlocks(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        for b in self.blocks:
            for p in b.parameters():
                p.requires_grad = True

    def run(self, x, attn_mask=None):
        """x: [L,B,D]。attn_mask（文本因果）按 seq 长切片后写入块的 attn_mask 属性。"""
        for b in self.blocks:
            if attn_mask is not None:
                T = x.shape[0]
                m = attn_mask[:T, :T].to(device=x.device, dtype=x.dtype)
                if hasattr(b, "attn_mask"):
                    b.attn_mask = m
            x = b(x)
        return x

    def forward(self, x, attn_mask=None):
        return self.run(x, attn_mask=attn_mask)
