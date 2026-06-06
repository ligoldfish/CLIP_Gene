from typing import Dict, List
import math, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities ----------

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------- Residual MLP Adapter (kept as-is) ----------
class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, drop: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        nn.init.normal_(self.down.weight, std=1e-4)
        nn.init.zeros_(self.up.weight)  # near-identity start
    def forward(self, x):
        return x + self.drop(self.up(self.act(self.down(x))))


# ---------- Shallow CNN blocks ----------
class VisionCNNUnit(nn.Module):
    """1x1 (down) → DWConv3x3 → 1x1 (up), residual+zero init on up.
       x: [B, D, H, W]
    """
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Conv2d(dim, bottleneck, kernel_size=1, bias=False)
        self.dw   = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, groups=bottleneck, bias=False)
        self.up   = nn.Conv2d(bottleneck, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, dim)  # light norm over channels
        self.act  = nn.GELU()
        zero_module(self.up)
    def forward(self, x):
        y = self.down(self.norm(x))
        y = self.dw(y)
        y = self.up(self.act(y))
        return x + y

class VisionShallowCNN(nn.Module):
    def __init__(self, dim: int, layers: int = 2, bottleneck: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(*[VisionCNNUnit(dim, bottleneck) for _ in range(layers)])
    def forward(self, x_bchw):
        return self.blocks(x_bchw)

class TextCausalCNNUnit(nn.Module):
    """1x1 (down) → causal DWConv1d(k=3) → 1x1 (up), residual+zero init.
       x: [B, D, T]
    """
    def __init__(self, dim: int, bottleneck: int = 64, k: int = 3):
        super().__init__()
        self.k = k
        self.down = nn.Conv1d(dim, bottleneck, kernel_size=1, bias=False)
        self.dw   = nn.Conv1d(bottleneck, bottleneck, kernel_size=k, groups=bottleneck, bias=False)
        self.up   = nn.Conv1d(bottleneck, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act  = nn.GELU()
        zero_module(self.up)
    def forward(self, x):
        # x: [B,D,T]; causal left pad
        B, D, T = x.shape
        y = self.norm(x.transpose(1,2)).transpose(1,2)  # LN over D
        y = self.down(y)
        y = F.pad(y, (self.k-1, 0))  # (left, right)
        y = self.dw(y)
        y = self.up(self.act(y))
        return x + y

class TextShallowCNN(nn.Module):
    def __init__(self, dim: int, layers: int = 1, bottleneck: int = 64, k: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([TextCausalCNNUnit(dim, bottleneck, k) for _ in range(layers)])
    def forward(self, x_bdt):
        for blk in self.blocks:
            x_bdt = blk(x_bdt)
        return x_bdt


# ---------- Gene block wrapper with Adapter + Prox ----------
class GeneBlockWrapper(nn.Module):
    def __init__(self, gene_block: nn.Module, dim: int, adapter_bottleneck: int = 64, freeze_gene: bool = True):
        super().__init__()
        self.gene_block = gene_block
        if freeze_gene:
            for p in self.gene_block.parameters():
                p.requires_grad = False
        # adapter after block output（基因全程冻结，去掉 prox 快照）
        self.adapter = ResidualAdapter(dim, bottleneck=adapter_bottleneck)
        # whether block.forward accepts attn_mask kw
        sig = inspect.signature(self.gene_block.forward)
        self._accepts_attn_kw = ("attn_mask" in sig.parameters)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            T = x.shape[0]
            m = attn_mask[:T, :T].to(device=x.device, dtype=x.dtype)
            if self._accepts_attn_kw:
                y = self.gene_block(x, attn_mask=m)
            else:
                if hasattr(self.gene_block, "attn_mask"):
                    self.gene_block.attn_mask = m
                y = self.gene_block(x)
        else:
            y = self.gene_block(x)
        y = self.adapter(y)
        return y

    def unfreeze(self):
        for p in self.gene_block.parameters():
            p.requires_grad = True


# ---------- Stitch layer (SN-Net 式，修非连续基因块的残差流断裂) ----------
class StitchLayer(nn.Module):
    """对 [..., D] 末维做仿射映射，对齐相邻基因块间/stem->首基因 的激活分布。

    full:  nn.Linear(D, D)，eye 初始化（LSQ 前回退为恒等）。
    低秩:  x + U(V x)，U 零初始化（残差恒等启动），LSQ 用 W-I 的截断 SVD 装入。
    LSQ 初始化后以低 LR 可训。
    """

    def __init__(self, dim: int, rank: int = None, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        if rank is None:                       # full DxD
            self.lin = nn.Linear(dim, dim, bias=bias)
            nn.init.eye_(self.lin.weight)
            if bias:
                nn.init.zeros_(self.lin.bias)
        else:                                  # low-rank residual
            self.V = nn.Linear(dim, rank, bias=False)
            self.U = nn.Linear(rank, dim, bias=bias)
            nn.init.normal_(self.V.weight, std=1e-3)
            nn.init.zeros_(self.U.weight)
            if bias:
                nn.init.zeros_(self.U.bias)

    def forward(self, x):                      # x: [..., D]
        if self.rank is None:
            return self.lin(x)
        return x + self.U(self.V(x))