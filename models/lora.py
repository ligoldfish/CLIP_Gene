"""LoRA 注入：MLP 用 LoRALinear；注意力用 LoRAMultiheadAttention（手写 SDPA + 零初始化低秩增量）。

openai ResidualAttentionBlock.attn = nn.MultiheadAttention（in_proj_weight 单 [3D,D] Parameter，
非 Linear），无法直接包 submodule。故原地替换为 LoRAMultiheadAttention：保留冻结的
in_proj/out_proj（param 名不变 → build 的冻结一致性断言仍过），手写注意力并加 q/k/v/out 的
零初始化(B=0)低秩增量，初始严格等价；forward 签名与 MHA 一致，保 attention() 的 self.attn_mask 机制。
"""
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """冻结 base nn.Linear + 低秩增量；B=0 初始 → 输出与 base 等价。"""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.scaling = (alpha or rank) / rank
        self.A = nn.Parameter(torch.zeros(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A.t()) @ self.B.t())


class LoRAMultiheadAttention(nn.Module):
    """冻结 nn.MultiheadAttention + q/k/v/out 低秩增量。手写自注意力(self-attn, q=k=v)。"""

    def __init__(self, mha: nn.MultiheadAttention, rank: int = 16,
                 lora_qkv: bool = True, lora_out: bool = True, alpha=None):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        # 复用冻结权重（保 param 名 attn.in_proj_weight / attn.out_proj.*）
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        self.in_proj_weight.requires_grad = False
        if self.in_proj_bias is not None:
            self.in_proj_bias.requires_grad = False
        for p in self.out_proj.parameters():
            p.requires_grad = False

        r, D = rank, self.embed_dim
        self.scaling_lora = (alpha or r) / r
        self.lora_qkv = lora_qkv
        if lora_qkv:
            for nm in ("q", "k", "v"):
                A = nn.Parameter(torch.zeros(r, D)); B = nn.Parameter(torch.zeros(D, r))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5)); nn.init.zeros_(B)
                setattr(self, f"{nm}_A", A); setattr(self, f"{nm}_B", B)
        self.lora_out = lora_out
        if lora_out:
            self.o_A = nn.Parameter(torch.zeros(r, D)); self.o_B = nn.Parameter(torch.zeros(D, r))
            nn.init.kaiming_uniform_(self.o_A, a=math.sqrt(5)); nn.init.zeros_(self.o_B)

    def _qkv(self, x):
        D = self.embed_dim
        Wq, Wk, Wv = self.in_proj_weight[:D], self.in_proj_weight[D:2 * D], self.in_proj_weight[2 * D:]
        if self.in_proj_bias is not None:
            bq, bk, bv = self.in_proj_bias[:D], self.in_proj_bias[D:2 * D], self.in_proj_bias[2 * D:]
        else:
            bq = bk = bv = None
        q, k, v = F.linear(x, Wq, bq), F.linear(x, Wk, bk), F.linear(x, Wv, bv)
        if self.lora_qkv:
            q = q + self.scaling_lora * ((x @ self.q_A.t()) @ self.q_B.t())
            k = k + self.scaling_lora * ((x @ self.k_A.t()) @ self.k_B.t())
            v = v + self.scaling_lora * ((x @ self.v_A.t()) @ self.v_B.t())
        return q, k, v

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        x = query                                  # openai 自注意力 q=k=v=x，[L,B,D]
        L, B, D = x.shape
        q, k, v = self._qkv(x)

        def split(t):
            return t.reshape(L, B * self.num_heads, self.head_dim).permute(1, 0, 2)  # [B*H, L, hd]

        qh, kh, vh = split(q), split(k), split(v)
        attn = torch.bmm(qh * self.scaling, kh.transpose(1, 2))   # [B*H, L, L]
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)               # 加性 mask [L,L] 广播
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, vh).permute(1, 0, 2).reshape(L, B, D)  # [L,B,D]
        o = self.out_proj(out)
        if self.lora_out:
            o = o + self.scaling_lora * ((out @ self.o_A.t()) @ self.o_B.t())
        return o, None


def inject_lora_into_gene_block(block, rank: int = 16, targets: Sequence[str] = ("attn", "mlp"),
                                lora_qkv: bool = True, lora_out: bool = True):
    """原地给 openai ResidualAttentionBlock 注入 LoRA。base 权重保持冻结、名不变(attn)/加 .base(mlp)。"""
    if "attn" in targets:
        block.attn = LoRAMultiheadAttention(block.attn, rank, lora_qkv, lora_out)
    if "mlp" in targets:
        # block.mlp = nn.Sequential(c_fc, gelu, c_proj)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, rank)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank)
    return block
