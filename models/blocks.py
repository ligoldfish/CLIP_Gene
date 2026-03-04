# models/blocks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFP32(nn.LayerNorm):
    """CLIP-style LayerNorm that is stable under fp16: normalize in fp32 then cast back."""
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    """CLIP MLP: Linear(d, 4d) -> QuickGELU -> Linear(4d, d)"""
    def __init__(self, d_model: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.c_fc = nn.Linear(d_model, hidden)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    """
    CLIP-compatible ResidualAttentionBlock:
      x = x + attn(LN(x))
      x = x + mlp(LN(x))
    Uses nn.MultiheadAttention with CLIP parameter naming.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=False)
        self.ln_1 = LayerNormFP32(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio)
        self.ln_2 = LayerNormFP32(d_model)
        self.attn_mask = attn_mask  # [L, L] float mask with -inf upper-tri

    def attention(self, x: torch.Tensor):
        # x: [seq, batch, d]
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        y = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return y

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BottleneckMLP(nn.Module):
    """
    Token-wise bottleneck MLP between shallow blocks and learngene.

    Modes:
      - bottleneck is None: use default bottleneck=max(64, d_model//4)
      - bottleneck > 0: residual bottleneck x <- x + W_up(Act(W_down(x)))
      - bottleneck <= 0: identity (disabled), useful for "shallow -> gene" directly
    """
    def __init__(self, d_model: int, bottleneck: int = None, dropout: float = 0.0):
        super().__init__()
        self.disabled = False

        if bottleneck is not None and int(bottleneck) <= 0:
            self.disabled = True
            self.down = None
            self.act = None
            self.drop = None
            self.up = None
            return

        if bottleneck is None:
            bottleneck = max(64, d_model // 4)

        self.down = nn.Linear(d_model, bottleneck)
        self.act = QuickGELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, d_model)

        # near-identity at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor):
        # x: [seq, batch, d]
        if self.disabled:
            return x
        return x + self.up(self.drop(self.act(self.down(x))))


class ProjectionHead(nn.Module):
    """
    Map tower output to shared low-dim space.
    - linear: Linear(d -> out)
    - mlp: Linear(d->h) + act + Linear(h->out)
    """
    def __init__(self, in_dim: int, out_dim: int, use_mlp: bool = False, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.use_mlp = use_mlp
        if not use_mlp:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            if hidden_dim is None:
                hidden_dim = in_dim
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.act = QuickGELU()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        if not self.use_mlp:
            return self.proj(x)
        return self.fc2(self.drop(self.act(self.fc1(x))))


def build_causal_attention_mask(context_len: int, device=None):
    # CLIP-style causal mask: allow attending to self and previous tokens
    mask = torch.full((context_len, context_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask
