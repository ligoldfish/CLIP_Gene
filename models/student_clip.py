"""统一学生模型：非连续冻结基因块 + 间隔缝合层 + 可选 shallow-CNN + logit_scale clamp。

替代旧的 models/geneclip.py 与 models/cnn_geneclip.py（两者 90% 重复，且 geneclip 的
attn_mask 处理对 stock openai-clip 已损坏）。全部用 models/blocks.py 的 robust 组件。

基因块全程冻结；可训部分 = adapter + 缝合层 + logit_scale(+ 可选 pos/proj)。
"""
import copy
from typing import List, Optional

import torch
import torch.nn as nn

from models.blocks import (
    GeneBlockWrapper,
    ResidualAdapter,  # noqa: F401  (供 trainable 分组 isinstance)
    StitchLayer,
    VisionShallowCNN,
    TextShallowCNN,
)


def _stitch_presence(sel: List[int], stitch_at_head: bool, n_total: int):
    """返回 (pre, mids, post) 布尔：哪些位置需要真缝合层。

    pre : stem/embed -> 首基因块（sel[0]!=0）
    mids: 相邻选中块间隔 >1
    post: 末基因块后（stitch_at_head 且 sel[-1] 非最后一层）
    """
    pre = len(sel) > 0 and sel[0] != 0
    mids = [(sel[k + 1] - sel[k]) > 1 for k in range(len(sel) - 1)]
    post = stitch_at_head and len(sel) > 0 and sel[-1] != (n_total - 1)
    return pre, mids, post


def _make_stitch(dim: int, use_full: bool, rank: int) -> nn.Module:
    return StitchLayer(dim, rank=None if use_full else rank)


class _StitchedTrunk(nn.Module):
    """共享的"基因块 + 间隔缝合"序列容器（视觉/文本编码器复用其 forward 逻辑）。"""

    def __init__(self, gene_blocks, sel, dim, adapter_bottleneck,
                 use_full_stitch, stitch_rank, stitch_at_head, n_total):
        super().__init__()
        self.sel = list(sel)
        self.blocks = nn.ModuleList([
            GeneBlockWrapper(b, dim=dim, adapter_bottleneck=adapter_bottleneck, freeze_gene=True)
            for b in gene_blocks
        ])
        pre, mids, post = _stitch_presence(self.sel, stitch_at_head, n_total)
        self.pre_stitch = _make_stitch(dim, use_full_stitch, stitch_rank) if pre else nn.Identity()
        self.mid_stitch = nn.ModuleList([
            _make_stitch(dim, use_full_stitch, stitch_rank) if m else nn.Identity() for m in mids
        ])
        self.post_stitch = _make_stitch(dim, use_full_stitch, stitch_rank) if post else nn.Identity()

    def run(self, x, attn_mask=None, return_hidden=False):
        """x: [L,B,D]（已是序列优先布局）。返回处理后的 x（同布局）。"""
        hidden = []
        x = self.pre_stitch(x)
        for k, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask) if attn_mask is not None else blk(x)
            if return_hidden:
                hidden.append(x)
            if k < len(self.blocks) - 1:
                x = self.mid_stitch[k](x)
        x = self.post_stitch(x)
        if return_hidden:
            return x, hidden
        return x


class StudentVisionEncoder(nn.Module):
    def __init__(self, teacher_visual, gene_blocks: List[nn.Module], sel: List[int], width: int,
                 adapter_bottleneck: int = 64, use_shallow_cnn: bool = False,
                 shallow_layers: int = 2, shallow_bottleneck: int = 64,
                 use_full_stitch: bool = True, stitch_rank: int = 32, stitch_at_head: bool = False,
                 n_total: int = 12, freeze_stem: bool = True, freeze_post: bool = True):
        super().__init__()
        self.conv1 = copy.deepcopy(teacher_visual.conv1)
        self.class_embedding = nn.Parameter(teacher_visual.class_embedding.detach().clone(),
                                            requires_grad=not freeze_stem)
        self.positional_embedding = nn.Parameter(teacher_visual.positional_embedding.detach().clone(),
                                                 requires_grad=not freeze_stem)
        self.ln_pre = copy.deepcopy(teacher_visual.ln_pre)
        self.ln_post = copy.deepcopy(teacher_visual.ln_post)
        self.proj = None if teacher_visual.proj is None else nn.Parameter(
            teacher_visual.proj.detach().clone(), requires_grad=not freeze_post)
        self.output_dim = teacher_visual.output_dim

        if freeze_stem:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.ln_pre.parameters():
                p.requires_grad = False
        if freeze_post:
            for p in self.ln_post.parameters():
                p.requires_grad = False

        self.use_shallow_cnn = use_shallow_cnn
        if use_shallow_cnn:
            self.shallow = VisionShallowCNN(dim=width, layers=shallow_layers, bottleneck=shallow_bottleneck)

        self.trunk = _StitchedTrunk(gene_blocks, sel, width, adapter_bottleneck,
                                    use_full_stitch, stitch_rank, stitch_at_head, n_total)

    def forward(self, x, return_hidden=False):
        x = self.conv1(x)                              # [B, D, H, W]
        if self.use_shallow_cnn:
            x = self.shallow(x)
        B, D, H, W = x.shape
        x = x.reshape(B, D, H * W).permute(0, 2, 1)    # [B, HW, D]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, D, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                 # [B, 1+HW, D]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                          # [L, B, D]
        out = self.trunk.run(x, return_hidden=return_hidden)
        x, hidden = out if return_hidden else (out, None)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return (x, hidden) if return_hidden else x


class StudentTextEncoder(nn.Module):
    def __init__(self, teacher_clip, gene_blocks: List[nn.Module], sel: List[int], width: int,
                 adapter_bottleneck: int = 64, use_shallow_cnn: bool = False,
                 shallow_layers: int = 1, shallow_bottleneck: int = 64,
                 use_full_stitch: bool = True, stitch_rank: int = 32, stitch_at_head: bool = False,
                 n_total: int = 12, freeze_embed: bool = True, freeze_head: bool = True):
        super().__init__()
        self.token_embedding = copy.deepcopy(teacher_clip.token_embedding)
        self.positional_embedding = nn.Parameter(teacher_clip.positional_embedding.detach().clone(),
                                                 requires_grad=not freeze_embed)
        self.ln_final = copy.deepcopy(teacher_clip.ln_final)
        self.text_projection = nn.Parameter(teacher_clip.text_projection.detach().clone(),
                                            requires_grad=not freeze_head)
        attn_mask = getattr(getattr(teacher_clip, "transformer", None), "attn_mask", None)
        if attn_mask is None:
            attn_mask = getattr(teacher_clip, "attn_mask", None)
        assert attn_mask is not None, "teacher 须提供文本 attn_mask"
        self.register_buffer("attn_mask", attn_mask.detach().clone().float(), persistent=False)

        if freeze_embed:
            for p in self.token_embedding.parameters():
                p.requires_grad = False
        if freeze_head:
            for p in self.ln_final.parameters():
                p.requires_grad = False

        self.use_shallow_cnn = use_shallow_cnn
        if use_shallow_cnn:
            self.shallow = TextShallowCNN(dim=width, layers=shallow_layers, bottleneck=shallow_bottleneck, k=3)

        self.trunk = _StitchedTrunk(gene_blocks, sel, width, adapter_bottleneck,
                                    use_full_stitch, stitch_rank, stitch_at_head, n_total)

    def forward(self, tokens, return_hidden=False):
        x = self.token_embedding(tokens).type(self.positional_embedding.dtype)  # [B,T,D]
        x = x + self.positional_embedding
        if self.use_shallow_cnn:
            x = x.permute(0, 2, 1).contiguous()        # [B,D,T]
            x = self.shallow(x)
            x = x.permute(0, 2, 1)                      # [B,T,D]
        x = x.permute(1, 0, 2)                          # [T,B,D]
        out = self.trunk.run(x, attn_mask=self.attn_mask, return_hidden=return_hidden)
        x, hidden = out if return_hidden else (out, None)
        x = x.permute(1, 0, 2)                          # [B,T,D]
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return (x, hidden) if return_hidden else x


class StudentMiniCLIP(nn.Module):
    def __init__(self, vision_enc: StudentVisionEncoder, text_enc: StudentTextEncoder,
                 logit_scale_init: float, logit_scale_max: float = 100.0):
        super().__init__()
        self.vision = vision_enc
        self.text = text_enc
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        self.logit_scale_max = logit_scale_max

    def encode_image(self, x):
        z = self.vision(x)
        return z / z.norm(dim=-1, keepdim=True)

    def encode_text(self, tokens):
        z = self.text(tokens)
        return z / z.norm(dim=-1, keepdim=True)

    def forward(self, images, tokens):
        zi = self.encode_image(images)
        zt = self.encode_text(tokens)
        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        logits_i = scale * (zi @ zt.t())
        logits_t = logits_i.t()
        return logits_i, logits_t

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
