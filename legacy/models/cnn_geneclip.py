from typing import List, Tuple
import math
import torch
import torch.nn as nn

from models.blocks import (
    VisionShallowCNN, TextShallowCNN, GeneBlockWrapper
)

class StudentVisionEncoder(nn.Module):
    def __init__(self, teacher_visual: nn.Module, visual_gene_blocks: List[nn.Module], width: int,
                 grid_hw: int, adapter_bottleneck: int = 64,
                 shallow_layers: int = 2, shallow_bottleneck: int = 64,
                 freeze_stem: bool = True, freeze_post: bool = True):
        super().__init__()
        # stem & heads from teacher (ViT only)
        self.conv1 = teacher_visual.conv1
        self.class_embedding = nn.Parameter(teacher_visual.class_embedding.detach().clone(), requires_grad=not freeze_stem)
        self.positional_embedding = nn.Parameter(teacher_visual.positional_embedding.detach().clone(), requires_grad=not freeze_stem)
        self.ln_pre = teacher_visual.ln_pre
        self.ln_post = teacher_visual.ln_post
        self.proj = None if teacher_visual.proj is None else nn.Parameter(teacher_visual.proj.detach().clone(), requires_grad=not freeze_post)
        self.output_dim = teacher_visual.output_dim

        if freeze_stem:
            for p in self.conv1.parameters(): p.requires_grad = False
            for p in self.ln_pre.parameters(): p.requires_grad = False
        if freeze_post:
            for p in self.ln_post.parameters(): p.requires_grad = False

        # NEW: shallow CNN before gene blocks (only on patches)
        self.grid_hw = grid_hw  # e.g., 7*7=49
        self.width = width
        self.shallow = VisionShallowCNN(dim=width, layers=shallow_layers, bottleneck=shallow_bottleneck)

        # gene blocks + adapters
        self.blocks = nn.ModuleList([
            GeneBlockWrapper(b, dim=width, adapter_bottleneck=adapter_bottleneck, freeze_gene=True)
            for b in visual_gene_blocks
        ])

    def forward(self, x):
        # conv1 → [N, D, H, W]
        x = self.conv1(x)
        # shallow CNN on patches
        x = self.shallow(x)
        # to tokens
        B, D, H, W = x.shape
        patch = x.view(B, D, H*W).permute(0, 2, 1)    # [B, HW, D]
        cls = self.class_embedding.to(patch.dtype).unsqueeze(0).expand(B, -1, -1)  # [B,1,D]
        x_tok = torch.cat([cls, patch], dim=1)        # [B, 1+HW, D]
        # +pos → ln_pre → LND
        x_tok = x_tok + self.positional_embedding.to(x_tok.dtype)
        x_tok = self.ln_pre(x_tok)
        x_seq = x_tok.permute(1, 0, 2)                # [L, B, D]
        for blk in self.blocks:
            x_seq = blk(x_seq)
        x_tok = x_seq.permute(1, 0, 2)
        x_cls = self.ln_post(x_tok[:, 0, :])
        if self.proj is not None:
            x_cls = x_cls @ self.proj
        return x_cls

    def prox_loss(self) -> torch.Tensor:
        return sum((blk.prox_loss() for blk in self.blocks), torch.tensor(0.0, device=self.positional_embedding.device))

    def unfreeze_last_n(self, n: int):
        for blk in self.blocks[-n:]:
            blk.unfreeze()


class StudentTextEncoder(nn.Module):
    def __init__(self, teacher_clip, text_gene_blocks: List[nn.Module], width: int,
                 adapter_bottleneck: int = 64, shallow_layers: int = 1, shallow_bottleneck: int = 64,
                 freeze_embed: bool = True, freeze_head: bool = True):
        super().__init__()
        # embedding & head from teacher CLIP
        self.token_embedding = teacher_clip.token_embedding
        self.positional_embedding = nn.Parameter(teacher_clip.positional_embedding.detach().clone(), requires_grad=not freeze_embed)
        self.ln_final = teacher_clip.ln_final
        self.text_projection = nn.Parameter(teacher_clip.text_projection.detach().clone(), requires_grad=not freeze_head)
        # attn mask (get or build)
        attn_mask = getattr(getattr(teacher_clip, 'transformer', None), 'attn_mask', None)
        if attn_mask is None:
            attn_mask = getattr(teacher_clip, 'attn_mask', None)
        if attn_mask is None:
            ctx_len = getattr(teacher_clip, 'context_length', None)
            if ctx_len is None:
                ctx_len = int(teacher_clip.positional_embedding.shape[0])
            m = torch.empty(ctx_len, ctx_len)
            m.fill_(float('-inf'))
            m.triu_(1)
            attn_mask = m
        self.register_buffer("attn_mask", attn_mask.clone().to(dtype=torch.float32), persistent=False)

        if freeze_embed:
            for p in self.token_embedding.parameters(): p.requires_grad = False
        if freeze_head:
            for p in self.ln_final.parameters(): p.requires_grad = False

        # NEW: shallow 1D causal CNN before gene blocks
        self.shallow = TextShallowCNN(dim=width, layers=shallow_layers, bottleneck=shallow_bottleneck, k=3)

        # gene blocks + adapters
        self.blocks = nn.ModuleList([
            GeneBlockWrapper(b, dim=width, adapter_bottleneck=adapter_bottleneck, freeze_gene=True)
            for b in text_gene_blocks
        ])

    def forward(self, tokens: torch.Tensor):
        # embeddings + pos → shallow causal CNN (on [B,D,T])
        x = self.token_embedding(tokens).type(self.positional_embedding.dtype)  # [B,T,D]
        x = x + self.positional_embedding
        x_bdt = x.permute(0, 2, 1).contiguous()  # [B,D,T]
        x_bdt = self.shallow(x_bdt)
        # back to sequence for transformer blocks
        x = x_bdt.permute(0, 2, 1)              # [B,T,D]
        x_seq = x.permute(1, 0, 2)              # [T,B,D]
        for blk in self.blocks:
            x_seq = blk(x_seq, attn_mask=self.attn_mask)
        x = x_seq.permute(1, 0, 2)              # [B,T,D]
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return x

    def prox_loss(self) -> torch.Tensor:
        return sum((blk.prox_loss() for blk in self.blocks), torch.tensor(0.0, device=self.positional_embedding.device))

    def unfreeze_last_n(self, n: int):
        for blk in self.blocks[-n:]:
            blk.unfreeze()


class StudentMiniCLIP(nn.Module):
    def __init__(self, vision_enc: StudentVisionEncoder, text_enc: StudentTextEncoder, logit_scale_init: float, logit_scale_max: float = 100.0):
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
    def prox_loss(self) -> torch.Tensor:
        return self.vision.prox_loss() + self.text.prox_loss()
    def unfreeze_gene_layers(self, n_last_v: int, n_last_t: int):
        if n_last_v > 0: self.vision.unfreeze_last_n(n_last_v)
        if n_last_t > 0: self.text.unfreeze_last_n(n_last_t)
