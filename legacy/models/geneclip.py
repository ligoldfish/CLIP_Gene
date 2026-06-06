from typing import List, Dict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 参数高效：Residual Adapter ----
class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, drop: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=1e-4)

    def forward(self, x):
        return x + self.drop(self.up(self.act(self.down(x))))


# ---- 基因层包装：冻结Block + Adapter + Prox快照 ----
class GeneBlockWrapper(nn.Module):
    def __init__(self, gene_block: nn.Module, dim: int, bottleneck: int = 64, drop: float = 0.0, freeze_gene=True):
        super().__init__()
        self.gene_block = gene_block
        if freeze_gene:
            for p in self.gene_block.parameters():
                p.requires_grad = False
        # 保存初始快照用于 Prox 正则
        self._gene_init = {n: p.detach().clone() for n, p in self.gene_block.named_parameters()}
        self.adapter = ResidualAdapter(dim, bottleneck=bottleneck, drop=drop)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            y = self.gene_block(x, attn_mask=attn_mask)
        else:
            y = self.gene_block(x)
        y = self.adapter(y)
        return y

    def prox_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for (n, p) in self.gene_block.named_parameters():
            init = self._gene_init[n].to(p.device)
            loss = loss + (p - init).pow(2).sum()
        return loss

    def unfreeze(self):
        for p in self.gene_block.parameters():
            p.requires_grad = True


class StudentVisionEncoder(nn.Module):
    def __init__(self, teacher_visual: nn.Module, visual_gene_blocks: List[nn.Module], width: int,
                 adapter_bottleneck: int = 64, freeze_stem: bool = True, freeze_post: bool = True):
        super().__init__()
        # Stem/Head 复制（可冻结）
        self.conv1 = copy.deepcopy(teacher_visual.conv1)
        self.class_embedding = nn.Parameter(teacher_visual.class_embedding.detach().clone(), requires_grad=not freeze_stem)
        self.positional_embedding = nn.Parameter(teacher_visual.positional_embedding.detach().clone(), requires_grad=not freeze_stem)
        self.ln_pre = copy.deepcopy(teacher_visual.ln_pre)
        self.ln_post = copy.deepcopy(teacher_visual.ln_post)
        self.proj = None if teacher_visual.proj is None else nn.Parameter(teacher_visual.proj.detach().clone(), requires_grad=not freeze_post)
        self.output_dim = teacher_visual.output_dim

        if freeze_stem:
            for p in self.conv1.parameters(): p.requires_grad = False
            for p in self.ln_pre.parameters(): p.requires_grad = False
        if freeze_post:
            for p in self.ln_post.parameters(): p.requires_grad = False

        # 基因层 + Adapter
        self.blocks = nn.ModuleList([
            GeneBlockWrapper(copy.deepcopy(b), dim=width, bottleneck=adapter_bottleneck, freeze_gene=True)
            for b in visual_gene_blocks
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def prox_loss(self) -> torch.Tensor:
        return sum((blk.prox_loss() for blk in self.blocks), torch.tensor(0.0, device=self.class_embedding.device))

    def unfreeze_last_n(self, n: int):
        for blk in self.blocks[-n:]:
            blk.unfreeze()


class StudentTextEncoder(nn.Module):
    def __init__(self, teacher_clip, text_gene_blocks: List[nn.Module], width: int,
                 adapter_bottleneck: int = 64, freeze_embed: bool = True, freeze_head: bool = True):
        super().__init__()
        # 这些模块在 OpenAI CLIP 顶层（不是 transformer 内部）
        self.token_embedding = copy.deepcopy(teacher_clip.token_embedding)
        self.positional_embedding = nn.Parameter(teacher_clip.positional_embedding.detach().clone(), requires_grad=not freeze_embed)
        self.ln_final = copy.deepcopy(teacher_clip.ln_final)
        self.text_projection = nn.Parameter(teacher_clip.text_projection.detach().clone(), requires_grad=not freeze_head)
        # 注意力 mask 位置在 transformer 或顶层，做兼容获取
        attn_mask = getattr(teacher_clip.transformer, 'attn_mask', None)
        if attn_mask is None:
            attn_mask = getattr(teacher_clip, 'attn_mask', None)
        assert attn_mask is not None, "Teacher clip should provide an attn_mask for text"
        self.register_buffer("attn_mask", attn_mask.clone(), persistent=False)

        if freeze_embed:
            for p in self.token_embedding.parameters(): p.requires_grad = False
        if freeze_head:
            for p in self.ln_final.parameters(): p.requires_grad = False

        self.blocks = nn.ModuleList([
            GeneBlockWrapper(copy.deepcopy(b), dim=width, bottleneck=adapter_bottleneck, freeze_gene=True)
            for b in text_gene_blocks
        ])

    def forward(self, tokens: torch.Tensor):
        x = self.token_embedding(tokens).type(self.positional_embedding.dtype)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        for blk in self.blocks:
            x = blk(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        # 取 EOT 位置（与 OpenAI CLIP 一致）
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return x

    def prox_loss(self) -> torch.Tensor:
        return sum((blk.prox_loss() for blk in self.blocks), torch.tensor(0.0, device=self.positional_embedding.device))

    def unfreeze_last_n(self, n: int):
        for blk in self.blocks[-n:]:
            blk.unfreeze()


class StudentMiniCLIP(nn.Module):
    def __init__(self, vision_enc: StudentVisionEncoder, text_enc: StudentTextEncoder, logit_scale_init: float):
        super().__init__()
        self.vision = vision_enc
        self.text = text_enc
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def encode_image(self, x):
        z = self.vision(x)
        return z / z.norm(dim=-1, keepdim=True)

    def encode_text(self, tokens):
        z = self.text(tokens)
        return z / z.norm(dim=-1, keepdim=True)

    def forward(self, images, tokens):
        zi = self.encode_image(images)
        zt = self.encode_text(tokens)
        scale = self.logit_scale.exp()
        logits_i = scale * (zi @ zt.t())
        logits_t = logits_i.t()
        return logits_i, logits_t

    def prox_loss(self) -> torch.Tensor:
        return self.vision.prox_loss() + self.text.prox_loss()

    def unfreeze_gene_layers(self, n_last_v: int, n_last_t: int):
        if n_last_v > 0:
            self.vision.unfreeze_last_n(n_last_v)
        if n_last_t > 0:
            self.text.unfreeze_last_n(n_last_t)
