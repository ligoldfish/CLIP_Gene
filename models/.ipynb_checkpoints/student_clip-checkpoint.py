# models/student_clip.py
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    LayerNormFP32,
    ResidualAttentionBlock,
    BottleneckMLP,
    ProjectionHead,
    build_causal_attention_mask,
)
from .learngene_loader import load_learngene_variant, load_multimodal_state


@dataclass
class StudentCLIPConfig:
    # I/O sizes
    image_resolution: int = 224
    patch_size: int = 32
    context_length: int = 77
    vocab_size: int = 49408

    # architecture
    shallow_layers: int = 2
    bottleneck_dim: Optional[int] = None
    bottleneck_dropout: float = 0.0

    # projection (shared low-dim)
    # Match CLIP's embedding dimension by default
    proj_dim: int = 512
    # Use CLIP-style linear projection by default (no MLP)
    proj_use_mlp: bool = False
    proj_hidden_dim: Optional[int] = None
    proj_dropout: float = 0.0

    # learngene loading
    gene_variant_dir: str = ""
    use_tleg: bool = False
    tleg_target_depth: int = 2
    tleg_strict: bool = False  # NEW: strict paper-faithful TLEG

    # freezing behavior (for training phases)
    freeze_learngene: bool = True

    # init / misc
    init_logit_scale: float = 1 / 0.07
    use_multimodal_init: bool = False  # try init proj/logit_scale from learngene_multimodal.pt

    # optional: CLIP stem initialization (copy from teacher CLIP; init-only)
    stem_init_from_clip: bool = False
    stem_init_clip_name: str = "ViT-B/32"
    freeze_stem_after_init: bool = False


class VisionStem(nn.Module):
    """Mini ViT stem: patch embedding + class token + pos embedding."""
    def __init__(self, width: int, image_resolution: int, patch_size: int):
        super().__init__()
        assert image_resolution % patch_size == 0
        self.grid = image_resolution // patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(self.grid * self.grid + 1, width))
        self.ln_pre = LayerNormFP32(width)

        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.class_embedding, std=0.01)

    def forward(self, x: torch.Tensor):
        # x: [B,3,H,W]
        x = self.conv1(x)  # [B,width,grid,grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B,width,grid^2]
        x = x.permute(0, 2, 1)  # [B,grid^2,width]
        cls = self.class_embedding.to(x.dtype)[None, None, :].expand(x.shape[0], 1, -1)  # [B,1,width]
        x = torch.cat([cls, x], dim=1)  # [B,1+grid^2,width]
        x = x + self.positional_embedding.to(x.dtype)[None, :, :]
        x = self.ln_pre(x)
        return x.permute(1, 0, 2)  # [seq,B,width]


class TextStem(nn.Module):
    """Token embedding + pos embedding."""
    def __init__(self, width: int, vocab_size: int, context_length: int):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, text: torch.Tensor):
        # text: [B, L]
        x = self.token_embedding(text)  # [B,L,width]
        x = x + self.positional_embedding.to(x.dtype)[None, :, :]
        return x.permute(1, 0, 2)  # [L,B,width]


class Tower(nn.Module):
    """
    A tower = shallow transformer blocks + bottleneck + learngene + ln_post + proj head.
    NOTE: width/head are per-tower (vision/text can differ).
    """
    def __init__(
        self,
        width: int,
        n_head: int,
        shallow_layers: int,
        bottleneck_dim: Optional[int],
        bottleneck_dropout: float,
        gene_module: nn.Module,
        is_text: bool,
        attn_mask: Optional[torch.Tensor],
        proj_dim: int,
        proj_use_mlp: bool,
        proj_hidden_dim: Optional[int],
        proj_dropout: float,
    ):
        super().__init__()
        self.width = width
        self.n_head = n_head
        self.is_text = is_text

        self.shallow = nn.ModuleList([
            ResidualAttentionBlock(d_model=width, n_head=n_head, attn_mask=attn_mask if is_text else None)
            for _ in range(shallow_layers)
        ])

        self.bottleneck = BottleneckMLP(d_model=width, bottleneck=bottleneck_dim, dropout=bottleneck_dropout)
        self.gene = gene_module

        self.ln_post = LayerNormFP32(width)
        self.proj = ProjectionHead(
            in_dim=width,
            out_dim=proj_dim,
            use_mlp=proj_use_mlp,
            hidden_dim=proj_hidden_dim,
            dropout=proj_dropout,
        )

    def forward_tokens(self, x_tokens: torch.Tensor, return_aux: bool = False):
        """Forward token sequence through shallow -> bottleneck -> gene."""
        shallow_outs = []
        for blk in self.shallow:
            x_tokens = blk(x_tokens)
            if return_aux:
                shallow_outs.append(x_tokens)

        x_pre_gene = self.bottleneck(x_tokens)
        x_post_gene = self.gene(x_pre_gene)
        x_post = self.ln_post(x_post_gene)

        if return_aux:
            aux = {
                "shallow": shallow_outs,
                "pre_gene": x_pre_gene,
                "post_gene": x_post_gene,
            }
            return x_post, aux
        return x_post

    def pool(self, x_tokens: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return pooled embedding:
          - vision: CLS token (token 0)
          - text: EOT token (argmax token id, as CLIP does)
        """
        if not self.is_text:
            pooled = x_tokens[0]  # [B,width]
        else:
            assert text is not None
            x = x_tokens.permute(1, 0, 2)  # [B,L,width]
            eot_idx = text.argmax(dim=-1)
            pooled = x[torch.arange(x.shape[0], device=x.device), eot_idx]  # [B,width]
        return pooled

    def forward(self, x_tokens: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_tokens = self.forward_tokens(x_tokens)
        pooled = self.pool(x_tokens, text=text)
        z = self.proj(pooled)
        z = F.normalize(z.float(), dim=-1, eps=1e-6).to(z.dtype)
        return z

    def forward_with_aux(self, x_tokens: torch.Tensor, text: Optional[torch.Tensor] = None):
        x_tokens, aux = self.forward_tokens(x_tokens, return_aux=True)
        pooled = self.pool(x_tokens, text=text)
        z = self.proj(pooled)
        z = F.normalize(z.float(), dim=-1, eps=1e-6).to(z.dtype)
        return z, aux


class StudentCLIP(nn.Module):
    """
    Dual-tower student model:
      vision:  stem_v -> shallow_v -> bottleneck -> learngene_v -> proj_v -> shared space
      text:    stem_t -> shallow_t -> bottleneck -> learngene_t -> proj_t -> shared space
    widths can differ (e.g., ViT-B/32: vision=768, text=512).
    """
    def __init__(self, cfg: StudentCLIPConfig, device: Optional[str] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert cfg.gene_variant_dir, "cfg.gene_variant_dir is required"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        self.dtype = dtype
        self.cfg = cfg
        # Expose shallow depth for init helpers / logging.
        # (Some init routines reference self.shallow_layers.)
        self.shallow_layers = cfg.shallow_layers

        # text causal mask
        text_mask = build_causal_attention_mask(cfg.context_length, device=dev)

        # Load learngene for each tower separately (widths may differ)
        gene_v, width_v, head_v = load_learngene_variant(
            cfg.gene_variant_dir,
            modality="visual",
            use_tleg=cfg.use_tleg,
            tleg_target_depth=cfg.tleg_target_depth,
            tleg_strict=getattr(cfg, "tleg_strict", False),
            attn_mask=None,
            device=dev,
            dtype=dtype,
        )
        gene_t, width_t, head_t = load_learngene_variant(
            cfg.gene_variant_dir,
            modality="text",
            use_tleg=cfg.use_tleg,
            tleg_target_depth=cfg.tleg_target_depth,
            tleg_strict=getattr(cfg, "tleg_strict", False),
            attn_mask=text_mask,
            device=dev,
            dtype=dtype,
        )

        self.width_v, self.head_v = width_v, head_v
        self.width_t, self.head_t = width_t, head_t

        # Stems
        self.vision_stem = VisionStem(width_v, cfg.image_resolution, cfg.patch_size).to(dev, dtype=dtype)
        self.text_stem = TextStem(width_t, cfg.vocab_size, cfg.context_length).to(dev, dtype=dtype)

        # Towers
        self.vision_tower = Tower(
            width=width_v,
            n_head=head_v,
            shallow_layers=cfg.shallow_layers,
            bottleneck_dim=cfg.bottleneck_dim,
            bottleneck_dropout=cfg.bottleneck_dropout,
            gene_module=gene_v,
            is_text=False,
            attn_mask=None,
            proj_dim=cfg.proj_dim,
            proj_use_mlp=cfg.proj_use_mlp,
            proj_hidden_dim=cfg.proj_hidden_dim,
            proj_dropout=cfg.proj_dropout,
        ).to(dev, dtype=dtype)

        self.text_tower = Tower(
            width=width_t,
            n_head=head_t,
            shallow_layers=cfg.shallow_layers,
            bottleneck_dim=cfg.bottleneck_dim,
            bottleneck_dropout=cfg.bottleneck_dropout,
            gene_module=gene_t,
            is_text=True,
            attn_mask=text_mask,
            proj_dim=cfg.proj_dim,
            proj_use_mlp=cfg.proj_use_mlp,
            proj_hidden_dim=cfg.proj_hidden_dim,
            proj_dropout=cfg.proj_dropout,
        ).to(dev, dtype=dtype)

        # optional: initialize stems / embeddings / post-LNs from a teacher CLIP (copy init only)
        if getattr(cfg, "stem_init_from_clip", False):
            self._init_from_clip_teacher(
                clip_name=getattr(cfg, "stem_init_clip_name", "ViT-B/32"),
                device=dev,
                dtype=dtype,
                freeze=getattr(cfg, "freeze_stem_after_init", False),
            )

        # logit scale (shared)
        init_ls = math.log(cfg.init_logit_scale)
        self.logit_scale = nn.Parameter(torch.tensor(init_ls, device=dev, dtype=dtype))

        # optional init from multimodal file
        if cfg.use_multimodal_init:
            mm = load_multimodal_state(cfg.gene_variant_dir)
            if "logit_scale" in mm:
                with torch.no_grad():
                    self.logit_scale.copy_(mm["logit_scale"].to(dev, dtype=dtype))

            # Init projection heads only if:
            # - you use linear head (proj_use_mlp=False)
            # - shapes align
            if not cfg.proj_use_mlp:
                if "text_projection" in mm:
                    tp = mm["text_projection"].to(dev, dtype=dtype)  # [width_t, embed_dim]
                    # our linear weight: [proj_dim, width_t]  (out,in)
                    if hasattr(self.text_tower.proj, "proj") and self.text_tower.proj.proj.weight.shape == tp.T.shape:
                        with torch.no_grad():
                            self.text_tower.proj.proj.weight.copy_(tp.T)

                if "visual.proj" in mm:
                    vp = mm["visual.proj"].to(dev, dtype=dtype)  # [width_v, embed_dim]
                    if hasattr(self.vision_tower.proj, "proj") and self.vision_tower.proj.proj.weight.shape == vp.T.shape:
                        with torch.no_grad():
                            self.vision_tower.proj.proj.weight.copy_(vp.T)

        # freeze gene if requested
        if cfg.freeze_learngene:
            self.vision_tower.gene.set_requires_grad(False)
            self.text_tower.gene.set_requires_grad(False)

    def _init_from_clip_teacher(self, clip_name: str, device: torch.device, dtype: torch.dtype, freeze: bool = False):
        """Copy CLIP stem / embeddings / post-LNs from a teacher CLIP model (init-only)."""
        try:
            import clip  # OpenAI CLIP
        except Exception as e:
            print(f"[StudentCLIP] clip import failed; skip stem init. err={e}")
            return

        teacher, _ = clip.load(clip_name, device="cpu", jit=False)
        teacher.eval()

        def _copy_param(dst: torch.Tensor, src: torch.Tensor, name: str):
            if dst.shape != src.shape:
                print(f"[StudentCLIP] shape mismatch for {name}: dst={tuple(dst.shape)} src={tuple(src.shape)}; skip.")
                return
            dst.copy_(src.to(device=device, dtype=dtype))

        with torch.no_grad():
            # ---- vision stem ----
            _copy_param(self.vision_stem.conv1.weight, teacher.visual.conv1.weight, "visual.conv1.weight")
            _copy_param(self.vision_stem.class_embedding, teacher.visual.class_embedding, "visual.class_embedding")
            _copy_param(self.vision_stem.positional_embedding, teacher.visual.positional_embedding, "visual.positional_embedding")
            _copy_param(self.vision_stem.ln_pre.weight, teacher.visual.ln_pre.weight, "visual.ln_pre.weight")
            _copy_param(self.vision_stem.ln_pre.bias, teacher.visual.ln_pre.bias, "visual.ln_pre.bias")

            # ---- vision post LN ----
            if hasattr(teacher.visual, "ln_post") and hasattr(self.vision_tower, "ln_post"):
                _copy_param(self.vision_tower.ln_post.weight, teacher.visual.ln_post.weight, "visual.ln_post.weight")
                _copy_param(self.vision_tower.ln_post.bias, teacher.visual.ln_post.bias, "visual.ln_post.bias")

            # ---- text stem ----
            _copy_param(self.text_stem.token_embedding.weight, teacher.token_embedding.weight, "token_embedding.weight")
            _copy_param(self.text_stem.positional_embedding, teacher.positional_embedding, "positional_embedding")

            # ---- text post LN ----
            if hasattr(teacher, "ln_final") and hasattr(self.text_tower, "ln_post"):
                _copy_param(self.text_tower.ln_post.weight, teacher.ln_final.weight, "ln_final.weight")
                _copy_param(self.text_tower.ln_post.bias, teacher.ln_final.bias, "ln_final.bias")

            # ---- strengthen text tower init (keep depth unchanged) ----
            # Initialize student text shallow blocks from teacher CLIP transformer blocks
            try:
                if hasattr(teacher, 'transformer') and hasattr(teacher.transformer, 'resblocks'):
                    src_blocks = list(teacher.transformer.resblocks)
                    dst_blocks = list(self.text_tower.shallow)  # ModuleList
                    n = min(self.cfg.shallow_layers, len(dst_blocks), len(src_blocks))
                    for i in range(n):
                        # state_dict keys are compatible between CLIP and our ResidualAttentionBlock
                        dst_blocks[i].load_state_dict(src_blocks[i].state_dict(), strict=False)
            except Exception as e:
                print(f'[StudentCLIP] text shallow init from CLIP failed; skip. err={e.__class__.__name__}: {e}')

            # ---- strengthen vision tower init (keep depth unchanged) ----
            # Initialize student vision shallow blocks from teacher CLIP visual transformer blocks
            try:
                if hasattr(teacher, 'visual') and hasattr(teacher.visual, 'transformer') and hasattr(teacher.visual.transformer, 'resblocks'):
                    src_blocks = list(teacher.visual.transformer.resblocks)
                    dst_blocks = list(self.vision_tower.shallow)  # ModuleList
                    n_copy = min(self.cfg.shallow_layers, len(src_blocks), len(dst_blocks))
                    if n_copy > 0:
                        for i in range(n_copy):
                            dst_blocks[i].load_state_dict(src_blocks[i].state_dict(), strict=False)
            except Exception as e:
                print(f'[StudentCLIP] vision shallow init from CLIP failed; skip. err={e.__class__.__name__}: {e}')

            # Initialize CLIP-style linear projections when shapes match
            def _copy_linear_proj(dst_head, src_proj, name: str):
                """Copy CLIP projection params into our linear projection head.

                - CLIP uses a Parameter shaped [in_dim, out_dim] (e.g., [width, embed_dim]).
                - nn.Linear stores weights as [out_dim, in_dim].
                This helper supports:
                  * ProjectionHead(linear) with .proj (nn.Linear)
                  * legacy heads with .net (nn.Linear)
                  * passing an nn.Linear directly
                """
                try:
                    if src_proj is None:
                        return

                    # find dst linear
                    net = None
                    if isinstance(dst_head, nn.Linear):
                        net = dst_head
                    else:
                        cand = getattr(dst_head, 'proj', None)
                        if isinstance(cand, nn.Linear):
                            net = cand
                        else:
                            cand = getattr(dst_head, 'net', None)
                            if isinstance(cand, nn.Linear):
                                net = cand

                    if net is None:
                        return

                    sp = src_proj.detach().to(device='cpu')
                    if sp.ndim != 2:
                        print(f'[StudentCLIP] skip {name}: src_proj ndim={sp.ndim} (expect 2)')
                        return

                    # copy with best-effort orientation
                    w = net.weight
                    if tuple(w.shape) == tuple(sp.shape):
                        w.copy_(sp.to(dtype=w.dtype))
                        if net.bias is not None:
                            net.bias.zero_()
                        return

                    # CLIP [in_dim,out_dim] -> Linear [out_dim,in_dim]
                    if tuple(w.shape) == (sp.shape[1], sp.shape[0]):
                        w.copy_(sp.t().to(dtype=w.dtype))
                        if net.bias is not None:
                            net.bias.zero_()
                        return

                    print(f'[StudentCLIP] shape mismatch for {name}: dst={tuple(w.shape)} src={tuple(sp.shape)}; skip.')
                except Exception as e:
                    print(f'[StudentCLIP] copy {name} failed; skip. err={e.__class__.__name__}: {e}')

            if hasattr(teacher, 'text_projection') and hasattr(self.text_tower, 'proj'):
                _copy_linear_proj(self.text_tower.proj, teacher.text_projection, 'text_projection')
            if hasattr(teacher, 'visual') and hasattr(teacher.visual, 'proj') and hasattr(self.vision_tower, 'proj'):
                _copy_linear_proj(self.vision_tower.proj, teacher.visual.proj, 'vision_projection')
            if hasattr(teacher, 'logit_scale') and hasattr(self, 'logit_scale'):
                _copy_param(self.logit_scale, teacher.logit_scale, 'logit_scale')

        if freeze:
            for p in self.vision_stem.parameters():
                p.requires_grad = False
            for p in self.text_stem.parameters():
                p.requires_grad = False
            for p in self.vision_tower.ln_post.parameters():
                p.requires_grad = False
            for p in self.text_tower.ln_post.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens
        v_tokens = self.vision_stem(image.to(next(self.vision_stem.parameters()).device, dtype=self.dtype))
        t_tokens = self.text_stem(text.to(next(self.text_stem.parameters()).device))

        # embeddings in shared low-dim space
        z_img = self.vision_tower(v_tokens)              # [B, proj_dim]
        z_txt = self.text_tower(t_tokens, text=text)     # [B, proj_dim]

        # logits
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * (z_img @ z_txt.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
