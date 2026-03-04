# models/learngene_loader.py
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import ResidualAttentionBlock
from .tleg import tleg_expand_any_gene

# ------------------------------------------------------------
# functional_call compatibility across PyTorch versions
# ------------------------------------------------------------
# - torch.nn.utils.stateless.functional_call (PyTorch >= 1.13)
# - torch.func.functional_call              (PyTorch >= 2.0)
try:
    from torch.nn.utils.stateless import functional_call as _functional_call  # type: ignore
except Exception:
    try:
        from torch.func import functional_call as _functional_call  # type: ignore
    except Exception:
        _functional_call = None  # type: ignore


def _new_resblock(
    width: int,
    heads: int,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Be compatible with different ResidualAttentionBlock signatures:
      - ResidualAttentionBlock(width, heads, attn_mask=..., device=..., dtype=...)
      - ResidualAttentionBlock(width, heads, attn_mask=...)
      - ResidualAttentionBlock(width, heads)
    """
    # try most-featured signature
    try:
        return ResidualAttentionBlock(width, heads, attn_mask=attn_mask, device=device, dtype=dtype)  # type: ignore
    except TypeError:
        pass
    # try attn_mask only
    try:
        return ResidualAttentionBlock(width, heads, attn_mask=attn_mask)  # type: ignore
    except TypeError:
        pass
    # fallback
    return ResidualAttentionBlock(width, heads)  # type: ignore


def _resolve_gene_ckpt(gene_dir: str, modality: str = "visual") -> str:
    """gene_dir can be a dir or a pt path; choose correct pt based on modality."""
    if os.path.isfile(gene_dir):
        return gene_dir

    m = (modality or "visual").lower()
    if m in ("vision", "visual", "image", "img"):
        fname = "learngene_visual.pt"
    elif m in ("text", "language", "lang"):
        fname = "learngene_text.pt"
    elif m in ("mm", "multi", "multimodal"):
        fname = "learngene_multimodal.pt"
    else:
        raise ValueError(f"Unknown modality={modality}. Use visual/text/multimodal.")

    p = os.path.join(gene_dir, fname)
    if os.path.isfile(p):
        return p

    fallback = [
        os.path.join(gene_dir, f"learngene_{m}.pt"),
        os.path.join(gene_dir, "learngene.pt"),
        os.path.join(gene_dir, "gene.pt"),
    ]
    for fp in fallback:
        if os.path.isfile(fp):
            return fp

    raise FileNotFoundError(
        f"Cannot find gene ckpt under {gene_dir}. Expected {fname} (or fallback {fallback})."
    )


def _pick_prefix_and_token(state: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    """
    Detect where the layer token appears inside keys.
    Support arbitrary prefix, e.g.:
      - visual.transformer.resblocks.11.attn.in_proj_weight
      - transformer.resblocks.11.*
      - blocks.0.*
      - gene.blocks.0.*
    Return: (token, prefix_before_token)
      token in {"resblocks.", "blocks."}
      prefix_before_token is substring before token (can be "")
    """
    counts: Dict[Tuple[str, str], int] = {}
    for k in state.keys():
        for token in ("resblocks.", "blocks."):
            pos = k.find(token)
            if pos != -1:
                pre = k[:pos]
                counts[(token, pre)] = counts.get((token, pre), 0) + 1

    if not counts:
        sample = list(state.keys())[:20]
        raise ValueError(
            "[learngene_loader] Unrecognized gene state_dict keys. "
            "Cannot find 'resblocks.' or 'blocks.' anywhere in keys.\n"
            f"Sample keys (first {len(sample)}):\n  - " + "\n  - ".join(sample)
        )

    (best_token, best_pre), best_cnt = max(counts.items(), key=lambda x: x[1])
    if best_cnt < max(5, int(0.1 * len(state))):
        warnings.warn(
            f"[learngene_loader] Detected token={best_token} prefix='{best_pre}' "
            f"from only {best_cnt}/{len(state)} keys; export may be unusual."
        )
    return best_token, best_pre


def _infer_layer_indices(state: Dict[str, torch.Tensor], token: str, pre: str) -> List[int]:
    """Infer layer indices i from keys containing: pre + token + f'{i}.'"""
    idxs = set()
    base = pre + token
    for k in state.keys():
        pos = k.find(base)
        if pos == -1:
            continue
        rest = k[pos + len(base):]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            idxs.add(int(head))
    if not idxs:
        sample = list(state.keys())[:20]
        raise ValueError(
            f"[learngene_loader] Failed to infer layer indices with token={token} pre='{pre}'.\n"
            f"Sample keys (first {len(sample)}):\n  - " + "\n  - ".join(sample)
        )
    return sorted(idxs)


def _extract_one_layer(state: Dict[str, torch.Tensor], token: str, pre: str, idx: int) -> Dict[str, torch.Tensor]:
    """
    Extract per-layer sd and strip the full prefix:
      pre + token + f"{idx}."
    so keys become "attn.in_proj_weight", "ln_1.weight", ...
    """
    layer_prefix = pre + token + f"{idx}."
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(layer_prefix):
            out[k[len(layer_prefix):]] = v
    return out


def _infer_width_and_heads(
    gene_layers: List[Dict[str, torch.Tensor]],
    meta_width,
    meta_heads
) -> Tuple[int, int]:
    width = meta_width
    heads = meta_heads

    if width is None:
        for sd in gene_layers:
            w = sd.get("attn.in_proj_weight", None)
            if torch.is_tensor(w) and w.ndim == 2:
                width = int(w.shape[1])
                break
        if width is None:
            raise ValueError("[learngene_loader] Cannot infer width from gene layer state_dict.")

    if heads is None:
        heads = width // 64 if width % 64 == 0 else max(1, width // 64)

    return int(width), int(heads)


class LearngeneModule(nn.Module):
    """Apply a list of ResidualAttentionBlock sequentially."""
    def __init__(self, blocks: nn.ModuleList, original_layer_indices: Optional[List[int]] = None):
        super().__init__()
        self.blocks = blocks
        self.original_layer_indices = original_layer_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def set_requires_grad(self, flag: bool = True):
        """Keep compatibility with old code (StudentCLIP expects this)."""
        for p in self.parameters():
            p.requires_grad_(flag)
        return self



class TLEGStrictPiecewiseModule(nn.Module):
    """
    Strict TLEG: execute expanded depth blocks via functional_call on a template block.
    expanded_layers: list of per-block state_dict with keys like 'ln_1.weight', 'attn.in_proj_weight', ...
    """
    def __init__(
        self,
        expanded_layers: List[Dict[str, torch.Tensor]],
        d_model: int,
        n_head: int,
        attn_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if _functional_call is None:
            raise RuntimeError(
                "functional_call is not available. Please use PyTorch >= 2.0 or >= 1.13."
            )
        # NOTE: expanded_layers are usually loaded on CPU (map_location="cpu").
        # In DDP / multi-GPU training, we must ensure the layer state dicts and template
        # live on the same device as the activation tensor `x`.
        self.expanded_layers = expanded_layers
        self.template = _new_resblock(d_model, n_head, attn_mask=attn_mask, device=device, dtype=dtype)

        # cache per-device moved state_dicts (each torchrun rank uses a single GPU)
        self._cached_device: Optional[torch.device] = None
        self._cached_expanded_layers: Optional[List[Dict[str, torch.Tensor]]] = None

    @staticmethod
    def _to_device_sd(sd: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if torch.is_tensor(v):
                out[k] = v.to(device=device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _ensure_cache(self, device: torch.device):
        if self._cached_expanded_layers is not None and self._cached_device == device:
            return
        self._cached_expanded_layers = [self._to_device_sd(sd, device) for sd in self.expanded_layers]
        self._cached_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device

        # Make sure template (and any internal buffers like attn_mask) is on the same device.
        # template has parameters, but be defensive anyway.
        p0 = next(self.template.parameters(), None)
        if p0 is None or p0.device != dev:
            self.template = self.template.to(dev)

        # Move per-layer weights/buffers to the same device as x (once per process/device).
        self._ensure_cache(dev)

        for sd in self._cached_expanded_layers:
            x = _functional_call(self.template, sd, (x,), {})
        return x


    def set_requires_grad(self, flag: bool = True):
        """
        Compatibility hook.

        Note:
        - strict 模式下真正“生效”的权重来自 expanded_layers（普通 Tensor dict，不是 nn.Parameter）
        - 所以训练时通常把 gene 当作 frozen（这和 strict 语义一致）
        - 这里主要是为了不让 StudentCLIP 调用时报错
        """
        for p in self.parameters():
            p.requires_grad_(flag)
        return self



def load_learngene_variant(
    variant_dir: str,
    modality: str = "visual",
    use_tleg: bool = False,
    tleg_target_depth: int = 0,
    tleg_strict: bool = False,
    piecewise: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, int, int]:
    """
    StudentCLIP 期望接口：返回 (gene_module, width, heads)

    兼容 export：
      - keys 可能是 blocks.<i>.* 或 resblocks.<i>.*
      - 也可能带任意前缀：visual.transformer.resblocks.<i>.* 等
    """
    pt_path = _resolve_gene_ckpt(variant_dir, modality=modality)
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"[learngene_loader] invalid gene export at {pt_path}: {type(obj)}")

    state = obj.get("state_dict", None) or obj.get("model", None) or obj.get("state", None)
    if state is None:
        if all(isinstance(k, str) for k in obj.keys()):
            state = obj
        else:
            raise KeyError(f"[learngene_loader] cannot find state_dict in {pt_path}")
    if not isinstance(state, dict):
        raise ValueError(f"[learngene_loader] state_dict is not a dict in {pt_path}")

    layers = obj.get("layers", None) or obj.get("layer_ids", None) or obj.get("layer_indices", None)

    token, pre = _pick_prefix_and_token(state)

    if layers is None:
        layers = _infer_layer_indices(state, token, pre)
    else:
        layers = [int(x) for x in list(layers)]

    gene_layers: List[Dict[str, torch.Tensor]] = []
    for idx in layers:
        sd_i = _extract_one_layer(state, token, pre, idx)
        if not sd_i:
            raise ValueError(
                f"[learngene_loader] Extracted empty layer sd for idx={idx} "
                f"using token={token} pre='{pre}'. Export keys may be inconsistent."
            )
        gene_layers.append(sd_i)

    meta_width = obj.get("width", None) or obj.get("d_model", None) or obj.get("embed_dim", None) or obj.get("hidden_size", None)
    meta_heads = obj.get("heads", None) or obj.get("n_head", None) or obj.get("num_heads", None)
    width, heads = _infer_width_and_heads(gene_layers, meta_width, meta_heads)

    # materialize base blocks
    base_blocks = nn.ModuleList([
        _new_resblock(width, heads, attn_mask=attn_mask, device=device, dtype=dtype)
        for _ in range(len(gene_layers))
    ])
    for blk, sd_i in zip(base_blocks, gene_layers):
        blk.load_state_dict(sd_i, strict=True)

    expanded_layers = gene_layers
    if use_tleg:
        if tleg_target_depth <= 0:
            raise ValueError("use_tleg=True but tleg_target_depth not set (>0).")
        if tleg_target_depth > len(gene_layers):
            expanded_layers = tleg_expand_any_gene(
                gene_layers,
                target_depth=tleg_target_depth,
                layer_ids=[int(x) for x in layers],
                strict_keys=tleg_strict,
            )

    if use_tleg and tleg_target_depth > len(gene_layers):
        if tleg_strict:
            gene = TLEGStrictPiecewiseModule(
                expanded_layers=expanded_layers,
                d_model=width,
                n_head=heads,
                attn_mask=attn_mask,
                device=device,
                dtype=dtype,
            )
        else:
            exp_blocks = nn.ModuleList([
                _new_resblock(width, heads, attn_mask=attn_mask, device=device, dtype=dtype)
                for _ in range(len(expanded_layers))
            ])
            for blk, sd_i in zip(exp_blocks, expanded_layers):
                blk.load_state_dict(sd_i, strict=True)
            gene = LearngeneModule(exp_blocks, original_layer_indices=[int(x) for x in layers])
    else:
        gene = LearngeneModule(base_blocks, original_layer_indices=[int(x) for x in layers])

    return gene, width, heads


def load_multimodal_state(variant_dir: str) -> Dict[str, torch.Tensor]:
    """Load multimodal gene state (used by multimodal init)."""
    pt_path = _resolve_gene_ckpt(variant_dir, modality="multimodal")
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise ValueError(f"Unsupported multimodal checkpoint format: {type(obj)}")
