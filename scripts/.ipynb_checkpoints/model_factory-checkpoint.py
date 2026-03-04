# scripts/model_factory.py
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn


class TokenizeCache:
    def __init__(self, tokenize_fn: Callable, max_size: int = 20000):
        self.fn = tokenize_fn
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.order: List[str] = []

    def __call__(self, texts: List[str]) -> torch.Tensor:
        outs = []
        for t in texts:
            if t in self.cache:
                outs.append(self.cache[t])
            else:
                tok = self.fn([t])[0]  # [ctx]
                self.cache[t] = tok
                self.order.append(t)
                if len(self.order) > self.max_size:
                    old = self.order.pop(0)
                    self.cache.pop(old, None)
                outs.append(tok)
        return torch.stack(outs, dim=0)


@dataclass
class ModelBundle:
    model: nn.Module
    tokenize: Callable[[List[str]], torch.Tensor]


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


def load_tinyclip_wkcn(pt_path: str, device: str) -> nn.Module:
    """
    Load wkcn/TinyCLIP .pt checkpoint using OpenAI-CLIP build_model(state_dict).
    """
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt.get("state_dict", None) or ckpt.get("model", None) or ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected ckpt type: {type(sd)}")
    sd = _strip_prefix(sd, "module.")

    import clip
    from clip.model import build_model
    model = build_model(sd).to(device)
    model.eval()
    return model


def create_model_bundle(args) -> ModelBundle:
    device = args.device

    if args.model == "ours":
        # IMPORTANT: this matches your traceback path: /root/codes/models/student_clip.py
        from models.student_clip import StudentCLIP  # noqa

        def _infer_patch_size(name: str) -> int:
            # Examples: "ViT-B/32" -> 32, "ViT-L/14" -> 14
            if not name:
                return 32
            if "/16" in name:
                return 16
            if "/14" in name:
                return 14
            if "/32" in name:
                return 32
            return 32

        # pick a CLIP name to infer ViT patch size (prefer teacher used for stem init)
        _ref_clip_name = getattr(args, "stem_init_clip_name", "") or getattr(args, "clip_name", "ViT-B/32")
        _image_size = int(getattr(args, "image_size", getattr(args, "image_resolution", 224)) or 224)

        # Build cfg with attribute access (Namespace/dataclass style)
        cfg_dict = {
            # ---- CLIP defaults (MUST exist for StudentCLIP) ----
            # If you use OpenAI CLIP ViT-B/32, the common defaults are:
            # context_length=77, vocab_size=49408, image_resolution=224
            "image_resolution": _image_size,
            "patch_size": _infer_patch_size(_ref_clip_name),
            "context_length": 77,
            "vocab_size": 49408,
            "init_logit_scale": 1 / 0.07,

            # your gene folder path (now points to outputs/lg_clip_adapter_rho)
            "gene_variant_dir": args.gene_dir,

            # architecture knobs (safe defaults)
            "shallow_layers": args.shallow_layers,
            "bottleneck_dim": None if args.bottleneck_dim <= 0 else args.bottleneck_dim,
            "bottleneck_dropout": args.bottleneck_dropout,

            "proj_dim": args.proj_dim,
            "proj_use_mlp": (args.proj_head == "mlp"),
            "proj_hidden_dim": None if args.proj_hidden_dim <= 0 else args.proj_hidden_dim,
            "proj_dropout": args.proj_dropout,

            # TLEG
            "use_tleg": args.use_tleg,
            "tleg_target_depth": args.tleg_target_depth,
            "tleg_strict": getattr(args, "tleg_strict", False),

            # init options
            "use_multimodal_init": args.use_multimodal_init,


            # CLIP stem initialization (copy vision/text stems + ln_post from teacher CLIP)
            "stem_init_from_clip": bool(getattr(args, "stem_init_from_clip", (not getattr(args, "disable_stem_init_from_clip", False)))),
            "stem_init_clip_name": getattr(args, "stem_init_clip_name", "ViT-B/32"),
            "freeze_stem_after_init": bool(getattr(args, "freeze_stem_after_init", False)),

            # start frozen; training loop will unfreeze later
            "freeze_learngene": True,
        }

        # If your project defines a StudentCLIPConfig dataclass, use it; otherwise fallback to SimpleNamespace
        try:
            from models.student_clip import StudentCLIPConfig  # type: ignore
            cfg = StudentCLIPConfig(**cfg_dict)  # dataclass init
        except Exception:
            cfg = SimpleNamespace(**cfg_dict)

        model = StudentCLIP(cfg, device=device).to(device)

        import clip
        tokenize = TokenizeCache(lambda x: clip.tokenize(x, truncate=True))
        return ModelBundle(model=model, tokenize=tokenize)

    if args.model == "clip":
        import clip
        model, _ = clip.load(args.clip_name, device=device, jit=False)
        model.eval()
        tokenize = TokenizeCache(lambda x: clip.tokenize(x, truncate=True))
        return ModelBundle(model=model, tokenize=tokenize)

    if args.model == "tinyclip":
        import clip
        model = load_tinyclip_wkcn(args.tinyclip_ckpt, device=device)
        tokenize = TokenizeCache(lambda x: clip.tokenize(x, truncate=True))
        return ModelBundle(model=model, tokenize=tokenize)

    raise ValueError(f"Unknown model: {args.model}")


def get_gene_params_by_attr(model: nn.Module) -> Optional[List[nn.Parameter]]:
    """
    Prefer attribute-based gene parameters: vision_tower.gene / text_tower.gene
    """
    try:
        gene_params = []
        if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "gene"):
            gene_params += list(model.vision_tower.gene.parameters())
        if hasattr(model, "text_tower") and hasattr(model.text_tower, "gene"):
            gene_params += list(model.text_tower.gene.parameters())
        return gene_params if len(gene_params) > 0 else None
    except Exception:
        return None


def split_param_groups(model: nn.Module, gene_keywords: List[str]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Return (new_params, gene_params).
    """
    gene_params = get_gene_params_by_attr(model)
    if gene_params is None:
        kws = [k.lower() for k in gene_keywords]
        gene_params = []
        for n, p in model.named_parameters():
            if any(k in n.lower() for k in kws):
                gene_params.append(p)

    gene_ids = set(id(p) for p in gene_params)
    new_params = [p for p in model.parameters() if id(p) not in gene_ids]
    return new_params, gene_params


def set_requires_grad(params: List[nn.Parameter], flag: bool):
    for p in params:
        p.requires_grad = flag
