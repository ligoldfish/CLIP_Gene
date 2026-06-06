from typing import Tuple, Dict, List
import re
import torch
import torch.nn as nn
import clip
import math

from configs.base_config import Config
from models.cnn_geneclip import StudentMiniCLIP, StudentVisionEncoder, StudentTextEncoder


def _extract_indices_from_state(state: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    idxs = set()
    for k in state.keys():
        m = pat.match(k)
        if m: idxs.add(int(m.group(1)))
    return sorted(list(idxs))


def _slice_teacher_blocks(teacher_blocks: nn.ModuleList, indices: List[int]) -> List[nn.Module]:
    return [teacher_blocks[i] for i in indices]


def _substate_for_layer(state: Dict[str, torch.Tensor], prefix: str, layer_idx: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    base = f"{prefix}{layer_idx}."
    for k, v in state.items():
        if k.startswith(base):
            out[k[len(base):]] = v
    return out


def _infer_grid_hw(visual) -> int:
    # positional_embedding: [1+HW, D]
    n = visual.positional_embedding.shape[0] - 1
    s = int(math.sqrt(n))
    assert s * s == n, f"positional embedding length-1 ({n}) is not a perfect square"
    return s * s


def build_student_from_gene(cfg: Config, device: str = "cuda"):
    teacher, preprocess = clip.load(cfg.CLIP_MODEL, device=device, jit=False)
    teacher = teacher.float().eval()
    # sanity check ViT
    assert hasattr(teacher.visual, 'transformer') and hasattr(teacher.visual, 'class_embedding'), "This script requires a ViT-based CLIP (e.g., ViT-B/32)."

    gene_state = torch.load(cfg.model_ckpt, map_location=device)

    vis_prefix = "visual.transformer.resblocks."
    txt_prefix = "transformer.resblocks."
    vis_idxs = _extract_indices_from_state(gene_state, vis_prefix)
    txt_idxs = _extract_indices_from_state(gene_state, txt_prefix)
    assert len(vis_idxs) + len(txt_idxs) > 0, "No resblocks found in cross_model.pt"

    vis_gene_blocks = _slice_teacher_blocks(teacher.visual.transformer.resblocks, vis_idxs)
    txt_gene_blocks = _slice_teacher_blocks(teacher.transformer.resblocks,        txt_idxs)
    for i, idx in enumerate(vis_idxs):
        sub = _substate_for_layer(gene_state, vis_prefix, idx)
        vis_gene_blocks[i].load_state_dict(sub, strict=False)
    for i, idx in enumerate(txt_idxs):
        sub = _substate_for_layer(gene_state, txt_prefix, idx)
        txt_gene_blocks[i].load_state_dict(sub, strict=False)

    width_v = teacher.visual.width
    width_t = teacher.transformer.width
    grid_hw = _infer_grid_hw(teacher.visual)

    vision_enc = StudentVisionEncoder(
        teacher_visual=teacher.visual,
        visual_gene_blocks=vis_gene_blocks,
        width=width_v,
        grid_hw=grid_hw,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
        shallow_layers=cfg.SHALLOW_CNN_LAYERS_V,
        shallow_bottleneck=cfg.SHALLOW_CNN_BOTTLENECK,
        freeze_stem=True,
        freeze_post=True,
    )
    text_enc = StudentTextEncoder(
        teacher_clip=teacher,
        text_gene_blocks=txt_gene_blocks,
        width=width_t,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
        shallow_layers=cfg.SHALLOW_CNN_LAYERS_T,
        shallow_bottleneck=cfg.SHALLOW_CNN_BOTTLENECK,
        freeze_embed=True,
        freeze_head=True,
    )

    logit_scale_init = float(teacher.logit_scale.detach().cpu())
    student = StudentMiniCLIP(
        vision_enc=vision_enc,
        text_enc=text_enc,
        logit_scale_init=logit_scale_init,
        logit_scale_max=cfg.LOGIT_SCALE_MAX,
    ).to(device)

    meta = {
        "visual_gene_indices": vis_idxs,
        "text_gene_indices": txt_idxs,
        "grid_hw": grid_hw,
    }
    return student, meta, preprocess
