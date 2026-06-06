from typing import Tuple
import re
import torch
import clip
import torch.nn as nn

from configs.base_config import Config
from models.geneclip import (
    StudentMiniCLIP,
    StudentVisionEncoder,
    StudentTextEncoder,
)


def _extract_indices_from_state(state: dict[str, torch.Tensor], prefix: str):
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    idxs = set()
    for k in state.keys():
        m = pat.match(k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(list(idxs))


def _slice_teacher_blocks(teacher_blocks: nn.ModuleList, indices: list[int]):
    return [teacher_blocks[i] for i in indices]


def _substate_for_layer(state: dict[str, torch.Tensor], prefix: str, layer_idx: int) -> dict[str, torch.Tensor]:
    out = {}
    base = f"{prefix}{layer_idx}."
    for k, v in state.items():
        if k.startswith(base):
            out[k[len(base):]] = v
    return out


def build_student_from_gene(cfg: Config, device: str = "cuda") -> Tuple[StudentMiniCLIP, dict, callable]:
    """根据 cross_model.pt 构建学生模型；返回 (student, meta, preprocess_fn)。"""
    # 1) 加载 Teacher CLIP（仅用于预处理与初始化）
    teacher, preprocess = clip.load(cfg.CLIP_MODEL, device=device, jit=False)
    teacher = teacher.float().eval()

    # 2) 读取基因权重
    gene_state = torch.load(cfg.model_ckpt, map_location=device)

    # 3) 解析层索引
    vis_prefix = "visual.transformer.resblocks."
    txt_prefix = "transformer.resblocks."
    vis_idxs = _extract_indices_from_state(gene_state, vis_prefix)
    txt_idxs = _extract_indices_from_state(gene_state, txt_prefix)
    assert len(vis_idxs) + len(txt_idxs) > 0, "cross_model.pt 中未找到任何 resblocks.* 参数"

    # 4) 从Teacher复制对应Block并覆盖为基因权重
    vis_gene_blocks = _slice_teacher_blocks(teacher.visual.transformer.resblocks, vis_idxs)
    txt_gene_blocks = _slice_teacher_blocks(teacher.transformer.resblocks, txt_idxs)
    for i, idx in enumerate(vis_idxs):
        sub = _substate_for_layer(gene_state, vis_prefix, idx)
        vis_gene_blocks[i].load_state_dict(sub, strict=False)
    for i, idx in enumerate(txt_idxs):
        sub = _substate_for_layer(gene_state, txt_prefix, idx)
        txt_gene_blocks[i].load_state_dict(sub, strict=False)

    width_v = teacher.visual.transformer.width
    width_t = teacher.transformer.width

    # 5) 组装学生编码器
    vision_enc = StudentVisionEncoder(
        teacher_visual=teacher.visual,
        visual_gene_blocks=vis_gene_blocks,
        width=width_v,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
        freeze_stem=True,
        freeze_post=True,
    )
    text_enc = StudentTextEncoder(
        teacher_clip=teacher,
        text_gene_blocks=txt_gene_blocks,
        width=width_t,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
        freeze_embed=True,
        freeze_head=True,
    )

    # 6) 组装 StudentMiniCLIP，初始化 logit_scale 与 teacher 一致
    logit_scale_init = float(teacher.logit_scale.detach().cpu())  # teacher 的参数本身就是 log(1/tau)
    student = StudentMiniCLIP(vision_enc, text_enc, logit_scale_init).to(device)

    meta = {
        "visual_gene_indices": vis_idxs,
        "text_gene_indices": txt_idxs,
        "adapter_bottleneck": cfg.ADAPTER_BOTTLENECK,
    }
    return student, meta, preprocess