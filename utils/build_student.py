"""从 gene-spec 构建预算-K 学生（替代 build_geneclip.py / build_cnn_geneclip.py）。

基因块按索引直接从**冻结 teacher** deepcopy（不再从坏掉的 model_ckpt 解析索引）。
含 `_assert_gene_params_loaded` 取代静默 strict=False。
"""
import copy
from typing import Dict, List, Optional, Tuple

import torch

from models.student_clip import StudentVisionEncoder, StudentTextEncoder, StudentMiniCLIP
from models.cmbi import load_gene_spec


def _truncate(ranking: List[int], scores: Dict, K: int, contiguous: bool, n_total: int) -> List[int]:
    if not contiguous:
        return ranking[:K]
    # 连续段消融：滑窗选 CM-BI 和最大的长 K 连续段
    def sc(i):
        return float(scores.get(str(i), scores.get(i, 0.0)))
    best_s, best_sum = 0, -1e30
    for s in range(0, n_total - K + 1):
        tot = sum(sc(i) for i in range(s, s + K))
        if tot > best_sum:
            best_sum, best_s = tot, s
    return list(range(best_s, best_s + K))


def _gaps(sel: List[int], stitch_at_head: bool, n_total: int) -> int:
    pre = 1 if (sel and sel[0] != 0) else 0
    mids = sum(1 for k in range(len(sel) - 1) if sel[k + 1] - sel[k] > 1)
    post = 1 if (stitch_at_head and sel and sel[-1] != n_total - 1) else 0
    return pre + mids + post


@torch.no_grad()
def _assert_gene_params_loaded(student, teacher, vis_sel, txt_sel):
    def check(wrappers, teacher_blocks, sel, name):
        assert len(wrappers) == len(sel), f"{name} 块数 {len(wrappers)} != K {len(sel)}"
        for w, idx in zip(wrappers, sel):
            tb = teacher_blocks[idx]
            tp = dict(tb.named_parameters())
            for n, p in w.gene_block.named_parameters():
                assert n in tp, f"{name} 块{idx} 缺参数 {n}"
                assert torch.equal(p.detach().cpu(), tp[n].detach().cpu()), \
                    f"{name} 块{idx} 参数 {n} 与 teacher 不一致（基因被意外重置）"
                assert p.requires_grad is False, f"{name} 块{idx} 参数 {n} 未冻结"
    check(student.vision.trunk.blocks, teacher.visual.transformer.resblocks, vis_sel, "vision")
    check(student.text.trunk.blocks, teacher.transformer.resblocks, txt_sel, "text")


def build_student_from_gene(
    cfg, gene_spec: Optional[Dict] = None, K: Optional[int] = None,
    device: str = "cuda", teacher=None, preprocess=None,
) -> Tuple[StudentMiniCLIP, Dict, object, object]:
    """返回 (student, meta, preprocess, teacher)。teacher 供蒸馏 + LSQ 复用。"""
    if teacher is None or preprocess is None:
        from utils.teacher_loader import load_teacher_offline
        teacher, preprocess = load_teacher_offline(cfg, device)
    teacher = teacher.float().eval()

    if gene_spec is None:
        gene_spec = load_gene_spec(cfg.GENE_SPEC_PATH)

    Kv = K if K is not None else (cfg.BUDGET_K_V if cfg.BUDGET_K_V is not None else cfg.BUDGET_K)
    Kt = K if K is not None else (cfg.BUDGET_K_T if cfg.BUDGET_K_T is not None else cfg.BUDGET_K)

    n_v = len(teacher.visual.transformer.resblocks)
    n_t = len(teacher.transformer.resblocks)
    vis_sel = sorted(_truncate(gene_spec["vision"]["ranking"], gene_spec["vision"].get("scores", {}),
                               Kv, cfg.CONTIGUOUS_SPAN, n_v))
    txt_sel = sorted(_truncate(gene_spec["text"]["ranking"], gene_spec["text"].get("scores", {}),
                               Kt, cfg.CONTIGUOUS_SPAN, n_t))

    vis_blocks = [copy.deepcopy(teacher.visual.transformer.resblocks[i]) for i in vis_sel]
    txt_blocks = [copy.deepcopy(teacher.transformer.resblocks[i]) for i in txt_sel]

    width_v = teacher.visual.transformer.width
    width_t = teacher.transformer.width

    vision_enc = StudentVisionEncoder(
        teacher.visual, vis_blocks, vis_sel, width=width_v,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK, use_shallow_cnn=cfg.USE_SHALLOW_CNN,
        shallow_layers=cfg.SHALLOW_CNN_LAYERS_V, shallow_bottleneck=cfg.SHALLOW_CNN_BOTTLENECK,
        use_full_stitch=cfg.USE_FULL_STITCH, stitch_rank=cfg.STITCH_RANK,
        stitch_at_head=cfg.STITCH_AT_HEAD, n_total=n_v,
    )
    text_enc = StudentTextEncoder(
        teacher, txt_blocks, txt_sel, width=width_t,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK, use_shallow_cnn=cfg.USE_SHALLOW_CNN,
        shallow_layers=cfg.SHALLOW_CNN_LAYERS_T, shallow_bottleneck=cfg.SHALLOW_CNN_BOTTLENECK,
        use_full_stitch=cfg.USE_FULL_STITCH, stitch_rank=cfg.STITCH_RANK,
        stitch_at_head=cfg.STITCH_AT_HEAD, n_total=n_t,
    )
    logit_scale_init = float(teacher.logit_scale.detach().cpu())
    student = StudentMiniCLIP(vision_enc, text_enc, logit_scale_init, cfg.LOGIT_SCALE_MAX).to(device)

    _assert_gene_params_loaded(student, teacher, vis_sel, txt_sel)
    # 缝合数自检
    n_vstitch = sum(1 for m in [student.vision.trunk.pre_stitch, student.vision.trunk.post_stitch]
                    if not isinstance(m, torch.nn.Identity)) + \
        sum(1 for m in student.vision.trunk.mid_stitch if not isinstance(m, torch.nn.Identity))
    assert n_vstitch == _gaps(vis_sel, cfg.STITCH_AT_HEAD, n_v), "视觉缝合层数与 gap 不符"

    meta = {
        "vision_sel": vis_sel, "text_sel": txt_sel, "Kv": Kv, "Kt": Kt,
        "vision_gaps": _gaps(vis_sel, cfg.STITCH_AT_HEAD, n_v),
        "text_gaps": _gaps(txt_sel, cfg.STITCH_AT_HEAD, n_t),
        "gene_spec_path": cfg.GENE_SPEC_PATH,
    }
    return student, meta, preprocess, teacher
