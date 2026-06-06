"""Cross-modal Block Influence (CM-BI) 选层。

在**冻结的预训练 teacher**上，对每个 resblock 做恒等跳块消融，量其对跨模态
affinity 矩阵的贡献。越重要的块跳掉后 affinity 变化越大 -> BI 越高。

判据（ShortGPT 的 Block-Influence 扩到联合 image-text affinity）：
  BI_i = 1 - E_batch[ cos( flatten(S_full), flatten(S_skip_i) ) ]
另算 ΔInfoNCE / ΔAcc 备用。视觉/文本分支独立打分排名。

附 grad-rho baseline（冻结 teacher 上的梯度活跃度），供消融对比。
"""
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from utils.teacher_forward import teacher_embeddings, assert_skip_effective


# =============================================================================
# 打分原语
# =============================================================================
def _infonce(S: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(S.size(0), device=S.device)
    return 0.5 * (F.cross_entropy(S, labels) + F.cross_entropy(S.t(), labels))


def _top1(S: torch.Tensor) -> float:
    labels = torch.arange(S.size(0), device=S.device)
    return (S.argmax(dim=1) == labels).float().mean().item()


def _bi_score(S_full: torch.Tensor, S_skip: torch.Tensor, score: str, scale: float) -> float:
    if score == "cos":
        a, b = S_full.flatten(), S_skip.flatten()
        return float(1.0 - F.cosine_similarity(a, b, dim=0).item())
    if score == "infonce":
        return float((_infonce(S_skip * scale) - _infonce(S_full * scale)).item())
    if score == "acc":
        return float(_top1(S_full * scale) - _top1(S_skip * scale))
    raise ValueError(f"未知 score: {score}")


# =============================================================================
# CM-BI 主流程
# =============================================================================
@torch.no_grad()
def compute_cmbi_ranking(teacher, calib_loader, device, score: str = "cos") -> Dict:
    teacher.eval()
    n_v = len(teacher.visual.transformer.resblocks)
    n_t = len(teacher.transformer.resblocks)
    scale = float(teacher.logit_scale.exp().clamp(max=100.0).item())

    acc_v: Dict[int, List[float]] = {i: [] for i in range(n_v)}
    acc_t: Dict[int, List[float]] = {j: [] for j in range(n_t)}

    checked = False
    for images, tokens in calib_loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        if not checked:
            assert_skip_effective(teacher, images, tokens, device)
            checked = True

        zi_full, zt_full = teacher_embeddings(teacher, images, tokens)
        assert zi_full.size(0) == zt_full.size(0), "affinity 需方阵：图文 batch 大小须一致"
        S_full = zi_full @ zt_full.t()

        # 视觉分支：逐块跳，复用 zt_full
        for i in range(n_v):
            zi_s, _ = teacher_embeddings(teacher, images, None, skip_vision=i)
            S_s = zi_s @ zt_full.t()
            acc_v[i].append(_bi_score(S_full, S_s, score, scale))
        # 文本分支：逐块跳，复用 zi_full
        for j in range(n_t):
            _, zt_s = teacher_embeddings(teacher, None, tokens, skip_text=j)
            S_s = zi_full @ zt_s.t()
            acc_t[j].append(_bi_score(S_full, S_s, score, scale))

    vis_scores = {i: float(np.mean(v)) for i, v in acc_v.items()}
    txt_scores = {j: float(np.mean(v)) for j, v in acc_t.items()}
    vis_rank = sorted(vis_scores, key=vis_scores.get, reverse=True)
    txt_rank = sorted(txt_scores, key=txt_scores.get, reverse=True)
    return {
        "vision": {"width": teacher.visual.transformer.width, "scores": vis_scores, "ranking": vis_rank},
        "text": {"width": teacher.transformer.width, "scores": txt_scores, "ranking": txt_rank},
    }


# =============================================================================
# grad-rho baseline（冻结 teacher 上的梯度活跃度，消融用）
# =============================================================================
def compute_grad_rho_ranking(teacher, calib_loader, device, threshold: float = 1e-3) -> Dict:
    teacher.eval()
    n_v = len(teacher.visual.transformer.resblocks)
    n_t = len(teacher.transformer.resblocks)
    for p in teacher.parameters():
        p.requires_grad_(True)
    acc_v: Dict[int, List[float]] = {i: [] for i in range(n_v)}
    acc_t: Dict[int, List[float]] = {j: [] for j in range(n_t)}
    scale = teacher.logit_scale.exp().clamp(max=100.0)

    for images, tokens in calib_loader:
        images = images.to(device); tokens = tokens.to(device)
        zi = teacher.encode_image(images).float()
        zt = teacher.encode_text(tokens).float()
        zi = zi / zi.norm(dim=-1, keepdim=True)
        zt = zt / zt.norm(dim=-1, keepdim=True)
        S = scale * (zi @ zt.t())
        loss = _infonce(S)
        teacher.zero_grad(set_to_none=True)
        loss.backward()
        _accum_rho(teacher.visual.transformer.resblocks, acc_v, threshold)
        _accum_rho(teacher.transformer.resblocks, acc_t, threshold)

    teacher.zero_grad(set_to_none=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    vis_scores = {i: float(np.mean(v)) for i, v in acc_v.items() if v}
    txt_scores = {j: float(np.mean(v)) for j, v in acc_t.items() if v}
    return {
        "vision": {"width": teacher.visual.transformer.width, "scores": vis_scores,
                   "ranking": sorted(vis_scores, key=vis_scores.get, reverse=True)},
        "text": {"width": teacher.transformer.width, "scores": txt_scores,
                 "ranking": sorted(txt_scores, key=txt_scores.get, reverse=True)},
    }


def _accum_rho(resblocks, acc: Dict[int, List[float]], threshold: float):
    for i, blk in enumerate(resblocks):
        flags = []
        for p in blk.parameters():
            if p.grad is not None:
                flags.append(1.0 if p.grad.abs().mean().item() > threshold else 0.0)
        if flags:
            acc[i].append(sum(flags) / len(flags))


# =============================================================================
# gene-spec I/O
# =============================================================================
def save_gene_spec(spec: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"[cmbi] gene-spec 已存 {path}")


def load_gene_spec(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
