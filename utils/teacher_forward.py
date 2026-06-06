"""可恒等跳块的 teacher 前向，供 CM-BI 选层与同源蒸馏共用。

openai resblock 返回同形 [L,B,D]，故"跳块" = 临时把 resblocks[idx] 换成返回输入的
恒等模块，encode 后还原。纯读、可逆、应在 .eval()+no_grad+fp32 下用。
"""
import contextlib
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _IdentityResblock(nn.Module):
    """openai ResidualAttentionBlock 的恒等替身：forward(x) 原样返回（忽略 attn_mask 属性）。"""

    def forward(self, x):  # x: [L, B, D]
        return x


@contextlib.contextmanager
def skip_resblock(resblocks: nn.Module, idx: Optional[int]):
    """临时把 resblocks[idx] 替换为恒等；idx=None 为 no-op。

    resblocks 为 openai 的 nn.Sequential，支持下标赋值；若某 clip 变体不支持，
    退化为 monkeypatch 该块 forward。
    """
    if idx is None:
        yield
        return
    orig = resblocks[idx]
    swapped = False
    try:
        resblocks[idx] = _IdentityResblock()
        swapped = True
    except Exception:  # noqa: BLE001
        # 退路：替换 forward
        orig_forward = orig.forward
        orig.forward = lambda x, *a, **k: x  # type: ignore
    try:
        yield
    finally:
        if swapped:
            resblocks[idx] = orig
        else:
            orig.forward = orig_forward  # type: ignore


@torch.no_grad()
def teacher_embeddings(
    teacher,
    images: Optional[torch.Tensor],
    tokens: Optional[torch.Tensor],
    skip_vision: Optional[int] = None,
    skip_text: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """返回 (img_emb, txt_emb)。skip_* = 要恒等跳过的 resblock 索引或 None。

    images / tokens 任一可为 None（只取另一分支，省算力）。
    """
    zi = zt = None
    with skip_resblock(teacher.visual.transformer.resblocks, skip_vision), \
         skip_resblock(teacher.transformer.resblocks, skip_text):
        if images is not None:
            zi = teacher.encode_image(images).float()
        if tokens is not None:
            zt = teacher.encode_text(tokens).float()
    if normalize:
        if zi is not None:
            zi = zi / zi.norm(dim=-1, keepdim=True)
        if zt is not None:
            zt = zt / zt.norm(dim=-1, keepdim=True)
    return zi, zt


@torch.no_grad()
def assert_skip_effective(teacher, images, tokens, device) -> None:
    """一次性自检：跳 vision/text 第 0 块确实改变 embedding（防 Sequential 替换未生效）。"""
    zi_full, zt_full = teacher_embeddings(teacher, images, tokens)
    zi_sv, _ = teacher_embeddings(teacher, images, None, skip_vision=0)
    _, zt_st = teacher_embeddings(teacher, None, tokens, skip_text=0)
    dv = (zi_full - zi_sv).abs().max().item()
    dt = (zt_full - zt_st).abs().max().item()
    assert dv > 1e-6, f"跳视觉块 0 未改变 embedding（max diff={dv}）—— 替换未生效"
    assert dt > 1e-6, f"跳文本块 0 未改变 embedding（max diff={dt}）—— 替换未生效"
