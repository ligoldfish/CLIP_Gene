"""前端块蒸馏预热：训练前端使其输出匹配 teacher 在首基因块输入处的隐状态。

镜像 stitch_init 的 hook 模式。只训前端（基因/adapter/缝合不动）。fp32 稳定。
在主训练与 LSQ 之前调用。
"""
import math

import torch
import torch.nn.functional as F

from utils.device import autocast


def _cos_warm(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def preheat_front_blocks(student, teacher, calib_loader, device, cfg):
    """返回 {loss, steps}。无前端 / 首基因索引为 0 的分支跳过。"""
    teacher.eval()
    student.eval()
    branches = []
    if getattr(student.vision, "front", None) is not None:
        sv = student.vision.trunk.sel[0]
        if sv > 0:
            branches.append(("vision", sv))
    if getattr(student.text, "front", None) is not None:
        st = student.text.trunk.sel[0]
        if st > 0:
            branches.append(("text", st))
    if not branches:
        print("[preheat] 无前端或首基因索引为 0，跳过")
        return {}

    do_v = any(b == "vision" for b, _ in branches)
    do_t = any(b == "text" for b, _ in branches)

    # teacher pre-hook 捕获首基因输入 = 目标 Y
    ycap, handles = {}, []
    for branch, idx in branches:
        blocks = teacher.visual.transformer.resblocks if branch == "vision" else teacher.transformer.resblocks

        def _hook(mod, args, key=branch):
            ycap[key] = args[0].detach()

        handles.append(blocks[idx].register_forward_pre_hook(_hook))

    fp = []
    if do_v:
        fp += list(student.vision.front.parameters())
    if do_t:
        fp += list(student.text.front.parameters())
    fp = [p for p in fp if p.requires_grad]
    # 防呆：前端参数不与基因/adapter 别名
    assert all(p.requires_grad for p in fp), "前端参数应全部可训"

    opt = torch.optim.AdamW(fp, lr=cfg.PREHEAT_LR, betas=(0.9, 0.98), eps=1e-8,
                            weight_decay=cfg.PREHEAT_WD)
    total = cfg.PREHEAT_EPOCHS * max(1, len(calib_loader))
    warm = int(0.05 * total)
    amp_dtype = torch.bfloat16 if cfg.PREHEAT_AMP else None

    step = 0
    last = 0.0
    try:
        for ep in range(cfg.PREHEAT_EPOCHS):
            for imgs, toks in calib_loader:
                imgs = imgs.to(device, non_blocking=True)
                toks = toks.to(device, non_blocking=True)
                ycap.clear()
                with torch.no_grad():
                    if do_v:
                        teacher.encode_image(imgs)
                    if do_t:
                        teacher.encode_text(toks)
                with autocast(enabled=cfg.PREHEAT_AMP, dtype=amp_dtype):
                    loss = imgs.new_zeros(())
                    if do_v:
                        Xv = student.vision.forward_front(imgs)
                        Yv = ycap["vision"].to(Xv.dtype)
                        assert Xv.shape == Yv.shape, f"vision front shape {Xv.shape} != {Yv.shape}"
                        loss = loss + F.mse_loss(Xv, Yv)
                    if do_t:
                        Xt = student.text.forward_front(toks)
                        Yt = ycap["text"].to(Xt.dtype)
                        assert Xt.shape == Yt.shape, f"text front shape {Xt.shape} != {Yt.shape}"
                        loss = loss + F.mse_loss(Xt, Yt)
                lr = cfg.PREHEAT_LR * _cos_warm(step, total, warm)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fp, cfg.MAX_GRAD_NORM)
                opt.step()
                step += 1
                last = loss.item()
            print(f"[preheat] ep{ep + 1}/{cfg.PREHEAT_EPOCHS} mse={last:.4f} (branches={[b for b,_ in branches]})")
    finally:
        for h in handles:
            h.remove()
    return {"loss": float(last), "steps": step}
