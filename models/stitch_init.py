"""缝合层最小二乘(LSQ)初始化。

把每个真缝合层初始化为：将其**学生侧输入分布 X** 映射到 **teacher 对应位置的
隐状态 Y**（teacher 中下一个被选基因块的输入）。这样非连续基因块拿到的输入分布
回到 in-distribution，残差流不再断裂。

用正规方程累积（XᵀX, XᵀY），避免存全部激活；float64 求解。
低秩缝合用 W−I 的截断 SVD 装入 U/V。
"""
from typing import List, Optional

import torch

from models.blocks import StitchLayer


class _StitchSpec:
    def __init__(self, module, branch: str, ykey):
        self.module = module
        self.branch = branch
        self.ykey = ykey            # ('vision', idx) | ('vision','final') | ('text', idx) ...
        self.dim = module.dim


def _collect_specs(encoder, branch: str) -> List[_StitchSpec]:
    trunk = encoder.trunk
    sel = trunk.sel
    specs = []
    if isinstance(trunk.pre_stitch, StitchLayer):
        specs.append(_StitchSpec(trunk.pre_stitch, branch, (branch, sel[0])))
    for k, m in enumerate(trunk.mid_stitch):
        if isinstance(m, StitchLayer):
            specs.append(_StitchSpec(m, branch, (branch, sel[k + 1])))
    if isinstance(trunk.post_stitch, StitchLayer):
        specs.append(_StitchSpec(trunk.post_stitch, branch, (branch, "final")))
    return specs


def _resblocks(teacher, branch):
    return teacher.visual.transformer.resblocks if branch == "vision" else teacher.transformer.resblocks


def _register_teacher_ycap(teacher, specs, ycap):
    """对 teacher 注册 hook，捕获各 ykey 处的隐状态 [L,B,D]。"""
    handles = []
    needed = {s.ykey for s in specs}
    for (branch, idx) in needed:
        blocks = _resblocks(teacher, branch)
        if idx == "final":
            def _post_hook(mod, inp, out, key=(branch, idx)):
                ycap[key] = out.detach()
            handles.append(blocks[-1].register_forward_hook(_post_hook))
        else:
            def _pre_hook(mod, args, key=(branch, idx)):
                ycap[key] = args[0].detach()
            handles.append(blocks[idx].register_forward_pre_hook(_pre_hook))
    return handles


def _make_xcap(xcap, key):
    def hook(mod, args):
        xcap[key] = args[0].detach()
    return hook


def _accum(acc, X, Y):
    # X,Y: [L,B,D] -> [(L*B), D]
    Xf = X.reshape(-1, X.shape[-1]).double()
    Yf = Y.reshape(-1, Y.shape[-1]).double()
    ones = torch.ones(Xf.shape[0], 1, dtype=torch.float64, device=Xf.device)
    Xa = torch.cat([Xf, ones], dim=1)              # [N, D+1]
    acc["XtX"] = acc["XtX"] + Xa.t() @ Xa          # [D+1, D+1]
    acc["XtY"] = acc["XtY"] + Xa.t() @ Yf          # [D+1, D]
    acc["n"] += Xf.shape[0]


def _solve(acc, ridge=1e-3):
    D1 = acc["XtX"].shape[0]
    A = acc["XtX"] + ridge * torch.eye(D1, dtype=torch.float64, device=acc["XtX"].device)
    P = torch.linalg.solve(A, acc["XtY"])          # [D+1, D],  Y ≈ [X|1] @ P
    W = P[:-1].t().contiguous()                    # nn.Linear weight: y = x@W^T + b -> W=[D,D]
    b = P[-1].contiguous()                         # [D]
    return W, b


@torch.no_grad()
def _load_into_stitch(stitch: StitchLayer, W: torch.Tensor, b: torch.Tensor):
    dev = stitch.lin.weight.device if stitch.rank is None else stitch.U.weight.device
    dt = stitch.lin.weight.dtype if stitch.rank is None else stitch.U.weight.dtype
    if stitch.rank is None:
        stitch.lin.weight.copy_(W.to(dev, dt))
        if stitch.lin.bias is not None:
            stitch.lin.bias.copy_(b.to(dev, dt))
    else:
        # x + U(Vx) ≈ W x + b  =>  U V ≈ W - I,  bias -> U.bias
        D = W.shape[0]
        M = W - torch.eye(D, dtype=W.dtype, device=W.device)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = stitch.rank
        Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :]    # Ur:[D,r] Sr:[r] Vr:[r,D]
        stitch.V.weight.copy_(Vr.to(dev, dt))                       # [r, D]
        stitch.U.weight.copy_((Ur * Sr.unsqueeze(0)).to(dev, dt))   # [D, r]
        if stitch.U.bias is not None:
            stitch.U.bias.copy_(b.to(dev, dt))


@torch.no_grad()
def lsq_init_stitches(student, teacher, calib_loader, device, max_batches: Optional[int] = None):
    """对 student 内所有真缝合层做 LSQ 初始化。无真缝合则直接返回。"""
    teacher.eval()
    student.eval()
    specs = _collect_specs(student.vision, "vision") + _collect_specs(student.text, "text")
    if not specs:
        print("[stitch] 无需缝合（基因块连续或单块）")
        return

    acc = {id(s.module): {"XtX": 0.0, "XtY": 0.0, "n": 0} for s in specs}
    xcap, ycap = {}, {}
    handles = []
    for s in specs:
        handles.append(s.module.register_forward_pre_hook(_make_xcap(xcap, id(s.module))))
    handles += _register_teacher_ycap(teacher, specs, ycap)

    n_batches = 0
    for bi, (imgs, toks) in enumerate(calib_loader):
        imgs = imgs.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)
        xcap.clear(); ycap.clear()
        teacher.encode_image(imgs); teacher.encode_text(toks)   # fills ycap
        student.encode_image(imgs); student.encode_text(toks)   # fills xcap
        for s in specs:
            X = xcap[id(s.module)]
            Y = ycap[s.ykey]
            _accum(acc[id(s.module)], X, Y)
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    for h in handles:
        h.remove()

    for s in specs:
        W, b = _solve(acc[id(s.module)])
        _load_into_stitch(s.module, W, b)
    print(f"[stitch] LSQ 初始化 {len(specs)} 个缝合层（{n_batches} batch）完成")
