"""设备/AMP 后端抽象：CUDA / Ascend-NPU / CPU 通吃。

新代码统一用本模块的 DEVICE / autocast() / make_grad_scaler() / manual_seed_all()，
不直接写 torch.cuda.*，从而 NPU(torch_npu) 与 GPU 同一份代码跑。

NPU 用法：环境装好 torch_npu 后，本模块自动探测 torch.npu.is_available() 并切到 npu，
AMP 走 torch.npu.amp，无需改业务代码。
"""
import contextlib
import torch

_BACKEND = "cpu"

# 探测 Ascend NPU（torch_npu 注册 torch.npu）
try:  # noqa: SIM105
    import torch_npu  # noqa: F401
    if hasattr(torch, "npu") and torch.npu.is_available():
        _BACKEND = "npu"
except Exception:  # noqa: BLE001
    pass

# 退到 CUDA
if _BACKEND == "cpu" and torch.cuda.is_available():
    _BACKEND = "cuda"

# 设备字符串（npu/cuda 用默认 0 号卡）
DEVICE = _BACKEND if _BACKEND != "cpu" else "cpu"


def backend() -> str:
    return _BACKEND


def is_npu() -> bool:
    return _BACKEND == "npu"


def autocast(enabled: bool = True, dtype=None):
    """混合精度上下文。dtype=None 走后端默认(fp16)；可传 torch.bfloat16。CPU 上 no-op。"""
    if not enabled:
        return contextlib.nullcontext()
    if _BACKEND == "npu":
        return torch.npu.amp.autocast(dtype=dtype) if dtype is not None else torch.npu.amp.autocast()
    if _BACKEND == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def make_grad_scaler():
    """AMP GradScaler。CPU 上 enabled=False（no-op）。"""
    if _BACKEND == "npu":
        return torch.npu.amp.GradScaler()
    if _BACKEND == "cuda":
        return torch.cuda.amp.GradScaler()
    return torch.cuda.amp.GradScaler(enabled=False)


def manual_seed_all(seed: int):
    torch.manual_seed(seed)
    if _BACKEND == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif _BACKEND == "npu":
        torch.npu.manual_seed_all(seed)


def empty_cache():
    if _BACKEND == "cuda":
        torch.cuda.empty_cache()
    elif _BACKEND == "npu":
        torch.npu.empty_cache()
