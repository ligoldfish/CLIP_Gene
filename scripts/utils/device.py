from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Optional, Tuple

import torch


_TORCH_NPU_IMPORTED = False


def try_import_torch_npu() -> bool:
    """Import torch_npu when available so that torch.device("npu") works."""

    global _TORCH_NPU_IMPORTED
    if _TORCH_NPU_IMPORTED:
        return True
    try:
        import torch_npu  # noqa: F401

        _TORCH_NPU_IMPORTED = True
        return True
    except Exception:
        return False


def npu_is_available() -> bool:
    try_import_torch_npu()
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_available"):
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def get_default_device() -> str:
    if npu_is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: Optional[str] = None, *, allow_cpu_fallback: bool = False) -> torch.device:
    requested = (device or "auto").lower()
    if requested in {"auto", "default"}:
        requested = get_default_device()

    if requested.startswith("npu"):
        try_import_torch_npu()
        if not npu_is_available() and not allow_cpu_fallback:
            raise RuntimeError("NPU device requested but torch_npu/torch.npu is not available.")
        return torch.device(requested if npu_is_available() else "cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available() and not allow_cpu_fallback:
            raise RuntimeError("CUDA device requested but torch.cuda is not available.")
        return torch.device(requested if torch.cuda.is_available() else "cpu")

    return torch.device(requested)


def is_npu_device(device) -> bool:
    return torch.device(device).type == "npu"


def is_cuda_device(device) -> bool:
    return torch.device(device).type == "cuda"


def is_accelerator_device(device) -> bool:
    return torch.device(device).type in {"cuda", "npu"}


def default_backend_for_device(device: Optional[str] = None) -> str:
    dev_type = resolve_device(device or "auto", allow_cpu_fallback=True).type
    if dev_type == "npu":
        return "hccl"
    if dev_type == "cuda":
        return "nccl"
    return "gloo"


def normalize_backend(backend: str, device: Optional[str] = None) -> str:
    if not backend or str(backend).lower() == "auto":
        return default_backend_for_device(device)
    return str(backend)


def set_device_for_distributed(local_rank: int, device: Optional[str] = None) -> torch.device:
    dev = resolve_device(device or "auto", allow_cpu_fallback=True)
    if dev.type == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if dev.type == "npu":
        try_import_torch_npu()
        torch.npu.set_device(local_rank)
        return torch.device("npu", local_rank)
    return dev


def enable_tf32(enable: bool = True) -> None:
    """Enable TF32 only on CUDA. NPU/CPU safely ignore this."""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        try:
            torch.backends.cudnn.benchmark = enable
        except Exception:
            pass


def npu_bf16_is_supported() -> bool:
    # torch_npu does not expose one stable API across releases; prefer conservative fp16.
    return False


def resolve_amp_dtype(device, amp_enabled: bool, requested: str = "auto") -> torch.dtype:
    if not amp_enabled:
        return torch.float16

    dev_type = torch.device(device).type
    req = str(requested or "auto").lower()
    if req == "auto":
        if dev_type == "cuda":
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if dev_type == "npu":
            return torch.bfloat16 if npu_bf16_is_supported() else torch.float16
        return torch.bfloat16
    if req in {"bf16", "bfloat16"}:
        if dev_type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            return torch.float16
        if dev_type == "npu" and not npu_bf16_is_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


def autocast_context(device, enabled: bool, dtype: torch.dtype):
    dev_type = torch.device(device).type
    if not enabled or dev_type == "cpu":
        return nullcontext()
    try:
        return torch.autocast(device_type=dev_type, dtype=dtype, enabled=True)
    except Exception:
        if dev_type == "cuda":
            return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        return nullcontext()


def make_grad_scaler(device, enabled: bool, **kwargs):
    dev_type = torch.device(device).type
    if not enabled:
        return torch.cuda.amp.GradScaler(enabled=False)
    if dev_type == "npu":
        try_import_torch_npu()
        amp_mod = getattr(getattr(torch, "npu", None), "amp", None)
        scaler_cls = getattr(amp_mod, "GradScaler", None)
        if scaler_cls is not None:
            return scaler_cls(enabled=True, **kwargs)
        return torch.cuda.amp.GradScaler(enabled=False)
    if dev_type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True, **kwargs)
    return torch.cuda.amp.GradScaler(enabled=False)


def scaler_is_enabled(scaler) -> bool:
    if scaler is None:
        return False
    try:
        return bool(scaler.is_enabled())
    except Exception:
        return False


def synchronize_device(device=None) -> None:
    dev = torch.device(device) if device is not None else torch.device(get_default_device())
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    elif dev.type == "npu" and npu_is_available():
        torch.npu.synchronize(dev)


def reset_peak_memory_stats(device=None) -> None:
    dev = torch.device(device) if device is not None else torch.device(get_default_device())
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)
    elif dev.type == "npu" and npu_is_available() and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats(dev)


def max_memory_allocated(device=None) -> float:
    dev = torch.device(device) if device is not None else torch.device(get_default_device())
    if dev.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(dev))
    if dev.type == "npu" and npu_is_available() and hasattr(torch.npu, "max_memory_allocated"):
        return float(torch.npu.max_memory_allocated(dev))
    return 0.0


def seed_accelerators(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))
