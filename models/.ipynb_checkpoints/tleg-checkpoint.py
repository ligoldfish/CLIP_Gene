# models/tleg.py
from typing import Dict, List, Optional, Sequence, Tuple
import math
import torch


def _common_keys(layer0: Dict[str, torch.Tensor], layer1: Dict[str, torch.Tensor], strict: bool) -> List[str]:
    k0 = set(layer0.keys())
    k1 = set(layer1.keys())
    if strict:
        if k0 != k1:
            only0 = sorted(list(k0 - k1))[:10]
            only1 = sorted(list(k1 - k0))[:10]
            raise ValueError(f"TLEG strict key mismatch. only_in_0={only0}, only_in_1={only1}")
        keys = sorted(list(k0))
    else:
        keys = sorted(list(k0 & k1))
        if len(keys) == 0:
            raise ValueError("No common parameter keys between layers for TLEG expansion.")
    return keys


@torch.no_grad()
def tleg_linear_expand_two_layers(
    layer0: Dict[str, torch.Tensor],
    layer1: Dict[str, torch.Tensor],
    target_depth: int,
    strict_keys: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """
    Simple TLEG-like expansion via linear interpolation between 2 layers:
      new_layer(k) = (1-alpha)*layer0 + alpha*layer1
      alpha = k/(target_depth-1)
    """
    assert target_depth >= 2, "target_depth must be >= 2"

    keys = _common_keys(layer0, layer1, strict=strict_keys)

    expanded: List[Dict[str, torch.Tensor]] = []
    for k in range(target_depth):
        alpha = float(k) / float(target_depth - 1)
        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            w0 = layer0[key]
            w1 = layer1[key]
            if w0.shape != w1.shape:
                raise ValueError(f"Shape mismatch for key={key}: {w0.shape} vs {w1.shape}")
            # align dtype/device
            w1 = w1.to(device=w0.device, dtype=w0.dtype)
            out[key] = ((1.0 - alpha) * w0 + alpha * w1).detach().clone()
        expanded.append(out)
    return expanded


def _allocate_steps(
    n_src: int,
    target_depth: int,
    positions: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Allocate (target_depth - 1) interpolation steps across (n_src - 1) segments.
    Each segment gets at least 1 step (so endpoints appear).
    Return list steps_per_segment length (n_src - 1), where segment i produces (steps_i + 1) layers incl. endpoints.
    """
    assert n_src >= 2
    segs = n_src - 1
    total_steps = target_depth - 1
    assert total_steps >= segs, "target_depth too small for piecewise interpolation"

    if positions is None or len(positions) != n_src:
        # uniform allocation
        base = total_steps // segs
        rem = total_steps % segs
        steps = [base + (1 if i < rem else 0) for i in range(segs)]
    else:
        # allocate proportional to gaps in positions
        gaps = [max(1, positions[i + 1] - positions[i]) for i in range(segs)]
        s = sum(gaps)
        raw = [total_steps * g / s for g in gaps]
        steps = [max(1, int(math.floor(x))) for x in raw]
        # adjust to match total_steps
        cur = sum(steps)
        # distribute remaining
        while cur < total_steps:
            # add to segment with largest fractional remainder
            fracs = [raw[i] - steps[i] for i in range(segs)]
            j = int(max(range(segs), key=lambda i: fracs[i]))
            steps[j] += 1
            cur += 1
        while cur > total_steps:
            # remove from segment with smallest fractional remainder but keep >=1
            fracs = [raw[i] - steps[i] for i in range(segs)]
            j = int(min([i for i in range(segs) if steps[i] > 1], key=lambda i: fracs[i]))
            steps[j] -= 1
            cur -= 1

    assert sum(steps) == total_steps
    assert all(s >= 1 for s in steps)
    return steps


@torch.no_grad()
def tleg_piecewise_expand_layers(
    layers: List[Dict[str, torch.Tensor]],
    target_depth: int,
    *,
    positions: Optional[Sequence[int]] = None,
    strict_keys: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """
    General TLEG expansion for N>=2 source layers using piecewise linear interpolation.

    - If N==2: reduce to two-layer interpolation.
    - If N>2: interpolate between consecutive layers, allocate target_depth across segments.
      Avoid duplicating joint endpoints.

    positions: optional list of "layer depth indices" (e.g., block ids). Useful for last2+6.
               len(positions) must equal len(layers). If provided, steps are allocated proportional to gaps.

    Returns list length = target_depth.
    """
    n = len(layers)
    assert n >= 2, "Need at least 2 layers for TLEG expansion."
    assert target_depth >= n, "target_depth must be >= number of source layers for expansion."

    if n == 2:
        return tleg_linear_expand_two_layers(layers[0], layers[1], target_depth, strict_keys=strict_keys)

    steps = _allocate_steps(n, target_depth, positions=positions)  # length n-1, sum = target_depth-1

    out_all: List[Dict[str, torch.Tensor]] = []
    for i in range(n - 1):
        l0 = layers[i]
        l1 = layers[i + 1]
        seg_depth = steps[i] + 1  # incl endpoints
        seg_layers = tleg_linear_expand_two_layers(l0, l1, seg_depth, strict_keys=strict_keys)
        if i > 0:
            # drop first to avoid duplicating joint endpoint
            seg_layers = seg_layers[1:]
        out_all.extend(seg_layers)

    assert len(out_all) == target_depth, f"Got {len(out_all)} layers, expected {target_depth}"
    return out_all


@torch.no_grad()
def tleg_expand_any_gene(
    gene_layers: List[Dict[str, torch.Tensor]],
    target_depth: int,
    *,
    layer_ids: Optional[Sequence[int]] = None,
    strict_keys: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """
    Convenience wrapper:
      - if target_depth <= len(gene_layers): return gene_layers (no expansion)
      - else piecewise expand to target_depth
    """
    if target_depth <= len(gene_layers):
        return gene_layers
    return tleg_piecewise_expand_layers(
        gene_layers, target_depth, positions=layer_ids, strict_keys=strict_keys
    )
