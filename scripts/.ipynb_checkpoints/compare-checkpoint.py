# scripts/inspect_ckpt.py
# Usage:
#   1) ckpt-only inspect:
#      python -m scripts.inspect_ckpt --ckpt outputs/pretrain_ours_last3/ckpt_last.pt
#
#   2) compare with a model config (requires your repo has scripts/model_factory.py):
#      python -m scripts.inspect_ckpt --ckpt outputs/pretrain_ours_last3/ckpt_last.pt \
#         --compare --model ours --gene_dir /root/gene_exports/last3 --shallow_layers 3 \
#         --use_tleg --tleg_target_depth 6 --proj_dim 256 --proj_head mlp
#
# python -m scripts.compare \
#   --ckpt outputs/pretrain_ours_last3/ckpt_last.pt
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Any, Tuple, List, Optional

import torch


# -------------------------
# Helpers
# -------------------------
def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_state_dict_from_pt(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Return (state_dict, meta_info).
    Supports:
      - {"model": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - raw state_dict
    """
    ckpt = torch.load(path, map_location="cpu")
    meta = {"ckpt_type": type(ckpt).__name__}

    if isinstance(ckpt, dict):
        # common containers
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
            meta["container_key"] = "model"
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
            meta["container_key"] = "state_dict"
        else:
            # maybe it's already a state dict (dict[str, tensor])
            # heuristic: if any value is a tensor
            if any(torch.is_tensor(v) for v in ckpt.values()):
                state = ckpt
                meta["container_key"] = "raw_dict"
            else:
                raise RuntimeError(f"Unrecognized ckpt dict structure keys={list(ckpt.keys())[:30]}")
    else:
        raise RuntimeError(f"Unsupported ckpt type: {type(ckpt)}")

    state = _strip_module_prefix(state)
    return state, meta


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def summarize_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    n_params = 0
    n_bytes = 0
    dtypes = Counter()
    shapes = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        n_params += v.numel()
        n_bytes += tensor_nbytes(v)
        dtypes[str(v.dtype)] += 1
        shapes[k] = list(v.shape)

    # top largest tensors
    largest = sorted(
        [(k, int(tensor_nbytes(v)), list(v.shape), str(v.dtype)) for k, v in state.items() if torch.is_tensor(v)],
        key=lambda x: x[1],
        reverse=True
    )[:20]

    return {
        "num_keys": len(state),
        "num_params": int(n_params),
        "size_mb": float(n_bytes) / (1024 ** 2),
        "dtype_hist": dict(dtypes),
        "largest_tensors_top20": largest,
        "shapes": shapes,  # optionally huge; keep for --dump_json
    }


def prefix_group_counts(keys: List[str], depth: int = 2) -> List[Tuple[str, int]]:
    c = Counter()
    for k in keys:
        parts = k.split(".")
        pref = ".".join(parts[:depth]) if len(parts) >= depth else k
        c[pref] += 1
    return c.most_common(50)


def find_max_block_index(keys: List[str], module_hint: str = "") -> Optional[int]:
    """
    Try to infer max 'blocks.<idx>' index from keys.
    If module_hint provided, only search keys containing that substring.
    """
    pat = re.compile(r"(?:^|\.)(?:blocks|resblocks)\.(\d+)(?:\.|$)")
    mx = None
    for k in keys:
        if module_hint and (module_hint not in k):
            continue
        m = pat.search(k)
        if m:
            idx = int(m.group(1))
            mx = idx if mx is None else max(mx, idx)
    return mx


def grep_keys(keys: List[str], contains: List[str]) -> List[str]:
    out = []
    for k in keys:
        ok = True
        for s in contains:
            if s not in k:
                ok = False
                break
        if ok:
            out.append(k)
    return out


def pretty_list(items: List[str], limit: int = 30) -> str:
    if not items:
        return "  (none)"
    head = items[:limit]
    s = "\n".join([f"  - {x}" for x in head])
    if len(items) > limit:
        s += f"\n  ... ({len(items) - limit} more)"
    return s


# -------------------------
# Optional: compare with current model code
# -------------------------
def try_build_model_and_compare(args, state: Dict[str, torch.Tensor]) -> None:
    """
    Compare checkpoint keys with the model created by create_model_bundle(args).
    Requires your repo has scripts/model_factory.py and scripts/utils/misc.py
    """
    try:
        from scripts.model_factory import create_model_bundle
        from scripts.utils.misc import unwrap_model
    except Exception as e:
        print(f"[COMPARE] Import failed: {e.__class__.__name__}: {e}")
        print("[COMPARE] Skip compare. (Make sure you're running inside your repo.)")
        return

    # Build model
    bundle = create_model_bundle(args)
    model = bundle.model
    base_model = unwrap_model(model)

    missing, unexpected = base_model.load_state_dict(state, strict=False)

    print("\n==============================")
    print("== Compare ckpt <-> model   ==")
    print("==============================")
    print(f"[COMPARE] model={args.model}")
    print(f"[COMPARE] missing={len(missing)} unexpected={len(unexpected)}")

    # show heads
    if len(missing) > 0:
        print("\n[COMPARE] Missing keys (first 50):")
        print(pretty_list(list(missing), limit=50))
    if len(unexpected) > 0:
        print("\n[COMPARE] Unexpected keys (first 50):")
        print(pretty_list(list(unexpected), limit=50))

    # quick diagnosis heuristics
    def frac_contains(lst, token: str) -> float:
        if not lst:
            return 0.0
        return sum(1 for x in lst if token in x) / float(len(lst))

    if len(missing) > 0:
        fm_tleg = frac_contains(missing, "tleg")
        fm_proj = frac_contains(missing, "proj")
        fm_bottleneck = frac_contains(missing, "bottleneck")
        fm_gene = frac_contains(missing, "gene")

        print("\n[COMPARE] Heuristic diagnosis (based on missing keys tokens):")
        print(f"  - missing contains 'tleg'      : {fm_tleg:.2%}")
        print(f"  - missing contains 'proj'      : {fm_proj:.2%}")
        print(f"  - missing contains 'bottleneck': {fm_bottleneck:.2%}")
        print(f"  - missing contains 'gene'      : {fm_gene:.2%}")

        if fm_tleg > 0.30:
            print("  => 很可能是 use_tleg / tleg_target_depth 与训练时不一致（eval/build 开了 tleg，但 ckpt 里没有）。")
        if fm_proj > 0.30:
            print("  => 很可能是 proj_head/proj_dim/proj_hidden_dim 与训练时不一致。")
        if fm_bottleneck > 0.10:
            print("  => 很可能是 bottleneck_dim/bottleneck_dropout 开关与训练时不一致。")
        if fm_gene > 0.10:
            print("  => 很可能是 gene_dir（last2/last3/26）对应结构与训练时不一致，或 gene 相关命名变更。")

    print("==============================\n")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser("Inspect a .pt checkpoint structure by state_dict keys + optional compare.")

    parser.add_argument("--ckpt", type=str, required=True, help="path to .pt checkpoint")
    parser.add_argument("--dump_json", type=str, default="", help="save a JSON report to this path")
    parser.add_argument("--prefix_depth", type=int, default=2, help="group key prefix depth for summary")
    parser.add_argument("--show_shapes", action="store_true", help="print example shapes for some important keys")

    # compare mode (optional)
    parser.add_argument("--compare", action="store_true", help="also build model using create_model_bundle and compare")

    # these args mimic your repo's create_model_bundle signature (minimal subset)
    parser.add_argument("--model", type=str, default="ours", choices=["ours", "clip", "tinyclip"])
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--shallow_layers", type=int, default=3)
    parser.add_argument("--bottleneck_dim", type=int, default=-1)
    parser.add_argument("--bottleneck_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_head", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--proj_hidden_dim", type=int, default=-1)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    parser.add_argument("--use_tleg", action="store_true", default=False)
    parser.add_argument("--tleg_target_depth", type=int, default=6)
    parser.add_argument("--use_multimodal_init", action="store_true", default=False)

    parser.add_argument("--disable_stem_init_from_clip", action="store_true", default=False)
    parser.add_argument("--stem_init_clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--freeze_stem_after_init", action="store_true", default=False)

    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--tinyclip_ckpt", type=str, default="")

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    state, meta = load_state_dict_from_pt(args.ckpt)
    keys = sorted(list(state.keys()))

    # core summary
    summ = summarize_state(state)
    print("===================================")
    print("== Checkpoint key structure report ==")
    print("===================================")
    print(f"[CKPT] path: {args.ckpt}")
    print(f"[CKPT] container: {meta.get('container_key', 'unknown')}  type: {meta.get('ckpt_type', 'unknown')}")
    print(f"[CKPT] num_keys: {summ['num_keys']}")
    print(f"[CKPT] num_params: {summ['num_params']:,}")
    print(f"[CKPT] size_mb: {summ['size_mb']:.2f} MB")
    print(f"[CKPT] dtype_hist: {summ['dtype_hist']}")

    # group by prefix
    print("\n== Top prefixes (by key count) ==")
    for pref, cnt in prefix_group_counts(keys, depth=args.prefix_depth):
        print(f"  {pref:<40} {cnt:>6}")

    # special token scans
    tokens = ["tleg", "gene", "proj", "bottleneck", "vision", "text", "logit_scale"]
    print("\n== Token presence (how many keys contain token) ==")
    for t in tokens:
        c = sum(1 for k in keys if t in k)
        print(f"  {t:<12}: {c}")

    # infer some "depth-ish" info from blocks indices
    hints = [
        ("vision_tower", "vision_tower"),
        ("text_tower", "text_tower"),
        ("vision", "vision"),
        ("text", "text"),
        ("gene", "gene"),
    ]
    print("\n== Infer max block/resblock index (best-effort) ==")
    for name, hint in hints:
        mx = find_max_block_index(keys, module_hint=hint)
        if mx is not None:
            print(f"  {name:<12}: max_index={mx}  (=> count approx {mx+1} blocks)")
    # show some important keys
    print("\n== Example keys for important patterns ==")
    for pattern in [["tleg"], ["proj"], ["bottleneck"], ["gene"], ["logit_scale"]]:
        matched = grep_keys(keys, pattern)
        print(f"\n[Keys containing {'+'.join(pattern)}] count={len(matched)}")
        print(pretty_list(matched, limit=30))

    # show shape samples (optional)
    if args.show_shapes:
        sample_keys = []
        # common candidates
        for cand in [
            "logit_scale",
            "vision_stem.conv1.weight",
            "text_stem.token_embedding.weight",
            "proj.weight",
            "proj_head",
        ]:
            for k in keys:
                if cand in k:
                    sample_keys.append(k)
            if len(sample_keys) > 40:
                break
        sample_keys = sample_keys[:40]
        print("\n== Sample shapes ==")
        for k in sample_keys:
            v = state[k]
            if torch.is_tensor(v):
                print(f"  {k:<70} shape={tuple(v.shape)} dtype={v.dtype}")

    # dump json (optional)
    if args.dump_json:
        report = {
            "ckpt_path": args.ckpt,
            "meta": meta,
            "summary": {k: v for k, v in summ.items() if k != "shapes"},
            "prefix_top": prefix_group_counts(keys, depth=args.prefix_depth),
            "token_counts": {t: sum(1 for k in keys if t in k) for t in tokens},
            "max_block_index": {name: find_max_block_index(keys, module_hint=hint) for name, hint in hints},
            "keys": keys,
            "shapes": summ["shapes"],
        }
        os.makedirs(os.path.dirname(args.dump_json) or ".", exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[SAVE] JSON report -> {args.dump_json}")

    # optional compare
    if args.compare:
        # align with your repo's expectation
        args.stem_init_from_clip = (not args.disable_stem_init_from_clip)
        try_build_model_and_compare(args, state)


if __name__ == "__main__":
    main()
