#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export CLIP "learngene" layers WITHOUT training.

It will create a root folder that contains 3 subfolders:
  - last2
  - last3
  - last2_plus6

Each subfolder saves:
  - learngene_visual.pt      (selected visual transformer blocks' ORIGINAL weights)
  - learngene_text.pt        (selected text transformer blocks' ORIGINAL weights)
  - learngene_multimodal.pt  (logit_scale / projections if present)
  - selected_layers.json     (metadata)

Usage:
python get_gene.py \
  --out_root ./gene_exports \
  --clip_model ViT-B/32 \
  --device cuda
"""

import os
import json
import argparse
from typing import Dict, List

import torch
import clip


def ensure_vit_clip(model):
    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer") or not hasattr(model.visual.transformer, "resblocks"):
        raise RuntimeError("This script expects a ViT-based OpenAI CLIP: model.visual.transformer.resblocks")
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "resblocks"):
        raise RuntimeError("This script expects OpenAI CLIP text transformer: model.transformer.resblocks")


def filter_blocks_state_dict(
    sd: Dict[str, torch.Tensor],
    prefix: str,
    layers: List[int],
    skip_if_contains: List[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Keep only keys under `prefix` and whose resblock index is in `layers`.
    Example prefix:
      - "visual.transformer.resblocks."
      - "transformer.resblocks."
    """
    if skip_if_contains is None:
        skip_if_contains = []

    out = {}
    layer_set = set(layers)
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        if any(s in k for s in skip_if_contains):
            continue
        # parse index after prefix: {prefix}{idx}.
        rest = k[len(prefix):]
        # rest like "11.attn.in_proj_weight"
        idx_str = rest.split(".", 1)[0]
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if idx in layer_set:
            out[k] = v.detach().cpu()
    return out


def save_variant(
    out_dir: str,
    clip_model_name: str,
    n_visual: int,
    n_text: int,
    visual_layers: List[int],
    text_layers: List[int],
    model,
):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()

    # save only ORIGINAL CLIP weights in those blocks (skip any possible adapters)
    skip_tokens = [".adapter_attn.", ".adapter_mlp.", "lora_", ".lora."]
    visual_gene = filter_blocks_state_dict(sd, "visual.transformer.resblocks.", visual_layers, skip_if_contains=skip_tokens)
    text_gene = filter_blocks_state_dict(sd, "transformer.resblocks.", text_layers, skip_if_contains=skip_tokens)

    # multimodal params (may vary by CLIP implementation/version)
    multimodal = {}
    for key in ["logit_scale", "text_projection", "visual.proj"]:
        if key in sd:
            multimodal[key] = sd[key].detach().cpu()

    torch.save(
        {"layers": visual_layers, "state_dict": visual_gene},
        os.path.join(out_dir, "learngene_visual.pt"),
    )
    torch.save(
        {"layers": text_layers, "state_dict": text_gene},
        os.path.join(out_dir, "learngene_text.pt"),
    )
    torch.save(
        {"state_dict": multimodal},
        os.path.join(out_dir, "learngene_multimodal.pt"),
    )

    meta = {
        "clip_model": clip_model_name,
        "n_visual_blocks": n_visual,
        "n_text_blocks": n_text,
        "visual_layers": visual_layers,
        "text_layers": text_layers,
        "saved_files": [
            "learngene_visual.pt",
            "learngene_text.pt",
            "learngene_multimodal.pt",
        ],
        "notes": "No training performed. These are original CLIP weights for selected transformer blocks.",
    }
    with open(os.path.join(out_dir, "selected_layers.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="root folder that will contain last2/last3/last2_plus6")
    ap.add_argument("--clip_model", type=str, default="ViT-B/32")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--mid_layer", type=int, default=6, help="the extra layer used in last2+mid (default 6)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, _ = clip.load(args.clip_model, device=device, jit=False)
    model = model.eval()

    ensure_vit_clip(model)
    n_visual = len(model.visual.transformer.resblocks)
    n_text = len(model.transformer.resblocks)

    # layer indices
    last2_v = [n_visual - 2, n_visual - 1]
    last3_v = [n_visual - 3, n_visual - 2, n_visual - 1]
    mid = args.mid_layer
    last2_plus_mid_v = sorted(set(last2_v + ([mid] if 0 <= mid < n_visual else [])))

    last2_t = [n_text - 2, n_text - 1]
    last3_t = [n_text - 3, n_text - 2, n_text - 1]
    last2_plus_mid_t = sorted(set(last2_t + ([mid] if 0 <= mid < n_text else [])))

    # output dirs
    out_last2 = os.path.join(args.out_root, "last2")
    out_last3 = os.path.join(args.out_root, "last3")
    out_last2p6 = os.path.join(args.out_root, f"last2_plus{mid}")

    os.makedirs(args.out_root, exist_ok=True)

    save_variant(out_last2, args.clip_model, n_visual, n_text, last2_v, last2_t, model)
    save_variant(out_last3, args.clip_model, n_visual, n_text, last3_v, last3_t, model)
    save_variant(out_last2p6, args.clip_model, n_visual, n_text, last2_plus_mid_v, last2_plus_mid_t, model)

    print(f"[DONE] Exported variants to: {args.out_root}")
    print(f"  - last2:         visual={last2_v}, text={last2_t}")
    print(f"  - last3:         visual={last3_v}, text={last3_t}")
    print(f"  - last2_plus{mid}: visual={last2_plus_mid_v}, text={last2_plus_mid_t}")


if __name__ == "__main__":
    main()
