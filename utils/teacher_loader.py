"""离线 teacher 加载：本地 HF safetensors -> openai-CLIP 布局。

为何需要：原代码 7 处 `clip.load("ViT-B/32")` 会联网下载。本地只有 HF 权重
（learngene_clip/clip-vit-base-patch32/model.safetensors，且 config.json 误标
vision-only，实为全量 CLIP）。本模块把 HF state_dict 的 key 名与张量打包重映射成
openai 布局，再用 `clip.model.build_model`（从张量形状反推架构）构建 teacher，
其结构与 `clip.load` 返回的对象完全一致，下游 build_student / cmbi 零改。

依赖：torch, safetensors, clip(pip，离线安全，仅 build_model + tokenize，不下载权重)。
verify 时额外用 transformers.CLIPModel 做数值对拍。
"""
import os
from typing import Dict, Tuple, Callable

import torch


# =============================================================================
# HF -> openai state_dict 转换
# =============================================================================
def hf_to_openai_state_dict(
    hf_sd: Dict[str, torch.Tensor],
    num_vision_layers: int = 12,
    num_text_layers: int = 12,
) -> Dict[str, torch.Tensor]:
    """纯 key/张量重映射（ground truth = 反向 HF 转换脚本）。不构建模型。"""
    out: Dict[str, torch.Tensor] = {}

    def g(k):  # 取 HF 张量（缺失即报错，暴露映射问题）
        if k not in hf_sd:
            raise KeyError(f"HF state_dict 缺少键: {k}")
        return hf_sd[k]

    # ---- 标量 / 投影（投影需转置：HF Linear [out,in] -> openai x@proj [in,out]）----
    out["logit_scale"] = g("logit_scale")
    out["visual.proj"] = g("visual_projection.weight").t().contiguous()
    out["text_projection"] = g("text_projection.weight").t().contiguous()

    # ---- 文本 embedding / head ----
    out["token_embedding.weight"] = g("text_model.embeddings.token_embedding.weight")
    out["positional_embedding"] = g("text_model.embeddings.position_embedding.weight")
    out["ln_final.weight"] = g("text_model.final_layer_norm.weight")
    out["ln_final.bias"] = g("text_model.final_layer_norm.bias")

    # ---- 视觉 stem / head ----
    out["visual.class_embedding"] = g("vision_model.embeddings.class_embedding")
    out["visual.conv1.weight"] = g("vision_model.embeddings.patch_embedding.weight")
    out["visual.positional_embedding"] = g("vision_model.embeddings.position_embedding.weight")
    out["visual.ln_pre.weight"] = g("vision_model.pre_layrnorm.weight")   # 注意 HF typo
    out["visual.ln_pre.bias"] = g("vision_model.pre_layrnorm.bias")
    out["visual.ln_post.weight"] = g("vision_model.post_layernorm.weight")
    out["visual.ln_post.bias"] = g("vision_model.post_layernorm.bias")

    def convert_layers(hf_prefix: str, oa_prefix: str, n: int):
        for i in range(n):
            hp = f"{hf_prefix}.{i}."
            op = f"{oa_prefix}.{i}."
            # attention QKV 打包：HF q/k/v 分离 -> openai in_proj 拼接(顺序 q,k,v)
            qw = g(hp + "self_attn.q_proj.weight")
            kw = g(hp + "self_attn.k_proj.weight")
            vw = g(hp + "self_attn.v_proj.weight")
            qb = g(hp + "self_attn.q_proj.bias")
            kb = g(hp + "self_attn.k_proj.bias")
            vb = g(hp + "self_attn.v_proj.bias")
            out[op + "attn.in_proj_weight"] = torch.cat([qw, kw, vw], dim=0).contiguous()
            out[op + "attn.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0).contiguous()
            out[op + "attn.out_proj.weight"] = g(hp + "self_attn.out_proj.weight")
            out[op + "attn.out_proj.bias"] = g(hp + "self_attn.out_proj.bias")
            # LayerNorm 名差异
            out[op + "ln_1.weight"] = g(hp + "layer_norm1.weight")
            out[op + "ln_1.bias"] = g(hp + "layer_norm1.bias")
            out[op + "ln_2.weight"] = g(hp + "layer_norm2.weight")
            out[op + "ln_2.bias"] = g(hp + "layer_norm2.bias")
            # MLP 名差异
            out[op + "mlp.c_fc.weight"] = g(hp + "mlp.fc1.weight")
            out[op + "mlp.c_fc.bias"] = g(hp + "mlp.fc1.bias")
            out[op + "mlp.c_proj.weight"] = g(hp + "mlp.fc2.weight")
            out[op + "mlp.c_proj.bias"] = g(hp + "mlp.fc2.bias")

    convert_layers("text_model.encoder.layers", "transformer.resblocks", num_text_layers)
    convert_layers("vision_model.encoder.layers", "visual.transformer.resblocks", num_vision_layers)
    return out


def _load_local_hf_state_dict(hf_dir: str) -> Dict[str, torch.Tensor]:
    st = os.path.join(hf_dir, "model.safetensors")
    bin_ = os.path.join(hf_dir, "pytorch_model.bin")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    if os.path.isfile(bin_):
        return torch.load(bin_, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{hf_dir} 下未找到 model.safetensors 或 pytorch_model.bin")


def _count_layers(hf_sd: Dict[str, torch.Tensor], tower: str) -> int:
    import re
    pat = re.compile(rf"^{tower}\.encoder\.layers\.(\d+)\.")
    idxs = {int(m.group(1)) for k in hf_sd for m in [pat.match(k)] if m}
    return max(idxs) + 1 if idxs else 12


def _build_openai_model(converted_sd: Dict[str, torch.Tensor], device: str):
    """primary: clip.model.build_model（从形状反推架构）。失败则给出清晰诊断。"""
    try:
        import clip
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "需要 openai `clip` 包（pip 安装即可，离线安全，不下载权重）。"
            f"原始错误: {e}"
        )
    try:
        model = clip.model.build_model(converted_sd)
    except Exception as e:  # noqa: BLE001
        # 打印 key 差异辅助定位映射错误
        raise RuntimeError(f"clip.model.build_model 失败，检查 HF->openai 映射: {e}")
    return model.float().to(device).eval()


def load_teacher_offline(cfg, device: str = "cuda") -> Tuple[torch.nn.Module, Callable]:
    """返回 (teacher_openai_layout, preprocess)。完全离线，不联网。"""
    hf_dir = getattr(cfg, "LOCAL_CLIP_DIR", "./clip-vit-base-patch32")
    hf_sd = _load_local_hf_state_dict(hf_dir)
    n_v = _count_layers(hf_sd, "vision_model")
    n_t = _count_layers(hf_sd, "text_model")
    converted = hf_to_openai_state_dict(hf_sd, num_vision_layers=n_v, num_text_layers=n_t)
    teacher = _build_openai_model(converted, device)
    # preprocess 复用已有标准 CLIP 变换，不走 clip.load
    from utils.data_loader import get_clip_preprocess
    preprocess = get_clip_preprocess()
    return teacher, preprocess


# =============================================================================
# 验收：与同一本地权重的 HF CLIPModel 数值对拍
# =============================================================================
@torch.no_grad()
def verify_conversion(cfg, device: str = "cpu", image_path: str = None) -> dict:
    """图像 embedding 余弦 > 0.999 即证明视觉转换(含 QKV 打包 + proj 转置)正确。

    文本侧因 openai/HF tokenizer 不同不强制 1.0，仅作信息项。
    """
    import torch.nn.functional as F
    from PIL import Image
    import clip

    teacher, preprocess = load_teacher_offline(cfg, device)

    # HF 参照：用默认 CLIPConfig()（恰为 ViT-B/32）构建全量 CLIPModel，再 load 本地权重，
    # 绕开本地 config.json 误标 vision-only 的坑。
    from transformers import CLIPModel, CLIPConfig
    hf = CLIPModel(CLIPConfig()).to(device).float().eval()
    hf_sd = _load_local_hf_state_dict(getattr(cfg, "LOCAL_CLIP_DIR", "./clip-vit-base-patch32"))
    missing, unexpected = hf.load_state_dict(hf_sd, strict=False)
    # 只允许 position_ids 之类 buffer 不匹配
    bad = [k for k in missing if "position_ids" not in k]
    assert not bad, f"HF 参照加载缺失关键键: {bad[:5]}"

    if image_path is None:
        image_path = _first_coco_val_image(cfg)
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    ours_i = teacher.encode_image(img).float()
    hf_i = hf.get_image_features(pixel_values=img).float()
    cos_i = F.cosine_similarity(ours_i, hf_i, dim=-1).mean().item()

    # 文本：两侧各用自己的 tokenizer（信息项）
    text = "a photo of a cat"
    ours_t = teacher.encode_text(clip.tokenize([text]).to(device)).float()
    cos_t = None
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer(
            vocab_file=os.path.join(cfg.LOCAL_CLIP_DIR, "vocab.json"),
            merges_file=os.path.join(cfg.LOCAL_CLIP_DIR, "merges.txt"),
        ) if os.path.isfile(os.path.join(cfg.LOCAL_CLIP_DIR, "vocab.json")) else None
        if tok is not None:
            ids = tok([text], padding="max_length", max_length=77, return_tensors="pt").to(device)
            hf_t = hf.get_text_features(**ids).float()
            cos_t = F.cosine_similarity(ours_t, hf_t, dim=-1).mean().item()
    except Exception:  # noqa: BLE001
        cos_t = None

    result = {
        "cos_image": cos_i,
        "cos_text": cos_t,
        "logit_scale_ours": float(teacher.logit_scale.exp().item()),
        "logit_scale_hf": float(hf.logit_scale.exp().item()),
        "passed": cos_i > 0.999,
    }
    assert result["passed"], f"图像 embedding 对拍失败 cos={cos_i:.5f}（检查 QKV 打包/proj 转置）"
    return result


def _first_coco_val_image(cfg) -> str:
    d = getattr(cfg, "COCO_VAL_IMG_DIR", None)
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(d, fn)
    raise FileNotFoundError("找不到 COCO val 图片用于验收；请传 image_path=")


if __name__ == "__main__":
    from configs.base_config import config
    print(verify_conversion(config, device=config.DEVICE))
