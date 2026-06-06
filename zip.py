#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pack_hf_models.py
在有网环境中将 Hugging Face 模型仓库下载为标准目录并打包为 .tar.gz，便于离线环境（如 AutoDL）使用。

用法示例：
  # 仅打包 TinyCLIP（推荐学生模型）
  python pack_hf_models.py --out ~/hf_pkgs \
    --model wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M

  # 同时打包 TinyCLIP + 教师 OpenCLIP ViT-B/16
  python pack_hf_models.py --out ~/hf_pkgs \
    --model wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M \
    --model laion/CLIP-ViT-B-16-laion2B-s34B-b88K
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

REQUIRED_FILES = [
    "config.json",
    # 至少存在以下任意一个权重文件：
    # "model.safetensors" / "pytorch_model.bin" / "tf_model.h5" / "model.ckpt.index" / "flax_model.msgpack"
]

WEIGHT_CANDIDATES = [
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
]

def ensure_deps():
    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("[!] 缺少依赖 huggingface_hub，请先安装：")
        print("    pip install -U huggingface_hub")
        sys.exit(1)

def snapshot(repo_id: str, out_root: Path, revision: str = None) -> Path:
    from huggingface_hub import snapshot_download

    # local_dir 取仓库名最后一段（避免 / 造成嵌套）
    name = repo_id.split("/")[-1]
    local_dir = (out_root / name).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] 下载 {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # 避免软链接在拷贝后失效
    )
    return local_dir

def is_valid_model_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    # 必须有 config.json
    if not (d / "config.json").is_file():
        return False
    # 至少存在一个权重文件
    for w in WEIGHT_CANDIDATES:
        if (d / w).is_file():
            return True
    return False

def make_tar_gz(src_dir: Path, out_root: Path) -> Path:
    tar_path = out_root / f"{src_dir.name}.tar.gz"
    print(f"[+] 打包 {src_dir} -> {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        # 将目录内容整体打包（解包后得到 src_dir/… 结构）
        tar.add(src_dir, arcname=src_dir.name)
    print(f"[✓] 完成：{tar_path}")
    return tar_path

def human_size(p: Path) -> str:
    try:
        total = 0
        for q in p.rglob("*"):
            if q.is_file():
                total += q.stat().st_size
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if total < 1024 or unit == "TB":
                return f"{total:.1f}{unit}"
            total /= 1024.0
    except Exception:
        return "unknown"

def main():
    ensure_deps()

    parser = argparse.ArgumentParser(description="下载并打包 Hugging Face 模型用于离线环境")
    parser.add_argument("--out", type=str, default="./swin", help="输出目录，用于保存模型目录与 .tar.gz")
    parser.add_argument("--model", type=str, action="append", default=["wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"],
                        help="模型仓库名（可重复传多次），如 wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
    parser.add_argument("--revision", type=str, default=None,
                        help="可选：指定模型版本（tag/commit），默认最新")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 可选：指定缓存目录，避免写入不期望路径
    os.environ.setdefault("HF_HOME", str(out_root / "_hf_cache"))

    ok_list = []
    for repo in args.model:
        try:
            model_dir = snapshot(repo, out_root, revision=args.revision)
            if not is_valid_model_dir(model_dir):
                print(f"[!] 目录结构缺少必要文件：{model_dir}")
                print("    需要至少包含 config.json 与一个权重文件（model.safetensors / pytorch_model.bin 等）")
                sys.exit(2)
            sz = human_size(model_dir)
            print(f"[i] 已下载：{model_dir}  (约 {sz})")
            tar_path = make_tar_gz(model_dir, out_root)
            ok_list.append((repo, model_dir, tar_path))
        except Exception as e:
            print(f"[x] 处理 {repo} 失败：{e}")
            sys.exit(3)

    print("\n=== 打包完成 ===")
    for repo, d, t in ok_list:
        print(f" - {repo}\n   目录: {d}\n   压缩: {t}")

    print("\n将上述 .tar.gz 拷到 AutoDL，例如放到 /root/autodl-tmp/ 后解包：")
    print("  cd /root/autodl-tmp && tar -xzf <文件名>.tar.gz")
    print("然后在离线环境使用：")
    print("  cfg.TINYCLIP_HF_ID = \"/root/autodl-tmp/<模型目录名>\"")
    print("  CLIPModel.from_pretrained(cfg.TINYCLIP_HF_ID, local_files_only=True)")

if __name__ == "__main__":
    main()
