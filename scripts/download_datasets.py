from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tasks.dataset_registry import DEFAULT_DATA_ROOT


COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations_trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "image_info_test2017": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
}

KARPATHY_CAPTION_DATASETS_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)

IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _path_exists(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()) if path.is_dir() else True)


def download_url(url: str, dst: Path, force: bool = False, dry_run: bool = False) -> Path:
    _mkdir(dst.parent)
    if dst.exists() and not force:
        _log(f"[SKIP] {dst} exists")
        return dst

    _log(f"[GET] {url}")
    _log(f"      -> {dst}")
    if dry_run:
        return dst

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ClipGeneDatasetDownloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", "0") or "0")
        done = 0
        last_print = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if total > 0 and now - last_print > 2.0:
                    pct = 100.0 * done / float(total)
                    _log(f"      {done / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB ({pct:.1f}%)")
                    last_print = now
    tmp.replace(dst)
    return dst


def _safe_target(root: Path, member_name: str) -> Path:
    root_abs = root.resolve()
    target = (root / member_name).resolve()
    if os.path.commonpath([str(root_abs), str(target)]) != str(root_abs):
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target


def extract_zip(zip_path: Path, dst: Path, force: bool = False, dry_run: bool = False) -> None:
    _mkdir(dst)
    _log(f"[UNZIP] {zip_path} -> {dst}")
    if dry_run:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        top_levels = set()
        for info in infos:
            _safe_target(dst, info.filename)
            name = info.filename.strip("/\\")
            if name:
                top_levels.add(name.split("/")[0].split("\\")[0])
        if top_levels and not force and all((dst / name).exists() for name in top_levels):
            _log(f"[SKIP] archive contents already present under {dst}")
            return
        zf.extractall(dst)


def extract_tar(tar_path: Path, dst: Path, force: bool = False, dry_run: bool = False) -> None:
    _mkdir(dst)
    _log(f"[UNTAR] {tar_path} -> {dst}")
    if dry_run:
        return
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        top_levels = set()
        for member in members:
            _safe_target(dst, member.name)
            name = member.name.strip("/\\")
            if name:
                top_levels.add(name.split("/")[0].split("\\")[0])
        if top_levels and not force and all((dst / name).exists() for name in top_levels):
            _log(f"[SKIP] archive contents already present under {dst}")
            return
        tf.extractall(dst, members=members)


def extract_archive(archive: Path, dst: Path, force: bool = False, dry_run: bool = False) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        extract_zip(archive, dst, force=force, dry_run=dry_run)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        extract_tar(archive, dst, force=force, dry_run=dry_run)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def copy_if_needed(src: Path, dst: Path, force: bool = False, dry_run: bool = False) -> None:
    if dst.exists() and not force:
        _log(f"[SKIP] {dst} exists")
        return
    _mkdir(dst.parent)
    _log(f"[COPY] {src} -> {dst}")
    if not dry_run:
        shutil.copy2(src, dst)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG"}
    return sum(1 for p in path.rglob("*") if p.suffix in exts)


def find_best_image_dir(root: Path) -> Optional[Path]:
    best: Tuple[int, Optional[Path]] = (0, None)
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        n = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if n > best[0]:
            best = (n, d)
    return best[1]


def link_or_copy_dir(src: Path, dst: Path, force: bool = False, dry_run: bool = False) -> None:
    if dst.exists() and count_images(dst) > 0 and not force:
        _log(f"[SKIP] image directory ready: {dst}")
        return
    if dst.exists() and force and not dry_run:
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    _mkdir(dst.parent)
    _log(f"[LINK] {src} -> {dst}")
    if dry_run:
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        _log("[WARN] symlink failed; falling back to copytree")
        shutil.copytree(src, dst)


def run_command(cmd: Sequence[str], dry_run: bool = False) -> None:
    _log("[RUN] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def write_note(path: Path, title: str, lines: Iterable[str], dry_run: bool = False) -> None:
    _mkdir(path.parent)
    text = "\n".join([title, "=" * len(title), "", *lines, ""]) + "\n"
    _log(f"[NOTE] {path}")
    if not dry_run:
        path.write_text(text, encoding="utf-8")


def download_coco(args) -> None:
    root = Path(args.root)
    coco_dir = root / "coco"
    dl_dir = root / "_downloads" / "coco"
    _mkdir(dl_dir)

    for split in args.coco_splits:
        url = COCO_URLS[split]
        zip_path = download_url(url, dl_dir / f"{split}.zip", force=args.force, dry_run=args.dry_run)
        if not args.no_extract:
            extract_zip(zip_path, coco_dir, force=args.force, dry_run=args.dry_run)

    if args.coco_annotations:
        zip_path = download_url(
            COCO_URLS["annotations_trainval2017"],
            dl_dir / "annotations_trainval2017.zip",
            force=args.force,
            dry_run=args.dry_run,
        )
        if not args.no_extract:
            extract_zip(zip_path, coco_dir, force=args.force, dry_run=args.dry_run)

    if "test2017" in args.coco_splits and args.coco_test_info:
        zip_path = download_url(
            COCO_URLS["image_info_test2017"],
            dl_dir / "image_info_test2017.zip",
            force=args.force,
            dry_run=args.dry_run,
        )
        if not args.no_extract:
            extract_zip(zip_path, coco_dir, force=args.force, dry_run=args.dry_run)


def download_karpathy(args) -> None:
    root = Path(args.root)
    dl_dir = root / "_downloads" / "karpathy"
    out_dir = root / "karpathy"
    _mkdir(dl_dir)
    zip_path = download_url(
        KARPATHY_CAPTION_DATASETS_URL,
        dl_dir / "caption_datasets.zip",
        force=args.force,
        dry_run=args.dry_run,
    )
    extract_zip(zip_path, out_dir, force=args.force, dry_run=args.dry_run)

    if args.dry_run:
        return

    coco_src = out_dir / "dataset_coco.json"
    flickr_src = out_dir / "dataset_flickr30k.json"
    if coco_src.exists():
        copy_if_needed(coco_src, root / "coco" / "annotations" / "dataset_coco.json", force=args.force)
    if flickr_src.exists():
        copy_if_needed(
            flickr_src,
            root / "flickr30k" / "annotations" / "dataset_flickr30k.json",
            force=args.force,
        )


def _download_cc3m_from_hf(args) -> None:
    out_dir = Path(args.root) / "cc3m" / "wds"
    _mkdir(out_dir)
    _log(f"[HF] dataset={args.cc3m_hf_repo} -> {out_dir}")
    if args.dry_run:
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "CC3M HF WebDataset download requires `pip install huggingface_hub`."
        ) from exc

    snapshot_download(
        repo_id=args.cc3m_hf_repo,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=args.cc3m_hf_allow_patterns,
        token=args.hf_token or None,
    )


def _prepare_cc3m_tsv(src: str, args) -> Path:
    if not src:
        raise ValueError("--cc3m_tsv is required when --cc3m_source img2dataset")
    root = Path(args.root)
    dl_dir = root / "_downloads" / "cc3m"
    _mkdir(dl_dir)
    if src.startswith(("http://", "https://")):
        raw = download_url(src, dl_dir / Path(src).name, force=args.force, dry_run=args.dry_run)
    else:
        raw = Path(src)
    prepared = dl_dir / "Train_GCC-training.with_header.tsv"
    _log(f"[TSV] preparing {raw} -> {prepared}")
    if args.dry_run:
        return prepared
    with open(raw, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        rest = f.read()
    if first.strip().lower().replace(" ", "") in {"caption\turl", "url\tcaption"}:
        text = first + rest
    else:
        text = "caption\turl\n" + first + rest
    prepared.write_text(text, encoding="utf-8")
    return prepared


def _download_cc3m_with_img2dataset(args) -> None:
    out_dir = Path(args.root) / "cc3m" / "wds"
    _mkdir(out_dir)
    tsv = _prepare_cc3m_tsv(args.cc3m_tsv, args)
    cmd = [
        "img2dataset",
        "--url_list",
        str(tsv),
        "--input_format",
        "tsv",
        "--url_col",
        "url",
        "--caption_col",
        "caption",
        "--output_format",
        "webdataset",
        "--output_folder",
        str(out_dir),
        "--processes_count",
        str(args.cc3m_processes),
        "--thread_count",
        str(args.cc3m_threads),
        "--image_size",
        str(args.cc3m_image_size),
        "--resize_mode",
        args.cc3m_resize_mode,
    ]
    run_command(cmd, dry_run=args.dry_run)


def download_cc3m(args) -> None:
    if args.cc3m_source == "hf_wds":
        _download_cc3m_from_hf(args)
    elif args.cc3m_source == "img2dataset":
        _download_cc3m_with_img2dataset(args)
    elif args.cc3m_source == "existing":
        path = Path(args.cc3m_existing_dir)
        if not path.exists():
            raise FileNotFoundError(f"--cc3m_existing_dir not found: {path}")
        target = Path(args.root) / "cc3m" / "wds"
        link_or_copy_dir(path, target, force=args.force, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown cc3m_source={args.cc3m_source}")


def download_flickr30k(args) -> None:
    # The captions/splits are in the Karpathy bundle and are safe to download.
    download_karpathy(args)

    root = Path(args.root)
    flickr_dir = root / "flickr30k"
    image_target = flickr_dir / "images"
    dl_dir = root / "_downloads" / "flickr30k"
    _mkdir(dl_dir)

    archive: Optional[Path] = None
    if args.flickr30k_source == "archive":
        if not args.flickr30k_archive:
            raise ValueError("--flickr30k_archive is required when --flickr30k_source archive")
        archive = Path(args.flickr30k_archive)
    elif args.flickr30k_source == "url":
        if not args.flickr30k_url:
            raise ValueError("--flickr30k_url is required when --flickr30k_source url")
        archive = download_url(
            args.flickr30k_url,
            dl_dir / Path(args.flickr30k_url).name,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif args.flickr30k_source == "kaggle":
        run_command(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                args.flickr30k_kaggle_dataset,
                "-p",
                str(dl_dir),
            ],
            dry_run=args.dry_run,
        )
        zips = sorted(dl_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        archive = zips[-1] if zips else None
    elif args.flickr30k_source == "manual":
        note = flickr_dir / "README_DOWNLOAD.txt"
        write_note(
            note,
            "Flickr30k image download required",
            [
                "The Karpathy split annotation is downloaded automatically.",
                "Flickr30k images may require accepting the dataset license.",
                "Put a local archive on the server and rerun:",
                "python -m scripts.download_datasets --datasets flickr30k --flickr30k_source archive "
                "--flickr30k_archive /path/to/flickr30k-images.zip",
                "Expected final path: <root>/flickr30k/images/*.jpg",
            ],
            dry_run=args.dry_run,
        )
        if args.strict:
            raise RuntimeError("Flickr30k image archive is required in --strict mode.")
        return

    if archive is None or (not args.dry_run and not archive.exists()):
        raise FileNotFoundError(f"Flickr30k archive not found: {archive}")

    extracted = flickr_dir / "_extracted"
    extract_archive(archive, extracted, force=args.force, dry_run=args.dry_run)
    if args.dry_run:
        return
    best = find_best_image_dir(extracted)
    if best is None or count_images(best) == 0:
        raise RuntimeError(f"Could not locate Flickr30k images under {extracted}")
    link_or_copy_dir(best, image_target, force=args.force)


def download_imagenet(args) -> None:
    root = Path(args.root)
    imagenet_dir = root / "imagenet"
    _mkdir(imagenet_dir)

    if args.imagenet_source == "archive":
        if not args.imagenet_val_archive:
            raise ValueError("--imagenet_val_archive is required when --imagenet_source archive")
        extract_archive(Path(args.imagenet_val_archive), imagenet_dir / "val", force=args.force, dry_run=args.dry_run)
    else:
        write_note(
            imagenet_dir / "README_DOWNLOAD.txt",
            "ImageNet local archive required",
            [
                "ImageNet requires license-gated access, so this script does not fetch images from a public URL.",
                "Download ILSVRC2012 validation images from the official portal, copy the archive to the server, then run:",
                "python -m scripts.download_datasets --datasets imagenet --imagenet_source archive "
                "--imagenet_val_archive /path/to/ILSVRC2012_img_val.tar "
                "--imagenet_val_labels /path/to/ImageNet_val_label.txt",
                "Expected final path for scripts.imagenet_zs: <root>/imagenet/val plus ImageNet_val_label.txt.",
            ],
            dry_run=args.dry_run,
        )
        if args.strict:
            raise RuntimeError("ImageNet archive is required in --strict mode.")

    class_index_dst = imagenet_dir / "imagenet_class_index.json"
    if args.imagenet_class_index_json:
        copy_if_needed(Path(args.imagenet_class_index_json), class_index_dst, force=args.force, dry_run=args.dry_run)
    else:
        download_url(IMAGENET_CLASS_INDEX_URL, class_index_dst, force=args.force, dry_run=args.dry_run)

    labels_dst = imagenet_dir / "ImageNet_val_label.txt"
    if args.imagenet_val_labels:
        copy_if_needed(Path(args.imagenet_val_labels), labels_dst, force=args.force, dry_run=args.dry_run)

    if not args.dry_run and labels_dst.exists() and (imagenet_dir / "val").exists():
        prepare_imagenet_imagefolder(imagenet_dir / "val", labels_dst, imagenet_dir / "val_imagefolder", force=args.force)

    if not args.dry_run and class_index_dst.exists():
        data = json.loads(class_index_dst.read_text(encoding="utf-8"))
        names = []
        for i in range(1000):
            entry = data.get(str(i), ["", f"class_{i}"])
            name = str(entry[1] if isinstance(entry, list) and len(entry) > 1 else entry)
            names.append(name.replace("_", " "))
        (imagenet_dir / "imagenet_classnames.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def prepare_imagenet_imagefolder(flat_val_dir: Path, labels_file: Path, out_dir: Path, force: bool = False) -> None:
    """Create an ImageFolder-compatible val tree from flat val images and labels."""
    if out_dir.exists() and count_images(out_dir) > 0 and not force:
        _log(f"[SKIP] ImageFolder tree ready: {out_dir}")
        return
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    _mkdir(out_dir)
    _log(f"[IMAGENET] preparing ImageFolder tree: {out_dir}")
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename, synset = parts[0], parts[1]
            src = flat_val_dir / filename
            if not src.exists():
                continue
            dst_dir = out_dir / synset
            _mkdir(dst_dir)
            dst = dst_dir / filename
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def download_cifar100(args) -> None:
    root = Path(args.root) / "cifar100"
    _mkdir(root)
    if args.dry_run:
        _log(f"[DRY] torchvision.datasets.CIFAR100(root={root}, download=True)")
        return
    try:
        import torchvision
    except Exception as exc:
        raise RuntimeError("CIFAR100 download requires torchvision.") from exc
    torchvision.datasets.CIFAR100(root=str(root), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(root), train=False, download=True)


def _paths_for_root(root: str) -> Dict[str, Dict[str, str]]:
    r = root.rstrip("/\\")
    return {
        "pretrain": {
            "cc3m_root": f"{r}/cc3m/wds",
            "coco_images": f"{r}/coco/train2017",
            "coco_captions": f"{r}/coco/annotations/captions_train2017.json",
        },
        "findgene": {
            "coco_img_dir": f"{r}/coco/train2017",
            "coco_ann_file": f"{r}/coco/annotations/captions_train2017.json",
        },
        "itm": {
            "coco_images": f"{r}/coco/train2017",
            "coco_captions": f"{r}/coco/annotations/captions_train2017.json",
            "coco_val_images": f"{r}/coco/val2017",
            "coco_val_captions": f"{r}/coco/annotations/captions_val2017.json",
            "coco_test_images": f"{r}/coco/val2017",
            "coco_test_captions": f"{r}/coco/annotations/captions_val2017.json",
            "flickr_images": f"{r}/flickr30k/images",
            "flickr_karpathy_json": f"{r}/flickr30k/annotations/dataset_flickr30k.json",
        },
        "retrieval": {
            "coco_images": f"{r}/coco/train2017",
            "coco_captions": f"{r}/coco/annotations/captions_train2017.json",
            "eval_coco_images": f"{r}/coco/train2017",
            "eval_coco_karpathy_json": f"{r}/coco/annotations/dataset_coco.json",
            "flickr_images": f"{r}/flickr30k/images",
            "flickr_ann": f"{r}/flickr30k/annotations/dataset_flickr30k.json",
            "eval_flickr_images": f"{r}/flickr30k/images",
            "eval_flickr_karpathy_json": f"{r}/flickr30k/annotations/dataset_flickr30k.json",
        },
        "multilabel": {
            "coco_train_img_dir": f"{r}/coco/train2017",
            "coco_train_instances_json": f"{r}/coco/annotations/instances_train2017.json",
            "coco_val_img_dir": f"{r}/coco/val2017",
            "coco_val_instances_json": f"{r}/coco/annotations/instances_val2017.json",
        },
        "imagenet": {
            "imagenet_val_dir": f"{r}/imagenet/val",
            "imagenet_val_imagefolder_dir": f"{r}/imagenet/val_imagefolder",
            "imagenet_val_labels": f"{r}/imagenet/ImageNet_val_label.txt",
            "class_index_json": f"{r}/imagenet/imagenet_class_index.json",
            "classnames_txt": f"{r}/imagenet/imagenet_classnames.txt",
        },
        "cifar100": {
            "data_root": f"{r}/cifar100",
        },
    }


def write_manifests(args) -> None:
    root = Path(args.root)
    paths = _paths_for_root(str(root))
    manifest = {
        "data_root": str(root),
        "env": {"CLIPGENE_DATA_ROOT": str(root)},
        "paths": paths,
    }
    _mkdir(root)
    json_path = root / "clipgene_data_paths.json"
    sh_path = root / "clipgene_data_env.sh"
    ps1_path = root / "clipgene_data_env.ps1"
    _log(f"[WRITE] {json_path}")
    if not args.dry_run:
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        sh_path.write_text(
            "\n".join(
                [
                    f"export CLIPGENE_DATA_ROOT='{root}'",
                    f"export CC3M_ROOT='{paths['pretrain']['cc3m_root']}'",
                    f"export COCO_TRAIN_IMG='{paths['itm']['coco_images']}'",
                    f"export COCO_VAL_IMG='{paths['itm']['coco_val_images']}'",
                    f"export COCO_CAP_TRAIN='{paths['itm']['coco_captions']}'",
                    f"export COCO_CAP_VAL='{paths['itm']['coco_val_captions']}'",
                    f"export COCO_INST_TRAIN='{paths['multilabel']['coco_train_instances_json']}'",
                    f"export COCO_INST_VAL='{paths['multilabel']['coco_val_instances_json']}'",
                    f"export COCO_KARPATHY_JSON='{paths['retrieval']['eval_coco_karpathy_json']}'",
                    f"export FLICKR_IMG='{paths['retrieval']['flickr_images']}'",
                    f"export FLICKR_JSON='{paths['retrieval']['flickr_ann']}'",
                    f"export IMAGENET_VAL_DIR='{paths['imagenet']['imagenet_val_dir']}'",
                    f"export IMAGENET_VAL_IMAGEFOLDER_DIR='{paths['imagenet']['imagenet_val_imagefolder_dir']}'",
                    f"export IMAGENET_VAL_LABELS='{paths['imagenet']['imagenet_val_labels']}'",
                    f"export IMAGENET_CLASS_INDEX_JSON='{paths['imagenet']['class_index_json']}'",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ps1_path.write_text(
            "\n".join(
                [
                    f"$env:CLIPGENE_DATA_ROOT = '{root}'",
                    f"$env:CC3M_ROOT = '{paths['pretrain']['cc3m_root']}'",
                    f"$env:COCO_TRAIN_IMG = '{paths['itm']['coco_images']}'",
                    f"$env:COCO_VAL_IMG = '{paths['itm']['coco_val_images']}'",
                    f"$env:COCO_CAP_TRAIN = '{paths['itm']['coco_captions']}'",
                    f"$env:COCO_CAP_VAL = '{paths['itm']['coco_val_captions']}'",
                    f"$env:COCO_INST_TRAIN = '{paths['multilabel']['coco_train_instances_json']}'",
                    f"$env:COCO_INST_VAL = '{paths['multilabel']['coco_val_instances_json']}'",
                    f"$env:COCO_KARPATHY_JSON = '{paths['retrieval']['eval_coco_karpathy_json']}'",
                    f"$env:FLICKR_IMG = '{paths['retrieval']['flickr_images']}'",
                    f"$env:FLICKR_JSON = '{paths['retrieval']['flickr_ann']}'",
                    f"$env:IMAGENET_VAL_DIR = '{paths['imagenet']['imagenet_val_dir']}'",
                    f"$env:IMAGENET_VAL_IMAGEFOLDER_DIR = '{paths['imagenet']['imagenet_val_imagefolder_dir']}'",
                    f"$env:IMAGENET_VAL_LABELS = '{paths['imagenet']['imagenet_val_labels']}'",
                    f"$env:IMAGENET_CLASS_INDEX_JSON = '{paths['imagenet']['class_index_json']}'",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    _log(f"[WRITE] {sh_path}")
    _log(f"[WRITE] {ps1_path}")


def parse_args():
    p = argparse.ArgumentParser("Download/localize datasets used by ClipGene experiments")
    p.add_argument("--root", type=str, default=DEFAULT_DATA_ROOT, help="Local dataset root on the server.")
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "coco", "karpathy", "cc3m", "flickr30k", "imagenet", "cifar100", "manifest"],
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--strict", action="store_true", help="Fail if license-gated datasets are not provided.")
    p.add_argument("--no_extract", action="store_true")

    p.add_argument("--coco_splits", type=str, nargs="+", default=["train2017", "val2017"], choices=list(COCO_URLS.keys())[:3])
    p.add_argument("--coco_annotations", action="store_true", default=True)
    p.add_argument("--no_coco_annotations", dest="coco_annotations", action="store_false")
    p.add_argument("--coco_test_info", action="store_true")

    p.add_argument("--cc3m_source", type=str, default="hf_wds", choices=["hf_wds", "img2dataset", "existing"])
    p.add_argument("--cc3m_hf_repo", type=str, default="pixparse/cc3m-wds")
    p.add_argument("--cc3m_hf_allow_patterns", type=str, nargs="+", default=["*.tar"])
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--cc3m_tsv", type=str, default="", help="Local path or URL to Train_GCC-training.tsv.")
    p.add_argument("--cc3m_existing_dir", type=str, default="")
    p.add_argument("--cc3m_processes", type=int, default=16)
    p.add_argument("--cc3m_threads", type=int, default=64)
    p.add_argument("--cc3m_image_size", type=int, default=256)
    p.add_argument("--cc3m_resize_mode", type=str, default="keep_ratio")

    p.add_argument("--flickr30k_source", type=str, default="manual", choices=["manual", "archive", "url", "kaggle"])
    p.add_argument("--flickr30k_archive", type=str, default="")
    p.add_argument("--flickr30k_url", type=str, default="")
    p.add_argument("--flickr30k_kaggle_dataset", type=str, default="hsankesara/flickr-image-dataset")

    p.add_argument("--imagenet_source", type=str, default="manual", choices=["manual", "archive"])
    p.add_argument("--imagenet_val_archive", type=str, default="")
    p.add_argument("--imagenet_val_labels", type=str, default="")
    p.add_argument("--imagenet_class_index_json", type=str, default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.datasets)
    if "all" in selected:
        selected = {"coco", "karpathy", "cc3m", "flickr30k", "imagenet", "cifar100", "manifest"}

    if "coco" in selected:
        download_coco(args)
    if "karpathy" in selected:
        download_karpathy(args)
    if "cc3m" in selected:
        download_cc3m(args)
    if "flickr30k" in selected:
        download_flickr30k(args)
    if "imagenet" in selected:
        download_imagenet(args)
    if "cifar100" in selected:
        download_cifar100(args)
    if "manifest" in selected:
        write_manifests(args)


if __name__ == "__main__":
    main()
