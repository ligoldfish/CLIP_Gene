# ClipGene Dataset Download Guide

All dataset paths are centralized under `CLIPGENE_DATA_ROOT`.

Default root:

```bash
export CLIPGENE_DATA_ROOT=/root/autodl-tmp/clipgene_data
```

## One-Command Setup

Download all automatically available datasets and write path manifests:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets all \
  --flickr30k_source manual \
  --imagenet_source manual
```

This downloads/localizes:

- COCO 2017 train/val images and train/val annotations.
- Karpathy caption split annotations for COCO and Flickr30k.
- CC3M WebDataset shards from Hugging Face by default.
- CIFAR100 through torchvision.
- ImageNet auxiliary class-index metadata.

Flickr30k images and ImageNet images are license-gated in many environments. The
script creates `README_DOWNLOAD.txt` files with the expected local paths when
manual archives are required.

## COCO Only

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets coco karpathy manifest
```

Expected paths:

```text
$CLIPGENE_DATA_ROOT/coco/train2017
$CLIPGENE_DATA_ROOT/coco/val2017
$CLIPGENE_DATA_ROOT/coco/annotations/captions_train2017.json
$CLIPGENE_DATA_ROOT/coco/annotations/captions_val2017.json
$CLIPGENE_DATA_ROOT/coco/annotations/instances_train2017.json
$CLIPGENE_DATA_ROOT/coco/annotations/instances_val2017.json
$CLIPGENE_DATA_ROOT/coco/annotations/dataset_coco.json
```

## CC3M

Default WebDataset download:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets cc3m manifest \
  --cc3m_source hf_wds
```

Alternative from the official CC3M TSV with `img2dataset`:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets cc3m manifest \
  --cc3m_source img2dataset \
  --cc3m_tsv /path/to/Train_GCC-training.tsv \
  --cc3m_processes 32 \
  --cc3m_threads 64
```

Training path:

```text
$CLIPGENE_DATA_ROOT/cc3m/wds
```

## Flickr30k

If you already have an image archive on the server:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets flickr30k manifest \
  --flickr30k_source archive \
  --flickr30k_archive /path/to/flickr30k-images.zip
```

If you use Kaggle credentials:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets flickr30k manifest \
  --flickr30k_source kaggle
```

Expected paths:

```text
$CLIPGENE_DATA_ROOT/flickr30k/images
$CLIPGENE_DATA_ROOT/flickr30k/annotations/dataset_flickr30k.json
```

## ImageNet

ImageNet images require official access. After placing the validation archive and
label file on the server:

```bash
python -m scripts.download_datasets \
  --root "$CLIPGENE_DATA_ROOT" \
  --datasets imagenet manifest \
  --imagenet_source archive \
  --imagenet_val_archive /path/to/ILSVRC2012_img_val.tar \
  --imagenet_val_labels /path/to/ImageNet_val_label.txt
```

Expected paths:

```text
$CLIPGENE_DATA_ROOT/imagenet/val
$CLIPGENE_DATA_ROOT/imagenet/val_imagefolder
$CLIPGENE_DATA_ROOT/imagenet/ImageNet_val_label.txt
$CLIPGENE_DATA_ROOT/imagenet/imagenet_class_index.json
$CLIPGENE_DATA_ROOT/imagenet/imagenet_classnames.txt
```

`scripts.imagenet_zs` uses the flat `val` folder plus `ImageNet_val_label.txt`.
The task builder in `tasks.zero_shot_imagenet` can use `val_imagefolder`.

## Generated Path Files

Every run with `--datasets manifest` writes:

```text
$CLIPGENE_DATA_ROOT/clipgene_data_paths.json
$CLIPGENE_DATA_ROOT/clipgene_data_env.sh
$CLIPGENE_DATA_ROOT/clipgene_data_env.ps1
```

Load the environment file before running experiments:

```bash
source "$CLIPGENE_DATA_ROOT/clipgene_data_env.sh"
```
