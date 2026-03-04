# codes/tasks/__init__.py
from .model_adapters import OpenAIClipAdapter, StudentCLIPAdapter, OpenCLIPAdapter
from .builders import (
    build_task_retrieval,
    build_task_itm,
    build_task_multilabel,
    build_task_zeroshot_imagenet,
)
from .retrieval import (
    encode_all_images,
    encode_all_texts,
    compute_retrieval_metrics,
)
