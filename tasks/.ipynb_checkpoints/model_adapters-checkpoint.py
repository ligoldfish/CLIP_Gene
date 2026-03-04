# codes/tasks/model_adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, Union

import torch
import torch.nn.functional as F

from .common import l2_normalize, default_clip_image_transform


class ModelAdapter(Protocol):
    """
    Unified interface for training/eval across models.
    Must provide:
      - preprocess: torchvision transform for PIL -> Tensor
      - tokenize(texts)->Tensor[int64] or model-specific tokens
      - encode_image(images)->[B,D] normalized
      - encode_text(tokens or texts)->[B,D] normalized
      - device, dtype
    """
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    def preprocess_pil(self, pil_image): ...
    def tokenize(self, texts: List[str]) -> torch.Tensor: ...
    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor: ...


@dataclass
class OpenAIClipAdapter:
    clip_model_name: str = "ViT-B/32"
    device_str: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        import clip  # openai clip
        dev = self.device_str if torch.cuda.is_available() and self.device_str.startswith("cuda") else "cpu"
        self._device = torch.device(dev)

        model, preprocess = clip.load(self.clip_model_name, device=self._device, jit=False)
        self.model = model.eval()
        self._preprocess = preprocess
        self._tokenize = clip.tokenize

        # OpenAI CLIP internally uses model.dtype (often fp16 on cuda by default)
        # Force fp32 unless you want fp16:
        if self.dtype == torch.float32:
            self.model = self.model.float()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype if hasattr(self.model, "dtype") else self.dtype

    def preprocess_pil(self, pil_image):
        return self._preprocess(pil_image)

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return self._tokenize(texts, truncate=True)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images.to(self.device))
        return l2_normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(text_tokens.to(self.device))
        return l2_normalize(feats.float(), dim=-1)


@dataclass
class StudentCLIPAdapter:
    """
    Wrap your StudentCLIP (ours) to match the adapter interface.
    """
    student_model: torch.nn.Module
    device_str: str = "cuda"
    image_size: int = 224

    def __post_init__(self):
        dev = self.device_str if torch.cuda.is_available() and self.device_str.startswith("cuda") else "cpu"
        self._device = torch.device(dev)
        self.model = self.student_model.to(self._device).eval()
        # Use CLIP-like transform by default
        self._preprocess = default_clip_image_transform(self.image_size)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        # use model parameter dtype
        return next(self.model.parameters()).dtype

    def preprocess_pil(self, pil_image):
        return self._preprocess(pil_image)

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        # StudentCLIP uses CLIP tokenizer (same as OpenAI clip), easiest is reuse clip.tokenize
        import clip
        return clip.tokenize(texts, truncate=True)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # use forward path to get embeddings: we expose tower enc via logits; easiest:
        # We can call internal stems + towers if you want, but simplest is:
        # add helper in training script. Here we compute via model forward is not needed.
        # We'll do explicit with existing modules in StudentCLIP:
        v_tokens = self.model.vision_stem(images.to(self.device, dtype=self.dtype))
        z = self.model.vision_tower(v_tokens)  # already normalized in our tower
        return z.float()

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        t_tokens = self.model.text_stem(text_tokens.to(self.device))
        z = self.model.text_tower(t_tokens, text=text_tokens.to(self.device))
        return z.float()


@dataclass
class OpenCLIPAdapter:
    """
    Adapter for open_clip models (can cover TinyCLIP depending on availability).
    Requires:
      pip install open_clip_torch
    """
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device_str: str = "cuda"

    def __post_init__(self):
        try:
            import open_clip
        except Exception as e:
            raise ImportError(
                "open_clip_torch is not installed. Install: pip install open_clip_torch"
            ) from e

        dev = self.device_str if torch.cuda.is_available() and self.device_str.startswith("cuda") else "cpu"
        self._device = torch.device(dev)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self._device
        )
        self.model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def preprocess_pil(self, pil_image):
        return self._preprocess(pil_image)

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return self._tokenizer(texts)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images.to(self.device))
        return l2_normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(text_tokens.to(self.device))
        return l2_normalize(feats.float(), dim=-1)
