"""统一评估接口：把 openai teacher 与 StudentMiniCLIP 包成同一 API。

二者都暴露 encode_image / encode_text / logit_scale，故鸭子类型即可。
"""
import torch


class EvalModel:
    def __init__(self, model, tokenize_fn, device="cuda", logit_scale_max=100.0):
        self.model = model.to(device).eval()
        self.tok = tokenize_fn
        self.device = device
        self.logit_scale_max = logit_scale_max

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        z = self.model.encode_image(images.to(self.device)).float()
        return z / z.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        toks = self.tok(texts).to(self.device)
        z = self.model.encode_text(toks).float()
        return z / z.norm(dim=-1, keepdim=True)

    @property
    def logit_scale_exp(self) -> float:
        return float(self.model.logit_scale.exp().clamp(max=self.logit_scale_max).item())
