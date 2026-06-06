from transformers import CLIPVisionModel
from transformers import CLIPModel
#from configs.base_config import config

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

print(model1)
# 输出应包含 encoder 属性：
# CLIPVisionTransformer(
#   (embeddings): CLIPVisionEmbeddings(...)
#   (encoder): CLIPEncoder(...)
#   (post_layernorm): LayerNorm(...)
# )

# import clip, importlib, sys
# print("clip module file:", clip.__file__)
# print("clip has available_models? ", hasattr(clip, "available_models"))
# try:
#     from importlib.metadata import version
#     print("dist version:", version("clip"))
# except Exception as e:
#     print("dist version not found:", e)

# print("has simple_tokenizer? ", hasattr(clip, "simple_tokenizer") or hasattr(__import__("clip.simple_tokenizer"), "simple_tokenizer"))
# print("models:", clip.available_models() if hasattr(clip, "available_models") else "N/A")

