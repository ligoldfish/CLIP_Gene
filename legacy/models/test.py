from transformers import CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
print(model)
