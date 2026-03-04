# demo_build_student.py
import torch
from models import StudentCLIP, StudentCLIPConfig

def main():
    cfg = StudentCLIPConfig(
        gene_variant_dir="/root/gene_exports/last2",  # 改成 last3 / last2_plus6 等
        use_tleg=True,             # 仅当 gene 是2层时有效（last2）
        tleg_target_depth=4,        # 2层 -> 扩展成4层
        proj_dim=256,
        proj_use_mlp=True,
        freeze_learngene=True,
        patch_size=32,
        image_resolution=224,
        context_length=77,
    )

    model = StudentCLIP(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    B = 2
    images = torch.randn(B, 3, 224, 224, device=next(model.parameters()).device)
    text = torch.randint(0, cfg.vocab_size, (B, cfg.context_length), device=next(model.parameters()).device)

    with torch.no_grad():
        li, lt = model(images, text)
    print("logits_per_image:", li.shape, "logits_per_text:", lt.shape)

if __name__ == "__main__":
    main()
