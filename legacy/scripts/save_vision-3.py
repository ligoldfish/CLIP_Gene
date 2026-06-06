# clip_cifar100/extract_features.py
import torch
from models.vision_model import CLIPVisionModel, FeatureExtractor
from configs.base_config import Config

def save_feature_extractor():
    # 加载训练好的模型
    model = CLIPVisionModel(pretrained_path=Config.MODEL_SAVE_PATH)
    model.to(Config.DEVICE)
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(Config.DEVICE)
    feature_extractor.eval()
    
    # 保存特征提取器
    torch.save(feature_extractor.state_dict(), Config.FEATURE_MODEL_SAVE_PATH)
    print(f"Saved feature extractor to {Config.FEATURE_MODEL_SAVE_PATH}")
    
    # 打印模型结构信息
    print("\nFeature extractor architecture:")
    print(feature_extractor)
    
    # 测试特征维度
    dummy_input = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
    features = feature_extractor(dummy_input)
    
    print("\nFeature dimensions:")
    for i, feat in enumerate(features):
        print(f"Layer {len(features)-i} (from last): {feat.shape}")

if __name__ == "__main__":
    save_feature_extractor()