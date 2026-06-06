"""CLIPgene 评估套件：零样本分类 / 图文检索 / 模态间隔 / 收敛成本。

本地保底：CIFAR-100 零样本 + COCO-val 检索 + modality_gap。
其余数据集经 registry 挂载、缺数据优雅跳过。
"""
