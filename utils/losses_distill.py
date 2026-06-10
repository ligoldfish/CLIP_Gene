"""同源蒸馏损失（teacher = 同一个 CLIP，迁移其自身对齐，不引入额外知识）。

L = α·InfoNCE + β·特征MSE + γ·affinity-KL
  - InfoNCE：对称交叉熵（CLIP 标准对比）
  - 特征 MSE：学生/ teacher 归一化 embedding 的 MSE（CLIP-KD 证最强）
  - affinity-KL：学生模仿 teacher 的图文相似度矩阵软标签（TinyCLIP affinity mimicking）
"""
import torch
import torch.nn.functional as F


def clip_contrastive(logits_i: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits_i.size(0), device=logits_i.device)
    return 0.5 * (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels))


def feature_mse(zi_s, zi_t, zt_s, zt_t) -> torch.Tensor:
    return F.mse_loss(zi_s, zi_t) + F.mse_loss(zt_s, zt_t)


def feature_distill(zi_s, zi_t, zt_s, zt_t, mode="cos") -> torch.Tensor:
    """特征蒸馏锚（学生贴 teacher，防遗忘）。

    "cos": 1-余弦距离（量级~0.5，与 clip/aff 同档，β=1 即有效梯度）—— 推荐，CLIP-KD 特征 mimicry 的归一形式。
    "mse": 归一向量 MSE（量级太小，需很大 β 才起作用）。
    """
    if mode == "cos":
        di = (1.0 - F.cosine_similarity(zi_s, zi_t, dim=-1)).mean()
        dt = (1.0 - F.cosine_similarity(zt_s, zt_t, dim=-1)).mean()
        return di + dt
    return F.mse_loss(zi_s, zi_t) + F.mse_loss(zt_s, zt_t)


def affinity_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """对称 KL（行+列），teacher 作软目标。乘 tau^2 标准 KD 缩放。"""
    s = student_logits / tau
    t = teacher_logits / tau
    # rows
    kl_r = F.kl_div(F.log_softmax(s, dim=1), F.softmax(t, dim=1), reduction="batchmean")
    # cols
    kl_c = F.kl_div(F.log_softmax(s.t(), dim=1), F.softmax(t.t(), dim=1), reduction="batchmean")
    return 0.5 * (kl_r + kl_c) * (tau * tau)


def compute_distill_loss(zi_s, zt_s, zi_t, zt_t, scale_s, scale_t, cfg):
    """返回 (total_loss, parts_dict)。所有 z 均已 L2 归一化。"""
    logits_i = scale_s * (zi_s @ zt_s.t())
    logits_t = logits_i.t()
    teacher_logits = scale_t * (zi_t @ zt_t.t())

    l_clip = clip_contrastive(logits_i, logits_t)
    l_feat = feature_distill(zi_s, zi_t, zt_s, zt_t, mode=getattr(cfg, "DISTILL_FEAT_MODE", "cos"))
    l_aff = affinity_kl(logits_i, teacher_logits.detach(), cfg.DISTILL_TAU)

    total = cfg.DISTILL_ALPHA * l_clip + cfg.DISTILL_BETA * l_feat + cfg.DISTILL_GAMMA * l_aff
    parts = {"clip": l_clip.item(), "feat": l_feat.item(), "aff": l_aff.item()}
    return total, parts
