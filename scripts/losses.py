# # scripts/losses.py
# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from scripts.utils.distributed import concat_all_gather, is_dist_avail_and_initialized

# def clip_contrastive_loss(image_features: torch.Tensor,
#                           text_features: torch.Tensor,
#                           logit_scale: torch.Tensor) -> torch.Tensor:
#     # 1) 先检查特征是否已经炸了（不要吞 NaN）
#     if (not torch.isfinite(image_features).all()) or (not torch.isfinite(text_features).all()):
#         return None
#         # rank = dist.get_rank() if dist.is_initialized() else 0
#         # if rank == 0:
#         #     print("[BAD] non-finite features!",
#         #           "img absmax=", image_features.abs().max().item(),
#         #           "txt absmax=", text_features.abs().max().item())
#         # raise RuntimeError("Non-finite features")

#     # 2) fp32 归一化 + eps 防止 0/0
#     img = F.normalize(image_features.float(), dim=-1, eps=1e-6)
#     txt = F.normalize(text_features.float(),  dim=-1, eps=1e-6)

#     # 3) gather
#     if is_dist_avail_and_initialized():
#         all_img = concat_all_gather(img)
#         all_txt = concat_all_gather(txt)
#         rank = dist.get_rank()
#         bsz = img.size(0)
#         targets = torch.arange(bsz, device=img.device) + rank * bsz
#     else:
#         all_img, all_txt = img, txt
#         bsz = img.size(0)
#         targets = torch.arange(bsz, device=img.device)

#     # 4) logits 用 fp32（amp 下更稳）
#     ls = logit_scale.float()
#     logits_i = ls * (img @ all_txt.t())
#     logits_t = ls * (txt @ all_img.t())

#     if (not torch.isfinite(logits_i).all()) or (not torch.isfinite(logits_t).all()):
#         rank = dist.get_rank() if dist.is_initialized() else 0
#         if rank == 0:
#             print("[BAD] non-finite logits!",
#                   "logits_i absmax=", logits_i.abs().max().item(),
#                   "logits_t absmax=", logits_t.abs().max().item(),
#                   "logit_scale=", ls.item())
#         raise RuntimeError("Non-finite logits")

#     loss_i = F.cross_entropy(logits_i, targets)
#     loss_t = F.cross_entropy(logits_t, targets)
#     return 0.5 * (loss_i + loss_t)

# scripts/losses.py
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scripts.utils.distributed import concat_all_gather, is_dist_avail_and_initialized

def clip_contrastive_loss(image_features: torch.Tensor,
                          text_features: torch.Tensor,
                          logit_scale: torch.Tensor) -> torch.Tensor:
    # # 1) 先检查特征是否已经炸了（不要吞 NaN）
    # if (not torch.isfinite(image_features).all()) or (not torch.isfinite(text_features).all()):
    #     # IMPORTANT (DDP): do NOT raise here. Let the training loop decide
    #     # whether to skip this batch *synchronously across ranks*.
    #     return None

    # 2) fp32 归一化 + eps 防止 0/0
    img = F.normalize(image_features.float(), dim=-1, eps=1e-6)
    txt = F.normalize(text_features.float(),  dim=-1, eps=1e-6)

    # 3) gather
    if is_dist_avail_and_initialized():
        all_img = concat_all_gather(img)
        all_txt = concat_all_gather(txt)
        rank = dist.get_rank()
        bsz = img.size(0)
        targets = torch.arange(bsz, device=img.device) + rank * bsz
    else:
        all_img, all_txt = img, txt
        bsz = img.size(0)
        targets = torch.arange(bsz, device=img.device)

    # 4) logits 用 fp32（amp 下更稳）
    ls = logit_scale.float()
    logits_i = ls * (img @ all_txt.t())
    logits_t = ls * (txt @ all_img.t())

    # if (not torch.isfinite(logits_i).all()) or (not torch.isfinite(logits_t).all()):
    #     return None
    if (not torch.isfinite(img).all()) or (not torch.isfinite(txt).all()):
        return None
    if (not torch.isfinite(logits_i).all()) or (not torch.isfinite(logits_t).all()):
        return None

    loss_i = F.cross_entropy(logits_i, targets)
    loss_t = F.cross_entropy(logits_t, targets)
    return 0.5 * (loss_i + loss_t)
