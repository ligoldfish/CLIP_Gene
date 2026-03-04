# scripts/utils/distributed.py
import os
import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl"):
    if dist.is_available() and dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dist.barrier()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
