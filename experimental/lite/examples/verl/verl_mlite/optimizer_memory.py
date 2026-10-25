# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Early communicator allocation and inactive allocator release for hybrid RL."""
import torch
import torch.distributed as dist


def release_inactive_cuda_cache(*, synchronize: bool = True) -> None:
    """Let colocated rollout/NCCL reuse allocator blocks, never live tensors."""
    if synchronize:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def release_post_backward_cache(policy: str, free_gib: float = 4.0) -> bool:
    """Optionally release inactive blocks; never synchronize or touch live tensors."""
    if policy == "off":
        return False
    if policy not in ("threshold", "always"):
        raise ValueError(f"Unknown post-backward cache policy: {policy!r}")
    if policy == "threshold" and torch.cuda.mem_get_info()[0] >= free_gib * 2**30:
        return False
    release_inactive_cuda_cache(synchronize=False)
    return True


def prewarm_grad_stats_groups(optimizer, device) -> None:
    """Materialize Core's lazy GN communicators before rollout fills the device.

    A zero scratch SUM uses exactly the optimizer's groups without touching model
    parameters, gradients, optimizer state or RNG. Deduplicate chained owners.
    Optimizers without a distributed grad-stats group need no communicator.
    """
    seen = set()
    for owner in getattr(optimizer, "chained_optimizers", [optimizer]):
        get_group = getattr(owner, "get_grad_stats_parallel_group", None)
        if get_group is None:
            continue
        group = get_group()
        if id(group) in seen:
            continue
        seen.add(id(group))
        scratch = torch.zeros((), dtype=torch.float32, device=device)
        dist.all_reduce(scratch, op=dist.ReduceOp.SUM, group=group)
        if scratch.item() != 0:
            raise RuntimeError("Grad-stats communicator prewarm changed zero scratch")
