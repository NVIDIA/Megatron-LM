"""Distributed HybridEP route-slot gate.

Run with an explicit NVLink-domain size, for example:

    NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4 \
      torchrun --standalone --nproc-per-node=4 hybridep_route_gate.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedHybridEPDispatcher,
)
from megatron.lite.primitive.parallel import ParallelState


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    local_experts = 2
    hidden_size = 16
    dispatcher = VLLMAlignedHybridEPDispatcher(
        world_size * local_experts,
        hidden_size,
        ParallelState(
            ep_size=world_size,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        hybridep_max_tokens_per_rank=128,
    )
    hidden = (
        torch.arange(
            2 * hidden_size, device="cuda", dtype=torch.float32
        ).reshape(2, hidden_size)
        + rank * 32
        + 1
    ).to(torch.bfloat16)
    local_start = rank * local_experts
    indices = torch.tensor(
        [
            [local_start, local_start],
            [
                local_start + 1,
                ((rank + 1) % world_size) * local_experts,
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    weights = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]],
        dtype=torch.float32,
        device="cuda",
    )

    dispatched, counts, _ = dispatcher.dispatch(hidden, weights, indices)
    expert_output = dispatched.clone()
    offset = 0
    for local_index, count in enumerate(counts.tolist()):
        end = offset + int(count)
        expert_output[offset:end].mul_(
            rank * local_experts + local_index + 1
        )
        offset = end
    if offset != expert_output.shape[0]:
        raise RuntimeError("HybridEP expert counts do not cover dispatched rows")
    actual = dispatcher.combine(expert_output)

    expected = torch.empty_like(actual)
    for token in range(hidden.shape[0]):
        accumulator = torch.zeros(
            hidden_size, dtype=torch.float32, device=hidden.device
        )
        for slot in range(indices.shape[1]):
            factor = int(indices[token, slot]) + 1
            accumulator.add_(
                (hidden[token] * factor).to(torch.bfloat16).float()
                * weights[token, slot]
            )
        expected[token].copy_(accumulator.to(torch.bfloat16))
    if not torch.equal(actual, expected):
        error = float((actual.float() - expected.float()).abs().max().item())
        raise AssertionError(
            f"HybridEP route-slot gate failed on rank {rank}: max_abs={error}"
        )
    if rank == 0:
        print(
            f"HybridEP route-slot gate passed bitwise at EP={world_size}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
