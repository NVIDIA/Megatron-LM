"""Quick-and-dirty test for multimem_all_gather_v.

Launch with:
    torchrun --nproc_per_node=NUM_GPUS test_agv.py

Each rank contributes a different number of tokens to exercise the variable-count path.
Rank r gets (BASE_TOKENS + r * TOKEN_STEP) tokens.
"""

import os
import sys

import torch
import torch.distributed as dist

HIDDEN_SIZE = 1024
BASE_TOKENS = 8
TOKEN_STEP = 4  # rank r has BASE_TOKENS + r * TOKEN_STEP tokens


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log(rank, f"init done, world_size={world_size}")

    from megatron.core.inference.communication.torch_symm_triton import are_tensors_nvls_eligible
    from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
        multimem_all_gather_v,
    )
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    # --- per-rank token counts and prefix sums ---
    local_tokens_per_rank = [BASE_TOKENS + r * TOKEN_STEP for r in range(world_size)]
    prefix_sums = [sum(local_tokens_per_rank[:r]) for r in range(world_size)]
    total_tokens = sum(local_tokens_per_rank)
    local_tokens = local_tokens_per_rank[rank]
    log(rank, f"local_tokens={local_tokens}, total_tokens={total_tokens}, prefix_sum={prefix_sums[rank]}")

    # --- local input: rank r fills with float(r) so placement is trivially verifiable ---
    input_tensor = torch.full(
        (local_tokens, HIDDEN_SIZE), float(rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    log(rank, f"input_tensor shape={input_tensor.shape}, dtype={input_tensor.dtype}")

    # --- rank_token_offset: scalar int32 CUDA tensor (fixed address for CUDA graphs) ---
    rank_token_offset = torch.tensor([prefix_sums[rank]], dtype=torch.int32, device="cuda")
    log(rank, f"rank_token_offset={rank_token_offset.item()}")

    # --- symmetric memory output buffer via SymmetricMemoryManager ---
    log(rank, "creating EP process group...")
    ep_group = dist.new_group(ranks=list(range(world_size)))
    log(rank, "getting symmetric memory buffer...")
    buf = SymmetricMemoryManager.get_buffer("ep", process_group=ep_group)
    log(rank, f"symm_mem_hdl={buf.symm_mem_hdl}")
    result = buf.maybe_get_tensor([total_tokens, HIDDEN_SIZE], dtype=torch.bfloat16)

    if result["handle"] is None:
        log(rank, "SKIP — symmetric memory unavailable (need Hopper+ NVLink)")
        dist.destroy_process_group()
        sys.exit(0)

    output_tensor = result["tensor"]
    symm_mem_hdl = result["handle"]
    log(rank, f"output_tensor shape={output_tensor.shape}, symm_mem_hdl rank={symm_mem_hdl.rank}")

    if not are_tensors_nvls_eligible(input_tensor):
        log(rank, "SKIP — input not NVLS-eligible (need 16-byte aligned size)")
        dist.destroy_process_group()
        sys.exit(0)
    log(rank, "tensors are NVLS-eligible")

    max_tokens = max(local_tokens_per_rank)  # same on all ranks — fixes CTA count symmetry

    # --- warmup ---
    log(rank, "starting warmup...")
    for i in range(3):
        log(rank, f"  warmup iteration {i}")
        multimem_all_gather_v(
            output_tensor, input_tensor, symm_mem_hdl,
            rank_token_offset=rank_token_offset,
            hidden_size=HIDDEN_SIZE,
            max_tokens=max_tokens,
        )
        log(rank, f"  warmup iteration {i} done")
    log(rank, "warmup done, synchronizing...")
    torch.cuda.synchronize()
    log(rank, "synchronized")

    # --- verify eager ---
    passed = True
    for r in range(world_size):
        start, end = prefix_sums[r], prefix_sums[r] + local_tokens_per_rank[r]
        expected = torch.full(
            (local_tokens_per_rank[r], HIDDEN_SIZE), float(r), dtype=torch.bfloat16, device="cuda"
        )
        if not torch.equal(output_tensor[start:end], expected):
            log(rank, f"EAGER FAIL: slice for rank {r} mismatch")
            passed = False
    if passed:
        log(rank, f"EAGER PASS — tokens per rank: {local_tokens_per_rank}, total: {total_tokens}")

    SymmetricMemoryManager.destroy("ep")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
