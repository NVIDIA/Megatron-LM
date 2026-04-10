# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Integration test: AGV → permute → unpermute → RSV.

Run with:
    torchrun --nproc_per_node=NUM_GPUS test_dispatch_combine.py
"""
import os
import sys

import torch
import torch.distributed as dist

HIDDEN = 2688
DTYPE = torch.bfloat16
BASE_TOKENS = 32
TOKEN_STEP = 16
ENGINE_MAX_TOKENS = 2048
NUM_LOCAL_EXPERTS = 2
TOPK = 2


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def prefix_sum(counts):
    return [sum(counts[:r]) for r in range(len(counts))]


def make_distributions(world_size):
    base, step = BASE_TOKENS, TOKEN_STEP
    return {
        "linear_inc": [base + r * step for r in range(world_size)],
        "uniform":    [base + (world_size // 2) * step] * world_size,
        "skewed":     [base * 4 if r == 0 else base for r in range(world_size)],
    }


def make_local_inputs(rank, local_tokens, num_total_experts):
    local_hidden = torch.full(
        (local_tokens, HIDDEN), float(rank + 1), dtype=DTYPE, device="cuda"
    )
    # Sample TOPK unique experts per token without replacement via topk on random scores.
    rand_scores = torch.rand(local_tokens, num_total_experts, device="cuda")
    local_routing = torch.topk(rand_scores, TOPK, dim=1).indices.to(torch.int64)
    local_probs = torch.full(
        (local_tokens, TOPK), 1.0 / TOPK, dtype=torch.float32, device="cuda"
    )
    return local_hidden, local_routing, local_probs


def verify_permute(rank, name, permuted_hidden, permuted_probs, permutation_map,
                   n_used, symm_hidden, local_tokens_per_rank, psums, world_size):
    """Every non-padding permuted row must match its source token in symm_hidden.

    permutation_map[pos] = original global token index (or -1 for padding).
    permuted_hidden[pos] must equal symm_hidden[permutation_map[pos]].
    permuted_probs[pos] must equal 1/TOPK.

    Only [0, n_used) is valid — elements beyond are uninitialized (torch.empty).
    """
    log(rank, f"    [{name}] n_used={n_used}, output_size={permuted_hidden.shape[0]}")

    valid_mask = permutation_map[:n_used] >= 0
    valid_pos = valid_mask.nonzero(as_tuple=True)[0]  # positions with real tokens

    if valid_pos.numel() == 0:
        log(rank, f"    [{name}] no valid positions — all tokens routed off-rank")
        return True

    src_indices = permutation_map[:n_used][valid_pos]  # original global token indices

    # Each permuted row must equal its source token's hidden state
    expected_hidden = symm_hidden[src_indices]        # [n_valid, HIDDEN] bf16
    actual_hidden = permuted_hidden[valid_pos]        # [n_valid, HIDDEN] bf16

    if not torch.equal(actual_hidden, expected_hidden):
        diff = (actual_hidden.float() - expected_hidden.float()).abs()
        log(rank, f"  FAIL [{name}] permuted_hidden mismatch: max_diff={diff.max().item():.4f}")
        bad = diff.max(dim=1).values.argmax().item()
        src = src_indices[bad].item()
        exp_val = float(src_indices[bad] // 1)
        for r in range(world_size):
            if psums[r] <= src < psums[r] + local_tokens_per_rank[r]:
                exp_val = float(r + 1)
                break
        log(rank, f"    first bad: pos={valid_pos[bad].item()}, src_token={src}, "
                  f"expected_fill={exp_val}, got={actual_hidden[bad, 0].item():.4f}")
        return False

    # Each permuted row must have prob 1/TOPK
    expected_prob = 1.0 / TOPK
    prob_diff = (permuted_probs[valid_pos] - expected_prob).abs()
    if prob_diff.max().item() > 1e-5:
        log(rank, f"  FAIL [{name}] permuted_probs mismatch: "
                  f"max_diff={prob_diff.max().item():.6f}")
        return False

    return True


def verify_rsv_output(rank, name, rsv_output):
    """After AGV → permute → unpermute → RSV, each rank's output should equal float(rank+1).

    probs sum to 1.0 (TOPK × 1/TOPK), so the weighted sum of hidden values for each
    token recovers the original hidden value float(source_rank+1). After RSV, rank r
    holds its own tokens, so all values should be float(rank+1).
    """
    expected_val = float(rank + 1)
    diff = (rsv_output - expected_val).abs()
    if diff.max().item() > 1e-2:
        log(rank, f"  FAIL [{name}] RSV output: max_diff={diff.max().item():.4f}, "
                  f"expected={expected_val}")
        return False
    return True


def run_case(rank, world_size, name, local_tokens_per_rank, psums, ep_group):
    from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
        multimem_all_gather_v,
        multimem_reduce_scatter_v,
    )
    from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    local_tokens = local_tokens_per_rank[rank]
    total_tokens = sum(local_tokens_per_rank)
    total_max_tokens = ENGINE_MAX_TOKENS * world_size
    num_total_experts = world_size * NUM_LOCAL_EXPERTS
    local_expert_start = rank * NUM_LOCAL_EXPERTS

    ep_max = max(local_tokens_per_rank)
    rank_token_offset = torch.tensor([psums[rank]], dtype=torch.int32, device="cuda")
    ep_max_tokens = torch.tensor([ep_max], dtype=torch.int32, device="cuda")
    valid_tokens_t = torch.tensor([total_tokens], dtype=torch.int32, device="cuda")

    # Allocate fixed-size symmetric buffers (bf16 inputs + fp32 unpermute output)
    hidden_buf  = SymmetricMemoryManager.get_buffer(f"{name}_h", process_group=ep_group)
    routing_buf = SymmetricMemoryManager.get_buffer(f"{name}_r", process_group=ep_group)
    probs_buf   = SymmetricMemoryManager.get_buffer(f"{name}_p", process_group=ep_group)
    unperm_buf  = SymmetricMemoryManager.get_buffer(f"{name}_u", process_group=ep_group)

    hidden_r  = hidden_buf.maybe_get_tensor([total_max_tokens, HIDDEN], dtype=DTYPE)
    routing_r = routing_buf.maybe_get_tensor([total_max_tokens, TOPK], dtype=torch.int64)
    probs_r   = probs_buf.maybe_get_tensor([total_max_tokens, TOPK], dtype=torch.float32)
    unperm_r  = unperm_buf.maybe_get_tensor([total_max_tokens, HIDDEN], dtype=torch.float32)

    if any(r["handle"] is None for r in (hidden_r, routing_r, probs_r, unperm_r)):
        return None

    symm_hidden    = hidden_r["tensor"]
    symm_hidden_hdl = hidden_r["handle"]
    symm_routing   = routing_r["tensor"]
    symm_routing_hdl = routing_r["handle"]
    symm_probs     = probs_r["tensor"]
    symm_probs_hdl = probs_r["handle"]
    symm_unperm    = unperm_r["tensor"]
    symm_unperm_hdl = unperm_r["handle"]

    local_hidden, local_routing, local_probs = make_local_inputs(
        rank, local_tokens, num_total_experts
    )

    # AGV all three input tensors into fixed-size symm buffers
    multimem_all_gather_v(
        symm_hidden, local_hidden, symm_hidden_hdl,
        rank_token_offset=rank_token_offset,
        ep_max_tokens=ep_max_tokens,
        engine_max_tokens=ENGINE_MAX_TOKENS,
    )
    multimem_all_gather_v(
        symm_routing, local_routing, symm_routing_hdl,
        rank_token_offset=rank_token_offset,
        ep_max_tokens=ep_max_tokens,
        engine_max_tokens=ENGINE_MAX_TOKENS,
    )
    multimem_all_gather_v(
        symm_probs, local_probs, symm_probs_hdl,
        rank_token_offset=rank_token_offset,
        ep_max_tokens=ep_max_tokens,
        engine_max_tokens=ENGINE_MAX_TOKENS,
    )
    torch.cuda.synchronize()

    # Permute using gathered tensors, gated by valid_tokens
    permuted_hidden, permuted_probs, permutation_map, offs = permute_tokens(
        symm_hidden,
        symm_probs,
        symm_routing,
        local_expert_start,
        NUM_LOCAL_EXPERTS,
        valid_tokens_t,
        alignment=1,
    )
    n_used = offs[-1].item()  # scalar: total rows used in permuted output
    n_used_t = offs[-1:]      # device tensor view for kernels

    passed_permute = verify_permute(
        rank, name, permuted_hidden, permuted_probs, permutation_map,
        n_used, symm_hidden, local_tokens_per_rank, psums, world_size,
    )

    # Unpermute directly into the symmetric fp32 buffer, then RSV to sum across all ranks.
    # After RSV each rank holds its own token slice with full weighted sum = float(rank+1).
    unpermute_tokens(
        permuted_hidden, permuted_probs, permutation_map, total_max_tokens, n_used_t, valid_tokens_t,
        out=symm_unperm,
    )

    rsv_output = torch.empty(local_tokens, HIDDEN, dtype=torch.float32, device="cuda")
    multimem_reduce_scatter_v(
        rsv_output, symm_unperm, symm_unperm_hdl,
        rank_token_offset=rank_token_offset,
        ep_max_tokens=ep_max_tokens,
        engine_max_tokens=ENGINE_MAX_TOKENS,
    )
    torch.cuda.synchronize()

    passed_rsv = verify_rsv_output(rank, name, rsv_output)

    passed = passed_permute and passed_rsv

    SymmetricMemoryManager.destroy(f"{name}_h")
    SymmetricMemoryManager.destroy(f"{name}_r")
    SymmetricMemoryManager.destroy(f"{name}_p")
    SymmetricMemoryManager.destroy(f"{name}_u")
    return passed


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log(rank, f"init done, world_size={world_size}")

    from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
    if not is_device_nvls_capable(torch.device("cuda")):
        log(rank, "SKIP ALL — requires Hopper+ GPU with NVLink (SM >= 9)")
        dist.destroy_process_group()
        sys.exit(0)

    ep_group = dist.new_group(ranks=list(range(world_size)))
    results = {}

    for dist_name, counts in make_distributions(world_size).items():
        psums = prefix_sum(counts)
        log(rank, f"--- {dist_name} | tokens={counts} ---")
        passed = run_case(rank, world_size, dist_name, counts, psums, ep_group)
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        log(rank, f"  {status} [{dist_name}]")
        results[dist_name] = passed

    n_skip = sum(1 for v in results.values() if v is None)
    n_pass = sum(1 for v in results.values() if v is True)
    n_fail = sum(1 for v in results.values() if v is False)
    log(rank, f"=== {n_pass} passed, {n_fail} failed, {n_skip} skipped "
              f"({len(results)} total) ===")

    dist.destroy_process_group()
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
