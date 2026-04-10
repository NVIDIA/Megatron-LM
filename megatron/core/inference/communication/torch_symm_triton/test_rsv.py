"""Tests for multimem_reduce_scatter_v.

Launch with:
    torchrun --nproc_per_node=NUM_GPUS test_rsv.py

Setup: each rank fills its entire symmetric buffer with float(rank + 1), simulating
each rank's local expert contribution to every token position. After RSV, rank r's
output slice (local_tokens rows) should equal the sum across all WS ranks:

    expected = sum(float(r + 1) for r in range(WS)) = WS * (WS + 1) / 2

Token distributions and tensor specs mirror test_agv.py. Only 128-bit-aligned
dtypes (bf16, fp32) are tested since RSV assumes 128-bit alignment.

Tests:
  Cross-product: 4 token distributions × 2 tensor specs (8 cases)
  multi_call:    second RSV with different fill overwrites first output
  ep_max_eq:     ep_max_tokens == engine_max_tokens (no CTA early-exit)
"""
import os
import sys
from itertools import product as iproduct

import torch
import torch.distributed as dist

BASE_TOKENS = 32
TOKEN_STEP = 16

# (name, hidden, dtype) — RSV is 128-bit only, so all rows must be 16B-aligned.
TENSOR_SPECS = [
    ("bf16_h2688", 2688, torch.bfloat16),  # 5376 B/row ✓
    ("fp32_h2688", 2688, torch.float32),   # 10752 B/row ✓
]


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def make_distributions(world_size, base=BASE_TOKENS, step=TOKEN_STEP):
    return {
        "linear_inc": [base + r * step for r in range(world_size)],
        "linear_dec": [base + (world_size - 1 - r) * step for r in range(world_size)],
        "uniform":    [base + (world_size // 2) * step] * world_size,
        "skewed":     [base * 4 if r == 0 else base for r in range(world_size)],
    }


def prefix_sum(counts):
    return [sum(counts[:r]) for r in range(len(counts))]


def expected_val(world_size):
    """Sum of float(r+1) for r in range(world_size) = WS*(WS+1)/2."""
    return float(world_size * (world_size + 1) // 2)


def verify(rank, name, output_tensor, local_tokens, hidden, dtype, exp_val):
    """Check that every element of output_tensor equals exp_val."""
    expected = torch.full((local_tokens, hidden), exp_val, dtype=dtype, device="cuda")
    if torch.equal(output_tensor, expected):
        return True
    # Report first mismatch for debugging.
    diff = (output_tensor - expected.to(output_tensor.dtype)).abs()
    log(rank, f"  FAIL [{name}]: max_diff={diff.max().item():.4f}, "
              f"first_bad_val={output_tensor.flatten()[diff.flatten().argmax()].item()}")
    return False


def run_rsv(symm_tensor, output_tensor, symm_mem_hdl,
            rank_token_offset, ep_max_tokens, engine_max_tokens, n_warmup=3):
    from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
        multimem_reduce_scatter_v,
    )
    for _ in range(n_warmup):
        multimem_reduce_scatter_v(
            output_tensor, symm_tensor, symm_mem_hdl,
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max_tokens,
        )
    torch.cuda.synchronize()


def run_case(
    name, rank, world_size,
    local_tokens_per_rank, psums,
    ep_group, buf_key,
    hidden, dtype,
    rank_token_offset, ep_max_tokens, engine_max_tokens,
    fill_scale=1.0,
):
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    total_tokens = sum(local_tokens_per_rank)
    buf = SymmetricMemoryManager.get_buffer(buf_key, process_group=ep_group)
    result = buf.maybe_get_tensor([total_tokens, hidden], dtype=dtype)
    if result["handle"] is None:
        return None  # skip

    symm_tensor = result["tensor"]
    symm_mem_hdl = result["handle"]

    local_tokens = local_tokens_per_rank[rank]
    output_tensor = torch.empty((local_tokens, hidden), dtype=dtype, device="cuda")

    # Fill the full symmetric buffer with fill_scale * float(rank + 1).
    # After RSV, output should equal fill_scale * WS*(WS+1)/2.
    symm_tensor.fill_(fill_scale * float(rank + 1))
    torch.cuda.synchronize()

    run_rsv(symm_tensor, output_tensor, symm_mem_hdl,
            rank_token_offset, ep_max_tokens, engine_max_tokens)

    exp = fill_scale * expected_val(world_size)
    passed = verify(rank, name, output_tensor, local_tokens, hidden, dtype, exp)
    SymmetricMemoryManager.destroy(buf_key)
    return passed


ENGINE_MAX_TOKENS = 2048  # fixed at model init; drives CTA grid size


def make_metadata(rank, local_tokens_per_rank):
    psums = prefix_sum(local_tokens_per_rank)
    ep_max = max(local_tokens_per_rank)  # per-iteration max; drives CTA early-exit
    rank_token_offset = torch.tensor([psums[rank]], dtype=torch.int32, device="cuda")
    ep_max_tokens = torch.tensor([ep_max], dtype=torch.int32, device="cuda")
    return psums, ep_max, rank_token_offset, ep_max_tokens


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log(rank, f"init done, world_size={world_size}, expected_val={expected_val(world_size)}")

    from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
    if not is_device_nvls_capable(torch.device("cuda")):
        log(rank, "SKIP ALL — requires Hopper+ GPU with NVLink (SM >= 9)")
        dist.destroy_process_group()
        sys.exit(0)

    ep_group = dist.new_group(ranks=list(range(world_size)))
    distributions = make_distributions(world_size)
    results = {}

    # ── Cross-product: distributions × tensor specs ────────────────────────────
    for (dist_name, counts), (tc_name, hidden, dtype) in iproduct(
        distributions.items(), TENSOR_SPECS
    ):
        psums, _, rto, emt = make_metadata(rank, counts)
        name = f"{dist_name}/{tc_name}"
        buf_key = f"ep_{dist_name}_{tc_name}"

        log(rank, f"--- {name} | tokens={counts} | dtype={dtype} ---")
        passed = run_case(
            name, rank, world_size, counts, psums,
            ep_group, buf_key,
            hidden, dtype,
            rto, emt, ENGINE_MAX_TOKENS,
        )
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        log(rank, f"  {status} [{name}]")
        results[name] = passed

    # ── Multi-call: second RSV with different fill must overwrite first output ──
    log(rank, "--- multi_call: second RSV overwrites first ---")
    counts = distributions["linear_inc"]
    psums, _, rto, emt = make_metadata(rank, counts)
    hidden, dtype = 2688, torch.bfloat16
    total_tokens = sum(counts)
    local_tokens = counts[rank]

    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    buf = SymmetricMemoryManager.get_buffer("ep_multi", process_group=ep_group)
    result = buf.maybe_get_tensor([total_tokens, hidden], dtype=dtype)
    if result["handle"] is not None:
        symm_tensor = result["tensor"]
        symm_mem_hdl = result["handle"]
        output_tensor = torch.empty((local_tokens, hidden), dtype=dtype, device="cuda")

        # Call 1: fill with scale=1
        symm_tensor.fill_(float(rank + 1))
        torch.cuda.synchronize()
        run_rsv(symm_tensor, output_tensor, symm_mem_hdl, rto, emt, ENGINE_MAX_TOKENS)

        # Call 2: fill with scale=2 — must fully overwrite call 1's output
        symm_tensor.fill_(2.0 * float(rank + 1))
        torch.cuda.synchronize()
        run_rsv(symm_tensor, output_tensor, symm_mem_hdl, rto, emt, ENGINE_MAX_TOKENS, n_warmup=1)

        exp = 2.0 * expected_val(world_size)
        passed = verify(rank, "multi_call", output_tensor, local_tokens, hidden, dtype, exp)
        log(rank, f"  {'PASS' if passed else 'FAIL'} [multi_call]")
        results["multi_call"] = passed
        SymmetricMemoryManager.destroy("ep_multi")

    # ── ep_max_tokens == engine_max_tokens: no CTA exits early ─────────────────
    log(rank, "--- ep_max_eq_engine: ep_max_tokens == engine_max_tokens ---")
    counts = distributions["linear_inc"]
    psums, _, rto, _ = make_metadata(rank, counts)
    # Set ep_max_tokens equal to ENGINE_MAX_TOKENS — no CTA exits early, all hit the barrier.
    emt_full = torch.tensor([ENGINE_MAX_TOKENS], dtype=torch.int32, device="cuda")
    passed = run_case(
        "ep_max_eq_engine", rank, world_size, counts, psums,
        ep_group, "ep_max_eq",
        hidden=2688, dtype=torch.bfloat16,
        rank_token_offset=rto, ep_max_tokens=emt_full, engine_max_tokens=ENGINE_MAX_TOKENS,
    )
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    log(rank, f"  {status} [ep_max_eq_engine]")
    results["ep_max_eq_engine"] = passed

    # ── Summary ────────────────────────────────────────────────────────────────
    n_skip = sum(1 for v in results.values() if v is None)
    n_pass = sum(1 for v in results.values() if v is True)
    n_fail = sum(1 for v in results.values() if v is False)
    log(rank, f"=== {n_pass} passed, {n_fail} failed, {n_skip} skipped "
              f"({len(results)} total) ===")

    dist.destroy_process_group()
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
