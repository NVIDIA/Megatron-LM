"""Exhaustive tests for multimem_all_gather_v (128-bit and 64-bit NVLS paths).

Launch with:
    torchrun --nproc_per_node=NUM_GPUS test_agv.py

Covers the cross-product of token distributions × tensor specs, plus multi-call
and ep_max_tokens edge-case tests.

Token distributions (rank r):
  linear_inc  BASE + r * STEP          (offset grows toward higher ranks)
  linear_dec  BASE + (WS-1-r) * STEP   (offset grows toward lower ranks)
  uniform     same count on all ranks  (degenerate variable case)
  skewed      rank 0 has 4×, others 1× (stress-tests prefix-sum calculation)

Tensor specs (name, hidden, dtype, expected BITS path):
  bf16_h2688   2688  bfloat16  128  (5376 B/row, 16B-aligned)
  fp32_topk6      6  float32    64  (  24 B/row, 8B-aligned only)
  fp32_topk22    22  float32    64  (  88 B/row, 8B-aligned only)
  int64_topk6     6  int64     128  (  48 B/row, 16B-aligned)
  int64_topk22   22  int64     128  ( 176 B/row, 16B-aligned)

Not tested here:
  - output_byte_offset != 0 (requires raw-buffer gymnastics with SymmetricMemoryManager)
  - numel_per_token > BLOCK_SIZE inner-loop path (needs hidden > 8192 for bf16)
"""
import os
import sys
from itertools import product as iproduct

import torch
import torch.distributed as dist

BASE_TOKENS = 32
TOKEN_STEP = 16

# (name, hidden, dtype, expected_bits)
TENSOR_SPECS = [
    ("bf16_h2688",   2688, torch.bfloat16, 128),
    ("fp32_topk6",      6, torch.float32,   64),
    ("fp32_topk22",    22, torch.float32,   64),
    ("int64_topk6",     6, torch.int64,    128),
    ("int64_topk22",   22, torch.int64,    128),
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


def verify(rank, world_size, name, output_tensor, local_tokens_per_rank, prefix_sums, fill_base=0.0):
    """Check that rank r's slice equals fill_base + float(r) for every r."""
    hidden = output_tensor.shape[1]
    dtype = output_tensor.dtype
    passed = True
    for r in range(world_size):
        start = prefix_sums[r]
        end = start + local_tokens_per_rank[r]
        expected = torch.full(
            (local_tokens_per_rank[r], hidden),
            fill_base + float(r),
            dtype=dtype, device="cuda",
        )
        if not torch.equal(output_tensor[start:end], expected):
            log(rank, f"  FAIL [{name}]: rank {r} slice mismatch")
            passed = False
    return passed


def run_agv(output_tensor, input_tensor, symm_mem_hdl,
            rank_token_offset, ep_max_tokens, engine_max_tokens, n_warmup=3):
    from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
        multimem_all_gather_v,
    )
    for _ in range(n_warmup):
        multimem_all_gather_v(
            output_tensor, input_tensor, symm_mem_hdl,
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max_tokens,
        )
    torch.cuda.synchronize()


def alloc_and_run(
    name, rank, world_size,
    local_tokens_per_rank, psums,
    ep_group, buf_key,
    hidden, dtype, expected_bits,
    rank_token_offset, ep_max_tokens, engine_max_tokens,
    fill_base=0.0,
):
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    # Verify BITS path selection before launching.
    row_bytes = hidden * torch.tensor([], dtype=dtype).element_size()
    actual_bits = 128 if row_bytes % 16 == 0 else 64
    assert actual_bits == expected_bits, (
        f"[{name}] BITS mismatch: expected {expected_bits}, got {actual_bits} "
        f"(hidden={hidden}, dtype={dtype}, row_bytes={row_bytes})"
    )

    total_tokens = sum(local_tokens_per_rank)
    buf = SymmetricMemoryManager.get_buffer(buf_key, process_group=ep_group)
    result = buf.maybe_get_tensor([total_tokens, hidden], dtype=dtype)
    if result["handle"] is None:
        return None  # symmetric memory unavailable — skip

    output_tensor = result["tensor"]
    symm_mem_hdl = result["handle"]

    local_tokens = local_tokens_per_rank[rank]
    input_tensor = torch.full(
        (local_tokens, hidden), fill_base + float(rank), dtype=dtype, device="cuda"
    ).contiguous()

    run_agv(output_tensor, input_tensor, symm_mem_hdl,
            rank_token_offset, ep_max_tokens, engine_max_tokens)

    passed = verify(rank, world_size, name, output_tensor,
                    local_tokens_per_rank, psums, fill_base)
    SymmetricMemoryManager.destroy(buf_key)
    return passed


def make_metadata(rank, local_tokens_per_rank):
    psums = prefix_sum(local_tokens_per_rank)
    engine_max_tokens = max(local_tokens_per_rank)
    rank_token_offset = torch.tensor([psums[rank]], dtype=torch.int32, device="cuda")
    ep_max_tokens = torch.tensor([engine_max_tokens], dtype=torch.int32, device="cuda")
    return psums, engine_max_tokens, rank_token_offset, ep_max_tokens


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
    distributions = make_distributions(world_size)
    results = {}

    # ── Cross-product: distributions × tensor specs ────────────────────────────
    for (dist_name, counts), (tc_name, hidden, dtype, expected_bits) in iproduct(
        distributions.items(), TENSOR_SPECS
    ):
        psums, engine_max, rto, emt = make_metadata(rank, counts)
        name = f"{dist_name}/{tc_name}"
        buf_key = f"ep_{dist_name}_{tc_name}"

        log(rank, f"--- {name} | tokens={counts} | BITS={expected_bits} ---")
        passed = alloc_and_run(
            name, rank, world_size, counts, psums,
            ep_group, buf_key,
            hidden, dtype, expected_bits,
            rto, emt, engine_max,
        )
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        log(rank, f"  {status} [{name}]")
        results[name] = passed

    # ── Multi-call: second AGV with different fill must overwrite first ─────────
    log(rank, "--- multi_call: second AGV overwrites first ---")
    counts = distributions["linear_inc"]
    psums, engine_max, rto, emt = make_metadata(rank, counts)
    hidden, dtype, expected_bits = 2688, torch.bfloat16, 128
    total_tokens = sum(counts)

    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    buf = SymmetricMemoryManager.get_buffer("ep_multi", process_group=ep_group)
    result = buf.maybe_get_tensor([total_tokens, hidden], dtype=dtype)
    if result["handle"] is not None:
        output_tensor = result["tensor"]
        symm_mem_hdl = result["handle"]
        local_tokens = counts[rank]

        # Call 1: fill with float(rank)
        inp1 = torch.full((local_tokens, hidden), float(rank), dtype=dtype, device="cuda").contiguous()
        run_agv(output_tensor, inp1, symm_mem_hdl, rto, emt, engine_max)

        # Call 2: fill with float(rank) + 100 — must fully overwrite call 1
        inp2 = torch.full((local_tokens, hidden), float(rank) + 100.0, dtype=dtype, device="cuda").contiguous()
        run_agv(output_tensor, inp2, symm_mem_hdl, rto, emt, engine_max, n_warmup=1)

        passed = verify(rank, world_size, "multi_call", output_tensor, counts, psums, fill_base=100.0)
        log(rank, f"  {'PASS' if passed else 'FAIL'} [multi_call]")
        results["multi_call"] = passed
        SymmetricMemoryManager.destroy("ep_multi")

    # ── ep_max_tokens == engine_max_tokens: no CTA exits early ─────────────────
    log(rank, "--- ep_max_eq_engine: ep_max_tokens == engine_max_tokens ---")
    counts = distributions["linear_inc"]
    psums, engine_max, rto, _ = make_metadata(rank, counts)
    # Set ep_max_tokens equal to engine_max — no early exit, all CTAs hit the barrier.
    emt_full = torch.tensor([engine_max], dtype=torch.int32, device="cuda")
    passed = alloc_and_run(
        "ep_max_eq_engine", rank, world_size, counts, psums,
        ep_group, "ep_max_eq",
        hidden=2688, dtype=torch.bfloat16, expected_bits=128,
        rank_token_offset=rto, ep_max_tokens=emt_full, engine_max_tokens=engine_max,
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
