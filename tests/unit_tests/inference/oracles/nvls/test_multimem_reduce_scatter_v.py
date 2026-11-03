# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the NVLS MoE dispatcher's combine: `_multimem_reduce_scatter_v_kernel`
(megatron/core/inference/communication/torch_symm_triton/variable_collectives.py:153, launched by
`multimem_reduce_scatter_v`, :250) and its batch-invariant alternative
`_ordered_reduce_scatter_v_kernel` (megatron/core/inference/moe/batch_invariant.py:309, launched
by `ordered_reduce_scatter_v`; the dispatcher picks it with batch_invariant_collective="ordered",
token_dispatcher_inference.py:666-682). The ordered kernel's audit record is in family 02.

Contract (from the launchers' docstrings and `NVLSAllGatherVDispatcher.token_combine`):
  Rank q of a group of ws ranks owns c_q tokens (counts differ between ranks and may be 0). The
  global rows are laid out by rank: rank q's rows start at off_q = c_0 + ... + c_{q-1}, and the
  buffers hold global_max = per_rank_max * ws rows. Every rank's symmetric buffer holds, at
  input_byte_offset, its partial P_q [global_max, H] (its expert outputs for every global row).
  rank_token_offset holds off_r and ep_max_tokens holds max_q c_q (device int32 scalars, the
  latter the same on every rank); per_rank_max_tokens is static and fixes the grid. Rank r's
  output is out[i] = sum_q P_q[off_r + i] for i < c_r (shape [c_r, H]).
  * multimem: fp32 (production: the dispatcher's buffer is fp32) or bf16 (fp32 accumulation,
    one rounding). The dispatcher documents the fp32 in-switch sum as correctly rounded,
    "bitwise-equal to an fp64 reference" (token_dispatcher_inference.py:667-671), which batch-
    invariant mode with batch_invariant_collective="multimem" relies on.
  * ordered: fp32 only; ((P_0 + P_1) + P_2) + ... in rank order, each add rounded to nearest in
    fp32, so the result depends on nothing else in the batch.
  Only rows [off_r, off_r + c_r) of the buffers are read and nothing but the output is written.
  The kernels barrier on entry (every partial is written before any rank reads) and not on exit
  (the next write to the buffer must follow a synchronizing collective: in the dispatcher, the
  next layer's all-gather-v; see test_multimem_all_gatherv_3tensor.py). Blocks at or past
  ep_max_tokens skip the barrier on every rank together.

Tolerance classes: exact on dyadic partials (sums exact in fp32 in any order) for both kernels;
multimem on N(0, 1) data bounded against the fp64 sum of the NCCL-gathered partials, calibrated
by NCCL's fp32 all-reduce rounded to the dtype, plus (fp32) the documented exactness against the
fp64 sum rounded once; ordered on N(0, 1) data exact against the rank-order fp32 sum in torch.

Structure-derived inputs: one block per token (grid min(per_rank_max, 128), grid-stride beyond
128 tokens per rank; per_rank_max in {1, 127, 128, 129, 300}); multimem blocks of
min(next_pow2(H/lanes), 1024) 16-byte lanes (4 fp32 or 8 bf16): fp32 H in {4, 128, 2048,
4092, 4096, 4100, 7168 (DeepSeek-V3, two windows)}, bf16 H in {8, 2048, 8184, 8192, 8200};
ordered blocks of min(next_pow2(H), 1024) elements: H in {4, 1023, 1024, 1025, 7168}. Counts:
uneven (one rank at per_rank_max, the last at 0), all at per_rank_max, one rank only, and all 0
(ep_max_tokens = 0: every block exits). input_byte_offset in {0, 1 MiB}; ws in {2, 4, 8};
combined pairwise. Memory: rows nobody owns are NaN in every partial (a read of them turns the
output NaN), the output is NaN-poisoned with guard bands, the buffer and the two device scalars
must be unchanged, and 64 KiB sentinels surround the buffer region. CUDA-graph replay with
changing counts, and back-to-back use, are in test_multimem_all_gatherv_3tensor.py.

#8868 (t3-r04f3) makes the ordered kernel's adds `add_rn_f32` inline asm, because LLVM may
reorder a chain of plain adds (it can turn it into a vector reduction that runs from the last
rank); `test_ordered_sum_is_rank_ordered` is the case it must pass, and it fails on main if that
reordering happens on the target compiler. #8879 (t3-r09f1) runs the multimem combine at EP =
world size inside the dispatcher; this file checks the kernel against the same claim directly.
"""

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
    multimem_reduce_scatter_v,
)
from megatron.core.inference.moe.batch_invariant import ordered_reduce_scatter_v
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MULTIMEM_ROWS = [("fp32", h) for h in (4, 128, 2048, 4092, 4096, 4100, 7168)] + [
    ("bf16", h) for h in (8, 2048, 8184, 8192, 8200)
]
ORDERED_HIDDEN = [4, 1023, 1024, 1025, 7168]
BYTE_OFFSETS = [0, 1 << 20]


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def sum_reference(parts):
    """The sum over ranks (dim 0), fp64 under ``ou.fp64_reference``."""
    return parts.sum(0)


def rank_ordered_sum(parts):
    """((P_0 + P_1) + P_2) + ...: torch's fp32 adds round to nearest one at a time."""
    acc = parts[0].clone()
    for q in range(1, parts.shape[0]):
        acc = acc + parts[q]
    return acc


class CombineCase:
    """Partials in every rank's buffer, the step metadata, and the NCCL references."""

    def __init__(self, g, hidden, dtype, per_rank_max, counts, byte_offset, data, seed):
        self.g, self.counts, self.byte_offset = g, counts, byte_offset
        ws = g.ws
        self.global_max = per_rank_max * ws
        self.per_rank_max = per_rank_max
        offsets = nu.offsets_from_counts(counts)
        self.total = sum(counts)
        self.rows = slice(offsets[g.rank], offsets[g.rank] + counts[g.rank])
        gen = g.rank_generator(seed)
        shape = (self.global_max, hidden)
        if data == "dyadic":
            partial = ou.dyadic_tensor(
                shape, dtype=dtype, device="cuda", generator=gen, **nu.DYADIC
            )
        elif data == "spread":
            # Full 24-bit significands, exponents over 2**-8 .. 2**8: the fp32 sum must round,
            # and the fp64 sum is exact (multiples of 2**-8 below 2**35: 43 bits).
            partial = ou.dyadic_tensor(
                shape,
                mantissa_bits=24,
                exponents=(-8, 8),
                dtype=dtype,
                device="cuda",
                generator=gen,
            )
        else:
            partial = torch.randn(shape, device="cuda", generator=gen).to(dtype)
        partial[self.total :] = float("nan")  # rows nobody owns: never read
        self.partial = partial
        self.parts = g.gather(partial)[:, self.rows]
        self.nccl_sum = g.all_reduce_fp32(partial[: self.total])[self.rows]
        symm = g.buffer()
        self.view = symm.view(shape, dtype, byte_offset)
        self.view.copy_(partial)
        region_end = byte_offset + partial.numel() * dtype.itemsize
        self.sentinels = [
            s
            for s in (symm.sentinel_before(byte_offset), symm.sentinel_after(region_end))
            if s is not None
        ]
        self.rank_token_offset = torch.tensor([offsets[g.rank]], dtype=torch.int32, device="cuda")
        self.ep_max_tokens = torch.tensor([max(counts)], dtype=torch.int32, device="cuda")
        self.out = ou.guarded_empty((counts[g.rank], hidden), dtype)

    def launch(self, launcher):
        return launcher(
            self.out.tensor,
            self.view,
            self.g.buffer().hdl,
            rank_token_offset=self.rank_token_offset,
            ep_max_tokens=self.ep_max_tokens,
            per_rank_max_tokens=self.per_rank_max,
            input_byte_offset=self.byte_offset,
        )

    def snapshot(self):
        """Copies taken after the launch (before any later collective changes the buffer)."""
        self.region_after = self.view.clone()

    def check_memory(self):
        ou.check_guards(output=self.out)
        ou.assert_exact(self.region_after, self.partial, what="buffer region (read-only)")
        for s in self.sentinels:
            s.check("buffer around the region")
        ou.assert_exact(
            self.rank_token_offset.cpu(),
            torch.tensor([self.rows.start]),
            what="rank_token_offset (read-only)",
        )
        ou.assert_exact(
            self.ep_max_tokens.cpu(),
            torch.tensor([max(self.counts)]),
            what="ep_max_tokens (read-only)",
        )


def _multimem_params():
    cases = ou.pairwise_cases(
        {
            "row": MULTIMEM_ROWS,
            "per_rank_max": nu.PER_RANK_MAX,
            "counts": nu.COUNTS,
            "byte_offset": BYTE_OFFSETS,
            "data": ["dyadic", "randn"],
            "ws": nu.GROUP_SIZES,
        }
    )
    params = []
    for c in cases:
        (dtype, hidden), prm = c["row"], c["per_rank_max"]
        sanitizer = (dtype, hidden, prm, c["ws"]) == ("fp32", 128, 127, 2)
        params.append(
            pytest.param(
                c["ws"],
                dtype,
                hidden,
                prm,
                c["counts"],
                c["byte_offset"],
                c["data"],
                id=f"ws{c['ws']}-{dtype}-H{hidden}-prm{prm}-{c['counts']}"
                f"-offset{c['byte_offset']}-{c['data']}",
                marks=[pytest.mark.sanitizer] if sanitizer else [],
            )
        )
    return params


def _ordered_params():
    cases = ou.pairwise_cases(
        {
            "hidden": ORDERED_HIDDEN,
            "per_rank_max": nu.PER_RANK_MAX,
            "counts": nu.COUNTS,
            "byte_offset": BYTE_OFFSETS,
            "data": ["dyadic", "randn"],
            "ws": nu.GROUP_SIZES,
        }
    )
    return [
        pytest.param(
            c["ws"],
            c["hidden"],
            c["per_rank_max"],
            c["counts"],
            c["byte_offset"],
            c["data"],
            id=f"ws{c['ws']}-H{c['hidden']}-prm{c['per_rank_max']}-{c['counts']}"
            f"-offset{c['byte_offset']}-{c['data']}",
        )
        for c in cases
    ]


@gpu_test
@pytest.mark.parametrize(
    "ws,dtype_name,hidden,per_rank_max,counts_pattern,byte_offset,data", _multimem_params()
)
def test_multimem_combine_matches_nccl(
    nvls_groups, ws, dtype_name, hidden, per_rank_max, counts_pattern, byte_offset, data
):
    """This rank's rows of the sum over ranks, read from the right rows only."""
    g = nvls_groups.get(ws)
    seed = nu.case_seed("rsv", ws, dtype_name, hidden, per_rank_max, counts_pattern, byte_offset)
    counts = nu.token_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    case = CombineCase(g, hidden, DTYPES[dtype_name], per_rank_max, counts, byte_offset, data, seed)
    result = case.launch(multimem_reduce_scatter_v)
    case.snapshot()

    def check():
        assert result is case.out.tensor, "multimem_reduce_scatter_v must return output_tensor"
        case.check_memory()
        out = case.out.tensor
        if data == "dyadic":
            ou.assert_exact(out, case.nccl_sum.to(out.dtype), what="combine (exact, dyadic)")
        else:
            ou.assert_close_to_reference(
                out,
                ou.fp64_reference(sum_reference, case.parts),
                case.nccl_sum.to(out.dtype),
                what="combine",
            )

    nu.agree(check, f"multimem combine, counts {counts}")


@gpu_test
@pytest.mark.parametrize(
    "hidden,per_rank_max,counts_pattern",
    [(7168, 129, "uneven"), (4100, 64, "full"), (128, 300, "uneven")],
)
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_multimem_fp32_sum_is_correctly_rounded(
    nvls_groups, ws, hidden, per_rank_max, counts_pattern
):
    """The dispatcher's claim for the fp32 in-switch sum (token_dispatcher_inference.py:667-671):
    bitwise equal to the fp64 sum rounded once, on fp32 values with full significands and
    exponents spread over 2**+-8 (their fp32 sums must round; their fp64 sums are exact).
    Batch-invariant mode with collective "multimem" depends on it."""
    g = nvls_groups.get(ws)
    seed = nu.case_seed("rsv-rounding", ws, hidden, per_rank_max, counts_pattern)
    counts = nu.token_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    case = CombineCase(g, hidden, torch.float32, per_rank_max, counts, 0, "spread", seed)
    case.launch(multimem_reduce_scatter_v)
    case.snapshot()

    def check():
        case.check_memory()
        expected = ou.fp64_reference(sum_reference, case.parts).to(torch.float32)
        ou.assert_exact(case.out.tensor, expected, what="fp32 combine vs fp64 sum rounded once")

    nu.agree(check, f"correctly rounded fp32 combine, counts {counts}")


@gpu_test
@pytest.mark.parametrize(
    "ws,hidden,per_rank_max,counts_pattern,byte_offset,data", _ordered_params()
)
def test_ordered_sum_is_rank_ordered(
    nvls_groups, ws, hidden, per_rank_max, counts_pattern, byte_offset, data
):
    """The batch-invariant combine adds the ranks' rows in rank order in fp32, bit for bit."""
    g = nvls_groups.get(ws)
    seed = nu.case_seed("ordered", ws, hidden, per_rank_max, counts_pattern, byte_offset, data)
    counts = nu.token_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    case = CombineCase(g, hidden, torch.float32, per_rank_max, counts, byte_offset, data, seed)
    result = case.launch(ordered_reduce_scatter_v)
    case.snapshot()

    def check():
        assert result is case.out.tensor, "ordered_reduce_scatter_v must return output_tensor"
        case.check_memory()
        ou.assert_exact(case.out.tensor, rank_ordered_sum(case.parts), what="rank-order sum")

    nu.agree(check, f"ordered combine, counts {counts}")
