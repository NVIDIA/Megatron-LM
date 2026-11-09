# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the NVLS MoE dispatcher's combine: `_multimem_reduce_scatter_v_kernel`
(megatron/core/inference/communication/torch_symm_triton/variable_collectives.py:153, launched by
`multimem_reduce_scatter_v`, :250) and its batch-invariant alternative
`_ordered_reduce_scatter_v_kernel` (megatron/core/inference/moe/batch_invariant.py:440, launched
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
  * multimem: fp32 (production: the dispatcher's buffer is fp32), added in an order the switch
    picks, or bf16 (fp32 accumulation, one rounding). Batch-invariant mode with
    batch_invariant_collective="multimem" (token_dispatcher_inference.py:666-675) needs a
    token's combined row to depend only on its partials: the same bits on every call, whatever
    its row position, the other tokens and the per-rank counts. It does not need correct
    rounding. Megatron allows that collective only where the switch provides this: on
    Blackwell. On Hopper the TransformerConfig refuses it, as deterministic training already
    refuses NVLS on every Hopper GPU (NVLS reductions there can depend on the multicast objects
    allocated before them, unless the driver and Fabric Manager carry the fix:
    NVIDIA/nccl#2360; docs/user-guide/deterministic-training.md). Findings O5-N3 and O5-N4.
  * ordered: fp32 only; ((P_0 + P_1) + P_2) + ... in rank order, each add rounded to nearest in
    fp32, so the result depends on nothing else in the batch.
  Only rows [off_r, off_r + c_r) of the buffers are read and nothing but the output is written.
  The kernels barrier on entry (every partial is written before any rank reads) and not on exit
  (the next write to the buffer must follow a synchronizing collective: in the dispatcher, the
  next layer's all-gather-v; see test_multimem_all_gatherv_3tensor.py). Blocks at or past
  ep_max_tokens skip the barrier on every rank together.

Tolerance classes:
  * exact on ``nu.EXACT`` partials (sums exact in bf16 under any order and rounding) for both
    kernels: addressing bit for bit on every architecture;
  * multimem fp32 on N(0, 1) data: the derived bound of recursive fp32 summation in any order,
    element by element (``nu.assert_fp32_sum_within_bound``); bf16: bounded against the fp64 sum
    of the NCCL-gathered partials, calibrated by NCCL's fp32 all-reduce rounded once to bf16;
  * multimem fp32 batch invariance: exact (bitwise) across calls, positions and co-batches, on
    Blackwell; on Hopper the config contract instead: "multimem" is refused (needs the fix PR
    s4f-multimem-rounding-doc; fails on main);
  * ordered on N(0, 1) data: exact against the rank-order fp32 sum in torch.

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
world size inside the dispatcher and asserts the correctly-rounded claim; it passed on H100 because
expert outputs carry few significant bits, so their fp32 sums are exact in any order.
`test_hopper_refuses_batch_invariant_multimem` fails on main: it is the contract the fix PR
s4f-multimem-rounding-doc adds (a config-time rejection on Hopper).

Since the medium-risk batch: production shapes at the whole node for both combines
(`test_production_shapes`: DeepSeek-V3, Nemotron 3 Nano and Nemotron 3 Super's latent MoE,
2048 tokens per step); NaN and infinities in the partials (`test_special_values_propagate`);
one compute-sanitizer case per kernel (`test_sanitizer_small_case`); and the int32 limits of
the row offsets (`test_int32_row_offsets`, `large`: 8 GiB of fp32 rows for the ordered kernel,
16 GiB for the multimem one; suspected findings O5-M1 and O5-M2 on main).
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


def blackwell_test(fn):
    """Blackwell only: where Megatron allows batch-invariant "multimem"."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.BLACKWELL)(fn))


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
        if data == "exact":
            partial = ou.dyadic_tensor(shape, dtype=dtype, device="cuda", generator=gen, **nu.EXACT)
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
            "data": ["exact", "randn"],
            "ws": nu.GROUP_SIZES,
        }
    )
    params = []
    for c in cases:
        (dtype, hidden), prm = c["row"], c["per_rank_max"]
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
            "data": ["exact", "randn"],
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
    run_multimem_combine(case, data)


def run_multimem_combine(case, data):
    """Launch the multimem combine on ``case`` and check it on every rank (see the module
    docstring for the tolerance classes)."""
    result = case.launch(multimem_reduce_scatter_v)
    case.snapshot()

    def check():
        assert result is case.out.tensor, "multimem_reduce_scatter_v must return output_tensor"
        case.check_memory()
        out = case.out.tensor
        if data == "exact":
            ou.assert_exact(out, case.nccl_sum.to(out.dtype), what="combine (exact sum)")
        elif out.dtype == torch.float32:
            nu.assert_fp32_sum_within_bound(out, case.parts, what="combine")
        else:
            ou.assert_close_to_reference(
                out,
                ou.fp64_reference(sum_reference, case.parts),
                case.nccl_sum.to(out.dtype),
                what="combine",
            )

    nu.agree(check, f"multimem combine, counts {case.counts}")


PROBES = 3
"""Probe tokens per rank in the batch-invariance test."""

SPREAD = dict(mantissa_bits=24, exponents=(-8, 8))
"""fp32 values with full significands and exponents over 2**+-8: their fp32 sums must round, so
the order of the adds shows in the bits (and their fp64 sums are exact)."""


@blackwell_test
@pytest.mark.parametrize("hidden", [7168, 128])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_multimem_fp32_combine_is_batch_invariant(nvls_groups, ws, hidden):
    """What batch-invariant mode with collective "multimem" needs from the fp32 in-switch sum:
    each rank owns PROBES probe tokens with fixed partials (a request stays on its EP rank);
    over several calls the per-rank counts, the probes' row positions, the co-batched tokens
    and the timing (a random rank is delayed) change, and every probe's combined row must keep
    its bits. Every call is also checked against the fp32 summation bound. Correct rounding is
    not required. Blackwell only: on Hopper Megatron refuses the collective
    (test_hopper_refuses_batch_invariant_multimem)."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    per_rank_max, probes, calls = 129, PROBES, 6
    global_max = per_rank_max * ws
    seed = nu.case_seed("rsv-batch-invariance", ws, hidden)
    shared = g.shared_generator(seed)
    gen = g.rank_generator(seed)
    # This rank's contribution to probe k of owner o.
    probe_parts = ou.dyadic_tensor(
        (ws, probes, hidden), dtype=torch.float32, device="cuda", generator=gen, **SPREAD
    )
    view = symm.view((global_max, hidden), torch.float32)
    results = []
    for _ in range(calls):
        counts = [int(c) for c in torch.randint(probes, per_rank_max + 1, (ws,), generator=shared)]
        offsets = nu.offsets_from_counts(counts)
        positions = [torch.randperm(c, generator=shared)[:probes].tolist() for c in counts]
        partial = ou.dyadic_tensor(
            (global_max, hidden), dtype=torch.float32, device="cuda", generator=gen, **SPREAD
        )
        partial[sum(counts) :] = float("nan")
        for owner in range(ws):
            for k, position in enumerate(positions[owner]):
                partial[offsets[owner] + position] = probe_parts[owner, k]
        rows = slice(offsets[g.rank], offsets[g.rank] + counts[g.rank])
        # The gather also orders every rank's previous combine (no exit barrier) before this
        # rank rewrites its buffer, as the next all-gather-v does in the dispatcher.
        parts = g.gather(partial)[:, rows]
        view.copy_(partial)
        out = torch.empty(counts[g.rank], hidden, dtype=torch.float32, device="cuda")
        g.maybe_sleep(shared)
        multimem_reduce_scatter_v(
            out,
            view,
            symm.hdl,
            rank_token_offset=torch.tensor([offsets[g.rank]], dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=per_rank_max,
        )
        results.append((out, parts, positions[g.rank], counts))
    nu.sync_world()

    def check():
        first = None
        for call, (out, parts, positions, counts) in enumerate(results):
            nu.assert_fp32_sum_within_bound(out, parts, what=f"call {call} (counts {counts})")
            probe_rows = out[positions]
            if first is None:
                first = probe_rows
            ou.assert_exact(
                probe_rows,
                first,
                what=f"call {call} (counts {counts}, rows {positions}): probe rows vs call 0",
            )

    nu.agree(check, "batch-invariant fp32 combine")


def batch_invariant_config(collective):
    """A minimal TransformerConfig in batch-invariant mode with the given EP combine collective
    (the rest set to what batch-invariant mode requires)."""
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4 if ou.device_sm_major() == ou.BLACKWELL else 3,
        batch_invariant_mode=True,
        batch_invariant_collective=collective,
    )


@ou.requires_sm(ou.HOPPER)
def test_hopper_refuses_batch_invariant_multimem():
    """On Hopper, batch-invariant mode with batch_invariant_collective="multimem" is refused when
    the config is built ("ordered", the default, is accepted). The in-switch sum there is not
    guaranteed to depend on a token's partials alone (NVLS reductions can depend on the
    multicast objects allocated before them unless the driver and Fabric Manager carry the fix,
    NVIDIA/nccl#2360), and Megatron does not check those versions, as for deterministic
    training. Fails on main; the fix PR s4f-multimem-rounding-doc adds the rejection."""
    batch_invariant_config("ordered")
    with pytest.raises((AssertionError, ValueError)):
        batch_invariant_config("multimem")


@blackwell_test
def test_blackwell_accepts_batch_invariant_multimem():
    """On Blackwell both collectives are accepted: test_multimem_fp32_combine_is_batch_invariant
    checks the in-switch sum gives what batch-invariant mode needs. Passes on main."""
    assert batch_invariant_config("ordered").batch_invariant_collective == "ordered"
    assert batch_invariant_config("multimem").batch_invariant_collective == "multimem"


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
    run_ordered_combine(case)


def run_ordered_combine(case):
    """Launch the ordered combine on ``case`` and check it bit for bit against the rank-order
    fp32 sum on every rank."""
    result = case.launch(ordered_reduce_scatter_v)
    case.snapshot()

    def check():
        assert result is case.out.tensor, "ordered_reduce_scatter_v must return output_tensor"
        case.check_memory()
        ou.assert_exact(case.out.tensor, rank_ordered_sum(case.parts), what="rank-order sum")

    nu.agree(check, f"ordered combine, counts {case.counts}")


COMBINES = {"multimem": multimem_reduce_scatter_v, "ordered": ordered_reduce_scatter_v}


def _production_params():
    cases = ou.pairwise_cases(
        {
            "combine": list(COMBINES),
            "model": list(nu.PRODUCTION_MOE),
            "counts": ["full", "uneven"],
            "data": ["exact", "randn"],
        }
    )
    return [
        pytest.param(
            c["combine"],
            c["model"],
            c["counts"],
            c["data"],
            id=f"{c['combine']}-{c['model']}-{c['counts']}-{c['data']}",
        )
        for c in cases
    ]


@gpu_test
@pytest.mark.parametrize("combine,model,counts_pattern,data", _production_params())
def test_production_shapes(nvls_groups, combine, model, counts_pattern, data):
    """The sweeps' checks at production shapes and group size (EP 8 on H100, EP 4 on GB200):
    the fp32 combine buffer of DeepSeek-V3 (H 7168: 1792 lanes, two windows of the multimem
    kernel, seven of the ordered one), Nemotron 3 Nano (H 2688) and Nemotron 3 Super's latent
    MoE (1024), with ``nu.PRODUCTION_TOKENS`` tokens per step (per_rank_max 256 at EP 8, 512 at
    EP 4, so the dispatcher's 128-block grid with a grid-stride loop), every rank at
    per_rank_max (a padded CUDA-graph step) or uneven counts (an eager step)."""
    ws = nu.PRODUCTION_GROUP_SIZE
    g = nvls_groups.get(ws)
    hidden, _ = nu.PRODUCTION_MOE[model]
    per_rank_max = nu.PRODUCTION_TOKENS // ws
    seed = nu.case_seed("rsv-production", combine, model, counts_pattern, data)
    counts = nu.production_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    case = CombineCase(g, hidden, torch.float32, per_rank_max, counts, 0, data, seed)
    if combine == "multimem":
        run_multimem_combine(case, data)
    else:
        run_ordered_combine(case)


@gpu_test
@pytest.mark.parametrize("combine", list(COMBINES))
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_special_values_propagate(nvls_groups, ws, combine):
    """NaN and infinities in an expert output reach the combined row as IEEE addition says
    (``nu.special_partials``: NaN stays NaN, infinities stay, +inf + -inf is NaN, finite
    columns exact; the sign of an all-zero sum is not asserted: the ordered kernel starts from
    +0.0). Every rank owns 3 tokens whose rows all hold the classes; fp32, the production
    dtype of both combines."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    parts = nu.special_partials(ws, 2 * ws, torch.float32)
    hidden = parts.shape[1]
    rows = 3
    view = symm.view((rows * ws, hidden), torch.float32)
    view.copy_(parts[g.rank].expand(rows * ws, hidden))
    out = ou.guarded_empty((rows, hidden), torch.float32)
    COMBINES[combine](
        out.tensor,
        view,
        symm.hdl,
        rank_token_offset=torch.tensor([rows * g.rank], dtype=torch.int32, device="cuda"),
        ep_max_tokens=torch.tensor([rows], dtype=torch.int32, device="cuda"),
        per_rank_max_tokens=rows,
    )
    got = out.tensor.clone()

    def check():
        ou.check_guards(output=out)
        for t in range(rows):
            nu.assert_ieee_sum(got[t], parts, what=f"{combine} combine, token {t}")

    nu.agree(check, f"{combine} combine of special values")


SANITIZER_CASES = {
    # combine: (hidden, why)
    "multimem": 4100,  # 1025 fp32 lanes: a full 1024-lane window, then one lane
    "ordered": 1025,  # a full 1024-element block, then one element
}


@gpu_test
@pytest.mark.sanitizer
@pytest.mark.parametrize("combine", list(SANITIZER_CASES))
def test_sanitizer_small_case(nvls_groups, combine):
    """The compute-sanitizer (memcheck) cases, one per kernel: ws 2, fp32 rows with a partial
    last window (SANITIZER_CASES), per_rank_max 129 (the 128-block grid and one grid-stride
    step), uneven counts (one rank empty), input_byte_offset 1 MiB (sentinels on both sides),
    N(0, 1) data, with the sweeps' checks. A dedicated function, so the runner selects it by
    name at any world size."""
    g = nvls_groups.get(2)
    hidden, per_rank_max = SANITIZER_CASES[combine], 129
    seed = nu.case_seed("rsv-sanitizer", combine)
    counts = nu.token_counts("uneven", 2, per_rank_max, g.shared_generator(seed))
    case = CombineCase(g, hidden, torch.float32, per_rank_max, counts, 1 << 20, "randn", seed)
    if combine == "multimem":
        run_multimem_combine(case, "randn")
    else:
        run_ordered_combine(case)


INT32_HIDDEN = 7168
"""Row width of the int32-limit cases (DeepSeek-V3's hidden size)."""

INT32_LIMITS = {
    # combine: (row width in the unit the kernel indexes, the int32 limit in that unit)
    # multimem: 16-byte lanes, scaled by 2 into 8-byte units (variable_collectives.py:239-243)
    "multimem": (INT32_HIDDEN // 4, 2**30),
    # ordered: fp32 elements, global_token * HIDDEN_SIZE + offsets (batch_invariant.py:486)
    "ordered": (INT32_HIDDEN, 2**31),
}


def _int32_params():
    labels = ["fits", "crosses", "row-starts-past"]
    params = []
    for k, label in enumerate(labels):  # every control first: a fault later poisons the process
        for combine, (row_units, limit) in INT32_LIMITS.items():
            rows = ou.int32_overflow_rows(row_units, limit=limit)[k]
            params.append(
                pytest.param(
                    combine, rows, id=f"{combine}-{rows}[int32-{label}]", marks=pytest.mark.large
                )
            )
    return params


@gpu_test
@pytest.mark.parametrize("combine,rows", _int32_params())
def test_int32_row_offsets(nvls_groups, combine, rows):
    """Row offsets at the int32 limit (`large`): the combine buffer holds ``rows`` fp32 rows of
    H 7168 and the last rank's rows reach the row whose offsets cross 2**31 in the unit the
    kernel indexes (INT32_LIMITS): 8 GiB of rows for the ordered kernel, 16 GiB for the
    multimem one. A dispatcher would reach these with EP 32 (ordered) or EP 64 (multimem) at
    the default 16384 max tokens and H 7168, e.g. on a GB200 NVL72 domain. Contract: the same sums
    as below the limit. ws 2, counts split in half, data in closed form
    (``nu.closed_form_rows``: every rank computes the expected sums without moving 16 GiB),
    exact. The `fits` ids are controls. Suspected on main (findings O5-M1, O5-M2): the row
    offsets are int32, so the `crosses` and `row-starts-past` ids read rows 16 or 32 GiB
    before the buffer: wrong sums or an illegal-address fault. A fault on one rank leaves its
    peer waiting in the next collective until the NCCL timeout: run each id in its own job."""
    g = nvls_groups.get(2)
    hidden = INT32_HIDDEN
    counts = [rows // 2, rows - rows // 2]
    first = counts[0] * g.rank
    mine = counts[g.rank]
    row_bytes = hidden * 4
    allocated = "large" in g.buffers
    need = mine * row_bytes + (0 if allocated else nu.LARGE_BUFFER_MB << 20) + (2 << 30)
    nu.require_free_memory_everywhere(need)
    symm = g.large_buffer()
    view = symm.view((rows, hidden), torch.float32)
    for start, count in nu.row_chunks(0, rows, hidden):
        view[start : start + count] = nu.closed_form_rows(
            g.rank, start, count, hidden, torch.float32
        )
    after = symm.sentinel_after(rows * row_bytes)
    out = torch.empty((mine, hidden), dtype=torch.float32, device="cuda")
    COMBINES[combine](
        out,
        view,
        symm.hdl,
        rank_token_offset=torch.tensor([first], dtype=torch.int32, device="cuda"),
        ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
        per_rank_max_tokens=max(counts),
    )

    def check():
        for start, count in nu.row_chunks(0, mine, hidden):
            expected = sum(
                nu.closed_form_rows(q, first + start, count, hidden, torch.float32)
                for q in range(2)
            )
            ou.assert_exact(
                out[start : start + count],
                expected,
                what=f"{combine} combine, global rows {first + start}-{first + start + count - 1}",
            )
        after.check("large buffer after the rows")

    nu.agree(check, f"{combine} combine of {rows} rows (counts {counts})")
