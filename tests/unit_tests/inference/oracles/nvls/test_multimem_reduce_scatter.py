# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_multimem_reduce_scatter_kernel`
(megatron/core/inference/communication/torch_symm_triton/collectives.py:175), launched by
`multimem_reduce_scatter` (collectives.py:356): the NVLS reduce-scatter of the inference-optimized
row-parallel linears (non-fused TP) and of the vocab-parallel embedding.

Contract (from the launcher's docstring and its callers, inference_layers.py:461-526, 616-656):
every rank q of a group of ws ranks has written its input P_q (N elements, bf16 or fp32) at
offset 0 of its symmetric buffer. Rank r's output is out_r[i] = sum_q P_q[r * N/ws + i] for
i < N/ws: bf16 inputs accumulate in fp32 and round once to nearest even (the kernel's
`multimem.ld_reduce ... .add.acc::f32.v4.bf16x2`); fp32 inputs are added in fp32 in an order the
switch picks. Nothing else is written: the buffers are only read. The kernel barriers on entry
(every rank's input is written before any rank reads) and not on exit: when it returns on one
rank, peers may still be reading their segments of its buffer, so the next write to the buffer
must be an all-gather of the same per-rank size (which writes only this rank's segment) or come
after a collective that synchronizes the group. Preconditions (checked): dtype bf16 or fp32 on
both sides, output bytes % 16 == 0, input numel == ws * output numel.

Tolerance classes:
  * exact on ``nu.EXACT`` inputs, whose sums are exact in bf16 under any order and rounding (the
    reference is NCCL's fp32 reduce-scatter, exact here): addressing, bit for bit, on every
    architecture;
  * bf16 on N(0, 1) inputs: bounded against the fp64 sum of the NCCL-gathered inputs,
    calibrated by NCCL's fp32 reduce-scatter rounded once to bf16, the contract's arithmetic;
  * fp32 on N(0, 1) inputs: the derived bound of recursive fp32 summation in any order,
    element by element (``nu.assert_fp32_sum_within_bound``). NCCL's ring is not a comparable
    trusted implementation here: its order differs from the switch's, and on H100 the switch's
    order differs between GPU sets (finding O5-N3);
  * the bf16 rounding itself: exact against the exact sum rounded to nearest even, on designed
    sums (ties, quarter and three-quarter ulps, sticky bits, carries, sign, binade, and for
    ws >= 3 fp32 accumulation), in `test_bf16_in_switch_sum_rounding`. Fails on GB200 (finding
    O5-N4); its message lists, per class, which rounding rule the hardware followed.

Structure-derived inputs: 16-byte chunks per thread (8 bf16 or 4 fp32: the two compiled
variants, REDUCE_F32 False/True), blocks of 1024 threads, grids of at most 128 blocks with a
grid-stride loop. Per-rank output chunks: 1, 1023/1024/1025, 131071/131072/131073 and 262145;
ws in {2, 4, 8}; combined pairwise with dtype and data. Memory: the output is NaN-poisoned with
guard bands, the input region of the buffer must be unchanged, and a 64 KiB sentinel follows it.
The production order row-parallel RS -> next layer's equal-size all-gather without a barrier is
run back to back with changing token counts and data, eagerly and as CUDA graphs (one per token
count, replayed in a changing order), with a random rank delayed before each step.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton.collectives import (
    multimem_all_gather,
    multimem_reduce_scatter,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

CHUNK_BYTES = 16
BLOCK_SIZE = 1024  # threads per block (collectives.py:229-233)
MAX_NUM_BLOCKS = 128  # kernel_config.py
CHUNKS_PER_RANK = (
    [1]
    + ou.boundary_sizes(BLOCK_SIZE, BLOCK_SIZE * MAX_NUM_BLOCKS, multiples=(), degenerate=False)
    + [2 * BLOCK_SIZE * MAX_NUM_BLOCKS + 1]
)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
SANITIZER_CASE = dict(chunks=1025, ws=2)


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def sum_reference(parts):
    """The sum over ranks (dim 0), fp64 under ``ou.fp64_reference``."""
    return parts.sum(0)


def make_input(shape, dtype, data, generator):
    if data == "exact":
        return ou.dyadic_tensor(shape, dtype=dtype, device="cuda", generator=generator, **nu.EXACT)
    return torch.randn(shape, device="cuda", generator=generator).to(dtype)


def check_sum(out, parts, nccl_rs_fp32, data, what):
    """``out`` against the sum over ranks of ``parts`` ([ws, ...]), as described above."""
    if data == "exact":
        ou.assert_exact(out, nccl_rs_fp32.to(out.dtype), what=f"{what} (exact sum)")
    elif out.dtype == torch.float32:
        nu.assert_fp32_sum_within_bound(out, parts, what=what)
    else:
        ou.assert_close_to_reference(
            out, ou.fp64_reference(sum_reference, parts), nccl_rs_fp32.to(out.dtype), what=what
        )


def _sweep_params():
    cases = ou.pairwise_cases(
        {
            "chunks": CHUNKS_PER_RANK,
            "dtype": list(DTYPES),
            "data": ["exact", "randn"],
            "ws": nu.GROUP_SIZES,
        }
    )
    params = []
    for c in cases:
        sanitizer = c["chunks"] == SANITIZER_CASE["chunks"] and c["ws"] == SANITIZER_CASE["ws"]
        params.append(
            pytest.param(
                c["ws"],
                c["chunks"],
                c["dtype"],
                c["data"],
                id=f"ws{c['ws']}-chunks{c['chunks']}-{c['dtype']}-{c['data']}",
                marks=[pytest.mark.sanitizer] if sanitizer else [],
            )
        )
    return params


@gpu_test
@pytest.mark.parametrize("ws,chunks,dtype_name,data", _sweep_params())
def test_reduce_scatter_matches_nccl(nvls_groups, ws, chunks, dtype_name, data):
    """This rank's output against NCCL / fp64; the buffer is read only; output fully written."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    dtype = DTYPES[dtype_name]
    per_rank = chunks * CHUNK_BYTES // dtype.itemsize
    seed = nu.case_seed("reduce-scatter", ws, chunks, dtype_name, data)
    x = make_input((ws * per_rank,), dtype, data, g.rank_generator(seed))
    parts = g.gather(x)[:, g.rank * per_rank : (g.rank + 1) * per_rank]
    nccl_rs = g.reduce_scatter_fp32(x)
    view = symm.view(x.shape, dtype)
    view.copy_(x)
    after = symm.sentinel_after(x.numel() * dtype.itemsize)
    out = ou.guarded_empty((per_rank,), dtype)
    result = multimem_reduce_scatter(out.tensor, view, symm.hdl)
    region = view.clone()

    def check():
        assert result is out.tensor, "multimem_reduce_scatter must return output_tensor"
        ou.check_guards(output=out)
        check_sum(out.tensor, parts, nccl_rs, data, what="reduce-scatter output")
        ou.assert_exact(region, x, what="input region of the buffer (read-only)")
        after.check("buffer after the input region")

    nu.agree(check, "reduce-scatter")


@gpu_test
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_reduce_scatter_then_equal_size_all_gather(nvls_groups, ws, mode):
    """The non-fused TP order on one buffer: the row-parallel GEMM output is written, reduce-
    scattered, and the next layer all-gathers the same per-rank size without a barrier (the
    rule main documents and #8709 keeps for equal sizes: each rank's all-gather writes only the
    segment no peer is still reducing). Token counts and data change every step and one random
    rank is delayed before each step. In graph mode each token count is one CUDA graph (copy in,
    RS, AG, copy out), replayed in a changing order: grids, pointers and barrier slots are baked
    per graph and shared between graphs. Every step is checked after the loop."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hidden, steps = 2048, 10
    sizes = [1, 3, 64, 129]
    shared = g.shared_generator(nu.case_seed("rs-ag", ws, mode))

    def step_ops(s):
        s.view.copy_(s.inp)
        multimem_reduce_scatter(s.rs_out, s.view, symm.hdl)
        multimem_all_gather(s.view, s.rs_out, symm.hdl, barrier_before=False)
        s.gathered.copy_(s.view)

    statics = {}
    for tpr in sizes:
        statics[tpr] = SimpleNamespace(
            inp=torch.zeros(tpr * ws, hidden, dtype=torch.bfloat16, device="cuda"),
            rs_out=torch.zeros(tpr, hidden, dtype=torch.bfloat16, device="cuda"),
            gathered=torch.zeros(tpr * ws, hidden, dtype=torch.bfloat16, device="cuda"),
            view=symm.view((tpr * ws, hidden), torch.bfloat16),
            graph=None,
        )
        step_ops(statics[tpr])  # compile and warm up
        if mode == "graph":
            torch.cuda.synchronize()
            statics[tpr].graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(statics[tpr].graph):
                step_ops(statics[tpr])
    nu.sync_world()

    order = [sizes[int(i)] for i in torch.randint(0, len(sizes), (steps,), generator=shared)]
    plan = []
    for i, tpr in enumerate(order):
        x = make_input(
            (tpr * ws, hidden),
            torch.bfloat16,
            "exact",
            g.rank_generator(nu.case_seed("rs-ag", ws, mode, i)),
        )
        plan.append(SimpleNamespace(tpr=tpr, x=x, nccl_rs=g.reduce_scatter_fp32(x)))
    results = []
    for step in plan:
        s = statics[step.tpr]
        s.inp.copy_(step.x)
        g.maybe_sleep(shared)
        if s.graph is None:
            step_ops(s)
        else:
            s.graph.replay()
        results.append(SimpleNamespace(rs_out=s.rs_out.clone(), gathered=s.gathered.clone()))
    for res in results:
        res.gathered_rs = g.gather(res.rs_out)

    def check():
        for i, (step, res) in enumerate(zip(plan, results)):
            ou.assert_exact(
                res.rs_out,
                step.nccl_rs.to(torch.bfloat16),
                what=f"step {i} (tpr {step.tpr}): reduce-scatter",
            )
            ou.assert_exact(
                res.gathered,
                res.gathered_rs.reshape(res.gathered.shape),
                what=f"step {i} (tpr {step.tpr}): all-gather after it",
            )

    nu.agree(check, f"{mode} RS -> AG steps in the order {order}")


TWO_TERM_SUMS = [
    # (class, a, b): one rank holds a, the next rank b, every other rank 0. All values are exact
    # in bf16 and every sum is exact in fp32. In [1, 2) a bf16 ulp is 2**-7.
    ("exact", 1.0, 2**-7),
    ("quarter-ulp", 1.0, 2**-9),
    ("tie-even-below", 1.0, 2**-8),
    ("tie-odd-below", 1.0 + 2**-7, 2**-8),
    ("three-quarter-ulp", 1.0, 3 * 2**-9),
    ("sticky-above", 1.0, 2**-20),
    ("sticky-below", 1.0, -(2**-20)),
    ("carry-to-tie", 1.5, 1.5 + 2**-7),
]
SMALL_TERM = 3 * 2**-10
"""For ws >= 3: 1.0 on one rank and SMALL_TERM (0.375 ulp of 1.0) on every other. Accumulated in
fp32 the sum is 1 + (ws - 1) * SMALL_TERM, which rounds to 1 + 2**-7 (ws=4) or 1 + 3 * 2**-7
(ws=8); accumulated in bf16 from the 1.0 on, every add rounds back to 1.0."""


def designed_sums(ws):
    """Class names and contributions [ws, n] (fp64, exact in bf16) of the designed sums: every
    class at three binades (2**-10, 1, 2**10), both signs, and with each rank holding the
    leading term."""
    names, columns = [], []
    for name, a, b in TWO_TERM_SUMS:
        for scale in (2**-10, 1.0, 2**10):
            for sign in (1.0, -1.0):
                for slot in range(ws):
                    v = [0.0] * ws
                    v[slot] = sign * scale * a
                    v[(slot + 1) % ws] = sign * scale * b
                    names.append(name)
                    columns.append(v)
    if ws >= 3:
        for scale in (2**-10, 1.0, 2**10):
            for sign in (1.0, -1.0):
                for slot in range(ws):
                    v = [sign * scale * SMALL_TERM] * ws
                    v[slot] = sign * scale
                    names.append("accumulate-in-fp32")
                    columns.append(v)
    while len(columns) % 8:  # whole 16-byte lanes
        names.append("zero")
        columns.append([0.0] * ws)
    return names, torch.tensor(columns, dtype=torch.float64).t().contiguous()


def rounding_rules(exact, contributions):
    """bf16 results of ``exact`` (fp32, exact sums) under the rounding rules a reduction might
    use, and of bf16 accumulation in rank order (each add rounded to nearest even)."""
    bits = exact.view(torch.int32)
    low = bits & 0xFFFF
    trunc = bits & -0x10000
    away = trunc + 0x10000
    inexact = low != 0
    negative = bits < 0
    odd = ((trunc >> 16) & 1) == 1

    def to_bf16(b):
        return (b >> 16).to(torch.int16).view(torch.bfloat16)

    rules = {
        "nearest-even": exact.to(torch.bfloat16),
        "toward-zero": to_bf16(trunc),
        "away-from-zero": to_bf16(torch.where(inexact, away, trunc)),
        "nearest-ties-away": to_bf16(torch.where(low >= 0x8000, away, trunc)),
        "to-odd": to_bf16(torch.where(inexact & ~odd, away, trunc)),
        "toward-+inf": to_bf16(torch.where(inexact & ~negative, away, trunc)),
        "toward--inf": to_bf16(torch.where(inexact & negative, away, trunc)),
    }
    for label, order in (
        ("bf16-accumulate-rank-order", contributions),
        ("bf16-accumulate-reverse", contributions.flip(0)),
    ):
        acc = order[0].to(torch.bfloat16)
        for term in order[1:]:
            acc = (acc.float() + term.float()).to(torch.bfloat16)
        rules[label] = acc
    return rules


def rounding_report(got, names, rules):
    """Per class: how many results each rule explains (bit for bit)."""
    lines = ["per class, results explained by each rule (matches / columns):"]
    for name in dict.fromkeys(names):
        idx = torch.tensor([i for i, n in enumerate(names) if n == name], device=got.device)
        g = got[idx].view(torch.int16)
        cells = [
            f"{rule} {int((g == r[idx].view(torch.int16)).sum())}" for rule, r in rules.items()
        ]
        lines.append(f"  {name} ({idx.numel()}): " + ", ".join(cells))
    return "\n".join(lines)


@gpu_test
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_bf16_in_switch_sum_rounding(nvls_groups, ws):
    """The bf16 in-switch sum accumulates in fp32 and rounds once to nearest even (the
    kernel's `.acc::f32`; what the unfused path and NCCL's fp32 accumulation give): checked bit
    for bit on designed sums whose exact value is known (see TWO_TERM_SUMS and SMALL_TERM).
    Every rank's shard holds every case, so each rank reads every case as the reducing GPU.

    Passes on H100. Fails on GB200 (finding O5-N4: the dyadic sweeps of the first GPU run saw
    1-2 ulp differences there, identical in both ws=2 groups, so deterministic); the failure
    message lists, per class, which rounding rule the results follow."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    names, contributions = designed_sums(ws)
    exact64 = contributions.sum(0)
    exact = exact64.to(torch.float32)
    assert torch.equal(exact.double(), exact64), "designed sums must be exact in fp32"
    assert torch.equal(contributions.to(torch.bfloat16).double(), contributions)
    n = contributions.shape[1]
    x = contributions[g.rank].repeat(ws).to(torch.bfloat16).cuda()
    view = symm.view(x.shape, torch.bfloat16)
    view.copy_(x)
    out = ou.guarded_empty((n,), torch.bfloat16)
    multimem_reduce_scatter(out.tensor, view, symm.hdl)
    got = out.tensor.clone()

    def check():
        ou.check_guards(output=out)
        rules = rounding_rules(exact.cuda(), contributions.cuda())
        try:
            ou.assert_exact(got, rules["nearest-even"], what="bf16 in-switch sum")
        except AssertionError as error:
            raise AssertionError(f"{error}\n{rounding_report(got, names, rules)}") from None

    nu.agree(check, "bf16 in-switch sum rounding")


@pytest.mark.parametrize(
    "case", ["valid", "tokens-not-a-multiple-of-ws", "dtype-mismatch", "unsupported-dtype"]
)
def test_launcher_checks_inputs(monkeypatch, case):
    """The launcher refuses inputs outside its contract before launching (CPU: the kernel is
    replaced by a recorder). Passes on main."""
    pytest.importorskip("triton")
    from megatron.core.inference.communication.torch_symm_triton import collectives, utils

    launches = []

    class Recorder:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(collectives, "_multimem_reduce_scatter_kernel", Recorder())
    monkeypatch.setattr(utils, "is_device_nvls_capable", lambda device: True)
    handle = SimpleNamespace(world_size=4, rank=2, multicast_ptr=0, signal_pad_ptrs_dev=0)
    inp, out = torch.zeros(4 * 64, dtype=torch.bfloat16), torch.zeros(64, dtype=torch.bfloat16)
    if case == "tokens-not-a-multiple-of-ws":
        inp = torch.zeros(4 * 64 + 8, dtype=torch.bfloat16)
    elif case == "dtype-mismatch":
        out = torch.zeros(64, dtype=torch.float32)
    elif case == "unsupported-dtype":
        inp, out = inp.half(), out.half()
    if case == "valid":
        collectives.multimem_reduce_scatter(out, inp, handle)
        assert len(launches) == 1
        return
    with pytest.raises(AssertionError):
        collectives.multimem_reduce_scatter(out, inp, handle)
    assert not launches
