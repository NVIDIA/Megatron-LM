# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_multimem_reduce_scatter_kernel`
(megatron/core/inference/communication/torch_symm_triton/collectives.py:175), launched by
`multimem_reduce_scatter` (collectives.py:356): the NVLS reduce-scatter of the inference-optimized
row-parallel linears (non-fused TP) and of the vocab-parallel embedding.

Contract (from the launcher's docstring and its callers, inference_layers.py:461-526, 616-656):
every rank q of a group of ws ranks has written its input P_q (N elements, bf16 or fp32) at
offset 0 of its symmetric buffer. Rank r's output is out_r[i] = sum_q P_q[r * N/ws + i] for
i < N/ws: bf16 inputs accumulate in fp32 and round once to bf16; fp32 inputs sum in fp32 (the
order across ranks is the switch's). Nothing else is written: the buffers are only read. The
kernel barriers on entry (every rank's input is written before any rank reads) and not on exit:
when it returns on one rank, peers may still be reading their segments of its buffer, so the next
write to the buffer must be an all-gather of the same per-rank size (which writes only this
rank's segment) or come after a collective that synchronizes the group. Preconditions (checked):
dtype bf16 or fp32 on both sides, output bytes % 16 == 0, input numel == ws * output numel.

Tolerance classes: exact on dyadic inputs (every sum is exact in fp32 in any order, so the
reference is NCCL's fp32 reduce-scatter rounded once to the dtype); on N(0, 1) inputs bounded
against the fp64 sum of the NCCL-gathered inputs, calibrated by NCCL's own fp32 reduce-scatter
rounded to the dtype (``ou.assert_close_to_reference``).

Structure-derived inputs: 16-byte chunks per thread (8 bf16 or 4 fp32: the two compiled
variants, REDUCE_F32 False/True), blocks of 1024 threads, grids of at most 128 blocks with a
grid-stride loop. Per-rank output chunks: 1, 1023/1024/1025, 131071/131072/131073 and 262145;
ws in {2, 4, 8}; combined pairwise with dtype and data. Memory: the output is NaN-poisoned with
guard bands, the input region of the buffer must be unchanged, and a 64 KiB sentinel follows it.
The production order row-parallel RS -> next layer's equal-size all-gather without a barrier is
run back to back with changing token counts and data, eagerly and as CUDA graphs (one per token
count, replayed in a changing order), with a random rank delayed before each step. The fp32 sum's
rounding against fp64 is checked strictly for the variable-count reduce-scatter, where the
dispatcher documents it (test_multimem_reduce_scatter_v.py).
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
    if data == "dyadic":
        return ou.dyadic_tensor(shape, dtype=dtype, device="cuda", generator=generator, **nu.DYADIC)
    return torch.randn(shape, device="cuda", generator=generator).to(dtype)


def check_sum(out, parts, nccl_rs_fp32, data, what):
    """``out`` against the sum over ranks of ``parts`` ([ws, ...]), as described above."""
    if data == "dyadic":
        ou.assert_exact(out, nccl_rs_fp32.to(out.dtype), what=f"{what} (exact on dyadic data)")
        return
    ou.assert_close_to_reference(
        out, ou.fp64_reference(sum_reference, parts), nccl_rs_fp32.to(out.dtype), what=what
    )


def _sweep_params():
    cases = ou.pairwise_cases(
        {
            "chunks": CHUNKS_PER_RANK,
            "dtype": list(DTYPES),
            "data": ["dyadic", "randn"],
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
            "dyadic",
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
