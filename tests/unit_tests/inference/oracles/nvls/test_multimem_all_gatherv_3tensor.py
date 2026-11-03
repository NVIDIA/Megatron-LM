# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_multimem_all_gatherv_3tensor_kernel`
(megatron/core/inference/communication/torch_symm_triton/variable_collectives.py:352), launched by
`multimem_all_gatherv_3tensor` (:621): the dispatch of the NVLS MoE dispatcher
(`NVLSAllGatherVDispatcher.token_dispatch`, token_dispatcher_inference.py:552-626), which gathers
each rank's hidden states, routing map and probabilities in one launch.

Contract (from the launcher's docstring and the dispatcher): rank q of a group of ws ranks owns
c_q tokens (counts differ and may be 0); its rows start at off_q = c_0 + ... + c_{q-1} of the
global layout, whose buffers hold global_max = per_rank_max * ws rows. For each of the three
tensors k, rank r passes in_k [c_r, w_k] (any dtype; row bytes a multiple of 8) and every rank's
out_k (a symmetric buffer view [global_max, w_k] at output_byte_offset_k, each tensor in its own
buffer or packed in one). rank_token_offset holds off_r, ep_max_tokens max_q c_q (device int32
scalars, the latter the same on every rank). After the call, on every rank,
out_k[off_q : off_q + c_q] == in_k of rank q, bit for bit, for every q, and nothing else is
written: rows [total, global_max) keep what they held, as do the bytes around each region and
the inputs. The kernel has an exit barrier and no entry barrier: peers must have finished
reading the previous contents (in the dispatcher, the preceding combine's entry barrier or the
metadata kernel orders this). Rows of 8 but not 16 bytes take the 64-bit path; rows that are not
a multiple of 8 bytes are refused (odd top-k fp32 probabilities).

Tolerance class: exact (bitwise; random bit patterns, so NaN payloads, -0.0 and -1 routing
entries must survive), against the rows gathered with NCCL.

Structure-derived inputs: one block per token (grid min(per_rank_max, 128); grid-stride beyond
128 tokens per rank: per_rank_max in {1, 127, 128, 129, 300}), blocks of max over the three
tensors of min(next_pow2(lanes), 1024) lanes; per-tensor 128- or 64-bit lanes. Hidden bf16 H in
{4 (64-bit, one lane), 8, 128, 2048, 4100 (64-bit, 1025 lanes: two windows), 7168, 8184, 8192,
8200, 16384}; routing/probs widths: top-k 8 (both 128-bit), 6 and 22 (fp32 rows 64-bit; 6 is
DeepSeek-V2-Lite's top-k, 22 is named in the kernel), and an int64 [3] / fp32 [2] pair (both
64-bit). Counts: uneven (one rank at per_rank_max, the last at 0), all full, one rank only, all
0 (ep_max_tokens = 0: every block exits). Separate buffers (production) or one buffer with the
three regions at offsets; ws in {2, 4, 8}; combined pairwise. Memory: every output region is
poisoned before the call and rows past the total must stay poisoned; 64 KiB sentinels around
each region; guarded inputs.

`test_dispatch_then_combine_steps` runs the dispatcher's per-layer order, step after step:
all-gather-v -> copy out -> poison the gathered buffers -> experts write the combine buffer ->
reduce-scatter-v (multimem, or the batch-invariant ordered combine), with the token counts
changing every step, a random rank delayed before every step, eagerly and as one CUDA graph per
local token count replayed with the counts (device scalars) changing between replays. Stale rows
from the previous step must never show up, and nothing may race (the combine's entry barrier is
what keeps the next all-gather-v from overwriting rows a peer has not consumed).
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton import variable_collectives
from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.moe.batch_invariant import ordered_reduce_scatter_v
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

HIDDEN = [4, 8, 128, 2048, 4100, 7168, 8184, 8192, 8200, 16384]
SIDE_TENSORS = {
    "topk8": ((torch.int64, 8), (torch.float32, 8)),
    "topk6": ((torch.int64, 6), (torch.float32, 6)),
    "topk22": ((torch.int64, 22), (torch.float32, 22)),
    "narrow": ((torch.int64, 3), (torch.float32, 2)),
}


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def _align(n, a=16):
    return -(-n // a) * a


def output_views(g, specs, global_max, layout):
    """[global_max, w] views for the three outputs and their (buffer, offset) placement."""
    views = []
    if layout == "separate":
        for name, (dtype, width) in zip(("main", "aux1", "aux2"), specs):
            views.append((g.buffer(name), 0, g.buffer(name).view((global_max, width), dtype)))
        return views
    offset = 0
    for dtype, width in specs:
        views.append((g.buffer(), offset, g.buffer().view((global_max, width), dtype, offset)))
        offset = _align(offset + global_max * width * dtype.itemsize + nu.SENTINEL_BYTES)
    return views


def _sweep_params():
    cases = ou.pairwise_cases(
        {
            "hidden": HIDDEN,
            "side": list(SIDE_TENSORS),
            "per_rank_max": nu.PER_RANK_MAX,
            "counts": nu.COUNTS,
            "layout": ["separate", "packed"],
            "ws": nu.GROUP_SIZES,
        }
    )
    params = []
    sanitizer_picked = False
    for c in cases:
        # One small case for the compute-sanitizer pass: the first at ws=2 with H=128.
        sanitizer = not sanitizer_picked and (c["hidden"], c["ws"]) == (128, 2)
        sanitizer_picked |= sanitizer
        params.append(
            pytest.param(
                c["ws"],
                c["hidden"],
                c["side"],
                c["per_rank_max"],
                c["counts"],
                c["layout"],
                id=f"ws{c['ws']}-H{c['hidden']}-{c['side']}-prm{c['per_rank_max']}"
                f"-{c['counts']}-{c['layout']}",
                marks=[pytest.mark.sanitizer] if sanitizer else [],
            )
        )
    return params


@gpu_test
@pytest.mark.parametrize("ws,hidden,side,per_rank_max,counts_pattern,layout", _sweep_params())
def test_all_gatherv_3tensor_matches_nccl(
    nvls_groups, ws, hidden, side, per_rank_max, counts_pattern, layout
):
    """Every rank's three outputs hold every rank's rows at its offset, bit for bit; rows past
    the total, the bytes around each region and the inputs are untouched."""
    g = nvls_groups.get(ws)
    seed = nu.case_seed("agv3", ws, hidden, side, per_rank_max, counts_pattern, layout)
    counts = nu.token_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    offsets = nu.offsets_from_counts(counts)
    total, mine, global_max = sum(counts), counts[g.rank], per_rank_max * ws
    specs = [(torch.bfloat16, hidden), *SIDE_TENSORS[side]]
    gen = g.rank_generator(seed)
    padded = [nu.random_bits((per_rank_max, width), dtype, gen) for dtype, width in specs]
    expected = [nu.concat_rows(g.gather(p), counts) for p in padded]
    inputs = [ou.guarded_like(p[:mine]) for p in padded]
    views = output_views(g, specs, global_max, layout)
    sentinels = []
    for buf, offset, view in views:
        ou.poison_(view)
        sentinels.append(buf.sentinel_after(offset + view.numel() * view.element_size()))
    rank_token_offset = torch.tensor([offsets[g.rank]], dtype=torch.int32, device="cuda")
    ep_max_tokens = torch.tensor([max(counts)], dtype=torch.int32, device="cuda")
    g.sync()  # no peer's stores may land before this rank's poison and sentinels
    result = multimem_all_gatherv_3tensor(
        *(v for _, _, v in views),
        *(i.tensor for i in inputs),
        *(buf.hdl for buf, _, _ in views),
        rank_token_offset=rank_token_offset,
        ep_max_tokens=ep_max_tokens,
        per_rank_max_tokens=per_rank_max,
        output_byte_offset_0=views[0][1],
        output_byte_offset_1=views[1][1],
        output_byte_offset_2=views[2][1],
    )
    gathered = [v.clone() for _, _, v in views]

    def check():
        assert all(r is v for r, (_, _, v) in zip(result, views)), "must return the outputs"
        for k, (got, exp) in enumerate(zip(gathered, expected)):
            ou.assert_exact(got[:total], exp, what=f"tensor {k} ({got.dtype}): gathered rows")
            ou.assert_poisoned(got[total:], what=f"tensor {k}: rows past the {total} gathered")
        for k, (buf, p) in enumerate(zip(inputs, padded)):
            ou.check_guards(**{f"input_{k}": buf})
            ou.assert_exact(buf.tensor, p[:mine], what=f"input {k} (read-only)")
        for s in sentinels:
            s.check("buffer after an output region")
        ou.assert_exact(
            rank_token_offset.cpu(),
            torch.tensor([offsets[g.rank]]),
            what="rank_token_offset (read-only)",
        )
        ou.assert_exact(
            ep_max_tokens.cpu(), torch.tensor([max(counts)]), what="ep_max_tokens (read-only)"
        )

    nu.agree(check, f"all-gather-v, counts {counts}")


@gpu_test
@pytest.mark.parametrize("combine", ["multimem", "ordered"])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_dispatch_then_combine_steps(nvls_groups, ws, mode, combine):
    """The dispatcher's order, step after step, with changing per-rank token counts (see the
    module docstring). Each step's gathered rows and combined sums are checked after the loop
    against references built with NCCL before it; rows past each step's total must still hold
    the poison written after the previous step."""
    g = nvls_groups.get(ws)
    hidden, topk, per_rank_max, steps = 2048, 6, 129, 10
    choices = [0, 1, 5, 64, per_rank_max]
    global_max = per_rank_max * ws
    specs = [(torch.bfloat16, hidden), (torch.int64, topk), (torch.float32, topk)]
    agv = output_views(g, specs, global_max, "separate")
    rsv_offset = _align(global_max * hidden * 2 + nu.SENTINEL_BYTES)
    rsv = g.buffer().view((global_max, hidden), torch.float32, rsv_offset)
    combine_fn = multimem_reduce_scatter_v if combine == "multimem" else ordered_reduce_scatter_v
    rank_token_offset = torch.zeros(1, dtype=torch.int32, device="cuda")
    ep_max_tokens = torch.zeros(1, dtype=torch.int32, device="cuda")
    static_in = [torch.zeros(per_rank_max, w, dtype=d, device="cuda") for d, w in specs]
    static_experts = torch.zeros(global_max, hidden, dtype=torch.float32, device="cuda")
    copies = [torch.empty(global_max, w, dtype=d, device="cuda") for d, w in specs]
    # Views of one allocation, so a 0-token output still has a valid pointer.
    out_storage = torch.empty(per_rank_max, hidden, dtype=torch.float32, device="cuda")
    outs = {c: out_storage[:c] for c in choices}

    def body(count):
        multimem_all_gatherv_3tensor(
            *(v for _, _, v in agv),
            *(t[:count] for t in static_in),
            *(buf.hdl for buf, _, _ in agv),
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            per_rank_max_tokens=per_rank_max,
        )
        for copy, (_, _, view) in zip(copies, agv):
            copy.copy_(view)
            ou.poison_(view)
        rsv.copy_(static_experts)
        combine_fn(
            outs[count],
            rsv,
            g.buffer().hdl,
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            per_rank_max_tokens=per_rank_max,
            input_byte_offset=rsv_offset,
        )

    def set_counts(counts):
        rank_token_offset.fill_(nu.offsets_from_counts(counts)[g.rank])
        ep_max_tokens.fill_(max(counts))

    for _, _, view in agv:
        ou.poison_(view)
    g.sync()  # the all-gather-v has no entry barrier: poison before any peer's stores
    graphs = {}
    for count in choices:  # compile and warm up with every rank at `count` tokens
        set_counts([count] * ws)
        body(count)
        if mode == "graph":
            torch.cuda.synchronize()
            graphs[count] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[count]):
                body(count)
    nu.sync_world()

    shared = g.shared_generator(nu.case_seed("dispatch-combine", ws, mode, combine))
    plan = []
    for i in range(steps):
        counts = [choices[int(j)] for j in torch.randint(0, len(choices), (ws,), generator=shared)]
        gen = g.rank_generator(nu.case_seed("dispatch-combine", ws, mode, combine, i))
        inputs = [nu.random_bits((per_rank_max, w), d, gen) for d, w in specs]
        experts = ou.dyadic_tensor(
            (global_max, hidden), dtype=torch.float32, device="cuda", generator=gen, **nu.DYADIC
        )
        rows = slice(
            nu.offsets_from_counts(counts)[g.rank],
            nu.offsets_from_counts(counts)[g.rank] + counts[g.rank],
        )
        plan.append(
            SimpleNamespace(
                counts=counts,
                inputs=inputs,
                experts=experts,
                gathered=[nu.concat_rows(g.gather(t), counts) for t in inputs],
                combined=g.all_reduce_fp32(experts)[rows],
            )
        )
    results = []
    for step in plan:
        count = step.counts[g.rank]
        for static, value in zip(static_in, step.inputs):
            static.copy_(value)
        static_experts.copy_(step.experts)
        set_counts(step.counts)
        g.maybe_sleep(shared)
        if mode == "graph":
            graphs[count].replay()
        else:
            body(count)
        results.append(([c.clone() for c in copies], outs[count].clone()))

    def check():
        for i, (step, (gathered, combined)) in enumerate(zip(plan, results)):
            total = sum(step.counts)
            for k, (got, exp) in enumerate(zip(gathered, step.gathered)):
                ou.assert_exact(got[:total], exp, what=f"step {i} {step.counts}: tensor {k}")
                ou.assert_poisoned(got[total:], what=f"step {i}: tensor {k} past row {total}")
            ou.assert_exact(combined, step.combined, what=f"step {i} {step.counts}: combine")

    nu.agree(check, f"{mode} dispatch/{combine} combine steps")


@pytest.mark.parametrize("topk", [2, 3])
def test_launcher_refuses_rows_not_a_multiple_of_8_bytes(monkeypatch, topk):
    """fp32 probabilities with an odd top-k (rows of 4 * topk bytes) cannot be gathered with
    64- or 128-bit stores: the launcher refuses them before launching; even top-k launches.
    Passes on main (CPU: the kernel is replaced by a recorder)."""
    pytest.importorskip("triton")
    launches = []

    class Recorder:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(variable_collectives, "_multimem_all_gatherv_3tensor_kernel", Recorder())
    monkeypatch.setattr(variable_collectives, "is_device_nvls_capable", lambda device: True)
    handle = SimpleNamespace(world_size=4, rank=1, multicast_ptr=0, signal_pad_ptrs_dev=0)
    meta = SimpleNamespace(numel=lambda: 1, dtype=torch.int32, is_cuda=True)
    hidden = torch.zeros(3, 128, dtype=torch.bfloat16)
    routing = torch.zeros(3, topk, dtype=torch.int64)
    probs = torch.zeros(3, topk, dtype=torch.float32)
    outputs = [torch.zeros(12, t.shape[1], dtype=t.dtype) for t in (hidden, routing, probs)]

    def launch():
        multimem_all_gatherv_3tensor(
            *outputs,
            hidden,
            routing,
            probs,
            handle,
            handle,
            handle,
            rank_token_offset=meta,
            ep_max_tokens=meta,
            per_rank_max_tokens=3,
        )

    if topk % 2 == 0:
        launch()
        assert len(launches) == 1
        return
    with pytest.raises(AssertionError):
        launch()
    assert not launches
