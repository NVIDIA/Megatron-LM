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

`test_dispatch_then_combine_steps` runs the dispatcher's order, step after step: the metadata
update (`fused_metadata_update`, the first MoE layer's; it computes the offsets and the maximum
count on device behind a barrier) -> all-gather-v -> copy out -> poison the gathered buffers ->
experts write the combine buffer -> reduce-scatter-v (multimem, or the batch-invariant ordered
combine). The token counts change every step, a random rank is delayed before every step, and it
runs eagerly and as one CUDA graph per local token count, replayed in a changing order. Stale
rows from the previous step must never show up. Within a step the combine's entry barrier keeps
the next all-gather-v from overwriting rows a peer has not consumed; between steps the metadata
barrier does, and it is the only barrier of a step in which no rank has tokens (the first GPU
run, without it, let a peer's next-step rows land before a delayed rank had copied out).

Since the medium-risk batch: production shapes at the whole node (`test_production_shapes`:
the dispatch of DeepSeek-V3, Nemotron 3 Nano and Nemotron 3 Super's latent MoE, 2048 tokens per
step, separate buffers); `test_sanitizer_small_case`, the compute-sanitizer case; and the int32
limit of the lane offsets (`test_int32_row_offsets`, `large`: 16 GiB of bf16 rows; suspected
finding O5-M2 on main).
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
from megatron.core.inference.moe.metadata import fused_metadata_update
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
    return [
        pytest.param(
            c["ws"],
            c["hidden"],
            c["side"],
            c["per_rank_max"],
            c["counts"],
            c["layout"],
            id=f"ws{c['ws']}-H{c['hidden']}-{c['side']}-prm{c['per_rank_max']}"
            f"-{c['counts']}-{c['layout']}",
        )
        for c in cases
    ]


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
    specs = [(torch.bfloat16, hidden), *SIDE_TENSORS[side]]
    run_all_gatherv_3tensor(g, specs, per_rank_max, counts, layout, seed)


def run_all_gatherv_3tensor(g, specs, per_rank_max, counts, layout, seed):
    """Gather random bit patterns of the three tensors ``specs`` ((dtype, width) each) with the
    per-rank ``counts`` and check every rank's outputs, the rows past the total, the bytes
    around each region, the inputs and the device scalars."""
    offsets = nu.offsets_from_counts(counts)
    total, mine, global_max = sum(counts), counts[g.rank], per_rank_max * g.ws
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
    meta = g.buffer("meta")
    step_metadata = torch.zeros(3, dtype=torch.int32, device="cuda")
    rank_token_offset, ep_max_tokens = step_metadata[1:2], step_metadata[2:3]
    static_in = [torch.zeros(per_rank_max, w, dtype=d, device="cuda") for d, w in specs]
    static_experts = torch.zeros(global_max, hidden, dtype=torch.float32, device="cuda")
    copies = [torch.empty(global_max, w, dtype=d, device="cuda") for d, w in specs]
    # Views of one allocation, so a 0-token output still has a valid pointer.
    out_storage = torch.empty(per_rank_max, hidden, dtype=torch.float32, device="cuda")
    outs = {c: out_storage[:c] for c in choices}

    def body(count):
        # The first MoE layer's metadata update: [total, this rank's offset, max count] from
        # every rank's count, behind a barrier. In the dispatcher this barrier is what orders
        # one step's reads of the gathered rows before the next step's all-gather-v (a step in
        # which no rank has tokens runs no other barrier).
        fused_metadata_update(count, meta.view((ws,), torch.int32), meta.hdl, step_metadata)
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

    for _, _, view in agv:
        ou.poison_(view)
    g.sync()  # the all-gather-v has no entry barrier: poison before any peer's stores
    graphs = {}
    for count in choices:  # compile and warm up with every rank at `count` tokens
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
            (global_max, hidden), dtype=torch.float32, device="cuda", generator=gen, **nu.EXACT
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


@gpu_test
@pytest.mark.parametrize("counts_pattern", ["full", "uneven"])
@pytest.mark.parametrize("model", list(nu.PRODUCTION_MOE))
def test_production_shapes(nvls_groups, model, counts_pattern):
    """The sweep's checks at production shapes and group size (EP 8 on H100, EP 4 on GB200):
    the dispatch of DeepSeek-V3 (H 7168, top-k 8: every tensor on 128-bit lanes), Nemotron 3
    Nano (H 2688, top-k 6: fp32 probabilities on 64-bit lanes) and Nemotron 3 Super's latent
    MoE (1024, top-k 22: 64-bit probabilities, 11 lanes of int64 routing), in separate buffers
    as the dispatcher allocates them, with ``nu.PRODUCTION_TOKENS`` tokens per step
    (per_rank_max 256 at EP 8, 512 at EP 4) and every rank full or uneven counts."""
    ws = nu.PRODUCTION_GROUP_SIZE
    g = nvls_groups.get(ws)
    hidden, topk = nu.PRODUCTION_MOE[model]
    per_rank_max = nu.PRODUCTION_TOKENS // ws
    seed = nu.case_seed("agv3-production", model, counts_pattern)
    counts = nu.production_counts(counts_pattern, ws, per_rank_max, g.shared_generator(seed))
    specs = [(torch.bfloat16, hidden), (torch.int64, topk), (torch.float32, topk)]
    run_all_gatherv_3tensor(g, specs, per_rank_max, counts, "separate", seed)


@gpu_test
@pytest.mark.sanitizer
def test_sanitizer_small_case(nvls_groups):
    """The compute-sanitizer (memcheck) case: ws 2, bf16 H 4100 (64-bit lanes: a full
    1024-lane window, then one lane), top-k 22 (128-bit int64 and 64-bit fp32 rows of 11
    lanes), per_rank_max 129 (the 128-block grid and one grid-stride step), uneven counts (one
    rank empty), the three regions packed in one buffer, with the sweep's checks. A dedicated
    function, so the runner selects it by name at any world size."""
    g = nvls_groups.get(2)
    seed = nu.case_seed("agv3-sanitizer")
    counts = nu.token_counts("uneven", 2, 129, g.shared_generator(seed))
    specs = [(torch.bfloat16, 4100), *SIDE_TENSORS["topk22"]]
    run_all_gatherv_3tensor(g, specs, 129, counts, "packed", seed)


INT32_HIDDEN = 7168
"""Hidden width of the int32-limit cases (DeepSeek-V3): 896 16-byte lanes of bf16."""


def _int32_params():
    lanes = INT32_HIDDEN * torch.bfloat16.itemsize // 16
    labels = ["fits", "crosses", "row-starts-past"]
    rows = ou.int32_overflow_rows(lanes, limit=2**30)
    return [
        pytest.param(n, id=f"{n}[int32-{label}]", marks=pytest.mark.large)
        for n, label in zip(rows, labels)
    ]


def _routing_rows(first, count, topk):
    """Closed-form int64 routing rows: global row r, slot k holds r * topk + k."""
    return torch.arange(
        first * topk, (first + count) * topk, dtype=torch.int64, device="cuda"
    ).view(count, topk)


@gpu_test
@pytest.mark.parametrize("rows", _int32_params())
def test_int32_row_offsets(nvls_groups, rows):
    """Row offsets at the int32 limit (`large`): the gathered hidden states are ``rows`` bf16
    rows of H 7168, and the last rank's rows reach the row whose 16-byte-lane offset, scaled
    by 2 into 8-byte units (variable_collectives.py:438-443), crosses 2**31: 16 GiB of rows.
    Routing (top-k 8 int64) and probabilities (fp32) are packed after them in the same buffer,
    at byte offsets past 2**31 (64-bit arguments). A dispatcher would reach this with EP 64 at
    32768 max tokens and H 7168 (at the default 16384 max tokens: EP 64 with H above 8192).
    Contract: every rank's rows land at their offsets, bit for bit, as below the limit. ws 2,
    counts split in half, data in closed form (``nu.closed_form_rows``), exact. The `fits` id
    is the control. Suspected on main (finding O5-M2): the lane offsets are int32, so the
    `crosses` and `row-starts-past` ids store rows 32 GiB before the multicast buffer: lost
    rows or an illegal-address fault. A fault on one rank leaves its peer waiting in the next
    collective until the NCCL timeout: run each id in its own job."""
    g = nvls_groups.get(2)
    hidden, topk = INT32_HIDDEN, 8
    counts = [rows // 2, rows - rows // 2]
    offsets = nu.offsets_from_counts(counts)
    first, mine = offsets[g.rank], counts[g.rank]
    allocated = "large" in g.buffers
    need = mine * hidden * 2 + (0 if allocated else nu.LARGE_BUFFER_MB << 20) + (2 << 30)
    nu.require_free_memory_everywhere(need)
    symm = g.large_buffer()
    specs = [(torch.bfloat16, hidden), (torch.int64, topk), (torch.float32, topk)]
    views, offset = [], 0
    for dtype, width in specs:
        view = symm.view((rows, width), dtype, offset)
        ou.poison_(view)
        views.append((offset, view))
        offset = _align(offset + view.numel() * view.element_size())
    after = symm.sentinel_after(offset)
    hidden_in = torch.empty((mine, hidden), dtype=torch.bfloat16, device="cuda")
    for start, count in nu.row_chunks(0, mine, hidden):
        hidden_in[start : start + count] = nu.closed_form_rows(
            g.rank, first + start, count, hidden, torch.bfloat16
        )
    inputs = [
        hidden_in,
        _routing_rows(first, mine, topk),
        nu.closed_form_rows(g.rank, first, mine, topk, torch.float32),
    ]
    g.sync()  # no entry barrier: every rank's poison before any peer's stores
    multimem_all_gatherv_3tensor(
        *(v for _, v in views),
        *inputs,
        symm.hdl,
        symm.hdl,
        symm.hdl,
        rank_token_offset=torch.tensor([first], dtype=torch.int32, device="cuda"),
        ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
        per_rank_max_tokens=max(counts),
        output_byte_offset_0=views[0][0],
        output_byte_offset_1=views[1][0],
        output_byte_offset_2=views[2][0],
    )

    def check():
        (_, out_h), (_, out_r), (_, out_p) = views
        for q, (lo, count) in enumerate(zip(offsets, counts)):
            for start, n in nu.row_chunks(lo, lo + count, hidden):
                ou.assert_exact(
                    out_h[start : start + n],
                    nu.closed_form_rows(q, start, n, hidden, torch.bfloat16),
                    what=f"hidden rows {start}-{start + n - 1} (rank {q})",
                )
            ou.assert_exact(
                out_r[lo : lo + count], _routing_rows(lo, count, topk), what=f"routing (rank {q})"
            )
            ou.assert_exact(
                out_p[lo : lo + count],
                nu.closed_form_rows(q, lo, count, topk, torch.float32),
                what=f"probabilities (rank {q})",
            )
        after.check("large buffer after the three regions")

    nu.agree(check, f"all-gather-v of {rows} rows (counts {counts})")


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
