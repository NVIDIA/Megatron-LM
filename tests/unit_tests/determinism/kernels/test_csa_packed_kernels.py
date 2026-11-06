# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay the packed layout, indexer metadata and CP collective operations."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import csa2
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    csa2_candidates,
    csa2_indexer,
    csa_indexer_loss_kernels,
    packed_layout,
    packed_sparse_attention,
    thd_utils,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    run_once,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_cp_compressor_layout_replays():
    seeded()
    cu = torch.tensor([0, 37, 37, 215, 512], device="cuda", dtype=torch.int32)
    hidden = torch.randn(256, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundary = torch.randn(16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, halo):
        return cp_utils.prepare_cp_compressor_input(x, halo, cu, 256, 2, 4)

    assert_replays_bit_exact(run, (hidden, boundary), contention=True, what="CSA packed layout")
    layout = run(hidden, boundary)

    def indices(comp_cu, row_map):
        return packed_layout.build_attention_indices(
            cu,
            256,
            256,
            16,
            16,
            4,
            64,
            cu_seqlens_compressed=comp_cu,
            seq_to_rank_row=row_map,
            compressed_rows=layout[1].shape[0] * 2,
            output_alignment=128,
        )

    assert_replays_bit_exact(indices, (layout[-2], layout[-1]), backward=False, contention=True)


class _SingleRankGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


@pytest.mark.parametrize("device_checks", [False, True], ids=["checks_off", "checks_on"])
def test_thd_metadata_device_checks_only_validate(monkeypatch, device_checks):
    """MCORE_CSA2_DEVICE_CHECKS only adds device asserts: the layouts are bit-identical with
    the checks on or off, off launches no check, and CPU metadata is validated either way."""
    seeded()
    cu = torch.tensor([0, 37, 37, 215, 300], device="cuda", dtype=torch.int32)
    padded = torch.tensor([0, 40, 40, 256, 512], device="cuda", dtype=torch.int32)
    config = SimpleNamespace(dsa_kernel_backend="cudnn", deterministic_mode=False)

    def packed():
        # Separate Q and KV tensors, so aliasing does not skip the Q/KV equality checks.
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu.clone(),
            cu_seqlens_kv=cu.clone(),
            cu_seqlens_q_padded=padded.clone(),
            cu_seqlens_kv_padded=padded.clone(),
            max_seqlen_q=256,
            max_seqlen_kv=256,
        )

    def build():
        layout = thd_utils.build_csa2_thd_layout(packed(), 512)
        layout.validate_compatible(packed(), 512)
        cp_utils.validate_packed_inputs(packed(), config, _SingleRankGroup())
        return layout, layout.for_compression(2)

    monkeypatch.setattr(thd_utils, "DEVICE_CHECKS", True)
    reference = build()
    torch.cuda.synchronize()

    launched = []
    assert_async = torch._assert_async

    def counting_assert_async(*args, **kwargs):
        launched.append(args)
        return assert_async(*args, **kwargs)

    monkeypatch.setattr(torch, "_assert_async", counting_assert_async)
    monkeypatch.setattr(thd_utils, "DEVICE_CHECKS", device_checks)
    actual = build()
    torch.cuda.synchronize()

    assert bool(launched) == device_checks
    for expected, value in zip(reference, actual):
        for field in dataclasses.fields(expected):
            got, want = getattr(value, field.name), getattr(expected, field.name)
            if isinstance(want, torch.Tensor):
                assert bytes_equal(got, want), field.name
            else:
                assert got == want, field.name

    bad = torch.tensor([1, 37, 37, 215, 300], dtype=torch.int32)
    with pytest.raises(ValueError, match="must start at zero"):
        thd_utils.build_csa2_thd_layout(
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=bad,
                cu_seqlens_kv=bad.clone(),
                max_seqlen_q=256,
                max_seqlen_kv=256,
            ),
            512,
        )


def _padding_values(case):
    """BF16 rows [T, 4, 64] and their padding flags. Padding rows hold undefined kernel
    output: NaN, infinities and signed zeros."""
    rows = {"mixed": 48, "all_padding": 6, "no_padding": 9, "empty": 0}[case]
    invalid = torch.zeros(rows, dtype=torch.bool, device="cuda")
    if case == "mixed":
        # Row 0 is padding, so unused list entries name a later valid row.
        invalid[[0, 1, 7, 8, 9, 30, 47]] = True
    elif case == "all_padding":
        invalid.fill_(True)
    values = torch.randn(rows, 4, 64, device="cuda").bfloat16()
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 3.0], device="cuda")
    fill = special[torch.arange(4 * 64, device="cuda") % special.numel()].view(4, 64)
    values[invalid] = fill.bfloat16()
    return invalid, values


@pytest.mark.parametrize("case", ["mixed", "all_padding", "no_padding", "empty"])
def test_sparse_pad_zero_matches_masked_fill(case):
    """In-place padding-row zeroing (MCORE_CSA2_SPARSE_PAD_ZERO) writes the bits masked_fill
    writes, forward and through its gradient hook, restores saved rows exactly, and replays
    bit-exactly under contention."""
    seeded()
    invalid, values = _padding_values(case)
    mask = invalid[:, None, None]
    rows, keep = thd_utils.padding_rows(invalid)
    assert rows.shape == keep.shape == (thd_utils.PAD_ROW_CAPACITY,)

    zeroed = values.clone()
    saved = thd_utils.save_rows(zeroed, rows)
    assert thd_utils.zero_rows_(zeroed, rows, keep) is zeroed
    assert bytes_equal(zeroed, values.masked_fill(mask, 0))
    thd_utils.restore_rows_(zeroed, rows, saved)
    assert bytes_equal(zeroed, values)

    weight = torch.randn(values.shape, device="cuda").bfloat16()
    upstream = torch.randn(values.shape, device="cuda").bfloat16().masked_fill(mask, float("nan"))

    def masked(x):
        return (x * 2).masked_fill(mask, 0) * weight

    def in_place(x):
        (y,) = thd_utils.zero_padding_rows_(invalid, x * 2)
        return y * weight

    leaf = values.clone().requires_grad_()
    grad_outputs = {"out": upstream}
    expected = run_once(masked, (leaf,), grad_outputs)
    actual = assert_replays_bit_exact(
        in_place, (leaf,), grad_outputs=grad_outputs, contention=True, what="pad-row zeroing"
    )
    for want, got in zip(expected, actual):
        assert want.keys() == got.keys()
        for name in want:
            assert bytes_equal(got[name], want[name]), name


def _spy_attention_kernels(padding, calls):
    """Deterministic stand-ins for the FlashMLA forward and the cuDNN backward.

    Rows without keys get undefined forward outputs, as in the real kernels, and backward
    scores q against lse like the real one, so a raw q padding row with lse 0 overflows.
    ``calls`` records what MCore hands the backward kernel.
    """

    def forward(q, kv, _indices, scale, *, attn_sink, **_kwargs):
        out = (q.float() * scale + kv.float().mean(0) + attn_sink.float()[None, :, None]).tanh()
        lse = q.float().mean(-1)
        out[padding] = float("nan")
        lse[padding] = float("inf")
        return out.to(q.dtype), lse, None

    def backward(q, kv, out, grad, lse, _sink, *_args, **_kwargs):
        calls.append([t.detach().clone() for t in (q, kv, out, grad, lse)])
        scores = torch.exp((q.float().mean(-1) - lse) * 64)
        ds = scores * (grad.float() * out.float()).sum(-1)
        dq = (ds[..., None] * q.float() + grad.float()).to(q.dtype)
        return dq, (kv.float() * ds.sum()).to(kv.dtype), ds.sum(0)

    return forward, backward


def _run_pad_zero_attention(monkeypatch, function, sparse_pad_zero, refill, inputs):
    """One CSA2 THD attention forward/backward with SPARSE_PAD_ZERO set as given."""
    for module in (csa2, packed_sparse_attention, csa2_indexer):
        monkeypatch.setattr(module, "SPARSE_PAD_ZERO", sparse_pad_zero)
    padding = inputs["padding"]
    calls = []
    forward, backward = _spy_attention_kernels(padding, calls)
    monkeypatch.setattr(packed_sparse_attention, "_csa_fwd_flash_mla", forward)
    monkeypatch.setattr(packed_sparse_attention, "_csa_sparse_attention_backward", backward)
    monkeypatch.setattr(packed_sparse_attention, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(packed_sparse_attention, "get_flash_mla_topk_alignment", lambda: 1)
    monkeypatch.setattr(csa2_indexer.fused_sparse_attention, "_csa_fwd_flash_mla", forward)
    monkeypatch.setattr(
        csa2_indexer.fused_sparse_attention, "_csa_sparse_attention_backward", backward
    )

    def indexer_loss(iq, ik, weights, *_args, **_kwargs):
        loss = iq.sum() + ik.sum() + weights.sum()
        return loss, (torch.ones_like(iq), torch.ones_like(ik), torch.ones_like(weights))

    monkeypatch.setattr(csa2_indexer, "_precompute_indexer_loss", indexer_loss)

    leaves = {
        name: inputs[name].detach().clone().requires_grad_()
        for name in ("q", "kv", "sink", "global_kv", "iq", "ik", "weights")
    }
    rows = padding.numel()
    # Producers whose backward does not read their output, like the q/KV up-projection.
    q_raw, kv_raw = leaves["q"] * 1.5, leaves["kv"] * 1.5
    query, key = csa2._zero_padding_rows_inplace(~padding, q_raw, kv_raw)
    zeros = torch.zeros(rows, 1, dtype=torch.int32, device="cuda")
    if function == "sparse_attention":
        out, _, _ = packed_sparse_attention.CSASparseAttnFunc.apply(
            query,
            key.squeeze(1),
            leaves["sink"],
            zeros,
            torch.ones(rows, dtype=torch.int32, device="cuda"),
            0.125,
            0,
            None,
            padding,
            None,
        )
        output = out.flatten(1)
    else:
        prepared = csa2_indexer.CSA2IndexerInputs(
            leaves["iq"],
            leaves["ik"],
            leaves["weights"],
            1,
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.full((rows,), 2, device="cuda"),
            ~padding,
            2,
            2,
            (rows,),
            thd_layout=SimpleNamespace(cp_size=1),
        )
        output, _ = csa2_indexer.fused_csa2_indexer_sparse_attn(
            prepared,
            query,
            key.squeeze(1),
            leaves["global_kv"],
            leaves["sink"],
            zeros,
            zeros,
            0.125,
            0.25,
            True,
            attention_indices=torch.zeros(rows, 2, dtype=torch.int32, device="cuda"),
            attention_topk_length=torch.where(padding, 0, 2).int(),
            q_padding_mask=padding,
        )
    if refill:
        # mla_up_proj recompute writes the producers' raw rows back into their storage.
        q_raw.data.copy_(leaves["q"].detach() * 1.5)
        kv_raw.data.copy_(leaves["kv"].detach() * 1.5)
    # The upstream gradient is shared: the add hands the same tensor to both branches.
    other = torch.zeros_like(output, requires_grad=True)
    upstream = inputs["upstream"].clone()
    differentiable = ("q", "kv", "sink", "global_kv")
    if function == "sparse_attention":
        differentiable = ("q", "kv", "sink")
    grads = torch.autograd.grad(
        output + other, [leaves[name] for name in differentiable] + [other], upstream
    )
    assert bytes_equal(upstream, inputs["upstream"]), "the shared upstream gradient changed"
    return output.detach(), grads, calls


@pytest.mark.parametrize("refill", [False, True], ids=["eager", "recompute_refill"])
@pytest.mark.parametrize("function", ["sparse_attention", "indexer_sparse_attention"])
def test_sparse_pad_zero_attention_matches_masked_fill(monkeypatch, function, refill):
    """CSA2 THD attention with in-place padding-row zeroing is bit-identical to masked_fill.

    Covers both autograd Functions (packed sparse attention and the joint attention/indexer
    loss), an upstream gradient shared with another consumer, and q/KV storage refilled with
    raw padding rows between forward and backward, as mla_up_proj recompute does. The kernels
    get identical q, out, dO and lse. Refilled KV padding rows only meet zero queries and zero
    gradients, and their gradient rows are zeroed, so only valid KV rows are compared.
    """
    seeded()
    rows, heads, dim = 12, 2, 8
    padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
    padding[[0, 5, 10, 11]] = True
    q = torch.randn(rows, heads, dim, device="cuda").bfloat16()
    kv = torch.randn(rows, 1, dim, device="cuda").bfloat16()
    # Large raw padding rows: unless re-zeroed, they overflow exp(score - lse) in backward.
    q[padding] = 200.0
    kv[padding] = -200.0
    inputs = dict(
        padding=padding,
        q=q,
        kv=kv,
        sink=torch.randn(heads, device="cuda"),
        global_kv=torch.randn(3, 1, dim, device="cuda").bfloat16(),
        iq=torch.randn(rows, 1, 2, device="cuda"),
        ik=torch.randn(2, 2, device="cuda"),
        weights=torch.randn(rows, 1, device="cuda"),
        upstream=torch.randn(rows, heads * dim, device="cuda")
        .bfloat16()
        .masked_fill(padding[:, None], float("nan")),
    )
    results = [
        _run_pad_zero_attention(monkeypatch, function, sparse_pad_zero, refill, inputs)
        for sparse_pad_zero in (False, True)
    ]
    (expected_out, expected_grads, expected_calls), (out, grads, calls) = results
    assert bytes_equal(out, expected_out)
    assert len(grads) == len(expected_grads)
    for got, want in zip(grads, expected_grads):
        assert bytes_equal(got, want)
    # Input gradients are finite; the other consumer receives the upstream gradient unchanged.
    *input_grads, other_grad = grads
    assert all(torch.isfinite(grad.float()).all() for grad in input_grads)
    assert bytes_equal(other_grad, inputs["upstream"])
    assert len(calls) == len(expected_calls) == 1
    valid_kv = torch.cat((~padding, torch.ones(3, dtype=torch.bool, device="cuda")))
    valid_kv = valid_kv[: calls[0][1].shape[0]]
    for index, (got, want) in enumerate(zip(calls[0], expected_calls[0])):
        if index == 1 and refill:
            got, want = got[valid_kv], want[valid_kv]
        assert bytes_equal(got, want), ("q", "kv", "out", "dO", "lse")[index]


def _selection_inputs(logical, physical, max_seqlen, local_rows, cp, ratio, heads, zero_padding):
    """Indexer inputs on the last CP rank of a packed batch."""
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(logical, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor(logical, dtype=torch.int32, device="cuda"),
        cu_seqlens_q_padded=torch.tensor(physical, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv_padded=torch.tensor(physical, dtype=torch.int32, device="cuda"),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    group = SimpleNamespace(size=lambda: cp, rank=lambda: cp - 1)
    layout = thd_utils.build_csa2_thd_layout(params, local_rows, cp_group=group)
    compressed = layout.for_compression(ratio)
    q = torch.randn(local_rows, heads, 128, device="cuda").bfloat16()
    k = torch.randn(compressed.capacity, 128, device="cuda").bfloat16()
    weights = torch.randn(local_rows, heads, device="cuda").bfloat16()
    if zero_padding:
        # csa2 zeroes x and qr on padding rows and the indexer projections have no bias, so
        # padding rows reach selection with q = w = 0 and tie on every causal candidate.
        q[~layout.valid_tokens] = 0
        weights[~layout.valid_tokens] = 0
    return csa2_indexer.prepare_csa2_indexer_inputs(q, k, weights, ratio, None, layout, compressed)


def _assert_padding_segments_keep_selection(monkeypatch, inputs, topk, precision):
    """fused_compact_topk returns the same bits with and without padding segments."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("compact selection requires SM100")
    pytest.importorskip("cudnn")
    csa2_indexer.fused_sparse_attention._ensure_dsa_namespace()
    dsa = csa2_indexer.fused_sparse_attention._DSA
    if not callable(getattr(dsa, "indexer_forward_top_k_wrapper", None)):
        pytest.skip("requires cuDNN compact Top-K")
    assert inputs.compact_topk_metadata[0].numel() > inputs.packed_metadata[0].numel()
    selections = []
    for pad_segments in (False, True):
        monkeypatch.setattr(csa2_indexer, "_COMPACT_TOPK_PAD_SEGMENTS", pad_segments)
        selections.append(csa2_indexer.fused_compact_topk(inputs, topk, precision))
    torch.cuda.synchronize()
    assert bytes_equal(selections[1], selections[0])
    assert bool((selections[0] >= 0).any()), "degenerate test: nothing selected"
    assert_replays_bit_exact(
        lambda: csa2_indexer.fused_compact_topk(inputs, topk, precision),
        (),
        backward=False,
        contention=True,
        what=f"CSA2 {precision} compact Top-K with padding segments",
    )


@pytest.mark.parametrize("zero_padding", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cp", [1, 2])
def test_compact_topk_padding_segments_keep_selection(
    monkeypatch, cp, ratio, heads, precision, zero_padding
):
    """MCORE_CSA2_TOPK_PAD_SEGMENTS: empty, padded and CP-cut segments select the same keys."""
    seeded()
    inputs = _selection_inputs(
        [0, 65, 65, 322], [0, 66, 66, 326], 260, 326 // cp, cp, ratio, heads, zero_padding
    )
    _assert_padding_segments_keep_selection(monkeypatch, inputs, 7, precision)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compact_topk_padding_segments_keep_selection_at_64k_cp16(monkeypatch, ratio):
    """The 64K CP16 shape: one 65535-token sample padded to 65536 rows, last rank, top-k 512,
    with production padding rows (q = w = 0)."""
    seeded()
    inputs = _selection_inputs([0, 65535], [0, 65536], 65536, 4096, 16, ratio, 32, True)
    _assert_padding_segments_keep_selection(monkeypatch, inputs, 512, "mxfp8")


def _assert_selection_ignores_chunk_budget(monkeypatch, inputs, chunk_mib, **selection):
    """fused_candidate_blocks returns the same bits for every score-chunk budget."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("CSA2 candidate scoring requires SM100")
    pytest.importorskip("cudnn")

    def select():
        tokens, blocks = csa2_candidates.fused_candidate_blocks(inputs, **selection)
        return tokens, blocks.indices, blocks.lengths

    results = []
    for mib in chunk_mib:
        monkeypatch.setattr(csa2_candidates, "_SCORE_CHUNK_MAX_BYTES", int(mib * (1 << 20)))
        results.append(select())
    torch.cuda.synchronize()
    for budget, result in zip(chunk_mib[1:], results[1:]):
        for name, got, want in zip(("token Top-K", "blocks", "lengths"), result, results[0]):
            assert bytes_equal(got, want), f"{name} at {budget} MiB"
    assert bool((results[0][0] >= 0).any()) and bool((results[0][1] >= 0).any())


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cp", [1, 2])
def test_candidate_selection_ignores_score_chunk_budget(monkeypatch, cp, ratio, precision):
    """MCORE_CSA2_CANDIDATE_CHUNK_MIB only sets how many query rows one dense-scorer launch
    takes: token Top-K and candidate blocks are bit-identical for single-row-group chunks,
    a ragged last chunk and one chunk, across empty, padded and CP-cut segments."""
    seeded()
    inputs = _selection_inputs(
        [0, 65, 65, 322], [0, 66, 66, 326], 260, 326 // cp, cp, ratio, 32, False
    )
    row_mib = ((inputs.packed_metadata[3] + 3) // 4 * 4) * 4 / (1 << 20)
    _assert_selection_ignores_chunk_budget(
        monkeypatch,
        inputs,
        (2048 * row_mib, 16 * row_mib, 37 * row_mib),
        topk_blocks=3,
        block_size=8,
        precision=precision,
        token_topk=7,
    )


def test_candidate_selection_ignores_score_chunk_budget_at_64k_cp16(monkeypatch):
    """The 64K CP16 candidate source (ratio 1, last rank): the previous 32 MiB budget (128-row
    chunks) and the new 152 MiB default (608-row chunks, a 448-row last chunk) agree."""
    seeded()
    inputs = _selection_inputs([0, 65535], [0, 65536], 65536, 4096, 16, 1, 32, True)
    _assert_selection_ignores_chunk_budget(
        monkeypatch,
        inputs,
        (32, 152),
        topk_blocks=2048,
        block_size=8,
        precision="mxfp8",
        token_topk=512,
    )


def test_packed_indexer_metadata_and_loss_replay():
    seeded()
    cu_q = torch.tensor([0, 65, 65, 256], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 32, 32, 96], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([63, 0, 65], device="cuda", dtype=torch.int32)
    scores = torch.randn(256, 64, device="cuda")
    candidates = torch.randint(-1, 70, (256, 32), device="cuda", dtype=torch.int32)

    def select(score, ids):
        lengths = packed_layout.build_seq_lens(cu_q, cu_k, 256, 4, offsets)
        return packed_layout.sanitize_topk(ids, score, lengths, output_width=128)

    assert_replays_bit_exact(select, (scores, candidates), backward=False, contention=True)
    indices, _ = select(scores, candidates)
    predict = torch.softmax(torch.randn_like(indices, dtype=torch.float32), dim=-1)
    target = torch.softmax(torch.randn_like(predict), dim=-1)

    def loss(t, p, ids):
        return csa_indexer_loss_kernels.sparse_kl_loss(t, p, ids, 0.2, True, 256)

    assert_replays_bit_exact(loss, (target, predict, indices), backward=False, contention=True)


def test_async_context_parallel_collectives_replay():
    """Launch/wait preserves autograd and bit-exact NCCL replay under the suite's Ring policy."""
    import torch.distributed as dist

    from megatron.core.tensor_parallel.mappings import (
        async_gather_from_sequence_parallel_region,
        async_reduce_scatter_along_first_dim,
    )
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        seeded()
        group = dist.group.WORLD
        value = torch.randn(128, 64, device="cuda", requires_grad=True)

        def run(x):
            gathered = async_gather_from_sequence_parallel_region(x, group=group).wait()
            return gathered.square()

        assert_replays_bit_exact(
            run, (value,), contention=True, what="async all-gather/reduce-scatter"
        )
        global_grad = torch.randn(group.size() * 128, 64, device="cuda")

        def reduce(x):
            return async_reduce_scatter_along_first_dim(x, group=group).wait()

        assert_replays_bit_exact(reduce, (global_grad,), backward=False, contention=True)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("ratio", [1, 2])
def test_cp_projected_compaction_replay_and_live_graph_maps(ratio):
    """Unique-source backward remains bit exact when captured padding maps change."""
    from megatron.core.ops.attention.csa.compressor_projection import (
        compact_cp_projection,
        cp_projection_maps,
    )

    seeded()
    sources = torch.arange(32, device="cuda").reshape(-1, ratio)
    valid = torch.ones(sources.shape[0], dtype=torch.bool, device="cuda")
    local = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    halo = torch.randn(ratio, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, boundary):
        mapping, inverse = cp_projection_maps(sources, valid, ratio, ratio, x.shape[0])
        return compact_cp_projection(x, boundary, mapping, inverse)

    assert_replays_bit_exact(
        run, (local, halo), contention=True, what="CSA2 projected CP compaction"
    )
    probe = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16)

    def forward_backward():
        output = run(local, halo)
        return output, *torch.autograd.grad(output, (local, halo), probe)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward_backward()
        forward_backward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward_backward()
    for shift in (0, 1):
        sources.copy_(torch.arange(32, device="cuda").roll(shift).reshape_as(sources))
        valid.fill_(True)
        valid[shift::3] = False
        expected = forward_backward()
        graph.replay()
        torch.cuda.synchronize()
        for actual, wanted in zip(captured, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
