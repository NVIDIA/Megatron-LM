# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB300 checks of the cuDNN frontend >= 1.29 DSA indexer integration (single GPU).

* MXFP8 vs BF16 dense indexer scorer on small packed THD layouts with CP1 and CP>1
  geometry: top-k recall, agreement of the MXFP8 scorer's own selected values with BF16's,
  the loss payload being the BF16 logits shifted per row, and the sparse indexer loss and its
  gradients.
* The selected-score recompute that feeds the indexer loss under MXFP8 (cuDNN
  ``sparse_indexer_score_recompute_wrapper``) against log-softmax of the FE BF16 scores (32
  and 64 indexer heads) within an FP32 summation-order error bound computed from the inputs
  (an MXFP8-input logit must exceed it), the MXFP8 path's loss and gradients against the
  BF16 path's given the same selection, and the MXFP8 loss-on call (score, select,
  recompute) without host syncs and under CUDA graph capture and replay.
* The explicit bottom-right ``q_causal_offsets`` of the BSHD scorer on a top-left frontend.
* ``clean_logits=False`` gives bitwise-identical top-k (when the frontend exposes it).

CP>1 layouts run on one GPU: the scorer only needs this rank's local query rows and the
globally ordered packed keys, which the test builds directly. Every test skips unless the
GPU, cuDNN frontend and Transformer Engine provide the required features.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.quantization import indexer_quantization
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_masking
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes

HEADS, HEAD_DIM = 32, 128
_K = 2048
LAYOUTS = [
    pytest.param((2048, 1024, 1024), 1, 0, 5, id="cp1-3seq-padded"),
    pytest.param((4096, 4096), 2, 0, 3, id="cp2-rank0"),
    pytest.param((4096, 4096), 2, 1, 3, id="cp2-rank1"),
    pytest.param((8192, 8192), 4, 3, 3, id="cp4-rank3"),
]


def _cudnn_dsa_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ is required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("cuDNN frontend DSA is not installed")
    dsa_cudnn_kernels._ensure_dsa_namespace()
    return DSA


def _require_mxfp8():
    dsa = _cudnn_dsa_or_skip()
    if not all(
        dsa_cudnn_kernels._accepts_named_parameter(dsa.indexer_forward_wrapper, name)
        for name in dsa_cudnn_kernels._MXFP8_SCORER_WRAPPER_PARAMETERS
    ):
        pytest.skip("cuDNN frontend lacks the MXFP8 dense indexer scorer (needs >= 1.29)")
    if not (indexer_quantization.HAVE_TE_MXFP8 and indexer_quantization.HAVE_TRITON):
        pytest.skip("Transformer Engine MXFP8 quantization or Triton is unavailable")


def _inputs(sq, sk, seed, outlier_channels=0, signed_weights=False, heads=HEADS):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((1, sq, heads, HEAD_DIM), generator=generator, device="cuda")
    k = torch.randn((1, sk, HEAD_DIM), generator=generator, device="cuda")
    if outlier_channels:
        # Without a Hadamard rotation a few large channels dominate each 1x32 block scale.
        channels = torch.randperm(HEAD_DIM, generator=generator, device="cuda")[:outlier_channels]
        q[..., channels] *= 16.0
        k[..., channels] *= 16.0
    if signed_weights:
        # Trained-indexer-like weights: logits stay within a few units of each other.
        w = torch.randn((1, sq, heads), generator=generator, device="cuda") * heads**-0.5
    else:
        w = torch.rand((1, sq, heads), generator=generator, device="cuda")
    return q.bfloat16(), k.bfloat16(), w.bfloat16()


def _require_recompute():
    dsa = _cudnn_dsa_or_skip()
    if getattr(dsa, "sparse_indexer_score_recompute_wrapper", None) is None:
        pytest.skip("cuDNN frontend lacks sparse_indexer_score_recompute_wrapper")


def _dense_reference(q, k, w):
    """FP32 ``[sq, sk]`` logits of the BF16 inputs (no TF32)."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dense = torch.einsum("thd,kd->thk", q[0].float(), k[0].float()).relu()
        return (dense * w[0].float().unsqueeze(-1)).sum(dim=1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _score(q, k, w, layout, cp_size, cp_rank, topk, options, deterministic=False):
    starts, ends, cu, max_seqlen = layout
    return dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        topk,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=cu,
        packed_cu_seqlens_k=cu,
        packed_max_seqlen_q=max_seqlen,
        packed_max_seqlen_k=max_seqlen,
        packed_cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
        deterministic=deterministic,
        scorer_options=options,
    )


def _scores_by_key(indices, scores, num_keys):
    """Scatter one row's selected scores to a ``(rows, num_keys)`` map; NaN elsewhere."""
    rows = indices.shape[1]
    dense = torch.full((rows, num_keys + 1), float("nan"), device=indices.device)
    idx = indices[0].long()
    dense.scatter_(1, torch.where(idx >= 0, idx, num_keys), scores[0].float())
    return dense[:, :num_keys]


def _recall(reference_indices, candidate_indices, num_keys):
    """Per-row |candidate ∩ reference| / |reference| over rows with a non-empty reference."""
    rows = reference_indices.shape[1]
    ref = reference_indices[0].long()
    cand = candidate_indices[0].long()
    member = torch.zeros((rows, num_keys + 1), dtype=torch.bool, device=ref.device)
    member.scatter_(1, torch.where(ref >= 0, ref, num_keys), True)
    member[:, num_keys] = False
    hits = torch.gather(member, 1, torch.where(cand >= 0, cand, num_keys)).sum(dim=1)
    sizes = (ref >= 0).sum(dim=1)
    valid = sizes > 0
    return hits[valid].float() / sizes[valid].float()


@pytest.mark.parametrize("outlier_channels", [0, 4], ids=["gaussian", "outliers"])
@pytest.mark.parametrize("seq_lens, cp_size, cp_rank, max_num_seqs", LAYOUTS)
def test_mxfp8_topk_overlaps_bf16(
    seq_lens, cp_size, cp_rank, max_num_seqs, outlier_channels, monkeypatch
):
    _require_mxfp8()
    _require_recompute()
    layout = fakes.packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs, device="cuda")
    sq = layout[0].numel()
    sk = sq * cp_size
    q, k, w = _inputs(sq, sk, seed=sq + cp_rank, outlier_channels=outlier_channels)
    topk = 512
    bf16 = _score(q, k, w, layout, cp_size, cp_rank, topk, dsa_cudnn_kernels.IndexerScorerOptions())
    mxfp8 = _score(
        q,
        k,
        w,
        layout,
        cp_size,
        cp_rank,
        topk,
        dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8"),
    )
    torch.testing.assert_close(mxfp8[1], bf16[1], rtol=0, atol=0)  # same per-row lengths

    recall = _recall(bf16[0], mxfp8[0], sk)
    print(
        f"[mxfp8-recall] layout={seq_lens} cp={cp_size}/{cp_rank} outliers={outlier_channels} "
        f"mean={recall.mean().item():.4f} min={recall.min().item():.4f}"
    )
    assert recall.mean().item() >= (0.80 if outlier_channels else 0.90)

    # The loss payload is the BF16 logits of the MXFP8 selection minus one log-sum-exp per
    # row: on keys both scorers selected, BF16 score - payload is flat within a row (keys
    # whose probability underflowed FLT_MIN are floored and left out). Slot order is
    # unspecified, so compare per key rather than per slot.
    bf16_dense = _scores_by_key(bf16[0], bf16[2], sk)
    payload_dense = _scores_by_key(mxfp8[0], mxfp8[2], sk)
    keep = (
        torch.isfinite(bf16_dense)
        & torch.isfinite(payload_dense)
        & (payload_dense > math.log(torch.finfo(torch.float32).tiny) + 1.0)
    )
    shift = bf16_dense - payload_dense
    spread = (
        torch.where(keep, shift, float("-inf")).amax(dim=1)
        - torch.where(keep, shift, float("inf")).amin(dim=1)
    )[keep.any(dim=1)]
    scale = bf16_dense[keep].abs().max()
    print(f"[mxfp8-payload] max per-row spread of bf16 - payload = {spread.max().item():.4e}")
    assert spread.max().item() <= 1e-4 * scale.item()

    # The MXFP8 scorer's own selected values (the loss never sees them; read here with the
    # recompute switched off) are within MXFP8 tolerance of BF16's on keys both selected.
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_recomputes_indexer_loss_scores", lambda precision: False
    )
    own = _score(
        q,
        k,
        w,
        layout,
        cp_size,
        cp_rank,
        topk,
        dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8"),
    )
    mxfp8_dense = _scores_by_key(own[0], own[2], sk)
    common = torch.isfinite(bf16_dense) & torch.isfinite(mxfp8_dense)
    rel_err = (mxfp8_dense - bf16_dense)[common].abs().max() / bf16_dense[common].abs().max()
    print(f"[mxfp8-scores] max |delta| / max |score| over common keys = {rel_err.item():.4e}")
    assert rel_err.item() < 0.1


def _sparse_loss(q, k, w, query, key, layout, cp_size, cp_rank, topk, precision):
    _starts, _ends, cu, max_seqlen = layout
    config = SimpleNamespace(
        kv_lora_rank=512,
        deterministic_mode=False,
        dsa_indexer_precision=precision,
        dsa_indexer_skip_masked_score_fill=False,
    )
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    q_sb = q.permute(1, 0, 2, 3).contiguous().requires_grad_(True)
    k_sb = k.permute(1, 0, 2).contiguous().requires_grad_(True)
    w_sb = w.permute(1, 0, 2).contiguous().requires_grad_(True)
    result = dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
        q_sb,
        k_sb,
        w_sb,
        topk,
        layout[0].to(torch.int32),
        layout[1].to(torch.int32),
        block_size=8192,
        query=query,
        key=key,
        softmax_scale=576**-0.5,
        loss_coeff=1.0,
        pg_collection=SimpleNamespace(tp=None),
        config=config,
        use_local_indexer_varlen=True,
        packed_seq_params=packed,
        cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
    )
    if result is None:
        pytest.skip("fused sparse indexer loss declined this layout (FlashMLA support)")
    _indices, _lengths, loss = result
    loss.backward()
    return loss.detach(), q_sb.grad, k_sb.grad, w_sb.grad


@pytest.mark.parametrize("seq_lens, cp_size, cp_rank, max_num_seqs", LAYOUTS)
def test_mxfp8_sparse_indexer_loss_tracks_bf16(seq_lens, cp_size, cp_rank, max_num_seqs):
    """Both arms feed the KL from BF16-input scores (MXFP8 recomputes the selected keys'),
    so the remaining gap comes from the MXFP8 selection alone."""
    _require_mxfp8()
    _require_recompute()
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("FlashMLA is required for the sparse indexer loss")
    layout = fakes.packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs, device="cuda")
    sq = layout[0].numel()
    sk = sq * cp_size
    q, k, w = _inputs(sq, sk, seed=31 + cp_rank)
    generator = torch.Generator(device="cuda").manual_seed(5)
    query = torch.randn((sq, 1, 64, 576), generator=generator, device="cuda").bfloat16()
    key = torch.randn((sk, 1, 1, 576), generator=generator, device="cuda").bfloat16()

    bf16 = _sparse_loss(q, k, w, query, key, layout, cp_size, cp_rank, 512, "bf16")
    mxfp8 = _sparse_loss(q, k, w, query, key, layout, cp_size, cp_rank, 512, "mxfp8")
    rel = ((mxfp8[0] - bf16[0]).abs() / bf16[0].abs().clamp(min=1e-12)).item()
    print(
        f"[mxfp8-loss] layout={seq_lens} cp={cp_size}/{cp_rank} bf16={bf16[0].item():.6e} "
        f"mxfp8={mxfp8[0].item():.6e} rel={rel:.3e}"
    )
    assert rel < 0.1
    for name, grad_bf16, grad_mxfp8 in zip(("q", "k", "w"), bf16[1:], mxfp8[1:]):
        cosine = torch.nn.functional.cosine_similarity(
            grad_bf16.float().flatten(), grad_mxfp8.float().flatten(), dim=0
        ).item()
        print(f"[mxfp8-grad] {name} cosine={cosine:.5f}")
        assert cosine > 0.9


# One FP32 rounding or truncation (tensor-core accumulation truncates) per operation.
_FP32_UNIT = 2.0**-23


def _gamma(n: int) -> float:
    """Higham's gamma_n = n u / (1 - n u): the relative error bound of n such operations."""
    return n * _FP32_UNIT / (1 - n * _FP32_UNIT)


def _mxfp8_round_trip(x):
    """FP32 values of BF16 ``x`` (..., 128) after an MXFP8 cast (E4M3, E8M0 1x32 scales)."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=2.0**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


def _selected_logit_terms(q, k, w, indices, lengths, rows_per_chunk=256):
    """Per selected key: ``A = sum_h |w_h| sum_t |q_ht k_t|`` (the error model's mass) and the
    logit of MXFP8-cast q and k (the negative control), both FP32 ``[1, sq, K]``."""
    sq, top_k = indices.shape[1], indices.shape[2]
    mass = torch.zeros((sq, top_k), device="cuda")
    control = torch.zeros((sq, top_k), device="cuda")
    q_mx, k_mx = _mxfp8_round_trip(q[0]), _mxfp8_round_trip(k[0])
    q_abs, k_abs, w32 = q[0].float().abs(), k[0].float().abs(), w[0].float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, sq, rows_per_chunk):
            rows = slice(start, start + rows_per_chunk)
            keys = indices[0, rows].long().clamp(min=0)
            mass[rows] = (
                torch.einsum("rkd,rhd->rkh", k_abs[keys], q_abs[rows])
                * w32[rows].abs().unsqueeze(1)
            ).sum(-1)
            control[rows] = (
                torch.einsum("rkd,rhd->rkh", k_mx[keys], q_mx[rows]).relu() * w32[rows].unsqueeze(1)
            ).sum(-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    valid = torch.arange(top_k, device="cuda").view(1, -1) < lengths[0].view(-1, 1)
    return torch.where(valid, mass, 0).unsqueeze(0), control.unsqueeze(0)


@pytest.mark.parametrize(
    "seq_lens, cp_size, cp_rank", [((8192, 4096, 4096), 1, 0), ((131072,), 32, 5)]
)
@pytest.mark.parametrize("heads", [32, 64])
def test_recompute_equals_log_softmax_of_fe_bf16_scores(heads, seq_lens, cp_size, cp_rank):
    """The MXFP8 path's loss payload: the selected keys' scores recomputed in FP32 from the
    BF16 inputs equal the log-softmax of the FE BF16 scorer's scores over the selection, with
    both indexer head counts MXFP8 accepts (the recompute kernel tiles them differently), up
    to FP32 summation order.

    Error model. Both kernels compute each selected logit ``s = sum_h w_h relu(q_h . k)`` in
    FP32 from BF16 q, k and w (a product of two BF16 values is exact in FP32), each in its own
    order and blocking. A term passes through at most n = 128 + H + 2 FP32 operations (127
    dot-product accumulations, the product with w_h, H - 1 head accumulations, 2 to spare),
    each with a relative error below u = 2^-23 (rounded or truncated). So each kernel's logit
    is within gamma_n * A of the exact one, A = sum_h |w_h| sum_t |q_ht k_t| (Higham, Accuracy
    and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1 and (3.5)). log p_i = s_i - LSE
    and the LSE is 1-Lipschitz in the max norm, so each kernel's log p_i is within
    gamma_n * (A_i + max_j A_j) of the exact one, plus its own log-softmax rounding, at most
    gamma_K (the normalising sum of K terms) + 4u (M + ln K), M the row's largest |s|. The two
    kernels differ by at most twice that, per selected element:

        bound_i = 2 gamma_n (A_i + max_j A_j) + 2 (gamma_K + 4u (M + ln K)).

    On these inputs it is 0.02-0.07, while FP32 order noise is about 2e-5 (the first GB300
    run measured 1.6e-5 to 1.9e-5 against an earlier fixed 1e-5 bound). Not loose for the
    defect the recompute guards against: a logit computed from MXFP8-cast q and k exceeds the
    bound by tens of times (negative control, asserted).
    """
    _require_recompute()
    layout = fakes.packed_zigzag_layout(seq_lens, cp_size, cp_rank, 5, device="cuda")
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * cp_size, seed=11, signed_weights=True, heads=heads)
    fe_bf16 = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True)
    indices, lengths, fe_scores = _score(q, k, w, layout, cp_size, cp_rank, _K, fe_bf16)
    payload = dsa_cudnn_kernels._recompute_selected_indexer_log_probs(
        q, k.unsqueeze(2), w, indices, topk_k=_K
    )
    valid = torch.arange(_K, device="cuda").view(1, 1, -1) < lengths.unsqueeze(-1)
    want = dsa_masking.masked_log_softmax(fe_scores, valid)
    got = dsa_masking.masked_log_softmax(payload, valid)
    mass, control_logits = _selected_logit_terms(q, k, w, indices, lengths)
    row_mass = mass.amax(dim=-1, keepdim=True)
    row_max = torch.where(valid, fe_scores.abs(), 0).amax(dim=-1, keepdim=True)
    bound = 2 * _gamma(HEAD_DIM + heads + 2) * (mass + row_mass) + 2 * (
        _gamma(_K) + 4 * _FP32_UNIT * (row_max + math.log(_K))
    )
    error = torch.where(valid, (got - want).abs(), 0)
    worst = int(error.flatten().argmax())
    ratio = (error / bound)[valid].max().item()
    control = dsa_masking.masked_log_softmax(control_logits, valid)
    control_ratio = (torch.where(valid, (control - want).abs(), 0) / bound)[valid].max().item()
    print(
        f"[mxfp8-recompute] {seq_lens} cp{cp_size}/{cp_rank} heads={heads}: "
        f"max |log p - FE BF16 log p| = {error.flatten()[worst].item():.3e} at |log p| "
        f"{want.flatten()[worst].abs().item():.3f}, |logit| "
        f"{fe_scores.flatten()[worst].abs().item():.3f}, A {mass.flatten()[worst].item():.1f}; "
        f"bound there {bound.flatten()[worst].item():.3e}; max error/bound {ratio:.2e}; "
        f"bound {bound[valid].min().item():.3e}..{bound[valid].max().item():.3e}; "
        f"MXFP8-cast control max error/bound {control_ratio:.1f}"
    )
    assert bool((error <= bound)[valid].all()), "the recompute exceeds the FP32 error bound"
    assert control_ratio > 1, "the bound would accept a logit from MXFP8-cast inputs"
    # A selected key whose logit trails the row maximum by > 90 underflows FLT_MIN.
    w_gap = w.clone()
    w_gap[0, 100] *= 1000
    gap = dsa_cudnn_kernels._recompute_selected_indexer_log_probs(
        q, k.unsqueeze(2), w_gap, indices, topk_k=_K
    )
    assert bool(torch.isfinite(gap[valid]).all())


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_loss_on_call_has_no_host_sync_and_replays_like_eager(heads, monkeypatch):
    """The MXFP8 loss-on call (score, select, recompute the selected keys' scores) runs under
    ``set_sync_debug_mode('error')`` and in a CUDA graph, whose replay selects and scores what
    the eager call does, also after the inputs change in place. Like every cuDNN DSA kernel,
    the recompute compiles at its first eager call (one variant per top-k and head count), so
    the capture follows eager calls, as full-iteration graphs do after their warm-up steps.
    Deterministic top-k, so the slots compare bitwise."""
    _require_mxfp8()
    _require_recompute()
    # 128K CP32 rank 5: 4096 query rows over 131072 keys.
    layout = fakes.packed_zigzag_layout((131072,), 32, 5, 5, device="cuda")
    sq = layout[0].numel()
    sk = sq * 32
    static = list(_inputs(sq, sk, seed=9, signed_weights=True, heads=heads))
    options = dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8")
    real = dsa_cudnn_kernels._recompute_selected_indexer_log_probs
    recomputes = []

    def counting(*args, **kwargs):
        recomputes.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_recompute_selected_indexer_log_probs", counting)

    def call():
        return _score(*static, layout, 32, 5, _K, options, deterministic=True)

    def same(actual, expected):
        assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=1e-6)

    call()  # first-use compiles (scorer, top-k, recompute) stay out of the sync check
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = call()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    same(captured, eager)
    new = _inputs(sq, sk, seed=10, signed_weights=True, heads=heads)
    for target, source in zip(static, new):
        target.copy_(source)
    graph.replay()
    torch.cuda.synchronize()
    same(captured, call())
    # warm-up, sync check, side stream, capture, last eager call: one recompute each
    assert len(recomputes) == 5
    print(f"[mxfp8-recompute-graph] heads={heads}: replay after a data swap equals eager")


def test_mxfp8_loss_and_gradients_equal_the_bf16_path_given_the_selection(monkeypatch):
    """Given the MXFP8 selection, the MXFP8 path's sparse indexer loss and gradients equal the
    BF16 path's (its FP32 scores of the same keys) up to FP32 summation order."""
    _require_mxfp8()
    _require_recompute()
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("FlashMLA is required for the sparse indexer loss")
    layout = fakes.packed_zigzag_layout((4096, 2048, 2048), 1, 0, 5, device="cuda")
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq, seed=13, signed_weights=True)
    generator = torch.Generator(device="cuda").manual_seed(5)
    query = torch.randn((sq, 1, 64, 576), generator=generator, device="cuda").bfloat16()
    key = torch.randn((sq, 1, 1, 576), generator=generator, device="cuda").bfloat16()
    real = dsa_cudnn_kernels._indexer_topk_bshd
    seen = {}

    def recording(*args, **kwargs):
        seen["selection"] = real(*args, **kwargs)
        return seen["selection"]

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", recording)
    mxfp8 = _sparse_loss(q, k, w, query, key, layout, 1, 0, _K, "mxfp8")
    indices, lengths, _payload = seen["selection"]
    valid = torch.arange(_K, device="cuda").view(1, 1, -1) < lengths.unsqueeze(-1)
    bf16_payload = torch.gather(_dense_reference(q, k, w), 1, indices[0].clamp(min=0).long())
    bf16_payload = bf16_payload.unsqueeze(0).masked_fill(~valid, torch.finfo(torch.float32).min)

    def same_selection(*_args, **_kwargs):
        return indices.clone(), lengths.clone(), bf16_payload.clone()

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", same_selection)
    bf16 = _sparse_loss(q, k, w, query, key, layout, 1, 0, _K, "bf16")
    rel = ((mxfp8[0] - bf16[0]).abs() / bf16[0].abs()).item()
    print(
        f"[mxfp8-loss-same-selection] mxfp8 {mxfp8[0].item():.6e} bf16 {bf16[0].item():.6e} rel {rel:.2e}"
    )
    assert rel <= 1e-5
    for name, grad_mxfp8, grad_bf16 in zip("qkw", mxfp8[1:], bf16[1:]):
        cosine = torch.nn.functional.cosine_similarity(
            grad_mxfp8.float().flatten(), grad_bf16.float().flatten(), dim=0
        ).item()
        print(f"[mxfp8-loss-same-selection] grad {name} cosine {cosine:.8f}")
        assert cosine > 1 - 1e-5


def test_bshd_bottom_right_offsets_on_this_frontend():
    """Cropped BSHD scoring keeps the bottom-right alignment on any installed frontend."""
    _cudnn_dsa_or_skip()
    sq, sk = 256, 1024
    q, k, w = _inputs(sq, sk, seed=3)
    scores = dsa_cudnn_kernels._indexer_forward_bshd_bottom_right(q, k.unsqueeze(2), w, ratio=1)
    reference = (
        torch.einsum("bqhd,bkd->bqhk", q.float(), k.float()).relu() * w.float().unsqueeze(-1)
    ).sum(dim=2)
    limits = torch.arange(sk - sq + 1, sk + 1, device="cuda").view(1, sq, 1)
    valid = torch.arange(sk, device="cuda").view(1, 1, sk) < limits
    torch.testing.assert_close(scores[valid], reference[valid], rtol=2e-2, atol=2e-2)
    assert torch.isneginf(scores[~valid]).all()


@pytest.mark.parametrize("seq_lens, cp_size, cp_rank, max_num_seqs", LAYOUTS)
def test_clean_logits_is_bitwise_identical(seq_lens, cp_size, cp_rank, max_num_seqs):
    _cudnn_dsa_or_skip()
    if not dsa_cudnn_kernels._supports_indexer_clean_logits():
        pytest.skip("the installed cuDNN frontend has no clean_logits switch")
    layout = fakes.packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs, device="cuda")
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * cp_size, seed=17)
    # Without deterministic mode the radix top-k writes its selection in an unspecified
    # slot order, so even two prefilled runs differ slot by slot. Deterministic mode
    # canonicalizes the selection, which makes a bitwise comparison meaningful.
    filled = _score(
        q,
        k,
        w,
        layout,
        cp_size,
        cp_rank,
        512,
        dsa_cudnn_kernels.IndexerScorerOptions(),
        deterministic=True,
    )
    skipped = _score(
        q,
        k,
        w,
        layout,
        cp_size,
        cp_rank,
        512,
        dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True),
        deterministic=True,
    )
    for expected, actual in zip(filled, skipped):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
