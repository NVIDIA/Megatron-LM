# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KPool migration contracts for PR 7522 head 0e92955e.

The compression references are independent torch math, including the separate
HF rounding profile from huggingface/transformers commit
93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py,
Glm5NextTextIndexer.get_pooled_states and forward. This is module-level coverage,
not official-checkpoint whole-model acceptance.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
    _compute_kpool_hf_scores,
    _kpool_compress_keys,
    _kpool_compress_keys_per_seg,
    _kpool_topk_from_score_chunks,
    _unfused_absorbed_dsa_fn,
    _validate_dsa_kpool_config,
    fused_qk_topk_kpool,
)


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def _hf_complete_pools(keys, gates, ape, pool_size=4):
    """Independent HF-layout softmax, key-dtype cast, multiply, and reduction."""
    usable = keys.shape[0] // pool_size * pool_size
    batch, width = keys.shape[1:]
    grouped_keys = keys[:usable].transpose(0, 1).reshape(batch, -1, pool_size, width)
    grouped_gates = gates[:usable].transpose(0, 1).reshape(batch, -1, pool_size, width)
    logits = grouped_gates.float() + ape.float()[None, None]
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    return (probabilities * grouped_keys).sum(dim=2).transpose(0, 1)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_kpool_default_preserves_pr_compression_exactly(dtype):
    """The new HF opt-in must not silently alter PR 7522's default profile."""
    torch.manual_seed(574)
    keys = torch.randn(13, 2, 128, device='cuda', dtype=dtype)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    grouped_keys = keys[:12].reshape(3, 4, 2, 128).float()
    logits = gates[:12].reshape(3, 4, 2, 128).float() + ape[None, :, None]
    exp_logits = (logits - logits.max(dim=1, keepdim=True).values).exp()
    expected = (
        (
            (exp_logits * grouped_keys).sum(dim=1, keepdim=True)
            / exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12)
        )
        .squeeze(1)
        .bfloat16()
    )
    default = _kpool_compress_keys(keys, gates, ape, 4)
    explicit = _kpool_compress_keys(keys, gates, ape, 4, kpool_key_dtype_compression=False)
    torch.testing.assert_close(default, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('length', [0, 3, 4, 9])
def test_hf_kpool_compression_matches_released_rounding_exactly(dtype, length):
    torch.manual_seed(575)
    keys = torch.randn(length, 2, 128, device='cuda', dtype=dtype)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    actual = _kpool_compress_keys(keys, gates, ape, 4, kpool_key_dtype_compression=True)
    expected = _hf_complete_pools(keys, gates, ape)
    assert actual.dtype == keys.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('lengths', [(3,), (3, 9, 5), (0, 3, 0, 9, 5), (1, 1, 1, 1)])
def test_hf_packed_pooling_resets_segments_and_pads_only_unused_slots(dtype, lengths):
    """Non-aligned documents, empty documents, tails and padding retain HF math."""
    torch.manual_seed(576)
    keys = torch.randn(sum(lengths), 2, 128, device='cuda', dtype=dtype)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device='cuda')
    actual, bases = _kpool_compress_keys_per_seg(
        keys, gates, ape, 4, cu, kpool_key_dtype_compression=True
    )
    expected_pools, expected_bases = [], []
    start = 0
    for length in lengths:
        expected_pools.append(
            _hf_complete_pools(keys[start : start + length], gates[start : start + length], ape)
        )
        expected_bases.extend(start + offset for offset in range(0, length - 3, 4))
        start += length
    expected = torch.cat(expected_pools, dim=0)
    assert actual.dtype == keys.dtype
    assert actual.shape == (sum(lengths) // 4, 2, 128)
    torch.testing.assert_close(actual[: len(expected_bases)], expected, rtol=0, atol=0)
    assert bases[: len(expected_bases)].tolist() == expected_bases
    assert torch.all(bases[len(expected_bases) :] == -1)


def _hf_padded_topk_reference(q, keys, gates, ape, weights, key_valid, topk=8):
    """Source-ordered native HF selection for a ``[batch, sequence]`` padding vector.

    Transcribes Glm5NextTextIndexer.forward, get_visible_tokens, get_pooled_states and
    append_visible_tail: pools start at each sample's first valid key, a pool is a
    candidate only when every slot is valid and its final token is visible, and the
    incomplete tail starts at ``first_key + visible - visible % pool``. Padded query rows
    select nothing. Selected pools fill slots ``[0, topk)`` and the tail fills
    ``[topk, topk + pool - 1)``, the fixed-width selector layout. Score column
    ``first_key // pool + j`` holds the sample's pool ``j``.

    No Megatron pooling, scoring, mask or selection helper is used. Shapes are
    reduced in token/head count only; the released indexer head width remains 128.
    """
    sequence, batch, _, width = q.shape
    pool_size = 4
    device = q.device
    positions = torch.arange(sequence, device=device)
    visible = (positions[None, None, :] <= positions[None, :, None]) & key_valid[:, None, :]

    number_of_pools = (sequence + pool_size - 1) // pool_size
    first_key = torch.where(
        key_valid.any(-1),
        key_valid.long().argmax(-1),
        torch.full((batch,), sequence, dtype=torch.long, device=device),
    )
    pool_offsets = torch.arange(number_of_pools * pool_size, device=device)
    pool_indices = first_key[:, None, None] + pool_offsets.view(1, number_of_pools, pool_size)
    batch_index = torch.arange(batch, device=device)[:, None, None]
    safe_indices = pool_indices.clamp(0, sequence - 1)
    grouped_keys = keys.transpose(0, 1)[batch_index, safe_indices]
    grouped_gates = gates.transpose(0, 1)[batch_index, safe_indices]
    grouped_valid = key_valid[batch_index, safe_indices] & (pool_indices < sequence)
    pool_valid = grouped_valid.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid, -1)
    logits = grouped_gates.float() + ape.float()[None, None]
    logits = logits.masked_fill(~grouped_valid[..., None], float('-inf'))
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    pool_keys = (probabilities * grouped_keys).sum(dim=2)

    # Retain the source's matrix-product reduction over all indexer heads.
    scores = torch.matmul(q.transpose(0, 1).float(), pool_keys.transpose(-1, -2).float()[:, None])
    scores = F.relu(scores * (width**-0.5))
    scores = torch.matmul(weights.transpose(0, 1).float().unsqueeze(-2), scores).squeeze(-2)

    pool_end = pool_indices[..., -1].clamp(0, sequence - 1)
    candidates = visible.gather(-1, pool_end[:, None, :].expand(batch, sequence, -1))
    candidates = candidates & pool_valid[:, None]
    select_k = min(topk // pool_size, number_of_pools)
    selected = scores.masked_fill(~candidates, torch.finfo(torch.float32).min)
    selected = selected.topk(select_k, dim=-1).indices
    selected_valid = candidates.gather(-1, selected)
    selected_tokens = pool_indices[batch_index, selected]
    selected_tokens = selected_tokens.masked_fill(~selected_valid[..., None], -1).flatten(-2)

    output = torch.full((batch, sequence, topk + 3), -1, device=device, dtype=torch.int32)
    output[..., : selected_tokens.size(-1)] = selected_tokens.to(torch.int32)
    visible_count = visible.long().sum(-1)
    tail_count = visible_count.remainder(pool_size)
    tail_offsets = torch.arange(pool_size - 1, device=device)
    tail = (first_key[:, None] + visible_count - tail_count)[..., None] + tail_offsets
    tail_valid = (tail_offsets < tail_count[..., None]) & (tail < sequence)
    tail_valid = tail_valid & visible.gather(-1, tail.clamp(0, sequence - 1))
    output[..., topk:] = tail.masked_fill(~tail_valid, -1).to(torch.int32)
    output = output.masked_fill(~key_valid[..., None], -1)

    all_scores = torch.full((batch, sequence, sequence // pool_size), float('-inf'), device=device)
    candidate_scores = scores.masked_fill(~candidates, float('-inf'))
    columns = first_key[:, None] // pool_size + torch.arange(number_of_pools, device=device)
    for sample in range(batch):
        # Complete valid pools end in range, so their columns exist in the score tensor.
        keep = pool_valid[sample]
        all_scores[sample][:, columns[sample, keep]] = candidate_scores[sample][:, keep]
    all_scores = all_scores.masked_fill(~key_valid[..., None], float('-inf'))
    return output, all_scores


def _hf_topk_reference(q, keys, gates, ape, weights, valid_start, valid_end, topk=8):
    """HF reference for one contiguous valid span shared by every sample."""
    key_valid = torch.zeros(q.size(1), keys.size(0), dtype=torch.bool, device=q.device)
    key_valid[:, valid_start:valid_end] = True
    return _hf_padded_topk_reference(q, keys, gates, ape, weights, key_valid, topk)


def _padding_causal_mask(key_valid):
    """Additive causal mask that hides padded keys and fully masks padded query rows."""
    positions = torch.arange(key_valid.size(-1), device=key_valid.device)
    visible = (positions[None, :] <= positions[:, None])[None]
    visible = visible & key_valid[:, None, :] & key_valid[:, :, None]
    mask = torch.zeros(visible.shape, device=key_valid.device)
    return mask.masked_fill(~visible, float('-inf'))


def _run_hf_selector(q, keys, gates, ape, weights, mask, index_topk, return_index_scores):
    return fused_qk_topk_kpool(
        q,
        keys,
        weights,
        index_topk=index_topk,
        pool_size=4,
        gate_score=gates,
        ape=ape,
        mask=mask,
        use_relu=True,
        always_select_tail=True,
        fp8_indexer=False,
        rotate_activation_enabled=False,
        return_index_scores=return_index_scores,
        kpool_key_dtype_compression=True,
        hf_scoring=True,
    )


def _assert_matches_hf_reference(actual, actual_scores, expected, expected_scores):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if actual_scores is not None:
        assert torch.equal(torch.isfinite(actual_scores), torch.isfinite(expected_scores))
        valid = torch.isfinite(expected_scores)
        torch.testing.assert_close(
            actual_scores[valid], expected_scores[valid], rtol=1e-6, atol=1e-5
        )


@pytest.mark.parametrize('return_index_scores', [False, True])
@pytest.mark.parametrize('valid_range', [(0, 17), (4, 16)])
@pytest.mark.parametrize('heads', [1, 32])
def test_hf_compression_selector_matches_independent_source_topk(
    return_index_scores, valid_range, heads
):
    """Exercise causal selection, score-free selection and aligned padding."""
    torch.manual_seed(577)
    sequence, batch, width = 17, 2, 128
    # Positive operands prevent ReLU-created zero-score ties, whose pool order
    # is unspecified when padded and unpadded candidate arrays have different sizes.
    q = torch.rand(sequence, batch, heads, width, device='cuda', dtype=torch.bfloat16)
    keys = torch.rand(sequence, batch, width, device='cuda', dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, width, device='cuda')
    weights = torch.rand(sequence, batch, heads, device='cuda') * (heads**-0.5)
    start, end = valid_range
    mask = torch.full((batch, sequence, sequence), float('-inf'), device='cuda')
    for query in range(start, end):
        mask[:, query, start : query + 1] = 0
    actual_scores, actual = fused_qk_topk_kpool(
        q,
        keys,
        weights,
        index_topk=8,
        pool_size=4,
        gate_score=gates,
        ape=ape,
        mask=mask,
        use_relu=True,
        always_select_tail=True,
        fp8_indexer=False,
        rotate_activation_enabled=False,
        return_index_scores=return_index_scores,
        kpool_key_dtype_compression=True,
        hf_scoring=True,
    )
    expected, expected_scores = _hf_topk_reference(q, keys, gates, ape, weights, start, end)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if return_index_scores:
        assert torch.equal(torch.isfinite(actual_scores), torch.isfinite(expected_scores))
        valid = torch.isfinite(expected_scores)
        torch.testing.assert_close(
            actual_scores[valid], expected_scores[valid], rtol=1e-6, atol=1e-5
        )


@pytest.mark.parametrize('return_index_scores', [False, True])
@pytest.mark.parametrize('mask_ndim', [2, 3])
def test_hf_selector_left_padding_counterexample(mask_ndim, return_index_scores):
    """Two padded keys and a one-pool budget: only HF's first-valid-key origin wins.

    Pools from the first valid key are [2:6], [6:10] and [10:14] with means 16, 10
    and 0, so the final query selects [2:6] and has no tail. Globally aligned pools
    would be [4:8] and [8:12] with means 6 and 4, selecting [4:8] plus tail [12:14].
    There is no score tie.
    """
    sequence, heads, width = 14, 32, 128
    values = torch.tensor([0, 0, 32, 32, 0, 0, 12, 12, 8, 8, 0, 0, 0, 0], device='cuda')
    keys = values[:, None, None].expand(sequence, 1, width).to(torch.bfloat16).contiguous()
    q = torch.ones(sequence, 1, heads, width, device='cuda', dtype=torch.bfloat16)
    gates = torch.zeros(sequence, 1, width, device='cuda')
    ape = torch.zeros(4, width, device='cuda')
    # Raw BF16 head-weight projections of 1, upcast and scaled by the head count.
    weights = torch.ones(sequence, 1, heads, device='cuda') * (heads**-0.5)
    key_valid = torch.arange(sequence, device='cuda')[None] >= 2
    mask = _padding_causal_mask(key_valid)
    if mask_ndim == 2:
        mask = mask[0]
    actual_scores, actual = _run_hf_selector(
        q, keys, gates, ape, weights, mask, 4, return_index_scores
    )
    assert actual[0, -1].tolist() == [2, 3, 4, 5, -1, -1, -1]
    expected, expected_scores = _hf_padded_topk_reference(
        q, keys, gates, ape, weights, key_valid, topk=4
    )
    _assert_matches_hf_reference(actual, actual_scores, expected, expected_scores)


def _check_padded_selector(spans, mask_ndim, heads, return_index_scores, sequence=24):
    """Compare the explicit-mask selector with the HF padding-vector reference."""
    torch.manual_seed(578)
    batch, width = len(spans), 128
    # Positive operands prevent ReLU-created zero-score ties.
    q = torch.rand(sequence, batch, heads, width, device='cuda', dtype=torch.bfloat16)
    keys = torch.rand(sequence, batch, width, device='cuda', dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, width, device='cuda')
    weights = torch.rand(sequence, batch, heads, device='cuda') * (heads**-0.5)
    key_valid = torch.zeros(batch, sequence, dtype=torch.bool, device='cuda')
    for batch_index, (start, end) in enumerate(spans):
        key_valid[batch_index, start:end] = True
    mask = _padding_causal_mask(key_valid)
    if mask_ndim == 2:
        assert all(span == spans[0] for span in spans)
        mask = mask[0]
    actual_scores, actual = _run_hf_selector(
        q, keys, gates, ape, weights, mask, 8, return_index_scores
    )
    expected, expected_scores = _hf_padded_topk_reference(q, keys, gates, ape, weights, key_valid)
    _assert_matches_hf_reference(actual, actual_scores, expected, expected_scores)


# Offsets 0 and 4 are aligned controls; the others start pools between global boundaries.
# Spans ending before the sequence end also right-pad keys and query rows.
_PADDED_SPANS = {f'left{offset}': (offset, 24) for offset in (0, 1, 2, 3, 4, 5, 6, 7)}
_PADDED_SPANS.update({'left3_right5': (3, 19), 'left4_right6': (4, 18), 'left6_right1': (6, 23)})


@pytest.mark.parametrize('return_index_scores', [False, True])
@pytest.mark.parametrize('heads', [1, 32])
@pytest.mark.parametrize('mask_ndim', [2, 3])
@pytest.mark.parametrize('span', list(_PADDED_SPANS.values()), ids=list(_PADDED_SPANS))
def test_hf_selector_matches_reference_with_contiguous_padding(
    span, mask_ndim, heads, return_index_scores
):
    _check_padded_selector([span, span], mask_ndim, heads, return_index_scores)


@pytest.mark.parametrize('return_index_scores', [False, True])
@pytest.mark.parametrize('heads', [1, 32])
def test_hf_selector_matches_reference_with_per_sample_padding(heads, return_index_scores):
    """Different origins per sample, an all-padding sample and a sub-pool sample."""
    spans = [(1, 23), (6, 19), (0, 0), (21, 24), (8, 21)]
    _check_padded_selector(spans, 3, heads, return_index_scores)


@pytest.mark.parametrize(
    ('pool_size', 'topk', 'loss_coeff', 'match'),
    [
        (0, 16, 0, 'positive'),
        (4, None, 0, 'must be set'),
        (4, 0, 0, 'positive'),
        (4, 15, 0, 'divisible'),
        (4, 16, 0.01, 'not supported'),
    ],
)
def test_kpool_rejects_invalid_selector_contract(pool_size, topk, loss_coeff, match):
    with pytest.raises(ValueError, match=match):
        _validate_dsa_kpool_config(
            SimpleNamespace(
                dsa_indexer_kpool=pool_size,
                dsa_indexer_topk=topk,
                dsa_indexer_loss_coeff=loss_coeff,
            )
        )


@pytest.mark.parametrize('loss_coeff', [None, 0.0])
def test_kpool_allows_frozen_selector_without_auxiliary_loss(loss_coeff):
    _validate_dsa_kpool_config(
        SimpleNamespace(
            dsa_indexer_kpool=4, dsa_indexer_topk=2048, dsa_indexer_loss_coeff=loss_coeff
        )
    )


def test_kpool_segment_compression_matches_independent_torch_reference():
    # GLM53 indexer head width is 128; only the number of tokens is reduced.
    torch.manual_seed(571)
    lengths = (3, 9, 5)
    cu = torch.tensor([0, 3, 12, 17], dtype=torch.int32, device='cuda')
    keys = torch.randn(sum(lengths), 1, 128, device='cuda', dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    pooled, bases = _kpool_compress_keys_per_seg(keys, gates, ape, 4, cu)
    reference, reference_bases = [], []
    start = 0
    for length in lengths:
        for offset in range(0, length - 3, 4):
            base = start + offset
            probabilities = torch.softmax(gates[base : base + 4].float() + ape[:, None], dim=0)
            reference.append((probabilities * keys[base : base + 4].float()).sum(0))
            reference_bases.append(base)
        start += length
    expected = torch.stack(reference).to(torch.bfloat16)
    actual = pooled[: len(reference)]
    assert bases[: len(reference)].tolist() == reference_bases
    assert torch.all(bases[len(reference) :] == -1)
    lhs, rhs = actual.flatten().double(), expected.flatten().double()
    assert F.cosine_similarity(lhs[None], rhs[None]).item() > 0.999
    assert (2 * (lhs * rhs).sum() / (lhs.square().sum() + rhs.square().sum())).item() > 0.999


def test_kpool_selection_without_scores_does_not_retain_autograd_graph():
    torch.manual_seed(572)
    q = torch.randn(7, 1, 2, 128, device='cuda', requires_grad=True)
    k = torch.randn(7, 1, 128, device='cuda', requires_grad=True)
    weights = torch.randn(7, 1, 2, device='cuda', requires_grad=True)
    positions = torch.arange(1, 8, device='cuda')
    with torch.no_grad():
        scores, indices = fused_qk_topk_kpool(
            q,
            k,
            weights,
            2048,
            4,
            gate_score=torch.zeros_like(k),
            ape=torch.zeros(4, 128, device='cuda'),
            varlen_starts=torch.zeros_like(positions),
            varlen_ends=positions,
            return_index_scores=False,
        )
    assert scores is None
    assert not indices.requires_grad
    assert indices.shape == (1, 7, 2051)
    assert q.grad is None and k.grad is None and weights.grad is None


def test_absorbed_sparse_attention_empty_rows_have_finite_zero_gradients():
    torch.manual_seed(573)
    query = torch.randn(2, 1, 2, 8, device='cuda', requires_grad=True)
    key = torch.randn(3, 1, 1, 8, device='cuda', requires_grad=True)
    indices = torch.tensor([[[-1, -1, -1], [0, 1, -1]]], device='cuda', dtype=torch.int32)
    output = _unfused_absorbed_dsa_fn(query, key, indices, 1 / 8**0.5, v_channels=8)
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    output.square().sum().backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    torch.testing.assert_close(query.grad[0], torch.zeros_like(query.grad[0]))


def _cpu_kpool_selector_inputs():
    torch.manual_seed(579)
    sequence, batch, heads, width, pool_size = 12, 2, 4, 16, 4
    q = torch.randn(sequence, batch, heads, width, requires_grad=True)
    k_pooled = torch.randn(sequence // pool_size, batch, width, requires_grad=True)
    weights = (torch.rand(sequence, batch, heads) * heads**-0.5).requires_grad_()
    return q, k_pooled, weights


@pytest.mark.parametrize('hf_scoring', [False, True])
def test_kpool_chunked_selection_records_no_autograd_state(hf_scoring):
    """Chunked KPool selection runs under no_grad in any caller grad mode.

    DSAIndexer.forward runs the selector in the caller's grad mode; without no_grad,
    grad mode would make it save operand and score tensors for a backward that never runs.
    """
    q, k_pooled, weights = _cpu_kpool_selector_inputs()
    kwargs = dict(
        index_topk=8,
        pool_size=4,
        pool_exists=None,
        mask=None,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        pool_token_base=torch.arange(k_pooled.size(0)) * 4,
        sk=q.size(0),
        use_relu=True,
        hf_scoring=hf_scoring,
    )
    saved = []

    def pack_hook(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with (
        torch.enable_grad(),
        torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor),
    ):
        selected = _kpool_topk_from_score_chunks(q=q, k_pooled=k_pooled, weights=weights, **kwargs)
        assert saved == []
        # Positive control: in this context the probe records autograd state for plain
        # differentiable scoring arithmetic.
        torch.einsum('sbhd,tbd->sbht', q, k_pooled)
        assert saved
    # The selection does not depend on the caller's grad mode.
    with torch.no_grad():
        expected = _kpool_topk_from_score_chunks(q=q, k_pooled=k_pooled, weights=weights, **kwargs)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)


def test_hf_kpool_scores_follow_caller_grad_mode_like_legacy_scores():
    """HF KPool scores follow the caller's grad mode, like the legacy index scores."""
    q, k_pooled, weights = _cpu_kpool_selector_inputs()
    with torch.enable_grad():
        hf_scores = _compute_kpool_hf_scores(q, k_pooled, weights)
        legacy_scores = _compute_index_scores(q, weights, k_pooled, use_relu=True)
    assert hf_scores.requires_grad
    assert legacy_scores.requires_grad
    hf_scores.sum().backward()
    for tensor in (q, k_pooled, weights):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    with torch.no_grad():
        assert not _compute_kpool_hf_scores(q, k_pooled, weights).requires_grad
        assert not _compute_index_scores(q, weights, k_pooled, use_relu=True).requires_grad
