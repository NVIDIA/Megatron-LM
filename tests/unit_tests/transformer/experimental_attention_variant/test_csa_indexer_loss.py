# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for routing compressed-THD CSA through ordinary DSA loss."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.csa import (
    _build_compressed_thd_indexer_metadata,
    _compressed_thd_topk_to_local,
    _compute_unfused_csa_non_compressed_lse,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    build_flat_topk_idxs,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    FusedDSAIndexerLoss,
    _normalize_indexer_teacher_target,
    fused_qk_topk_naive,
)
from tests.unit_tests.test_utilities import Utils


def _causal_compressed_mask(sq, sk, ratio, device):
    columns = torch.arange(sk, device=device).unsqueeze(0).expand(sq, -1)
    positions = torch.arange(1, sq + 1, device=device).unsqueeze(1)
    return torch.where(columns >= positions // ratio, float('-inf'), 0.0).unsqueeze(0)


class _SingleRankGroup:
    @staticmethod
    def size():
        return 1


class _SingleRankPGCollection:
    tp = _SingleRankGroup()


def _make_legacy_abi_case(seed=23):
    generator = torch.Generator().manual_seed(seed)
    sq, sk, batch = 8, 2, 1
    index_heads, index_dim = 2, 4
    attention_heads, attention_dim = 2, 5
    q = torch.randn(sq, batch, index_heads, index_dim, generator=generator)
    weights = torch.randn(sq, batch, index_heads, generator=generator)
    k = torch.randn(sk, batch, index_dim, generator=generator)
    query = torch.randn(sq, batch, attention_heads, attention_dim, generator=generator)
    key = torch.randn(sk, batch, attention_heads, attention_dim, generator=generator)
    mask = _causal_compressed_mask(sq, sk, ratio=4, device='cpu')
    return q, weights, k, query, key, mask


def _run_legacy_abi_case(num_inputs):
    q, weights, k, query, key, mask = _make_legacy_abi_case()
    q = q.requires_grad_(True)
    weights = weights.requires_grad_(True)
    k = k.requires_grad_(True)
    args = [
        q,
        weights,
        k,
        query,
        key,
        key.shape[-1] ** -0.5,
        2,
        0.7,
        mask,
        True,
        _SingleRankPGCollection(),
        None,
        None,
        None,
        None,
        False,
        True,
        None,
    ]
    topk_indices, loss = FusedDSAIndexerLoss.apply(*args[:num_inputs])
    loss.backward()
    return topk_indices, loss.detach(), q.grad, weights.grad, k.grad


def _run_compressed_thd_topk(q, k, weights, topk, cu_q, cu_comp, ratio):
    starts, ends, valid_rows, offsets = _build_compressed_thd_indexer_metadata(
        cu_q, cu_comp, total_q=q.shape[0], ratio=ratio
    )
    _, topk_global = fused_qk_topk_naive(
        q.unsqueeze(1),
        k.unsqueeze(1),
        weights.unsqueeze(1),
        topk,
        varlen_starts=starts,
        varlen_ends=ends,
    )
    return _compressed_thd_topk_to_local(topk_global, offsets), valid_rows


def _run_compressed_thd_loss(
    q,
    weights,
    k,
    query,
    key,
    *,
    topk,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    pg_collection,
    calculate_per_token_loss,
    cu_q,
    cu_comp,
    ratio,
):
    starts, ends, valid_rows, offsets = _build_compressed_thd_indexer_metadata(
        cu_q, cu_comp, total_q=q.shape[0], ratio=ratio
    )
    topk_global, loss = FusedDSAIndexerLoss.apply(
        q.unsqueeze(1),
        weights.unsqueeze(1),
        k.unsqueeze(1),
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        None,
        sparse_loss,
        pg_collection,
        starts,
        ends,
        None,
        valid_rows,
        calculate_per_token_loss,
        True,
    )
    return _compressed_thd_topk_to_local(topk_global, offsets), loss


@pytest.mark.parametrize("num_inputs", [14, 17, 18])
def test_custom_autograd_backward_matches_legacy_arity(num_inputs):
    reference = _run_legacy_abi_case(18)
    actual = _run_legacy_abi_case(num_inputs)
    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


def test_corrected_teacher_normalizes_tiny_positive_mass():
    target = torch.tensor([[[1.0e-30, 2.0e-30]]], dtype=torch.float32)

    corrected = _normalize_indexer_teacher_target(
        target, torch.zeros((1, 1, 1), dtype=torch.float32)
    )
    legacy = _normalize_indexer_teacher_target(target, None)

    torch.testing.assert_close(
        corrected, torch.tensor([[[1.0 / 3.0, 2.0 / 3.0]]]), rtol=1e-6, atol=0
    )
    assert corrected.sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(
        legacy, target / torch.tensor(1.0e-10, dtype=torch.float32), rtol=0, atol=0
    )


def test_non_compressed_lse_chunking_and_all_invalid_window():
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(5, 2, 3, 4, generator=generator)
    kv = torch.randn(5, 2, 4, generator=generator)
    sink = torch.randn(3, generator=generator)
    window_indices = torch.tensor(
        [[[0, -1], [0, 1], [1, 2], [2, 3], [3, 4]], [[0, -1], [0, 1], [1, 2], [2, 3], [3, 4]]]
    )

    row_chunks = _compute_unfused_csa_non_compressed_lse(
        query, kv, sink, window_indices, softmax_scale=0.5, chunk_size=1
    )
    full_chunk = _compute_unfused_csa_non_compressed_lse(
        query, kv, sink, window_indices, softmax_scale=0.5, chunk_size=10
    )
    torch.testing.assert_close(row_chunks, full_chunk, rtol=0, atol=0)

    all_invalid = torch.full_like(window_indices, -1)
    sink_only = _compute_unfused_csa_non_compressed_lse(
        query, kv, sink, all_invalid, softmax_scale=0.5
    )
    expected = sink.view(1, -1, 1).expand(2, -1, 5)
    torch.testing.assert_close(sink_only, expected, rtol=0, atol=0)


def test_thd_non_compressed_lse_uses_global_interleaved_kv_offsets():
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            [[1.0, 1.0], [-1.0, 0.5]],
            [[0.25, -0.5], [0.75, 1.0]],
        ]
    )
    kv_full_thd = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [100.0, 100.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [200.0, 200.0],
            [300.0, 300.0],
        ]
    )
    local_window = torch.tensor([[0, -1], [0, 1], [0, -1], [0, 1]])
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_kv_full = torch.tensor([0, 3, 7], dtype=torch.int32)
    global_window, _ = build_flat_topk_idxs(
        local_window, batch_size=-1, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv_full
    )
    assert torch.equal(
        global_window, torch.tensor([[0, -1], [0, 1], [3, -1], [3, 4]], dtype=torch.int32)
    )

    sink = torch.tensor([0.25, -0.5])
    actual = _compute_unfused_csa_non_compressed_lse(
        query, kv_full_thd, sink, global_window, softmax_scale=0.5, chunk_size=1
    )
    gathered = kv_full_thd.index_select(0, global_window.clamp_min(0).reshape(-1)).reshape(4, 2, 2)
    logits = torch.einsum('rhd,rkd->rhk', query.float(), gathered.float()) * 0.5
    logits = logits.masked_fill((global_window < 0).unsqueeze(1), float('-inf'))
    expected = (
        torch.logaddexp(torch.logsumexp(logits, dim=-1), sink.view(1, -1))
        .transpose(0, 1)
        .unsqueeze(0)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestCompressedThdMetadata:
    def test_two_segments_and_padding_map_exactly(self):
        cu_q = torch.tensor([0, 8, 13], dtype=torch.int32)
        cu_comp = torch.tensor([0, 2, 3], dtype=torch.int32)

        starts, ends, valid_rows, offsets = _build_compressed_thd_indexer_metadata(
            cu_q, cu_comp, total_q=15, ratio=4
        )

        assert torch.equal(starts, torch.tensor([0] * 8 + [2] * 5 + [0, 0]))
        assert torch.equal(ends, torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0]))
        assert torch.equal(valid_rows, torch.tensor([True] * 13 + [False, False]))
        assert torch.equal(offsets, starts)

    def test_per_segment_padding_uses_physical_offsets_and_real_lengths(self):
        physical_cu_q = torch.tensor([0, 12, 24], dtype=torch.int32)
        real_cu_q = torch.tensor([0, 9, 20], dtype=torch.int32)
        physical_cu_comp = torch.tensor([0, 3, 6], dtype=torch.int32)

        starts, ends, valid_rows, offsets = _build_compressed_thd_indexer_metadata(
            physical_cu_q, physical_cu_comp, total_q=24, ratio=4, cu_seqlens_q_unpadded=real_cu_q
        )

        expected_valid = torch.tensor([True] * 9 + [False] * 3 + [True] * 11 + [False])
        assert torch.equal(valid_rows, expected_valid)
        assert torch.equal(starts[:9], torch.zeros(9, dtype=torch.int64))
        assert torch.equal(starts[9:12], torch.zeros(3, dtype=torch.int64))
        assert torch.equal(starts[12:23], torch.full((11,), 3, dtype=torch.int64))
        assert starts[23] == 0
        assert torch.equal(offsets, starts)
        assert ends[8] == 2
        assert torch.equal(ends[9:12], torch.zeros(3, dtype=torch.int64))
        assert ends[22] == 5
        assert ends[23] == 0

    @pytest.mark.parametrize(
        "cu_q,cu_comp,ratio,match",
        [
            (torch.tensor([0, 4]), torch.tensor([0, 1]), 0, "ratio must be positive"),
            (torch.tensor([0, 4]), torch.tensor([0, 1, 2]), 4, "same non-empty"),
        ],
    )
    def test_invalid_metadata_raises(self, cu_q, cu_comp, ratio, match):
        with pytest.raises(ValueError, match=match):
            _build_compressed_thd_indexer_metadata(cu_q, cu_comp, total_q=4, ratio=ratio)


class TestCompressedThdDsaLoss:
    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp']
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('sparse_loss', [False, True], ids=['dense_loss', 'sparse_loss'])
    @pytest.mark.parametrize(
        'calculate_per_token_loss', [False, True], ids=['row_mean', 'token_sum']
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_segment_matches_sbhd_value_topk_and_gradients(
        self, sparse_loss, calculate_per_token_loss
    ):
        torch.manual_seed(0)
        sq, ratio = 32, 4
        sk = sq // ratio
        num_heads, head_dim = 4, 64
        idx_nh, idx_hd = 4, 32
        topk = 4
        softmax_scale = head_dim**-0.5
        loss_coeff = 0.5
        device = 'cuda'

        q_sbhd = torch.randn(sq, 1, idx_nh, idx_hd, device=device).requires_grad_(True)
        w_sbhd = torch.randn(sq, 1, idx_nh, device=device).requires_grad_(True)
        k_sbhd = torch.randn(sk, 1, idx_hd, device=device).requires_grad_(True)
        query_sbhd = torch.randn(sq, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        key_sbhd = torch.randn(sk, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)

        topk_sbhd, loss_sbhd = FusedDSAIndexerLoss.apply(
            q_sbhd,
            w_sbhd,
            k_sbhd,
            query_sbhd,
            key_sbhd,
            softmax_scale,
            topk,
            loss_coeff,
            _causal_compressed_mask(sq, sk, ratio, device),
            sparse_loss,
            self.pg_collection,
            None,
            None,
            None,
            None,
            calculate_per_token_loss,
            True,
        )
        loss_sbhd.backward()

        q_thd = q_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        w_thd = w_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        k_thd = k_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_comp = torch.tensor([0, sk], dtype=torch.int32, device=device)

        topk_thd, loss_thd = _run_compressed_thd_loss(
            q_thd,
            w_thd,
            k_thd,
            query_sbhd.squeeze(1),
            key_sbhd.squeeze(1),
            topk=topk,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            sparse_loss=sparse_loss,
            pg_collection=self.pg_collection,
            calculate_per_token_loss=calculate_per_token_loss,
            cu_q=cu_q,
            cu_comp=cu_comp,
            ratio=ratio,
        )
        loss_thd.backward()

        torch.testing.assert_close(loss_thd, loss_sbhd, rtol=1e-5, atol=1e-5)
        assert torch.equal(topk_thd, topk_sbhd.squeeze(0))
        torch.testing.assert_close(q_thd.grad, q_sbhd.grad.squeeze(1), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(w_thd.grad, w_sbhd.grad.squeeze(1), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_thd.grad, k_sbhd.grad.squeeze(1), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('calculate_per_token_loss', [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multisegment_zero_compressed_segment_normalization(self, calculate_per_token_loss):
        torch.manual_seed(7)
        seg_q_lens = [2, 8]
        seg_comp_lens = [0, 2]
        total_q, total_comp = sum(seg_q_lens), sum(seg_comp_lens)
        ratio = 4
        idx_nh, idx_hd = 4, 32
        num_heads, head_dim = 4, 64
        topk = 2
        softmax_scale = head_dim**-0.5
        loss_coeff = 0.5
        device = 'cuda'

        q = torch.randn(total_q, idx_nh, idx_hd, device=device)
        w = torch.randn(total_q, idx_nh, device=device)
        k = torch.randn(total_comp, idx_hd, device=device)
        query = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        key = torch.randn(total_comp, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        cu_q = torch.tensor([0, 2, 10], dtype=torch.int32, device=device)
        cu_comp = torch.tensor([0, 0, 2], dtype=torch.int32, device=device)

        topk_thd, loss_thd = _run_compressed_thd_loss(
            q,
            w,
            k,
            query,
            key,
            topk=topk,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            sparse_loss=False,
            pg_collection=self.pg_collection,
            calculate_per_token_loss=calculate_per_token_loss,
            cu_q=cu_q,
            cu_comp=cu_comp,
            ratio=ratio,
        )
        assert (topk_thd[: seg_q_lens[0]] == -1).all()

        segment_losses = []
        for segment, (sq, sk) in enumerate(zip(seg_q_lens, seg_comp_lens)):
            if sk == 0:
                continue
            qs, qe = int(cu_q[segment].item()), int(cu_q[segment + 1].item())
            ks, ke = int(cu_comp[segment].item()), int(cu_comp[segment + 1].item())
            _, loss_segment = FusedDSAIndexerLoss.apply(
                q[qs:qe].unsqueeze(1),
                w[qs:qe].unsqueeze(1),
                k[ks:ke].unsqueeze(1),
                query[qs:qe],
                key[ks:ke],
                softmax_scale,
                topk,
                loss_coeff,
                _causal_compressed_mask(sq, sk, ratio, device),
                False,
                self.pg_collection,
                None,
                None,
                None,
                None,
                calculate_per_token_loss,
                True,
            )
            segment_losses.append(loss_segment if calculate_per_token_loss else loss_segment * sq)

        expected = torch.stack(segment_losses).sum()
        if not calculate_per_token_loss:
            expected = expected / total_q
        torch.testing.assert_close(loss_thd, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('sparse_loss', [False, True])
    @pytest.mark.parametrize('calculate_per_token_loss', [False, True])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_static_padding_is_excluded_from_loss_and_gradients(
        self, sparse_loss, calculate_per_token_loss
    ):
        torch.manual_seed(11)
        real_q, padded_q = 13, 16
        real_k, padded_k = 3, 5
        ratio, topk = 4, 3
        idx_nh, idx_hd = 2, 16
        num_heads, head_dim = 2, 32
        device = 'cuda'

        q_init = torch.randn(padded_q, idx_nh, idx_hd, device=device)
        w_init = torch.randn(padded_q, idx_nh, device=device)
        k_init = torch.randn(padded_k, idx_hd, device=device)
        query = torch.randn(padded_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        key = torch.randn(padded_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        cu_q = torch.tensor([0, 8, real_q], dtype=torch.int32, device=device)
        cu_comp = torch.tensor([0, 2, real_k], dtype=torch.int32, device=device)

        q_padded = q_init.clone().requires_grad_(True)
        w_padded = w_init.clone().requires_grad_(True)
        k_padded = k_init.clone().requires_grad_(True)
        topk_padded, loss_padded = _run_compressed_thd_loss(
            q_padded,
            w_padded,
            k_padded,
            query,
            key,
            topk=topk,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.5,
            sparse_loss=sparse_loss,
            pg_collection=self.pg_collection,
            calculate_per_token_loss=calculate_per_token_loss,
            cu_q=cu_q,
            cu_comp=cu_comp,
            ratio=ratio,
        )
        loss_padded.backward()

        # Keep the padded shapes identical so the oracle uses the same GEMM
        # accumulation shape. Only perturb values that metadata marks as
        # padding; none may affect loss, top-k, or real-row gradients.
        q_alt_init = q_init.clone()
        w_alt_init = w_init.clone()
        k_alt_init = k_init.clone()
        query_alt = query.clone()
        key_alt = key.clone()
        q_alt_init[real_q:] = torch.randn_like(q_alt_init[real_q:]) * 100
        w_alt_init[real_q:] = torch.randn_like(w_alt_init[real_q:]) * 100
        k_alt_init[real_k:] = torch.randn_like(k_alt_init[real_k:]) * 100
        query_alt[real_q:] = torch.randn_like(query_alt[real_q:]) * 100
        key_alt[real_k:] = torch.randn_like(key_alt[real_k:]) * 100

        q_alt = q_alt_init.requires_grad_(True)
        w_alt = w_alt_init.requires_grad_(True)
        k_alt = k_alt_init.requires_grad_(True)
        topk_alt, loss_alt = _run_compressed_thd_loss(
            q_alt,
            w_alt,
            k_alt,
            query_alt,
            key_alt,
            topk=topk,
            softmax_scale=head_dim**-0.5,
            loss_coeff=0.5,
            sparse_loss=sparse_loss,
            pg_collection=self.pg_collection,
            calculate_per_token_loss=calculate_per_token_loss,
            cu_q=cu_q,
            cu_comp=cu_comp,
            ratio=ratio,
        )
        loss_alt.backward()

        torch.testing.assert_close(loss_padded, loss_alt, rtol=0, atol=0)
        torch.testing.assert_close(topk_padded, topk_alt, rtol=0, atol=0)
        assert (topk_padded[real_q:] == -1).all()
        torch.testing.assert_close(q_padded.grad[:real_q], q_alt.grad[:real_q], rtol=0, atol=0)
        torch.testing.assert_close(w_padded.grad[:real_q], w_alt.grad[:real_q], rtol=0, atol=0)
        torch.testing.assert_close(k_padded.grad[:real_k], k_alt.grad[:real_k], rtol=0, atol=0)
        assert torch.count_nonzero(q_padded.grad[real_q:]) == 0
        assert torch.count_nonzero(w_padded.grad[real_q:]) == 0
        assert torch.count_nonzero(k_padded.grad[real_k:]) == 0
        assert torch.count_nonzero(q_alt.grad[real_q:]) == 0
        assert torch.count_nonzero(w_alt.grad[real_q:]) == 0
        assert torch.count_nonzero(k_alt.grad[real_k:]) == 0


class TestCompressedThdTopk:
    def test_global_to_local_mapping_never_crosses_segments(self):
        torch.manual_seed(3)
        seg_q_lens = [8, 4]
        seg_comp_lens = [2, 1]
        total_q, total_comp = sum(seg_q_lens), sum(seg_comp_lens)
        ratio, requested_topk = 4, 4
        effective_topk = min(requested_topk, total_comp)
        q = torch.randn(total_q, 2, 16)
        k = torch.randn(total_comp, 16)
        weights = torch.randn(total_q, 2)
        cu_q = torch.tensor([0, 8, 12], dtype=torch.int32)
        cu_comp = torch.tensor([0, 2, 3], dtype=torch.int32)

        topk, valid_rows = _run_compressed_thd_topk(
            q, k, weights, effective_topk, cu_q, cu_comp, ratio
        )

        assert topk.shape == (total_q, effective_topk)
        assert topk.dtype == torch.int64
        assert valid_rows.all()
        for start, end, capacity in [(0, 8, 2), (8, 12, 1)]:
            valid = topk[start:end][topk[start:end] >= 0]
            assert (valid < capacity).all()
        for row, position in enumerate(list(range(8)) + list(range(4))):
            valid_count = min((position + 1) // ratio, seg_comp_lens[0 if row < 8 else 1])
            assert (topk[row, valid_count:] == -1).all()
