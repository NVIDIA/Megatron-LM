# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
    FusedDSAIndexerLoss,
    _build_causal_mask_from_positions,
    _build_fused_indexer_varlen_bounds,
    _compute_index_scores,
    _fused_qk_topk_lighting,
    _fused_qk_topk_lighting_with_streaming_sparse_kl,
    _generate_varlen_mask_params,
    _get_cp_positions_from_layout,
    _scatter_topk_into_index_mask,
    _unfused_absorbed_dsa_fn,
    compute_dsa_indexer_loss,
    compute_dsa_indexer_loss_topk_sparse,
    fused_qk_topk_naive,
    rotate_activation,
    unfused_dsa_fn,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed.

    This is a simple identity-like transformation that preserves shape and applies scaling.
    """
    return x * scale


def _build_packed_causal_mask_for_test(
    query_idx: torch.Tensor, key_idx: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Build packed-sequence causal mask for tests."""
    query_idx = query_idx.to(dtype=torch.int64)
    key_idx = key_idx.to(dtype=torch.int64)
    cu_seqlens = cu_seqlens.to(device=query_idx.device, dtype=torch.int64)

    boundaries = cu_seqlens[1:]
    query_seq_id = torch.searchsorted(boundaries, query_idx, right=True)
    key_seq_id = torch.searchsorted(boundaries, key_idx, right=True)
    valid = (query_seq_id.unsqueeze(-1) == key_seq_id.unsqueeze(0)) & (
        key_idx.unsqueeze(0) <= query_idx.unsqueeze(-1)
    )
    mask = torch.zeros(
        (query_idx.numel(), key_idx.numel()), dtype=torch.float32, device=query_idx.device
    )
    mask.masked_fill_(~valid, float("-inf"))
    return mask


def _fake_lighting_indexer_for_test(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_w: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    index_topk: int,
    topk_indices: torch.Tensor | None = None,
):
    """Reference fake indexer for testing fused batched loop plumbing."""
    del topk_indices

    # [sq, h, d] @ [sk, d]^T -> [sq, h, sk]
    logits = torch.einsum("qhd,kd->qhk", index_q.float(), index_k.float())
    logits = torch.relu(logits) * index_w.float().unsqueeze(-1)
    logits = logits.sum(dim=1)  # [sq, sk]

    key_pos = torch.arange(index_k.size(0), dtype=torch.int64, device=logits.device)
    valid = (key_pos.unsqueeze(0) >= starts.to(torch.int64).unsqueeze(-1)) & (
        key_pos.unsqueeze(0) < ends.to(torch.int64).unsqueeze(-1)
    )
    logits = logits.masked_fill(~valid, float("-inf"))

    topk_k = min(index_topk, logits.size(-1))
    topk_scores, topk_idx = torch.topk(logits, topk_k, dim=-1)
    topk_idx = topk_idx.to(torch.int32)
    topk_idx = topk_idx.masked_fill(topk_scores == float("-inf"), -1)
    return topk_scores, topk_idx


def _fake_fused_scores_indices_for_test(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    index_topk: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fused-style batched/chunked top-k scores+indices with fake indexer."""
    sq, b = q.size(0), q.size(1)
    scores_out, idx_out = None, None
    for bi in range(b):
        for s0 in range(0, sq, block_size):
            s1 = min(s0 + block_size, sq)
            scores_chunk, idx_chunk = _fake_lighting_indexer_for_test(
                q[:, bi][s0:s1],
                k[:, bi],
                weights[:, bi][s0:s1],
                starts[s0:s1],
                ends[s0:s1],
                index_topk,
            )
            if scores_out is None:
                scores_out = torch.empty(
                    (b, sq, scores_chunk.size(-1)),
                    dtype=scores_chunk.dtype,
                    device=scores_chunk.device,
                )
            if idx_out is None:
                idx_out = torch.empty(
                    (b, sq, idx_chunk.size(-1)), dtype=idx_chunk.dtype, device=idx_chunk.device
                )
            scores_out[bi, s0:s1].copy_(scores_chunk)
            idx_out[bi, s0:s1].copy_(idx_chunk)
    assert scores_out is not None and idx_out is not None
    return scores_out, idx_out


class _FakeTPGroup:
    def size(self) -> int:
        return 1


class _FakeCPGroup:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeTPGroup()


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in dsa module if not installed."""
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            mock_hadamard_transform,
        ):
            yield
    else:
        yield


class TestDSACPPositionHelpers:
    """Test helper utilities used for DSAttention context-parallel masking."""

    def test_allgather_layout_positions(self):
        """Allgather CP layout should map to contiguous query and global key positions."""
        query_pos, key_pos = _get_cp_positions_from_layout(
            sq=4, skv=8, cp_size=2, cp_rank=1, cp_comm_type="allgather", device=torch.device("cpu")
        )
        assert query_pos.tolist() == [4, 5, 6, 7]
        assert key_pos.tolist() == list(range(8))

    def test_position_based_causal_mask(self):
        """Position-based causal mask should mask keys with strictly larger positions."""
        query_pos = torch.tensor([0, 2], dtype=torch.int64)
        key_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        mask = _build_causal_mask_from_positions(query_pos, key_pos)
        expected = torch.tensor(
            [[0.0, float("-inf"), float("-inf"), float("-inf")], [0.0, 0.0, 0.0, float("-inf")]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(mask, expected, rtol=0, atol=0)

    def test_packed_position_based_causal_mask(self):
        """Packed causal mask should block cross-sequence attention using cu_seqlens boundaries."""
        # Two packed sequences: [0,1,2] and [3,4]
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        query_idx = torch.tensor([1, 3, 4], dtype=torch.int64)
        key_idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        mask = _build_packed_causal_mask_for_test(query_idx, key_idx, cu_seqlens)
        expected = torch.tensor(
            [
                [0.0, 0.0, float("-inf"), float("-inf"), float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), 0.0, float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(mask, expected, rtol=0, atol=0)

    def test_topk_uses_key_length(self):
        """Top-k selection should be bounded by key length, not query length."""
        sq, skv, bsz, nheads, dim = 4, 7, 1, 2, 8
        topk = 6
        q = torch.randn(sq, bsz, nheads, dim, dtype=torch.float32)
        k = torch.randn(skv, bsz, dim, dtype=torch.float32)
        weights = torch.randn(sq, bsz, nheads, dtype=torch.float32)

        _, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask=None)
        assert topk_indices.shape == (bsz, sq, topk)

    def test_cp_packed_varlen_end_to_end_matches_dense_mask(self):
        """CP+THD multi-sequence varlen path should match dense packed mask end-to-end."""
        # Simulate cp_size=2 allgather layout with local query chunk and global keys.
        cp_size, cp_rank = 2, 1
        sq, skv = 4, 8
        bsz, nheads, dim, vdim = 1, 2, 8, 6
        topk = 4
        softmax_scale = dim**-0.5

        # Three packed sequences in global stream: [0,1,2], [3,4], [5,6,7]
        cu_seqlens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        query_idx, key_idx = _get_cp_positions_from_layout(
            sq=sq,
            skv=skv,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_comm_type="allgather",
            device=torch.device("cpu"),
        )

        # Build varlen starts/ends for local query rows.
        starts_all, ends_all = _generate_varlen_mask_params(cu_seqlens.to(torch.int64))
        starts = starts_all.index_select(0, query_idx)
        ends = ends_all.index_select(0, query_idx)

        q = torch.randn(sq, bsz, nheads, dim, dtype=torch.float32)
        k_for_index = torch.randn(skv, bsz, dim, dtype=torch.float32)
        weights = torch.randn(sq, bsz, nheads, dtype=torch.float32)
        query = torch.randn(sq, bsz, nheads, dim, dtype=torch.float32)
        key = torch.randn(skv, bsz, nheads, dim, dtype=torch.float32)
        value = torch.randn(skv, bsz, nheads, vdim, dtype=torch.float32)

        dense_mask = _build_packed_causal_mask_for_test(query_idx, key_idx, cu_seqlens)
        _, dense_idx = fused_qk_topk_naive(q, k_for_index, weights, topk, mask=dense_mask)
        out_dense = unfused_dsa_fn(query, key, value, dense_idx, softmax_scale, mask=dense_mask)

        _, varlen_idx = fused_qk_topk_naive(
            q,
            k_for_index,
            weights,
            topk,
            mask=None,
            varlen_starts=starts,
            varlen_ends=ends,
            key_positions=key_idx,
        )
        out_varlen = unfused_dsa_fn(
            query,
            key,
            value,
            varlen_idx,
            softmax_scale,
            mask=None,
            varlen_starts=starts,
            varlen_ends=ends,
            key_positions=key_idx,
        )

        torch.testing.assert_close(out_varlen, out_dense, rtol=0, atol=0)

    def test_cp_packed_varlen_uneven_rank_lengths_matches_dense_mask(self, monkeypatch):
        """CP+THD varlen path should match dense mask under uneven per-rank query lengths."""
        # Simulate cp_size=2, cp_rank=1, local query lengths [3, 5].
        cp_size, cp_rank = 2, 1
        local_lengths = [3, 5]
        sq, skv = local_lengths[cp_rank], sum(local_lengths)
        bsz, nheads, dim, vdim = 1, 2, 8, 6
        topk = 4
        softmax_scale = dim**-0.5

        fake_cp_group = _FakeCPGroup(cp_size)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def _fake_all_gather(out, local_len, group=None):
            del local_len, group
            for i, tensor in enumerate(out):
                tensor.copy_(
                    torch.tensor([local_lengths[i]], dtype=tensor.dtype, device=tensor.device)
                )

        monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)

        # Packed global stream has three sequences: [0,1], [2,3,4], [5,6,7]
        cu_seqlens = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
        query_idx, key_idx = _get_cp_positions_from_layout(
            sq=sq,
            skv=skv,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_comm_type="allgather",
            device=torch.device("cpu"),
            cp_group=fake_cp_group,
        )
        assert query_idx.tolist() == [3, 4, 5, 6, 7]

        starts_all, ends_all = _generate_varlen_mask_params(cu_seqlens.to(torch.int64))
        starts = starts_all.index_select(0, query_idx)
        ends = ends_all.index_select(0, query_idx)

        q = torch.randn(sq, bsz, nheads, dim, dtype=torch.float32)
        k_for_index = torch.randn(skv, bsz, dim, dtype=torch.float32)
        weights = torch.randn(sq, bsz, nheads, dtype=torch.float32)
        query = torch.randn(sq, bsz, nheads, dim, dtype=torch.float32)
        key = torch.randn(skv, bsz, nheads, dim, dtype=torch.float32)
        value = torch.randn(skv, bsz, nheads, vdim, dtype=torch.float32)

        dense_mask = _build_packed_causal_mask_for_test(query_idx, key_idx, cu_seqlens)
        _, dense_idx = fused_qk_topk_naive(q, k_for_index, weights, topk, mask=dense_mask)
        out_dense = unfused_dsa_fn(query, key, value, dense_idx, softmax_scale, mask=dense_mask)

        _, varlen_idx = fused_qk_topk_naive(
            q,
            k_for_index,
            weights,
            topk,
            mask=None,
            varlen_starts=starts,
            varlen_ends=ends,
            key_positions=key_idx,
        )
        out_varlen = unfused_dsa_fn(
            query,
            key,
            value,
            varlen_idx,
            softmax_scale,
            mask=None,
            varlen_starts=starts,
            varlen_ends=ends,
            key_positions=key_idx,
        )

        torch.testing.assert_close(out_varlen, out_dense, rtol=0, atol=0)

    def test_fused_topk_batched_loop_matches_reference(self):
        """Fused batched/chunked top-k loop should match per-batch reference outputs."""
        sq, skv, bsz, heads, dim = 9, 13, 3, 4, 8
        topk = 5
        block_size = 4

        q = torch.randn(sq, bsz, heads, dim, dtype=torch.float32)
        k = torch.randn(skv, bsz, dim, dtype=torch.float32)
        weights = torch.randn(sq, bsz, heads, dtype=torch.float32)
        starts = torch.zeros(sq, dtype=torch.int32)
        ends = torch.arange(1, sq + 1, dtype=torch.int32).clamp_max(skv)

        expected = []
        for bi in range(bsz):
            _, ref_idx = _fake_lighting_indexer_for_test(
                q[:, bi], k[:, bi], weights[:, bi], starts, ends, topk
            )
            expected.append(ref_idx)
        expected = torch.stack(expected, dim=0)

        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.lighting_indexer",
            _fake_lighting_indexer_for_test,
        ):
            got = _fused_qk_topk_lighting(
                q=q,
                k=k,
                weights=weights,
                index_topk=topk,
                starts=starts,
                ends=ends,
                block_size=block_size,
            )

        assert got is not None
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)

    def test_fused_streaming_sparse_kl_matches_reference(self):
        """Streaming fused sparse-KL path should match reference top-k sparse KL."""
        sq, skv, bsz, heads, dim = 10, 12, 2, 4, 8
        topk = 6
        block_size = 4
        softmax_scale = dim**-0.5

        q = torch.randn(sq, bsz, heads, dim, dtype=torch.float32)
        k = torch.randn(skv, bsz, dim, dtype=torch.float32)
        weights = torch.randn(sq, bsz, heads, dtype=torch.float32)
        # MQA key for target attention distribution.
        query = torch.randn(sq, bsz, heads, dim, dtype=torch.float32)
        key = torch.randn(skv, bsz, 1, dim, dtype=torch.float32)
        starts = torch.zeros(sq, dtype=torch.int32)
        ends = torch.arange(1, sq + 1, dtype=torch.int32).clamp_max(skv)
        fake_pg = _FakePGCollection()

        ref_scores, ref_idx = _fake_fused_scores_indices_for_test(
            q, k, weights, starts, ends, topk, block_size
        )
        ref_loss = compute_dsa_indexer_loss_topk_sparse(
            index_topk_scores=ref_scores,
            topk_indices=ref_idx,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            pg_collection=fake_pg,
        )

        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.lighting_indexer",
            _fake_lighting_indexer_for_test,
        ):
            fused_out = _fused_qk_topk_lighting_with_streaming_sparse_kl(
                q=q,
                k=k,
                weights=weights,
                index_topk=topk,
                starts=starts,
                ends=ends,
                block_size=block_size,
                query=query,
                key=key,
                softmax_scale=softmax_scale,
                loss_coeff=1.0,
                pg_collection=fake_pg,
            )

        assert fused_out is not None
        got_idx, got_loss = fused_out
        assert torch.equal(got_idx, ref_idx)
        torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-5)

    def test_fused_bounds_disable_on_per_batch_mask_mismatch(self):
        """Fused bounds should disable when batched masks are not identical."""
        sq, skv, bsz = 5, 7, 2
        base_mask = torch.triu(
            torch.full((sq, skv), float("-inf"), dtype=torch.float32), diagonal=1
        )
        mask = base_mask.unsqueeze(0).expand(bsz, -1, -1).clone()
        out = _build_fused_indexer_varlen_bounds(
            sq=sq,
            skv=skv,
            device=mask.device,
            mask=mask,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
        )
        assert out is not None

        # Change one batch mask so masks are no longer identical.
        mask[1, 0, 0] = float("-inf")
        out_mismatch = _build_fused_indexer_varlen_bounds(
            sq=sq,
            skv=skv,
            device=mask.device,
            mask=mask,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
        )
        assert out_mismatch is None

    def test_scatter_topk_chunked_matches_manual_with_negative_indices(self):
        """Chunked top-k scatter should match manual behavior for -1 invalid indices."""
        b, sq, skv = 2, 4, 6
        topk_indices = torch.tensor(
            [
                [[0, 2, -1], [1, -1, -1], [2, 4, 5], [3, -1, 0]],
                [[5, 4, 1], [0, -1, 2], [3, -1, -1], [1, 2, 3]],
            ],
            dtype=torch.int32,
        )
        got = torch.full((b, sq, skv), float("-inf"), dtype=torch.float32)
        _scatter_topk_into_index_mask(got, topk_indices, seq_chunk_size=2)

        expected = torch.full((b, sq, skv), float("-inf"), dtype=torch.float32)
        topk_i64 = topk_indices.to(torch.int64)
        valid = topk_i64 >= 0
        b_idx, q_idx, t_idx = torch.where(valid)
        k_idx = topk_i64[b_idx, q_idx, t_idx]
        expected[b_idx, q_idx, k_idx] = 0.0

        assert torch.equal(got, expected)


class TestDSAAbsorbedParityCPU:
    """CPU parity tests for absorbed DSA rewrite."""

    def test_absorbed_path_matches_non_absorbed_output(self):
        """Absorbed attention + up_v projection should match non-absorbed attention output."""
        torch.manual_seed(1234)

        sq, skv, bsz, nheads = 6, 6, 1, 3
        qk_dim, qk_pos_dim = 5, 2
        kv_lora_rank, vdim = 4, 3
        softmax_scale = (qk_dim + qk_pos_dim) ** -0.5

        # Build synthetic tensors consistent with the absorbed rewrite equations.
        q_no_pe = torch.randn(sq, bsz, nheads, qk_dim, dtype=torch.float32)
        q_pos = torch.randn(sq, bsz, nheads, qk_pos_dim, dtype=torch.float32)
        kv_latent = torch.randn(skv, bsz, kv_lora_rank, dtype=torch.float32)
        k_pos_shared = torch.randn(skv, bsz, 1, qk_pos_dim, dtype=torch.float32)

        up_k_weight = torch.randn(nheads, qk_dim, kv_lora_rank, dtype=torch.float32)
        up_v_weight = torch.randn(nheads, vdim, kv_lora_rank, dtype=torch.float32)

        # Non-absorbed tensors.
        query_non_abs = torch.cat([q_no_pe, q_pos], dim=-1).contiguous()
        k_no_pe = torch.einsum("sbk,hqk->sbhq", kv_latent, up_k_weight)
        key_non_abs = torch.cat([k_no_pe, k_pos_shared.expand(-1, -1, nheads, -1)], dim=-1)
        value_non_abs = torch.einsum("sbk,hvk->sbhv", kv_latent, up_v_weight).contiguous()

        # Absorbed tensors.
        q_content_abs = torch.einsum("sbhq,hqk->sbhk", q_no_pe, up_k_weight)
        query_abs = torch.cat([q_content_abs, q_pos], dim=-1).contiguous()
        key_abs = torch.cat([kv_latent.unsqueeze(2), k_pos_shared], dim=-1).contiguous()

        # Use full-key support and causal masking in both paths.
        topk_indices = (
            torch.arange(skv, dtype=torch.int64).view(1, 1, skv).expand(bsz, sq, skv).contiguous()
        )
        causal_mask = torch.triu(
            torch.full((sq, skv), float("-inf"), dtype=torch.float32), diagonal=1
        )

        out_non_abs = unfused_dsa_fn(
            query_non_abs, key_non_abs, value_non_abs, topk_indices, softmax_scale, mask=causal_mask
        )
        out_abs_latent = _unfused_absorbed_dsa_fn(
            query_abs,
            key_abs,
            topk_indices,
            softmax_scale,
            v_channels=kv_lora_rank,
            mask=causal_mask,
        )
        out_abs = torch.einsum("sbhc,hdc->sbhd", out_abs_latent, up_v_weight).contiguous()
        out_abs = out_abs.view(sq, bsz, -1)

        torch.testing.assert_close(out_abs, out_non_abs, rtol=1e-4, atol=1e-5)


class TestRotateActivation:
    """Test rotate_activation function."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_rotate_activation_shape(self):
        """Test that rotate_activation preserves shape."""
        batch_size = 2
        seq_len = 16
        hidden_size = 128

        x = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16).cuda()
        output = rotate_activation(x)

        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16

    def test_rotate_activation_dtype_check(self):
        """Test that rotate_activation only accepts bfloat16."""
        x = torch.randn(16, 2, 128, dtype=torch.float32).cuda()

        with pytest.raises(AssertionError, match="only support bf16"):
            rotate_activation(x)


@pytest.mark.parametrize("seqlen_and_topk", [[16, 32], [64, 32]])
class TestComputeDSAIndexerLoss:
    """Test compute_dsa_indexer_loss function."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_shape(self, seqlen_and_topk):
        """Test that indexer loss returns a scalar."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
        assert loss >= 0  # KL divergence should be non-negative

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_sparse(self, seqlen_and_topk):
        """Test sparse indexer loss computation."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss_sparse = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=True,
            pg_collection=self.pg_collection,
        )

        loss_dense = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        # Sparse loss should be different from dense loss
        if seqlen > index_topk:
            assert loss_sparse != loss_dense
        else:
            assert loss_sparse == loss_dense
        assert loss_sparse >= 0
        assert loss_dense >= 0


class TestDSAIndexerLossAutoScaler:
    """Test DSAIndexerLossAutoScaler autograd function."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass(self):
        """Test that forward pass preserves output."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)
        indexer_loss = torch.tensor(0.5).cuda()
        indexer_loss.requires_grad_(True)

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        assert torch.allclose(result, output, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass(self):
        """Test that backward pass triggers indexer loss backward and scales gradient correctly."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)

        # Create indexer_loss with computation graph
        # This simulates compute_dsa_indexer_loss which computes KL divergence
        dummy_input = torch.randn(10).cuda()
        dummy_input.requires_grad_(True)
        indexer_loss = dummy_input.mean()

        # Set loss scale
        scale = torch.tensor(2.0).cuda()
        DSAIndexerLossAutoScaler.set_loss_scale(scale)

        # Apply the autograd function
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        # Trigger backward
        main_loss = result.sum()
        main_loss.backward()

        # Check that gradients flow back to output
        assert output.grad is not None, "Gradient should flow back to parameters"

        # Check that indexer_loss backward was triggered
        assert dummy_input.grad is not None, "Indexer loss backward should be triggered"

        # Verify the gradient is scaled correctly
        expected_grad_per_element = scale.item() / len(dummy_input)
        assert torch.allclose(
            dummy_input.grad,
            torch.full_like(dummy_input, expected_grad_per_element),
            rtol=0,
            atol=0,
        ), f"Gradient should be scaled by loss scale, expected {expected_grad_per_element}, got {dummy_input.grad[0].item()}"


class TestFusedDSAIndexerLossGradient:
    """Test that FusedDSAIndexerLoss manual backward matches autograd backward."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_indexer_loss_gradient_matches_autograd(self):
        """
        Test that the manually written backward in FusedDSAIndexerLoss produces
        the same gradients as PyTorch autograd on the unfused implementation.
        """
        batch_size = 2
        num_heads = 4
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        for seqlen, index_topk in [[16, 8], [32, 16], [64, 32]]:
            for sparse_loss in [False, True]:
                tag = f"[seqlen={seqlen}, topk={index_topk}, sparse={sparse_loss}]"
                torch.manual_seed(42)

                q_ref = (
                    torch.randn(
                        seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32
                    )
                    .cuda()
                    .requires_grad_(True)
                )
                weights_ref = (
                    torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32)
                    .cuda()
                    .requires_grad_(True)
                )
                k_ref = (
                    torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32)
                    .cuda()
                    .requires_grad_(True)
                )
                query = torch.randn(
                    seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                ).cuda()
                key = torch.randn(
                    seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                ).cuda()
                mask = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(),
                    diagonal=1,
                )

                # Method 1: Autograd (reference)
                index_scores_ref = _compute_index_scores(q_ref, weights_ref, k_ref)
                index_scores_masked = index_scores_ref + mask.unsqueeze(0)
                topk_k = min(index_topk, seqlen)
                topk_indices = index_scores_masked.topk(topk_k, dim=-1)[1]

                loss_ref = compute_dsa_indexer_loss(
                    index_scores=index_scores_masked,
                    topk_indices=topk_indices,
                    query=query,
                    key=key,
                    softmax_scale=softmax_scale,
                    loss_coeff=loss_coeff,
                    sparse_loss=sparse_loss,
                    pg_collection=self.pg_collection,
                )
                loss_ref.backward()

                grad_q_ref = q_ref.grad.clone()
                grad_weights_ref = weights_ref.grad.clone()
                grad_k_ref = k_ref.grad.clone()

                # Method 2: FusedDSAIndexerLoss (manual backward)
                q_fused = q_ref.detach().clone().requires_grad_(True)
                weights_fused = weights_ref.detach().clone().requires_grad_(True)
                k_fused = k_ref.detach().clone().requires_grad_(True)

                topk_indices_fused, loss_fused = FusedDSAIndexerLoss.apply(
                    q_fused,
                    weights_fused,
                    k_fused,
                    query.detach(),
                    key.detach(),
                    softmax_scale,
                    index_topk,
                    loss_coeff,
                    mask,
                    sparse_loss,
                    self.pg_collection,
                    None,
                    None,
                    None,
                )
                loss_fused.backward()

                grad_q_fused = q_fused.grad
                grad_weights_fused = weights_fused.grad
                grad_k_fused = k_fused.grad

                # Compare
                assert torch.allclose(
                    loss_fused, loss_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} Loss mismatch: fused={loss_fused.item()}, ref={loss_ref.item()}"

                assert torch.equal(
                    topk_indices_fused, topk_indices
                ), f"{tag} Top-k indices mismatch between fused and reference"

                assert torch.allclose(
                    grad_q_fused, grad_q_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_q mismatch: max diff = {(grad_q_fused - grad_q_ref).abs().max().item()}"

                assert torch.allclose(
                    grad_weights_fused, grad_weights_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_weights mismatch: max diff = {(grad_weights_fused - grad_weights_ref).abs().max().item()}"

                assert torch.allclose(
                    grad_k_fused, grad_k_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_k mismatch: max diff = {(grad_k_fused - grad_k_ref).abs().max().item()}"


class TestFusedDSAIndexerLossGradientTP:
    """Test FusedDSAIndexerLoss gradient consistency across different TP sizes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_indexer_loss_gradient_tp_consistency(self):
        """
        Test that FusedDSAIndexerLoss produces consistent gradients across TP ranks
        and matches TP=1 baseline.

        Tests all combinations of sparse_loss=[False, True] and TP=[2, 4] in a
        single test to minimise process-group init/destroy overhead.
        """
        seqlen = 64
        index_topk = 32
        batch_size = 2
        num_heads = 8
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        # =============================================
        # Compute TP=1 baselines for both sparse_loss values
        # =============================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])

        # Create inputs (shared across all variants)
        q_input = torch.randn(
            seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32
        ).cuda()
        weights_input = torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32).cuda()
        k_input = torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32).cuda()
        query_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(), diagonal=1
        )

        # {sparse_loss: (topk_indices, loss, grad_q, grad_weights, grad_k)}
        baselines = {}
        for sparse_loss in [False, True]:
            q_tp1 = q_input.clone().requires_grad_(True)
            weights_tp1 = weights_input.clone().requires_grad_(True)
            k_tp1 = k_input.clone().requires_grad_(True)

            topk_indices_tp1, loss_tp1 = FusedDSAIndexerLoss.apply(
                q_tp1,
                weights_tp1,
                k_tp1,
                query_input.detach(),
                key_input.detach(),
                softmax_scale,
                index_topk,
                loss_coeff,
                mask,
                sparse_loss,
                pg_collection_tp1,
                None,
                None,
                None,
            )
            loss_tp1.backward()

            baselines[sparse_loss] = (
                topk_indices_tp1.clone(),
                loss_tp1.detach().clone(),
                q_tp1.grad.clone(),
                weights_tp1.grad.clone(),
                k_tp1.grad.clone(),
            )

        Utils.destroy_model_parallel()

        # =============================================
        # Test each TP size against baselines
        # =============================================
        for tensor_model_parallel_size in [2, 4]:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            torch.manual_seed(42)
            model_parallel_cuda_manual_seed(42)

            pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            # query and key split along heads for TP
            head_per_rank = num_heads // tensor_model_parallel_size
            start_head = tp_rank * head_per_rank
            end_head = (tp_rank + 1) * head_per_rank
            query_tpn = query_input[:, :, start_head:end_head, :].clone()
            key_tpn = key_input[:, :, start_head:end_head, :].clone()

            for sparse_loss in [False, True]:
                topk_indices_tp1, loss_tp1_value, grad_q_tp1, grad_weights_tp1, grad_k_tp1 = (
                    baselines[sparse_loss]
                )
                tag = f"[TP={tensor_model_parallel_size}, sparse_loss={sparse_loss}]"

                q_tpn = q_input.clone().requires_grad_(True)
                weights_tpn = weights_input.clone().requires_grad_(True)
                k_tpn = k_input.clone().requires_grad_(True)

                topk_indices_tpn, loss_tpn = FusedDSAIndexerLoss.apply(
                    q_tpn,
                    weights_tpn,
                    k_tpn,
                    query_tpn.detach(),
                    key_tpn.detach(),
                    softmax_scale,
                    index_topk,
                    loss_coeff,
                    mask,
                    sparse_loss,
                    pg_collection_tpn,
                    None,
                    None,
                    None,
                )
                loss_tpn.backward()

                # Loss should be the same
                assert torch.allclose(
                    loss_tpn, loss_tp1_value, rtol=1e-5, atol=1e-5
                ), f"{tag} Loss mismatch: got {loss_tpn.item()}, TP=1 got {loss_tp1_value.item()}"

                # Top-k indices should be the same
                assert torch.equal(
                    topk_indices_tpn, topk_indices_tp1
                ), f"{tag} Top-k indices mismatch between TP=1 and TP=N"

                # Gradients should match (indexer params are duplicated across TP)
                assert torch.allclose(
                    q_tpn.grad, grad_q_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_q mismatch: max diff = {(q_tpn.grad - grad_q_tp1).abs().max().item()}"

                assert torch.allclose(
                    weights_tpn.grad, grad_weights_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_weights mismatch: max diff = {(weights_tpn.grad - grad_weights_tp1).abs().max().item()}"

                assert torch.allclose(
                    k_tpn.grad, grad_k_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_k mismatch: max diff = {(k_tpn.grad - grad_k_tp1).abs().max().item()}"

                # Check gradients are identical across all TP ranks
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    for grad_tensor, name in [
                        (q_tpn.grad, "grad_q"),
                        (weights_tpn.grad, "grad_weights"),
                        (k_tpn.grad, "grad_k"),
                    ]:
                        grad_list = [torch.zeros_like(grad_tensor) for _ in range(tp_size)]
                        torch.distributed.all_gather(
                            grad_list, grad_tensor, group=pg_collection_tpn.tp
                        )

                        for i in range(1, tp_size):
                            assert torch.allclose(
                                grad_list[0], grad_list[i], rtol=0, atol=0
                            ), f"{tag} {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


@pytest.mark.parametrize("seqlen", [16, 64])
class TestDSAIndexer:
    """Test DSA Indexer module basic functionality with TP=1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        cls = request.cls
        cls.index_topk = 32
        cls.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=cls.index_topk,
        )

        # Create indexer submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        cls.indexer = DSAIndexer(cls.config, indexer_submodules, cls.pg_collection)

        yield
        Utils.destroy_model_parallel()

    def test_dsa_indexer_constructor(self, seqlen):
        """Test indexer initialization."""
        assert isinstance(self.indexer, DSAIndexer)
        assert self.indexer.hidden_size == 256
        assert self.indexer.index_n_heads == 8
        assert self.indexer.index_head_dim == 64
        assert self.indexer.index_topk == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward(self, seqlen):
        """Test indexer forward pass."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass
        topk_indices = self.indexer(x, qr)

        # Check output shape
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_with_scores(self, seqlen):
        """Test indexer forward pass with scores."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass with scores
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr)

        # Check output shapes
        assert index_scores.shape == (batch_size, seqlen, seqlen)
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_with_scores_packed_thd(self, seqlen):
        """Test indexer forward_with_scores works with packed THD inputs."""
        batch_size = 1
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=x.device)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_kv=seqlen,
        )
        token_idx = torch.arange(seqlen, dtype=torch.int64, device=x.device)
        mask = _build_packed_causal_mask_for_test(token_idx, token_idx, cu_seqlens)

        index_scores, topk_indices = self.indexer.forward_with_scores(
            x, qr, mask=mask, packed_seq_params=packed_seq_params
        )

        assert index_scores.shape == (batch_size, seqlen, seqlen)
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_with_mask(self, seqlen):
        """Test indexer with attention mask."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()
        mask = torch.triu(
            torch.full((batch_size, seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(),
            diagonal=1,
        )

        # Forward pass with mask
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr, mask=mask)

        # Check that masked positions are not selected
        # For causal mask, topk_indices[b, i, :] should all be <= i (except for the case that
        # i < index_topk).
        for b in range(batch_size):
            for i in range(seqlen):
                assert torch.all(topk_indices[b, i] <= max(self.index_topk, i))


class TestDSAttention:
    """Test DSAttention module basic functionality with TP=1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        # Create MLA config with sparse attention parameters
        cls.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=False,
        )

        # Create sparse attention submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        cls.sparse_attention = DSAttention(
            config=cls.config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=cls.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_dsa_constructor(self):
        """Test sparse attention initialization."""
        assert isinstance(self.sparse_attention, DSAttention)
        assert hasattr(self.sparse_attention, 'indexer')
        assert isinstance(self.sparse_attention.indexer, DSAIndexer)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward(self):
        """Test sparse attention forward pass."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.cuda()

        # Create input tensors [seq_len, batch, num_heads, head_dim]
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Check output shape
        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_backward(self):
        """Test sparse attention backward pass with indexer loss."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.train()
        self.sparse_attention.cuda()

        # Create input tensors
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for inputs
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Check that indexer parameters have gradients
        for name, param in self.sparse_attention.indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Indexer parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_topk_selection(self):
        """Test that sparse attention correctly selects top-k indices."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.eval()
        self.sparse_attention.cuda()

        # Create input tensors
        query = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        value = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        with torch.no_grad():
            # Get topk indices from indexer
            _, topk_indices = self.sparse_attention.indexer.forward_with_scores(x, qr)

            # Forward pass
            output = self.sparse_attention(
                query=query,
                key=key,
                value=value,
                x=x,
                qr=qr,
                attention_mask=attention_mask,
                attn_mask_type=AttnMaskType.causal,
            )

        # Check that topk_indices are valid
        assert torch.all(topk_indices >= 0)
        assert torch.all(topk_indices < seq_len)
        assert topk_indices.shape[2] == min(self.config.dsa_indexer_topk, seq_len)


# ======================================================================================
# Tensor Parallel Consistency Tests
# ======================================================================================


class TestIndexerTensorParallel:
    """Test DSA Indexer with different TP sizes and SP settings, compare with TP=1 baseline."""

    TP_SIZES = [2, 4, 8]
    SP_VALUES = [False, True]

    def _create_config(self, sequence_parallel=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )

    def _create_indexer(self, config, pg_collection):
        """Helper to create indexer."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        return DSAIndexer(config, indexer_submodules, pg_collection)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_weight_consistency(self):
        """Test that indexer weights are identical across ALL GPUs."""
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            world_size = torch.distributed.get_world_size()

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer = self._create_indexer(config, pg_collection).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                # Check that all weights are identical across ALL ranks
                if world_size > 1:
                    for name, param in indexer.named_parameters():
                        param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                        torch.distributed.all_gather(param_list, param.data)

                        for i in range(1, world_size):
                            assert torch.allclose(
                                param_list[0], param_list[i], rtol=0, atol=0
                            ), f"{tag} Parameter {name} differs between rank 0 and rank {i} (world)"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_consistency(self):
        """Test that indexer gives consistent results across different TP sizes and SP settings."""
        seq_len = 64
        batch_size = 2

        # TP=1 baseline (once for all TP sizes; TP=1 doesn't use SP)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(sequence_parallel=False)
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tp1 = self._create_indexer(config_tp1, pg_collection_tp1).cuda()

        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()

        index_scores_tp1, topk_indices_tp1 = indexer_tp1.forward_with_scores(x_input, qr_input)
        loss_tp1 = index_scores_tp1.sum()
        loss_tp1.backward()

        indexer_tp1_grads = {
            name: param.grad.clone().cpu()
            for name, param in indexer_tp1.named_parameters()
            if param.grad is not None
        }

        Utils.destroy_model_parallel()

        # Test each TP size with both SP values
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config_tpn = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer_tpn = self._create_indexer(config_tpn, pg_collection_tpn).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                if sequence_parallel:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    seq_per_rank = seq_len // tensor_model_parallel_size
                    start_idx = tp_rank * seq_per_rank
                    end_idx = (tp_rank + 1) * seq_per_rank
                    x_tpn = x_input[start_idx:end_idx]
                    qr_tpn = qr_input[start_idx:end_idx]
                else:
                    x_tpn = x_input
                    qr_tpn = qr_input

                index_scores_tpn, topk_indices_tpn = indexer_tpn.forward_with_scores(x_tpn, qr_tpn)
                loss_tpn = index_scores_tpn.sum()
                loss_tpn.backward()

                assert index_scores_tpn.shape == index_scores_tp1.shape
                assert topk_indices_tpn.shape == topk_indices_tp1.shape
                assert torch.allclose(
                    index_scores_tpn, index_scores_tp1, rtol=0, atol=0
                ), f"{tag} Index scores mismatch vs TP=1"
                assert torch.equal(
                    topk_indices_tpn, topk_indices_tp1
                ), f"{tag} Top-k indices mismatch vs TP=1"

                for name, param in indexer_tpn.named_parameters():
                    if param.grad is not None and name in indexer_tp1_grads:
                        assert torch.allclose(
                            param.grad.cpu(), indexer_tp1_grads[name], rtol=0, atol=0
                        ), f"{tag} Indexer gradient {name} mismatch vs TP=1"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_gradient_sync(self):
        """Test that gradients are properly synchronized within TP group."""
        seq_len = 64
        batch_size = 2

        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer = self._create_indexer(config, pg_collection).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                x_input = torch.randn(
                    seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16
                ).cuda()
                qr_input = torch.randn(
                    seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16
                ).cuda()

                if sequence_parallel:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    seq_per_rank = seq_len // tp_size
                    start_idx = tp_rank * seq_per_rank
                    end_idx = (tp_rank + 1) * seq_per_rank
                    x = x_input[start_idx:end_idx]
                    qr = qr_input[start_idx:end_idx]
                else:
                    x = x_input
                    qr = qr_input

                index_scores, topk_indices = indexer.forward_with_scores(x, qr)
                loss = index_scores.sum()
                loss.backward()

                for name, param in indexer.named_parameters():
                    if param.requires_grad:
                        assert param.grad is not None, f"{tag} Parameter {name} has no gradient"

                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    for name, param in indexer.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                            torch.distributed.all_gather(
                                grad_list, param.grad, group=pg_collection.tp
                            )

                            for i in range(1, tp_size):
                                assert torch.allclose(
                                    grad_list[0], grad_list[i], rtol=0, atol=0
                                ), f"{tag} Gradient for {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


class TestDSAttentionTensorParallel:
    """Test DSAttention with different TP sizes, SP settings, and sparse indexer loss."""

    TP_SIZES = [2, 4]
    SP_VALUES = [False, True]
    SPARSE_VALUES = [False, True]

    def _create_config(self, sequence_parallel=False, use_sparse_indexer_loss=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=use_sparse_indexer_loss,
        )

    def _create_sparse_attention(self, config, pg_collection):
        """Helper to create sparse attention."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        return DSAttention(
            config=config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=pg_collection,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_weight_consistency(self):
        """Test that sparse attention indexer weights are identical across ALL GPUs."""
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            world_size = torch.distributed.get_world_size()

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    config = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    if world_size > 1:
                        for name, param in sparse_attention.indexer.named_parameters():
                            param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                            torch.distributed.all_gather(param_list, param.data)

                            for i in range(1, world_size):
                                torch.testing.assert_close(
                                    param_list[0], param_list[i], rtol=0, atol=0
                                )

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward_consistency(self):
        """Test that sparse attention gives consistent results across different TP, SP, and sparse loss settings."""
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

        seq_len = 64
        batch_size = 2

        # TP=1 baselines: one per use_sparse_indexer_loss value (SP is always False for TP=1)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        baselines = {}  # {sparse_loss: (output, indexer_grads, q_grad, k_grad, v_grad)}
        for use_sparse_indexer_loss in self.SPARSE_VALUES:
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123)

            config_tp1 = self._create_config(
                sequence_parallel=False, use_sparse_indexer_loss=use_sparse_indexer_loss
            )
            pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp']
            )
            sparse_attention_tp1 = self._create_sparse_attention(
                config_tp1, pg_collection_tp1
            ).cuda()

            num_heads = config_tp1.num_attention_heads
            head_dim = config_tp1.hidden_size // num_heads

            query_input = (
                torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
                .cuda()
                .requires_grad_(True)
            )
            key_input = (
                torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
                .cuda()
                .requires_grad_(True)
            )
            value_input = (
                torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
                .cuda()
                .requires_grad_(True)
            )
            x_input = torch.randn(
                seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
            ).cuda()
            qr_input = torch.randn(
                seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
            ).cuda()
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
            attention_mask = torch.tril(attention_mask)

            sparse_attention_tp1.train()
            output_tp1 = sparse_attention_tp1(
                query=query_input,
                key=key_input,
                value=value_input,
                x=x_input,
                qr=qr_input,
                attention_mask=attention_mask,
                attn_mask_type=AttnMaskType.causal,
            )

            loss_tp1 = output_tp1.sum()
            loss_tp1.backward()

            baselines[use_sparse_indexer_loss] = (
                output_tp1.detach().clone(),
                {
                    name: param.grad.clone()
                    for name, param in sparse_attention_tp1.indexer.named_parameters()
                    if param.grad is not None
                },
                query_input.grad.clone().cpu(),
                key_input.grad.clone().cpu(),
                value_input.grad.clone().cpu(),
                num_heads,
                head_dim,
            )

        Utils.destroy_model_parallel()

        # Test each TP size with all (SP, sparse) combos
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    (
                        output_tp1,
                        indexer_tp1_grads,
                        query_tp1_grad,
                        key_tp1_grad,
                        value_tp1_grad,
                        num_heads,
                        head_dim,
                    ) = baselines[use_sparse_indexer_loss]

                    config_tpn = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention_tpn = self._create_sparse_attention(
                        config_tpn, pg_collection_tpn
                    ).cuda()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    query_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    key_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    value_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    x_input_tpn = torch.randn(
                        seq_len, batch_size, config_tpn.hidden_size, dtype=torch.bfloat16
                    ).cuda()
                    qr_input_tpn = torch.randn(
                        seq_len, batch_size, config_tpn.q_lora_rank, dtype=torch.bfloat16
                    ).cuda()
                    attention_mask_tpn = torch.ones(
                        batch_size, 1, seq_len, seq_len, dtype=torch.bool
                    ).cuda()
                    attention_mask_tpn = torch.tril(attention_mask_tpn)

                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    if sequence_parallel:
                        seq_per_rank = seq_len // tensor_model_parallel_size
                        start_idx = tp_rank * seq_per_rank
                        end_idx = (tp_rank + 1) * seq_per_rank
                        x_tpn = x_input_tpn[start_idx:end_idx]
                        qr_tpn = qr_input_tpn[start_idx:end_idx]
                    else:
                        x_tpn = x_input_tpn
                        qr_tpn = qr_input_tpn

                    head_per_rank = num_heads // tensor_model_parallel_size
                    start_head = tp_rank * head_per_rank
                    end_head = (tp_rank + 1) * head_per_rank
                    query_tpn = (
                        query_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )
                    key_tpn = (
                        key_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )
                    value_tpn = (
                        value_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )

                    sparse_attention_tpn.train()
                    output_tpn = sparse_attention_tpn(
                        query=query_tpn,
                        key=key_tpn,
                        value=value_tpn,
                        x=x_tpn,
                        qr=qr_tpn,
                        attention_mask=attention_mask_tpn,
                        attn_mask_type=AttnMaskType.causal,
                    )

                    loss_tpn = output_tpn.sum()
                    loss_tpn.backward()

                    output_tpn_gathered = gather_from_tensor_model_parallel_region(
                        output_tpn, group=pg_collection_tpn.tp
                    )
                    assert output_tpn_gathered.shape == output_tp1.shape
                    assert torch.allclose(
                        output_tpn_gathered.detach(), output_tp1, rtol=0, atol=0
                    ), f"{tag} Sparse attention outputs mismatch vs TP=1"

                    for name, param in sparse_attention_tpn.indexer.named_parameters():
                        if param.grad is not None and name in indexer_tp1_grads:
                            torch.testing.assert_close(
                                param.grad, indexer_tp1_grads[name], rtol=1e-5, atol=1e-5
                            )

                    sq, b, nh, hd = query_tpn.grad.shape
                    query_grad_gathered = gather_from_tensor_model_parallel_region(
                        query_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)
                    key_grad_gathered = gather_from_tensor_model_parallel_region(
                        key_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)
                    value_grad_gathered = gather_from_tensor_model_parallel_region(
                        value_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)

                    assert torch.allclose(
                        query_grad_gathered.cpu(), query_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Query gradient mismatch vs TP=1"
                    assert torch.allclose(
                        key_grad_gathered.cpu(), key_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Key gradient mismatch vs TP=1"
                    assert torch.allclose(
                        value_grad_gathered.cpu(), value_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Value gradient mismatch vs TP=1"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_gradient_sync(self):
        """Test that indexer gradients are properly synchronized within TP group."""
        seq_len = 64
        batch_size = 2

        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    config = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
                    sparse_attention.train()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    num_heads = config.num_attention_heads
                    head_dim = config.hidden_size // num_heads

                    query_input = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    key_input = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    value_input = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    x_input = torch.randn(
                        seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16
                    ).cuda()
                    qr_input = torch.randn(
                        seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16
                    ).cuda()

                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    if sequence_parallel:
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                        seq_per_rank = seq_len // tp_size
                        start_idx = tp_rank * seq_per_rank
                        end_idx = (tp_rank + 1) * seq_per_rank
                        x = x_input[start_idx:end_idx]
                        qr = qr_input[start_idx:end_idx]
                    else:
                        x = x_input
                        qr = qr_input

                    head_per_rank = num_heads // tensor_model_parallel_size
                    start_head = tp_rank * head_per_rank
                    end_head = (tp_rank + 1) * head_per_rank
                    query = query_input[:, :, start_head:end_head, :]
                    key = key_input[:, :, start_head:end_head, :]
                    value = value_input[:, :, start_head:end_head, :]

                    attention_mask = torch.ones(
                        batch_size, 1, seq_len, seq_len, dtype=torch.bool
                    ).cuda()
                    attention_mask = torch.tril(attention_mask)

                    query.requires_grad_(True)
                    key.requires_grad_(True)
                    value.requires_grad_(True)

                    output = sparse_attention(
                        query=query,
                        key=key,
                        value=value,
                        x=x,
                        qr=qr,
                        attention_mask=attention_mask,
                        attn_mask_type=AttnMaskType.causal,
                    )

                    loss = output.sum()
                    loss.backward()

                    assert query.grad is not None
                    assert key.grad is not None
                    assert value.grad is not None

                    for name, param in sparse_attention.indexer.named_parameters():
                        if param.requires_grad:
                            assert (
                                param.grad is not None
                            ), f"{tag} Indexer parameter {name} has no gradient"

                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    if tp_size > 1:
                        for name, param in sparse_attention.indexer.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                                torch.distributed.all_gather(
                                    grad_list, param.grad, group=pg_collection.tp
                                )

                                for i in range(1, tp_size):
                                    assert torch.allclose(
                                        grad_list[0], grad_list[i], rtol=0, atol=0
                                    ), f"{tag} Gradient for {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


@pytest.mark.internal
class TestDSAModuleSpecDispatch:
    """Tests for get_dsa_module_spec_for_backend and get_experimental_attention_variant_module_spec."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _make_dsa_config(self, **kwargs):
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            **kwargs,
        )

    def test_get_experimental_attention_variant_module_spec_dsa(self):
        """get_experimental_attention_variant_module_spec dispatches to DSA for variant='dsa'."""
        config = self._make_dsa_config(experimental_attention_variant="dsa")
        spec = get_experimental_attention_variant_module_spec(config)
        assert spec.module == MLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention

    def test_get_dsa_module_spec_for_backend(self):
        """get_dsa_module_spec_for_backend returns the correct full spec structure."""
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        config = self._make_dsa_config()
        backend = TESpecProvider()
        spec = get_dsa_module_spec_for_backend(config, backend=backend)
        assert spec.module == MLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention
        assert spec.submodules.core_attention.submodules.indexer.module == DSAIndexer
        assert spec.params["attn_mask_type"] == AttnMaskType.causal

    def test_get_dsa_module_spec_requires_mla(self):
        """get_dsa_module_spec_for_backend rejects configs without MLA."""
        from megatron.core.transformer import TransformerConfig as _TransformerConfig

        config = _TransformerConfig(num_layers=2, hidden_size=256, num_attention_heads=4)
        with pytest.raises(AssertionError, match="only MLA supports sparse attention"):
            get_dsa_module_spec_for_backend(config, backend=None)

    def test_get_dsa_module_spec_rejects_qk_l2_norm(self):
        """get_dsa_module_spec_for_backend rejects configs with qk_l2_norm=True."""
        config = self._make_dsa_config(qk_l2_norm=True)
        with pytest.raises(AssertionError, match="qk_l2_norm is not supported"):
            get_dsa_module_spec_for_backend(config, backend=None)
