# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from unittest.mock import patch

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CSA_OPERATION_DETERMINISM,
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
    _apply_rope,
    _compute_unfused_csa_non_compressed_lse,
    _get_compress_causal_mask_cached,
    _get_compress_valid_counts_cached,
    _pool_compressor_values,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    FusedDSAIndexerLoss,
    compute_dsa_indexer_loss,
    fused_qk_topk_naive,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed."""
    return x * scale


class _DisabledContextTracker:
    """Track whether a projection runs inside the FP8-disabled context."""

    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __call__(self, _config, is_init=False):
        assert not is_init
        return self

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.depth -= 1
        return False


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in both dsa and csa modules if not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ===========================================================================
# Helper function tests
# ===========================================================================


class _SingleRankTP:
    @staticmethod
    def size():
        return 1


class _SingleRankPG:
    tp = _SingleRankTP()


class _TwoRankTP:
    @staticmethod
    def size():
        return 2


class _TwoRankPG:
    tp = _TwoRankTP()


def test_csa_rejects_explicit_attention_mask():
    """The native SBHD slice must not silently ignore padding or document boundaries."""
    with pytest.raises(ValueError, match="implicit causal mask"):
        CompressedSparseAttention.forward(
            None, query=None, key=None, value=None, attention_mask=torch.zeros(1, 1, 1, 1)
        )


def test_all_csa_operations_declare_determinism():
    """Every eager CSA operation declares a valid bit-exact determinism status."""
    valid_statuses = {"deterministic", "nondeterministic", "unknown"}
    assert set(CSA_OPERATION_DETERMINISM) == {
        "unfused_sparse_attention",
        "non_compressed_lse",
        "compressor_pooling",
    }
    assert set(CSA_OPERATION_DETERMINISM.values()) <= valid_statuses


def test_compressed_causal_metadata_is_cached_and_correct():
    ratio, seqlen, n_compressed = 4, 12, 3
    device_str = "cpu"
    _get_compress_causal_mask_cached.cache_clear()
    _get_compress_valid_counts_cached.cache_clear()

    mask = _get_compress_causal_mask_cached(ratio, seqlen, n_compressed, device_str)
    valid_counts = _get_compress_valid_counts_cached(ratio, seqlen, device_str)

    assert mask is _get_compress_causal_mask_cached(ratio, seqlen, n_compressed, device_str)
    assert valid_counts is _get_compress_valid_counts_cached(ratio, seqlen, device_str)
    expected_counts = torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    expected_mask = torch.where(
        torch.arange(n_compressed).unsqueeze(0) >= expected_counts, float("-inf"), 0.0
    )
    torch.testing.assert_close(valid_counts, expected_counts)
    torch.testing.assert_close(mask, expected_mask)
    assert mask.unsqueeze(0).expand(2, -1, -1).stride(0) == 0


def test_unfused_csa_non_compressed_lse_matches_window_and_sink_oracle():
    torch.manual_seed(17)
    seqlen_q, batch_size, n_kv = 3, 2, 5
    num_heads, head_dim = 2, 4
    query = torch.randn(seqlen_q, batch_size, num_heads, head_dim, requires_grad=True)
    kv_full = torch.randn(n_kv, batch_size, head_dim, requires_grad=True)
    sink = torch.randn(num_heads, requires_grad=True)
    window_indices = torch.tensor([[[-1, 0], [0, 1], [1, 2]], [[-1, 0], [0, 2], [2, 4]]])

    expected = torch.empty(batch_size, num_heads, seqlen_q)
    with torch.no_grad():
        for batch in range(batch_size):
            for row in range(seqlen_q):
                for head in range(num_heads):
                    logits = [sink[head]]
                    for key_index in window_indices[batch, row]:
                        if key_index >= 0:
                            logits.append(
                                torch.dot(query[row, batch, head], kv_full[key_index, batch])
                            )
                    expected[batch, head, row] = torch.logsumexp(torch.stack(logits), dim=0)

    actual = _compute_unfused_csa_non_compressed_lse(
        query, kv_full, sink, window_indices, softmax_scale=1.0
    )

    assert actual.shape == (batch_size, num_heads, seqlen_q)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    for teacher_tensor in (query, kv_full, sink):
        assert teacher_tensor.grad is None


def _independent_csa_indexer_loss(
    index_scores,
    topk_indices,
    query,
    compressed_kv,
    window_kv,
    window_indices,
    sink,
    *,
    sparse_loss,
    loss_coeff,
):
    """Compute a small-loop CSA teacher oracle with the complete denominator."""
    batch_size, seqlen_q, n_compressed = index_scores.shape
    num_heads = query.shape[2]
    losses = []
    for batch in range(batch_size):
        for row in range(seqlen_q):
            selected = (
                topk_indices[batch, row].tolist() if sparse_loss else list(range(n_compressed))
            )
            target = []
            for compressed_index in selected:
                head_mass = 0.0
                for head in range(num_heads):
                    non_compressed_logits = [sink[head]]
                    for window_index in window_indices[batch, row]:
                        if window_index >= 0:
                            non_compressed_logits.append(
                                torch.dot(query[row, batch, head], window_kv[window_index, batch])
                            )
                    compressed_logits = [
                        torch.dot(query[row, batch, head], compressed_kv[key_index, batch])
                        for key_index in selected
                    ]
                    denominator = torch.logsumexp(
                        torch.stack(non_compressed_logits + compressed_logits), dim=0
                    )
                    selected_position = selected.index(compressed_index)
                    head_mass = head_mass + torch.exp(
                        compressed_logits[selected_position] - denominator
                    )
                target.append(head_mass)
            target = torch.stack(target)
            target = target / target.sum()
            predict_log = torch.log_softmax(index_scores[batch, row, selected], dim=-1)
            losses.append((target * (torch.log(target) - predict_log)).sum())
    return torch.stack(losses).mean() * loss_coeff


@pytest.mark.parametrize("sparse_loss", [False, True], ids=["dense", "sparse"])
def test_csa_indexer_loss_uses_full_attention_denominator(sparse_loss):
    torch.manual_seed(29)
    seqlen_q, batch_size, num_heads, head_dim = 4, 1, 2, 3
    n_compressed, index_heads, index_dim = 3, 2, 2
    index_topk, loss_coeff = 2, 0.7

    q = torch.randn(seqlen_q, batch_size, index_heads, index_dim, requires_grad=True)
    weights = torch.randn(seqlen_q, batch_size, index_heads, requires_grad=True)
    k = torch.randn(n_compressed, batch_size, index_dim, requires_grad=True)
    query = torch.randn(seqlen_q, batch_size, num_heads, head_dim, requires_grad=True)
    window_kv = torch.randn(seqlen_q, batch_size, head_dim, requires_grad=True)
    compressed_kv = torch.randn(n_compressed, batch_size, head_dim, requires_grad=True)
    sink = torch.randn(num_heads, requires_grad=True)
    window_indices = torch.tensor([[[-1, 0], [0, 1], [1, 2], [2, 3]]])
    non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
        query, window_kv, sink, window_indices, softmax_scale=1.0
    )
    key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, num_heads, -1)
    compressed_mask = torch.zeros(seqlen_q, n_compressed)

    q_reference = q.detach().clone().requires_grad_(True)
    weights_reference = weights.detach().clone().requires_grad_(True)
    k_reference = k.detach().clone().requires_grad_(True)
    index_scores_reference, topk_reference = fused_qk_topk_naive(
        q_reference, k_reference, weights_reference, index_topk
    )
    loss_reference = compute_dsa_indexer_loss(
        index_scores_reference,
        topk_reference,
        query.detach(),
        key_for_loss.detach(),
        1.0,
        loss_coeff,
        sparse_loss,
        _SingleRankPG(),
        mask=compressed_mask,
        non_compressed_lse=non_compressed_lse,
    )
    loss_reference.backward()

    topk_actual, loss_actual = FusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key_for_loss,
        1.0,
        index_topk,
        loss_coeff,
        compressed_mask,
        sparse_loss,
        _SingleRankPG(),
        None,
        None,
        None,
        None,
        False,
        True,
        non_compressed_lse,
    )
    loss_actual.backward()

    independent_loss = _independent_csa_indexer_loss(
        index_scores_reference.detach(),
        topk_reference,
        query.detach(),
        compressed_kv.detach(),
        window_kv.detach(),
        window_indices,
        sink.detach(),
        sparse_loss=sparse_loss,
        loss_coeff=loss_coeff,
    )

    torch.testing.assert_close(loss_actual, independent_loss)
    torch.testing.assert_close(loss_actual, loss_reference)
    torch.testing.assert_close(topk_actual, topk_reference)
    torch.testing.assert_close(q.grad, q_reference.grad)
    torch.testing.assert_close(weights.grad, weights_reference.grad)
    torch.testing.assert_close(k.grad, k_reference.grad)
    for teacher_tensor in (query, window_kv, compressed_kv, sink):
        assert teacher_tensor.grad is None


class TestGetWindowTopkIdxs:
    """Test get_window_topk_idxs helper."""

    def test_basic_shape(self):
        batch_size, seqlen, window_size = 2, 16, 4
        idxs = get_window_topk_idxs(window_size, batch_size, seqlen, torch.device("cpu"))
        assert idxs.shape == (batch_size, seqlen, window_size)

    def test_causal_no_future(self):
        """Indices should never exceed the query position."""
        seqlen, window_size = 32, 8
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        for i in range(seqlen):
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert torch.all(valid <= i), f"Position {i} has future indices"

    def test_invalid_marked_minus_one(self):
        """Early positions that cannot fill the window should use -1."""
        seqlen, window_size = 8, 4
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs[0, 0, 0] == -1 or idxs[0, 0, 0] == 0
        for pos in range(window_size, seqlen):
            assert torch.all(idxs[0, pos] >= 0), f"Position {pos} has invalid -1"

    def test_window_larger_than_seqlen(self):
        """Window larger than sequence length should still work."""
        seqlen, window_size = 4, 16
        idxs = get_window_topk_idxs(window_size, 1, seqlen, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, window_size)


class TestGetCompressTopkIdxs:
    """Test get_compress_topk_idxs helper."""

    def test_basic_shape(self):
        ratio, batch_size, seqlen, offset = 4, 2, 32, 32
        idxs = get_compress_topk_idxs(ratio, batch_size, seqlen, offset, torch.device("cpu"))
        n_compressed = seqlen // ratio
        assert idxs.shape == (batch_size, seqlen, n_compressed)

    def test_offset_applied(self):
        """Valid indices should be >= offset."""
        ratio, seqlen, offset = 4, 32, 100
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        valid = idxs[idxs >= 0]
        if valid.numel() > 0:
            assert torch.all(valid >= offset), "Valid indices should be offset"

    def test_causal_no_future(self):
        """Compressed indices should respect causality."""
        ratio, seqlen, offset = 4, 32, 32
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        for i in range(seqlen):
            n_valid = (i + 1) // ratio
            valid = idxs[0, i][idxs[0, i] >= 0]
            assert valid.numel() <= n_valid, f"Position {i} has too many valid compressed indices"

    def test_ratio_128(self):
        """Test with large compression ratio."""
        ratio, seqlen, offset = 128, 256, 256
        idxs = get_compress_topk_idxs(ratio, 1, seqlen, offset, torch.device("cpu"))
        assert idxs.shape == (1, seqlen, seqlen // ratio)


# ===========================================================================
# unfused_compressed_sparse_attn tests
# ===========================================================================


def test_unfused_sparse_attention_rejects_sink_head_mismatch():
    with pytest.raises(ValueError, match="one value per query head"):
        unfused_compressed_sparse_attn(
            query=torch.zeros(2, 1, 2, 4),
            kv_full=torch.zeros(2, 1, 4),
            attn_sink=torch.zeros(1),
            topk_indices=torch.zeros(1, 2, 1, dtype=torch.int32),
            softmax_scale=1.0,
        )


class TestUnfusedCompressedSparseAttn:
    """Test the unfused compressed sparse attention kernel."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_shape(self):
        """Test output shape of unfused compressed sparse attention."""
        sq, b, np_, hn = 16, 2, 4, 64
        n_kv = sq + sq // 4
        topk = 8

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()
        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )

        assert output.shape == (sq, b, np_ * hn)
        assert output.dtype == query.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_invalid_indices_masked(self):
        """Test that -1 indices are properly masked."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.bfloat16).cuda()
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.bfloat16).cuda()
        attn_sink = torch.zeros(np_, dtype=torch.float32).cuda()

        topk_indices = torch.full((b, sq, topk), -1, dtype=torch.int32).cuda()
        topk_indices[:, :, 0] = 0
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_flow(self):
        """Test that gradients flow through sparse attention."""
        sq, b, np_, hn = 8, 1, 2, 32
        n_kv = sq
        topk = 4

        query = torch.randn(sq, b, np_, hn, dtype=torch.float32).cuda().requires_grad_(True)
        kv_full = torch.randn(n_kv, b, hn, dtype=torch.float32).cuda().requires_grad_(True)
        attn_sink = torch.nn.Parameter(torch.zeros(np_, dtype=torch.float32).cuda())

        topk_indices = torch.randint(0, n_kv, (b, sq, topk), dtype=torch.int32).cuda()
        softmax_scale = hn**-0.5

        output = unfused_compressed_sparse_attn(
            query, kv_full, attn_sink, topk_indices, softmax_scale
        )
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert kv_full.grad is not None
        assert attn_sink.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_repeated_and_invalid_indices_match_loop_oracle(self):
        """Batch-flattened gather preserves forward and backward sparse-attention semantics."""
        torch.manual_seed(41)
        sq, b, np_, hn, n_kv = 3, 2, 2, 4, 5
        topk_indices = torch.tensor(
            [[[0, 0, -1], [1, 3, 1], [4, -1, 2]], [[2, -1, 2], [4, 0, -1], [1, 1, 3]]],
            dtype=torch.int32,
            device="cuda",
        )
        query = torch.randn(sq, b, np_, hn, device="cuda", requires_grad=True)
        kv_full = torch.randn(n_kv, b, hn, device="cuda", requires_grad=True)
        attn_sink = torch.randn(np_, device="cuda", requires_grad=True)
        grad_output = torch.randn(sq, b, np_ * hn, device="cuda")

        with patch("torch.gather", side_effect=AssertionError("expanded gather must not be used")):
            actual = unfused_compressed_sparse_attn(
                query, kv_full, attn_sink, topk_indices, softmax_scale=0.5
            )
        (actual * grad_output).sum().backward()
        actual_grads = (query.grad.clone(), kv_full.grad.clone(), attn_sink.grad.clone())

        query_ref = query.detach().clone().requires_grad_(True)
        kv_ref = kv_full.detach().clone().requires_grad_(True)
        sink_ref = attn_sink.detach().clone().requires_grad_(True)
        rows = []
        for row in range(sq):
            batches = []
            for batch in range(b):
                heads = []
                for head in range(np_):
                    valid_indices = topk_indices[batch, row]
                    valid_indices = valid_indices[valid_indices >= 0].long()
                    logits = (
                        torch.einsum(
                            "h,kh->k", query_ref[row, batch, head], kv_ref[valid_indices, batch]
                        )
                        * 0.5
                    )
                    probabilities = torch.softmax(
                        torch.cat([logits, sink_ref[head : head + 1]]), dim=0
                    )
                    heads.append(
                        torch.einsum("k,kh->h", probabilities[:-1], kv_ref[valid_indices, batch])
                    )
                batches.append(torch.cat(heads))
            rows.append(torch.stack(batches))
        expected = torch.stack(rows)
        (expected * grad_output).sum().backward()

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads, (query_ref.grad, kv_ref.grad, sink_ref.grad)
        ):
            torch.testing.assert_close(actual_grad, expected_grad)


# ===========================================================================
# Compressor tests
# ===========================================================================


def test_compressor_pooling_matches_fp32_forward_and_backward_oracle():
    torch.manual_seed(43)
    kv = torch.randn(2, 4, 2, 6, dtype=torch.bfloat16, requires_grad=True)
    score = torch.randn(2, 4, 2, 6, dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(2, 2, 6, dtype=torch.bfloat16)

    actual = _pool_compressor_values(kv, score, torch.bfloat16)
    (actual * grad_output).sum().backward()
    actual_grads = (kv.grad.clone(), score.grad.clone())

    kv_ref = kv.detach().clone().requires_grad_(True)
    score_ref = score.detach().clone().requires_grad_(True)
    expected = (
        (kv_ref.float() * torch.softmax(score_ref, dim=1, dtype=torch.float32))
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    (expected * grad_output).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[0], kv_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[1], score_ref.grad, rtol=0, atol=0)


def _make_mla_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    csa_compress_ratios=None,
    csa_window_size=8,
    csa_dense_mode=False,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    dsa_indexer_use_sparse_loss=False,
):
    """Helper to create MLATransformerConfig for CSA tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0] * num_layers
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        csa_dense_mode=csa_dense_mode,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
    )


def _make_compressor_submodules():
    """Create Compressor submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressorSubmodules(
        linear_wkv=ModuleSpec(module=TELinear),
        linear_wgate=ModuleSpec(module=TELinear),
        norm=ModuleSpec(module=TENorm),
    )


def _make_csa_indexer_submodules():
    """Create CSAIndexer submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_weights_proj=ModuleSpec(module=TELinear),
        compressor=partial(Compressor, submodules=_make_compressor_submodules()),
    )


def _make_csa_submodules():
    """Create CompressedSparseAttention submodules spec."""
    return CompressedSparseAttentionSubmodules(
        compressor=partial(Compressor, submodules=_make_compressor_submodules()),
        indexer=partial(CSAIndexer, submodules=_make_csa_indexer_submodules()),
    )


def test_compressed_sparse_attention_rejects_tensor_parallelism():
    config = _make_mla_config(num_attention_heads=2, csa_compress_ratios=[0] * 4)
    with pytest.raises(ValueError, match="tensor-parallel size 1"):
        CompressedSparseAttention(
            config=config,
            submodules=CompressedSparseAttentionSubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=_TwoRankPG(),
            compress_ratio=0,
        )


def test_compressed_sparse_attention_rejects_query_head_mismatch():
    config = _make_mla_config(num_attention_heads=2, csa_compress_ratios=[0] * 4)
    attention = CompressedSparseAttention(
        config=config,
        submodules=CompressedSparseAttentionSubmodules(),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=_SingleRankPG(),
        compress_ratio=0,
    )

    with pytest.raises(ValueError, match="query head count"):
        attention(
            query=torch.zeros(2, 1, 1, config.v_head_dim), key=None, value=None, attention_mask=None
        )


# ===========================================================================
# Compressor tests
# ===========================================================================


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressor:
    """Test Compressor module."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[4, 128, 4, 128])
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_output_shape(self, compress_ratio):
        """Test that compressor produces correct output shape."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)

        expected_len = seq_len // compress_ratio
        assert output is not None
        assert output.shape == (expected_len, batch_size, head_dim)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_too_short_input(self, compress_ratio):
        """Test that compressor returns None when input is shorter than compress_ratio."""
        short_len = compress_ratio - 1
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = torch.randn(short_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        output = compressor(x)
        assert output is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compressor_gradient_flow(self, compress_ratio):
        """Test that gradients flow through the compressor."""
        seq_len = 256
        batch_size = 2
        head_dim = self.config.v_head_dim

        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

        x = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        output = compressor(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in compressor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_projection_disables_fp8(self, compress_ratio, monkeypatch):
        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=self.config.v_head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()
        tracker = _DisabledContextTracker()
        calls = []

        for name, projection in (
            ('linear_wkv', compressor.linear_wkv),
            ('linear_wgate', compressor.linear_wgate),
        ):
            original_forward = projection.forward

            def checked_forward(*args, _name=name, _forward=original_forward, **kwargs):
                assert tracker.depth > 0, f"{_name} ran outside the FP8-disabled context"
                calls.append(_name)
                return _forward(*args, **kwargs)

            monkeypatch.setattr(projection, 'forward', checked_forward)

        monkeypatch.setattr(
            'megatron.core.transformer.experimental_attention_variant.csa.get_fp8_disabled_context',
            tracker,
        )
        x = torch.randn(
            compress_ratio * 2, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        )
        compressor(x)

        assert calls == ['linear_wkv', 'linear_wgate']
        assert tracker.entries == 1


# ===========================================================================
# CSAIndexer tests
# ===========================================================================


@pytest.mark.parametrize("seqlen", [32, 128])
class TestCSAIndexer:
    """Test CSAIndexer module basic functionality."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.compress_ratio = 4
        cls.config = _make_mla_config(csa_compress_ratios=[4, 4, 4, 4], dsa_indexer_topk=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.indexer = CSAIndexer(
            config=cls.config,
            submodules=_make_csa_indexer_submodules(),
            compress_ratio=cls.compress_ratio,
            rotary_pos_emb=cls.rotary_pos_emb,
            pg_collection=cls.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_csa_indexer_constructor(self, seqlen):
        """Test CSAIndexer initialization."""
        assert isinstance(self.indexer, CSAIndexer)
        assert self.indexer.compress_ratio == self.compress_ratio
        assert self.indexer.index_n_heads == self.config.dsa_indexer_n_heads
        assert self.indexer.index_head_dim == self.config.dsa_indexer_head_dim
        assert self.indexer.index_topk == self.config.dsa_indexer_topk

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward(self, seqlen):
        """Test CSAIndexer forward pass."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        index_scores, topk_indices = self.indexer(x, qr)
        n_compressed = seqlen // self.compress_ratio
        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)

        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_forward_before_topk(self, seqlen):
        """Test CSAIndexer forward_before_topk."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        q, k, weights = self.indexer.forward_before_topk(x, qr)

        assert q.shape == (
            seqlen,
            batch_size,
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        n_compressed = seqlen // self.compress_ratio
        assert k.shape == (n_compressed, batch_size, self.config.dsa_indexer_head_dim)
        assert weights.shape == (seqlen, batch_size, self.config.dsa_indexer_n_heads)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_weights_projection_disables_fp8(self, seqlen, monkeypatch):
        tracker = _DisabledContextTracker()
        self.indexer.cuda()
        original_forward = self.indexer.linear_weights_proj.forward

        def checked_forward(*args, **kwargs):
            assert tracker.depth > 0, "indexer weights projection ran under FP8"
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(self.indexer.linear_weights_proj, 'forward', checked_forward)
        monkeypatch.setattr(
            'megatron.core.transformer.experimental_attention_variant.csa.get_fp8_disabled_context',
            tracker,
        )
        x = torch.randn(seqlen, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        weights = self.indexer._project_weights(x)

        assert weights.shape == (seqlen, 1, self.config.dsa_indexer_n_heads)
        assert tracker.entries == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csa_indexer_with_mask(self, seqlen):
        """Test CSAIndexer with causal mask."""
        batch_size = 2
        self.indexer.cuda()

        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        n_compressed = seqlen // self.compress_ratio
        causal_mask = torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(seqlen, -1)
        positions = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
        causal_mask = (
            torch.where(causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        index_scores, topk_indices = self.indexer(x, qr, mask=causal_mask)

        effective_topk = min(self.config.dsa_indexer_topk, n_compressed)
        assert index_scores.shape == (batch_size, seqlen, n_compressed)
        assert topk_indices.shape == (batch_size, seqlen, effective_topk)


# ===========================================================================
# CompressedSparseAttention tests
# ===========================================================================


class TestCompressedSparseAttentionRatio1:
    """Test CompressedSparseAttention with compress_ratio=1 (window-only)."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[0, 0, 0, 0], csa_window_size=8)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        cls.csa = CompressedSparseAttention(
            config=cls.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=cls.pg_collection,
            rotary_pos_emb=rotary_pos_emb,
            compress_ratio=0,
        )

        yield
        Utils.destroy_model_parallel()

    def test_ratio1_no_compressor(self):
        """With ratio=1, compressor and indexer should not be built."""
        assert self.csa.compressor is None
        assert self.csa.indexer is None

    def test_mtp_layer_number_is_offset(self):
        """MTP attention layers are numbered after all decoder layers."""
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            compress_ratio=0,
            is_mtp_layer=True,
        )

        assert csa.layer_number == self.config.num_layers + 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_forward(self):
        """Test forward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ratio1_backward(self):
        """Test backward pass with window-only attention."""
        seq_len = 32
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        self.csa.train()
        self.csa.cuda()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = self.csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None


@pytest.mark.parametrize("compress_ratio", [4, 128])
class TestCompressedSparseAttentionCompressed:
    """Test CompressedSparseAttention with compress_ratio > 1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128],
            csa_window_size=8,
            dsa_indexer_topk=8,
            dsa_indexer_loss_coeff=1.0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _get_layer_number(self, compress_ratio):
        """Return a layer_number (1-indexed) whose compress_ratio matches."""
        for i, r in enumerate(self.config.csa_compress_ratios):
            if r == compress_ratio:
                return i + 1
        raise ValueError(f"No layer with compress_ratio={compress_ratio}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self, compress_ratio):
        """Test that compressor/indexer are conditionally built."""
        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        assert csa.compressor is not None
        if compress_ratio == 4:
            assert csa.indexer is not None
        elif compress_ratio == 128:
            assert csa.indexer is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self, compress_ratio):
        """Test forward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward(self, compress_ratio):
        """Test backward pass with compressed attention."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.train()

        query = (
            torch.randn(seq_len, batch_size, np_, hn, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, 1, hn, dtype=torch.float32).cuda().requires_grad_(True)
        )
        value = key.clone().detach().requires_grad_(True)
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None

        for name, param in csa.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eval_mode(self, compress_ratio):
        """Test forward pass in eval mode."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        layer_number = self._get_layer_number(compress_ratio)
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        ).cuda()
        csa.eval()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        with torch.no_grad():
            output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


# ===========================================================================
# _apply_rope tests
# ===========================================================================


class TestApplyRope:
    """Test ``_apply_rope`` — the layout-aware RoPE wrapper used by
    Compressor / CSAIndexer / hybrid-attention callers.

    Behaviours covered:

    * 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
      inputs both work (3-D gets a temporary head-dim unsqueeze).
    * Only the trailing ``pos_dim`` components are rotated; the leading
      ``nope_dim`` slice is bit-exact unchanged.
    * Both ``RotaryEmbedding`` (returns ``Tensor``) and
      ``YarnRotaryEmbedding`` (returns ``(emb, mscale)`` tuple) — DSv4
      hybrid silently swaps the class based on ``compress_ratio``.
    * Both unfused and fused (``config.apply_rope_fusion=True``) paths
      produce the same output (within bf16 precision).
    * For ``ratio > 1`` the rotary table is built at
      ``rotary_seq_len * ratio`` and strided by ``ratio``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(0)
        model_parallel_cuda_manual_seed(0)
        cls = request.cls
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        # head_dim 32 = nope 24 + pos 8
        cls.config = _make_mla_config(v_head_dim=32, qk_pos_emb_head_dim=8)
        yield
        Utils.destroy_model_parallel()

    def _make_rotary(self, kind: str):
        from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding

        pos_dim = self.config.qk_pos_emb_head_dim
        if kind == 'rope':
            return RotaryEmbedding(
                pos_dim, rotary_percent=1.0, rotary_base=10000, cp_group=self.pg_collection.cp
            )
        if kind == 'yarn':
            return YarnRotaryEmbedding(
                pos_dim,
                rotary_base=40000,
                scaling_factor=40,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=0.0,
                cp_group=self.pg_collection.cp,
            )
        raise ValueError(kind)

    def _config_with(self, *, apply_rope_fusion: bool):
        # Reuse the class-level config; only flip the fusion flag.
        cfg = self.config
        cfg.apply_rope_fusion = apply_rope_fusion
        return cfg

    _ROTARY_FUSION_COMBOS = [
        pytest.param('rope', False, id='rope-unfused'),
        pytest.param('rope', True, id='rope-fused'),
        pytest.param('yarn', False, id='yarn-unfused'),
        pytest.param('yarn', True, id='yarn-fused'),
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(("rotary_kind", "apply_rope_fusion"), _ROTARY_FUSION_COMBOS)
    @pytest.mark.parametrize("input_ndim", [3, 4], ids=['3d', '4d'])
    @pytest.mark.parametrize("ratio", [1, 4], ids=['ratio_1', 'ratio_4'])
    def test_apply_rope(self, rotary_kind, apply_rope_fusion, input_ndim, ratio):
        """Output shape == input shape; no NaN; nope-dim slice is
        bit-exact unchanged. Sweeps the valid combinations of rotary
        class × apply_rope_fusion × input rank × ratio. Yarn's
        tuple-return is covered by the ``'yarn-*'`` combos.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads = 8, 2, 4
        cfg = self._config_with(apply_rope_fusion=apply_rope_fusion)

        shape = (seq, batch, head_dim) if input_ndim == 3 else (seq, batch, heads, head_dim)
        x = torch.randn(*shape, dtype=torch.bfloat16, device='cuda')
        # ``fused_mla_rope_inplace`` mutates the input — give it a copy so
        # the nope-dim equality check below still has the original.
        out = _apply_rope(
            x.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert not torch.isnan(out).any()
        # The leading nope_dim slice is the identity portion of RoPE.
        assert torch.equal(
            out[..., :nope], x[..., :nope]
        ), "RoPE must not touch the first nope_dim components"
        # Trailing pos_dim should rotate at non-zero positions.
        pe_changed = (out[..., nope:] != x[..., nope:]).any(dim=-1).flatten()
        assert pe_changed[
            1:
        ].any(), "RoPE should rotate the trailing pos_dim components for seq > 0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_3d_input_matches_4d_with_single_head(self, rotary_kind):
        """For a single-head input, the 3-D ``(s, b, d)`` and 4-D
        ``(s, b, 1, d)`` invocations must produce numerically identical
        output (3-D path just inserts a temporary head dim).
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch = 8, 2
        cfg = self._config_with(apply_rope_fusion=False)

        x_3d = torch.randn(seq, batch, head_dim, dtype=torch.bfloat16, device='cuda')
        x_4d = x_3d.unsqueeze(-2)

        out_3d = _apply_rope(
            x_3d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_4d = _apply_rope(
            x_4d,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )

        assert out_3d.shape == x_3d.shape
        assert out_4d.shape == x_4d.shape
        assert torch.equal(out_3d, out_4d.squeeze(-2))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("rotary_kind", ['rope', 'yarn'])
    def test_ratio_strides_rotary_table(self, rotary_kind):
        """For ``ratio > 1``, the rotary table is built at
        ``rotary_seq_len * ratio`` and strided by ``ratio``. The result
        with ``ratio=k`` must equal an ``apply_rope`` call on the same
        positions of a length-``rotary_seq_len * k`` table.
        """
        rotary = self._make_rotary(rotary_kind).cuda()
        nope = self.config.v_head_dim - self.config.qk_pos_emb_head_dim
        pos = self.config.qk_pos_emb_head_dim
        head_dim = nope + pos
        seq, batch, heads, ratio = 4, 1, 2, 4
        cfg = self._config_with(apply_rope_fusion=False)

        x_comp = torch.randn(seq, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda')
        out_comp = _apply_rope(
            x_comp.clone(),
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        x_full = torch.zeros(
            seq * ratio, batch, heads, head_dim, dtype=torch.bfloat16, device='cuda'
        )
        x_full[::ratio][:seq] = x_comp
        out_full = _apply_rope(
            x_full,
            nope,
            pos,
            rotary,
            cfg,
            rotary_seq_len=seq * ratio,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        out_ref = out_full[::ratio][:seq]

        assert torch.allclose(out_comp, out_ref, rtol=1e-3, atol=1e-3), (
            f"ratio={ratio} stride mismatch: "
            f"max abs diff = {(out_comp - out_ref).abs().max().item():.3e}"
        )


# ===========================================================================
# csa_dense_mode tests
# ===========================================================================


class TestCompressedSparseAttentionDenseMode:
    """Test that csa_dense_mode=True disables the indexer for ratio=4 layers."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(
            csa_compress_ratios=[4, 128, 4, 128], csa_window_size=8, csa_dense_mode=True
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_disables_indexer_for_ratio4(self):
        """With csa_dense_mode=True, ratio=4 layers should NOT build an indexer."""
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        assert csa.compress_ratio == 4
        assert csa.compressor is not None, "Compressor should still be built"
        assert csa.indexer is None, "Indexer should be disabled in dense mode"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_mode_forward_ratio4(self):
        """Forward pass should work for ratio=4 in dense mode (uses all compressed positions)."""
        seq_len = 256
        batch_size = 2
        np_ = self.config.num_attention_heads
        hn = self.config.v_head_dim

        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
        ).cuda()

        query = torch.randn(seq_len, batch_size, np_, hn, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, 1, hn, dtype=torch.bfloat16).cuda()
        value = key.clone()
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        output = csa(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

        assert output.shape == (seq_len, batch_size, np_ * hn)
        assert not torch.isnan(output).any()


class TestCSAHighPrecisionParams:
    """Reference-checkpoint FP32 parameters survive BF16 model conversion."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _make_mla_config(csa_compress_ratios=[4, 4, 4, 4])
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ape_and_attn_sink_stay_fp32_after_bf16_conversion(self):
        from megatron.core.transformer.module import Float16Module

        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=4,
            name="decoder.layers.0.self_attention.core_attention",
        )

        assert csa.attn_sink.dtype == torch.float32
        assert csa.compressor.ape.dtype == torch.float32
        assert csa.indexer.compressor.ape.dtype == torch.float32

        bf16_module = Float16Module(config=self.config, module=csa)

        assert bf16_module.module.attn_sink.dtype == torch.float32
        assert bf16_module.module.compressor.ape.dtype == torch.float32
        assert bf16_module.module.indexer.compressor.ape.dtype == torch.float32
        assert bf16_module.module.compressor.linear_wkv.weight.dtype == torch.bfloat16
        assert bf16_module.module.compressor.linear_wgate.weight.dtype == torch.bfloat16
