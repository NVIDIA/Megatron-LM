# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_layout,
    dsa_masking,
)
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    assert_similarity as _assert_similarity,
)
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    run_absorbed_mla_dsa_parity,
)


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("kernel_backend", ["cudnn", "tilelang"])
@pytest.mark.parametrize("seqlen", [1024, 2048])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("use_sparse_loss", [False, True], ids=["dense_loss", "sparse_loss"])
def test_fused_absorbed_mla_dsa_matches_native(
    kernel_backend: str, seqlen: int, calculate_per_token_loss: bool, use_sparse_loss: bool
) -> None:
    run_absorbed_mla_dsa_parity(
        kernel_backend=kernel_backend,
        seqlen=seqlen,
        attention_backend=AttnBackend.auto,
        calculate_per_token_loss=calculate_per_token_loss,
        use_sparse_loss=use_sparse_loss,
        num_iterations=10,
    )


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("seqlen", [1024, 2048, 4096])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("use_sparse_loss", [False, True], ids=["dense_loss", "sparse_loss"])
def test_unfused_absorbed_mla_dsa_matches_native(
    seqlen: int, calculate_per_token_loss: bool, use_sparse_loss: bool
) -> None:
    run_absorbed_mla_dsa_parity(
        kernel_backend="none",
        seqlen=seqlen,
        attention_backend=AttnBackend.unfused,
        calculate_per_token_loss=calculate_per_token_loss,
        use_sparse_loss=use_sparse_loss,
        num_iterations=10,
    )


def _skip_if_fused_dsa_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused DSA parity")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cudnn fused DSA path requires SM90+")
    missing = []
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        missing.append("cudnn-frontend DSA (nvidia-cudnn-frontend[cutedsl]>=1.24.1)")
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        missing.append("flash_mla")
    if missing:
        pytest.skip(f"fused DSA dependencies are not available: {', '.join(missing)}")


def test_bytes_to_chunk_rows_stays_within_budget_below_alignment():
    bytes_per_row = 3 * 1024 * 1024
    max_bytes = 1024 * 1024 * 1024

    chunk_rows = dsa_cudnn_kernels._bytes_to_chunk_rows(
        n_rows=2048, bytes_per_row=bytes_per_row, max_bytes=max_bytes, alignment=512
    )

    assert chunk_rows == max_bytes // bytes_per_row
    assert chunk_rows * bytes_per_row <= max_bytes


class _SingleRankTensorParallel:
    def size(self) -> int:
        return 1


class _SingleRankProcessGroups:
    tp = _SingleRankTensorParallel()


class _PackedCpCudnnConfig:
    dsa_kernel_backend = "cudnn"
    attention_backend = AttnBackend.auto
    kv_lora_rank = 512

    def __init__(self, calculate_per_token_loss: bool):
        self.calculate_per_token_loss = calculate_per_token_loss


def _make_packed_cp_varlen_cudnn_case(*, calculate_per_token_loss: bool):
    """Build a small real-kernel packed-THD CP-style absorbed-MLA case."""
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    device = torch.device("cuda")
    cp_size = 2
    cp_rank = 1
    # Keep this as one packed sequence so the test exercises the production
    # single_packed_thd_sequence fast path rather than an impossible flag combo.
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32, device=device)
    skv = int(cu_seqlens[-1].item())
    batch = 1
    heads = 8
    attn_dim = 576
    latent_v_channels = 512
    projected_v_channels = 32
    indexer_heads = 64
    indexer_dim = 128
    index_topk = 64
    loss_coeff = 0.01
    softmax_scale = attn_dim**-0.5

    query_positions, _key_reorder_idx = (
        dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
        )
    )
    varlen_starts, varlen_ends = dsa_masking.generate_varlen_mask_params_for_positions(
        cu_seqlens, query_positions
    )
    sq = query_positions.numel()

    query_global = torch.randn((skv, batch, heads, attn_dim), device=device, dtype=torch.bfloat16)
    key = torch.randn((skv, batch, 1, attn_dim), device=device, dtype=torch.bfloat16)
    q_indexer_global = torch.randn(
        (skv, batch, indexer_heads, indexer_dim), device=device, dtype=torch.bfloat16
    )
    k_indexer = torch.randn((skv, batch, indexer_dim), device=device, dtype=torch.bfloat16)
    weights_global = torch.randn((skv, batch, indexer_heads), device=device, dtype=torch.bfloat16)
    up_v_weight = torch.randn(
        (heads, projected_v_channels, latent_v_channels), device=device, dtype=torch.bfloat16
    )
    query_valid_rows = torch.ones((batch, sq), dtype=torch.bool, device=device)

    query = query_global.index_select(0, query_positions).contiguous()
    q_indexer = q_indexer_global.index_select(0, query_positions).contiguous()
    weights = weights_global.index_select(0, query_positions).contiguous()

    config = _PackedCpCudnnConfig(calculate_per_token_loss=calculate_per_token_loss)
    pg_collection = _SingleRankProcessGroups()
    return {
        "config": config,
        "pg_collection": pg_collection,
        "query": query,
        "key": key,
        "up_v_weight": up_v_weight,
        "q_indexer": q_indexer,
        "k_indexer": k_indexer,
        "weights": weights,
        "index_topk": index_topk,
        "softmax_scale": softmax_scale,
        "loss_coeff": loss_coeff,
        "varlen_starts": varlen_starts,
        "varlen_ends": varlen_ends,
        "query_valid_rows": query_valid_rows,
        "cp_size": cp_size,
        "cp_rank": cp_rank,
        "latent_v_channels": latent_v_channels,
    }


def _reference_absorbed_output_and_sparse_loss(case, topk_indices: torch.Tensor):
    index_scores, _topk_ref = dsa_module.fused_qk_topk_naive(
        case["q_indexer"],
        case["k_indexer"],
        case["weights"],
        case["index_topk"],
        mask=None,
        varlen_starts=case["varlen_starts"],
        varlen_ends=case["varlen_ends"],
        key_positions=None,
        use_relu=True,
    )
    latent_out = dsa_module._unfused_absorbed_dsa_fn(
        case["query"],
        case["key"],
        topk_indices,
        case["softmax_scale"],
        case["latent_v_channels"],
        mask=None,
        varlen_starts=case["varlen_starts"],
        varlen_ends=case["varlen_ends"],
        key_positions=None,
    )
    output = torch.einsum("sbhc,hdc->sbhd", latent_out, case["up_v_weight"]).contiguous()
    output = output.view(output.size(0), output.size(1), -1)

    key_for_loss = case["key"].expand(-1, -1, case["query"].size(2), -1)
    loss = dsa_module.compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        case["query"],
        key_for_loss,
        case["softmax_scale"],
        case["loss_coeff"],
        sparse_loss=True,
        pg_collection=case["pg_collection"],
        mask=None,
        varlen_starts=case["varlen_starts"],
        varlen_ends=case["varlen_ends"],
        key_positions=None,
        query_valid_rows=case["query_valid_rows"],
        calculate_per_token_loss=case["config"].calculate_per_token_loss,
    )
    return output, loss


def test_cudnn_indexer_topk_varlen_uses_logical_query_positions(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            b, sq, _, _ = q_bshd.shape
            sk = k_bshd.size(1)
            key_ids = torch.arange(sk, dtype=torch.float32).view(1, 1, sk).expand(b, sq, sk)
            query_ids = torch.arange(sq).view(1, sq, 1)
            scores = key_ids.clone().masked_fill(
                torch.arange(sk).view(1, 1, sk) > query_ids, float("-inf")
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    q = torch.ones((1, 2, 1, 1))
    k = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    w = torch.ones((1, 2, 1))
    starts = torch.tensor([3, 6], dtype=torch.int64)
    ends = torch.tensor([5, 8], dtype=torch.int64)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q, k, w, topk=4, varlen_starts=starts, varlen_ends=ends, key_positions=None
    )

    expected = torch.tensor([[[4, 3, -1, -1], [7, 6, -1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(topk_indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_score_chunks_preserve_global_query_offsets(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            b, sq, idx_nh, _ = q_bshd.shape
            sk = k_bshd.size(1)
            scores = torch.zeros((b, sq, sk), dtype=torch.float32)
            k_bdk = k_bshd[:, :, 0, :].float().transpose(1, 2).contiguous()
            for head_idx in range(idx_nh):
                head_scores = torch.bmm(q_bshd[:, :, head_idx, :].float(), k_bdk).relu()
                head_scores = head_scores * w_bsh[:, :, head_idx].float().unsqueeze(-1)
                scores = scores + head_scores
            local_seq_lens = ((torch.arange(sq) + 1) // ratio).clamp(max=sk)
            key_ids = torch.arange(sk).view(1, 1, sk)
            scores.masked_fill_(key_ids >= local_seq_lens.view(1, sq, 1), float("-inf"))
            return {"scores": scores * sm_scale}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            values, indices = scores.topk(top_k, dim=-1)
            result = {"indices": indices.to(torch.int32)}
            if return_val:
                result["values"] = values
            return result

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda b, sq, sk: 2)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 4, 1, 1)),
        torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1),
        torch.ones((1, 4, 1)),
        topk=4,
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices,
        torch.tensor(
            [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]], dtype=torch.int32
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_tie_break_prefers_lower_indices(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            return {"scores": torch.zeros((1, 1, 4), dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is False
            values, indices = scores_flat.topk(top_k, dim=-1)
            return {"indices": indices.to(torch.int32), "values": values if return_val else None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 4, 1)),
        torch.zeros((1, 1, 1)),
        topk=2,
        varlen_starts=torch.tensor([0], dtype=torch.int64),
        varlen_ends=torch.tensor([4], dtype=torch.int64),
        return_scores=False,
        use_local_indexer_varlen=True,
    )

    torch.testing.assert_close(topk_indices, torch.tensor([[[0, 1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32))


def test_cudnn_indexer_topk_repeated_varlen_ends_keep_distinct_query_rows(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            raise AssertionError("repeated varlen ends must not scatter through one score row")

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            values, indices = scores.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 2, 1, 1)),
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        torch.tensor([[[1.0], [-1.0]]]),
        topk=2,
        varlen_starts=torch.tensor([0, 0], dtype=torch.int64),
        varlen_ends=torch.tensor([2, 2], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        return_topk_scores=True,
    )

    torch.testing.assert_close(
        topk_indices, torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(topk_length, torch.tensor([[2, 2]], dtype=torch.int32))
    torch.testing.assert_close(topk_scores, torch.tensor([[[2.0, 1.0], [-1.0, -2.0]]]))


def test_cudnn_indexer_topk_nonlocal_varlen_ends_avoid_global_score_rows(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            raise AssertionError("non-local varlen ends must not allocate global score rows")

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 2, 1, 1)),
        torch.arange(8, dtype=torch.float32).view(1, 8, 1),
        torch.ones((1, 2, 1)),
        topk=4,
        varlen_starts=torch.tensor([5, 6], dtype=torch.int64),
        varlen_ends=torch.tensor([7, 8], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[5, 6, -1, -1], [6, 7, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(topk_length, torch.tensor([[2, 2]], dtype=torch.int32))


def test_cudnn_indexer_topk_local_varlen_keeps_compact_query_rows(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            raise AssertionError("local varlen must score with logical ends before masking")

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 2, 1, 1)),
        torch.arange(8, dtype=torch.float32).view(1, 8, 1),
        torch.ones((1, 2, 1)),
        topk=4,
        varlen_starts=torch.tensor([3, 6], dtype=torch.int64),
        varlen_ends=torch.tensor([5, 8], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        use_local_indexer_varlen=True,
    )

    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[3, 4, -1, -1], [6, 7, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_multi_packed_cp_uses_segmented_thd(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(
            q, k, weights, ratio, sm_scale, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        ):
            del weights, ratio, sm_scale
            seen["q_shape"] = q.shape
            seen["k_shape"] = k.shape
            seen["cu_q"] = cu_seqlens_q.clone()
            seen["cu_k"] = cu_seqlens_k.clone()
            seen["max_q"] = max_seqlen_q
            seen["max_k"] = max_seqlen_k
            scores = torch.full((q.size(0), max_seqlen_k), float("-inf"))
            for segment in range(cu_seqlens_q.numel() - 1):
                q_start = int(cu_seqlens_q[segment])
                q_end = int(cu_seqlens_q[segment + 1])
                k_length = int(cu_seqlens_k[segment + 1] - cu_seqlens_k[segment])
                q_length = q_end - q_start
                for row in range(q_length):
                    valid_k = k_length - q_length + row + 1
                    scores[q_start + row, :valid_k] = torch.arange(valid_k, dtype=torch.float32)
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores, seq_lens, top_k, next_n, return_val):
            del next_n
            masked_scores = scores.clone()
            key_ids = torch.arange(scores.size(1)).view(1, -1)
            masked_scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            values, indices = masked_scores.topk(top_k, dim=-1)
            return {"indices": indices.to(torch.int32), "values": values if return_val else None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    query_positions = torch.tensor([0, 1, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23])
    starts = torch.tensor([0] * 4 + [8] * 8)
    ends = query_positions + 1
    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 12, 1, 1)),
        torch.ones((1, 24, 1)),
        torch.ones((1, 12, 1)),
        topk=3,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=torch.tensor([0, 8, 24], dtype=torch.int32),
        packed_cu_seqlens_k=torch.tensor([0, 8, 24], dtype=torch.int32),
        packed_max_seqlen_q=16,
        packed_max_seqlen_k=16,
        packed_cp_size=2,
        local_packed_cp_rank=0,
    )

    assert seen["q_shape"] == torch.Size([12, 1, 1])
    assert seen["k_shape"] == torch.Size([30, 1, 1])
    torch.testing.assert_close(seen["cu_q"], torch.tensor([0, 2, 4, 8, 12], dtype=torch.int32))
    torch.testing.assert_close(seen["cu_k"], torch.tensor([0, 2, 10, 14, 30], dtype=torch.int32))
    assert seen["max_q"] == 4
    assert seen["max_k"] == 16
    expected_indices = []
    expected_scores = []
    expected_lengths = []
    for position, start in zip(query_positions.tolist(), starts.tolist()):
        valid_count = position - start + 1
        row = list(range(position, max(start - 1, position - 3), -1))
        expected_indices.append(row + [-1] * (3 - len(row)))
        local_scores = [value - start for value in row]
        expected_scores.append(
            local_scores + [torch.finfo(torch.float32).min] * (3 - len(local_scores))
        )
        expected_lengths.append(min(valid_count, 3))
    expected_indices = torch.tensor([expected_indices], dtype=torch.int32)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(
        topk_length, torch.tensor([expected_lengths], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        topk_scores, torch.tensor([expected_scores], dtype=torch.float32), rtol=0, atol=0
    )


def test_cudnn_split_topk_threads_multi_packed_cp_metadata(monkeypatch):
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8, 24], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8, 24], dtype=torch.int32),
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    q = torch.ones((12, 1, 1, 1))
    k = torch.ones((24, 1, 1))
    weights = torch.ones((12, 1, 1))
    starts = torch.zeros(12, dtype=torch.int64)
    ends = torch.arange(1, 13, dtype=torch.int64)
    seen = {}

    def fake_indexer_topk(q_bshd, _k_bsd, _w_bsh, topk, **kwargs):
        seen["loss_off"] = kwargs
        return (
            torch.zeros((q_bshd.size(0), q_bshd.size(1), topk), dtype=torch.int32),
            torch.full((q_bshd.size(0), q_bshd.size(1)), topk, dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    dsa_cudnn_kernels.run_fused_qk_topk(
        q,
        k,
        weights,
        index_topk=3,
        starts=starts,
        ends=ends,
        block_size=128,
        use_local_indexer_varlen=True,
        packed_seq_params=packed_seq_params,
        cp_size=2,
    )

    def fake_sparse_loss_apply(*args):
        seen["loss_on"] = args
        return (
            torch.zeros((1, 12, 3), dtype=torch.int32),
            torch.full((1, 12), 3, dtype=torch.int32),
            torch.tensor(0.0),
        )

    monkeypatch.setattr(
        dsa_cudnn_kernels,
        "FusedQKTopKWithSparseLossFunc",
        SimpleNamespace(apply=fake_sparse_loss_apply),
    )
    dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
        q,
        k,
        weights,
        index_topk=3,
        starts=starts,
        ends=ends,
        block_size=128,
        query=torch.ones((12, 1, 64, 576)),
        key=torch.ones((24, 1, 1, 576)),
        softmax_scale=1.0,
        loss_coeff=0.001,
        pg_collection=SimpleNamespace(tp=None),
        config=SimpleNamespace(kv_lora_rank=512),
        use_local_indexer_varlen=True,
        packed_seq_params=packed_seq_params,
        cp_size=2,
    )

    loss_off = seen["loss_off"]
    torch.testing.assert_close(loss_off["packed_cu_seqlens_q"], packed_seq_params.cu_seqlens_q)
    torch.testing.assert_close(loss_off["packed_cu_seqlens_k"], packed_seq_params.cu_seqlens_kv)
    assert loss_off["packed_max_seqlen_q"] == 16
    assert loss_off["packed_max_seqlen_k"] == 16
    assert loss_off["packed_cp_size"] == 2

    loss_on = seen["loss_on"]
    torch.testing.assert_close(loss_on[18], packed_seq_params.cu_seqlens_q)
    torch.testing.assert_close(loss_on[19], packed_seq_params.cu_seqlens_kv)
    assert loss_on[20:23] == (16, 16, 2)


def test_cudnn_indexer_topk_single_packed_cp_uses_absolute_seq_lens(monkeypatch):
    seen = {"forward_shapes": [], "topk_seq_lens": []}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            seen["forward_shapes"].append((q_bshd.shape[1], k_bshd.shape[1]))
            b, sq, _, _ = q_bshd.shape
            sk = k_bshd.shape[1]
            return {
                "scores": torch.arange(sk, dtype=torch.float32).view(1, 1, sk).expand(b, sq, sk)
            }

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            seen["topk_seq_lens"].append(seq_lens.detach().clone())
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            values, indices = scores.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 4, 1, 1)),
        torch.arange(8, dtype=torch.float32).view(1, 8, 1),
        torch.ones((1, 4, 1)),
        topk=3,
        varlen_starts=torch.zeros(4, dtype=torch.int64),
        varlen_ends=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=1,
    )

    assert seen["forward_shapes"] == [(2, 4), (2, 6)]
    assert [seq_lens.tolist() for seq_lens in seen["topk_seq_lens"]] == [[3, 4], [5, 6]]
    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3]]], dtype=torch.int32),
    )
    torch.testing.assert_close(topk_length, torch.tensor([[3, 3, 3, 3]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores,
        torch.tensor([[[2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [4.0, 3.0, 2.0], [5.0, 4.0, 3.0]]]),
    )


def test_cudnn_indexer_topk_single_packed_cp_prefix_crops_keys_per_chunk(monkeypatch):
    seen = {"forward_shapes": []}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            seen["forward_shapes"].append((q_bshd.shape[1], k_bshd.shape[1]))
            b, sq, _, _ = q_bshd.shape
            sk = k_bshd.shape[1]
            return {
                "scores": torch.arange(sk, dtype=torch.float32).view(1, 1, sk).expand(b, sq, sk)
            }

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            values, indices = scores.topk(top_k, dim=-1)
            return {"indices": indices.to(torch.int32), "values": values if return_val else None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda *_args: 1)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 2, 1, 1)),
        torch.arange(8, dtype=torch.float32).view(1, 8, 1),
        torch.ones((1, 2, 1)),
        topk=2,
        varlen_starts=torch.zeros(2, dtype=torch.int64),
        varlen_ends=torch.tensor([1, 2], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=1,
        local_packed_cp_query_len=8,
    )

    assert seen["forward_shapes"] == [(1, 1), (1, 2)]
    torch.testing.assert_close(topk_indices, torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[1, 2]], dtype=torch.int32))


def test_cudnn_indexer_topk_single_packed_cp_real_kernel_uses_bottom_right_alignment(monkeypatch):
    _skip_if_fused_dsa_unavailable()
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda *_args: 1)

    query = torch.zeros((1, 4, 32, 128), device="cuda", dtype=torch.bfloat16)
    query[..., 0, 0] = 1
    key = torch.zeros((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    key[..., 0] = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    weights = torch.zeros((1, 4, 32), device="cuda", dtype=torch.bfloat16)
    weights[..., 0] = 1
    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        query,
        key,
        weights,
        topk=2,
        varlen_starts=torch.zeros(4, device="cuda", dtype=torch.int64),
        varlen_ends=torch.tensor([3, 4, 13, 14], device="cuda", dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=1,
    )

    order = topk_indices.argsort(dim=-1)
    sorted_indices = torch.gather(topk_indices, dim=-1, index=order)
    sorted_scores = torch.gather(topk_scores, dim=-1, index=order)
    torch.testing.assert_close(
        sorted_indices.cpu(),
        torch.tensor([[[1, 2], [2, 3], [11, 12], [12, 13]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length.cpu(), torch.tensor([[2, 2, 2, 2]], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        sorted_scores.cpu(),
        torch.tensor([[[1.0, 2.0], [2.0, 3.0], [11.0, 12.0], [12.0, 13.0]]]),
        rtol=0,
        atol=0,
    )


def test_cudnn_indexer_topk_indices_only_filters_masked_varlen_scores(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            scores = (
                torch.arange(8, dtype=torch.float32)
                .view(1, 1, 8)
                .expand(q_bshd.size(0), q_bshd.size(1), 8)
                .clone()
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is False
            scores = scores_flat.clone()
            key_ids = torch.arange(scores.size(1), device=scores.device).view(1, -1)
            scores.masked_fill_(key_ids >= seq_lens.view(-1, 1), float("-inf"))
            return {"indices": scores.topk(top_k, dim=-1).indices.to(torch.int32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 2, 1, 1)),
        torch.arange(8, dtype=torch.float32).view(1, 8, 1),
        torch.ones((1, 2, 1)),
        topk=4,
        varlen_starts=torch.tensor([3, 6], dtype=torch.int64),
        varlen_ends=torch.tensor([5, 8], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[[3, 4, -1, -1], [6, 7, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        topk_length, torch.tensor([[2, 2]], dtype=torch.int32), rtol=0, atol=0
    )


def test_cudnn_indexer_topk_indices_only_compacts_invalid_prefix_entries(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale):
            scores = (
                torch.arange(8, dtype=torch.float32)
                .view(1, 1, 8)
                .expand(q_bshd.size(0), q_bshd.size(1), 8)
                .clone()
            )
            return {"scores": scores}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is False
            indices = torch.tensor([[4, 6, 3, 5]], dtype=torch.int32)
            return {"indices": indices.to(device=scores_flat.device)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 8, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        varlen_starts=torch.tensor([3], dtype=torch.int64),
        varlen_ends=torch.tensor([5], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
    )

    torch.testing.assert_close(
        topk_indices, torch.tensor([[[3, 4, -1, -1]]], dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32), rtol=0, atol=0)


def test_cudnn_indexer_topk_can_return_topk_scores(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            return {"scores": torch.tensor([[[0.5, float("-inf"), 2.0, 1.0]]], dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            seen["return_val"] = return_val
            values, indices = scores_flat.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 4, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        return_scores=False,
        return_topk_scores=True,
    )

    assert seen["return_val"] is True
    torch.testing.assert_close(topk_indices, torch.tensor([[[0, -1, -1, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[1]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores,
        torch.tensor(
            [
                [
                    [
                        0.5,
                        torch.finfo(torch.float32).min,
                        torch.finfo(torch.float32).min,
                        torch.finfo(torch.float32).min,
                    ]
                ]
            ]
        ),
    )


def test_cudnn_indexer_topk_tie_break_does_not_bias_selected_scores(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            return {"scores": torch.zeros((1, 1, 4), dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            seen["scores_flat"] = scores_flat.detach().clone()
            assert return_val is True
            values, indices = scores_flat.topk(top_k, dim=-1)
            return {"indices": indices.to(torch.int32), "values": values if return_val else None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 4, 1)),
        torch.zeros((1, 1, 1)),
        topk=2,
        varlen_starts=torch.tensor([0], dtype=torch.int64),
        varlen_ends=torch.tensor([4], dtype=torch.int64),
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
    )

    assert seen["scores_flat"][0, 0] > seen["scores_flat"][0, 1]
    torch.testing.assert_close(topk_indices, torch.tensor([[[0, 1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32))
    torch.testing.assert_close(topk_scores, torch.zeros((1, 1, 2), dtype=torch.float32))


def test_cudnn_indexer_topk_scores_reapply_varlen_bounds(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            return {"scores": torch.zeros((1, 1, 8), dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            assert return_val is True
            indices = torch.tensor([[5, 3, 4, 1]], dtype=torch.int32)
            return {
                "indices": indices,
                "values": torch.gather(scores_flat, dim=1, index=indices.long()),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.zeros((1, 1, 1, 1)),
        torch.zeros((1, 8, 1)),
        torch.zeros((1, 1, 1)),
        topk=4,
        varlen_starts=torch.tensor([3], dtype=torch.int64),
        varlen_ends=torch.tensor([5], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
    )

    torch.testing.assert_close(topk_indices, torch.tensor([[[3, 4, -1, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores,
        torch.tensor(
            [[[0.0, 0.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]
        ),
    )


def test_cudnn_attention_topk_preparation_preserves_valid_prefix():
    topk_indices = torch.tensor([[[3, -1, 1, 2], [4, 0, -1, -1]]], dtype=torch.int32)

    prepared, topk_length = dsa_cudnn_kernels._prepare_attention_topk_indices(topk_indices, sk=5)

    torch.testing.assert_close(
        prepared, torch.tensor([[[1, 2, 3, -1], [0, 4, -1, -1]]], dtype=torch.int32)
    )
    torch.testing.assert_close(topk_length, torch.tensor([[3, 2]], dtype=torch.int32))


def test_cudnn_split_topk_hook_uses_indexer_topk(monkeypatch):
    seen = {}

    def fake_indexer_topk(
        q_bshd,
        k_bsd,
        w_bsh,
        topk,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        return_scores=True,
        return_topk_scores=False,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        packed_cu_seqlens_q=None,
        packed_cu_seqlens_k=None,
        packed_max_seqlen_q=None,
        packed_max_seqlen_k=None,
        packed_cp_size=1,
    ):
        seen["q_shape"] = q_bshd.shape
        seen["k_shape"] = k_bsd.shape
        seen["w_shape"] = w_bsh.shape
        seen["topk"] = topk
        seen["varlen_starts"] = varlen_starts
        seen["varlen_ends"] = varlen_ends
        seen["key_positions"] = key_positions
        seen["return_scores"] = return_scores
        seen["return_topk_scores"] = return_topk_scores
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        seen["single_packed_thd_sequence"] = single_packed_thd_sequence
        seen["local_packed_cp_rank"] = local_packed_cp_rank
        seen["local_packed_cp_query_start"] = local_packed_cp_query_start
        seen["local_packed_cp_query_len"] = local_packed_cp_query_len
        return (
            torch.tensor([[[1, 0, -1], [2, 1, 0]]], dtype=torch.int32),
            torch.tensor([[2, 3]], dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)

    starts = torch.tensor([0, 1], dtype=torch.int32)
    ends = torch.tensor([2, 4], dtype=torch.int32)
    topk_indices, topk_length = dsa_cudnn_kernels.run_fused_qk_topk(
        q=torch.zeros((2, 1, 3, 4), dtype=torch.bfloat16),
        k=torch.zeros((4, 1, 4), dtype=torch.bfloat16),
        weights=torch.zeros((2, 1, 3), dtype=torch.bfloat16),
        index_topk=3,
        starts=starts,
        ends=ends,
        block_size=128,
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=7,
    )

    torch.testing.assert_close(
        topk_indices, torch.tensor([[[1, 0, -1], [2, 1, 0]]], dtype=torch.int32)
    )
    torch.testing.assert_close(topk_length, torch.tensor([[2, 3]], dtype=torch.int32))
    assert seen["q_shape"] == (1, 2, 3, 4)
    assert seen["k_shape"] == (1, 4, 4)
    assert seen["w_shape"] == (1, 2, 3)
    assert seen["topk"] == 3
    torch.testing.assert_close(seen["varlen_starts"], starts)
    torch.testing.assert_close(seen["varlen_ends"], ends)
    assert seen["key_positions"] is None
    assert seen["return_scores"] is False
    assert seen["return_topk_scores"] is False
    assert seen["use_local_indexer_varlen"] is True
    assert seen["single_packed_thd_sequence"] is True
    assert seen["local_packed_cp_rank"] == 7


@pytest.mark.parametrize("hook", ["split_topk", "split_topk_loss", "full_fusion"])
def test_cudnn_fused_hooks_reject_non_relu_scoring(hook):
    with pytest.raises(RuntimeError, match="dsa_indexer_scoring_relu=True"):
        if hook == "split_topk":
            dsa_cudnn_kernels.run_fused_qk_topk(
                q=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                k=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                weights=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                index_topk=1,
                starts=torch.tensor([0], dtype=torch.int32),
                ends=torch.tensor([1], dtype=torch.int32),
                block_size=128,
                use_relu=False,
            )
        elif hook == "split_topk_loss":
            dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
                q=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                k=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                weights=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                index_topk=1,
                starts=torch.tensor([0], dtype=torch.int32),
                ends=torch.tensor([1], dtype=torch.int32),
                block_size=128,
                query=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                key=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                softmax_scale=1.0,
                loss_coeff=0.01,
                pg_collection=object(),
                use_relu=False,
            )
        else:

            class Config:
                dsa_kernel_backend = "cudnn"
                attention_backend = AttnBackend.auto
                kv_lora_rank = 4
                calculate_per_token_loss = True

            dsa_cudnn_kernels.run_fused_dsa_attention(
                config=Config(),
                query=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                key=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                value=None,
                up_v_weight=None,
                q_indexer=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
                k_indexer=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                indexer_weights=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
                indexer_topk=1,
                softmax_scale=1.0,
                loss_coeff=0.0,
                sparse_loss=False,
                calculate_per_token_loss=True,
                absorbed_mla=True,
                cp_size=1,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=None,
                varlen_starts=None,
                varlen_ends=None,
                key_positions=None,
                query_valid_rows=None,
                use_relu=False,
            )


def test_cudnn_split_topk_with_loss_returns_precomputed_indexer_grads(monkeypatch):
    class Config:
        kv_lora_rank = 512

    seen = {}

    def fake_indexer_topk(
        q_bshd,
        k_bsd,
        w_bsh,
        topk,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        return_scores=True,
        return_topk_scores=False,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        packed_cu_seqlens_q=None,
        packed_cu_seqlens_k=None,
        packed_max_seqlen_q=None,
        packed_max_seqlen_k=None,
        packed_cp_size=1,
    ):
        seen["return_scores"] = return_scores
        seen["return_topk_scores"] = return_topk_scores
        seen["varlen_starts"] = varlen_starts
        seen["varlen_ends"] = varlen_ends
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        seen["single_packed_thd_sequence"] = single_packed_thd_sequence
        seen["local_packed_cp_rank"] = local_packed_cp_rank
        seen["local_packed_cp_query_start"] = local_packed_cp_query_start
        seen["local_packed_cp_query_len"] = local_packed_cp_query_len
        return (
            torch.tensor([[[1, 0], [2, -1]]], dtype=torch.int32),
            torch.tensor([[2, 1]], dtype=torch.int32),
            torch.tensor([[[3.0, 1.0], [2.0, torch.finfo(torch.float32).min]]]),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["flash_topk"] = topk_idxs.detach().clone()
        seen["flash_topk_length"] = topk_length.detach().clone()
        seen["d_v"] = d_v
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    def fake_sparse_loss_and_grads(**kwargs):
        seen["loss_topk"] = kwargs["topk_indices_cmp"].detach().clone()
        seen["tp_group"] = kwargs["tp_group"]
        q_grad = torch.ones_like(kwargs["q_idx_bshd"]).permute(1, 0, 2, 3).contiguous()
        k_grad = torch.full_like(kwargs["k_idx_bsd"], 2.0).permute(1, 0, 2).contiguous()
        w_grad = torch.full_like(kwargs["w_bsh"], 3.0).permute(1, 0, 2).contiguous()
        return torch.tensor(4.0, dtype=torch.float32), q_grad, k_grad, w_grad

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_compute_sparse_indexer_loss_and_grads", fake_sparse_loss_and_grads
    )

    q = torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros((3, 1, 1), dtype=torch.bfloat16, requires_grad=True)
    weights = torch.zeros((2, 1, 1), dtype=torch.bfloat16, requires_grad=True)

    class FakeProcessGroupCollection:
        tp = object()

    pg_collection = FakeProcessGroupCollection()
    topk_indices, topk_length, indexer_loss = dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
        config=Config(),
        q=q,
        k=k,
        weights=weights,
        index_topk=2,
        starts=torch.tensor([0, 1], dtype=torch.int32),
        ends=torch.tensor([2, 3], dtype=torch.int32),
        block_size=128,
        query=torch.zeros((2, 1, 1, Config.kv_lora_rank), dtype=torch.bfloat16),
        key=torch.zeros((3, 1, 1, Config.kv_lora_rank), dtype=torch.bfloat16),
        softmax_scale=1.0,
        loss_coeff=0.01,
        pg_collection=pg_collection,
        calculate_per_token_loss=False,
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=5,
    )

    torch.testing.assert_close(topk_indices, torch.tensor([[[0, 1], [2, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2, 1]], dtype=torch.int32))
    assert indexer_loss.item() == 4.0
    assert seen["return_scores"] is False
    assert seen["return_topk_scores"] is True
    assert seen["use_local_indexer_varlen"] is True
    assert seen["single_packed_thd_sequence"] is True
    assert seen["local_packed_cp_rank"] == 5
    assert seen["tp_group"] is pg_collection.tp
    assert seen["d_v"] == Config.kv_lora_rank
    torch.testing.assert_close(
        seen["flash_topk"], torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["flash_topk_length"], torch.tensor([2, 1], dtype=torch.int32))
    torch.testing.assert_close(
        seen["loss_topk"], torch.tensor([[[1, 0], [2, -1]]], dtype=torch.int32)
    )

    indexer_loss.backward()

    torch.testing.assert_close(q.grad, torch.ones_like(q))
    torch.testing.assert_close(k.grad, torch.full_like(k, 2.0))
    torch.testing.assert_close(weights.grad, torch.full_like(weights, 3.0))


def test_cudnn_split_topk_with_loss_declines_unsupported_flashmla_value_dim(monkeypatch):
    class Config:
        kv_lora_rank = 4

    def fail_apply(*_args, **_kwargs):
        raise AssertionError("unsupported FlashMLA value dim must decline before apply")

    monkeypatch.setattr(dsa_cudnn_kernels.FusedQKTopKWithSparseLossFunc, "apply", fail_apply)

    assert (
        dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
            config=Config(),
            q=torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
            k=torch.zeros((3, 1, 1), dtype=torch.bfloat16),
            weights=torch.zeros((2, 1, 1), dtype=torch.bfloat16),
            index_topk=2,
            starts=torch.tensor([0, 1], dtype=torch.int32),
            ends=torch.tensor([2, 3], dtype=torch.int32),
            block_size=128,
            query=torch.zeros((2, 1, 1, 8), dtype=torch.bfloat16),
            key=torch.zeros((3, 1, 1, 8), dtype=torch.bfloat16),
            softmax_scale=1.0,
            loss_coeff=0.01,
            pg_collection=object(),
            use_relu=True,
        )
        is None
    )


def test_flash_mla_topk_alignment_uses_sm100_block(monkeypatch):
    # ``_device_sm`` is lru_cached and calls ``get_device_capability(device_index)``, so the
    # patch must accept the device arg, and the cache must be cleared so the simulated SM takes
    # effect — and cleared again afterward so the faked ``(10, 0)`` cannot leak into later
    # real-device tests (which would mis-pad heads and fault the real kernel).
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (10, 0))
    dsa_cudnn_kernels._device_sm.cache_clear()
    try:
        assert dsa_cudnn_kernels._get_topk_alignment() == 512
    finally:
        dsa_cudnn_kernels._device_sm.cache_clear()


def test_dsa_fwd_flash_mla_pads_topk_to_flashmla_block(monkeypatch):
    seen = {}

    def fake_flash_mla(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, indexer_topk):
        seen["indices"] = indices
        seen["topk_length"] = topk_length
        return (
            torch.zeros((q.size(0), q.size(1), d_v), dtype=q.dtype),
            torch.zeros((q.size(0), q.size(1)), dtype=torch.float32),
            torch.zeros((q.size(0), q.size(1)), dtype=torch.float32),
        )

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_flash_mla", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_flash_mla_sparse_fwd", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_get_topk_alignment", lambda: 512)
    monkeypatch.setattr(dsa_cudnn_kernels, "_get_head_padding", lambda num_heads: num_heads)

    topk_idxs = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32)
    topk_length = torch.tensor([3, 3], dtype=torch.int32)

    dsa_cudnn_kernels._dsa_fwd_flash_mla(
        torch.zeros((2, 2, 4), dtype=torch.bfloat16),
        torch.zeros((5, 4), dtype=torch.bfloat16),
        topk_idxs,
        1.0,
        d_v=4,
        attn_sink=torch.full((2,), float("-inf"), dtype=torch.float32),
        topk_length=topk_length,
    )

    assert seen["indices"].shape == (2, 1, 512)
    torch.testing.assert_close(seen["indices"][:, 0, :3], topk_idxs)
    torch.testing.assert_close(
        seen["indices"][:, 0, 3:], torch.full((2, 509), -1, dtype=torch.int32)
    )
    torch.testing.assert_close(seen["topk_length"], topk_length)


def test_cudnn_indexer_topk_scores_local_varlen_keeps_compact_query_rows(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio, sm_scale):
            raise AssertionError("local varlen must score with logical ends before masking")

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, next_n, return_val):
            values, indices = scores_flat.topk(top_k, dim=-1)
            if return_val:
                return {"indices": indices.to(torch.int32), "values": values}
            return {"indices": indices.to(torch.int32), "values": None}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    topk_indices, topk_length, topk_scores = dsa_cudnn_kernels._indexer_topk_bshd(
        torch.ones((1, 1, 1, 1)),
        torch.arange(4, dtype=torch.float32).view(1, 4, 1),
        torch.ones((1, 1, 1)),
        topk=4,
        varlen_starts=torch.tensor([2], dtype=torch.int64),
        varlen_ends=torch.tensor([4], dtype=torch.int64),
        key_positions=None,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
    )

    torch.testing.assert_close(topk_indices, torch.tensor([[[3, 2, -1, -1]]], dtype=torch.int32))
    torch.testing.assert_close(topk_length, torch.tensor([[2]], dtype=torch.int32))
    torch.testing.assert_close(
        topk_scores,
        torch.tensor(
            [[[3.0, 2.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]
        ),
    )


def test_cudnn_sparse_attn_target_uses_frontend_wrapper(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            softmax_scale,
            qhead_per_kv_head=None,
            topk_indices_global=True,
        ):
            seen["q_attn"] = q_attn
            seen["k_attn"] = k_attn
            seen["lse_is_contiguous"] = lse.is_contiguous()
            seen["topk_indices_is_contiguous"] = topk_indices.is_contiguous()
            seen["softmax_scale"] = softmax_scale
            seen["qhead_per_kv_head"] = qhead_per_kv_head
            seen["topk_indices_global"] = topk_indices_global
            return {"target": torch.ones_like(topk_indices, dtype=torch.float32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    q = torch.randn((1, 2, 8, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 3, 4), dtype=torch.bfloat16)
    lse = torch.randn((1, 2, 8), dtype=torch.float32).transpose(1, 2).transpose(1, 2)
    topk_indices = torch.tensor([[[0, 1, -1], [2, 0, 1]]], dtype=torch.int32)

    target = dsa_cudnn_kernels._compute_attn_target(
        q, k, lse, topk_indices, topk_length=None, softmax_scale=0.5, qhead_per_kv_head=8
    )

    torch.testing.assert_close(target, torch.ones_like(target))
    assert seen["q_attn"] is q
    assert seen["k_attn"] is k
    assert seen["lse_is_contiguous"] is True
    assert seen["topk_indices_is_contiguous"] is True
    assert seen["softmax_scale"] == 0.5
    assert seen["qhead_per_kv_head"] == 8
    assert seen["topk_indices_global"] is False


def test_cudnn_sparse_attn_target_pads_small_local_head_count(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            softmax_scale,
            qhead_per_kv_head=None,
            topk_indices_global=True,
        ):
            seen["q_attn_shape"] = q_attn.shape
            seen["lse_shape"] = lse.shape
            seen["q_real"] = q_attn[:, :, :4, :].clone()
            seen["q_pad_abs_sum"] = q_attn[:, :, 4:, :].abs().sum()
            seen["lse_real"] = lse[:, :, :4].clone()
            seen["lse_pad_is_inf"] = torch.isinf(lse[:, :, 4:]).all()
            seen["qhead_per_kv_head"] = qhead_per_kv_head
            seen["topk_indices_global"] = topk_indices_global
            return {"target": torch.ones_like(topk_indices, dtype=torch.float32)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    q = torch.randn((1, 2, 4, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 3, 4), dtype=torch.bfloat16)
    lse = torch.randn((1, 2, 4), dtype=torch.float32)
    topk_indices = torch.tensor([[[0, 1, -1], [2, 0, 1]]], dtype=torch.int32)

    target = dsa_cudnn_kernels._compute_attn_target(
        q, k, lse, topk_indices, topk_length=None, softmax_scale=0.5, qhead_per_kv_head=4
    )

    torch.testing.assert_close(target, torch.ones_like(target))
    assert seen["q_attn_shape"] == (1, 2, 8, 4)
    assert seen["lse_shape"] == (1, 2, 8)
    torch.testing.assert_close(seen["q_real"], q)
    assert seen["q_pad_abs_sum"].item() == 0
    torch.testing.assert_close(seen["lse_real"], lse)
    assert seen["lse_pad_is_inf"].item() is True
    assert seen["qhead_per_kv_head"] == 8
    assert seen["topk_indices_global"] is False


def test_cudnn_sparse_loss_uses_selected_topk_scores(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_indexer_score_recompute_wrapper(*args, **kwargs):
            raise AssertionError("sparse indexer predict should use selected top-k scores")

        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["index_score"] = index_score
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_indexer_topk(
        q_bshd,
        k_bsd,
        w_bsh,
        topk,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        return_scores=True,
        return_topk_scores=False,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        packed_cu_seqlens_q=None,
        packed_cu_seqlens_k=None,
        packed_max_seqlen_q=None,
        packed_max_seqlen_k=None,
        packed_cp_size=1,
    ):
        del (
            single_packed_thd_sequence,
            local_packed_cp_rank,
            local_packed_cp_query_start,
            local_packed_cp_query_len,
            packed_cu_seqlens_q,
            packed_cu_seqlens_k,
            packed_max_seqlen_q,
            packed_max_seqlen_k,
            packed_cp_size,
        )
        seen["return_scores"] = return_scores
        seen["return_topk_scores"] = return_topk_scores
        return (
            torch.tensor([[[3, 1, 2, -1]]], dtype=torch.int32),
            torch.tensor([[3]], dtype=torch.int32),
            torch.tensor([[[3.0, 2.0, 0.0, torch.finfo(torch.float32).min]]]),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["flash_topk"] = topk_idxs
        seen["flash_topk_length"] = topk_length
        return torch.zeros((1, 1, d_v), dtype=q.dtype), torch.zeros((1, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        seen["loss_topk"] = topk_indices
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=True,
        d_v=1,
    )

    assert seen["return_scores"] is False
    assert seen["return_topk_scores"] is True
    torch.testing.assert_close(seen["flash_topk"], torch.tensor([[1, 2, 3, -1]], dtype=torch.int32))
    torch.testing.assert_close(seen["flash_topk_length"], torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(
        seen["loss_topk"], torch.tensor([[[3, 1, 2, -1]]], dtype=torch.int32)
    )
    expected = torch.softmax(torch.tensor([3.0, 2.0, 0.0]), dim=0)
    torch.testing.assert_close(seen["index_score"][0, 0, :3], expected)
    torch.testing.assert_close(seen["index_score"][0, 0, 3:], torch.zeros(125))


def test_cudnn_sparse_loss_masks_invalid_query_rows_for_backward(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["attn_score"] = attn_score
            seen["index_score"] = index_score
            seen["topk_indices"] = topk_indices
            seen["loss_coeff"] = loss_coeff
            seen["grad_loss"] = grad_loss
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[3, 1, 2, -1], [2, 0, 1, -1]]], dtype=torch.int32),
            torch.tensor([[3, 3]], dtype=torch.int32),
            torch.tensor(
                [[[3.0, 2.0, 0.0, torch.finfo(torch.float32).min], [4.0, 1.0, 0.0, -1.0]]],
                dtype=torch.float32,
            ),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.25, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
        query_valid_rows=torch.tensor([[True, False]], dtype=torch.bool),
    )

    torch.testing.assert_close(
        seen["topk_indices"][0, 0, :4], torch.tensor([3, 1, 2, 0], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["topk_indices"][0, 1, :4], torch.zeros(4, dtype=torch.int32))
    torch.testing.assert_close(
        seen["topk_indices"][0, :, 4:], torch.zeros((2, 124), dtype=torch.int32)
    )
    torch.testing.assert_close(seen["attn_score"][0, 1], torch.zeros(128))
    torch.testing.assert_close(seen["index_score"][0, 1], torch.zeros(128))
    assert seen["loss_coeff"] == 0.01
    torch.testing.assert_close(seen["grad_loss"], torch.tensor(2.0))


def test_cudnn_sparse_loss_reduces_attention_target_across_tp(monkeypatch):
    seen = {"all_reduce_calls": 0}

    class FakeTPGroup:
        def size(self):
            return 2

    tp_group = FakeTPGroup()

    class FakeDSA:
        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            del index_score, topk_indices, sm_scale, loss_coeff, grad_loss, block_I
            seen["attn_score"] = attn_score.detach().clone()
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    def fake_attn_target(_q, _k, _lse, topk_indices, *args, **kwargs):
        seen["loss_topk"] = topk_indices.detach().clone()
        del args, kwargs
        return torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    def fake_all_reduce(tensor, group=None, async_op=False):
        assert group is tp_group
        assert async_op is False
        seen["all_reduce_calls"] += 1
        tensor.add_(torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=tensor.dtype))

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads(
        q_idx_bshd=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        k_idx_bsd=torch.zeros((1, 4, 1), dtype=torch.bfloat16),
        w_bsh=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        topk_indices_cmp=torch.tensor([[[2, 0, 1, -1]]], dtype=torch.int32),
        topk_length_cmp=torch.tensor([[3]], dtype=torch.int32),
        indexer_score_payload=torch.tensor([[[2.0, 0.0, 1.0, -1.0]]], dtype=torch.float32),
        query=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
        kv_full=torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        lse=torch.zeros((1, 1), dtype=torch.float32),
        softmax_scale=1.0,
        loss_coeff=1.0,
        query_valid_rows=None,
        calculate_per_token_loss=True,
        tp_group=tp_group,
    )

    assert seen["all_reduce_calls"] == 1
    torch.testing.assert_close(
        seen["loss_topk"], torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["attn_score"][0, 0, :4], torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_cudnn_sparse_backward_uses_batch_major_topk_indices_for_batched_kv(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            index_score,
            topk_indices,
            sm_scale,
            loss_coeff,
            grad_loss,
            block_I,
        ):
            seen["topk_indices"] = topk_indices
            seen["loss_coeff"] = loss_coeff
            seen["grad_loss"] = grad_loss
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[3, 1, -1]], [[0, 2, -1]]], dtype=torch.int32),
            torch.tensor([[2], [2]], dtype=torch.int32),
            torch.tensor([[[3.0, 2.0, -1.0]], [[4.0, 1.0, -1.0]]], dtype=torch.float32),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    def fake_attn_target(q_attn, k_attn, lse, topk_indices, *args, **kwargs):
        return torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_attn_target)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 2, 1), dtype=torch.bfloat16),
        torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 2, 1), dtype=torch.bfloat16),
        torch.zeros((1, 2, 1), dtype=torch.bfloat16),
        indexer_topk=3,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
    )

    torch.testing.assert_close(
        seen["topk_indices"][:, 0, :3], torch.tensor([[3, 1, 0], [4, 6, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        seen["topk_indices"][:, 0, 3:], torch.zeros((2, 125), dtype=torch.int32)
    )
    assert seen["loss_coeff"] == 0.01
    torch.testing.assert_close(seen["grad_loss"], torch.tensor(1.0))


@pytest.mark.parametrize("source", ["full_fusion", "split_attention"])
def test_cudnn_attention_backward_sanitizes_ignored_topk_slots(monkeypatch, source):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(
            q, kv, out, dO, lse, attn_sink, topk_indices, softmax_scale, topk_length
        ):
            seen["bwd_q_shape"] = q.shape
            seen["bwd_topk"] = topk_indices.detach().clone()
            seen["bwd_topk_length"] = topk_length.detach().clone()
            return {"dq": torch.ones_like(q), "dkv": torch.zeros_like(kv)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        return (
            torch.tensor([[[-1, -1, -1, -1], [1, 2, -1, -1]]], dtype=torch.int32),
            torch.tensor([[0, 2]], dtype=torch.int32),
            None,
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["fwd_topk"] = topk_idxs.detach().clone()
        seen["fwd_topk_length"] = topk_length.detach().clone()
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)

    if source == "full_fusion":
        query = torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16, requires_grad=True)
        monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
        output, _indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
            query,
            torch.zeros((4, 1, 1), dtype=torch.bfloat16, requires_grad=True),
            torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
            torch.zeros((4, 1, 1), dtype=torch.bfloat16),
            torch.zeros((2, 1, 1), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.0,
            sparse_loss=True,
            calculate_per_token_loss=False,
            d_v=1,
        )
        expected_fwd_topk = torch.tensor([[-1, -1, -1, -1], [1, 2, -1, -1]], dtype=torch.int32)
        expected_bwd_topk = torch.tensor([[1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
    else:
        value_dim = 512
        query = torch.zeros((2, 1, 1, value_dim), dtype=torch.bfloat16, requires_grad=True)
        key = torch.zeros((4, 1, 1, value_dim), dtype=torch.bfloat16, requires_grad=True)
        output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
            query=query,
            key=key,
            topk_indices=torch.tensor([[[-1, -1, -1], [2, -1, 1]]], dtype=torch.int32),
            softmax_scale=1.0,
            v_channels=value_dim,
        )
        assert output is not None
        expected_fwd_topk = torch.tensor([[-1, -1, -1], [1, 2, -1]], dtype=torch.int32)
        expected_bwd_topk = torch.tensor([[1, 2, 0], [0, 0, 0]], dtype=torch.int32)

    output.float().sum().backward()

    torch.testing.assert_close(seen["fwd_topk"], expected_fwd_topk)
    torch.testing.assert_close(seen["fwd_topk_length"], torch.tensor([0, 2], dtype=torch.int32))
    assert seen["bwd_q_shape"] == (query.size(0) * query.size(1), query.size(2), query.size(3))
    # Backward compacts nonempty rows and appends one dummy row to avoid empty cuDNN launches.
    torch.testing.assert_close(seen["bwd_topk"], expected_bwd_topk)
    torch.testing.assert_close(seen["bwd_topk_length"], torch.tensor([2, 1], dtype=torch.int32))
    torch.testing.assert_close(query.grad[0], torch.zeros_like(query.grad[0]))
    torch.testing.assert_close(query.grad[1], torch.ones_like(query.grad[1]))
    if source == "split_attention":
        torch.testing.assert_close(key.grad, torch.zeros_like(key.grad))


def test_cudnn_full_fusion_skips_sparse_bwd_compaction_for_nonempty_local_varlen(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(
            q, kv, out, dO, lse, attn_sink, topk_indices, softmax_scale, topk_length
        ):
            del out, dO, lse, attn_sink, softmax_scale
            seen["bwd_q_shape"] = q.shape
            seen["bwd_topk"] = topk_indices.detach().clone()
            seen["bwd_topk_length"] = topk_length.detach().clone()
            return {"dq": torch.ones_like(q), "dkv": torch.zeros_like(kv)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_indexer_topk(*args, **kwargs):
        assert kwargs["use_local_indexer_varlen"] is True
        return (
            torch.tensor([[[0, -1, -1], [0, 1, -1]]], dtype=torch.int32),
            torch.tensor([[1, 2]], dtype=torch.int32),
            None,
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        del kv, topk_idxs, softmax_scale, attn_sink
        seen["fwd_topk_length"] = topk_length.detach().clone()
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)

    query = torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16, requires_grad=True)
    output, _indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
        query,
        torch.zeros((4, 1, 1), dtype=torch.bfloat16, requires_grad=True),
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((4, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1), dtype=torch.bfloat16),
        indexer_topk=3,
        softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=True,
        calculate_per_token_loss=False,
        d_v=1,
        varlen_starts=torch.tensor([0, 0], dtype=torch.int64),
        varlen_ends=torch.tensor([1, 2], dtype=torch.int64),
        use_local_indexer_varlen=True,
    )

    output.float().sum().backward()

    assert seen["bwd_q_shape"] == (2, 1, 1)
    torch.testing.assert_close(seen["fwd_topk_length"], torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(seen["bwd_topk_length"], torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(
        seen["bwd_topk"], torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(query.grad, torch.ones_like(query.grad))


def test_cudnn_sparse_attention_uses_supplied_topk_length(monkeypatch):
    seen = {}

    def fail_prepare(*_args, **_kwargs):
        raise AssertionError("prepared cuDNN top-k should not be compacted again")

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        seen["topk"] = topk_idxs.detach().clone()
        seen["topk_length"] = topk_length.detach().clone()
        return torch.zeros((2, 1, d_v), dtype=q.dtype), torch.zeros((2, 1), dtype=torch.float32)

    monkeypatch.setattr(dsa_cudnn_kernels, "_prepare_attention_topk_indices", fail_prepare)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)

    value_dim = 512
    topk_indices = torch.tensor([[[2, 1, -1], [-1, -1, -1]]], dtype=torch.int32)
    output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=torch.zeros((2, 1, 1, value_dim), dtype=torch.bfloat16),
        key=torch.zeros((4, 1, 1, value_dim), dtype=torch.bfloat16),
        topk_indices=topk_indices,
        softmax_scale=1.0,
        v_channels=value_dim,
        topk_length=torch.tensor([[2, 0]], dtype=torch.int32),
    )

    assert output is not None
    torch.testing.assert_close(
        seen["topk"], torch.tensor([[2, 1, -1], [-1, -1, -1]], dtype=torch.int32)
    )
    torch.testing.assert_close(seen["topk_length"], torch.tensor([2, 0], dtype=torch.int32))
    torch.testing.assert_close(
        topk_indices, torch.tensor([[[2, 1, -1], [-1, -1, -1]]], dtype=torch.int32)
    )


def test_cudnn_sparse_attention_declines_unsupported_flashmla_value_dim(monkeypatch):
    def fail_flash_mla(*_args, **_kwargs):
        raise AssertionError("unsupported FlashMLA value dim must decline before FlashMLA")

    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fail_flash_mla)

    output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=torch.zeros((2, 1, 1, 8), dtype=torch.bfloat16),
        key=torch.zeros((4, 1, 1, 8), dtype=torch.bfloat16),
        topk_indices=torch.tensor([[[2, 1, -1], [-1, -1, -1]]], dtype=torch.int32),
        softmax_scale=1.0,
        v_channels=4,
    )

    assert output is None


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_cudnn_attention_backward_supports_small_local_head_count():
    _skip_if_fused_dsa_unavailable()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    sq = 64
    batch_size = 1
    num_heads = 8
    attn_dim = 576
    latent_v_channels = 512
    indexer_heads = 64
    indexer_dim = 128

    query = torch.randn(
        (sq, batch_size, num_heads, attn_dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv_full = torch.randn(
        (sq, batch_size, attn_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    q_indexer = torch.randn(
        (sq, batch_size, indexer_heads, indexer_dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_indexer = torch.randn(
        (sq, batch_size, indexer_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weights = torch.randn(
        (sq, batch_size, indexer_heads), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    output, indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk=2048,
        softmax_scale=attn_dim**-0.5,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=latent_v_channels,
    )

    assert torch.isfinite(output).all()
    assert indexer_loss.item() == 0.0
    output.float().mul(torch.randn_like(output).float()).sum().backward()

    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(kv_full.grad).all()
    assert q_indexer.grad is None
    assert k_indexer.grad is None
    assert weights.grad is None


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_cudnn_attention_backward_pads_small_local_head_count(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(
            q, kv, out, grad_out, lse, attn_sink, indices, **_kwargs
        ):
            seen["shapes"] = (
                q.shape,
                out.shape,
                grad_out.shape,
                lse.shape,
                attn_sink.shape,
                indices.shape,
            )
            return {"dq": torch.zeros_like(q), "dkv": torch.zeros_like(kv)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)

    sq, batch_size, num_heads, attn_dim, value_dim, skv = 2, 1, 8, 4, 3, 4
    grad_query, grad_kv = dsa_cudnn_kernels._run_sparse_attention_backward(
        q_flat=torch.zeros((sq * batch_size, num_heads, attn_dim), device="cuda"),
        kv_flat=torch.zeros((skv * batch_size, attn_dim), device="cuda"),
        attn_sink=torch.zeros(num_heads, device="cuda"),
        global_idxs=torch.zeros((sq * batch_size, 2), dtype=torch.int32, device="cuda"),
        out_flat=torch.zeros((sq * batch_size, num_heads, value_dim), device="cuda"),
        lse=torch.zeros((sq * batch_size, num_heads), device="cuda"),
        topk_length=torch.full((sq * batch_size,), 2, dtype=torch.int32, device="cuda"),
        softmax_scale=1.0,
        sq=sq,
        b=batch_size,
        num_heads=num_heads,
        d=attn_dim,
        skv=skv,
        grad_output=torch.zeros((sq, batch_size, num_heads, value_dim), device="cuda"),
        all_rows_nonempty=True,
    )

    assert seen["shapes"] == (
        torch.Size([2, 64, 4]),
        torch.Size([2, 64, 3]),
        torch.Size([2, 64, 3]),
        torch.Size([2, 64]),
        torch.Size([64]),
        torch.Size([2, 2]),
    )
    assert grad_query.shape == (sq, batch_size, num_heads, attn_dim)
    assert grad_kv.shape == (skv, batch_size, attn_dim)


def test_cudnn_indexer_backward_head_padding_slices_to_actual_heads():
    q = torch.randn(1, 2, 32, 4)
    w = torch.randn(1, 2, 32)

    padded_q, padded_w, actual_heads = dsa_cudnn_kernels._pad_indexer_heads_for_backward(q, w)

    assert actual_heads == 32
    assert padded_q.shape == (1, 2, 64, 4)
    assert padded_w.shape == (1, 2, 64)
    torch.testing.assert_close(padded_q[:, :, :32], q)
    torch.testing.assert_close(padded_w[:, :, :32], w)
    torch.testing.assert_close(padded_q[:, :, 32:], torch.zeros(1, 2, 32, 4))
    torch.testing.assert_close(padded_w[:, :, 32:], torch.zeros(1, 2, 32))

    grad_q, grad_w = dsa_cudnn_kernels._slice_indexer_backward_head_grads(
        padded_q, padded_w, actual_heads
    )
    torch.testing.assert_close(grad_q, q)
    torch.testing.assert_close(grad_w, w)


def test_cudnn_flash_mla_sm100_uses_smallest_supported_head_padding(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_current_sm", lambda: (10, 0))

    assert dsa_cudnn_kernels._flash_mla_head_padding(32) == 64
    assert dsa_cudnn_kernels._flash_mla_head_padding(64) == 64
    assert dsa_cudnn_kernels._flash_mla_head_padding(128) == 128
    assert dsa_cudnn_kernels._flash_mla_head_padding(256) == 256
    assert dsa_cudnn_kernels._flash_mla_head_padding(48) is None


def test_cudnn_sparse_backward_topk_padding_aligns_to_block_size():
    attn_score = torch.ones(1, 2, 3)
    index_score = torch.full((1, 2, 3), 2.0)
    topk_indices = torch.tensor([[[0, 1, 2], [2, 1, -1]]], dtype=torch.int32)

    padded_attn, padded_index, padded_topk = dsa_cudnn_kernels._pad_sparse_backward_topk(
        attn_score, index_score, topk_indices, block_size=4
    )

    assert padded_attn.shape == (1, 2, 4)
    assert padded_index.shape == (1, 2, 4)
    assert padded_topk.shape == (1, 2, 4)
    torch.testing.assert_close(padded_attn[..., :3], attn_score)
    torch.testing.assert_close(padded_index[..., :3], index_score)
    torch.testing.assert_close(
        padded_topk[..., :3], torch.tensor([[[0, 1, 2], [2, 1, 0]]], dtype=torch.int32)
    )
    torch.testing.assert_close(padded_attn[..., 3], torch.zeros(1, 2))
    torch.testing.assert_close(padded_index[..., 3], torch.zeros(1, 2))
    torch.testing.assert_close(padded_topk[..., 3], torch.zeros((1, 2), dtype=torch.int32))


def test_cudnn_attn_target_pads_small_local_head_count(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            softmax_scale,
            qhead_per_kv_head=None,
            topk_indices_global=False,
        ):
            del k_attn, lse, softmax_scale
            seen["q_shape"] = q_attn.shape
            seen["topk_shape"] = topk_indices.shape
            seen["qhead_per_kv_head"] = qhead_per_kv_head
            seen["topk_indices_global"] = topk_indices_global
            return {
                "target": torch.zeros(
                    (q_attn.size(0), q_attn.size(1), topk_indices.size(2)), dtype=torch.float32
                )
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    target = dsa_cudnn_kernels._compute_attn_target(
        q_attn_bshd=torch.zeros((1, 2, 4, 8), dtype=torch.bfloat16),
        k_attn_bsd=torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        lse=torch.zeros((1, 2, 4), dtype=torch.float32),
        topk_indices=torch.zeros((1, 2, 4), dtype=torch.int32),
        topk_length=None,
        softmax_scale=1.0,
        qhead_per_kv_head=4,
    )

    assert target.shape == (1, 2, 4)
    assert seen["q_shape"] == (1, 2, 8, 8)
    assert seen["topk_shape"] == (1, 2, 4)
    assert seen["qhead_per_kv_head"] == 8
    assert seen["topk_indices_global"] is False


def test_cudnn_dense_attn_lse_uses_full_causal_kv():
    q = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0]], [[2.0]], [[100.0]]]], dtype=torch.float32)

    lse = dsa_cudnn_kernels._compute_dense_attn_lse(q, k, softmax_scale=1.0, qhead_per_kv_head=2)

    expected = torch.tensor(
        [
            [
                [1.0, 2.0],
                [
                    torch.logsumexp(torch.tensor([3.0, 6.0]), dim=0),
                    torch.logsumexp(torch.tensor([4.0, 8.0]), dim=0),
                ],
            ]
        ]
    )
    torch.testing.assert_close(lse, expected)


def test_cudnn_dense_loss_recomputes_full_kv_lse(monkeypatch):
    seen = {}
    dense_lse = torch.tensor([[[0.25, 0.5], [0.75, 1.0]]], dtype=torch.float32)
    sparse_lse = torch.full((2, 2), 99.0, dtype=torch.float32)

    class FakeDSA:
        @staticmethod
        def dense_attn_score_recompute_wrapper(
            q_attn, k_attn, lse, softmax_scale, qhead_per_kv_head, ratio
        ):
            del q_attn, k_attn, softmax_scale, qhead_per_kv_head, ratio
            seen["dense_lse"] = lse.detach().clone()
            return {
                "out": torch.tensor([[[1.0, 0.0, 0.0], [0.25, 0.75, 0.0]]]),
                "denom": torch.ones((1, 2), dtype=torch.float32),
            }

        @staticmethod
        def dense_indexer_backward_wrapper(
            q_indexer,
            weights,
            k_indexer,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale,
            loss_coeff,
            grad_loss,
            ratio,
            block_I,
        ):
            del attn_score, attn_l1norm, index_score, index_lse
            del sm_scale, loss_coeff, grad_loss, ratio, block_I
            return {
                "d_index_q": torch.zeros_like(q_indexer),
                "d_index_k": torch.zeros_like(k_indexer),
                "d_weights": torch.zeros_like(weights),
            }

    def fake_indexer_topk(*args, **kwargs):
        del args
        seen["return_scores"] = kwargs["return_scores"]
        seen["return_topk_scores"] = kwargs["return_topk_scores"]
        return (
            torch.tensor([[[0], [1]]], dtype=torch.int32),
            torch.tensor([[1, 1]], dtype=torch.int32),
            torch.tensor(
                [[[0.0, float("-inf"), float("-inf")], [0.0, 1.0, float("-inf")]]],
                dtype=torch.float32,
            ),
        )

    def fake_flash_mla(q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        del kv, topk_idxs, softmax_scale, attn_sink, topk_length
        return torch.zeros((q.size(0), q.size(1), d_v), dtype=q.dtype), sparse_lse

    def fake_dense_lse(q_attn, k_attn, softmax_scale, qhead_per_kv_head):
        seen["dense_lse_helper_called"] = True
        assert q_attn.shape == (1, 2, 2, 1)
        assert k_attn.shape == (1, 3, 1, 1)
        assert softmax_scale == 1.0
        assert qhead_per_kv_head == 2
        return dense_lse

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_dense_attn_lse", fake_dense_lse)

    dsa_cudnn_kernels.fused_indexer_sparse_attn(
        torch.zeros((2, 1, 2, 1), dtype=torch.bfloat16),
        torch.zeros((3, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1, 1), dtype=torch.bfloat16),
        torch.zeros((3, 1, 1), dtype=torch.bfloat16),
        torch.zeros((2, 1, 1), dtype=torch.bfloat16),
        indexer_topk=1,
        softmax_scale=1.0,
        loss_coeff=1.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=1,
    )

    assert seen["return_scores"] is True
    assert seen["return_topk_scores"] is False
    assert seen["dense_lse_helper_called"] is True
    torch.testing.assert_close(seen["dense_lse"], dense_lse)
    assert not torch.allclose(seen["dense_lse"], sparse_lse.reshape(1, 2, 2))


def test_cudnn_full_fusion_declines_absorbed_mla_without_up_v_weight(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 4
        calculate_per_token_loss = True

    def fail_fused_indexer_sparse_attn(*_args, **_kwargs):
        raise AssertionError("missing up_v_weight must decline before fused attention runs")

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fail_fused_indexer_sparse_attn
    )

    sq = 2
    batch = 1
    heads = 2
    hidden = 8
    idx_heads = 2
    idx_hidden = 4
    assert (
        dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sq, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=None,
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sq, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.0,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=1,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
            query_valid_rows=None,
            use_relu=True,
        )
        is None
    )


def test_cudnn_full_fusion_declines_unsupported_flashmla_value_dim(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 4
        calculate_per_token_loss = True

    def fail_fused_indexer_sparse_attn(*_args, **_kwargs):
        raise AssertionError("unsupported FlashMLA value dim must decline before fused attention")

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fail_fused_indexer_sparse_attn
    )

    sq = 2
    batch = 1
    heads = 2
    hidden = 8
    idx_heads = 2
    idx_hidden = 4
    assert (
        dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sq, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=torch.zeros(
                (heads, Config.kv_lora_rank, Config.kv_lora_rank), dtype=torch.bfloat16
            ),
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sq, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.0,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=1,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
            query_valid_rows=None,
            use_relu=True,
        )
        is None
    )


def test_cudnn_full_fusion_accepts_varlen_when_indexer_loss_disabled(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 512
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        tp_group=None,
    ):
        seen["sparse_loss"] = sparse_loss
        seen["loss_coeff"] = loss_coeff
        seen["varlen_starts"] = varlen_starts
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.zeros((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = Config.kv_lora_rank
    idx_heads = 2
    idx_hidden = 4
    up_v_weight = torch.zeros(
        (heads, Config.kv_lora_rank, Config.kv_lora_rank), dtype=torch.bfloat16
    )
    output = dsa_cudnn_kernels.run_fused_dsa_attention(
        config=Config(),
        query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
        key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
        value=None,
        up_v_weight=up_v_weight,
        q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
        k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
        indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=True,
        absorbed_mla=True,
        cp_size=2,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=object(),
        varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
        varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
        key_positions=None,
        query_valid_rows=None,
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
    )

    assert output is not None
    assert seen["sparse_loss"] is False
    assert seen["loss_coeff"] == 0.0
    assert seen["use_local_indexer_varlen"] is True
    torch.testing.assert_close(seen["varlen_starts"], torch.tensor([0, 2], dtype=torch.int64))


def test_cudnn_full_fusion_skips_varlen_dense_indexer_loss_under_no_grad(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 512
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        tp_group=None,
    ):
        seen["loss_coeff"] = loss_coeff
        seen["sparse_loss"] = sparse_loss
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.zeros((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = Config.kv_lora_rank
    idx_heads = 2
    idx_hidden = 4
    up_v_weight = torch.zeros(
        (heads, Config.kv_lora_rank, Config.kv_lora_rank), dtype=torch.bfloat16
    )
    with torch.no_grad():
        output = dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=up_v_weight,
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=2,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=object(),
            varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
            varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
            key_positions=None,
            query_valid_rows=None,
            use_relu=True,
            use_local_indexer_varlen=True,
            single_packed_thd_sequence=True,
        )

    assert output is not None
    assert seen["loss_coeff"] == 0.0
    assert seen["sparse_loss"] is False


def test_cudnn_full_fusion_accepts_local_varlen_for_sparse_indexer_loss(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 512
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        tp_group=None,
    ):
        seen["sparse_loss"] = sparse_loss
        seen["loss_coeff"] = loss_coeff
        seen["use_local_indexer_varlen"] = use_local_indexer_varlen
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.ones((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = Config.kv_lora_rank
    idx_heads = 2
    idx_hidden = 4
    up_v_weight = torch.zeros(
        (heads, Config.kv_lora_rank, Config.kv_lora_rank), dtype=torch.bfloat16
    )
    output = dsa_cudnn_kernels.run_fused_dsa_attention(
        config=Config(),
        query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
        key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
        value=None,
        up_v_weight=up_v_weight,
        q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
        k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
        indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
        indexer_topk=4,
        softmax_scale=1.0,
        loss_coeff=0.01,
        sparse_loss=True,
        calculate_per_token_loss=True,
        absorbed_mla=True,
        cp_size=2,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=object(),
        varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
        varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
        key_positions=None,
        query_valid_rows=None,
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
    )

    assert output is not None
    assert seen["sparse_loss"] is True
    assert seen["loss_coeff"] == 0.01
    assert seen["use_local_indexer_varlen"] is True


def test_cudnn_full_fusion_declines_varlen_dense_indexer_loss(monkeypatch):
    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 512
        calculate_per_token_loss = True

    sq = 2
    sk = 4
    batch = 1
    heads = 2
    hidden = Config.kv_lora_rank
    idx_heads = 2
    idx_hidden = 4
    up_v_weight = torch.zeros(
        (heads, Config.kv_lora_rank, Config.kv_lora_rank), dtype=torch.bfloat16
    )

    def fail_fused_indexer_sparse_attn(*_args, **_kwargs):
        raise AssertionError("dense indexer loss with varlen bounds must decline full fusion")

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fail_fused_indexer_sparse_attn
    )

    assert (
        dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=up_v_weight,
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=2,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=object(),
            varlen_starts=torch.tensor([0, 2], dtype=torch.int64),
            varlen_ends=torch.tensor([1, 4], dtype=torch.int64),
            key_positions=None,
            query_valid_rows=None,
            use_relu=True,
            use_local_indexer_varlen=True,
        )
        is None
    )


def test_cudnn_full_fusion_strips_flagged_plain_causal_varlen(monkeypatch):
    """``varlen_is_plain_causal`` drives the plain-causal normalization without a device sync.

    ``build_dsattention_forward_mask`` emits explicit plain-causal bounds and flags them via
    ``varlen_is_plain_causal``. The hook must normalize those bounds back to ``None`` based on the
    flag alone (instead of a ``torch.equal`` host/device sync), so dense indexer loss takes the
    fused no-varlen path. Identical bounds without the flag are left intact and decline fusion.
    """

    class Config:
        dsa_kernel_backend = "cudnn"
        attention_backend = AttnBackend.auto
        kv_lora_rank = 512
        calculate_per_token_loss = True

    seen = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff=0.0,
        sparse_loss=False,
        calculate_per_token_loss=False,
        d_v=0,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        tp_group=None,
    ):
        seen["called"] = True
        seen["varlen_starts"] = varlen_starts
        seen["varlen_ends"] = varlen_ends
        seen["loss_coeff"] = loss_coeff
        return torch.zeros(
            (query.size(0), query.size(1), query.size(2) * d_v),
            dtype=query.dtype,
            device=query.device,
        ), torch.zeros((), dtype=torch.float32, device=query.device)

    monkeypatch.setattr(
        dsa_cudnn_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    sq = sk = 4
    batch, heads = 1, 2
    hidden = Config.kv_lora_rank
    idx_heads, idx_hidden = 2, 4
    up_v_weight = torch.zeros((heads, hidden, hidden), dtype=torch.bfloat16)

    def run(varlen_is_plain_causal):
        seen.clear()
        return dsa_cudnn_kernels.run_fused_dsa_attention(
            config=Config(),
            query=torch.zeros((sq, batch, heads, hidden), dtype=torch.bfloat16),
            key=torch.zeros((sk, batch, 1, hidden), dtype=torch.bfloat16),
            value=None,
            up_v_weight=up_v_weight,
            q_indexer=torch.zeros((sq, batch, idx_heads, idx_hidden), dtype=torch.bfloat16),
            k_indexer=torch.zeros((sk, batch, idx_hidden), dtype=torch.bfloat16),
            indexer_weights=torch.zeros((sq, batch, idx_heads), dtype=torch.bfloat16),
            indexer_topk=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            calculate_per_token_loss=True,
            absorbed_mla=True,
            cp_size=1,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            varlen_starts=torch.zeros(sq, dtype=torch.int64),
            varlen_ends=torch.arange(1, sq + 1, dtype=torch.int64),
            key_positions=None,
            query_valid_rows=None,
            varlen_is_plain_causal=varlen_is_plain_causal,
            use_relu=True,
        )

    # Flagged: bounds are normalized to None, so dense indexer loss runs on the fused path.
    flagged_output = run(varlen_is_plain_causal=True)
    assert flagged_output is not None
    assert seen["called"] is True
    assert seen["varlen_starts"] is None
    assert seen["varlen_ends"] is None
    assert seen["loss_coeff"] == 0.01

    # Unflagged: the identical bounds are left intact, so dense indexer loss declines fusion.
    assert run(varlen_is_plain_causal=False) is None
    assert "called" not in seen


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_cudnn_full_fusion_real_kernel_packed_cp_varlen_matches_reference():
    _skip_if_fused_dsa_unavailable()
    case = _make_packed_cp_varlen_cudnn_case(calculate_per_token_loss=False)

    fused = dsa_cudnn_kernels.run_fused_dsa_attention(
        config=case["config"],
        query=case["query"],
        key=case["key"],
        value=None,
        up_v_weight=case["up_v_weight"],
        q_indexer=case["q_indexer"],
        k_indexer=case["k_indexer"],
        indexer_weights=case["weights"],
        indexer_topk=case["index_topk"],
        softmax_scale=case["softmax_scale"],
        loss_coeff=case["loss_coeff"],
        sparse_loss=True,
        calculate_per_token_loss=case["config"].calculate_per_token_loss,
        absorbed_mla=True,
        cp_size=case["cp_size"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=object(),
        varlen_starts=case["varlen_starts"],
        varlen_ends=case["varlen_ends"],
        key_positions=None,
        query_valid_rows=case["query_valid_rows"],
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=case["cp_rank"],
        pg_collection=case["pg_collection"],
    )

    assert fused is not None
    fused_output, fused_loss = fused

    _ref_scores, ref_topk = dsa_module.fused_qk_topk_naive(
        case["q_indexer"],
        case["k_indexer"],
        case["weights"],
        case["index_topk"],
        mask=None,
        varlen_starts=case["varlen_starts"],
        varlen_ends=case["varlen_ends"],
        key_positions=None,
        use_relu=True,
    )
    ref_output, ref_loss = _reference_absorbed_output_and_sparse_loss(case, ref_topk)

    _assert_similarity(fused_output, ref_output, eps=5e-3)
    torch.testing.assert_close(fused_loss, ref_loss, rtol=5e-2, atol=5e-3)


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_cudnn_split_real_kernel_packed_cp_varlen_matches_reference():
    _skip_if_fused_dsa_unavailable()
    case = _make_packed_cp_varlen_cudnn_case(calculate_per_token_loss=True)

    topk_indices, topk_length, fused_loss = dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
        config=case["config"],
        q=case["q_indexer"],
        k=case["k_indexer"],
        weights=case["weights"],
        index_topk=case["index_topk"],
        starts=case["varlen_starts"].to(dtype=torch.int32),
        ends=case["varlen_ends"].to(dtype=torch.int32),
        block_size=8192,
        query=case["query"],
        key=case["key"],
        softmax_scale=case["softmax_scale"],
        loss_coeff=case["loss_coeff"],
        pg_collection=case["pg_collection"],
        query_valid_rows=case["query_valid_rows"],
        calculate_per_token_loss=case["config"].calculate_per_token_loss,
        use_relu=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_rank=case["cp_rank"],
    )
    fused_latent = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=case["query"],
        key=case["key"],
        topk_indices=topk_indices,
        softmax_scale=case["softmax_scale"],
        v_channels=case["latent_v_channels"],
        topk_length=topk_length,
    )
    assert fused_latent is not None
    fused_output = torch.einsum("sbhc,hdc->sbhd", fused_latent, case["up_v_weight"]).contiguous()
    fused_output = fused_output.view(fused_output.size(0), fused_output.size(1), -1)

    ref_output, ref_loss = _reference_absorbed_output_and_sparse_loss(case, topk_indices)
    _assert_similarity(fused_output, ref_output, eps=5e-3)
    torch.testing.assert_close(fused_loss, ref_loss, rtol=5e-2, atol=5e-3)
