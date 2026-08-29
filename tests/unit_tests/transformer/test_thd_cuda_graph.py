# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for THD format with CUDA Graph support.

Padding helpers and dataclass round-trip (any GPU count, fast):
    torchrun --nproc_per_node 1 -m pytest -xvs \
        tests/unit_tests/transformer/test_thd_cuda_graph.py \
        -k "Pad or Decompose"

End-to-end no-graph vs graph bitwise loss/grad_norm match for
Moonlight-16B and Qwen3-8B with TP2_CP2_PP2 + sequence packing
(requires 8 GPUs, slow ~5 min per run, 4 runs total). Moonlight covers
MoE router/preprocess graph capture with router fusion; Qwen3 is dense and
covers attention graph capture:
    pytest -xvs tests/unit_tests/transformer/test_thd_cuda_graph.py::TestE2EBitwise

The E2E test directly subprocesses `torchrun pretrain_gpt.py` -- the same
command exercised by test_moonlight_qwen3_bitwise.sh -- with both
cuda_graph_impl=none and cuda_graph_impl=transformer_engine, then compares
the per-iteration loss / grad_norm lines. They must be exactly equal.
"""

import os
import re
import socket
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule import _build_thd_padding_mask
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    _resolve_thd_padding_lengths,
    _same_tensor_view,
    extend_thd_padding_before_cp_slice,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
    resolve_thd_tail_padding_policy,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

os.environ.setdefault('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


_REQUIRES_TWO_RANKS = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2 or torch.cuda.device_count() < 2,
    reason="requires torchrun with at least 2 GPUs",
)


# =============================================================================
# Helpers (shared by the lightweight unit tests)
# =============================================================================


def _make_cu(seqlens, device="cuda"):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    for i, s in enumerate(seqlens):
        cu[i + 1] = cu[i] + s
    return cu


def _make_psp(seqlens):
    cu = _make_cu(seqlens)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
    )


def _make_same_view_cu_pair(boundaries):
    storage = torch.tensor([-1, *boundaries, -1], dtype=torch.int32)
    q = storage[1:-1]
    kv = storage[1:-1]
    assert q is not kv
    assert q.data_ptr() == kv.data_ptr()
    return q, kv


def _make_same_view_psp(valid_boundaries, padded_boundaries):
    cu_q, cu_kv = _make_same_view_cu_pair(valid_boundaries)
    cu_q_padded, cu_kv_padded = _make_same_view_cu_pair(padded_boundaries)
    max_seqlen = max(
        right - left
        for boundaries in (valid_boundaries, padded_boundaries)
        for left, right in zip(boundaries, boundaries[1:])
    )
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        cu_seqlens_q_padded=cu_q_padded,
        cu_seqlens_kv_padded=cu_kv_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        pad_between_seqs=valid_boundaries != padded_boundaries,
    )


def _assert_qkv_views_are_shared(packed_seq_params):
    for q, kv in (
        (packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_kv),
        (packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_kv_padded),
    ):
        assert q is kv
        assert q.data_ptr() == kv.data_ptr()
        assert q.storage_offset() == kv.storage_offset()
        assert q.shape == kv.shape
        assert q.stride() == kv.stride()


def _build_layer(H, nh, nkv, ffn, max_seqlen, max_num_seqs, tp=1, sp=False):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    config = TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=nh,
        num_query_groups=nkv,
        ffn_hidden_size=ffn,
        max_seqlen_per_dp_cp_rank=max_seqlen,
        thd_max_packed_sequences=max_num_seqs,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        bf16=True,
    )
    model_parallel_cuda_manual_seed(42)
    return (
        TransformerLayer(
            config, get_gpt_layer_with_transformer_engine_spec().submodules, layer_number=1
        )
        .cuda()
        .bfloat16()
    )


def _build_chunk(H, nh, nkv, ffn, max_seqlen, max_num_seqs):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=nh,
        num_query_groups=nkv,
        ffn_hidden_size=ffn,
        max_seqlen_per_dp_cp_rank=max_seqlen,
        thd_max_packed_sequences=max_num_seqs,
        bf16=True,
        cuda_graph_impl="transformer_engine",
        cuda_graph_granularity="chunk",
        cuda_graph_modules=[],
        cuda_graph_dynamic_microbatches=True,
        sequence_packing_scheduler="dp_balanced",
        pad_packed_seq_alignment="max",
        use_cpu_initialization=True,
    )
    model_parallel_cuda_manual_seed(42)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=128,
        max_sequence_length=max_seqlen,
        position_embedding_type="rope",
    ).cuda()
    return model.decoder


# =============================================================================
# 1. pad_sequence_for_thd correctness
# =============================================================================


@pytest.mark.internal
@pytest.mark.parametrize("cuda_graph_static,expected_max_num_seqs", [(False, 32), (True, 32)])
def test_pad_to_max_resolves_padding_kwargs(cuda_graph_static, expected_max_num_seqs):
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=8192,
        thd_max_packed_sequences=32,
        cuda_graph_static=cuda_graph_static,
    )

    assert alignment is None
    assert target_len == 8192
    assert max_num_seqs == expected_max_num_seqs


@pytest.mark.internal
def test_resolve_thd_tail_padding_policy():
    from types import SimpleNamespace

    # Default (unset): append an ordinary dummy sequence.
    assert resolve_thd_tail_padding_policy(SimpleNamespace()) == "append_dummy_seq"
    assert (
        resolve_thd_tail_padding_policy(SimpleNamespace(thd_tail_padding_policy=None))
        == "append_dummy_seq"
    )
    # Explicit policies are returned as-is.
    assert (
        resolve_thd_tail_padding_policy(SimpleNamespace(thd_tail_padding_policy="extend_last"))
        == "extend_last"
    )
    with pytest.raises(AssertionError, match="Unsupported thd_tail_padding_policy"):
        resolve_thd_tail_padding_policy(SimpleNamespace(thd_tail_padding_policy="bogus"))


class TestResolveThdPaddingLengths:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "source,target_len,alignment,expected",
        [
            ("tokens", None, 64, (80, 80, 128, 128)),
            ("labels", 256, None, (80, 80, 256, 256)),
            ("metadata", None, 64, (80, 80, 128, 128)),
        ],
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_non_cp_length_resolution_contract(self, source, target_len, alignment, expected):
        """Resolve lengths from local tensors when present, otherwise from THD metadata."""
        tokens, labels = None, None
        psp = _make_psp([50, 30])

        if source == "tokens":
            tokens = torch.ones(1, 80, device="cuda")
            psp = PackedSeqParams(qkv_format="thd")
            expected_device = tokens.device
        elif source == "labels":
            labels = torch.ones(1, 80, device="cuda")
            expected_device = labels.device
        else:
            expected_device = psp.cu_seqlens_q.device

        local_actual, global_actual, local_target, global_target, mask_device = (
            _resolve_thd_padding_lengths(
                tokens, labels, None, None, psp, target_len=target_len, alignment=alignment
            )
        )

        assert (local_actual, global_actual, local_target, global_target) == expected
        assert mask_device == expected_device

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_tensor_requires_cu_seqlens(self):
        """All-None tensor inputs need cu_seqlens to build a padding mask."""
        psp = PackedSeqParams(qkv_format="thd")

        with pytest.raises(AssertionError, match="cu_seqlens_q must be available"):
            _resolve_thd_padding_lengths(
                None, None, None, None, psp, target_len=128, alignment=None
            )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_metadata_only_uses_physical_padded_endpoint(self):
        """Physical storage, not compact valid-token count, determines padding length."""
        cu_valid = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu_valid, cu_seqlens_q_padded=cu_padded
        )

        resolved = _resolve_thd_padding_lengths(
            None, None, None, None, psp, target_len=None, alignment=6
        )

        assert resolved[:4] == (8, 8, 12, 12)
        assert resolved[4] == cu_padded.device

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_tensor_alignment_uses_local_target_and_global_tail(self):
        """CP-local padding tail determines the global padded endpoint."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        tokens = torch.ones(1, 1600, device="cuda")
        psp = _make_psp([1600, 1600])

        local_actual, global_actual, local_target, global_target, mask_device = (
            _resolve_thd_padding_lengths(
                tokens, None, None, None, psp, target_len=None, alignment=128
            )
        )

        assert (local_actual, global_actual, local_target, global_target) == (
            1600,
            3200,
            1664,
            3328,
        )
        assert mask_device == tokens.device

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_tensor_target_len_scales_global_target(self):
        """Fixed target_len is CP-local and scales to a global endpoint."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        tokens = torch.ones(1, 80, device="cuda")
        psp = _make_psp([140])

        local_actual, global_actual, local_target, global_target, mask_device = (
            _resolve_thd_padding_lengths(
                tokens, None, None, None, psp, target_len=128, alignment=None
            )
        )

        assert (local_actual, global_actual, local_target, global_target) == (80, 140, 128, 256)
        assert mask_device == tokens.device

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "alignment,target_len,expected_global_target", [(128, None, 256), (None, 128, 256)]
    )
    @_REQUIRES_TWO_RANKS
    def test_cp_no_tensor_partitions_actual_and_target_lengths(
        self, alignment, target_len, expected_global_target
    ):
        """Without local tensors, CP-local lengths come from THD partition indices."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core import parallel_state
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        psp = _make_psp([140])
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        expected_local_actual = get_thd_partitioned_indices(
            psp.cu_seqlens_q, 140, cp_size, cp_rank
        ).numel()
        expected_local_target = target_len or (
            (expected_local_actual + alignment - 1) // alignment * alignment
        )

        local_actual, global_actual, local_target, global_target, mask_device = (
            _resolve_thd_padding_lengths(
                None, None, None, None, psp, target_len=target_len, alignment=alignment
            )
        )

        assert (local_actual, global_actual, local_target, global_target) == (
            expected_local_actual,
            140,
            expected_local_target,
            expected_global_target,
        )
        assert mask_device == psp.cu_seqlens_q.device

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_no_tensor_alignment_matches_local_tensor_path(self):
        """Intermediate PP stages apply alignment to the CP-local length."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        from megatron.core import parallel_state
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        global_actual = 2048
        alignment = 2048
        psp = _make_psp([global_actual])
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        local_actual = get_thd_partitioned_indices(
            psp.cu_seqlens_q, global_actual, cp_size, cp_rank
        ).numel()
        metadata_lengths = _resolve_thd_padding_lengths(
            None, None, None, None, psp, target_len=None, alignment=alignment
        )[:4]
        tensor_lengths = _resolve_thd_padding_lengths(
            torch.ones(1, local_actual, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=None,
            alignment=alignment,
        )[:4]

        assert metadata_lengths == tensor_lengths
        assert metadata_lengths == (local_actual, global_actual, alignment, alignment * cp_size)


class TestPadSequenceForThdAliasPreservation:

    @pytest.mark.internal
    def test_same_tensor_view_edge_cases_fail_closed(self):
        base = torch.arange(6)
        same_view = base[:4]
        same_view_alias = base[:4]
        assert _same_tensor_view(same_view, same_view)
        assert _same_tensor_view(same_view, same_view_alias)
        assert not _same_tensor_view(None, None)
        assert not _same_tensor_view(same_view, None)

        empty = torch.empty(0)
        assert not _same_tensor_view(empty, empty)
        assert not _same_tensor_view(empty, empty.view_as(empty))

        meta = torch.empty(4, device="meta")
        assert not _same_tensor_view(meta, meta)
        assert not _same_tensor_view(meta, meta.view_as(meta))

        assert not _same_tensor_view(same_view, base[1:5])
        assert not _same_tensor_view(same_view, base[:3])

    @pytest.mark.internal
    def test_same_tensor_view_fake_tensor_fails_closed(self):
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode
        except ImportError:
            pytest.skip("FakeTensorMode is not available in this PyTorch version.")

        with FakeTensorMode():
            fake = torch.empty(4, device="cuda")
            fake_alias = fake.view_as(fake)

        assert not _same_tensor_view(fake, fake)
        assert not _same_tensor_view(fake, fake_alias)

    @pytest.mark.internal
    def test_append_dummy_sequence_preserves_qkv_views(self):
        psp = _make_same_view_psp([0, 3, 5], [0, 3, 5])

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 5), None, None, None, psp, target_len=8, cp_size=1, cp_rank=0
        )

        expected = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        assert torch.equal(padded.cu_seqlens_q, expected)
        assert torch.equal(padded.cu_seqlens_q_padded, expected)
        _assert_qkv_views_are_shared(padded)
        assert padded.cu_seqlens_q.data_ptr() != padded.cu_seqlens_q_padded.data_ptr()

    @pytest.mark.internal
    def test_extend_last_sequence_preserves_qkv_views(self):
        psp = _make_same_view_psp([0, 3, 5], [0, 4, 7])

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 7),
            None,
            None,
            None,
            psp,
            target_len=10,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )

        assert torch.equal(padded.cu_seqlens_q, torch.tensor([0, 3, 5], dtype=torch.int32))
        assert torch.equal(padded.cu_seqlens_q_padded, torch.tensor([0, 4, 10], dtype=torch.int32))
        _assert_qkv_views_are_shared(padded)
        assert padded.cu_seqlens_q.data_ptr() != padded.cu_seqlens_q_padded.data_ptr()

    @pytest.mark.internal
    def test_extend_last_missing_padded_metadata_inherits_valid_qkv_view(self):
        cu_q, cu_kv = _make_same_view_cu_pair([0, 3, 5])
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            pad_between_seqs=False,
        )

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 5),
            None,
            None,
            None,
            psp,
            target_len=8,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )

        assert torch.equal(padded.cu_seqlens_q, torch.tensor([0, 3, 5], dtype=torch.int32))
        assert torch.equal(padded.cu_seqlens_q_padded, torch.tensor([0, 3, 8], dtype=torch.int32))
        _assert_qkv_views_are_shared(padded)
        assert padded.cu_seqlens_q.data_ptr() != padded.cu_seqlens_q_padded.data_ptr()

    @pytest.mark.internal
    def test_extend_last_missing_padded_metadata_keeps_distinct_valid_qkv_views(self):
        cu_q = torch.tensor([0, 3, 5], dtype=torch.int32)
        cu_kv = cu_q.clone()
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            pad_between_seqs=False,
        )

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 5),
            None,
            None,
            None,
            psp,
            target_len=8,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )

        expected = torch.tensor([0, 3, 8], dtype=torch.int32)
        assert torch.equal(padded.cu_seqlens_q_padded, expected)
        assert torch.equal(padded.cu_seqlens_kv_padded, expected)
        assert padded.cu_seqlens_q_padded is not padded.cu_seqlens_kv_padded
        assert padded.cu_seqlens_q_padded.data_ptr() != padded.cu_seqlens_kv_padded.data_ptr()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_extend_last_missing_padded_metadata_keeps_cp1_thd_scorer_eligible(self):
        from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

        storage = torch.tensor([-1, 0, 3, 5, -1], dtype=torch.int32, device="cuda")
        cu_q = storage[1:-1]
        cu_kv = storage[1:-1]
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            pad_between_seqs=False,
        )

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 5, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=8,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )

        assert dsa_cudnn_kernels._packed_thd_kernel_applicable(
            b=1,
            sq=8,
            sk=8,
            device=padded.cu_seqlens_q_padded.device,
            packed_cu_seqlens_q=padded.cu_seqlens_q_padded,
            packed_cu_seqlens_k=padded.cu_seqlens_kv_padded,
            packed_max_seqlen_q=padded.max_seqlen_q,
            packed_max_seqlen_k=padded.max_seqlen_kv,
            cp_size=1,
            cp_rank=0,
            local_packed_cp_query_start=0,
            local_packed_cp_query_len=None,
        )

    @pytest.mark.internal
    def test_fixed_capacity_only_padding_preserves_qkv_views(self):
        psp = _make_same_view_psp([0, 3, 8], [0, 4, 8])

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 8),
            None,
            None,
            None,
            psp,
            target_len=8,
            max_num_seqs=5,
            cp_size=1,
            cp_rank=0,
        )

        assert torch.equal(padded.cu_seqlens_q, torch.tensor([0, 3, 8, 8, 8, 8], dtype=torch.int32))
        assert torch.equal(
            padded.cu_seqlens_q_padded, torch.tensor([0, 4, 8, 8, 8, 8], dtype=torch.int32)
        )
        _assert_qkv_views_are_shared(padded)
        assert padded.cu_seqlens_q.data_ptr() != padded.cu_seqlens_q_padded.data_ptr()

    @pytest.mark.internal
    def test_distinct_equal_qkv_views_remain_distinct(self):
        cu_q = torch.tensor([0, 3, 5], dtype=torch.int32)
        cu_kv = cu_q.clone()
        cu_q_padded = cu_q.clone()
        cu_kv_padded = cu_q.clone()
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_q_padded=cu_q_padded,
            cu_seqlens_kv_padded=cu_kv_padded,
            max_seqlen_q=3,
            max_seqlen_kv=3,
        )

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 5),
            None,
            None,
            None,
            psp,
            target_len=8,
            max_num_seqs=5,
            cp_size=1,
            cp_rank=0,
        )

        expected = torch.tensor([0, 3, 5, 8, 8, 8], dtype=torch.int32)
        assert torch.equal(padded.cu_seqlens_q, expected)
        assert torch.equal(padded.cu_seqlens_kv, expected)
        assert torch.equal(padded.cu_seqlens_q_padded, expected)
        assert torch.equal(padded.cu_seqlens_kv_padded, expected)
        assert padded.cu_seqlens_q is not padded.cu_seqlens_kv
        assert padded.cu_seqlens_q.data_ptr() != padded.cu_seqlens_kv.data_ptr()
        assert padded.cu_seqlens_q_padded is not padded.cu_seqlens_kv_padded
        assert padded.cu_seqlens_q_padded.data_ptr() != padded.cu_seqlens_kv_padded.data_ptr()


class TestPadSequenceForThd:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generic_alignment_appends_dummy_padding_sequence(self):
        """Default THD padding covers tail slots with an ordinary dummy sequence."""
        seqlens, total_T = [50, 30], 80
        psp = _make_psp(seqlens)
        # Use the non-default mode so losing it during rebuild cannot silently pass as zigzag.
        psp.cp_partition_mode = "contiguous"
        orig = psp.cu_seqlens_q.clone()
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"), None, None, None, psp, alignment=64
        )
        assert p_tok.shape == (1, 128)
        expected = torch.cat((orig, torch.tensor([128], dtype=orig.dtype, device=orig.device)))
        assert torch.equal(p.cu_seqlens_q, expected)
        assert torch.equal(p.cu_seqlens_q_padded, expected)
        assert p.pad_between_seqs is False
        assert p.cp_partition_mode == "contiguous"
        assert mask.shape == (1, 128)
        assert not mask[0, :total_T].any() and mask[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_default_dummy_preserves_existing_valid_and_physical_gaps(self):
        """The legacy ordinary dummy adds only the tail length to compact metadata."""
        cu_valid = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid.clone(),
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded.clone(),
            max_seqlen_q=4,
            max_seqlen_kv=4,
        )
        initial_padding_mask = torch.tensor(
            [[False, False, False, True, False, False, True, True]], dtype=torch.bool, device="cuda"
        )

        p_tok, _, _, _, padded, mask = pad_sequence_for_thd(
            torch.ones(1, 8, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=12,
            padding_mask=initial_padding_mask,
        )

        expected_valid = torch.tensor([0, 3, 5, 9], dtype=torch.int32, device="cuda")
        expected_padded = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
        assert p_tok.shape == (1, 12)
        assert torch.equal(padded.cu_seqlens_q, expected_valid)
        assert torch.equal(padded.cu_seqlens_kv, expected_valid)
        assert torch.equal(padded.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(padded.cu_seqlens_kv_padded, expected_padded)
        assert padded.pad_between_seqs is True
        assert padded.max_seqlen_q == 4
        assert padded.max_seqlen_kv == 4
        assert torch.equal(
            mask,
            torch.tensor(
                [[False, False, False, True, False, False, True, True, True, True, True, True]],
                dtype=torch.bool,
                device="cuda",
            ),
        )

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_alignment_uses_global_cu_seqlens_length(self):
        """CP-local token length must not cap global packed-sequence padding."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        psp = _make_psp([140])
        local_T = 80
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            torch.ones(1, local_T, device="cuda"), None, None, None, psp, alignment=128
        )

        assert p_tok.shape[-1] >= local_T
        assert p.cu_seqlens_q[-1].item() == 256
        assert p.cu_seqlens_q_padded[-1].item() == 256
        assert p.max_seqlen_q == 140
        assert p.max_seqlen_kv == 140
        assert mask.shape[-1] == p_tok.shape[-1]
        assert not mask[0, :local_T].any()

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_alignment_covers_local_padding_tail(self):
        """CP-local padding can create a global tail even when global length is aligned."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        psp = _make_psp([1600, 1600])
        local_T = 1600
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            torch.ones(1, local_T, device="cuda"), None, None, None, psp, alignment=128
        )

        assert p_tok.shape[-1] == 1664
        assert p.cu_seqlens_q[-1].item() == 3328
        assert p.cu_seqlens_q_padded[-1].item() == 3328
        assert p.max_seqlen_q == 1600
        assert p.max_seqlen_kv == 1600
        assert mask.shape[-1] == p_tok.shape[-1]
        assert not mask[0, :local_T].any()
        assert mask[0, local_T:].all()

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_dummy_rejects_tail_that_cannot_be_partitioned(self):
        """An ordinary dummy sequence must satisfy TE's 2*CP divisibility rule."""
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        with pytest.raises(AssertionError, match="must be divisible"):
            pad_sequence_for_thd(
                torch.ones(1, 2, device="cuda"), None, None, None, _make_psp([4]), target_len=3
            )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cp_non_dummy_padding_rejects_slice_first_call(self):
        """Non-dummy metadata must be extended before CP partitions the tensors."""
        with pytest.raises(AssertionError, match="before CP slicing"):
            pad_sequence_for_thd(
                torch.ones(1, 2, device="cuda"),
                None,
                None,
                None,
                _make_psp([4]),
                target_len=3,
                tail_padding_policy="extend_last",
                cp_size=2,
                cp_rank=0,
            )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_without_dummy_extends_last_padded_sequence(self):
        """Non-dummy tail padding belongs to the final real sequence."""
        cu_valid = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid.clone(),
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded.clone(),
            max_seqlen_q=4,
            max_seqlen_kv=4,
            pad_between_seqs=True,
        )
        original = {
            "cu_seqlens_q": psp.cu_seqlens_q.clone(),
            "cu_seqlens_kv": psp.cu_seqlens_kv.clone(),
            "cu_seqlens_q_padded": psp.cu_seqlens_q_padded.clone(),
            "cu_seqlens_kv_padded": psp.cu_seqlens_kv_padded.clone(),
        }
        initial_padding_mask = torch.tensor(
            [[False, False, False, True, False, False, True, True]], dtype=torch.bool, device="cuda"
        )
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            torch.ones(1, 8, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=10,
            tail_padding_policy="extend_last",
            padding_mask=initial_padding_mask,
        )
        expected_padded = torch.tensor([0, 4, 10], dtype=torch.int32, device="cuda")
        assert p_tok.shape == (1, 10)
        assert torch.equal(p.cu_seqlens_q, cu_valid)
        assert torch.equal(p.cu_seqlens_kv, cu_valid)
        assert torch.equal(p.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(p.cu_seqlens_kv_padded, expected_padded)
        assert p.max_seqlen_q == 6
        assert p.max_seqlen_kv == 6
        assert p.pad_between_seqs is True
        assert torch.equal(
            p.seq_idx,
            torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.int32, device="cuda"),
        )
        for name, value in original.items():
            assert torch.equal(getattr(psp, name), value)
        assert torch.equal(
            mask,
            torch.tensor(
                [[False, False, False, True, False, False, True, True, True, True]],
                dtype=torch.bool,
                device="cuda",
            ),
        )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cp_non_dummy_padding_extends_globally_before_slice(self):
        """The supported slice-then-pad recipe for the non-scheduler SFT path.

        With CP > 1 the caller extends the global physical endpoint and pads
        tensors before CP slicing; the post-slice pad_sequence_for_thd call is
        then a metadata no-op instead of tripping the slice-first assertion.
        """
        cp_size = 2
        cu_valid = torch.tensor([0, 4, 12], dtype=torch.int32, device="cuda")
        cu_padded = cu_valid.clone()
        max_seqlen = torch.tensor([8], dtype=torch.int32, device="cuda")

        cu_padded, max_seqlen, global_target_len = extend_thd_padding_before_cp_slice(
            cu_padded,
            max_seqlen,
            alignment=4,
            target_len=None,
            cp_size=cp_size,
            cp_partition_mode="zigzag",
        )
        assert global_target_len == 16
        assert torch.equal(cu_padded, torch.tensor([0, 4, 16], dtype=torch.int32, device="cuda"))
        assert int(max_seqlen.item()) == 12

        padding_mask = _build_thd_padding_mask(cu_valid, cu_padded)
        assert padding_mask.tolist() == [False] * 12 + [True] * 4

        tokens = torch.arange(1, global_target_len + 1, device="cuda").unsqueeze(0)
        tokens[0, 12:] = 0
        expected_local_rows = {
            0: [0, 3, 4, 5, 6, 13, 14, 15],  # zigzag chunks 0 and 3 of each sequence
            1: [1, 2, 7, 8, 9, 10, 11, 12],  # zigzag chunks 1 and 2 of each sequence
        }
        for cp_rank, rows in expected_local_rows.items():
            index = torch.tensor(rows, dtype=torch.int64, device="cuda")
            local_tokens = tokens.index_select(1, index)
            local_mask = padding_mask.unsqueeze(0).index_select(1, index)
            psp = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_valid,
                cu_seqlens_kv=cu_valid.clone(),
                cu_seqlens_q_padded=cu_padded,
                cu_seqlens_kv_padded=cu_padded.clone(),
                max_seqlen_q=int(max_seqlen.item()),
                max_seqlen_kv=int(max_seqlen.item()),
            )
            p_tok, _, _, _, p, mask = pad_sequence_for_thd(
                local_tokens,
                None,
                None,
                None,
                psp,
                alignment=4,
                tail_padding_policy="extend_last",
                padding_mask=local_mask,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            assert torch.equal(p_tok, local_tokens)
            assert torch.equal(p.cu_seqlens_q, cu_valid)
            assert torch.equal(p.cu_seqlens_q_padded, cu_padded)
            assert p.max_seqlen_q == 12
            assert torch.equal(mask, local_mask)

    @pytest.mark.internal
    @_REQUIRES_TWO_RANKS
    def test_cp_non_dummy_pretrain_gpt_flow_does_not_assert(self):
        """Regression for the non-scheduler SFT path with CP > 1.

        pretrain_gpt passes no explicit CP geometry, so pad_sequence_for_thd
        resolves it from parallel_state. After the global pre-slice extension
        the call must succeed on every rank instead of raising the slice-first
        assertion.
        """
        from megatron.core import parallel_state

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        cu_valid = torch.tensor([0, 4, 12], dtype=torch.int32, device="cuda")
        cu_padded, max_seqlen, global_target_len = extend_thd_padding_before_cp_slice(
            cu_valid.clone(),
            torch.tensor([8], dtype=torch.int32, device="cuda"),
            alignment=4,
            target_len=None,
            cp_size=cp_size,
            cp_partition_mode="zigzag",
        )
        padding_mask = _build_thd_padding_mask(cu_valid, cu_padded).unsqueeze(0)

        local_rows = [0, 3, 4, 5, 6, 13, 14, 15] if cp_rank == 0 else [1, 2, 7, 8, 9, 10, 11, 12]
        index = torch.tensor(local_rows, dtype=torch.int64, device="cuda")
        local_tokens = torch.ones(1, global_target_len, device="cuda").index_select(1, index)
        local_mask = padding_mask.index_select(1, index)
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid.clone(),
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded.clone(),
            max_seqlen_q=int(max_seqlen.item()),
            max_seqlen_kv=int(max_seqlen.item()),
        )
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            local_tokens,
            None,
            None,
            None,
            psp,
            alignment=4,
            tail_padding_policy="extend_last",
            padding_mask=local_mask,
        )
        assert torch.equal(p_tok, local_tokens)
        assert torch.equal(p.cu_seqlens_q, cu_valid)
        assert torch.equal(p.cu_seqlens_q_padded, cu_padded)
        assert p.max_seqlen_q == 12
        assert torch.equal(mask, local_mask)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_without_dummy_creates_padded_metadata(self):
        """Equal input boundaries diverge when the last sequence absorbs the tail."""
        psp = _make_psp([5, 3])

        _, _, _, _, padded, mask = pad_sequence_for_thd(
            torch.ones(1, 8, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=12,
            tail_padding_policy="extend_last",
        )

        expected_valid = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
        expected_padded = torch.tensor([0, 5, 12], dtype=torch.int32, device="cuda")
        assert torch.equal(padded.cu_seqlens_q, expected_valid)
        assert torch.equal(padded.cu_seqlens_kv, expected_valid)
        assert torch.equal(padded.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(padded.cu_seqlens_kv_padded, expected_padded)
        assert padded.max_seqlen_q == 7
        assert padded.max_seqlen_kv == 7
        assert padded.pad_between_seqs is True
        assert not mask[0, :8].any()
        assert mask[0, 8:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shapes_and_data_preservation(self):
        """Shapes are static; original data intact; padding zero-filled."""
        seqlens, max_seqlen, max_num_seqs = [100, 50, 30], 256, 8
        total_T = sum(seqlens)
        tokens = torch.arange(total_T, device="cuda").unsqueeze(0).float()
        p_tok, p_lab, p_loss, p_pos, p_params, p_mask = pad_sequence_for_thd(
            tokens,
            tokens.clone(),
            torch.ones(1, total_T, device="cuda"),
            torch.arange(total_T, device="cuda").unsqueeze(0),
            _make_psp(seqlens),
            target_len=max_seqlen,
            max_num_seqs=max_num_seqs,
        )
        for t in (p_tok, p_lab, p_loss, p_pos):
            assert t.shape == (1, max_seqlen)
        for cu in (
            p_params.cu_seqlens_q,
            p_params.cu_seqlens_kv,
            p_params.cu_seqlens_q_padded,
            p_params.cu_seqlens_kv_padded,
        ):
            assert cu.shape[0] == max_num_seqs + 1
        expected_cu = torch.tensor(
            [0, 100, 150, 180, 256, 256, 256, 256, 256], dtype=torch.int32, device="cuda"
        )
        assert torch.equal(p_params.cu_seqlens_q, expected_cu)
        assert torch.equal(p_params.cu_seqlens_kv, expected_cu)
        assert torch.equal(p_params.cu_seqlens_q_padded, expected_cu)
        assert torch.equal(p_params.cu_seqlens_kv_padded, expected_cu)
        assert p_params.max_seqlen_q == max_seqlen
        assert p_params.max_seqlen_kv == max_seqlen
        assert p_params.pad_between_seqs is True
        assert p_mask.shape == (1, max_seqlen) and p_mask.dtype == torch.bool
        assert torch.equal(p_tok[0, :total_T], tokens[0])
        assert (p_tok[0, total_T:] == 0).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eager_pad_to_max_appends_dummy_padding_sequence(self):
        """Default eager pad-to-max covers the tail with a dummy sequence."""
        seqlens, total_T, target_len = [50, 30], 80, 8192
        psp = _make_psp(seqlens)
        orig_cu = psp.cu_seqlens_q.clone()
        alignment, pad_target_len, max_num_seqs = get_thd_padding_kwargs(
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=target_len,
            thd_max_packed_sequences=None,
            cuda_graph_static=False,
        )

        p_tok, _, _, _, p_params, p_mask = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            psp,
            alignment=alignment,
            target_len=pad_target_len,
            max_num_seqs=max_num_seqs,
        )

        assert p_tok.shape == (1, target_len)
        expected = torch.cat(
            (orig_cu, torch.tensor([target_len], dtype=orig_cu.dtype, device=orig_cu.device))
        )
        assert torch.equal(p_params.cu_seqlens_q, expected)
        assert torch.equal(p_params.cu_seqlens_q_padded, expected)
        assert p_params.cu_seqlens_q.shape[0] == orig_cu.shape[0] + 1
        assert p_params.max_seqlen_q == target_len - total_T
        assert p_params.max_seqlen_kv == target_len - total_T
        assert p_params.total_tokens == target_len
        assert p_params.pad_between_seqs is False
        assert p_mask.shape == (1, target_len)
        assert not p_mask[0, :total_T].any()
        assert p_mask[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_mask_boundary(self):
        """False at real positions, True at padding (MoE aux-loss contract)."""
        seqlens, total_T, max_seqlen = [60, 40], 100, 128
        _, _, _, _, _, m = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            _make_psp(seqlens),
            target_len=max_seqlen,
            max_num_seqs=4,
        )
        assert not m[0, :total_T].any() and m[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_mask_preserves_existing_padding(self):
        """Existing THD padding and appended tail padding are merged in one helper."""
        seqlens, total_T, max_seqlen = [4, 4], 8, 10
        padding_mask = torch.tensor(
            [[False, False, False, True, False, False, True, True]], dtype=torch.bool, device="cuda"
        )

        _, _, _, _, _, m = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            _make_psp(seqlens),
            target_len=max_seqlen,
            max_num_seqs=4,
            padding_mask=padding_mask,
        )

        assert torch.equal(
            m,
            torch.tensor(
                [[False, False, False, True, False, False, True, True, True, True]],
                dtype=torch.bool,
                device="cuda",
            ),
        )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cp_metadata_only_uses_local_padding_mask(self, monkeypatch):
        """A middle PP stage must not repartition already CP-sliced physical rows."""
        packed_seq_params = _make_psp([15])
        cu_seqlens_padded = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
        packed_seq_params.cu_seqlens_q_padded = cu_seqlens_padded
        packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_padded.clone()
        padding_mask = torch.tensor([[False, False, False, True]], dtype=torch.bool, device="cuda")

        def fail_if_called(*_args, **_kwargs):
            pytest.fail("already sliced padding_mask must avoid TE repartitioning")

        monkeypatch.setattr(
            "megatron.core.extensions.transformer_engine.get_thd_partitioned_indices",
            fail_if_called,
        )

        _, _, _, _, padded_params, padded_mask = pad_sequence_for_thd(
            None,
            None,
            None,
            None,
            packed_seq_params,
            target_len=4,
            padding_mask=padding_mask,
            cp_size=4,
            cp_rank=3,
        )

        assert torch.equal(padded_mask, padding_mask)
        assert padded_params.cu_seqlens_q.tolist() == [0, 15]
        assert padded_params.cu_seqlens_q_padded.tolist() == [0, 16]

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cu_seqlens_fill_value(self):
        """Static cu padding repeats the dummy endpoint."""
        seqlens, total_T = [50, 30], 80
        _, _, _, _, p, _ = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            _make_psp(seqlens),
            target_len=128,
            max_num_seqs=32,
        )
        assert p.cu_seqlens_q[0] == 0 and p.cu_seqlens_q[2] == 80
        assert (p.cu_seqlens_q[3:] == 128).all()
        assert p.cu_seqlens_q_padded[0] == 0 and p.cu_seqlens_q_padded[2] == 80
        assert (p.cu_seqlens_q_padded[3:] == 128).all()
        assert p.pad_between_seqs is True

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_non_dummy_tail_padding_extends_static_padded_endpoint(self):
        """Static entry padding repeats the extended final padded endpoint."""
        cu_valid = torch.tensor([0, 18, 44, 52, 96, 118], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 24, 56, 64, 112, 144], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid.clone(),
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded.clone(),
            max_seqlen_q=48,
            max_seqlen_kv=48,
            pad_between_seqs=True,
        )

        _, _, _, _, padded, _ = pad_sequence_for_thd(
            torch.ones(1, 144, device="cuda"),
            None,
            None,
            None,
            psp,
            target_len=160,
            max_num_seqs=8,
            tail_padding_policy="extend_last",
        )

        expected_valid = torch.tensor(
            [0, 18, 44, 52, 96, 118, 118, 118, 118], dtype=torch.int32, device="cuda"
        )
        expected_padded = torch.tensor(
            [0, 24, 56, 64, 112, 160, 160, 160, 160], dtype=torch.int32, device="cuda"
        )
        assert torch.equal(padded.cu_seqlens_q, expected_valid)
        assert torch.equal(padded.cu_seqlens_kv, expected_valid)
        assert torch.equal(padded.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(padded.cu_seqlens_kv_padded, expected_padded)
        assert padded.max_seqlen_q == 160
        assert padded.max_seqlen_kv == 160
        assert padded.pad_between_seqs is True

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_none_inputs(self):
        """Non-pre_process PP: mask from cu_seqlens when all tensors None."""
        seqlens, total_T, max_seqlen = [50, 30], 80, 128
        _, _, _, _, _, mask = pad_sequence_for_thd(
            None, None, None, None, _make_psp(seqlens), target_len=max_seqlen, max_num_seqs=4
        )
        assert mask.shape == (1, max_seqlen)
        assert not mask[0, :total_T].any() and mask[0, total_T:].all()


# =============================================================================
# 2. PackedSeqParams decompose / reconstruct
# =============================================================================


class TestDecomposeReconstruct:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reconstruct_preserves_cu_tensors_and_uses_conservative_padding_flag(self):
        """Reconstruction preserves cu tensors and uses a graph-static padding flag."""
        psp = _make_psp([100, 50, 30])
        orig = {
            k: getattr(psp, k).clone()
            for k in (
                'cu_seqlens_q',
                'cu_seqlens_kv',
                'cu_seqlens_q_padded',
                'cu_seqlens_kv_padded',
            )
        }
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        # Use the non-default mode so losing it during reconstruction is observable.
        layer.config.cp_partition_mode = "contiguous"
        kw = {'packed_seq_params': psp, 'other': 'kept'}
        TransformerLayer._decompose_packed_seq_params_to_kwargs(kw)
        assert 'packed_seq_params' not in kw and 'cu_seqlens_q' in kw
        layer._reconstruct_packed_seq_params_from_kwargs(kw)
        r = kw['packed_seq_params']
        assert r.qkv_format == 'thd' and r.max_seqlen_q == 128
        assert r.pad_between_seqs is True
        assert r.cp_partition_mode == "contiguous"
        for k, v in orig.items():
            assert torch.equal(getattr(r, k), v)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reconstruct_preserves_inter_sequence_padding_semantics(self):
        """CUDA graph reconstruction conservatively enables valid/physical gaps."""
        cu_valid = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        cu_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_valid,
            cu_seqlens_kv=cu_valid.clone(),
            cu_seqlens_q_padded=cu_padded,
            cu_seqlens_kv_padded=cu_padded.clone(),
            max_seqlen_q=4,
            max_seqlen_kv=4,
            pad_between_seqs=True,
        )
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'packed_seq_params': psp}

        TransformerLayer._decompose_packed_seq_params_to_kwargs(kw)
        layer._reconstruct_packed_seq_params_from_kwargs(kw)

        reconstructed = kw['packed_seq_params']
        assert reconstructed.pad_between_seqs is True
        assert not torch.equal(reconstructed.cu_seqlens_q, reconstructed.cu_seqlens_q_padded)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_noop_without_packed_seq_params(self):
        """No-ops on non-THD kwargs (SBHD path)."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'hidden_states': torch.randn(10, 1, 256, device="cuda")}
        keys = set(kw.keys())
        TransformerLayer._decompose_packed_seq_params_to_kwargs(kw)
        assert set(kw.keys()) == keys
        layer._reconstruct_packed_seq_params_from_kwargs(kw)
        assert set(kw.keys()) == keys


class TestStaticInputs:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_static_padding_mask_is_unmasked_for_capture(self):
        """Capture-time padding_mask must not mark every static token as padding."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        layer.config.sequence_packing_scheduler = "dp_balanced"
        layer.config.cuda_graph_impl = "transformer_engine"

        static_inputs = layer.get_layer_static_inputs(seq_length=128, micro_batch_size=1)

        assert static_inputs["padding_mask"].shape == (1, 128)
        assert not static_inputs["padding_mask"].any()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_hash_router_input_ids_use_per_rank_capacity(self):
        """Hash routing input IDs must use the fixed [1, local_tokens] THD shape."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        layer.config.context_parallel_size = 2
        layer.config.sequence_packing_scheduler = "dp_balanced"
        layer.config.cuda_graph_impl = "transformer_engine"
        layer.config.moe_n_hash_layers = 1
        layer.is_moe_layer = True
        layer.mlp.router = torch.nn.Identity()
        layer.mlp.router.is_hash_layer = True

        static_inputs = layer.get_layer_static_inputs(seq_length=256, micro_batch_size=2)

        assert static_inputs["input_ids"].shape == (1, 128)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_chunk_static_inputs_and_packed_sequence_round_trip(self):
        block = _build_chunk(256, 4, 4, 1024, 128, 8)

        static_inputs = block.get_layer_static_inputs(seq_length=128, micro_batch_size=1)
        assert static_inputs["hidden_states"].shape == (128, 1, 256)
        assert static_inputs["cu_seqlens_q"].shape == (9,)
        assert static_inputs["cu_seqlens_kv_padded"].shape == (9,)
        assert static_inputs["padding_mask"].shape == (1, 128)
        assert not static_inputs["padding_mask"].any()

        packed_seq_params = _make_psp([64, 32])
        kwargs = {'packed_seq_params': packed_seq_params}
        block._decompose_packed_seq_params_to_kwargs(kwargs)
        block._reconstruct_packed_seq_params_from_kwargs(kwargs)

        reconstructed = kwargs['packed_seq_params']
        assert reconstructed.pad_between_seqs is True
        assert reconstructed.max_seqlen_q == 128
        assert torch.equal(reconstructed.cu_seqlens_q, packed_seq_params.cu_seqlens_q)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dynamic_cp_chunk_uses_capture_metadata_and_runtime_graph_bank(self, monkeypatch):
        from megatron.core import parallel_state
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        block = _build_chunk(256, 4, 4, 1024, 128, 8)
        groups = {2: object(), 4: object()}
        block.config.context_parallel_size = 4
        block.config.dynamic_context_parallel = True
        block.config._cuda_graph_capture_dynamic_cp = (2, groups[2])

        static_inputs = block.get_layer_static_inputs(seq_length=512, micro_batch_size=1)
        assert static_inputs["cu_seqlens_q"][-1].item() == 256

        packed_seq_params = _make_psp([128, 128])
        kwargs = {'packed_seq_params': packed_seq_params}
        block._decompose_packed_seq_params_to_kwargs(kwargs)
        block._reconstruct_packed_seq_params_from_kwargs(kwargs)
        reconstructed = kwargs['packed_seq_params']
        assert reconstructed.max_seqlen_q == 256
        assert reconstructed.local_cp_size == 2
        assert reconstructed.cp_group is groups[2]

        graph_calls = []

        def graph_for(cp_size):
            def replay(*args, **kwargs):
                graph_calls.append((cp_size, args, kwargs))
                return cp_size

            return replay

        graph_banks = {cp_size: [graph_for(cp_size)] for cp_size in groups}
        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.callables_per_chunk = [[block]]
        helper.num_microbatches = 1
        helper.config = SimpleNamespace(overlap_moe_expert_parallel_comm=False)
        helper._graphs_created = False
        helper._install_captured_graphs(
            [(cp_size, groups[cp_size]) for cp_size in groups], graph_banks
        )
        assert block.cuda_graphs_by_dynamic_cp_size == graph_banks
        assert helper._graphs_created is True
        monkeypatch.setattr(
            parallel_state,
            'get_dynamic_data_context_parallel_groups',
            lambda group_size: groups[group_size],
        )

        hidden_states = torch.ones(128, 1, 256, dtype=torch.bfloat16, device="cuda")
        for cp_size, logical_microbatch in ((2, 2), (4, 16), (2, 33)):
            runtime_psp = _make_psp([128])
            runtime_psp.local_cp_size = cp_size
            runtime_psp.cp_group = groups[cp_size]
            block.current_microbatch = logical_microbatch
            assert (
                block._te_cuda_graph_replay(hidden_states, packed_seq_params=runtime_psp) == cp_size
            )
            assert block.cuda_graphs is block.cuda_graphs_by_dynamic_cp_size[cp_size]

        assert [cp_size for cp_size, _, _ in graph_calls] == [2, 4, 2]

        for missing_metadata in (None, SimpleNamespace(local_cp_size=None, cp_group=groups[2])):
            with pytest.raises(RuntimeError, match="requires packed sequence metadata"):
                block._activate_dynamic_cp_cuda_graph(missing_metadata)

        wrong_group = _make_psp([128])
        wrong_group.local_cp_size = 2
        wrong_group.cp_group = groups[4]
        with pytest.raises(RuntimeError, match="process group"):
            block._te_cuda_graph_replay(hidden_states, packed_seq_params=wrong_group)

        unknown_size = _make_psp([128])
        unknown_size.local_cp_size = 8
        unknown_size.cp_group = object()
        with pytest.raises(RuntimeError, match="available sizes"):
            block._te_cuda_graph_replay(hidden_states, packed_seq_params=unknown_size)


class TestDynamicMicrobatchSlots:

    @pytest.mark.internal
    def test_completed_helper_rejects_recapture_before_communicator_work(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._capture_finished = True
        helper._get_dynamic_cp_capture_contexts = lambda: pytest.fail(
            "recapture reached communicator setup"
        )

        with pytest.raises(RuntimeError, match="capture has already been finished"):
            helper.create_cudagraphs()

    @pytest.mark.internal
    def test_parent_cp_transport_is_warmed_by_helper_constructor(self, monkeypatch):
        from megatron.core.pipeline_parallel import p2p_communication
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        parent_group = object()
        capture_contexts = ((8, object()), (4, object()))
        calls = []
        monkeypatch.setattr(cuda_graphs, 'HAVE_TE_GRAPHS', True)
        monkeypatch.setattr(p2p_communication, 'P2PCommunicator', lambda **kwargs: object())
        monkeypatch.setattr(TECudaGraphHelper, '_discover_layers', lambda self: None)
        monkeypatch.setattr(
            TECudaGraphHelper, '_publish_thd_rotary_seq_lens', lambda self, lengths: None
        )
        monkeypatch.setattr(TECudaGraphHelper, '_should_share_dynamic_cp_pool', lambda self: True)
        monkeypatch.setattr(
            TECudaGraphHelper, '_get_dynamic_cp_capture_contexts', lambda self: capture_contexts
        )
        monkeypatch.setattr(
            TECudaGraphHelper,
            '_warmup_dynamic_cp_communicators',
            lambda self, contexts, group: calls.append(('warmup', contexts, group)),
        )

        helper = TECudaGraphHelper(
            model=[],
            config=SimpleNamespace(
                cuda_graph_impl='transformer_engine', max_seqlen_per_dp_cp_rank=None
            ),
            seq_length=1,
            micro_batch_size=1,
            pg_collection=SimpleNamespace(
                tp=object(), dp=object(), dp_cp=parent_group, pp=object()
            ),
        )

        assert helper._reuse_parent_cp_transport is True
        assert calls == [('warmup', capture_contexts, parent_group)]

    @pytest.mark.internal
    def test_parent_cp_transport_is_limited_to_captured_attention(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
        from megatron.core.transformer.enums import CudaGraphModule

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._should_share_dynamic_cp_pool = lambda: True

        helper.config = SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.attn], cp_comm_type="p2p"
        )
        assert helper._should_reuse_dynamic_cp_p2p_transport()

        helper.config = SimpleNamespace(cuda_graph_modules=[])
        assert helper._should_reuse_dynamic_cp_p2p_transport()

        helper.config = SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.attn], cp_comm_type="a2a"
        )
        assert not helper._should_reuse_dynamic_cp_p2p_transport()

        helper.config = SimpleNamespace(cuda_graph_modules=[CudaGraphModule.moe_router])
        assert not helper._should_reuse_dynamic_cp_p2p_transport()

        assert helper._should_use_dynamic_cp_parent_router_reduction()
        helper.config = SimpleNamespace(cuda_graph_modules=[CudaGraphModule.attn])
        assert not helper._should_use_dynamic_cp_parent_router_reduction()

    @pytest.mark.internal
    @pytest.mark.parametrize("cp_comm_type", ("a2a", "all_gather", "a2a+p2p"))
    def test_dynamic_cp_graph_attention_rejects_non_p2p_transport(self, cp_comm_type):
        from megatron.core.transformer.enums import CudaGraphModule

        with pytest.raises(ValueError, match="currently supports cp_comm_type='p2p' only"):
            TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                dynamic_context_parallel=True,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn],
                cuda_graph_dynamic_microbatches=True,
                cp_comm_type=cp_comm_type,
                max_seqlen_per_dp_cp_rank=128,
                pad_packed_seq_alignment=128,
                thd_max_packed_sequences=2,
            )

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "fp8_attention_flag",
        ("fp8_dot_product_attention", "fp8_multi_head_attention", "custom_recipe"),
    )
    def test_dynamic_cp_graph_attention_rejects_fp8_attention(self, fp8_attention_flag):
        from megatron.core.enums import Fp8Recipe
        from megatron.core.transformer.enums import CudaGraphModule

        kwargs = (
            {"fp8_recipe": Fp8Recipe.custom, "fp8_quantizer_factory": "unused.factory"}
            if fp8_attention_flag == "custom_recipe"
            else {fp8_attention_flag: True}
        )
        with pytest.raises(ValueError, match="does not support FP8 DPA/MHA or custom"):
            TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                dynamic_context_parallel=True,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn],
                cuda_graph_dynamic_microbatches=True,
                cp_comm_type="p2p",
                fp8="hybrid",
                max_seqlen_per_dp_cp_rank=128,
                pad_packed_seq_alignment=128,
                thd_max_packed_sequences=2,
                **kwargs,
            )

    @pytest.mark.internal
    @pytest.mark.parametrize("fp4_attention_flag", ("fp8_dot_product_attention", "custom_recipe"))
    def test_dynamic_cp_graph_attention_rejects_fp4_attention(self, fp4_attention_flag):
        from megatron.core.enums import Fp4Recipe
        from megatron.core.transformer.enums import CudaGraphModule

        kwargs = (
            {"fp4_recipe": Fp4Recipe.custom, "fp4_quantizer_factory": "unused.factory"}
            if fp4_attention_flag == "custom_recipe"
            else {fp4_attention_flag: True}
        )
        with pytest.raises(ValueError, match="does not support FP8 DPA/MHA or custom"):
            TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                dynamic_context_parallel=True,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.attn],
                cuda_graph_dynamic_microbatches=True,
                cp_comm_type="p2p",
                fp4="e2m1",
                max_seqlen_per_dp_cp_rank=128,
                pad_packed_seq_alignment=128,
                thd_max_packed_sequences=2,
                **kwargs,
            )

    @pytest.mark.internal
    def test_router_only_graph_warms_parent_collective_without_p2p(self, monkeypatch):
        from megatron.core.pipeline_parallel import p2p_communication
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
        from megatron.core.transformer.enums import CudaGraphModule

        parent_group = object()
        calls = []
        monkeypatch.setattr(cuda_graphs, 'HAVE_TE_GRAPHS', True)
        monkeypatch.setattr(p2p_communication, 'P2PCommunicator', lambda **kwargs: object())
        monkeypatch.setattr(TECudaGraphHelper, '_discover_layers', lambda self: None)
        monkeypatch.setattr(
            TECudaGraphHelper, '_publish_thd_rotary_seq_lens', lambda self, lengths: None
        )
        monkeypatch.setattr(TECudaGraphHelper, '_should_share_dynamic_cp_pool', lambda self: True)
        monkeypatch.setattr(
            TECudaGraphHelper,
            '_warmup_dynamic_cp_parent_collective',
            lambda self, group: calls.append(('parent', group)),
        )

        config = SimpleNamespace(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.moe_router],
            max_seqlen_per_dp_cp_rank=None,
        )
        helper = TECudaGraphHelper(
            model=[],
            config=config,
            seq_length=1,
            micro_batch_size=1,
            pg_collection=SimpleNamespace(
                tp=object(), dp=object(), dp_cp=parent_group, pp=object()
            ),
        )

        assert helper._reuse_parent_cp_transport is False
        assert calls == [('parent', parent_group)]

    @pytest.mark.internal
    def test_dynamic_cp_graph_bank_and_capture_contexts(self, monkeypatch):
        from megatron.core import parallel_state
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        groups = {size: object() for size in (1, 2, 4, 8)}
        monkeypatch.setattr(
            parallel_state,
            'get_dynamic_data_context_parallel_groups',
            lambda group_size: groups[group_size],
        )
        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            dynamic_context_parallel=True, min_dynamic_context_parallel_size=1
        )
        helper.dp_cp_group = SimpleNamespace(size=lambda: 8)
        assert helper._get_dynamic_cp_capture_contexts() == [
            (size, groups[size]) for size in (8, 4, 2, 1)
        ]

        bank = {size: [f'cp{size}'] for size in groups}
        activated_static_input_banks = []
        layer = SimpleNamespace(
            cuda_graphs=[],
            cuda_graphs_by_dynamic_cp_size=bank,
            activate_te_cuda_graph_static_hidden_inputs=activated_static_input_banks.append,
        )
        params = SimpleNamespace(local_cp_size=4, cp_group=groups[4])
        TransformerLayer._activate_dynamic_cp_cuda_graph(layer, params)
        assert layer.cuda_graphs is bank[4]
        assert activated_static_input_banks == [4]

    @pytest.mark.internal
    def test_failed_capture_retry_restores_capture_only_state(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._capture_finished = False
        helper._reuse_parent_cp_transport = True
        helper._thd_rotary_seq_lens = {2: 4096}
        helper._uses_mhc_direct_write_arena = lambda: False
        logical_group = object()
        capture_contexts = [(2, logical_group)]
        helper._get_dynamic_cp_capture_contexts = lambda: capture_contexts
        helper._should_reuse_dynamic_cp_p2p_transport = lambda: True
        helper._should_use_dynamic_cp_parent_router_reduction = lambda: True
        helper._should_share_dynamic_cp_pool = lambda: False
        warmup_calls = []
        helper._warmup_dynamic_cp_communicators = lambda contexts, group: warmup_calls.append(
            ('p2p', tuple(contexts), group)
        )
        helper._warmup_dynamic_cp_parent_collective = lambda group: warmup_calls.append(
            ('router', group)
        )
        helper._get_cuda_graph_input_data = lambda: ([], {})
        helper._validate_mhc_static_hidden_inputs = lambda sample_args: None
        helper._set_dynamic_cp_capture_context = lambda context: None
        helper._finish_capturing = lambda start_time: None
        helper._publish_dynamic_cp_graph_microbatch_limit = lambda: None
        helper._abort_capturing = lambda captured_graphs: helper._clear_thd_rotary_seq_lens()
        helper.flattened_callables = []
        helper.dp_cp_group = object()
        helper.config = SimpleNamespace()
        helper.chunks_with_decoder = []
        helper.callables_per_chunk = []

        attempts = iter((RuntimeError("capture failed"), 0.0))

        def start_capturing():
            assert helper.config._cuda_graph_thd_rotary_seq_lens == {2: 4096}
            result = next(attempts)
            if isinstance(result, BaseException):
                raise result
            return result

        helper._start_capturing = start_capturing

        with pytest.raises(RuntimeError, match="capture failed"):
            helper.create_cudagraphs()
        assert helper._reuse_parent_cp_transport is True
        assert helper._capture_finished is False
        assert not hasattr(helper.config, '_cuda_graph_thd_rotary_seq_lens')

        helper.create_cudagraphs()

        assert warmup_calls == [
            ('router', helper.dp_cp_group),
            ('p2p', tuple(capture_contexts), helper.dp_cp_group),
            ('router', helper.dp_cp_group),
            ('p2p', tuple(capture_contexts), helper.dp_cp_group),
        ]
        assert helper._reuse_parent_cp_transport is True
        assert helper._capture_finished is True
        assert helper.config._cuda_graph_thd_rotary_seq_lens == {2: 4096}

    @pytest.mark.internal
    def test_failed_capture_restores_offload_and_dynamic_cp_warmup_state(self, monkeypatch):
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        calls = []
        monkeypatch.setattr(off_interface, 'enable_offload', lambda: calls.append('enable'))
        monkeypatch.setattr(off_interface, 'reset', lambda: calls.append('reset'))
        monkeypatch.setattr(cuda_graphs, '_set_capture_end', lambda: calls.append('capture_end'))
        monkeypatch.setattr(cuda_graphs.gc, 'collect', lambda: calls.append('gc'))
        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty_cache'))

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._graphs_created = True
        helper.config = SimpleNamespace(fine_grained_activation_offloading=True)
        helper.callables_per_chunk = []
        helper._should_share_dynamic_cp_pool = lambda: True
        helper._clear_moe_cudagraph_tensor_attrs = lambda: calls.append('clear_moe')
        helper._reset_after_capture = lambda: calls.append('reset_capture')
        helper._clear_thd_rotary_seq_lens = lambda: calls.append('clear_rope')

        helper._abort_capturing({})

        assert calls.index('enable') < calls.index('reset')
        assert 'clear_moe' in calls
        assert 'reset_capture' in calls
        assert 'clear_rope' in calls
        assert helper._graphs_created is False

    @pytest.mark.internal
    def test_thd_capture_rope_and_dummy_boundaries_share_sample_limit(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            max_seqlen_per_dp_cp_rank=4096,
            pad_packed_seq_alignment=4096,
            thd_max_packed_sequences=32,
        )
        helper.seq_length = 16384
        helper.thd_sequence_length_upper_bound = None

        rope_len, boundaries = helper._get_thd_capture_rotary_layout(4)
        assert rope_len == 16384
        assert boundaries[:3] == (0, 16384, 16384)

        rope_len, boundaries = helper._get_thd_capture_rotary_layout(8)
        assert rope_len == 16384
        assert boundaries[:4] == (0, 16384, 32768, 32768)

        helper.thd_sequence_length_upper_bound = 12288
        rope_len, boundaries = helper._get_thd_capture_rotary_layout(8)
        assert rope_len == 12288
        assert boundaries[:5] == (0, 12288, 24576, 32768, 32768)

        # Do not exceed the token capacity when the configured sample upper bound
        # is larger than this capture variant can hold.
        helper.seq_length = 65536
        helper.thd_sequence_length_upper_bound = None
        rope_len, boundaries = helper._get_thd_capture_rotary_layout(8)
        assert rope_len == 32768
        assert boundaries[:3] == (0, 32768, 32768)

        # Correctness fallback: one boundary slot cannot represent two bounded
        # sequences, so retain the original full-capacity RoPE table.
        helper.seq_length = 16384
        helper.config.thd_max_packed_sequences = 1
        rope_len, boundaries = helper._get_thd_capture_rotary_layout(8)
        assert rope_len == 32768
        assert boundaries == (0, 32768)

    @pytest.mark.internal
    def test_thd_capture_rope_limits_reach_every_vpp_chunk(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace()
        chunk_configs = [SimpleNamespace(), SimpleNamespace()]
        helper.model = [
            SimpleNamespace(config=helper.config, module=SimpleNamespace(config=config))
            for config in chunk_configs
        ]
        helper.chunks_with_decoder = [model.module for model in helper.model]
        rotary_seq_lens = {4: 16384, 8: 16384}

        helper._publish_thd_rotary_seq_lens(rotary_seq_lens)

        for config in (helper.config, *chunk_configs):
            assert config._cuda_graph_thd_rotary_seq_lens is rotary_seq_lens

        helper._clear_thd_rotary_seq_lens()
        for config in (helper.config, *chunk_configs):
            assert not hasattr(config, '_cuda_graph_thd_rotary_seq_lens')

    @pytest.mark.internal
    @pytest.mark.parametrize("reset_fails", (False, True))
    def test_delete_graphs_releases_capture_only_state(self, monkeypatch, reset_fails):
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        class Graph:
            reset_count = 0

            def reset(self):
                self.reset_count += 1
                if reset_fails:
                    raise RuntimeError("reset failed")

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._graphs_created = True
        helper.config = SimpleNamespace(
            _cuda_graph_num_microbatches=8, _cuda_graph_thd_rotary_seq_lens={2: 4096}
        )
        chunk_config = SimpleNamespace(_cuda_graph_thd_rotary_seq_lens={2: 4096})
        helper.chunks_with_decoder = [SimpleNamespace(config=chunk_config)]
        layer = torch.nn.Identity()
        graph = Graph()
        later_graph = Graph()
        layer.cuda_graphs = [graph, later_graph]
        layer.cuda_graphs_by_dynamic_cp_size = {}
        layer.cuda_graph_manual_hooks = []
        helper.callables_per_chunk = [[layer]]
        helper.tp_group = object()
        helper.dp_cp_group = object()
        monkeypatch.setattr(cuda_graphs, 'is_te_min_version', lambda version: True)
        monkeypatch.setattr(cuda_graphs, 'log_on_each_pipeline_stage', lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

        if reset_fails:
            with pytest.raises(RuntimeError, match="reset failed"):
                helper.delete_cuda_graphs()
        else:
            helper.delete_cuda_graphs()

        assert graph.reset_count == 1
        assert later_graph.reset_count == 1
        assert helper._graphs_created is False
        assert layer.cuda_graphs == []
        assert layer.cuda_graphs_by_dynamic_cp_size == {}
        assert not hasattr(helper.config, '_cuda_graph_num_microbatches')
        assert not hasattr(helper.config, '_cuda_graph_thd_rotary_seq_lens')
        assert not hasattr(chunk_config, '_cuda_graph_thd_rotary_seq_lens')

    @pytest.mark.internal
    def test_delete_graphs_without_graphable_layers_releases_state(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._graphs_created = False
        helper.config = SimpleNamespace(
            _cuda_graph_num_microbatches=8, _cuda_graph_thd_rotary_seq_lens={2: 4096}
        )
        helper.chunks_with_decoder = []
        helper.callables_per_chunk = []
        helper.tp_group = object()
        helper.dp_cp_group = object()
        monkeypatch.setattr(cuda_graphs, 'is_te_min_version', lambda version: True)
        monkeypatch.setattr(cuda_graphs, 'log_on_each_pipeline_stage', lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

        helper.delete_cuda_graphs()

        assert not hasattr(helper.config, '_cuda_graph_num_microbatches')
        assert not hasattr(helper.config, '_cuda_graph_thd_rotary_seq_lens')

    @pytest.mark.internal
    def test_delete_graphs_preserves_first_error_and_runs_remaining_cleanup(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        class Graph:
            def reset(self):
                raise RuntimeError("reset failed")

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper._graphs_created = True
        helper.config = SimpleNamespace(_cuda_graph_num_microbatches=8)
        helper.chunks_with_decoder = []
        layer = torch.nn.Identity()
        layer.cuda_graphs = [Graph()]
        layer.cuda_graphs_by_dynamic_cp_size = {}
        layer.cuda_graph_manual_hooks = []
        helper.callables_per_chunk = [[layer]]
        helper.tp_group = object()
        helper.dp_cp_group = object()
        cleanup_calls = []

        def fail_rope_cleanup():
            cleanup_calls.append("rope")
            raise RuntimeError("rope cleanup failed")

        helper._clear_thd_rotary_seq_lens = fail_rope_cleanup
        monkeypatch.setattr(cuda_graphs, 'is_te_min_version', lambda version: True)
        monkeypatch.setattr(cuda_graphs, 'log_on_each_pipeline_stage', lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

        with pytest.raises(RuntimeError, match="reset failed"):
            helper.delete_cuda_graphs()

        assert cleanup_calls == ["rope"]
        assert not hasattr(helper.config, '_cuda_graph_num_microbatches')
        assert layer.cuda_graphs == []

    @pytest.mark.internal
    def test_thd_graph_runtime_rope_uses_capture_sample_limit(self):
        from megatron.core.models.gpt.gpt_model import GPTModel

        model = GPTModel.__new__(GPTModel)
        object.__setattr__(
            model,
            'config',
            SimpleNamespace(
                context_parallel_size=4, _cuda_graph_thd_rotary_seq_lens={4: 16384, 8: 16384}
            ),
        )

        assert (
            model._bound_thd_rotary_seq_len(
                32768, SimpleNamespace(qkv_format='thd', local_cp_size=8)
            )
            == 16384
        )
        assert (
            model._bound_thd_rotary_seq_len(
                32768, SimpleNamespace(qkv_format='thd', local_cp_size=None)
            )
            == 16384
        )
        assert (
            model._bound_thd_rotary_seq_len(
                32768, SimpleNamespace(qkv_format='sbhd', local_cp_size=8)
            )
            == 32768
        )
        assert model._bound_thd_rotary_seq_len(32768, None) == 32768

    @pytest.mark.internal
    def test_thd_capture_dummy_boundaries_are_seeded_in_place(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        original = torch.zeros(5, dtype=torch.int32)
        static_inputs = {
            name: original.clone()
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            )
        }
        boundaries = (0, 16384, 32768, 32768, 32768)

        TECudaGraphHelper._seed_thd_capture_cu_seqlens(static_inputs, boundaries)

        for value in static_inputs.values():
            assert value.tolist() == list(boundaries)

        with pytest.raises(RuntimeError, match="has 4 entries, but 5 boundaries"):
            TECudaGraphHelper._seed_thd_capture_cu_seqlens(
                {"cu_seqlens_q": torch.zeros(4, dtype=torch.int32)}, boundaries
            )

    @pytest.mark.internal
    def test_mla_rope_tensor_follows_te_graph_lifetime(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention

        mla = MLASelfAttention.__new__(MLASelfAttention)
        torch.nn.Module.__init__(mla)
        mla.config = SimpleNamespace(cuda_graph_impl='transformer_engine')
        monkeypatch.setattr(cuda_graphs, 'is_graph_capturing', lambda: True)

        tensor = torch.ones(1)
        tensor_ref = weakref.ref(tensor)
        mla._retain_cuda_graph_rope_tensors(tensor)
        del tensor
        assert tensor_ref() is not None

        TECudaGraphHelper._clear_cuda_graph_state(mla)
        assert tensor_ref() is None

    @pytest.mark.internal
    def test_pp2_slots_track_max_outstanding_microbatches(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        order = [1, 1, -1, 1, -1, 1, -1, -1]

        assert TECudaGraphHelper._get_required_num_microbatch_slots_from_order(order, 1) == 2

    @pytest.mark.internal
    def test_vpp_slots_track_each_chunk_liveness(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        order = [1, 1, 1, 2, 2, 2, -2, 1, -2, 1, -2, 2, -1, 2, -1, -1, -2, -2, -1, -1]

        assert TECudaGraphHelper._get_required_num_microbatch_slots_from_order(order, 2) == 5

    @pytest.mark.internal
    def test_dp_balanced_thd_capture_upper_bound_uses_max_sequence_length(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        assert (
            TECudaGraphHelper._get_dp_balanced_thd_max_num_microbatches(
                global_batch_size=64,
                dp_size=1,
                cp_size=1,
                max_seqlen_per_dp_cp_rank=4096,
                max_sequence_length=4096,
                max_num_seqs=8,
            )
            == 64
        )
        assert (
            TECudaGraphHelper._get_dp_balanced_thd_max_num_microbatches(
                global_batch_size=64,
                dp_size=1,
                cp_size=2,
                max_seqlen_per_dp_cp_rank=4096,
                max_sequence_length=4096,
                max_num_seqs=8,
            )
            == 32
        )
        assert (
            TECudaGraphHelper._get_dp_balanced_thd_max_num_microbatches(
                global_batch_size=2,
                dp_size=1,
                cp_size=1,
                max_seqlen_per_dp_cp_rank=4096,
                max_sequence_length=4096,
                max_num_seqs=8,
                max_subsamples_per_item=4,
            )
            == 8
        )

    @pytest.mark.internal
    def test_dp_balanced_thd_capture_upper_bound_aligns_vpp_groups(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        assert (
            TECudaGraphHelper._get_dp_balanced_thd_max_num_microbatches(
                global_batch_size=18,
                dp_size=1,
                cp_size=1,
                max_seqlen_per_dp_cp_rank=4096,
                max_sequence_length=2048,
                microbatch_group_size_per_vp_stage=8,
                max_num_seqs=8,
            )
            == 16
        )

    @pytest.mark.internal
    def test_dynamic_cp_capture_upper_bound_uses_scheduler_rank_fill(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        get_bound = TECudaGraphHelper._get_default_dynamic_cp_thd_max_num_microbatches

        assert get_bound(64, 8, 8192, 8192, microbatch_group_size_per_vp_stage=4) == 8
        assert get_bound(64, 8, 8192, 16384, microbatch_group_size_per_vp_stage=4) == 16
        assert get_bound(64, 8, 8192, 65536, microbatch_group_size_per_vp_stage=4) == 64
        assert (
            get_bound(
                64, 8, 8192, 8192, max_subsamples_per_item=2, microbatch_group_size_per_vp_stage=4
            )
            == 16
        )
        assert get_bound(9, 8, 8192, 8192, microbatch_group_size_per_vp_stage=8) == 8

    @pytest.mark.internal
    def test_dynamic_cp_capture_upper_bound_covers_scheduler_outputs(self):
        from megatron.core.datasets.data_schedule_utils import (
            align_sample_id_groups,
            next_hdp_group_packing_aware,
        )
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        length_sets = (
            [8192] * 64,
            [128] * 64,
            [8192 if index % 2 else 128 for index in range(64)],
            [128 + (index * 7919) % 8065 for index in range(64)],
            list(range(128, 8193, 128)),
        )
        bound = TECudaGraphHelper._get_default_dynamic_cp_thd_max_num_microbatches(
            64, 8, 8192, 8192, microbatch_group_size_per_vp_stage=4
        )

        for lengths in length_sets:
            remaining = list(enumerate(lengths))
            groups = []
            while remaining:
                _, remaining, _, sample_ids = next_hdp_group_packing_aware(
                    remaining,
                    total_gpus=8,
                    max_seq_len_per_rank=8192,
                    min_cp_size=1,
                    max_num_seqs=31,
                )
                groups.append(sample_ids)
            aligned_groups = align_sample_id_groups(groups, 4)
            assert len(aligned_groups) <= bound

    @pytest.mark.internal
    def test_dynamic_slot_liveness_only_includes_vpp_aligned_counts(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        get_counts = TECudaGraphHelper._get_dynamic_slot_liveness_microbatch_counts

        assert get_counts(4) == (1, 2, 3, 4)
        assert get_counts(8, 4) == (4, 8)
        assert get_counts(16, 8) == (8, 16)
        with pytest.raises(ValueError, match="not aligned"):
            get_counts(10, 4)

    @pytest.mark.internal
    def test_pp4_vpp4_liveness_union_matches_fixed_cp4_schedule(self, monkeypatch):
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import (
            get_pp_rank_microbatches,
            get_schedule_table,
        )
        from megatron.core.transformer.cuda_graphs import (
            TECudaGraphHelper,
            convert_schedule_table_to_order,
        )

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 4)
        monkeypatch.setattr(
            parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: 4
        )

        expected_color_counts = (60, 60, 48, 48)
        for pp_rank, expected_colors in enumerate(expected_color_counts):
            monkeypatch.setattr(
                parallel_state, "get_pipeline_model_parallel_rank", lambda rank=pp_rank: rank
            )
            conflicts_by_count = {}
            colors_by_count = {}
            for num_microbatches in range(4, 65, 4):
                _, _, warmup, _ = get_pp_rank_microbatches(
                    num_microbatches, 4, 4, forward_only=False
                )
                order = convert_schedule_table_to_order(
                    warmup, 4, get_schedule_table(num_microbatches, 4, 4)
                )
                colors, conflicts = TECudaGraphHelper._build_saved_tensor_liveness_colors(
                    order, 8, [3, 3, 3, 3], return_conflicts=True
                )
                colors_by_count[num_microbatches] = colors
                conflicts_by_count[num_microbatches] = conflicts

            union_conflicts = {
                frame: set().union(*(conflicts[frame] for conflicts in conflicts_by_count.values()))
                for frame in conflicts_by_count[4]
            }
            for num_microbatches in range(12, 65, 4):
                assert conflicts_by_count[num_microbatches] == union_conflicts
            assert conflicts_by_count[32] == union_conflicts
            assert len(set(colors_by_count[32].values())) == expected_colors

    @pytest.mark.internal
    def test_dynamic_cp_variant_order_runs_one_schedule_per_variant(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        order = [1, 2, 1, 2, -1, -2, -1, -2]
        combined = TECudaGraphHelper._build_slot_aliased_variant_order(
            order, 2, 2, canonical_variant=0
        )

        assert combined == [1, 2, 1, 2, -1, -2, -1, -2, 3, 4, 3, 4, -3, -4, -3, -4]

    @pytest.mark.internal
    def test_slot_memory_colors_follow_pp_vpp_liveness(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        sequential = TECudaGraphHelper._build_saved_tensor_liveness_colors(
            [1, 1, -1, -1, 2, 2, -2, -2], 2, [1, 1]
        )
        overlapping = TECudaGraphHelper._build_saved_tensor_liveness_colors(
            [1, 1, 2, 2, -2, -2, -1, -1], 2, [1, 1]
        )

        assert sequential[(0, 0, 0)] != sequential[(0, 1, 0)]
        assert overlapping[(0, 0, 0)] != overlapping[(0, 1, 0)]
        assert len(set(overlapping.values())) > len(set(sequential.values()))
        with pytest.raises(RuntimeError, match="slot period is shorter"):
            TECudaGraphHelper._build_saved_tensor_liveness_colors([1, 1, -1, -1], 1, [1])

    @pytest.mark.internal
    def test_user_grad_colors_separate_adjacent_vpp_chunks(self):
        """An overlapped PP send must survive the next chunk's backward graph."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        colors = TECudaGraphHelper._build_user_grad_liveness_colors(
            [1, 2, -2, -1], num_slots=1, num_model_chunks=2
        )

        assert colors[(0, 0)] != colors[(1, 0)]
        assert len(set(colors.values())) == 2

    @pytest.mark.internal
    def test_dynamic_cp_variants_emit_one_compact_slot_plan(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.num_microbatches = 2
        helper.num_model_chunks = 1
        helper.num_layers_per_chunk = [1]
        helper.flattened_callables = [torch.nn.Identity()]
        helper._set_dynamic_cp_capture_context = lambda context: None
        order = [1, 1, -1, 1, -1, 1, -1, -1]

        def make_bank(cp_size, offset):
            return (
                cp_size,
                object(),
                [(offset + index,) for index in range(2)],
                {
                    '_order': order,
                    '_num_layers_per_chunk': [1],
                    '_reuse_graph_input_output_buffers': True,
                },
            )

        banks = [make_bank(2, 0), make_bank(1, 10)]
        callables, sample_args, kwargs = helper._get_dynamic_cp_variant_capture_data(banks)
        colors = TECudaGraphHelper._build_saved_tensor_liveness_colors(order, 2, [1])
        grad_colors = TECudaGraphHelper._build_user_grad_liveness_colors(order, 2, 1)
        expected_slots = tuple(
            (
                colors[(0, logical_slot % 2, 0)],
                logical_slot,
                variant * 2 + logical_slot,
                0,
                0,
                variant,
                grad_colors[(0, logical_slot % 2)],
            )
            for variant in range(2)
            for logical_slot in range(2)
        )

        assert len(callables) == 2
        assert sample_args == tuple((index,) for index in range(2)) + tuple(
            (10 + index,) for index in range(2)
        )
        assert kwargs['_graph_memory_slots'] == expected_slots
        assert kwargs['_num_layers_per_chunk'] == [1, 1]
        assert not any(
            key in kwargs
            for key in (
                '_saved_tensor_memory_alias_groups',
                '_slot_io_memory_alias_groups',
                '_slot_io_liveness_groups',
                '_warmup_plan_alias_groups',
            )
        )

    @pytest.mark.internal
    def test_dynamic_cp_slot_plan_covers_unaligned_capture_order(self):
        """A synthetic W-slot capture order must participate in liveness coloring."""
        from megatron.core.pipeline_parallel.schedules import get_schedule_table
        from megatron.core.transformer.cuda_graphs import (
            TECudaGraphHelper,
            convert_schedule_table_to_order,
        )

        def get_order(num_microbatches):
            # PP2 rank 0, VPP2, group size 4 has six warmup virtual microbatches.
            return tuple(
                convert_schedule_table_to_order(6, 2, get_schedule_table(num_microbatches, 2, 4))
            )

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.num_microbatches = 7
        helper.num_model_chunks = 2
        helper.num_layers_per_chunk = [1, 1]
        helper.flattened_callables = [torch.nn.Identity(), torch.nn.Identity()]
        helper._set_dynamic_cp_capture_context = lambda context: None
        helper._dynamic_slot_liveness_orders = (get_order(4), get_order(8))
        base_order = get_order(7)
        bank = (
            1,
            object(),
            [(index,) for index in range(14)],
            {
                '_order': base_order,
                '_num_layers_per_chunk': [1, 1],
                '_reuse_graph_input_output_buffers': True,
            },
        )

        _, _, kwargs = helper._get_dynamic_cp_variant_capture_data([bank])
        saved_arenas = {
            (model_chunk, physical_slot): saved_arena
            for (saved_arena, physical_slot, _, model_chunk, _, _, _) in kwargs[
                '_graph_memory_slots'
            ]
        }

        # These frames overlap only in the synthetic M=W=7 capture schedule, not in
        # either scheduler-reachable M=4/8 schedule.
        assert saved_arenas[(0, 2)] != saved_arenas[(1, 5)]
        assert saved_arenas[(0, 3)] != saved_arenas[(1, 6)]

    @pytest.mark.internal
    def test_dynamic_cp_capture_propagates_te_rebound_static_inputs(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.num_microbatches = 2
        helper.num_model_chunks = 1
        helper.num_layers_per_chunk = [1]
        helper.flattened_callables = [torch.nn.Identity()]
        helper._set_dynamic_cp_capture_context = lambda context: None
        order = [1, 1, -1, 1, -1, 1, -1, -1]
        banks = [
            (
                cp_size,
                object(),
                [(cp_size, slot) for slot in range(2)],
                {
                    '_order': order,
                    '_num_layers_per_chunk': [1],
                    '_reuse_graph_input_output_buffers': True,
                },
            )
            for cp_size in (2, 1)
        ]
        rebound_inputs = ((object(),), (object(),))

        def capture_graph_bank(_callables, sample_args, _kwargs, *, paged_stash_order=None):
            assert paged_stash_order == order
            assert isinstance(sample_args, list)
            sample_args[1] = rebound_inputs[0]
            sample_args[3] = rebound_inputs[1]
            return [object() for _ in sample_args]

        helper._capture_graph_bank = capture_graph_bank
        monkeypatch.setattr(cuda_graphs, 'log_on_each_pipeline_stage', lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

        captured_graphs = {}
        helper._capture_dynamic_cp_variants(banks, captured_graphs)

        assert banks[0][2][1] is rebound_inputs[0]
        assert banks[1][2][1] is rebound_inputs[1]
        assert len(captured_graphs[2]) == 2
        assert len(captured_graphs[1]) == 2

    @pytest.mark.internal
    def test_shared_slots_publish_logical_not_physical_microbatch_limit(self):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            dynamic_context_parallel=True,
            cuda_graph_dynamic_microbatches=True,
            _cuda_graph_num_microbatches=32,
        )
        helper.pp_group = SimpleNamespace(size=lambda: 8)
        helper.num_microbatches = 16
        helper._dynamic_slot_liveness_limit = 64

        helper._publish_dynamic_cp_graph_microbatch_limit()
        assert helper.config._cuda_graph_num_microbatches == 64

        helper._dynamic_slot_liveness_limit = None
        helper.num_microbatches = 32
        helper._publish_dynamic_cp_graph_microbatch_limit()
        assert helper.config._cuda_graph_num_microbatches == 32

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "fallback", ("overlap_moe_expert_parallel_comm", "delay_wgrad_compute")
    )
    def test_shared_slots_fall_back_for_unsupported_graph_orders(self, fallback):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            dynamic_context_parallel=True,
            cuda_graph_dynamic_microbatches=True,
            overlap_moe_expert_parallel_comm=False,
            delay_wgrad_compute=False,
        )
        setattr(helper.config, fallback, True)

        assert not helper._should_share_dynamic_cp_pool()

    @pytest.mark.internal
    def test_pp8_vpp4_uses_sixteen_noncolliding_slots(self):
        from megatron.core.pipeline_parallel.schedules import get_schedule_table
        from megatron.core.transformer.cuda_graphs import (
            TECudaGraphHelper,
            convert_schedule_table_to_order,
        )

        pp_size = 8
        num_model_chunks = 4
        microbatch_group_size = 8
        num_microbatches = pp_size * num_model_chunks * 4
        schedule_table = get_schedule_table(
            num_microbatches, num_model_chunks, microbatch_group_size
        )
        forward_ops = list(schedule_table)
        backward_ops = [
            (microbatch, num_model_chunks - chunk - 1) for microbatch, chunk in schedule_table
        ]

        for pp_rank in range(pp_size):
            num_warmup = (pp_size - pp_rank - 1) * 2
            num_warmup += (num_model_chunks - 1) * microbatch_group_size
            order = convert_schedule_table_to_order(num_warmup, num_model_chunks, schedule_table)
            num_slots = TECudaGraphHelper._get_required_num_microbatch_slots_from_order(
                order, num_model_chunks
            )
            assert num_slots == 16

            events = [('forward', *op) for op in forward_ops[:num_warmup]]
            for index in range(num_warmup, len(forward_ops)):
                events.append(('forward', *forward_ops[index]))
                events.append(('backward', *backward_ops[index - num_warmup]))
            events.extend(('backward', *op) for op in backward_ops[-num_warmup:])
            live_slots = [dict() for _ in range(num_model_chunks)]
            for phase, microbatch, chunk in events:
                slot = microbatch % num_slots
                if phase == 'forward':
                    assert slot not in live_slots[chunk]
                    live_slots[chunk][slot] = microbatch
                else:
                    assert live_slots[chunk].pop(slot) == microbatch
            assert all(not slots for slots in live_slots)


# =============================================================================
# 3. E2E no-graph vs graph bitwise loss/grad_norm match
#    Subprocess-launches `torchrun pretrain_gpt.py` -- same recipe as
#    test_moonlight_qwen3_bitwise.sh -- and asserts the per-iteration
#    metric strings are byte-identical between the two runs.
# =============================================================================

# Common args shared across both models.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_VARLEN_JSON = (
    '{"mode":"distribution","type":"lognormal",'
    '"format":"thd","min_seq_len":512,"max_seq_len":4096,'
    '"mean_seq_len":3072,"lognormal_sigma":1.1}'
)

_QWEN3_VARLEN_JSON = (
    '{"mode":"distribution","type":"lognormal",'
    '"format":"thd","min_seq_len":128,"max_seq_len":1024,'
    '"mean_seq_len":512,"lognormal_sigma":0.8}'
)

_TRAIN_ITERS = 5

_COMMON_ARGS = [
    "--seq-length",
    "4096",
    "--max-position-embeddings",
    "8192",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "64",
    "--train-iters",
    str(_TRAIN_ITERS),
    "--lr",
    "1e-5",
    "--min-lr",
    "1e-6",
    "--lr-decay-style",
    "cosine",
    "--lr-warmup-iters",
    "1",
    "--weight-decay",
    "0.01",
    "--clip-grad",
    "1.0",
    "--seed",
    "1234",
    "--te-rng-tracker",
    "--bf16",
    "--tensor-model-parallel-size",
    "2",
    "--pipeline-model-parallel-size",
    "2",
    "--context-parallel-size",
    "2",
    "--swiglu",
    "--disable-bias-linear",
    "--sequence-parallel",
    "--use-varlen-dataset",
    "--mock-data",
    "--tokenizer-type",
    "NullTokenizer",
    "--varlen-mock-dataset-config-json",
    _VARLEN_JSON,
    "--sequence-packing-scheduler",
    "dp_balanced",
    "--max-seqlen-per-dp-cp-rank",
    "4096",
    "--pad-packed-seq-alignment",
    "max",
    "--thd-tail-padding-policy",
    "extend_last",
    "--calculate-per-token-loss",
    "--transformer-impl",
    "transformer_engine",
    "--attention-dropout",
    "0",
    "--hidden-dropout",
    "0",
    "--no-bias-swiglu-fusion",
    "--no-gradient-accumulation-fusion",
    "--no-save-optim",
    "--no-save-rng",
    "--save-interval",
    "999999",
    "--eval-interval",
    "999999",
    "--eval-iters",
    "1",
    "--log-interval",
    "1",
    "--no-check-for-nan-in-loss-and-grad",
    "--deterministic-mode",
    "--thd-max-packed-sequences",
    "8",
]


def _with_arg_replacements(args, replacements):
    args = list(args)
    for name, value in replacements.items():
        idx = args.index(name)
        args[idx + 1] = value
    return args


_QWEN3_COMMON_ARGS = _with_arg_replacements(
    _COMMON_ARGS,
    {
        "--seq-length": "1024",
        "--varlen-mock-dataset-config-json": _QWEN3_VARLEN_JSON,
        "--max-seqlen-per-dp-cp-rank": "512",
    },
)


_MOONLIGHT_ARGS = _COMMON_ARGS + [
    "--num-layers",
    "27",
    "--hidden-size",
    "2048",
    "--ffn-hidden-size",
    "11264",
    "--num-attention-heads",
    "16",
    "--decoder-first-pipeline-num-layers",
    "13",
    "--decoder-last-pipeline-num-layers",
    "14",
    "--expert-model-parallel-size",
    "4",
    "--expert-tensor-parallel-size",
    "1",
    "--multi-latent-attention",
    "--kv-lora-rank",
    "512",
    "--qk-head-dim",
    "128",
    "--qk-pos-emb-head-dim",
    "64",
    "--v-head-dim",
    "128",
    "--num-experts",
    "64",
    "--moe-ffn-hidden-size",
    "1408",
    "--moe-router-topk",
    "6",
    "--moe-shared-expert-intermediate-size",
    "2816",
    "--moe-layer-freq",
    "([0]+[1]*26)",
    "--moe-token-dispatcher-type",
    "flex",
    "--moe-flex-dispatcher-backend",
    "hybridep",
    "--moe-router-fusion",
    "--moe-router-score-function",
    "sigmoid",
    "--moe-router-topk-scaling-factor",
    "2.446",
    "--moe-router-load-balancing-type",
    "aux_loss",
    "--moe-aux-loss-coeff",
    "0.001",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-5",
    "--rotary-base",
    "50000",
    "--vocab-size",
    "163840",
]

_QWEN3_ARGS = _QWEN3_COMMON_ARGS + [
    "--num-layers",
    "36",
    "--hidden-size",
    "4096",
    "--ffn-hidden-size",
    "12288",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--max-position-embeddings",
    "40960",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "1000000",
    "--untie-embeddings-and-output-weights",
    "--vocab-size",
    "151936",
    "--moe-token-dispatcher-type",
    "flex",
    "--moe-flex-dispatcher-backend",
    "hybridep",
]

_ATTN_CUDA_GRAPH_ARGS = [
    "--cuda-graph-impl",
    "transformer_engine",
    "--cuda-graph-dynamic-microbatches",
    "--cuda-graph-modules",
    "attn",
]

_MOE_CUDA_GRAPH_ARGS = _ATTN_CUDA_GRAPH_ARGS + ["moe_preprocess", "moe_router"]


def _get_available_port(preferred):
    """Return preferred if free, otherwise ask the OS for an available localhost port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("Could not find an available localhost port")


def _run_pretrain(model_args, cuda_graph_args, master_port, extra_env=None, timeout=900):
    """Subprocess-launch `torchrun pretrain_gpt.py` once and capture stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["NCCL_ALGO"] = "^NVLS"
    # Strip any inherited torchrun env so this subprocess starts a fresh group.
    for k in list(env.keys()):
        if k.startswith(
            (
                "TORCHELASTIC_",
                "MASTER_",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "GROUP_RANK",
                "LOCAL_WORLD_SIZE",
            )
        ):
            env.pop(k, None)
    # Clear pytest-conftest env vars that disable TE attention backends
    # (set by tests/unit_tests/conftest.py::set_env). Pretrain needs at
    # least one of fused/flash attention to build the model.
    env.pop("NVTE_FLASH_ATTN", None)
    env.pop("NVTE_FUSED_ATTN", None)
    if extra_env:
        env.update(extra_env)

    cmd = (
        [
            "torchrun",
            "--nproc_per_node",
            "8",
            "--nnodes",
            "1",
            "--master_addr",
            "localhost",
            "--master_port",
            str(_get_available_port(master_port)),
            "pretrain_gpt.py",
        ]
        + model_args
        + cuda_graph_args
    )

    result = subprocess.run(
        cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout
    )
    return result


_ITER_START_RE = re.compile(r"iteration\s+(\d+)/\s*\d+ \|")


def _extract_metrics(stdout):
    """Extract deterministic per-iteration fields from a training log.

    Captured torchrun stdout interleaves writes from multiple ranks at the byte
    level (no newline between rank-0's iter line and rank-7's "Number of
    parameters" line, e.g.). So we cannot rely on full-line matching: we locate
    each `iteration N/M |` marker and pull the deterministic fields by name
    from a small window after it. Wall-clock `elapsed time per iteration`
    is intentionally excluded.
    """
    results = []
    for m in _ITER_START_RE.finditer(stdout):
        window = stdout[m.start() : m.start() + 800]
        lr = re.search(r"learning rate:\s*(\S+)", window)
        lm_loss = re.search(r"lm loss:\s*(\S+)", window)
        grad_norm = re.search(r"grad norm:\s*(\S+)", window)
        if not (lr and lm_loss and grad_norm):
            continue
        parts = [f"iter={m.group(1)}", f"lr={lr.group(1)}", f"lm_loss={lm_loss.group(1)}"]
        parts.append(f"grad_norm={grad_norm.group(1)}")
        results.append(" | ".join(parts))
    return results


@pytest.mark.internal
@pytest.mark.skip(reason="Temporarily disabled until the required Transformer Engine PR lands.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.parametrize(
    "model_name,model_args,cuda_graph_args,base_port",
    [
        ("moonlight", _MOONLIGHT_ARGS, _MOE_CUDA_GRAPH_ARGS, 29660),
        ("qwen3", _QWEN3_ARGS, _ATTN_CUDA_GRAPH_ARGS, 29662),
    ],
)
class TestE2EBitwise:
    """End-to-end bitwise comparison: pretrain_gpt.py noGraph vs cudaGraph.

    Each test launches `torchrun pretrain_gpt.py` twice -- once without CUDA
    graphs and once with `cuda_graph_impl=transformer_engine` -- using the same
    model/test settings as test_moonlight_qwen3_bitwise.sh. Moonlight covers
    attn/moe_preprocess/moe_router graphs with router fusion; Qwen3 covers attn
    graphs because this test's Qwen3 recipe is dense.
    Asserts the per-iteration `lm loss / grad norm` lines are byte-identical.

    Slow (~5 min per model). Marked `internal` so CI can opt-in.
    """

    def test_no_graph_vs_graph(self, model_name, model_args, cuda_graph_args, base_port):
        # No graph baseline.
        r1 = _run_pretrain(model_args, cuda_graph_args=[], master_port=base_port)
        assert r1.returncode == 0, (
            f"[{model_name}] noGraph pretrain failed (rc={r1.returncode})\n"
            f"--- stdout (tail) ---\n{r1.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r1.stderr[-2000:]}"
        )
        metrics_eager = _extract_metrics(r1.stdout)
        assert len(metrics_eager) == _TRAIN_ITERS, (
            f"[{model_name}] noGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_eager)}\n"
            f"--- stdout (tail) ---\n{r1.stdout[-2000:]}"
        )

        # CUDA graph capture.
        r2 = _run_pretrain(model_args, cuda_graph_args=cuda_graph_args, master_port=base_port + 1)
        assert r2.returncode == 0, (
            f"[{model_name}] cudaGraph pretrain failed (rc={r2.returncode})\n"
            f"--- stdout (tail) ---\n{r2.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r2.stderr[-2000:]}"
        )
        metrics_graph = _extract_metrics(r2.stdout)
        assert len(metrics_graph) == _TRAIN_ITERS, (
            f"[{model_name}] cudaGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_graph)}\n"
            f"--- stdout (tail) ---\n{r2.stdout[-2000:]}"
        )

        # Bitwise compare per iteration.
        for i, (a, b) in enumerate(zip(metrics_eager, metrics_graph)):
            assert a == b, f"[{model_name}] iter {i+1} differs:\n" f"  eager: {a}\n" f"  graph: {b}"
