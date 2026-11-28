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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule import _build_thd_padding_mask
from megatron.core.extensions.transformer_engine import HAVE_TE
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


@pytest.mark.internal
@pytest.mark.parametrize("input_path", ["embedding", "precomputed", "pipeline"])
@pytest.mark.parametrize("mtp_enabled", [False, True])
def test_hybrid_padding_mask_sp_scatter_uses_model_tp_group(monkeypatch, input_path, mtp_enabled):
    """Align the decoder mask once while preserving the full mask for MTP."""
    from megatron.core import tensor_parallel
    from megatron.core.models.hybrid import hybrid_model

    tp_group = object()
    scatter_calls = []
    mtp_masks = []
    input_ids = torch.arange(8).view(1, 8)
    caller_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    local_hidden = torch.zeros(4, 1, 8)

    def fake_sp_scatter(input_, group=None):
        scatter_calls.append((input_.dtype, group))
        return input_.narrow(0, 0, input_.shape[0] // 2)

    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", fake_sp_scatter)

    class FakeEmbedding:
        scatter_to_sequence_parallel = False

        def __call__(self, input_ids, position_ids):
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 8)

    class FakeDecoder:
        input_tensor = local_hidden

        def __call__(self, *, hidden_states, padding_mask, **_kwargs):
            if hidden_states is None:
                hidden_states = self.input_tensor
            assert padding_mask.shape == (1, hidden_states.shape[0])
            return hidden_states

    class FakeMTP:
        def prepare_cp_layout(self, **kwargs):
            assert kwargs["padding_mask"] is caller_padding_mask
            return SimpleNamespace(**kwargs)

        def __call__(self, *, hidden_states, padding_mask, **_kwargs):
            assert padding_mask is caller_padding_mask
            mtp_masks.append(padding_mask)
            return hidden_states

    monkeypatch.setattr(hybrid_model, "resolve_cp_group", lambda group, params: None)
    monkeypatch.setattr(hybrid_model, "prepare_mtp_sequence_roll_context", lambda **kwargs: None)
    model = SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=False,
            moe_paged_stash=False,
            sequence_parallel=True,
            multi_latent_attention=False,
            moe_num_hash_layers=0,
            mtp_num_layers=1 if mtp_enabled else None,
            freeze_base_model_for_mtp=False,
        ),
        hybrid_layer_config_list=None,
        pre_process=input_path != "pipeline",
        post_process=False,
        embedding=FakeEmbedding(),
        decoder=FakeDecoder(),
        pg_collection=SimpleNamespace(tp=tp_group, cp=None),
        position_embedding_type=None,
        share_embeddings_and_output_weights=False,
        mtp_process=mtp_enabled,
        mtp=FakeMTP(),
    )
    output = hybrid_model.HybridModel.forward(
        model,
        input_ids=input_ids,
        position_ids=input_ids,
        attention_mask=None,
        padding_mask=caller_padding_mask,
        decoder_input=local_hidden if input_path == "precomputed" else None,
    )

    assert output.shape == (4, 1, 8)
    expected_scatter = [(torch.float32, tp_group)] if input_path == "embedding" else []
    assert scatter_calls == expected_scatter + [(torch.bool, tp_group)]
    assert len(mtp_masks) == int(mtp_enabled)


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
        expected_local_target = get_thd_partitioned_indices(
            psp.cu_seqlens_q, expected_global_target, cp_size, cp_rank
        ).numel()

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
    @pytest.mark.parametrize("scope", ["layer", "block"])
    def test_reconstruct_preserves_cu_tensors_and_uses_conservative_padding_flag(self, scope):
        """Reconstruction preserves cu tensors and uses a graph-static padding flag — for a layer
        callable and for the chunk-wise block callable alike."""
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
        if scope == "layer":
            layer = _build_layer(256, 4, 4, 1024, 128, 8)
        else:
            layer = _build_chunk_gpt_model(256, 4, 4, 1024, 128, 8).decoder
        # Use the non-default mode so losing it during reconstruction is observable.
        layer.config.cp_partition_mode = "contiguous"
        kw = {'packed_seq_params': psp, 'other': 'kept'}
        layer._decompose_packed_seq_params_to_kwargs(kw)
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

        layer._decompose_packed_seq_params_to_kwargs(kw)
        layer._reconstruct_packed_seq_params_from_kwargs(kw)

        reconstructed = kw['packed_seq_params']
        assert reconstructed.pad_between_seqs is True
        assert not torch.equal(reconstructed.cu_seqlens_q, reconstructed.cu_seqlens_q_padded)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_layout_cp_batch_kwargs_are_dropped_at_replay(self):
        """A hybrid model called with a ContextParallelBatch (pretrain_hybrid.py always passes
        one) forwards its per-layout metadata to the decoder next to packed_seq_params and
        padding_mask. Graph replay accepts tensors only; for a single layout the dicts repeat
        packed_seq_params / padding_mask and the plan is None, so all three are dropped. So is
        the plan the data pipeline builds for every pack when linear_cp_layout differs from
        attention_cp_layout, on a stack that converts no layout (its eager forward ignores it
        too); only a stack that would convert layouts per layer refuses it."""
        block = _build_chunk_gpt_model(256, 4, 4, 1024, 128, 8).decoder
        psp = _make_psp([64, 64])
        padding_mask = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
        # The decoder keywords of HybridModel.forward.
        kw = {
            'hidden_states': torch.randn(128, 1, 256, device="cuda"),
            'attention_mask': None,
            'inference_context': None,
            'rotary_pos_emb': None,
            'packed_seq_params': psp,
            'padding_mask': padding_mask,
            'padding_mask_by_layout': {'contiguous': padding_mask},
            'packed_seq_params_by_layout': {'contiguous': psp},
            'cp_layout_plan': None,
            'input_ids': None,
        }
        block._decompose_packed_seq_params_to_kwargs(kw)
        assert not kw.keys() & {
            'packed_seq_params_by_layout',
            'padding_mask_by_layout',
            'cp_layout_plan',
        }
        assert kw['padding_mask'] is padding_mask
        assert all(v is None or torch.is_tensor(v) for v in kw.values())
        for layout_manager in (None, SimpleNamespace(requires_conversion=False)):
            block._cp_layout_manager = layout_manager
            unused_plan = {'cp_layout_plan': object()}
            block._decompose_packed_seq_params_to_kwargs(unused_plan)
            assert unused_plan == {}
        block._cp_layout_manager = SimpleNamespace(requires_conversion=True)
        with pytest.raises(RuntimeError, match="layout plan"):
            block._decompose_packed_seq_params_to_kwargs({'cp_layout_plan': object()})

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_noop_without_packed_seq_params(self):
        """No-ops on non-THD kwargs (SBHD path)."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'hidden_states': torch.randn(10, 1, 256, device="cuda")}
        keys = set(kw.keys())
        layer._decompose_packed_seq_params_to_kwargs(kw)
        assert set(kw.keys()) == keys
        layer._reconstruct_packed_seq_params_from_kwargs(kw)
        assert set(kw.keys()) == keys


def _build_chunk_gpt_model(
    H,
    nh,
    nkv,
    ffn,
    max_seqlen,
    max_num_seqs,
    *,
    tp=1,
    sp=False,
    mtp_layers=0,
    vocab=128,
    cuda_graph_impl="chunk_wise",
    packed=True,
    position_embedding_type="rope",
    max_sequence_length=None,
    pre_process=True,
    **config_overrides,
):
    """GPTModel of a chunk-graph training run: packed sequences (THD) with a static capacity of
    ``max_seqlen`` tokens, or with ``packed=False`` fixed-length batches of that sequence length.
    ``cuda_graph_impl="none"`` builds the same run without graphs (see
    ``_build_chunk_gpt_model_and_eager_twin``)."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    config_kwargs = dict(
        num_layers=2,
        hidden_size=H,
        num_attention_heads=nh,
        num_query_groups=nkv,
        ffn_hidden_size=ffn,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mtp_num_layers=mtp_layers or None,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
        use_cpu_initialization=True,
    )
    if packed:
        config_kwargs.update(
            max_seqlen_per_dp_cp_rank=max_seqlen,
            thd_max_packed_sequences=max_num_seqs,
            cuda_graph_dynamic_microbatches=cuda_graph_impl != "none",
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
        )
    config_kwargs.update(config_overrides)
    config = TransformerConfig(**config_kwargs)
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        enable_hyper_connection=config.enable_hyper_connections,
    )
    mtp_spec = (
        get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True)
        if mtp_layers
        else None
    )
    model_parallel_cuda_manual_seed(42)
    return _in_bf16(
        GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab,
            max_sequence_length=max_sequence_length or max_seqlen,
            pre_process=pre_process,
            position_embedding_type=position_embedding_type,
            mtp_block_spec=mtp_spec,
        )
    )


def _in_bf16(model):
    """The model on the GPU with its parameters in BF16, as a ``--bf16`` run holds them
    (``Float16Module``). The layers create theirs in ``params_dtype`` already; a learned
    position table is created in FP32, and with it the eager decoder would get FP32 hidden
    states where the graph's static input is BF16."""
    from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

    return convert_module_to_dtype_except_fp32_marked(model.cuda(), torch.bfloat16)


def _as_training_model_and_eager_twin(model, twin):
    """Give the eager twin the model's weights and put both in training mode."""
    twin.load_state_dict(model.state_dict(), strict=True)
    model.train()
    twin.train()
    # The capture helper's post-capture reset calls the DDP wrapper's zero_grad_buffer(); the
    # tests drive the bare model and zero the gradients themselves (_train_step).
    model.zero_grad_buffer = lambda: None
    return model, twin


def _build_chunk_gpt_model_and_eager_twin(*args, **kwargs):
    """A chunk-graph GPTModel and its eager twin: the same run with ``cuda_graph_impl="none"``
    and the same weights. The twin never takes a chunk-graph code path (no post-process block,
    no chunk-specific checkpoint or precision selection), so it is the reference a "matches
    eager" comparison needs; the chunk-configured model's own forward before the capture runs
    the post-process block and every ``chunk_wise``-gated branch already."""
    model = _build_chunk_gpt_model(*args, **kwargs)
    twin = _build_chunk_gpt_model(*args, cuda_graph_impl="none", **kwargs)
    return _as_training_model_and_eager_twin(model, twin)


def _build_chunk_hybrid_model(
    max_seqlen,
    max_num_seqs,
    *,
    tp=1,
    sp=False,
    cp=1,
    vocab=128,
    cuda_graph_impl="chunk_wise",
    mtp_depths=1,
    mtp_hsm=False,
):
    """HybridModel (``*-`` decoder, ``mtp_depths`` ``*-`` MTP depths) of a packed-sequence
    chunk-graph run: attention and MLP are separate layers of the ``HybridStack`` the decoder
    graph captures, and the MTP's inner stack runs inside the post-process graph."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=512,
        is_hybrid_model=True,
        mtp_num_layers=mtp_depths,
        mtp_hsm=mtp_hsm,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        context_parallel_size=cp,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
        cuda_graph_dynamic_microbatches=cuda_graph_impl != "none",
        sequence_packing_scheduler="dp_balanced",
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=max_seqlen,
        thd_max_packed_sequences=max_num_seqs,
        use_cpu_initialization=True,
    )
    model_parallel_cuda_manual_seed(42)
    return _in_bf16(
        HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab,
            max_sequence_length=max_seqlen,
            hybrid_layer_pattern="*-" + "/*-" * mtp_depths,
            position_embedding_type="rope",
        )
    )


def _build_chunk_hybrid_model_and_eager_twin(*args, **kwargs):
    """Like ``_build_chunk_gpt_model_and_eager_twin``, for ``_build_chunk_hybrid_model``."""
    model = _build_chunk_hybrid_model(*args, **kwargs)
    twin = _build_chunk_hybrid_model(*args, cuda_graph_impl="none", **kwargs)
    return _as_training_model_and_eager_twin(model, twin)


# Drop-and-pad MoE (static token shapes, which a captured block needs) whose routers read the
# padding mask: the z-loss and the load-balancing loss are taken over the valid tokens only.
# With top-2 of 4 experts no expert receives more than one copy of every token, so a capacity
# factor of 2 never drops a token.
_CHUNK_MOE = dict(
    num_moe_experts=4,
    moe_router_topk=2,
    moe_token_dispatcher_type="alltoall",
    moe_expert_capacity_factor=2.0,
    moe_pad_expert_input_to_capacity=True,
    moe_router_load_balancing_type="aux_loss",
    moe_aux_loss_coeff=0.1,
    moe_z_loss_coeff=0.1,
)


@pytest.mark.internal
class TestChunkStaticInputs:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_decoder_static_inputs(self):
        block = _build_chunk_gpt_model(256, 4, 4, 1024, 128, 8).decoder

        static_inputs = block.get_layer_static_inputs(seq_length=128, micro_batch_size=1)
        assert static_inputs["hidden_states"].shape == (128, 1, 256)
        assert static_inputs["cu_seqlens_q"].shape == (9,)
        assert static_inputs["cu_seqlens_kv_padded"].shape == (9,)
        assert static_inputs["padding_mask"].shape == (1, 128)
        assert not static_inputs["padding_mask"].any()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_postprocess_block_is_attached_and_excluded_from_state_dict(self):
        model = _build_chunk_gpt_model(256, 4, 4, 1024, 128, 8)
        block = model.postprocess_block
        assert block is not None and block.post_process and not block.pre_process
        assert block.output_layer is model.output_layer
        assert not any(key.startswith('postprocess_block.') for key in model.state_dict())
        assert not any(key.startswith('postprocess_block.') for key in model.sharded_state_dict())
        # Loading the model's own state dict must not report the shared block parameters missing.
        model.load_state_dict(model.state_dict(), strict=True)
        static_inputs = block.get_layer_static_inputs(seq_length=128, micro_batch_size=1)
        assert static_inputs["labels"].shape == (1, 128)
        assert static_inputs["hidden_states"].shape == (128, 1, 256)
        # Without MTP the post-process consumes neither the tokens nor the padding mask, and a
        # last stage without MTP does not even receive tokens / position ids (a captured keyword
        # that is None at replay would make the graphed callable raise).
        assert "input_ids" not in static_inputs
        assert "position_ids" not in static_inputs
        assert "padding_mask" not in static_inputs

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("context_parallel_size", [1, 2])
    def test_fixed_length_decoder_attention_mask_follows_the_dataloader_layout(
        self, context_parallel_size
    ):
        """Fixed-length batches with ``create_attention_mask_in_dataloader`` carry a
        ``[batch, 1, sequence / cp, sequence]`` mask (True: masked): this rank's query rows
        against every key. The block captures a causal sample of exactly that layout."""
        micro_batch_size, seq_length, hidden = 2, 32, 64
        block = _build_chunk_gpt_model(
            hidden, 4, 4, 128, seq_length, 8, packed=False, create_attention_mask_in_dataloader=True
        ).decoder
        block.config.context_parallel_size = context_parallel_size
        local_seq_length = seq_length // context_parallel_size

        static_inputs = block.get_layer_static_inputs(seq_length, micro_batch_size)

        assert static_inputs["hidden_states"].shape == (local_seq_length, micro_batch_size, hidden)
        attention_mask = static_inputs["attention_mask"]
        assert attention_mask.dtype == torch.bool
        assert attention_mask.shape == (micro_batch_size, 1, local_seq_length, seq_length)
        causal = ~torch.tril(torch.ones(local_seq_length, seq_length, dtype=torch.bool)).cuda()
        assert torch.equal(attention_mask, causal.expand(micro_batch_size, 1, -1, -1))
        # No packed metadata, and no mask input unless a layer received one in eager warm-up.
        assert "cu_seqlens_q" not in static_inputs
        assert "padding_mask" not in static_inputs

        # A run whose batches carry no attention mask captures none: the replay could not
        # refresh it.
        block.config.create_attention_mask_in_dataloader = False
        assert "attention_mask" not in block.get_layer_static_inputs(seq_length, micro_batch_size)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("packed", [True, False])
    def test_mhc_decoder_takes_the_residual_streams_on_later_stages(self, packed):
        """Hyper connections hand the ``num_residual_streams`` expanded streams from one decoder
        chunk to the next; only the chunk that owns the embedding receives the model width."""
        seq_length, hidden, streams = 32, 64, 4
        micro_batch_size = 1 if packed else 2
        for pre_process, width in ((True, hidden), (False, hidden * streams)):
            block = _build_chunk_gpt_model(
                hidden,
                4,
                4,
                128,
                seq_length,
                8,
                packed=packed,
                pre_process=pre_process,
                enable_hyper_connections=True,
                num_residual_streams=streams,
                mhc_sinkhorn_iterations=5,
            ).decoder
            assert block.pre_process is pre_process

            static_inputs = block.get_layer_static_inputs(seq_length, micro_batch_size)

            hidden_states = static_inputs["hidden_states"]
            assert hidden_states.shape == (seq_length, micro_batch_size, width), pre_process
            assert hidden_states.dtype == torch.bfloat16 and hidden_states.requires_grad
            assert hidden_states.is_cuda

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("packed, micro_batch_size", [(True, 1), (False, 1), (False, 2)])
    def test_hash_routed_decoder_takes_token_ids_in_its_own_token_frame(
        self, packed, micro_batch_size
    ):
        """The block-scope row of ``test_hash_router_input_ids_follow_static_token_layout``, on a
        real block whose first layer routes by token id. With context parallelism the block's
        hidden states hold this rank's tokens only, and the hash router asserts one id per row
        of its logits: ``[1, tokens]`` for a pack, ``[batch, tokens / cp]`` for fixed length."""
        seq_length, vocab = 64, 128
        block = _build_chunk_gpt_model(
            64,
            4,
            4,
            128,
            seq_length,
            8,
            packed=packed,
            vocab=vocab,
            moe_num_hash_layers=1,
            hash_moe_vocab_size=vocab,
            **_CHUNK_MOE,
        ).decoder
        assert block.layers[0].mlp.router.is_hash_layer
        assert not block.layers[1].mlp.router.is_hash_layer
        # Two context-parallel ranks, each holding seq_length tokens of every sample.
        block.config.context_parallel_size = 2

        static_inputs = block.get_layer_static_inputs(seq_length * 2, micro_batch_size)

        assert static_inputs["hidden_states"].shape[:2] == (seq_length, micro_batch_size)
        assert static_inputs["input_ids"].shape == (micro_batch_size, seq_length)
        assert static_inputs["input_ids"].dtype == torch.long

        # Without a hash-routed layer in the block no ids are captured (the model hands the
        # decoder none either).
        block.layers[0].mlp.router.is_hash_layer = False
        assert "input_ids" not in block.get_layer_static_inputs(seq_length * 2, micro_batch_size)


@pytest.mark.internal
class TestChunkTokenMetadata:
    """``padding_mask`` and the hash router's ``input_ids`` of a chunk callable.

    A graph input has one fixed shape and the graph one fixed keyword set, while the models
    differ in what they hand the block: ``GPTModel`` scatters the padding mask to the
    sequence-parallel shard on the first stage only and never the ids, ``HybridModel`` scatters
    both. The block captures both in its own token frame (that of its ``hidden_states``) and
    replay brings the runtime metadata to it, as the layers do in eager mode."""

    @staticmethod
    def _block(hidden_shape=(4, 1, 8), *, thd=True, moe=False, **config):
        from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphBlockMixin
        from megatron.core.transformer.moe.moe_layer import MoELayer

        class Base(torch.nn.Module):
            def get_layer_static_inputs(self, seq_length, micro_batch_size):
                return {"hidden_states": torch.ones(hidden_shape, requires_grad=True)}

            def _is_thd_cuda_graph(self):
                return thd

        class Block(ChunkCudaGraphBlockMixin, Base):
            pass

        block = Block()
        block.pre_process = True
        block.config = SimpleNamespace(
            **{
                "sequence_parallel": True,
                "enable_hyper_connections": False,
                "max_seqlen_per_dp_cp_rank": 8,
                "context_parallel_size": 1,
                "thd_max_packed_sequences": 4,
                "create_attention_mask_in_dataloader": False,
                "moe_n_hash_layers": 0,
                **config,
            }
        )
        # Rank 1 of a TP group of two owns the second half of the CP-local tokens.
        block.tp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
        if moe:
            block.mlp = MoELayer.__new__(MoELayer)
            torch.nn.Module.__init__(block.mlp)
        return block

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("pre_process", [True, False])
    def test_thd_samples_use_the_block_token_frame_on_every_stage(self, pre_process):
        """8 CP-local tokens, TP 2 with sequence parallelism: the block sees 4 tokens."""
        block = self._block(moe_n_hash_layers=1)
        block.pre_process = pre_process
        block.router = torch.nn.Identity()
        block.router.is_hash_layer = True

        static_inputs = block.get_layer_static_inputs(seq_length=8, micro_batch_size=1)

        assert static_inputs["padding_mask"].shape == (1, 4)
        assert static_inputs["padding_mask"].dtype == torch.bool
        assert not static_inputs["padding_mask"].any()
        assert static_inputs["input_ids"].shape == (1, 4)
        assert static_inputs["input_ids"].dtype == torch.long
        assert static_inputs["cu_seqlens_q"].shape == (5,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_length_hash_ids_follow_context_parallelism(self):
        """Fixed-length batch of 2 x 16 tokens with CP 2: the block's hidden states are
        [8, 2, H] and the hash router hashes exactly those 8 tokens per sample (the per-layer
        capture's contract, test_hash_router_input_ids_follow_static_token_layout)."""
        block = self._block(
            (8, 2, 8),
            thd=False,
            sequence_parallel=False,
            context_parallel_size=2,
            moe_n_hash_layers=1,
        )
        block.router = torch.nn.Identity()
        block.router.is_hash_layer = True

        static_inputs = block.get_layer_static_inputs(seq_length=16, micro_batch_size=2)

        assert static_inputs["input_ids"].shape == (2, 8)
        # No layer of the block received a padding mask in the eager warm-up steps.
        assert "padding_mask" not in static_inputs
        assert block._chunk_graph_captures_padding_mask is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_length_capture_reserves_the_padding_mask_a_layer_received(self):
        """Like the per-layer capture: a layer that saw a mask during the eager warm-up steps
        makes the block graph take one, in the block's token frame."""
        block = self._block((8, 2, 8), thd=False, sequence_parallel=False)
        block.layer = torch.nn.Identity()
        block.layer._padding_mask_shape_seen = (2, 8)

        static_inputs = block.get_layer_static_inputs(seq_length=8, micro_batch_size=2)

        assert static_inputs["padding_mask"].shape == (2, 8)
        assert not static_inputs["padding_mask"].any()
        assert block._chunk_graph_captures_padding_mask is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("cuda_graph_impl", ["chunk_wise", "transformer_engine"])
    def test_fixed_length_capture_reserves_the_padding_mask_the_block_received(
        self, cuda_graph_impl
    ):
        """The hyper-connection layers consume the mask without recording it per layer, so a
        chunk-wise block also notes a mask where the model hands it over."""
        block = self._block(
            (8, 2, 8), thd=False, sequence_parallel=False, cuda_graph_impl=cuda_graph_impl
        )
        block.forward = lambda hidden_states, padding_mask=None: hidden_states

        block(torch.zeros(8, 2, 8))
        assert "padding_mask" not in block.get_layer_static_inputs(seq_length=8, micro_batch_size=2)

        block(torch.zeros(8, 2, 8), padding_mask=torch.zeros(2, 8, dtype=torch.bool))
        static_inputs = block.get_layer_static_inputs(seq_length=8, micro_batch_size=2)

        if cuda_graph_impl == "chunk_wise":
            assert static_inputs["padding_mask"].shape == (2, 8)
            assert block._chunk_graph_captures_padding_mask is True
        else:
            assert "padding_mask" not in static_inputs

    def test_replay_shards_full_length_metadata_like_the_layers_do(self):
        block = self._block()
        block._chunk_graph_captures_padding_mask = True
        hidden_states = torch.zeros(4, 1, 8)
        kwargs = {
            "padding_mask": torch.tensor([[False] * 6 + [True] * 2]),
            "input_ids": torch.arange(8).unsqueeze(0),
        }

        block._align_token_metadata_to_block(hidden_states, kwargs)

        assert kwargs["padding_mask"].tolist() == [[False, False, True, True]]
        assert kwargs["input_ids"].tolist() == [[4, 5, 6, 7]]

    @pytest.mark.parametrize("sequence_parallel", [True, False])
    def test_replay_keeps_metadata_that_is_already_in_the_block_frame(self, sequence_parallel):
        block = self._block(sequence_parallel=sequence_parallel)
        block._chunk_graph_captures_padding_mask = True
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)
        input_ids = torch.arange(4).unsqueeze(0)
        kwargs = {"padding_mask": padding_mask, "input_ids": input_ids}

        block._align_token_metadata_to_block(torch.zeros(4, 1, 8), kwargs)

        assert kwargs["padding_mask"] is padding_mask
        assert kwargs["input_ids"] is input_ids

    def test_replay_without_a_mask_feeds_an_all_false_one_to_a_graph_that_takes_a_mask(self):
        block = self._block()
        block._chunk_graph_captures_padding_mask = True
        kwargs = {}

        block._align_token_metadata_to_block(torch.zeros(4, 2, 8), kwargs)

        assert kwargs["padding_mask"].shape == (2, 4)
        assert kwargs["padding_mask"].dtype == torch.bool
        assert not kwargs["padding_mask"].any()

    def test_mask_arriving_at_a_graph_captured_without_one(self):
        """The graphed callable would drop the keyword silently. Nothing reads it in a block
        without MoE routers; with them the replay would diverge from eager, so it raises."""
        hidden_states = torch.zeros(4, 1, 8)
        mask = torch.zeros(1, 4, dtype=torch.bool)

        block = self._block()
        block._chunk_graph_captures_padding_mask = False
        kwargs = {"padding_mask": mask}
        block._align_token_metadata_to_block(hidden_states, kwargs)
        assert "padding_mask" not in kwargs

        block = self._block(moe=True)
        block._chunk_graph_captures_padding_mask = False
        with pytest.raises(RuntimeError, match="captured without a padding_mask input"):
            block._align_token_metadata_to_block(hidden_states, {"padding_mask": mask})
        # No mask, no complaint.
        kwargs = {}
        block._align_token_metadata_to_block(hidden_states, kwargs)
        assert kwargs == {}

    def test_postprocess_block_keeps_the_callers_full_length_metadata(self):
        """The post-process captures the full-length tokens and mask (the MTP rolls them)."""
        from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphPostProcessBlock

        block = object.__new__(ChunkCudaGraphPostProcessBlock)
        object.__setattr__(block, "config", SimpleNamespace(sequence_parallel=True))
        padding_mask = torch.zeros(1, 8, dtype=torch.bool)
        input_ids = torch.arange(8).unsqueeze(0)
        kwargs = {"padding_mask": padding_mask, "input_ids": input_ids}

        block._align_token_metadata_to_block(torch.zeros(4, 1, 8), kwargs)

        assert kwargs["padding_mask"] is padding_mask
        assert kwargs["input_ids"] is input_ids

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_length_graph_refuses_packed_seq_params(self):
        """A graph captured from fixed-length samples has no cu_seqlens inputs; the graphed
        callable would drop them and replay causal attention across the sequence boundaries."""
        block = self._block(thd=False)
        with pytest.raises(RuntimeError, match="captured for fixed-length batches"):
            block._decompose_packed_seq_params_to_kwargs({"packed_seq_params": _make_psp([4, 4])})
        # Nothing to refuse without packed metadata.
        kwargs = {"packed_seq_params": None}
        block._decompose_packed_seq_params_to_kwargs(kwargs)
        assert kwargs == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_microbatch_scheduled_on_another_cp_group_is_refused(self):
        """The context-parallel collectives of a captured block are those of the static CP
        group; a microbatch that dynamic context parallelism scheduled on a group of another
        size would be replayed on the wrong ranks."""
        block = self._block(context_parallel_size=2)
        packed_seq_params = _make_psp([4, 4])

        packed_seq_params.local_cp_size = 1
        with pytest.raises(RuntimeError, match="context-parallel group of size 1"):
            block._decompose_packed_seq_params_to_kwargs({"packed_seq_params": packed_seq_params})

        # The static size, spelled out or left to the default, replays.
        for local_cp_size in (2, None):
            packed_seq_params.local_cp_size = local_cp_size
            kwargs = {"packed_seq_params": packed_seq_params}
            block._decompose_packed_seq_params_to_kwargs(kwargs)
            assert kwargs["cu_seqlens_q"] is packed_seq_params.cu_seqlens_q
            assert "packed_seq_params" not in kwargs

    def test_packed_chunkwise_cp_linear_layers_are_refused(self):
        """Their CP metadata is derived on the host from each pack's sequence boundaries. The
        stack only runs them chunkwise with more than one context-parallel rank: its flag names
        the layers that would, whatever the size of the CP group."""
        block = self._block(context_parallel_size=2)
        block._has_linear_layer_with_chunkwise_cp = True
        with pytest.raises(ValueError, match="chunkwise context parallelism"):
            block.get_layer_static_inputs(seq_length=8, micro_batch_size=1)

        block = self._block(context_parallel_size=1)
        block._has_linear_layer_with_chunkwise_cp = True
        assert "cu_seqlens_q" in block.get_layer_static_inputs(seq_length=8, micro_batch_size=1)


def _thd_batch(seqlens, vocab, seed, capacity, max_num_seqs):
    """One packed micro-batch padded to the static THD capacity exactly like the data pipeline
    (``--pad-packed-seq-alignment max --thd-max-packed-sequences N``: tokens to ``capacity`` with a
    dummy tail sequence, cu_seqlens to ``N + 1`` entries); identical on every rank (CPU seed)."""
    total = sum(seqlens)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, vocab, (1, total + 1), generator=generator).cuda()
    position_ids = torch.cat([torch.arange(length) for length in seqlens]).unsqueeze(0).cuda()
    alignment, target_len, padded_num_seqs = get_thd_padding_kwargs(
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=capacity,
        thd_max_packed_sequences=max_num_seqs,
        cuda_graph_static=True,
    )
    input_ids, labels, loss_mask, position_ids, packed_seq_params, padding_mask = (
        pad_sequence_for_thd(
            tokens[:, :-1].contiguous(),
            tokens[:, 1:].contiguous(),
            torch.ones(1, total, dtype=torch.float32, device="cuda"),
            position_ids,
            _make_psp(seqlens),
            alignment=alignment,
            target_len=target_len,
            max_num_seqs=padded_num_seqs,
            # The pack is padded before any context-parallel slicing (the callers slice it
            # themselves where they need a rank's share), so ``capacity`` is its global length
            # whatever the context-parallel size of the test.
            cp_size=1,
            cp_rank=0,
        )
    )
    assert input_ids.shape == (1, capacity)
    assert int(packed_seq_params.cu_seqlens_q_padded[-1]) == capacity
    assert packed_seq_params.cu_seqlens_q.numel() == max_num_seqs + 1
    return dict(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
    )


def _train_step(model, batch, **forward_kwargs):
    """Forward + backward without an optimizer step; returns (per-token loss, {name: grad}).
    ``forward_kwargs`` are further keyword arguments of the model call."""
    for param in model.parameters():
        # Caller-owned leaf-grad buffers: with a leaf grad of None, AccumulateGrad may adopt TE's
        # recyclable static grad buffer instead of copying out of it.
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        else:
            param.grad.zero_()
    per_token_loss = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        padding_mask=batch["padding_mask"],
        packed_seq_params=batch["packed_seq_params"],
        **forward_kwargs,
    )
    # Under graphs the model returns a view of TE's recyclable static output, owned only until
    # its backward: retain the values before crossing that boundary.
    loss_values = per_token_loss.detach().float().clone()
    loss = (per_token_loss.float() * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
    loss.backward()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss_values, grads


def _train_step_with_mtp_loss(model, batch, **forward_kwargs):
    """``_train_step`` plus the MTP loss of the step, which never reaches the model output (it
    is accumulated for logging); ``None`` when no model with MTP ran in this test."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    MTPLossLoggingHelper.clean_loss_in_tracker()
    per_token_loss, grads = _train_step(model, batch, **forward_kwargs)
    torch.cuda.synchronize()
    mtp_loss = MTPLossLoggingHelper.tracker.get("loss_sums")
    return per_token_loss, grads, None if mtp_loss is None else mtp_loss.clone()


def _assert_matches_eager(result, reference, label):
    """One training step (``_train_step_with_mtp_loss``) against the eager twin's step on the
    same microbatch: per-token loss, MTP loss and every parameter gradient."""
    per_token_loss, grads, mtp_loss = result
    ref_per_token_loss, ref_grads, ref_mtp_loss = reference
    torch.testing.assert_close(per_token_loss, ref_per_token_loss, rtol=1e-3, atol=1e-3)
    assert (mtp_loss is None) == (ref_mtp_loss is None), label
    if mtp_loss is not None:
        torch.testing.assert_close(mtp_loss, ref_mtp_loss, rtol=1e-3, atol=1e-3)
    assert grads.keys() == ref_grads.keys(), label
    for name, ref_grad in ref_grads.items():
        error = (grads[name] - ref_grad).norm()
        assert error <= 2e-2 * ref_grad.norm() + 1e-6, (
            label,
            name,
            error.item(),
            ref_grad.norm().item(),
        )


def _grad_distance(grads, ref_grads, prefix=""):
    """Relative L2 distance of two ``{name: grad}`` dicts over the parameters under ``prefix``."""
    names = [name for name in ref_grads if name.startswith(prefix)]
    assert names, prefix
    error = sum((grads[name] - ref_grads[name]).square().sum() for name in names)
    norm = sum(ref_grads[name].square().sum() for name in names)
    return (error / norm).sqrt().item()


def _pretrain_hybrid_batch(model, batch):
    """One packed microbatch (``_thd_batch``) as ``pretrain_hybrid.py`` hands it to the model:
    ``get_batch`` returns the packing scheduler's ``ContextParallelBatch`` (built by
    ``get_batches_on_this_cp_rank``, metadata finalized by ``prepare_packed_seq_params``), and
    ``forward_step`` passes the tensors and ``PackedSeqParams`` of its boundary layout next to
    the ``ContextParallelBatch`` itself (``_pretrain_hybrid_step``)."""
    from megatron.core.context_parallel import ContextParallelBatch, get_batches_on_this_cp_rank
    from megatron.training.utils import prepare_packed_seq_params

    config, pg_collection = model.config, model.pg_collection
    packed_seq_params = batch["packed_seq_params"]
    scheduler_batch = {
        "tokens": batch["input_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
        "position_ids": batch["position_ids"],
        "padding_mask": batch["padding_mask"],
        "cu_seqlens": packed_seq_params.cu_seqlens_q.unsqueeze(0),
        "cu_seqlens_padded": packed_seq_params.cu_seqlens_q_padded.unsqueeze(0),
        "max_seqlen": torch.tensor(packed_seq_params.max_seqlen_q, dtype=torch.int32).cuda(),
        "local_cp_size": None,
        "dynamic_cp_group": None,
    }
    additional_layouts = set()
    if config.linear_cp_layout != config.attention_cp_layout:
        additional_layouts.add(config.attention_cp_layout)
    cp_batch = get_batches_on_this_cp_rank(
        scheduler_batch,
        boundary_layout=config.linear_cp_layout,
        dynamic_cp=False,
        cp_group=pg_collection.cp,
        additional_layouts=additional_layouts,
        sequence_parallel=config.sequence_parallel,
        tp_group=pg_collection.tp,
        tp_cp_group=None,
        tokens_per_sample=None,
    )
    prepare_packed_seq_params(cp_batch.get_packed_seq_params(), config, cp_group=pg_collection.cp)
    assert isinstance(cp_batch, ContextParallelBatch) and cp_batch.thd_plan is None

    view = cp_batch.get_batch()
    return dict(
        input_ids=view["tokens"],
        labels=view["labels"],
        position_ids=view["position_ids"],
        loss_mask=view["loss_mask"],
        padding_mask=view["padding_mask"],
        packed_seq_params=cp_batch.get_packed_seq_params(),
        cp_batch=cp_batch,
    )


def _pretrain_hybrid_step(model, batch, **forward_kwargs):
    """``_train_step_with_mtp_loss`` with the model call of ``pretrain_hybrid.py``'s
    ``forward_step`` (``batch`` from ``_pretrain_hybrid_batch``)."""
    return _train_step_with_mtp_loss(model, batch, cp_batch=batch["cp_batch"], **forward_kwargs)


def _fixed_length_batch(micro_batch_size, seq_length, vocab, seed, padding_mask=None):
    """One fixed-length (SBHD) micro-batch with the dataloader's causal attention mask (True:
    masked), optionally with a batch-first padding mask (True: padding) whose tokens do not
    count in the loss."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, vocab, (micro_batch_size, seq_length + 1), generator=generator)
    loss_mask = torch.ones(micro_batch_size, seq_length, dtype=torch.float32)
    if padding_mask is not None:
        loss_mask = loss_mask * (~padding_mask.cpu()).to(loss_mask.dtype)
    causal = ~torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool))
    return dict(
        input_ids=tokens[:, :-1].contiguous().cuda(),
        labels=tokens[:, 1:].contiguous().cuda(),
        position_ids=torch.arange(seq_length).repeat(micro_batch_size, 1).cuda(),
        loss_mask=loss_mask.cuda(),
        attention_mask=causal.tile(micro_batch_size, 1, 1, 1).cuda(),
        padding_mask=padding_mask,
        packed_seq_params=None,
    )


def _record_graph_replays(monkeypatch):
    """Record every CUDA graph replay from here on; returns the list the graphs go to."""
    replays = []
    original_replay = torch.cuda.CUDAGraph.replay

    def recording_replay(graph, *args, **kwargs):
        replays.append(graph)
        return original_replay(graph, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", recording_replay)
    return replays


class _ChunkCaptureEnv:
    """The process-wide state a chunk-graph capture test sets up (the ``chunk_capture``
    fixture puts it back)."""

    def __init__(self):
        self.cleanups = []
        self.stream = None

    def setup(self, tensor_model_parallel_size=1, micro_batch_size=1, context_parallel_size=1):
        """Model-parallel state, the micro-batch count the capture sizes its Transformer Engine
        warm-up from (two microbatches per step) and the Transformer Engine RNG tracker."""
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
            init_num_microbatches_calculator,
        )
        from megatron.core.tensor_parallel.random import initialize_rng_tracker

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )
        if self.stream is None:
            # Read after the rank selected its device: setting a stream makes its device the
            # current one, so a stream saved before that would move the rank back to device 0.
            self.stream = torch.cuda.current_stream()
        # A failed earlier test can leave the calculator initialized.
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=2 * micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )
        # The conftest fixture puts the previous tracker back after the test.
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def capture(self, model, seq_length, micro_batch_size=1):
        """Capture the chunk graphs of ``model``; returns the capture helper."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        helper = TECudaGraphHelper(
            model=[model],
            config=model.config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )
        # Transformer Engine wraps the __call__ of the graphed classes while it captures and
        # unwraps them only when the capture returns.
        class_calls = {
            type(block): vars(type(block)).get("__call__") for block in helper.flattened_callables
        }

        def release():
            if helper.graphs_created():
                helper.delete_cuda_graphs()
            for cls, call in class_calls.items():
                if call is not None:
                    cls.__call__ = call
                elif "__call__" in vars(cls):
                    del cls.__call__

        self.cleanups.append(release)
        if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
            # The eager steps before a capture leave autograd nodes that refer to their stream,
            # which a capture on another stream has to redirect. The redirection is a
            # process-wide switch that every capture path turns on and leaves on: back to
            # PyTorch's default, so that this capture turns it on itself (the fixture puts the
            # previous setting back).
            torch.autograd.graph.set_override_stale_capture_stream(False)
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert len(model.decoder.cuda_graphs) > 0
        return helper

    def capture_and_replay_against_eager(
        self, monkeypatch, model, twin, batches, seq_length, **options
    ):
        """The life of a chunk-graph run on ``batches``, every step compared with the eager
        twin's step on the same microbatch (no optimizer step anywhere, the weights stay
        equal): eager steps of the chunk-configured model (the ``cuda_graph_warmup_steps``,
        which already take the chunk-graph forward path), the capture, then two rounds of
        replays in different orders, so every replay must reflect the microbatch it was given.

        Options: ``step`` (``_train_step_with_mtp_loss``), ``micro_batch_size`` (1) and
        ``replays_per_step``, the graph replays a post-capture step launches at least (4:
        forward and backward of the decoder and of the post-process). Returns the helper."""
        step = options.pop("step", _train_step_with_mtp_loss)
        micro_batch_size = options.pop("micro_batch_size", 1)
        replays_per_step = options.pop("replays_per_step", 4)
        assert not options, options

        reference = [step(twin, batch) for batch in batches]
        for first, second in zip(reference, reference[1:]):
            assert not torch.allclose(first[0], second[0]), "batches must differ"
        for index, batch in enumerate(batches):
            _assert_matches_eager(step(model, batch), reference[index], ("eager", index))

        helper = self.capture(model, seq_length, micro_batch_size)

        replays = _record_graph_replays(monkeypatch)
        indices = list(range(len(batches)))
        for order in (indices, indices[::-1]):
            for index in order:
                before = len(replays)
                result = step(model, batches[index])
                assert len(replays) - before >= replays_per_step, (index, len(replays) - before)
                _assert_matches_eager(result, reference[index], ("replay", index))
        return helper


@pytest.fixture
def chunk_capture():
    """A ``_ChunkCaptureEnv`` whose process-wide state is put back even when the test fails:
    the captured graphs (and with them the shared graph pool), the ``__call__`` of the graphed
    classes, the capture flags, autograd's redirection of stale stream references, the current
    CUDA stream, the micro-batch calculator, the MoE metrics tracker and the model-parallel
    state."""
    from contextlib import ExitStack

    from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.moe.moe_logging import (
        MoEMetricsTracker,
        get_moe_metrics_tracker,
        set_moe_metrics_tracker,
    )

    # The process-wide MoE metrics tracker keeps its entries, sized by the layer count of the
    # first model that recorded into it: the test records into a tracker of its own.
    moe_metrics_tracker = get_moe_metrics_tracker()
    set_moe_metrics_tracker(MoEMetricsTracker())
    # A capture turns the redirection on for the rest of the process; PyTorch keeps the getter
    # of the switch private.
    stale_stream_override = getattr(torch._C, "_override_stale_capture_stream", None)
    redirects_stale_streams = None if stale_stream_override is None else stale_stream_override()
    env = _ChunkCaptureEnv()
    try:
        yield env
    finally:
        # Callbacks run last-registered first, each even when an earlier one raised.
        with ExitStack() as stack:
            stack.callback(Utils.destroy_model_parallel)
            stack.callback(set_moe_metrics_tracker, moe_metrics_tracker)
            stack.callback(destroy_num_microbatches_calculator)
            if env.stream is not None:
                stack.callback(torch.cuda.set_stream, env.stream)
            if redirects_stale_streams is not None:
                stack.callback(
                    torch.autograd.graph.set_override_stale_capture_stream, redirects_stale_streams
                )
            # Only still set if a capture raised.
            stack.callback(cuda_graphs._set_capture_end)
            if cuda_graphs.HAVE_TE_GRAPHS:
                stack.callback(cuda_graphs.te_set_capture_end)
            for cleanup in env.cleanups:
                stack.callback(cleanup)
            torch.cuda.synchronize()


# Packings under the static capacity of 128 tokens and the bound of 4 sequences (the padding
# tail becomes a dummy sequence, so at most 3 real ones).
_THD_PACKINGS = ([120], [60, 60], [32, 88], [16, 40, 60])
# The same with no document longer than a 64-entry position table.
_THD_SHORT_DOCUMENT_PACKINGS = ([64, 56], [60, 60], [32, 64], [16, 40, 60])
# Packs that fill the capacity exactly: no padding tail, hence no dummy sequence and no masked
# token (one sample of the full length; two documents that add up to it).
_THD_FULL_PACKINGS = ([128], [40, 88])


@pytest.mark.internal
class TestChunkGraphRegression:
    """Real chunk-graph capture and replay on small models, every step checked against the
    eager twin (``cuda_graph_impl="none"``, same weights) on the same microbatch.

    Packed sequences (THD), where the post-process (MTP, LM head, loss) is a second captured
    callable: tensor parallelism with sequence parallelism, with and without MTP; activation
    recompute recorded inside the graphs (full, selective ``core_attn`` and ``mlp``); hyper
    connections; the Transformer Engine fused cross entropy; a learned position table shorter
    than a pack; a ``HybridModel`` called the way ``pretrain_hybrid.py`` calls it; the calls
    and configurations that keep the post-process eager after the decoder replay. Fixed-length
    batches: the padding mask and hash-router token ids of a MoE block as graph inputs.
    Numerical comparisons: no dropout, no MoE routing randomness.
    """

    @_REQUIRES_TWO_RANKS
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("mtp_layers", [1, 0])
    def test_thd_tp2_sp_mtp_capture_replay_matches_eager(
        self, chunk_capture, monkeypatch, mtp_layers
    ):
        chunk_capture.setup(tensor_model_parallel_size=2)
        tokens, max_num_seqs, vocab = 128, 4, 128
        model, twin = _build_chunk_gpt_model_and_eager_twin(
            256, 4, 4, 512, tokens, max_num_seqs, tp=2, sp=True, mtp_layers=mtp_layers, vocab=vocab
        )
        # LM head + loss (+ MTP when present) are captured as the post-process callable; the
        # eager twin has no such block.
        assert model.postprocess_block is not None and twin.postprocess_block is None
        assert model.mtp_process == bool(mtp_layers)
        batches = [
            _thd_batch(seqlens, vocab, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(_THD_PACKINGS + _THD_FULL_PACKINGS)
        ]
        # The full packs carry no padded token; every other pack does.
        padded = [bool(batch["loss_mask"].eq(0).any()) for batch in batches]
        assert padded == [True] * len(_THD_PACKINGS) + [False] * len(_THD_FULL_PACKINGS)

        helper = chunk_capture.capture_and_replay_against_eager(
            monkeypatch, model, twin, batches, tokens
        )

        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        assert len(model.postprocess_block.cuda_graphs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "builder_kwargs, recomputed_module",
        [
            pytest.param(
                dict(
                    recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
                ),
                "decoder.layers.0",
                id="full_recompute",
            ),
            pytest.param(
                dict(recompute_granularity="selective", recompute_modules=["core_attn"]),
                "decoder.layers.0.self_attention.core_attention",
                id="core_attn_recompute",
            ),
            pytest.param(
                # Without bias, as in the recipes that recompute the MLP: with one, the MLP hands
                # the checkpoint its bias parameter as an output, which the eager checkpoint of
                # the twin cannot return (its backward re-enters itself).
                dict(
                    recompute_granularity="selective",
                    recompute_modules=["mlp"],
                    add_bias_linear=False,
                ),
                "decoder.layers.0.mlp",
                id="mlp_recompute",
            ),
            pytest.param(
                dict(
                    enable_hyper_connections=True, num_residual_streams=4, mhc_sinkhorn_iterations=5
                ),
                None,
                id="hyper_connections",
            ),
            pytest.param(
                dict(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te"),
                None,
                id="te_fused_cross_entropy",
            ),
            pytest.param(
                dict(position_embedding_type="learned_absolute", max_sequence_length=64),
                None,
                id="short_learned_position_table",
            ),
        ],
    )
    def test_thd_capture_replay_matches_eager(
        self, chunk_capture, monkeypatch, builder_kwargs, recomputed_module
    ):
        """Packed sequences with one MTP layer, one configuration per row:

        * activation recompute: the checkpoint is recorded inside the graphs, so every pass of
          the capture runs the checkpointed module twice (in the forward, and again when the
          backward recomputes it), like every eager step;
        * hyper connections: the residual streams stay inside the decoder graph and the
          pre-contraction streams are a second input of the post-process graph (for the MTP);
        * Transformer Engine fused cross entropy: the loss is part of the post-process graph,
          so its backward has to be the capturable variant;
        * a learned position table shorter than a pack: the post-process sample feeds the MTP
          embedding position ids, which must be valid rows of that table.
        """
        chunk_capture.setup()
        tokens, max_num_seqs, vocab = 128, 4, 128
        model, twin = _build_chunk_gpt_model_and_eager_twin(
            256, 4, 4, 512, tokens, max_num_seqs, mtp_layers=1, vocab=vocab, **builder_kwargs
        )
        assert model.postprocess_block is not None and model.mtp_process
        if builder_kwargs.get("cross_entropy_fusion_impl") == "te":
            # The loss is recorded into the post-process graph, so the model holds the variant
            # whose backward is capturable (it creates no host constant on the device to compare
            # the incoming gradient with); the eager twin holds the default one.
            assert model.vocab_parallel_cross_entropy.keywords == {"cuda_graph_capturable": True}
            assert twin.vocab_parallel_cross_entropy.keywords == {"cuda_graph_capturable": False}
        packings = (
            _THD_SHORT_DOCUMENT_PACKINGS
            if "max_sequence_length" in builder_kwargs
            else _THD_PACKINGS
        )
        batches = [
            _thd_batch(seqlens, vocab, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(packings)
        ]

        # Forward calls of the recomputed module and of a module no checkpoint covers.
        calls = {}
        if recomputed_module is not None:
            for name, owner in (("model", model), ("twin", twin)):
                for kind, path in (
                    ("recomputed", recomputed_module),
                    ("plain", "decoder.final_layernorm"),
                ):
                    counter = calls[name, kind] = []
                    handle = owner.get_submodule(path).register_forward_hook(
                        lambda *_, counter=counter: counter.append(None)
                    )
                    chunk_capture.cleanups.append(handle.remove)

        helper = chunk_capture.capture_and_replay_against_eager(
            monkeypatch, model, twin, batches, tokens
        )

        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        assert len(model.postprocess_block.cuda_graphs) > 0
        if recomputed_module is not None:
            # The eager reference recomputes: once in the forward, once in the backward.
            assert len(calls["twin", "plain"]) == len(batches)
            assert len(calls["twin", "recomputed"]) == 2 * len(batches)
            # So do the chunk-configured model's eager steps and every pass of the capture
            # (Transformer Engine's warm-up passes and the recorded forward + backward); the
            # replays run no Python.
            assert len(calls["model", "plain"]) > len(batches)
            assert len(calls["model", "recomputed"]) == 2 * len(calls["model", "plain"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tensor_model_parallel_size, context_parallel_size",
        [
            (1, 1),
            pytest.param(2, 1, marks=_REQUIRES_TWO_RANKS),
            pytest.param(1, 2, marks=_REQUIRES_TWO_RANKS),
        ],
        ids=["tp1", "tp2", "cp2"],
    )
    def test_hybrid_capture_replay_matches_eager_called_like_pretrain_hybrid(
        self, chunk_capture, monkeypatch, tensor_model_parallel_size, context_parallel_size
    ):
        """``pretrain_hybrid.py`` passes every microbatch's ``ContextParallelBatch``, so the
        ``HybridStack`` receives the per-layout dicts (``packed_seq_params_by_layout``,
        ``padding_mask_by_layout``) and the layout plan next to the tensors on every call,
        including the replays, which accept tensors only. With sequence parallelism the model
        hands the stack the padding mask already scattered to its token frame. With context
        parallelism every rank captures its share of the window (the zigzag slice of a pack of
        ``context_parallel_size`` windows), and the attention layers exchange tokens over the
        CP group inside the block graph."""
        tp, cp = tensor_model_parallel_size, context_parallel_size
        chunk_capture.setup(tensor_model_parallel_size=tp, context_parallel_size=cp)
        tokens, max_num_seqs, vocab = 128, 4, 128
        model, twin = _build_chunk_hybrid_model_and_eager_twin(
            tokens, max_num_seqs, tp=tp, sp=tp > 1, cp=cp, vocab=vocab
        )
        assert model.postprocess_block is not None and twin.postprocess_block is None
        assert model.mtp_process
        # A pack spans the windows of all CP ranks; its documents are multiples of 2 x CP.
        batches = [
            _pretrain_hybrid_batch(
                model,
                _thd_batch(
                    [cp * length for length in seqlens], vocab, seed, cp * tokens, max_num_seqs
                ),
            )
            for seed, seqlens in enumerate(_THD_PACKINGS + _THD_FULL_PACKINGS)
        ]
        assert all(batch["input_ids"].shape[-1] == tokens for batch in batches)

        helper = chunk_capture.capture_and_replay_against_eager(
            monkeypatch, model, twin, batches, tokens, step=_pretrain_hybrid_step
        )

        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        assert len(model.postprocess_block.cuda_graphs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hybrid_mtp_hidden_state_mixing_rolls_its_history_like_eager(
        self, chunk_capture, monkeypatch
    ):
        """MTP hidden-state mixing (``mtp_hsm``) rolls the older hidden states once per depth and
        picks the roll by whether the padded boundaries of the pack are a tensor of their own.
        ``pretrain_hybrid.py``'s data path hands one tensor in both roles, which selects the
        plain packed roll; the capture rebuilds the metadata from separate graph inputs and
        would record the other roll, whose document ends differ for documents of odd length.
        The rebuilt metadata therefore shares the tensors the way the eager calls saw them.

        The mixing always picks the oldest entry here (it samples per token otherwise), so
        every step depends on the rolled history and the two models compute the same thing."""
        chunk_capture.setup()
        tokens, max_num_seqs, vocab = 128, 4, 128
        model, twin = _build_chunk_hybrid_model_and_eager_twin(
            tokens, max_num_seqs, vocab=vocab, mtp_depths=2, mtp_hsm=True
        )
        assert model.postprocess_block is not None and twin.postprocess_block is None
        # Documents of odd length.
        batches = [
            _pretrain_hybrid_batch(model, _thd_batch(seqlens, vocab, seed, tokens, max_num_seqs))
            for seed, seqlens in enumerate(([33, 57, 31], [45, 75], [121], [17, 41, 59]))
        ]
        for batch in batches:
            packed_seq_params = batch["packed_seq_params"]
            assert packed_seq_params.cu_seqlens_q_padded is packed_seq_params.cu_seqlens_q

        mixes = []

        def oldest_entry(high, size, device=None, **kwargs):
            mixes.append(tuple(size))
            return torch.zeros(size, dtype=torch.long, device=device)

        monkeypatch.setattr(torch, "randint", oldest_entry)

        # The roll that eager never takes on this data path: every call of the translation
        # that prepares it, in an eager step, a capture or (none) a replay.
        from megatron.core.transformer import cuda_graphs, multi_token_prediction

        local_rolls = []
        translate = multi_token_prediction._packed_seq_params_for_local_hsm_roll

        def recording_translate(*args, **kwargs):
            local_rolls.append(cuda_graphs.is_graph_capturing())
            return translate(*args, **kwargs)

        monkeypatch.setattr(
            multi_token_prediction, "_packed_seq_params_for_local_hsm_roll", recording_translate
        )

        helper = chunk_capture.capture_and_replay_against_eager(
            monkeypatch, model, twin, batches, tokens, step=_pretrain_hybrid_step
        )

        assert mixes, "the hidden-state mixing never ran"
        assert not local_rolls, f"local hidden-state roll taken (while capturing: {local_rolls})"
        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        assert len(model.postprocess_block.cuda_graphs) > 0
        assert model.postprocess_block._packed_boundaries_shared_seen is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
    def test_calls_the_captured_postprocess_cannot_serve_run_the_eager_one(
        self, chunk_capture, monkeypatch, model_type
    ):
        """The captured post-process takes the tensors of a plain training call. A call that
        conditions the MTP or the LM head on anything else (an MTP input mask, per-call logits
        gathering, extra block keywords, an inference sequence offset) has to reach the eager
        post-process with exactly those arguments, after the decoder replay: the graphed
        callable would drop them without a word."""
        chunk_capture.setup()
        tokens, max_num_seqs, vocab = 128, 4, 128
        batch = _thd_batch([32, 40, 50], vocab, 0, tokens, max_num_seqs)
        if model_type == "gpt":
            model, twin = _build_chunk_gpt_model_and_eager_twin(
                256, 4, 4, 512, tokens, max_num_seqs, mtp_layers=1, vocab=vocab
            )
            step, eager_postprocess = _train_step_with_mtp_loss, "_postprocess"
        else:
            model, twin = _build_chunk_hybrid_model_and_eager_twin(
                tokens, max_num_seqs, vocab=vocab
            )
            step, eager_postprocess = _pretrain_hybrid_step, "_postprocess_after_decoder"
            batch = _pretrain_hybrid_batch(model, batch)
        model_class = type(model)

        generator = torch.Generator(device="cpu").manual_seed(7)
        mtp_input_mask = (torch.rand(1, tokens, generator=generator) > 0.3).cuda()
        fallbacks = {
            "mtp_input_mask": dict(mtp_input_mask=mtp_input_mask),
            "runtime_gather_output": dict(runtime_gather_output=True),
        }
        if model_type == "gpt":
            fallbacks["extra_block_kwargs"] = dict(extra_block_kwargs={"context": None})

        plain_reference = step(twin, batch)
        reference = {name: step(twin, batch, **kwargs) for name, kwargs in fallbacks.items()}
        # The mask changes what the MTP is trained on.
        masked_distance = _grad_distance(reference["mtp_input_mask"][1], plain_reference[1], "mtp.")
        assert masked_distance > 0.1, masked_distance

        helper = chunk_capture.capture(model, tokens)
        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        replays = _record_graph_replays(monkeypatch)

        # Which post-process a call takes: the block (a graph replay), or the model's eager
        # method called by forward itself, with which arguments.
        block_calls, eager_calls = [], []
        block_class = type(model.postprocess_block)
        block_call = block_class.__call__
        eager_method = getattr(model_class, eager_postprocess)

        def recording_block_call(block, *args, **kwargs):
            block_calls.append(kwargs)
            return block_call(block, *args, **kwargs)

        def recording_eager_postprocess(owner, *args, **kwargs):
            assert not args
            eager_calls.append(kwargs)
            return eager_method(owner, *args, **kwargs)

        monkeypatch.setattr(block_class, "__call__", recording_block_call)
        monkeypatch.setattr(model_class, eager_postprocess, recording_eager_postprocess)

        def run(**kwargs):
            block_calls.clear()
            eager_calls.clear()
            before = len(replays)
            result = step(model, batch, **kwargs)
            return result, len(replays) - before

        # A plain call replays both callables and runs no eager post-process.
        result, replayed = run()
        assert len(block_calls) == 1 and not eager_calls
        assert replayed >= 4
        _assert_matches_eager(result, plain_reference, "plain")

        for name, kwargs in fallbacks.items():
            result, replayed = run(**kwargs)
            assert not block_calls, name
            assert len(eager_calls) == 1, name
            for key, value in kwargs.items():
                assert eager_calls[0][key] is value, name
            # Decoder forward and backward still replay.
            assert replayed >= 2, name
            _assert_matches_eager(result, reference[name], name)

        if model_type == "gpt":
            # Only an inference pre-process produces a sequence offset; attention reads it for
            # flash decoding alone, so the numbers are those of the plain call.
            sequence_len_offset = torch.zeros(1, dtype=torch.int32, device="cuda")
            preprocess = model._preprocess

            def preprocess_with_sequence_len_offset(*args, **kwargs):
                output = list(preprocess(*args, **kwargs))
                output[4] = sequence_len_offset
                return tuple(output)

            monkeypatch.setattr(model, "_preprocess", preprocess_with_sequence_len_offset)
            result, replayed = run()
            assert not block_calls
            assert len(eager_calls) == 1
            assert eager_calls[0]["sequence_len_offset"] is sequence_len_offset
            assert replayed >= 2
            _assert_matches_eager(result, plain_reference, "sequence_len_offset")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lm_head_forward_hook_keeps_the_postprocess_eager(self, chunk_capture, monkeypatch):
        """Forward hooks on the LM head (logits savers, distillation captures) have to see the
        logits of every microbatch, and a replay runs no Python: with a hook registered when
        the capture helper is built, the decoder is the only captured callable and the
        post-process stays eager behind its replay."""
        chunk_capture.setup()
        tokens, max_num_seqs, vocab = 128, 4, 128
        model, twin = _build_chunk_gpt_model_and_eager_twin(
            256, 4, 4, 512, tokens, max_num_seqs, mtp_layers=1, vocab=vocab
        )
        logits_seen = {}
        for name, owner in (("model", model), ("twin", twin)):
            seen = logits_seen[name] = []
            handle = owner.output_layer.register_forward_hook(
                lambda module, inputs, output, seen=seen: seen.append(
                    output[0].detach().float().clone()
                )
            )
            chunk_capture.cleanups.append(handle.remove)
        batches = [
            _thd_batch(seqlens, vocab, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(_THD_PACKINGS[:3])
        ]

        reference, reference_logits = [], []
        for batch in batches:
            logits_seen["twin"].clear()
            reference.append(_train_step_with_mtp_loss(twin, batch))
            reference_logits.append(list(logits_seen["twin"]))
        # The LM head runs for the MTP depth and for the main loss.
        assert all(reference_logits)
        assert not torch.allclose(reference_logits[0][-1], reference_logits[1][-1])

        helper = chunk_capture.capture(model, tokens)
        assert helper.flattened_callables == [model.decoder]
        assert not model.postprocess_block.cuda_graphs
        assert "forward hooks" in (model.postprocess_block.eager_only_reason() or "")
        # The capture ran the decoder only.
        assert not logits_seen["model"]

        replays = _record_graph_replays(monkeypatch)
        for index in (2, 0, 1, 0):
            logits_seen["model"].clear()
            before = len(replays)
            result = _train_step_with_mtp_loss(model, batches[index])
            # Decoder forward and backward replay; nothing else is a graph.
            assert len(replays) - before >= 2, index
            _assert_matches_eager(result, reference[index], index)
            # The hook fired on this step, with this step's logits.
            assert len(logits_seen["model"]) == len(reference_logits[index]), index
            for logits, ref_logits in zip(logits_seen["model"], reference_logits[index]):
                torch.testing.assert_close(logits, ref_logits, rtol=2e-2, atol=2e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_config_logger_keeps_the_postprocess_eager(self, chunk_capture, monkeypatch, tmp_path):
        """``config_logger_dir`` dumps the inputs and logits of every forward from the
        post-process: a blocking device-to-host copy that a capture cannot record and a replay
        would never repeat. The dump includes the decoder input, which the post-process block
        does not take, so the forward calls the model's own post-process."""
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        chunk_capture.setup()
        model = _build_chunk_gpt_model(64, 4, 4, 128, 32, 4)
        model.train()

        def captured_callables():
            return TECudaGraphHelper(
                model=[model], config=model.config, seq_length=32, micro_batch_size=1
            ).flattened_callables

        assert model.postprocess_block.eager_only_reason() is None
        assert captured_callables() == [model.decoder, model.postprocess_block]

        # Set after the model is built, so no module dumps its construction arguments.
        model.config.config_logger_dir = str(tmp_path)
        assert captured_callables() == [model.decoder]
        assert "config_logger_dir" in (model.postprocess_block.eager_only_reason() or "")

        block_calls, eager_calls = [], []
        block_class = type(model.postprocess_block)
        block_call = block_class.__call__
        eager_postprocess = GPTModel._postprocess

        def recording_block_call(block, *args, **kwargs):
            block_calls.append(kwargs)
            return block_call(block, *args, **kwargs)

        def recording_eager_postprocess(owner, **kwargs):
            eager_calls.append(kwargs)
            return eager_postprocess(owner, **kwargs)

        monkeypatch.setattr(block_class, "__call__", recording_block_call)
        monkeypatch.setattr(GPTModel, "_postprocess", recording_eager_postprocess)

        _train_step(model, _thd_batch([8, 12], 128, 0, 32, 4))

        assert not block_calls
        assert len(eager_calls) == 1
        # The embedding output of this (pre-process) stage, as the dump records it.
        assert eager_calls[0]["decoder_input"] is not None
        assert list(tmp_path.iterdir())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        ("hash_layers", "hyper_connections"),
        [(0, False), (1, False), (0, True)],
        ids=["learned_router", "hash_router", "hyper_connections"],
    )
    def test_fixed_length_moe_capture_replays_the_padding_mask(
        self, chunk_capture, monkeypatch, hash_layers, hyper_connections
    ):
        """Fixed-length batches carry a padding mask only for some datasets. The eager warm-up
        steps of this run receive one, so the block graph takes a mask input in the block's
        token frame, and every replay routes with the mask of its own microbatch (the z-loss
        and the load-balancing loss count valid tokens only), or with an all-False one when a
        batch has none. With ``hash_layers`` the first MoE layer routes by token id: the ids
        are a graph input too. The hyper-connection layers hand the mask to the router
        themselves, so the block is what notes that a mask arrived."""
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        micro_batch_size, seq_length, vocab = 2, 64, 128
        chunk_capture.setup(micro_batch_size=micro_batch_size)
        # The scale of the auxiliary losses is one process-wide tensor, created by the first
        # backward that needs it: a fresh one for this test, the previous one back afterwards.
        monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
        model, twin = _build_chunk_gpt_model_and_eager_twin(
            256,
            4,
            4,
            512,
            seq_length,
            None,
            packed=False,
            vocab=vocab,
            moe_num_hash_layers=hash_layers,
            hash_moe_vocab_size=vocab if hash_layers else None,
            **(
                dict(
                    enable_hyper_connections=True, num_residual_streams=4, mhc_sinkhorn_iterations=5
                )
                if hyper_connections
                else {}
            ),
            **_CHUNK_MOE,
        )
        assert model.postprocess_block is None  # fixed length: the post-process stays eager
        routers = [layer.mlp.router for layer in model.decoder.layers]
        assert [router.is_hash_layer for router in routers] == [hash_layers > 0, False]

        def mask(*valid_lengths):
            padding_mask = torch.zeros(micro_batch_size, seq_length, dtype=torch.bool)
            for sample, valid in enumerate(valid_lengths):
                padding_mask[sample, valid:] = True
            return padding_mask.cuda()

        # Masks over real tokens: the tail of one sample, different tails of both, none.
        batches = [
            _fixed_length_batch(micro_batch_size, seq_length, vocab, seed, padding_mask)
            for seed, padding_mask in enumerate((mask(64, 40), mask(24, 56), None, mask(48, 64)))
        ]
        # The mask changes the router gradients: the same tokens with and without it.
        unmasked = dict(batches[0], padding_mask=None)
        learned_router = "decoder.layers.1.mlp.router.weight"
        mask_effect = _grad_distance(
            _train_step(twin, batches[0])[1], _train_step(twin, unmasked)[1], learned_router
        )
        assert mask_effect > 0.1, mask_effect

        chunk_capture.capture_and_replay_against_eager(
            monkeypatch,
            model,
            twin,
            batches,
            seq_length,
            micro_batch_size=micro_batch_size,
            replays_per_step=2,
        )

        assert model.decoder._chunk_graph_captures_padding_mask is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_mask_at_a_moe_graph_captured_without_one_is_refused(
        self, chunk_capture, monkeypatch
    ):
        """No mask reached the block during the eager warm-up steps, so its graph has no mask
        input: the graphed callable would drop a later mask and the routers would count the
        padding tokens. Batches without a mask replay."""
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        micro_batch_size, seq_length, vocab = 2, 64, 128
        chunk_capture.setup(micro_batch_size=micro_batch_size)
        monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
        model, twin = _build_chunk_gpt_model_and_eager_twin(
            256, 4, 4, 512, seq_length, None, packed=False, vocab=vocab, **_CHUNK_MOE
        )
        batches = [
            _fixed_length_batch(micro_batch_size, seq_length, vocab, seed) for seed in range(2)
        ]

        chunk_capture.capture_and_replay_against_eager(
            monkeypatch,
            model,
            twin,
            batches,
            seq_length,
            micro_batch_size=micro_batch_size,
            replays_per_step=2,
        )

        assert model.decoder._chunk_graph_captures_padding_mask is False
        padding_mask = torch.zeros(micro_batch_size, seq_length, dtype=torch.bool, device="cuda")
        padding_mask[1, 40:] = True
        masked = _fixed_length_batch(micro_batch_size, seq_length, vocab, 0, padding_mask)
        with pytest.raises(RuntimeError, match="captured without a padding_mask input"):
            _train_step(model, masked)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_delayed_scaling_updates_fp8_scales_on_replay(self, chunk_capture, monkeypatch):
        """Replay bypasses TransformerBlock.forward and the fp8_autocast it opens around the
        layers. TE's graphed callable registers the block's FP8 metas with the global amax buffer
        and runs the delayed-scaling reduce / scale update only inside an enclosing autocast, so
        the chunk replay opens one itself; without it every scale would keep its capture-time
        value for the rest of training (the control phase below)."""
        from contextlib import nullcontext

        from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphBlockMixin

        if not HAVE_TE:
            pytest.skip("Transformer Engine not available")
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        fp8_available, reason = FP8GlobalStateManager.is_fp8_available()
        if not fp8_available:
            pytest.skip(f"FP8 not available on this device: {reason}")

        chunk_capture.setup()
        tokens, max_num_seqs, vocab = 128, 4, 128
        model = _build_chunk_gpt_model(
            256,
            4,
            4,
            512,
            tokens,
            max_num_seqs,
            vocab=vocab,
            fp8="hybrid",
            fp8_recipe="delayed",
            fp8_amax_history_len=4,
            fp8_amax_compute_algo="max",
        )
        model.train()
        model.zero_grad_buffer = lambda: None
        batches = [
            _thd_batch(seqlens, vocab, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(([120], [60, 60], [32, 88]))
        ]

        chunk_capture.capture(model, tokens)

        # The modules whose forward quantizes under this recipe (the attention's FP8 meta
        # exists but records nothing while fp8_dpa is off). TE restores the pre-capture FP8
        # state after the capture: no amax has been reduced yet, every scale is still the
        # initial one.
        fp8_linears = [
            module
            for module in model.decoder.modules()
            if isinstance(module, (te.Linear, te.LayerNormLinear, te.LayerNormMLP))
        ]
        assert fp8_linears, "the decoder graph holds no FP8 modules"

        def forward_state():
            return [
                (
                    m.fp8_meta["scaling_fwd"].scale.clone(),
                    m.fp8_meta["scaling_fwd"].amax_history.clone(),
                )
                for m in fp8_linears
            ]

        initial = forward_state()
        assert all(torch.count_nonzero(history) == 0 for _, history in initial)
        assert all(torch.all(scale == 1) for scale, _ in initial)

        # Control: without the autocast around the replay nothing reduces the recorded amax
        # or recomputes a scale, however many steps run.
        with monkeypatch.context() as control:
            control.setattr(
                ChunkCudaGraphBlockMixin,
                "_chunk_graph_replay_quantization_context",
                lambda self: nullcontext(),
            )
            for batch in batches[:2]:
                _train_step(model, batch)
            torch.cuda.synchronize()
            for (scale, _), (initial_scale, _) in zip(forward_state(), initial):
                assert torch.equal(scale, initial_scale), (scale, initial_scale)
            assert all(torch.count_nonzero(history[1:]) == 0 for _, history in forward_state())

        for batch in batches:
            _train_step(model, batch)
        torch.cuda.synchronize()

        for (scale, history), (initial_scale, _) in zip(forward_state(), initial):
            # Each replay's forward exit reduced the amax the graph recorded and rolled it
            # into the history, and recomputed the scale from it.
            assert torch.count_nonzero(history[1:]) > 0, history
            assert not torch.equal(scale, initial_scale), (scale, initial_scale)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hybrid_mtp_stack_is_not_a_chunk_callable():
    """Only the decoder stack of a hybrid model is a chunk CUDA graph callable. The MTP stack
    (``HybridStack(..., is_mtp_layer=True)``) runs inside the post-process callable's capture
    and must not re-enter the chunk-capture adapter there (it would open a second offload
    handler and bump the paged-stash microbatch counter as if it were a new decoder chunk)."""
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import cuda_graphs

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=128,
            is_hybrid_model=True,
            mtp_num_layers=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            cuda_graph_impl="chunk_wise",
            cuda_graph_modules=[],
            cuda_graph_dynamic_microbatches=True,
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=64,
            thd_max_packed_sequences=4,
        )
        model_parallel_cuda_manual_seed(42)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=64,
            hybrid_layer_pattern="*-/*-",
            position_embedding_type="rope",
        )
        model.train()
        mtp_stack = model.mtp.layers[0].mtp_model_layer
        assert isinstance(mtp_stack, HybridStack) and mtp_stack.is_mtp_layer
        assert model.decoder.is_cuda_graph_chunk_callable is True
        assert mtp_stack.is_cuda_graph_chunk_callable is False
        cuda_graphs._set_capture_start()
        try:
            assert model.decoder._should_call_te_cudagraph()
            assert not mtp_stack._should_call_te_cudagraph()
        finally:
            cuda_graphs._set_capture_end()
    finally:
        Utils.destroy_model_parallel()


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
        assert layer._cuda_graph_padding_mask_shape == (1, 128)
        mask = static_inputs["padding_mask"]
        assert layer._te_cuda_graph_padding_mask(mask, mask.device) is mask
        torch.testing.assert_close(layer._te_cuda_graph_padding_mask(None, mask.device), mask)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("micro_batch_size", [1, 2])
    def test_hash_router_input_ids_follow_static_token_layout(self, packed, micro_batch_size):
        """THD flattens batches; SBHD retains the static batch and local sequence axes."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        layer.config.context_parallel_size = 2
        layer.config.sequence_packing_scheduler = "dp_balanced" if packed else None
        layer.config.cuda_graph_impl = "transformer_engine"
        layer.config.moe_num_hash_layers = 1
        layer.is_moe_layer = True
        layer.mlp.router = torch.nn.Identity()
        layer.mlp.router.is_hash_layer = True

        static_inputs = layer.get_layer_static_inputs(
            seq_length=256, micro_batch_size=micro_batch_size
        )

        expected = (
            (1, 128) if packed else (micro_batch_size, static_inputs["hidden_states"].shape[0])
        )
        assert static_inputs["input_ids"].shape == expected


@pytest.mark.internal
@pytest.mark.parametrize(
    "pre_process,post_process,pipeline_size,expected",
    [
        (False, False, 2, True),
        (False, True, 2, True),
        (False, False, 1, True),
        (True, False, 2, False),
    ],
)
def test_full_local_padding_mask_covers_every_non_preprocess_chunk(
    pre_process, post_process, pipeline_size, expected
):
    """Intermediate and last PP/VPP chunks both consume an unscattered local THD mask."""
    from types import SimpleNamespace

    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

    helper = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=True, pipeline_model_parallel_size=pipeline_size)
    )
    layer = SimpleNamespace(_is_thd_cuda_graph=lambda: True)
    chunk = SimpleNamespace(pre_process=pre_process, post_process=post_process)

    assert (
        TECudaGraphHelper._needs_full_local_padding_mask(
            helper, layer, chunk, {"padding_mask": object()}
        )
        is expected
    )


@pytest.mark.internal
def test_direct_layer_route_decompose_rejects_wrong_rank_before_flattening(monkeypatch):
    """Direct/MTP replay must validate the eager host rank tag before TE drops it."""
    from megatron.core import parallel_state
    from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

    class RankOneCPGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 1

    class RankZeroCPGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    cu = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    plan = cp_balanced_indexer.build_graph_dynamic_plan(cu, RankOneCPGroup(), 16)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    cp_balanced_indexer.attach_graph_dynamic_plan(packed, plan)
    layer = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=2, max_seqlen_per_dp_cp_rank=8),
        pg_collection=SimpleNamespace(cp=RankZeroCPGroup()),
        _uses_graph_dynamic_dsa_route=lambda: True,
        _set_te_cuda_graph_route_replay_state=lambda _packed: pytest.fail(
            "wrong-rank route reached replay-state setup"
        ),
    )
    # A conflicting global rank proves the direct path consults its explicit group.
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 1)

    with pytest.raises(ValueError, match="current rank-local contract"):
        TransformerLayer._decompose_packed_seq_params_to_kwargs(
            layer, {"packed_seq_params": packed}
        )


def _make_cpu_route_arena_block(num_slots=2):
    """Construct only the TransformerBlock state exercised by route-arena unit tests."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = object.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        dsa_cp_balance_indexer_graph_dynamic_packs=True,
        context_parallel_size=2,
        max_seqlen_per_dp_cp_rank=8,
    )
    block.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    block.current_microbatch = 0
    block._te_cuda_graph_route_metadata_arenas = ()
    block._te_cuda_graph_route_metadata_arena_ptrs = ()
    logical_route_numel = 20
    arenas = tuple(
        (
            torch.zeros(12, dtype=torch.int32),
            torch.cat(
                (
                    torch.zeros(logical_route_numel, dtype=torch.int64),
                    torch.full((slot + 1,), slot + 1, dtype=torch.int64),
                )
            ),
        )
        for slot in range(num_slots)
    )
    block.set_te_cuda_graph_route_metadata_arenas(arenas, logical_route_numel=logical_route_numel)
    return block, arenas


class TestGraphDynamicRouteMetadataArena:

    @pytest.mark.internal
    def test_block_stages_exactly_two_owners_once_and_preserves_source(self, monkeypatch):
        """Layer count does not multiply route copies; the caller's PSP stays untouched."""
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        block, arenas = _make_cpu_route_arena_block(num_slots=2)
        block.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
        block.current_microbatch = 3  # slot 1 by modulo
        source_layout = torch.arange(12, dtype=torch.int32)
        source_route = torch.arange(20, dtype=torch.int64)
        source = SimpleNamespace(route_buffers=(source_layout, source_route))

        monkeypatch.setattr(
            cp_balanced_indexer,
            "get_graph_dynamic_plan_buffers",
            lambda params: params.route_buffers,
        )
        monkeypatch.setattr(
            cp_balanced_indexer,
            "get_graph_dynamic_plan",
            lambda _params: {"cp_size": 2, "cp_rank": 0, "l_local": 8, "route_padding": 0},
        )

        def attach(
            params, layout_i32, route_i64, _cp_size, _l_local, *, route_padding=0, cp_rank=None
        ):
            params.route_buffers = (layout_i32, route_i64)
            params.route_padding = route_padding
            params.cp_rank = cp_rank

        monkeypatch.setattr(cp_balanced_indexer, "attach_graph_dynamic_plan_buffers", attach)

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            staged = block._stage_te_cuda_graph_route_metadata(source)

        copy_events = [event for event in prof.events() if event.name == "aten::copy_"]
        assert len(copy_events) == 2
        assert staged is not source
        assert staged.route_buffers[0] is arenas[1][0]
        assert staged.route_buffers[1] is arenas[1][1]
        assert staged.route_padding == 2
        assert staged.cp_rank == 0
        assert staged._te_cuda_graph_route_microbatch_id == 3
        assert staged._te_cuda_graph_route_slot == 1
        assert torch.equal(arenas[1][0], source_layout)
        assert torch.equal(arenas[1][1][: source_route.numel()], source_route)
        assert torch.equal(arenas[1][1][source_route.numel() :], torch.full((2,), 2))
        assert torch.count_nonzero(arenas[0][0]) == 0
        assert torch.count_nonzero(arenas[0][1][: source_route.numel()]) == 0
        assert source.route_buffers[0] is source_layout
        assert source.route_buffers[1] is source_route

    @pytest.mark.internal
    def test_block_rejects_unknown_rank_source_before_staging(self, monkeypatch):
        """A tensor-only reconstructed plan cannot be relabeled as a runtime rank."""
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        block, _ = _make_cpu_route_arena_block(num_slots=2)
        source = SimpleNamespace(
            route_buffers=(torch.arange(12, dtype=torch.int32), torch.arange(20, dtype=torch.int64))
        )
        monkeypatch.setattr(
            cp_balanced_indexer,
            "get_graph_dynamic_plan_buffers",
            lambda params: params.route_buffers,
        )
        monkeypatch.setattr(
            cp_balanced_indexer,
            "get_graph_dynamic_plan",
            lambda _params: {"cp_size": 2, "cp_rank": None, "l_local": 8, "route_padding": 0},
        )

        with pytest.raises(ValueError, match="exact rank-local source plan"):
            block._stage_te_cuda_graph_route_metadata(source)

    @pytest.mark.internal
    def test_slot_modulo_schema_and_pointer_guards(self):
        block, arenas = _make_cpu_route_arena_block(num_slots=2)
        selected = block.get_te_cuda_graph_route_metadata_arena(5)
        assert selected[0] is arenas[1][0]
        assert selected[1] is arenas[1][1]

        with pytest.raises(TypeError, match="int32 layout, int64 route"):
            block.set_te_cuda_graph_route_metadata_arenas(
                ((torch.zeros(12, dtype=torch.int64), torch.zeros(21, dtype=torch.int64)),),
                logical_route_numel=20,
            )
        with pytest.raises(ValueError, match="must not alias"):
            block.set_te_cuda_graph_route_metadata_arenas(
                (arenas[0], arenas[0]), logical_route_numel=20
            )

        block._te_cuda_graph_route_metadata_arenas = ((arenas[0][0].clone(), arenas[0][1]),) + (
            arenas[1],
        )
        with pytest.raises(RuntimeError, match="changed address"):
            block.get_te_cuda_graph_route_metadata_arena(0)

    @staticmethod
    def _make_capture_helper(num_chunks=2, num_slots=2, num_layers=2):
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        class RouteLayer:
            def _uses_graph_dynamic_dsa_route(self):
                return True

        class ArenaBlock:
            def __init__(self, layers):
                self.layers = layers
                self.arenas = ()
                self.pg_collection = SimpleNamespace(
                    cp=SimpleNamespace(size=lambda: 2, rank=lambda: 0)
                )

            def set_te_cuda_graph_route_metadata_arenas(
                self, arenas, *, logical_route_numel, cp_rank=None
            ):
                self.arenas = tuple(arenas)
                self.logical_route_numel = logical_route_numel
                self.cp_rank = cp_rank

            def clear_te_cuda_graph_route_metadata_arenas(self):
                self.arenas = ()

        helper = object.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            dsa_cp_balance_indexer_graph_dynamic_packs=True,
            context_parallel_size=2,
            overlap_moe_expert_parallel_comm=False,
            delay_wgrad_compute=False,
        )
        helper.num_microbatches = num_slots
        helper.callables_per_chunk = []
        helper.callables_per_chunk_is_mtp = []
        helper.chunks_with_decoder = []
        for _ in range(num_chunks):
            layers = [RouteLayer() for _ in range(num_layers)]
            block = ArenaBlock(layers)
            helper.callables_per_chunk.append(layers)
            helper.callables_per_chunk_is_mtp.append([False] * num_layers)
            helper.chunks_with_decoder.append(SimpleNamespace(decoder=block))
        helper.flattened_callables = [
            layer for layers in helper.callables_per_chunk for layer in layers
        ]

        sample_kwargs = []
        for sample_idx in range(len(helper.flattened_callables) * num_slots):
            sample_kwargs.append(
                {
                    "dsa_cp_graph_layout_buffer": torch.full((12,), sample_idx, dtype=torch.int32),
                    "dsa_cp_graph_route_buffer": torch.full((20,), sample_idx, dtype=torch.int64),
                    "unrelated": torch.tensor(sample_idx),
                }
            )
        return helper, sample_kwargs

    @pytest.mark.internal
    def test_capture_shares_within_chunk_slot_and_isolates_vpp_chunks_and_slots(self):
        helper, sample_kwargs = self._make_capture_helper()
        unrelated_ptrs = [kwargs["unrelated"].data_ptr() for kwargs in sample_kwargs]
        route_arenas = helper._canonicalize_graph_dynamic_route_inputs(sample_kwargs)

        all_ptrs = set()
        route_lengths = set()
        for chunk_idx in range(2):
            chunk_base = chunk_idx * 2 * 2
            for slot in range(2):
                first_idx = chunk_base + slot * 2
                second_idx = first_idx + 1
                first_pair = (
                    sample_kwargs[first_idx]["dsa_cp_graph_layout_buffer"],
                    sample_kwargs[first_idx]["dsa_cp_graph_route_buffer"],
                )
                second_pair = (
                    sample_kwargs[second_idx]["dsa_cp_graph_layout_buffer"],
                    sample_kwargs[second_idx]["dsa_cp_graph_route_buffer"],
                )
                assert first_pair[0].data_ptr() == second_pair[0].data_ptr()
                assert first_pair[1].data_ptr() == second_pair[1].data_ptr()
                assert first_pair[0].data_ptr() not in all_ptrs
                assert first_pair[1].data_ptr() not in all_ptrs
                all_ptrs.update((first_pair[0].data_ptr(), first_pair[1].data_ptr()))
                assert first_pair[1].numel() not in route_lengths
                route_lengths.add(first_pair[1].numel())
        assert [kwargs["unrelated"].data_ptr() for kwargs in sample_kwargs] == unrelated_ptrs

        helper._attach_graph_dynamic_route_arenas(sample_kwargs, route_arenas)
        assert all(len(chunk.decoder.arenas) == 2 for chunk in helper.chunks_with_decoder)

        old_ptrs = set(all_ptrs)
        helper._clear_graph_dynamic_route_arenas()
        assert all(not chunk.decoder.arenas for chunk in helper.chunks_with_decoder)

        next_kwargs = self._make_capture_helper()[1]
        next_arenas = helper._canonicalize_graph_dynamic_route_inputs(next_kwargs)
        helper._attach_graph_dynamic_route_arenas(next_kwargs, next_arenas)
        next_ptrs = {
            tensor.data_ptr()
            for chunk in helper.chunks_with_decoder
            for pair in chunk.decoder.arenas
            for tensor in pair
        }
        assert old_ptrs.isdisjoint(next_ptrs)

    @pytest.mark.internal
    def test_post_capture_pointer_drift_fails_closed(self):
        helper, sample_kwargs = self._make_capture_helper(num_chunks=1)
        route_arenas = helper._canonicalize_graph_dynamic_route_inputs(sample_kwargs)
        sample_kwargs[0]["dsa_cp_graph_layout_buffer"] = sample_kwargs[0][
            "dsa_cp_graph_layout_buffer"
        ].clone()

        with pytest.raises(RuntimeError, match="split one canonical DSA route arena"):
            helper._attach_graph_dynamic_route_arenas(sample_kwargs, route_arenas)

    @pytest.mark.internal
    def test_post_te_rebind_uses_final_common_pair(self):
        helper, sample_kwargs = self._make_capture_helper(num_chunks=1, num_slots=1)
        route_arenas = helper._canonicalize_graph_dynamic_route_inputs(sample_kwargs)
        expected_pair = route_arenas[(0, 0)]["pair"]
        final_pair = (torch.zeros_like(expected_pair[0]), torch.zeros_like(expected_pair[1]))
        for graph_idx in route_arenas[(0, 0)]["graph_indices"]:
            sample_kwargs[graph_idx]["dsa_cp_graph_layout_buffer"] = final_pair[0]
            sample_kwargs[graph_idx]["dsa_cp_graph_route_buffer"] = final_pair[1]

        helper._attach_graph_dynamic_route_arenas(sample_kwargs, route_arenas)
        attached = helper.chunks_with_decoder[0].decoder.arenas[0]
        assert attached[0] is final_pair[0]
        assert attached[1] is final_pair[1]

    @pytest.mark.internal
    def test_rank_without_route_callable_is_noop_and_mtp_keeps_direct_fallback(self):
        helper, _sample_kwargs = self._make_capture_helper(num_chunks=1)
        helper.callables_per_chunk[0] = []
        helper.callables_per_chunk_is_mtp[0] = []
        helper.flattened_callables = []
        assert helper._canonicalize_graph_dynamic_route_inputs([]) == {}

        helper, sample_kwargs = self._make_capture_helper(num_chunks=1, num_slots=1, num_layers=1)
        helper.callables_per_chunk_is_mtp = [[True]]
        original_ptrs = (
            sample_kwargs[0]["dsa_cp_graph_layout_buffer"].data_ptr(),
            sample_kwargs[0]["dsa_cp_graph_route_buffer"].data_ptr(),
        )
        assert helper._canonicalize_graph_dynamic_route_inputs(sample_kwargs) == {}
        assert (
            sample_kwargs[0]["dsa_cp_graph_layout_buffer"].data_ptr(),
            sample_kwargs[0]["dsa_cp_graph_route_buffer"].data_ptr(),
        ) == original_ptrs

    @pytest.mark.internal
    def test_transformer_and_hybrid_stacks_share_route_arena_owner_methods(self):
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.transformer.module import MegatronModule
        from megatron.core.transformer.transformer_block import TransformerBlock

        assert (
            TransformerBlock._stage_te_cuda_graph_route_metadata
            is MegatronModule._stage_te_cuda_graph_route_metadata
        )
        assert (
            HybridStack._stage_te_cuda_graph_route_metadata
            is MegatronModule._stage_te_cuda_graph_route_metadata
        )

        hybrid = object.__new__(HybridStack)
        torch.nn.Module.__init__(hybrid)
        hybrid.config = SimpleNamespace(context_parallel_size=1)
        hybrid.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0))
        arenas = ((torch.zeros(12, dtype=torch.int32), torch.zeros(21, dtype=torch.int64)),)
        hybrid.set_te_cuda_graph_route_metadata_arenas(arenas, logical_route_numel=20)
        selected = hybrid.get_te_cuda_graph_route_metadata_arena(0)
        assert selected[0] is arenas[0][0]
        assert selected[1] is arenas[0][1]

    @pytest.mark.internal
    def test_set_current_microbatch_updates_owning_block(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs

        block = SimpleNamespace(layers=[])
        wrapped = SimpleNamespace(decoder=block)

        def get_wrapped(_model, attr, **_kwargs):
            if attr == "decoder":
                return wrapped
            if attr == "vision_model":
                return None
            raise AssertionError(attr)

        monkeypatch.setattr(cuda_graphs, "get_attr_wrapped_model", get_wrapped)
        cuda_graphs.set_current_microbatch(object(), 7)
        assert block.current_microbatch == 7

    def test_set_current_microbatch_follows_a_language_model_wrapper(self, monkeypatch):
        """The graph helper captures the callables of the model a wrapper holds as
        ``language_model`` (decoder block and post-process block). The wrapper exposes that
        model's decoder but not its post-process block, which has to follow the microbatch as
        well, or it would replay slot 0 for every microbatch."""
        from megatron.core.transformer import cuda_graphs

        block = SimpleNamespace(layers=[], config=SimpleNamespace(cuda_graph_impl="chunk_wise"))
        postprocess_block = SimpleNamespace()
        language_model = SimpleNamespace(decoder=block, postprocess_block=postprocess_block)
        wrapper = SimpleNamespace(decoder=block, language_model=language_model)

        def get_wrapped(_model, attr, **_kwargs):
            return wrapper if attr == "decoder" else None

        monkeypatch.setattr(cuda_graphs, "get_attr_wrapped_model", get_wrapped)
        cuda_graphs.set_current_microbatch(object(), 3)
        assert block.current_microbatch == 3
        assert postprocess_block.current_microbatch == 3

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_te_shared_route_arenas_replay_30_dynamic_packs(self, monkeypatch):
        """Real TE graphs share one two-copy route arena per stack slot for 30 replays."""
        import transformer_engine.pytorch.graph as te_graph

        from megatron.core import parallel_state
        from megatron.core.transformer.cuda_graphs import (
            TECudaGraphHelper,
            _set_capture_end,
            _set_capture_start,
        )
        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer
        from megatron.core.transformer.module import GraphableMegatronModule
        from megatron.core.transformer.transformer_block import TransformerBlock
        from megatron.core.utils import is_te_min_version

        if not is_te_min_version("2.7.0"):
            pytest.skip("route-arena integration requires TE >= 2.7")

        # Keep this test topology-local. The route builder and owner attach still use the
        # production CP=2, rank-0 contract, but no process group or collective is required.
        monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)
        config = SimpleNamespace(
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[],
            fine_grained_activation_offloading=False,
            delay_offload_until_cuda_graph=False,
            # TransformerConfig derives graph dynamic packs from the dp_balanced scheduler.
            # This single-chunk schedule is PP=1, so TE slot selection keeps modulo reuse.
            sequence_packing_scheduler="dp_balanced",
            pipeline_model_parallel_size=1,
            dsa_cp_balance_indexer_graph_dynamic_packs=True,
            context_parallel_size=2,
            max_seqlen_per_dp_cp_rank=8,
            overlap_moe_expert_parallel_comm=False,
            delay_wgrad_compute=False,
        )

        class LocalCPGroup:
            @staticmethod
            def size():
                return 2

            @staticmethod
            def rank():
                return 0

        class RouteToyLayer(GraphableMegatronModule):
            def __init__(self, layer_number):
                # A tiny test-only Graphable module avoids constructing a full TransformerLayer,
                # while exercising GraphableMegatronModule's real TE replay/slot selection.
                torch.nn.Module.__init__(self)
                self.config = config
                self.pg_collection = SimpleNamespace(cp=LocalCPGroup())
                self.layer_number = layer_number
                self.gain = torch.nn.Parameter(
                    torch.linspace(
                        0.9 + 0.1 * layer_number, 1.2 + 0.1 * layer_number, 4, device="cuda"
                    )
                )
                self.bias = torch.nn.Parameter(
                    torch.linspace(-0.2, 0.1, 4, device="cuda") + 0.05 * layer_number
                )
                self.cuda_graphs = []
                self.cuda_graph_manual_hooks = []
                self._te_cuda_graph_route_replay_state = None

            @staticmethod
            def _uses_graph_dynamic_dsa_route():
                return True

            def _decompose_packed_seq_params_to_kwargs(self, kwargs):
                # Exercise the production rank check and staged-slot tag plumbing.
                return TransformerLayer._decompose_packed_seq_params_to_kwargs(self, kwargs)

            def _te_cuda_graph_replay(self, *args, **kwargs):
                # The production replay wrapper owns PackedSeqParams decomposition.
                return TransformerLayer._te_cuda_graph_replay(self, *args, **kwargs)

            def _te_cuda_graph_replay_impl(self, args, kwargs, context):
                assert context is None
                return GraphableMegatronModule._te_cuda_graph_replay(self, *args, **kwargs)

            def forward(
                self,
                hidden_states,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                dsa_cp_graph_layout_buffer,
                dsa_cp_graph_route_buffer,
                is_first_microbatch=False,
            ):
                del is_first_microbatch
                signal = (
                    dsa_cp_graph_layout_buffer[:4].to(hidden_states.dtype).sum() * 0.01
                    + dsa_cp_graph_route_buffer[:4].to(hidden_states.dtype).sum() * 0.001
                    + cu_seqlens_q.to(hidden_states.dtype).sum() * 0.0001
                    + cu_seqlens_kv.to(hidden_states.dtype).sum() * 0.0001
                    + cu_seqlens_q_padded.to(hidden_states.dtype).sum() * 0.0001
                    + cu_seqlens_kv_padded.to(hidden_states.dtype).sum() * 0.0001
                )
                return hidden_states * (self.gain + signal * 0.001) + self.bias * signal * 0.01

        graph_layers = torch.nn.ModuleList((RouteToyLayer(0), RouteToyLayer(1)))
        eager_layers = torch.nn.ModuleList((RouteToyLayer(0), RouteToyLayer(1)))
        for graph_layer, eager_layer in zip(graph_layers, eager_layers):
            eager_layer.load_state_dict(graph_layer.state_dict())
            graph_layer.train()
            eager_layer.train()

        block = object.__new__(TransformerBlock)
        torch.nn.Module.__init__(block)
        block.config = config
        block.layers = graph_layers
        block.pg_collection = SimpleNamespace(cp=LocalCPGroup())
        block.current_microbatch = 0
        block.clear_te_cuda_graph_route_metadata_arenas()

        helper = object.__new__(TECudaGraphHelper)
        helper.config = config
        helper.num_microbatches = 2
        helper.callables_per_chunk = [list(graph_layers)]
        helper.callables_per_chunk_is_mtp = [[False, False]]
        helper.chunks_with_decoder = [SimpleNamespace(decoder=block)]
        helper.flattened_callables = list(graph_layers)

        # These are three genuine fixed-capacity zigzag plans. K=5 for each plan,
        # CP=2, L=8, and every sequence length is divisible by 2*CP.
        source_packs = []
        for boundaries in ((0, 4, 8, 16, 16), (0, 4, 4, 8, 16), (0, 0, 4, 8, 16)):
            cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
            built = cp_balanced_indexer.build_graph_dynamic_plan(cu, LocalCPGroup(), 16)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu.clone(),
                cu_seqlens_q_padded=cu.clone(),
                cu_seqlens_kv_padded=cu.clone(),
                max_seqlen_q=16,
                max_seqlen_kv=16,
            )
            cp_balanced_indexer.attach_graph_dynamic_plan_buffers(
                packed, built["layout_i32"], built["route_i64"], 2, 8, cp_rank=0
            )
            source_packs.append(packed)

        source_pairs = tuple(
            cp_balanced_indexer.get_graph_dynamic_plan_buffers(packed) for packed in source_packs
        )
        logical_route_numel = source_pairs[0][1].numel()
        assert logical_route_numel == 26
        assert all(
            cp_balanced_indexer.get_graph_dynamic_plan(packed)["route_padding"] == 0
            for packed in source_packs
        )
        source_ptrs = tuple(tuple(owner.data_ptr() for owner in pair) for pair in source_pairs)
        source_snapshots = tuple(tuple(owner.clone() for owner in pair) for pair in source_pairs)
        cu_names = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
        source_cu_ptrs = tuple(
            tuple(getattr(packed, name).data_ptr() for name in cu_names) for packed in source_packs
        )
        source_cu_snapshots = tuple(
            tuple(getattr(packed, name).clone() for name in cu_names) for packed in source_packs
        )
        source_cu_ptr_set = {ptr for ptrs in source_cu_ptrs for ptr in ptrs}

        sample_args = tuple(
            (torch.full((8, 4), 0.125 * (sample_idx + 1), device="cuda", requires_grad=True),)
            for sample_idx in range(4)
        )
        sample_kwargs = [
            {
                # TE uses sample tensors as mutable static replay surfaces. Keep capture
                # inputs disjoint from runtime pack metadata, whose ownership stays with the
                # data pipeline and must survive later graph-input copies unchanged.
                "cu_seqlens_q": source_packs[0].cu_seqlens_q.clone(),
                "cu_seqlens_kv": source_packs[0].cu_seqlens_kv.clone(),
                "cu_seqlens_q_padded": source_packs[0].cu_seqlens_q_padded.clone(),
                "cu_seqlens_kv_padded": source_packs[0].cu_seqlens_kv_padded.clone(),
                "dsa_cp_graph_layout_buffer": source_pairs[0][0],
                "dsa_cp_graph_route_buffer": source_pairs[0][1],
            }
            for _ in range(4)
        ]
        capture_cu_ptrs = {kwargs[name].data_ptr() for kwargs in sample_kwargs for name in cu_names}
        assert len(capture_cu_ptrs) == len(sample_kwargs) * len(cu_names)
        assert capture_cu_ptrs.isdisjoint(source_cu_ptr_set)
        route_arenas = helper._canonicalize_graph_dynamic_route_inputs(sample_kwargs)

        graphed = ()
        _set_capture_start()
        try:
            # This toy contains no RNG operation. Other tests in the same worker may leave
            # TE's process-global auxiliary RNG registry populated with legacy Tensor states,
            # while TE's graph-safe capture path assumes every auxiliary entry is a Generator
            # and calls get_state() on it. Isolate this RNG-free capture from those unrelated
            # tracker entries while preserving TE's normal default CUDA generator handling.
            with monkeypatch.context() as capture_patch:
                capture_patch.setattr(te_graph, "get_all_rng_states", lambda: {})
                graphed = te_graph.make_graphed_callables(
                    tuple(graph_layers),
                    sample_args,
                    sample_kwargs=sample_kwargs,
                    allow_unused_input=True,
                    num_warmup_iters=1,
                    _order=[1, -1, 1, -1],
                    _num_layers_per_chunk=[2],
                    retain_graph_in_backward=False,
                    _reuse_graph_input_output_buffers=True,
                    fp8_enabled=False,
                )
        finally:
            _set_capture_end()

        try:
            assert len(graphed) == 4
            assert {
                kwargs[name].data_ptr() for kwargs in sample_kwargs for name in cu_names
            }.isdisjoint(source_cu_ptr_set)
            helper._attach_graph_dynamic_route_arenas(sample_kwargs, route_arenas)
            arenas = block._te_cuda_graph_route_metadata_arenas
            assert tuple(pair[1].numel() for pair in arenas) == (
                logical_route_numel + 1,
                logical_route_numel + 2,
            )
            assert len({pair[1].shape for pair in arenas}) == 2

            # The final TE-owned K=5 views, not only the source plan, must satisfy
            # cuDNN Frontend's 16-byte packed-layout pointer contract.
            for slot, pair in enumerate(arenas):
                final_plan = cp_balanced_indexer._graph_dynamic_plan_from_buffers(
                    pair[0], pair[1], 2, 8, route_padding=slot + 1, cp_rank=0
                )
                for layout_name in ("head_layout", "tail_layout", "output_layout"):
                    assert all(tensor.data_ptr() % 16 == 0 for tensor in final_plan[layout_name])

            # The two layer graphs for a slot must expose the same final TE static owners.
            for slot in range(2):
                arena_ptrs = tuple(owner.data_ptr() for owner in arenas[slot])
                for layer_number in range(2):
                    graph_idx = slot * 2 + layer_number
                    assert (
                        sample_kwargs[graph_idx]["dsa_cp_graph_layout_buffer"].data_ptr(),
                        sample_kwargs[graph_idx]["dsa_cp_graph_route_buffer"].data_ptr(),
                    ) == arena_ptrs
                graph_layers[0].cuda_graphs.append(graphed[slot * 2])
                graph_layers[1].cuda_graphs.append(graphed[slot * 2 + 1])

            arena_ptrs = tuple(tuple(owner.data_ptr() for owner in pair) for pair in arenas)
            suffixes = tuple(
                pair[1]
                .narrow(0, logical_route_numel, pair[1].numel() - logical_route_numel)
                .clone()
                for pair in arenas
            )
            assert torch.equal(suffixes[0], torch.ones(1, dtype=torch.int64, device="cuda"))
            assert torch.equal(suffixes[1], torch.full((2,), 2, dtype=torch.int64, device="cuda"))

            for layer in tuple(graph_layers) + tuple(eager_layers):
                for parameter in layer.parameters():
                    # Keep optimizer-owned grad buffers alive across TE graph replay. If a
                    # leaf grad starts as None, AccumulateGrad may adopt TE's recyclable static
                    # grad-input buffer instead of copying from it, extending that buffer's
                    # lifetime beyond the graph schedule that owns it.
                    parameter.grad = torch.zeros_like(parameter)

            hidden_template = torch.linspace(-0.75, 0.75, 32, device="cuda").view(8, 4)
            grad_output = torch.linspace(0.5, -0.5, 32, device="cuda").view(8, 4)
            first_outputs = {}
            for replay_idx in range(30):
                for layer in tuple(graph_layers) + tuple(eager_layers):
                    for parameter in layer.parameters():
                        parameter.grad.zero_()

                pack_idx = replay_idx % 3
                source = source_packs[pack_idx]
                source_pair = source_pairs[pack_idx]
                slot = replay_idx % 2
                other_slot = 1 - slot
                block.current_microbatch = replay_idx

                selected_versions = tuple(owner._version for owner in arenas[slot])
                other_versions = tuple(owner._version for owner in arenas[other_slot])
                other_snapshot = tuple(owner.clone() for owner in arenas[other_slot])
                staged = block._stage_te_cuda_graph_route_metadata(source)
                staged_pair = cp_balanced_indexer.get_graph_dynamic_plan_buffers(staged)
                assert tuple(owner.data_ptr() for owner in staged_pair) == arena_ptrs[slot]
                torch.testing.assert_close(staged_pair[0], source_pair[0], rtol=0, atol=0)
                torch.testing.assert_close(
                    staged_pair[1][:logical_route_numel], source_pair[1], rtol=0, atol=0
                )
                assert tuple(owner._version for owner in arenas[slot]) == tuple(
                    version + 1 for version in selected_versions
                )

                graph_hidden = hidden_template.clone().requires_grad_(True)
                # As with parameters, use a caller-owned leaf-grad buffer so the assertion
                # cannot observe a TE static dgrad allocation after the shared graph pool has
                # reused it. Give graph and eager separate upstream grads as TE is free to use
                # an incoming backward tensor as graph workspace.
                graph_hidden.grad = torch.zeros_like(graph_hidden)
                graph_grad_output = grad_output.clone()
                eager_grad_output = grad_output.clone()
                graph_output = graph_hidden
                for layer in graph_layers:
                    # Simulate checkpoint recompute after the global layer state advanced to
                    # another slot. The staged invocation must still select its retained graph.
                    layer.current_microbatch = replay_idx + 1
                    graph_output = layer(graph_output, packed_seq_params=staged)
                    assert layer._te_cuda_graph_route_replay_state is None
                # TE returns a detached view of its recyclable static output. With
                # input/output-buffer reuse enabled, that view is owned only until its
                # scheduled consumer/backward; retain the value before crossing that boundary.
                graph_output_value = graph_output.detach().clone()
                graph_output.backward(graph_grad_output)

                eager_hidden = hidden_template.clone().requires_grad_(True)
                eager_hidden.grad = torch.zeros_like(eager_hidden)
                eager_output = eager_hidden
                for layer in eager_layers:
                    eager_output = layer(
                        eager_output,
                        cu_seqlens_q=source.cu_seqlens_q,
                        cu_seqlens_kv=source.cu_seqlens_kv,
                        cu_seqlens_q_padded=source.cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=source.cu_seqlens_kv_padded,
                        dsa_cp_graph_layout_buffer=source_pair[0],
                        dsa_cp_graph_route_buffer=source_pair[1],
                    )
                eager_output_value = eager_output.detach().clone()
                eager_output.backward(eager_grad_output)

                torch.testing.assert_close(graph_output_value, eager_output_value, rtol=0, atol=0)
                torch.testing.assert_close(graph_hidden.grad, eager_hidden.grad, rtol=0, atol=0)
                for graph_layer, eager_layer in zip(graph_layers, eager_layers):
                    for (graph_name, graph_param), (eager_name, eager_param) in zip(
                        graph_layer.named_parameters(), eager_layer.named_parameters()
                    ):
                        assert graph_name == eager_name
                        assert graph_param.grad is not None and eager_param.grad is not None
                        torch.testing.assert_close(
                            graph_param.grad, eager_param.grad, rtol=0, atol=0
                        )

                # Exactly the stack-level two copies may update owner versions. A drift to the
                # other layer slot would trigger TE input copies and mutate that static arena.
                assert tuple(owner._version for owner in arenas[slot]) == tuple(
                    version + 1 for version in selected_versions
                )
                assert tuple(owner._version for owner in arenas[other_slot]) == other_versions
                for actual, expected in zip(arenas[other_slot], other_snapshot):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for current_suffix, expected_suffix in zip(
                    (
                        pair[1].narrow(
                            0, logical_route_numel, pair[1].numel() - logical_route_numel
                        )
                        for pair in arenas
                    ),
                    suffixes,
                ):
                    torch.testing.assert_close(current_suffix, expected_suffix, rtol=0, atol=0)
                assert (
                    tuple(tuple(owner.data_ptr() for owner in pair) for pair in arenas)
                    == arena_ptrs
                )
                assert (
                    tuple(tuple(owner.data_ptr() for owner in pair) for pair in source_pairs)
                    == source_ptrs
                )
                for actual_pair, expected_pair in zip(source_pairs, source_snapshots):
                    for actual, expected in zip(actual_pair, expected_pair):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert (
                    tuple(
                        tuple(getattr(packed, name).data_ptr() for name in cu_names)
                        for packed in source_packs
                    )
                    == source_cu_ptrs
                )
                for packed, expected_pack in zip(source_packs, source_cu_snapshots):
                    for name, expected in zip(cu_names, expected_pack):
                        torch.testing.assert_close(getattr(packed, name), expected, rtol=0, atol=0)

                if pack_idx in first_outputs:
                    torch.testing.assert_close(
                        graph_output_value, first_outputs[pack_idx], rtol=0, atol=0
                    )
                else:
                    first_outputs[pack_idx] = graph_output_value

            assert len(first_outputs) == 3
            assert not torch.equal(first_outputs[0], first_outputs[1])
            assert not torch.equal(first_outputs[1], first_outputs[2])
        finally:
            torch.cuda.synchronize()
            for graph in graphed:
                if hasattr(graph, "reset"):
                    graph.reset()
            for layer in graph_layers:
                layer.cuda_graphs.clear()
            block.clear_te_cuda_graph_route_metadata_arenas()


class TestDynamicMicrobatchSlots:

    @pytest.mark.internal
    def test_capture_count_includes_topology_liveness_floor(self):
        """A small capture-step GBS must not discard the PP/VPP slot lower bound."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        assert TECudaGraphHelper._get_dynamic_capture_num_microbatches(1, 1, 2) == 2
        assert TECudaGraphHelper._get_dynamic_capture_num_microbatches(2, 5, 3) == 5

    @pytest.mark.internal
    def test_production_capture_plan_enforces_run_level_gbs_contract(self, monkeypatch):
        """Real slot selection sizes fixed-GBS packing and rejects a later GBS increase."""
        from types import SimpleNamespace

        from megatron.core.transformer import cuda_graphs

        helper = object.__new__(cuda_graphs.TECudaGraphHelper)
        helper.config = SimpleNamespace(
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=4096,
            virtual_pipeline_model_parallel_size=None,
            thd_max_packed_sequences=None,
        )
        helper.dp_group = SimpleNamespace(size=lambda: 1)
        helper.dp_cp_group = SimpleNamespace(size=lambda: 2)
        helper.micro_batch_size = 1
        helper.seq_length = 4096
        helper.thd_sequence_length_upper_bound = 4096
        monkeypatch.setattr(cuda_graphs, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(cuda_graphs, "get_current_running_global_batch_size", lambda: 8)
        monkeypatch.setattr(cuda_graphs, "get_global_batch_size_upper_bound", lambda: 8)

        plan = helper._get_dynamic_capture_plan(
            auto_num_slots=3, microbatch_group_size_per_vp_stage=2
        )
        assert plan == (4, 1, 8, 8, 4, "thd_varlen_upper_bound")

        plan = helper._get_dynamic_capture_plan(
            auto_num_slots=5, microbatch_group_size_per_vp_stage=2
        )
        assert plan[0] == 5

        monkeypatch.setattr(cuda_graphs, "get_current_running_global_batch_size", lambda: 1)
        with pytest.raises(ValueError, match="step_batch_size_schedule"):
            helper._get_dynamic_capture_plan(auto_num_slots=3, microbatch_group_size_per_vp_stage=2)

    @staticmethod
    def _slot_agreement_helper(monkeypatch, *, pipeline_size, dynamic_slots=True):
        """A helper stand-in on pipeline rank 1 and the pipeline-group reductions it issues."""
        from types import SimpleNamespace

        from megatron.core import parallel_state
        from megatron.core.transformer import cuda_graphs

        helper = object.__new__(cuda_graphs.TECudaGraphHelper)
        helper.pp_group = SimpleNamespace(size=lambda: pipeline_size)
        helper.num_model_chunks = 1
        helper.flattened_callables = []
        helper._captures_dsa_metric_writes = False
        helper.config = SimpleNamespace(
            cuda_graph_dynamic_microbatches=dynamic_slots,
            overlap_moe_expert_parallel_comm=False,
            microbatch_group_size_per_vp_stage=None,
            moe_paged_stash=False,
        )
        monkeypatch.setattr(
            parallel_state, "get_pipeline_model_parallel_world_size", lambda: pipeline_size
        )
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(
            parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None
        )
        reductions = []

        def record_reduction(tensor, op=None, group=None):
            reductions.append((int(tensor.item()), op, group))
            tensor.fill_(7)  # a peer stage reports a larger floor

        monkeypatch.setattr(torch.distributed, "all_reduce", record_reduction)
        return helper, reductions

    @pytest.mark.internal
    def test_rank_without_graphable_layers_joins_the_slot_agreement(self, monkeypatch):
        """A stage with no decoder layer reduces the slot floor with its pipeline peers.

        The slot floor of a run with dynamic microbatch slots is reduced over the pipeline group
        while the capture inputs are built. A rank with nothing to capture (an embedding-only or
        loss-only stage) skips that step; without its own reduction, its next pipeline-group
        collective (the packing scheduler's "out of data" flag) pairs with the peers' slot floor.
        """
        from megatron.core.transformer import cuda_graphs

        helper, reductions = self._slot_agreement_helper(monkeypatch, pipeline_size=4)
        # What a rank with layers does while it builds the capture inputs.
        assert helper._agree_on_dynamic_microbatch_slots(4) == 7
        with_layers = list(reductions)
        assert len(with_layers) == 1
        floor, op, group = with_layers[0]
        assert floor >= 1 and op == torch.distributed.ReduceOp.MAX and group is helper.pp_group

        # The capture entry point of a rank without layers issues the same reduction.
        reductions.clear()
        monkeypatch.setattr(cuda_graphs, "validate_moe_cuda_graph_support", lambda config: None)
        helper._start_capturing = lambda: 0.0
        finished = []
        helper._finish_capturing = finished.append
        helper.create_cudagraphs()
        assert reductions == with_layers
        assert finished == [0.0]

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "pipeline_size, dynamic_slots", [(1, True), (4, False)], ids=["pp1", "fixed-slots"]
    )
    def test_rank_without_graphable_layers_adds_no_collective_otherwise(
        self, monkeypatch, pipeline_size, dynamic_slots
    ):
        """Without pipeline peers, or with one slot per microbatch, nothing is reduced."""
        helper, reductions = self._slot_agreement_helper(
            monkeypatch, pipeline_size=pipeline_size, dynamic_slots=dynamic_slots
        )
        helper._join_capture_collectives()
        assert reductions == []

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


def _run_pretrain(model_args, cuda_graph_args, master_port):
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
        cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=900
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
