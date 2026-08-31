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
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    _resolve_thd_padding_lengths,
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
def test_hybrid_padding_mask_sp_scatter_uses_model_tp_group(monkeypatch):
    """Hybrid embedding and padding-mask SP splits must use the same explicit TP group."""
    from megatron.core import tensor_parallel
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    tp_group = object()
    scatter_calls = []

    def fake_sp_scatter(input_, group=None):
        scatter_calls.append((input_.dtype, group))
        return input_.narrow(0, 0, input_.shape[0] // 2)

    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", fake_sp_scatter)

    class FakeEmbedding:
        scatter_to_sequence_parallel = False

        def __call__(self, input_ids, position_ids):
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 8)

    class FakeDecoder:
        def __call__(self, *, hidden_states, padding_mask, **_kwargs):
            assert padding_mask.shape == (1, hidden_states.shape[0])
            return hidden_states

    model = SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=False,
            moe_paged_stash=False,
            sequence_parallel=True,
            multi_latent_attention=False,
            moe_n_hash_layers=0,
            mtp_num_layers=None,
        ),
        pre_process=True,
        post_process=False,
        embedding=FakeEmbedding(),
        decoder=FakeDecoder(),
        pg_collection=SimpleNamespace(tp=tp_group),
        position_embedding_type=None,
        share_embeddings_and_output_weights=False,
        mtp_process=False,
    )
    input_ids = torch.arange(8).view(1, 8)
    output = HybridModel.forward(
        model,
        input_ids=input_ids,
        position_ids=input_ids,
        attention_mask=None,
        padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
    )

    assert output.shape == (4, 1, 8)
    assert scatter_calls == [(torch.float32, tp_group), (torch.bool, tp_group)]


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
    def test_noop_without_packed_seq_params(self):
        """No-ops on non-THD kwargs (SBHD path)."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'hidden_states': torch.randn(10, 1, 256, device="cuda")}
        keys = set(kw.keys())
        layer._decompose_packed_seq_params_to_kwargs(kw)
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


def _make_balanced_dynamic_pack_config(monkeypatch, **overrides):
    """Build a minimal valid opt-in config while isolating optional backend probes."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
    kwargs = dict(
        context_parallel_size=2,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=128,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=8,
        csa_compress_ratios=[4, 4, 4, 4],
        csa_dense_mode=False,
        dsa_kernel_backend="none",
        dsa_cp_balance_indexer=True,
        dsa_cp_balance_indexer_graph_dynamic_packs=True,
        cuda_graph_impl="transformer_engine",
        cuda_graph_dynamic_microbatches=True,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return _make_config(**kwargs)


@pytest.mark.internal
@pytest.mark.parametrize("virtual_pipeline_size", [None, 2])
def test_balanced_dynamic_packs_opt_in_allows_attention_graph_with_pp_vpp(
    monkeypatch, virtual_pipeline_size
):
    config = _make_balanced_dynamic_pack_config(
        monkeypatch, virtual_pipeline_model_parallel_size=virtual_pipeline_size
    )

    assert config.pipeline_model_parallel_size == 2
    assert config.virtual_pipeline_model_parallel_size == virtual_pipeline_size
    assert config.dsa_cp_balance_indexer_graph_dynamic_packs


@pytest.mark.internal
def test_balanced_dynamic_packs_allows_verified_safe_row_limit_boundary(monkeypatch):
    """Each half-call may contain exactly 32768 rows; only larger calls are unsafe."""
    config = _make_balanced_dynamic_pack_config(monkeypatch, max_seqlen_per_dp_cp_rank=2 * 32768)

    assert config.max_seqlen_per_dp_cp_rank // 2 == 32768


@pytest.mark.internal
def test_balanced_static_pack_graph_keeps_pp_rejection(monkeypatch):
    with pytest.raises(ValueError, match="graph_dynamic_packs=True"):
        _make_balanced_dynamic_pack_config(
            monkeypatch, dsa_cp_balance_indexer_graph_dynamic_packs=False
        )


@pytest.mark.internal
def test_balanced_static_pack_pp_allows_graph_scope_outside_attention(monkeypatch):
    config = _make_balanced_dynamic_pack_config(
        monkeypatch, dsa_cp_balance_indexer_graph_dynamic_packs=False, cuda_graph_modules=["mlp"]
    )

    assert config.pipeline_model_parallel_size == 2
    assert not config.dsa_cp_balance_indexer_graph_dynamic_packs


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dsa_cp_balance_indexer": False}, "requires dsa_cp_balance_indexer=True"),
        ({"cuda_graph_impl": "local"}, "requires cuda_graph_impl='transformer_engine'"),
        ({"cuda_graph_impl": "full_iteration"}, "requires cuda_graph_impl='transformer_engine'"),
        ({"cuda_graph_modules": ["mlp"]}, "requires CUDA graph capture to include attention"),
        (
            {"sequence_packing_scheduler": "default_dynamic_cp"},
            "requires sequence_packing_scheduler='dp_balanced'",
        ),
        (
            {"dynamic_context_parallel": True},
            "Dynamic context parallelism requires sequence_packing_scheduler=default_dynamic_cp",
        ),
        (
            {"cuda_graph_dynamic_microbatches": False},
            "with PP/VPP requires cuda_graph_dynamic_microbatches=True",
        ),
        ({"max_seqlen_per_dp_cp_rank": 127}, "requires an even max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": 0}, "requires a positive max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": -2}, "requires a positive max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": 65538}, "above the verified-safe limit"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not yet support overlap_moe_expert_parallel_comm or delay_wgrad_compute",
        ),
        (
            {"delay_wgrad_compute": True},
            "does not yet support overlap_moe_expert_parallel_comm or delay_wgrad_compute",
        ),
    ],
)
def test_balanced_dynamic_packs_validates_opt_in_contract(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_balanced_dynamic_pack_config(monkeypatch, **overrides)


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
