# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for static-shape THD Transformer Engine CUDA graphs."""

import pytest
import torch

from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _make_cu(seqlens, device="cpu"):
    lengths = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), lengths.cumsum(0))
    )


def _make_packed_seq_params(seqlens, device="cpu"):
    cu_seqlens = _make_cu(seqlens, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
        local_cp_size=1,
        pad_between_seqs=False,
    )


@pytest.mark.parametrize(
    "cuda_graph_static,expected_max_num_seqs", [(False, None), (True, 32)]
)
def test_pad_to_max_resolves_static_cu_capacity(
    cuda_graph_static, expected_max_num_seqs
):
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=8192,
        thd_max_packed_sequences=32,
        cuda_graph_static=cuda_graph_static,
    )

    assert alignment is None
    assert target_len == 8192
    assert max_num_seqs == expected_max_num_seqs


def test_static_padding_appends_dummy_sequence_and_fixes_all_shapes():
    packed_seq_params = _make_packed_seq_params([100, 50, 30])
    packed_seq_params.cp_partition_mode = "contiguous"
    total_tokens = 180
    tokens = torch.arange(total_tokens).reshape(1, -1)

    padded = pad_sequence_for_thd(
        tokens,
        tokens.clone(),
        torch.ones(1, total_tokens),
        torch.arange(total_tokens).reshape(1, -1),
        packed_seq_params,
        target_len=256,
        max_num_seqs=8,
    )
    padded_tokens, labels, loss_mask, position_ids, params, padding_mask = padded

    for tensor in (padded_tokens, labels, loss_mask, position_ids):
        assert tensor.shape == (1, 256)
    for cu_seqlens in (
        params.cu_seqlens_q,
        params.cu_seqlens_kv,
        params.cu_seqlens_q_padded,
        params.cu_seqlens_kv_padded,
    ):
        assert cu_seqlens.shape == (9,)
        assert cu_seqlens.tolist() == [0, 100, 150, 180, 256, 256, 256, 256, 256]
    assert params.max_seqlen_q == 256
    assert params.max_seqlen_kv == 256
    assert params.pad_between_seqs is False
    assert params.local_cp_size == packed_seq_params.local_cp_size
    assert params.cp_partition_mode == "contiguous"
    assert padding_mask.shape == (1, 256)
    assert not padding_mask[0, :total_tokens].any()
    assert padding_mask[0, total_tokens:].all()


def test_alignment_padding_preserves_metadata_without_dummy_sequence():
    packed_seq_params = _make_packed_seq_params([50, 30])
    original_cu_seqlens = packed_seq_params.cu_seqlens_q.clone()

    padded_tokens, _, _, _, params, padding_mask = pad_sequence_for_thd(
        torch.ones(1, 80),
        None,
        None,
        None,
        packed_seq_params,
        alignment=64,
        pad_by_appending_dummy_seq=False,
    )

    assert padded_tokens.shape == (1, 128)
    assert torch.equal(params.cu_seqlens_q, original_cu_seqlens)
    assert params.max_seqlen_q == 50
    assert params.pad_between_seqs is False
    assert not padding_mask[0, :80].any()
    assert padding_mask[0, 80:].all()


def test_padding_merges_existing_mask_with_tail():
    packed_seq_params = _make_packed_seq_params([4, 4])
    existing_mask = torch.tensor(
        [[False, False, False, True, False, False, True, True]]
    )

    *_, padding_mask = pad_sequence_for_thd(
        torch.ones(1, 8),
        None,
        None,
        None,
        packed_seq_params,
        target_len=10,
        max_num_seqs=4,
        padding_mask=existing_mask,
    )

    assert padding_mask.tolist() == [
        [False, False, False, True, False, False, True, True, True, True]
    ]


def test_metadata_only_padding_uses_explicit_cp_geometry():
    packed_seq_params = _make_packed_seq_params([15])
    packed_seq_params.cu_seqlens_q_padded = torch.tensor(
        [0, 16], dtype=torch.int32
    )
    packed_seq_params.cu_seqlens_kv_padded = (
        packed_seq_params.cu_seqlens_q_padded.clone()
    )
    existing_mask = torch.tensor([[False, False, False, True]])

    *_, params, padding_mask = pad_sequence_for_thd(
        None,
        None,
        None,
        None,
        packed_seq_params,
        target_len=4,
        padding_mask=existing_mask,
        cp_size=4,
        cp_rank=3,
    )

    assert torch.equal(padding_mask, existing_mask)
    assert params.cu_seqlens_q.tolist() == [0, 15, 16]
    assert params.cu_seqlens_q_padded.tolist() == [0, 16, 16]


def test_dynamic_slot_liveness_for_pp_and_vpp_orders():
    pp_order = [1, 1, -1, 1, -1, 1, -1, -1]
    vpp_order = [
        1,
        1,
        1,
        2,
        2,
        2,
        -2,
        1,
        -2,
        1,
        -2,
        2,
        -1,
        2,
        -1,
        -1,
        -2,
        -2,
        -1,
        -1,
    ]

    assert TECudaGraphHelper._get_required_num_microbatch_slots_from_order(
        pp_order, 1
    ) == 2
    assert TECudaGraphHelper._get_required_num_microbatch_slots_from_order(
        vpp_order, 2
    ) == 5


def test_dp_balanced_capture_upper_bound_accounts_for_cp_and_vpp():
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


def test_thd_te_graph_rejects_moe_tp_sp_token_count_mismatch():
    with pytest.raises(
        ValueError,
        match="tensor_parallel_size > 1 with sequence_parallel",
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            num_moe_experts=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            max_seqlen_per_dp_cp_rank=128,
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
            cuda_graph_impl="transformer_engine",
        )


def test_local_graph_does_not_require_static_thd_padding_contract():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        max_seqlen_per_dp_cp_rank=128,
        sequence_packing_scheduler="dp_balanced",
        cuda_graph_impl="local",
    )

    assert config.pad_packed_seq_alignment is None


@pytest.mark.parametrize(
    "cuda_graph_kwargs",
    [
        {"cuda_graph_impl": "transformer_engine"},
        {"external_cuda_graph": True},
    ],
)
def test_te_cuda_graph_rejects_mhc_selective_recompute(cuda_graph_kwargs):
    with pytest.raises(
        NotImplementedError,
        match="'mhc' in recompute_modules is not supported",
    ):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            enable_hyper_connections=True,
            num_residual_streams=2,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            **cuda_graph_kwargs,
        )


@pytest.mark.parametrize(
    "unsupported_recompute_kwargs",
    [
        {"cuda_graph_impl": "transformer_engine"},
        {"external_cuda_graph": True},
        {
            "cuda_graph_impl": "local",
            "cuda_graph_modules": ["moe_router"],
            "fine_grained_activation_offloading": True,
            "offload_modules": ["expert_fc1"],
            "num_moe_experts": 4,
        },
    ],
)
def test_mhc_recompute_warning_is_suppressed_when_unsupported(
    recwarn, unsupported_recompute_kwargs
):
    TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        enable_hyper_connections=True,
        num_residual_streams=2,
        recompute_granularity=None,
        recompute_modules=[],
        **unsupported_recompute_kwargs,
    )

    assert not any(
        "Consider adding 'mhc'" in str(warning.message) for warning in recwarn
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_transformer_layer_static_thd_inputs_use_prefixed_contract():
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            max_seqlen_per_dp_cp_rank=128,
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
            thd_max_packed_sequences=8,
            cuda_graph_impl="transformer_engine",
            cp_partition_mode="contiguous",
            bf16=True,
        )
        model_parallel_cuda_manual_seed(42)
        attention_layer_spec = hybrid_stack_spec.submodules.attention_layer
        layer = (
            TransformerLayer(
                config,
                attention_layer_spec.submodules,
                layer_number=1,
            )
            .cuda()
            .bfloat16()
        )

        static_inputs = layer.get_layer_static_inputs(
            seq_length=128, micro_batch_size=4
        )
        packed_seq_params = static_inputs.pop("packed_seq_params")
        assert packed_seq_params.cp_partition_mode == "contiguous"
        _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(
            layer, static_inputs, packed_seq_params
        )
        assert (
            layer._get_te_cuda_graph_packed_seq_params_static_metadata()[
                "cp_partition_mode"
            ]
            == "contiguous"
        )

        assert static_inputs["hidden_states"].shape == (128, 1, 256)
        assert static_inputs["hidden_states"].dtype == torch.bfloat16
        assert static_inputs["padding_mask"].shape == (1, 128)
        assert not static_inputs["padding_mask"].any()
        assert not any(key.startswith("cu_seqlens_") for key in static_inputs)
        flattened_keys = {
            key
            for key in static_inputs
            if key.startswith(CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX)
        }
        assert len(flattened_keys) == 4
        assert all(static_inputs[key].shape == (9,) for key in flattened_keys)
    finally:
        Utils.destroy_model_parallel()
