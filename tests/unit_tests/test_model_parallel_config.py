# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest
import torch

import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args, validate_args


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (False, None),
        (True, ["ag", "rs"]),
        ([], ["ag", "rs"]),
        (["rs", "ag", "rs"], ["ag", "rs"]),
    ],
)
def test_gtp_nccl_ub_is_normalized(value, expected):
    config = ModelParallelConfig(gtp_remat_nccl_ub=value, gtp_expert_remat_nccl_ub=value)

    assert config.gtp_remat_nccl_ub == expected
    assert config.gtp_expert_remat_nccl_ub == expected


def test_dynamic_cp_selects_default_scheduler_and_variable_lengths():
    config = ModelParallelConfig(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=4096)

    assert config.sequence_packing_scheduler == "default_dynamic_cp"
    assert config.variable_seq_lengths is True


def test_dynamic_cp_rejects_non_dynamic_scheduler():
    with pytest.raises(ValueError, match="requires.*default_dynamic_cp"):
        ModelParallelConfig(
            dynamic_context_parallel=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=4096,
        )


def test_default_dynamic_cp_scheduler_requires_dynamic_cp():
    with pytest.raises(ValueError, match="requires.*dynamic_context_parallel=True"):
        ModelParallelConfig(
            sequence_packing_scheduler="default_dynamic_cp", max_seqlen_per_dp_cp_rank=4096
        )


def test_dynamic_cp_rejects_deterministic_mode():
    with pytest.raises(ValueError, match="does not support deterministic_mode"):
        ModelParallelConfig(
            dynamic_context_parallel=True, deterministic_mode=True, max_seqlen_per_dp_cp_rank=4096
        )


def test_static_packing_preserves_deterministic_mode():
    config = ModelParallelConfig(
        sequence_packing_scheduler="dp_balanced",
        deterministic_mode=True,
        max_seqlen_per_dp_cp_rank=4096,
    )

    assert config.deterministic_mode
    assert not config.dynamic_context_parallel


@pytest.mark.parametrize("alignment", [None, "32", "max"])
def test_packed_padding_cli_normalizes_alignment(monkeypatch, alignment):
    argv = ["test_model_parallel_config.py"]
    if alignment is not None:
        argv += ["--pad-packed-seq-alignment", alignment]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    config = ModelParallelConfig(
        pad_packed_seq_alignment=args.pad_packed_seq_alignment, max_seqlen_per_dp_cp_rank=128
    )
    assert config.pad_packed_seq_alignment == (32 if alignment == "32" else alignment)


def _te_cross_entropy_training_args(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    # Let validate_args derive a global batch size that is valid for the
    # active data-parallel size in distributed unit-test jobs.
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = 'te'
    return args


def test_te_cross_entropy_loss_fusion_is_allowed_in_model_parallel_config():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


def test_te_cross_entropy_loss_fusion_is_allowed_by_training_args(monkeypatch):
    args = _te_cross_entropy_training_args(monkeypatch)
    monkeypatch.setattr('megatron.training.arguments.is_te_min_version', lambda version: True)

    validated_args = validate_args(args)

    assert validated_args.cross_entropy_loss_fusion
    assert validated_args.cross_entropy_fusion_impl == 'te'


def test_te_cross_entropy_loss_fusion_requires_te_2_19(monkeypatch):
    args = _te_cross_entropy_training_args(monkeypatch)
    monkeypatch.setattr('megatron.training.arguments.is_te_min_version', lambda version: False)

    with pytest.raises(AssertionError, match="disabled due to stability issues"):
        validate_args(args)


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize(
    "alignment,capacity,valid",
    [("max", 18, False), ("max", 24, True), (6, 24, True), (8, 24, True)],
)
def test_packed_padding_preserves_sequence_parallel_divisibility(
    cp_size, alignment, capacity, valid
):
    kwargs = dict(
        tensor_model_parallel_size=4,
        context_parallel_size=cp_size,
        sequence_packing_scheduler="dp_balanced",
        pad_packed_seq_alignment=alignment,
        max_seqlen_per_dp_cp_rank=capacity,
    )
    # Without sequence parallelism, TP does not constrain the padded token length.
    ModelParallelConfig(**kwargs)
    if valid:
        config = ModelParallelConfig(sequence_parallel=True, **kwargs)
        from megatron.core.datasets.data_schedule_utils import get_packed_sequence_alignment

        effective_alignment, _ = get_packed_sequence_alignment(config, 4)
        assert effective_alignment % 4 == 0
        assert capacity % effective_alignment == 0
    else:
        with pytest.raises(AssertionError, match="sequence-parallel TP divisibility"):
            ModelParallelConfig(sequence_parallel=True, **kwargs)


def _fixed_p2p_shape_config(config_cls=ModelParallelConfig, **overrides):
    kwargs = dict(
        pipeline_p2p_fixed_shape=True,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=2048,
        pad_packed_seq_alignment="max",
    )
    kwargs.update(overrides)
    return config_cls(**kwargs)


@pytest.mark.parametrize("alignment", ["max", 2048])
def test_pipeline_p2p_fixed_shape_accepts_full_length_padding(alignment):
    config = _fixed_p2p_shape_config(pad_packed_seq_alignment=alignment)

    assert config.pipeline_p2p_fixed_shape
    assert config.variable_seq_lengths


def test_pipeline_p2p_fixed_shape_requires_sequence_packing():
    with pytest.raises(ValueError, match="requires a sequence_packing_scheduler"):
        _fixed_p2p_shape_config(sequence_packing_scheduler=None)


@pytest.mark.parametrize("alignment", [None, 128])
def test_pipeline_p2p_fixed_shape_requires_full_length_padding(alignment):
    with pytest.raises(ValueError, match="requires pad_packed_seq_alignment='max'"):
        _fixed_p2p_shape_config(pad_packed_seq_alignment=alignment)


def test_pipeline_p2p_fixed_shape_rejects_dynamic_context_parallel():
    with pytest.raises(ValueError, match="not supported with dynamic_context_parallel"):
        _fixed_p2p_shape_config(sequence_packing_scheduler=None, dynamic_context_parallel=True)


def test_pipeline_p2p_fixed_shape_requires_tp_divisible_length_with_sequence_parallel():
    with pytest.raises(ValueError, match="divisible by tensor_model_parallel_size"):
        _fixed_p2p_shape_config(
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            max_seqlen_per_dp_cp_rank=18,
            pad_packed_seq_alignment=18,
        )


def test_pipeline_p2p_fixed_shape_warns_when_mtp_standalone():
    with pytest.warns(UserWarning, match="no effect when mtp_standalone"):
        _fixed_p2p_shape_config(mtp_standalone=True)


def test_pipeline_p2p_fixed_shape_warns_for_layout_derived_standalone_mtp(monkeypatch):
    # The flexible layout derives mtp_standalone after the base config has been validated.
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _: True)
    with pytest.warns(UserWarning, match="no effect when mtp_standalone"):
        config = _fixed_p2p_shape_config(
            TransformerConfig,
            num_layers=3,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=4,
            pipeline_dtype=torch.bfloat16,
            pipeline_model_parallel_layout="Et|t|tm|L",
            mtp_num_layers=1,
        )

    assert config.mtp_standalone
