# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.training.arguments import parse_args, validate_args


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
    [("max", 18, False), ("max", 24, True), (6, 24, False), (8, 24, True)],
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
        ModelParallelConfig(sequence_parallel=True, **kwargs)
    else:
        with pytest.raises(AssertionError, match="sequence-parallel TP divisibility"):
            ModelParallelConfig(sequence_parallel=True, **kwargs)
