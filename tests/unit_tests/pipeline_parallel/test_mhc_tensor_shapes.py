# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.schedules import _get_pipeline_hidden_size, get_tensor_shapes
from megatron.core.transformer.transformer_config import TransformerConfig


class FakeProcessGroup:
    """Small process-group stand-in for schedule shape tests."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def make_config(**overrides):
    values = {
        'hidden_size': 64,
        'enable_hyper_connections': True,
        'num_residual_streams': 4,
        'sequence_parallel': False,
        'variable_seq_lengths': False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def get_shapes(config, *, pp_rank=0, pp_size=2, is_recv=True, tp_size=1, cp_size=1):
    return get_tensor_shapes(
        seq_length=32,
        micro_batch_size=2,
        decoder_seq_length=None,
        config=config,
        tp_group=FakeProcessGroup(0, tp_size),
        cp_group=FakeProcessGroup(0, cp_size),
        pp_group=FakeProcessGroup(pp_rank, pp_size),
        is_recv=is_recv,
    )


@pytest.mark.parametrize(
    "pp_rank,is_recv,hidden_size",
    [
        (0, True, 64),
        (0, False, 256),
        (1, True, 256),
        (1, False, 256),
        (2, True, 256),
        (2, False, 256),
        (3, True, 256),
        (3, False, 64),
    ],
)
def test_non_interleaved_mhc_uses_expanded_shape_between_stages(pp_rank, is_recv, hidden_size):
    shapes = get_shapes(make_config(), pp_rank=pp_rank, pp_size=4, is_recv=is_recv)

    assert shapes == [(32, 2, hidden_size)]


@pytest.mark.parametrize("enable_hyper_connections", [False, True])
@pytest.mark.parametrize("is_recv", [False, True])
def test_single_pipeline_stage_keeps_model_hidden_size(enable_hyper_connections, is_recv):
    config = make_config(enable_hyper_connections=enable_hyper_connections)

    assert get_shapes(config, pp_size=1, is_recv=is_recv) == [(32, 2, 64)]


def test_non_mhc_pipeline_keeps_model_hidden_size():
    config = make_config(enable_hyper_connections=False)

    for rank in range(4):
        assert get_shapes(config, pp_rank=rank, pp_size=4, is_recv=True) == [(32, 2, 64)]
        assert get_shapes(config, pp_rank=rank, pp_size=4, is_recv=False) == [(32, 2, 64)]


def test_legacy_get_tensor_shapes_call_without_pp_group_is_unchanged():
    config = make_config()

    shapes = get_tensor_shapes(
        seq_length=32,
        micro_batch_size=2,
        decoder_seq_length=None,
        config=config,
        tp_group=FakeProcessGroup(0, 1),
        cp_group=FakeProcessGroup(0, 1),
    )

    assert shapes == [(32, 2, 64)]


@pytest.mark.parametrize(
    "enabled,pp_size,hidden_size", [(False, 1, 64), (False, 4, 64), (True, 1, 64), (True, 4, 256)]
)
def test_interleaved_pipeline_uses_one_shape_for_all_active_edges(enabled, pp_size, hidden_size):
    config = make_config(enable_hyper_connections=enabled)

    assert _get_pipeline_hidden_size(config, pp_group=FakeProcessGroup(0, pp_size)) == hidden_size


def test_mhc_shape_preserves_context_and_sequence_parallel_scaling():
    config = make_config(sequence_parallel=True)

    shapes = get_shapes(config, pp_rank=1, pp_size=4, is_recv=True, tp_size=2, cp_size=4)

    assert shapes == [(4, 2, 256)]


def test_mhc_shape_uses_decoder_sequence_length():
    config = make_config()

    shapes = get_tensor_shapes(
        seq_length=32,
        micro_batch_size=2,
        decoder_seq_length=48,
        config=config,
        tp_group=FakeProcessGroup(0, 1),
        cp_group=FakeProcessGroup(0, 2),
        pp_group=FakeProcessGroup(1, 4),
        is_recv=True,
    )

    assert shapes == [(24, 2, 256)]


def test_variable_sequence_length_shape_is_unchanged():
    config = make_config(variable_seq_lengths=True)

    assert get_shapes(config, pp_rank=1, pp_size=4, is_recv=True) == [()]


def test_native_mhc_transformer_config_drives_pipeline_shape():
    config = TransformerConfig(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        enable_hyper_connections=True,
        num_residual_streams=4,
        use_cpu_initialization=True,
    )

    assert get_shapes(config, pp_rank=0, pp_size=2, is_recv=False) == [(32, 2, 256)]
