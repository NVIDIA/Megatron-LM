# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, _has_megatron_fsdp_v2
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


class _FSDPWrapper(torch.nn.Module):
    def __init__(self, use_megatron_fsdp_v2):
        super().__init__()
        self.ddp_config = SimpleNamespace(use_megatron_fsdp_v2=use_megatron_fsdp_v2)


def test_has_megatron_fsdp_v2_walks_nested_model_list():
    root = torch.nn.Module()
    v1_child = _FSDPWrapper(use_megatron_fsdp_v2=False)
    v2_child = _FSDPWrapper(use_megatron_fsdp_v2=True)

    assert not _has_megatron_fsdp_v2([root, v1_child])
    assert _has_megatron_fsdp_v2([root, torch.nn.Sequential(v2_child)])


def test_v2_fsdp_forward_only_stays_eager(mocker):
    mocker.patch("megatron.core.full_cuda_graph.torch.cuda.Stream")
    eager_result = object()
    forward_backward_func = mocker.Mock(return_value=eager_result)
    wrapper = FullCudaGraphWrapper(forward_backward_func)
    data_read = mocker.patch.object(wrapper, "data_read")
    data_iterator = [iter(())]

    result = wrapper(
        model=[torch.nn.Sequential(_FSDPWrapper(use_megatron_fsdp_v2=True))],
        data_iterator=data_iterator,
        num_microbatches=1,
        seq_length=None,
        forward_only=True,
    )

    assert result is eager_result
    data_read.assert_not_called()
    assert forward_backward_func.call_args.kwargs["data_iterator"] is data_iterator


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_forward_backward_func_with_full_cuda_graph(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)

    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    # Wrapping the forward_backward_func with FullCudaGraphWrapper enables full iteration CUDA graphs.
    forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    num_microbatches = 4

    # CUDA graph warmup
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    # CUDA graph capture and replay
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()
