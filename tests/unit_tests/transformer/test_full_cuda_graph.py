# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    clone_tensors_in_struct,
    copy_tensors_in_struct,
)
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_clone_tensors_in_struct_replaces_none_tensor_slots():
    target = copy_tensors_in_struct(
        {
            "tokens": torch.zeros(2, dtype=torch.int64),
            "optional": None,
            "nested": {"value": None},
        }
    )

    updated = clone_tensors_in_struct(
        target,
        {
            "tokens": torch.ones(2, dtype=torch.int64),
            "optional": torch.arange(2, dtype=torch.int64),
            "nested": {"value": torch.full((2,), 3.0)},
        },
    )

    assert updated is target
    assert torch.equal(updated["tokens"].cpu(), torch.ones(2, dtype=torch.int64))
    assert torch.equal(updated["optional"].cpu(), torch.arange(2, dtype=torch.int64))
    assert torch.equal(updated["nested"]["value"].cpu(), torch.full((2,), 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_buffer_loader_keeps_buffer_ids_separate():
    StaticBufferLoader.static_buffers = {'training': {}, 'validation': {}}
    loader = StaticBufferLoader()

    first = loader({"tokens": torch.zeros(1), "optional": None}, "training", 0, buffer_id=0)
    second = loader(
        {"tokens": torch.ones(1), "optional": torch.ones(1)}, "training", 0, buffer_id=1
    )

    assert first is StaticBufferLoader.static_buffers["training"][0][0]
    assert second is StaticBufferLoader.static_buffers["training"][1][0]
    assert first is not second
    assert first["optional"] is None
    assert torch.equal(second["optional"].cpu(), torch.ones(1))


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
