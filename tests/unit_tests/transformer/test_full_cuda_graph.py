# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import gc
import warnings
from unittest.mock import Mock, patch

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    get_shared_capture_stream,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _reset_full_cuda_graph_state():
    """Drop process-global graph inputs and outputs between unit tests."""
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}


@pytest.fixture(autouse=True)
def reset_full_cuda_graph_state():
    """Keep the full-iteration wrapper's class state isolated per test."""
    _reset_full_cuda_graph_state()
    MTPLossLoggingHelper.tracker = {}
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _reset_full_cuda_graph_state()
    MTPLossLoggingHelper.tracker = {}
    Utils.destroy_model_parallel()
    gc.collect()


def test_ddp_grad_accumulators_share_full_cuda_graph_stream():
    """Retained DDP AccumulateGrad nodes must use the full-iteration capture stream."""

    class RetainingDataParallel(torch.nn.Module):
        """Minimal DDP wrapper that retains parameter AccumulateGrad nodes."""

        def __init__(self, *, module, **_):
            super().__init__()
            self.module = module
            self.grad_accumulators = []
            for param in module.parameters():
                expanded_param = param.expand_as(param)
                grad_accumulator = expanded_param.grad_fn.next_functions[0][0]
                grad_accumulator.register_hook(lambda *_: None)
                self.grad_accumulators.append(grad_accumulator)

        def forward(self, inputs):
            """Run the wrapped module."""
            return self.module(inputs)

    assert torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch is not None
    model = torch.nn.Linear(4, 4, device="cuda")
    model.config = Mock(cuda_graph_impl="full_iteration")
    ddp_config = Mock(
        num_buckets=None,
        bucket_size=1024,
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
    )
    process_groups = Mock()
    with patch(
        "megatron.training.models.dist_utils.DistributedDataParallel", RetainingDataParallel
    ):
        wrapped_model = _ddp_wrap(
            [model],
            data_parallel_random_init=False,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            pg_collection=process_groups,
        )[0]

    capture_stream = get_shared_capture_stream()
    current_stream = torch.cuda.current_stream()
    capture_stream.wait_stream(current_stream)
    static_input = torch.ones(2, 4, device="cuda")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with torch.cuda.stream(capture_stream):
            wrapped_model(static_input).sum().backward()
            wrapped_model.zero_grad(set_to_none=False)

            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph, stream=capture_stream):
                wrapped_model(static_input).sum().backward()

    cuda_graph.replay()
    torch.cuda.synchronize()

    stream_mismatch_warnings = [
        warning
        for warning in caught_warnings
        if "AccumulateGrad node's stream does not match" in str(warning.message)
    ]
    assert not stream_mismatch_warnings
    assert all(param.grad is not None for param in wrapped_model.parameters())


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


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="full-iteration MTP test requires TransformerEngine RNG tracking",
)
def test_mtp_bshd_training_with_full_cuda_graph():
    """Capture and replay a full forward-backward iteration containing MTP BSHD rolls."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    sequence_length = 16
    micro_batch_size = 1
    vocab_size = 128
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        cuda_graph_impl="full_iteration",
        cuda_graph_modules=[],
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=layer_spec, use_transformer_engine=True
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        position_embedding_type="rope",
    ).cuda()
    model.train()
    assert model.mtp_process

    tokens = torch.arange(sequence_length, dtype=torch.int64).repeat(micro_batch_size, 1)
    batch = {
        'tokens': tokens,
        'labels': (tokens + 1) % vocab_size,
        'loss_mask': torch.ones_like(tokens, dtype=torch.float32),
        'position_ids': torch.arange(sequence_length, dtype=torch.int64).repeat(
            micro_batch_size, 1
        ),
    }

    def forward_step_func(data_iterator, model):
        microbatch = next(data_iterator)
        output = model(
            input_ids=microbatch['tokens'],
            position_ids=microbatch['position_ids'],
            attention_mask=None,
            labels=microbatch['labels'],
            loss_mask=microbatch['loss_mask'],
            packed_seq_params=None,
        )

        def loss_func(output_tensor):
            loss_mask = microbatch['loss_mask']
            loss = (output_tensor.float() * loss_mask).sum() / loss_mask.sum()
            return loss, {'lm loss': loss.detach()}

        return output, loss_func

    forward_backward_func = FullCudaGraphWrapper(
        get_forward_backward_func(), cuda_graph_warmup_steps=1
    )
    losses = []
    for _ in range(3):
        model.zero_grad(set_to_none=False)
        reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[iter([batch])],
            model=[model],
            num_microbatches=1,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        assert len(reduced) == 1
        losses.append(reduced[0]['lm loss'].detach().clone())

    assert FullCudaGraphWrapper.cuda_graph['training'] is not None
    assert all(torch.isfinite(loss) for loss in losses)
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(losses[2], losses[0], rtol=1e-3, atol=1e-3)

    mtp_grads = [
        parameter.grad for parameter in model.mtp.parameters() if parameter.grad is not None
    ]
    assert mtp_grads
    assert all(torch.isfinite(grad).all() for grad in mtp_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in mtp_grads)
