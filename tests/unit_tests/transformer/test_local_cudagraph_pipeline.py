# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.schedules import custom_backward, deallocate_output_tensor
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    _CudaGraphRunner,
    create_cudagraphs,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def test_make_pipeline_output_viewless_guard_and_passthrough():
    """Exercise the helper's guard, non-tensor passthrough, and autograd preservation."""
    runner = object.__new__(_CudaGraphRunner)

    runner.is_last_layer = False
    runner.deallocate_pipeline_outputs = True
    sentinel = torch.randn(4)
    assert runner._make_pipeline_output_viewless(sentinel) is sentinel

    runner.is_last_layer = True
    runner.deallocate_pipeline_outputs = False
    assert runner._make_pipeline_output_viewless(sentinel) is sentinel

    runner.deallocate_pipeline_outputs = True
    view_base = torch.randn(4, 4, requires_grad=True)
    tensor_view = view_base[0]
    out = runner._make_pipeline_output_viewless((tensor_view, None))

    assert out[0]._base is None
    assert torch.equal(out[0], tensor_view)
    assert out[0].data_ptr() == tensor_view.data_ptr()
    assert out[1] is None

    out[0].sum().backward()
    assert view_base.grad is not None
    expected_grad = torch.zeros_like(view_base)
    expected_grad[0] = 1
    assert torch.equal(view_base.grad, expected_grad)


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestLocalCudagraphPipelineOutput:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    def test_record_and_replay_outputs_support_pipeline_deallocation(self):
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            cuda_graph_impl="local",
            cuda_graph_modules=[],
            cuda_graph_warmup_steps=1,
            deallocate_pipeline_outputs=True,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        block = TransformerBlock(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            post_process=False,
            post_layer_norm=False,
        ).cuda()
        block.train()
        for param in block.parameters():
            param.main_grad = torch.zeros_like(param)

        sequence_length = 32
        hidden_states = torch.randn(
            (sequence_length, 1, config.hidden_size), device="cuda", requires_grad=True
        )
        attention_mask = torch.ones(
            (1, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
        )

        record_out = block(hidden_states=hidden_states, attention_mask=attention_mask)
        expected_shape = record_out.shape
        assert torch.isfinite(record_out).all()
        assert record_out._base is None

        record_grad = torch.ones_like(record_out)
        deallocate_output_tensor(record_out, deallocate_pipeline_outputs=True)
        custom_backward(record_out, record_grad)
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        hidden_states.grad = None
        create_cudagraphs()

        replay_out = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert replay_out.shape == expected_shape
        assert torch.isfinite(replay_out).all()
        assert replay_out._base is None

        replay_grad = torch.ones_like(replay_out)
        deallocate_output_tensor(replay_out, deallocate_pipeline_outputs=True)
        custom_backward(replay_out, replay_grad)
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

        for layer in block.layers:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph
        torch.cuda.synchronize()
