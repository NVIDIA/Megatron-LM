# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_layer_with_transformer_engine_spec()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_block = self.parallel_transformer_block
        assert isinstance(parallel_transformer_block, TransformerBlock)
        num_weights = sum([p.numel() for p in parallel_transformer_block.parameters()])
        assert num_weights == 100096
        assert parallel_transformer_block.num_layers_per_pipeline_rank == 2
        assert len(parallel_transformer_block.layers) == 2
        layer_0: TransformerLayer = parallel_transformer_block._get_layer(0)
        assert layer_0.layer_number == 1
        layer_1: TransformerLayer = parallel_transformer_block._get_layer(1)
        assert layer_1.layer_number == 2

    def test_gpu_forward(self):
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config

        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_gpu_forward_full_checkpoint(self):
        self._run_full_checkpoint_test(fp8=None)

    def test_gpu_forward_full_checkpoint_fp8(self):
        self._run_full_checkpoint_test(fp8="e4m3")

    def test_gpu_forward_selective_checkpoint(self):
        self._run_selective_checkpoint_test(fp8=None)

    def test_gpu_forward_selective_checkpoint_fp8(self):
        self._run_selective_checkpoint_test(fp8="e4m3")

    def _run_full_checkpoint_test(self, fp8):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'full'
        config.recompute_method = 'block'
        config.fp8 = fp8
        config.recompute_num_layers = config.num_layers
        full_transformer_block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec()
        )
        assert full_transformer_block.config.recompute_granularity == 'full'
        assert full_transformer_block.config.recompute_method == 'block'
        assert full_transformer_block.config.fp8 == fp8

        sequence_length = 32
        micro_batch_size = 2
        full_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = full_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def _run_selective_checkpoint_test(self, fp8):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'selective'
        config.fp8 = fp8
        selective_transformer_block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec()
        )
        assert selective_transformer_block.config.recompute_granularity == 'selective'
        assert selective_transformer_block.checkpoint_core_attention
        assert selective_transformer_block.config.fp8 == fp8

        sequence_length = 32
        micro_batch_size = 2
        selective_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = selective_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size


class TestPipelineParallelTransformerBlock:
    @pytest.mark.parametrize(
        "num_layers, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, "
        "include_embedding_in_pipeline_split, include_loss_in_pipeline_split, "
        "first_pipeline_num_layers, last_pipeline_num_layers, should_assert_error",
        [
            # Last pipeline stage has specified layers
            (60, 5, None, False, False, None, 4, False),
            # Uneven PP 6*[8]+[6]+[6]=60
            (60, 8, None, False, False, 6, 6, False),
            # Even PP
            (64, 4, None, False, False, None, None, False),
            # Even VPP
            (64, 4, 8, False, False, None, None, False),
            # First pipeline stage has specified layers
            # Should distribute remaining layers evenly among other stages
            (60, 6, None, False, False, 5, None, False),
            # Uneven distribution leading to assertion error
            (101, 8, None, False, False, 13, 13, True),
            # Include embedding in pipeline split without virtual PP
            (63, 4, None, True, False, None, None, False),
            # Include loss in pipeline split without virtual PP
            (63, 4, None, False, True, None, None, False),
            # Include embedding and loss in pipeline split without virtual PP
            (62, 4, None, True, True, None, None, False),
            # Include embedding and loss with virtual PP
            (62, 4, 2, True, True, None, None, False),
            # num_layers not divisible by pipeline size without embedding/loss
            (65, 4, None, False, False, None, None, True),
            # num_layers not divisible by pipeline size with embedding/loss
            (65, 4, None, True, True, None, None, True),
            # Uneven distribution with specified first pipeline layers causing error
            (61, 4, None, False, False, 12, None, True),
            # Too few layers for the number of pipeline stages
            (2, 4, None, False, False, None, None, True),
            # Uneven PP with embedding included (should assert per code)
            (60, 6, None, True, False, 5, 5, True),
            # Virtual PP where num_layers not divisible by total virtual stages
            (50, 2, 7, False, False, None, None, True),
            # Edge case where num_layers per virtual rank is zero
            (4, 4, 4, False, False, None, None, True),
        ],
    )
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_layer_builder(
        self,
        num_layers,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        include_embedding_in_pipeline_split,
        include_loss_in_pipeline_split,
        first_pipeline_num_layers,
        last_pipeline_num_layers,
        should_assert_error,
    ):
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        )
        context = (
            pytest.raises((AssertionError, ValueError)) if should_assert_error else nullcontext()
        )
        with context:
            transformer_config = TransformerConfig(
                num_layers=num_layers,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
                include_embedding_in_pipeline_split=include_embedding_in_pipeline_split,
                include_loss_in_pipeline_split=include_loss_in_pipeline_split,
                first_pipeline_num_layers=first_pipeline_num_layers,
                last_pipeline_num_layers=last_pipeline_num_layers,
                pipeline_dtype=torch.bfloat16,
                hidden_size=128,
                num_attention_heads=16,
            )
            total_build_layers = 0
            for i in range(pipeline_model_parallel_size):
                parallel_state.set_pipeline_model_parallel_rank(i)
                if virtual_pipeline_model_parallel_size is not None:
                    for j in range(virtual_pipeline_model_parallel_size):
                        parallel_state.set_virtual_pipeline_model_parallel_rank(j)
                        num_layers_to_build = get_num_layers_to_build(transformer_config)
                        total_build_layers += num_layers_to_build
                else:
                    num_layers_to_build = get_num_layers_to_build(transformer_config)
                    total_build_layers += num_layers_to_build
        if not should_assert_error:
            assert (
                total_build_layers == num_layers
            ), f"total build layers {total_build_layers} should be equal to num_layers {num_layers}"
        parallel_state.set_pipeline_model_parallel_world_size(None)
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)
