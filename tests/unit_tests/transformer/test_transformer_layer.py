# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        assert isinstance(parallel_transformer_layer, TransformerLayer)
        assert parallel_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_transformer_layer.parameters()])
        assert num_weights == 1884

    def test_gpu_forward(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states, context = parallel_transformer_layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_chunked_mlp(self):
        with torch.no_grad():

            def test(
                num_layers,
                hidden_size,
                num_attention_heads,
                mlp_chunks_for_prefill,
                hidden_states,
                inference_context,
            ):

                transformer_config = TransformerConfig(
                    num_layers=2,
                    hidden_size=12,
                    num_attention_heads=4,
                    mlp_chunks_for_prefill=4,
                    add_bias_linear=True,
                    use_cpu_initialization=True,
                )
                parallel_transformer_layer = TransformerLayer(
                    transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
                )

                parallel_transformer_layer.cuda()

                hidden_states, context = parallel_transformer_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                )

                return hidden_states, context

            num_layers = 2
            hidden_size = 12
            num_attention_heads = 4

            sequence_length = 32
            micro_batch_size = 2

            # [sequence length, batch size, hidden size]
            input_hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
            input_hidden_states = input_hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            inference_context = StaticInferenceContext(
                max_batch_size=micro_batch_size, max_sequence_length=sequence_length
            )

            outputs = {}

            for mlp_chunks_for_prefill in [1, 4]:
                hidden_states, context = test(
                    num_layers,
                    hidden_size,
                    num_attention_heads,
                    mlp_chunks_for_prefill,
                    input_hidden_states,
                    inference_context,
                )
                assert hidden_states.shape[0] == sequence_length
                assert hidden_states.shape[1] == micro_batch_size
                assert hidden_states.shape[2] == hidden_size

                outputs[mlp_chunks_for_prefill] = (hidden_states, context)

        assert torch.equal(outputs[1][0], outputs[4][0])

    def test_get_layer_offset(self):
        config = self.parallel_transformer_layer.config
        assert get_transformer_layer_offset(config) == 0

    @pytest.mark.parametrize(
        "config_params,expected_offsets",
        [
            # Test case 1: Both first and last stages set (30 layers: 8+6+6+10)
            (
                {
                    "num_layers": 30,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": 8,
                    "num_layers_in_last_pipeline_stage": 10,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-3
                    (0, 1): 15,  # Stage 0, VP 1: layers 15-18
                    (1, 0): 4,  # Stage 1, VP 0: layers 4-6
                    (1, 1): 19,  # Stage 1, VP 1: layers 19-21
                    (2, 0): 7,  # Stage 2, VP 0: layers 7-9
                    (2, 1): 22,  # Stage 2, VP 1: layers 22-24
                    (3, 0): 10,  # Stage 3, VP 0: layers 10-14
                    (3, 1): 25,  # Stage 3, VP 1: layers 25-29
                },
            ),
            # Test case 2: Only first stage set (26 layers: 8+6+6+6)
            (
                {
                    "num_layers": 26,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": 8,
                    "num_layers_in_last_pipeline_stage": None,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-3
                    (0, 1): 13,  # Stage 0, VP 1: layers 13-16
                    (1, 0): 4,  # Stage 1, VP 0: layers 4-6
                    (1, 1): 17,  # Stage 1, VP 1: layers 17-19
                    (2, 0): 7,  # Stage 2, VP 0: layers 7-9
                    (2, 1): 20,  # Stage 2, VP 1: layers 20-22
                    (3, 0): 10,  # Stage 3, VP 0: layers 10-12
                    (3, 1): 23,  # Stage 3, VP 1: layers 23-25
                },
            ),
            # Test case 3: Only last stage set (26 layers: 6+6+6+8)
            (
                {
                    "num_layers": 26,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": None,
                    "num_layers_in_last_pipeline_stage": 8,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-2
                    (0, 1): 13,  # Stage 0, VP 1: layers 13-15
                    (1, 0): 3,  # Stage 1, VP 0: layers 3-5
                    (1, 1): 16,  # Stage 1, VP 1: layers 16-18
                    (2, 0): 6,  # Stage 2, VP 0: layers 6-8
                    (2, 1): 19,  # Stage 2, VP 1: layers 19-21
                    (3, 0): 9,  # Stage 3, VP 0: layers 9-12
                    (3, 1): 22,  # Stage 3, VP 1: layers 22-25
                },
            ),
            # Test case 4: Even distribution (24 layers: 6+6+6+6)
            (
                {
                    "num_layers": 24,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": None,
                    "num_layers_in_last_pipeline_stage": None,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-2
                    (0, 1): 12,  # Stage 0, VP 1: layers 12-14
                    (1, 0): 3,  # Stage 1, VP 0: layers 3-5
                    (1, 1): 15,  # Stage 1, VP 1: layers 15-17
                    (2, 0): 6,  # Stage 2, VP 0: layers 6-8
                    (2, 1): 18,  # Stage 2, VP 1: layers 18-20
                    (3, 0): 9,  # Stage 3, VP 0: layers 9-11
                    (3, 1): 21,  # Stage 3, VP 1: layers 21-23
                },
            ),
        ],
    )
    def test_get_layer_offset_parametrized(self, config_params, expected_offsets):
        """
        Parametrized test for get_transformer_layer_offset with different configurations.
        Tests various combinations of first/last stage settings and virtual pipeline sizes.

        This test verifies that the layer offset calculation correctly handles:
        - Asymmetric pipeline stages (different layer counts per stage)
        - Virtual pipeline parallelism (splitting physical stages into virtual stages)
        - Various combinations of first/last stage configurations

        The expected_offsets dictionary maps (pipeline_rank, vp_stage) tuples to
        the expected starting layer index for that stage combination.
        """

        config = TransformerConfig(
            hidden_size=512, num_attention_heads=8, use_cpu_initialization=True, **config_params
        )

        for (pipeline_rank, vp_stage), expected_offset in expected_offsets.items():
            original_get_pipeline_rank = parallel_state.get_pipeline_model_parallel_rank
            parallel_state.get_pipeline_model_parallel_rank = lambda: pipeline_rank

            try:
                actual_offset = get_transformer_layer_offset(config, vp_stage)
                assert actual_offset == expected_offset, (
                    f"Expected offset {expected_offset} for pipeline rank {pipeline_rank}, "
                    f"VP stage {vp_stage}, but got {actual_offset}"
                )
            finally:
                parallel_state.get_pipeline_model_parallel_rank = original_get_pipeline_rank

    @pytest.mark.parametrize('order', ['tp-pp-dp', 'tp-dp-pp'])
    @pytest.mark.parametrize('tp_pp', [(4, 2), (1, 1), (8, 1), (2, 2)])
    def test_sharded_state_dict(self, tp_pp, order):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(*tp_pp, order=order)

        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True
        )
        parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
        )

        sharded_state_dict = parallel_transformer_layer.sharded_state_dict()

        extra_states = {k: v for k, v in sharded_state_dict.items() if k.endswith('extra_state')}
        sharded_tensors = {
            k: v for k, v in sharded_state_dict.items() if not k.endswith('extra_state')
        }
        assert all(isinstance(t, ShardedObject) for t in extra_states.values())
        assert all(isinstance(t, ShardedTensor) for t in sharded_tensors.values())

        # Test all local shapes
        tensor_local_shapes = {k: v.local_shape for k, v in sharded_tensors.items()}
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert tensor_local_shapes == get_tensor_shapes_for_tp(transformer_config, tp_size)

        # Test all global shapes. Prepend num layers in front of expected shapes
        tensor_global_shapes = {k: v.global_shape for k, v in sharded_tensors.items()}
        expected_global_shapes = get_tensor_shapes_for_tp(transformer_config, 1)
        assert tensor_global_shapes == expected_global_shapes

        # Test ShardedTensor keys
        for state_dict_key, sh_ten in sharded_tensors.items():
            assert state_dict_key == sh_ten.key

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)


def get_tensor_shapes_for_tp(transformer_config, tp_size):
    hs = transformer_config.hidden_size
    return {
        'mlp.linear_fc1.layer_norm_weight': (hs,),
        'mlp.linear_fc1.layer_norm_bias': (hs,),
        'mlp.linear_fc1.weight': (hs * 4 // tp_size, hs),
        'mlp.linear_fc1.bias': (hs * 4 // tp_size,),
        'mlp.linear_fc2.weight': (hs, hs * 4 // tp_size),
        'mlp.linear_fc2.bias': (hs,),
        'self_attention.linear_proj.weight': (hs, hs // tp_size),
        'self_attention.linear_proj.bias': (hs,),
        'self_attention.linear_qkv.layer_norm_weight': (hs,),
        'self_attention.linear_qkv.layer_norm_bias': (hs,),
        'self_attention.linear_qkv.weight': (hs * 3 // tp_size, hs),
        'self_attention.linear_qkv.bias': (hs * 3 // tp_size,),
    }
