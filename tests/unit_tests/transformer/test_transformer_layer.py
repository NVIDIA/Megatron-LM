# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import CheckpointManager, model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
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
            transformer_config, get_gpt_layer_with_transformer_engine_submodules()
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
                    transformer_config, get_gpt_layer_with_transformer_engine_submodules()
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
            parallel_state.set_pipeline_model_parallel_rank(pipeline_rank)

            try:
                actual_offset = get_transformer_layer_offset(config, vp_stage)
                assert actual_offset == expected_offset, (
                    f"Expected offset {expected_offset} for pipeline rank {pipeline_rank}, "
                    f"VP stage {vp_stage}, but got {actual_offset}"
                )
            finally:
                parallel_state.set_pipeline_model_parallel_rank(original_get_pipeline_rank)

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
            transformer_config, get_gpt_layer_with_transformer_engine_submodules()
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


class TestTransformerLayerWithHyperConnectionRecompute:
    """Test TransformerLayer with HyperConnection and MHC block recomputation."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_layer_with_hyper_connection(self, hidden_size=64, num_streams=4):
        """Create a HyperConnectionTransformerLayer with hyper connection enabled."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            recompute_hyper_connections=True,
            recompute_granularity='selective',
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layer = HyperConnectionTransformerLayer(config, layer_spec.submodules)
        layer.cuda()
        return layer, config

    def test_forward_with_hyper_connection_recompute(self):
        """
        Test that TransformerLayer forward works correctly with HyperConnection
        and MHC block recomputation enabled.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        layer, config = self._create_layer_with_hyper_connection(hidden_size, num_streams)
        layer.train()  # Enable training mode for recomputation

        # Input shape: [seq_len, batch_size, n * hidden_size] for hyper connections
        n_channels = num_streams * hidden_size
        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        # Create manager for MHC block recomputation
        manager = CheckpointManager()

        # Forward pass with recompute manager
        manager.is_last_layer_in_recompute_block = True
        output, context = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            mhc_recompute_manager=manager,
        )

        # Verify output shape
        assert output.shape == (
            seq_len,
            batch_size,
            n_channels,
        ), f"Expected output shape {(seq_len, batch_size, n_channels)}, got {output.shape}"

        # Register unified recompute hook at block boundary.
        manager.discard_all_outputs_and_register_unified_recompute(output)

        # Backward pass should work without error
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert hidden_states.grad is not None, "Gradients should be computed for hidden_states"
        assert hidden_states.grad.shape == hidden_states.shape

    def test_intermediate_layer_with_recompute(self):
        """
        Test TransformerLayer as an intermediate layer (not last in block).
        In this case, MLP BDA should also be checkpointed.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        layer, config = self._create_layer_with_hyper_connection(hidden_size, num_streams)
        layer.train()

        n_channels = num_streams * hidden_size
        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        manager = CheckpointManager()

        # Forward pass - NOT the last layer in block
        manager.is_last_layer_in_recompute_block = False
        output, context = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            mhc_recompute_manager=manager,
        )

        # Verify output shape
        assert output.shape == (seq_len, batch_size, n_channels)

        # Backward pass should work
        loss = output.sum()
        # For intermediate layers, we need to pass output to next layer
        # Here we just register the recompute hook on output for testing
        manager.discard_all_outputs_and_register_unified_recompute(loss)

        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

    def test_multiple_layers_chain_with_recompute(self):
        """
        Test multiple TransformerLayers chained together with a single
        CheckpointManager, simulating TransformerBlock behavior.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2
        num_layers = 3

        # Create multiple layers
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            recompute_hyper_connections=True,
            recompute_granularity='selective',
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layers = [
            HyperConnectionTransformerLayer(
                config, layer_spec.submodules, layer_number=i + 1
            ).cuda()
            for i in range(num_layers)
        ]

        for layer in layers:
            layer.train()

        n_channels = num_streams * hidden_size
        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        # Single manager for all layers (like TransformerBlock)
        manager = CheckpointManager()

        # Forward through all layers
        h = hidden_states
        for i, layer in enumerate(layers):
            is_last = i == num_layers - 1
            manager.is_last_layer_in_recompute_block = is_last
            h, _ = layer(
                hidden_states=h, attention_mask=attention_mask, mhc_recompute_manager=manager
            )
            if is_last:
                manager.discard_all_outputs_and_register_unified_recompute(h)

        # Backward pass
        loss = h.sum()
        loss.backward()

        # Verify gradients
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape
        # Check that gradient is non-trivial (not all zeros)
        assert hidden_states.grad.abs().sum() > 0


class TestMHCRecomputeMemorySaving:
    """Verify that recompute_hyper_connections actually reduces peak GPU memory."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _run_forward_backward(
        num_layers,
        hidden_size,
        num_streams,
        seq_len,
        batch_size,
        use_recompute,
        recompute_block_size=2,
    ):
        """Run a full forward + backward pass and return (peak memory, output grad).

        When use_recompute=True, a new CheckpointManager is created every
        `recompute_block_size` layers, mirroring TransformerBlock's
        _build_mhc_recompute_layer_plan logic.
        """
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            recompute_hyper_connections=use_recompute,
            recompute_granularity='selective' if use_recompute else None,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layers = [
            HyperConnectionTransformerLayer(
                config, layer_spec.submodules, layer_number=i + 1
            ).cuda()
            for i in range(num_layers)
        ]
        for layer in layers:
            layer.train()

        n_channels = num_streams * hidden_size
        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        manager = CheckpointManager() if use_recompute else None

        h = hidden_states
        for i, layer in enumerate(layers):
            is_last_in_block = (i == num_layers - 1) or ((i + 1) % recompute_block_size == 0)
            kwargs = dict(hidden_states=h, attention_mask=attention_mask)
            if manager is not None:
                manager.is_last_layer_in_recompute_block = is_last_in_block
                kwargs['mhc_recompute_manager'] = manager
            h, _ = layer(**kwargs)
            if manager is not None and is_last_in_block:
                manager.discard_all_outputs_and_register_unified_recompute(h)
                if i < num_layers - 1:
                    manager = CheckpointManager()

        loss = h.sum()
        loss.backward()
        torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated()
        grad = hidden_states.grad.clone()

        del layers, hidden_states, h, loss, manager
        torch.cuda.empty_cache()

        return peak_mem, grad

    def test_recompute_reduces_peak_memory(self):
        """Peak memory with recompute (block_size=2) should be lower than without."""
        num_layers = 8
        hidden_size = 128
        num_streams = 4
        seq_len = 64
        batch_size = 4

        peak_no_recompute, _ = self._run_forward_backward(
            num_layers, hidden_size, num_streams, seq_len, batch_size, use_recompute=False
        )
        peak_recompute, _ = self._run_forward_backward(
            num_layers,
            hidden_size,
            num_streams,
            seq_len,
            batch_size,
            use_recompute=True,
            recompute_block_size=2,
        )

        saving_pct = (peak_no_recompute - peak_recompute) / peak_no_recompute * 100

        assert peak_recompute < peak_no_recompute, (
            f"Recompute should reduce peak memory, but got "
            f"no_recompute={peak_no_recompute / 1e6:.1f}MB vs "
            f"recompute={peak_recompute / 1e6:.1f}MB "
            f"(saving={saving_pct:.1f}%)"
        )


class TestMHCWithCudaGraph:
    """Test HyperConnectionTransformerLayer compatibility with CUDA graphs.

    CUDA graph capture requires static computation graphs and fixed tensor shapes.
    These tests verify that the mHC layer properly supports the CUDA graph interface
    defined in GraphableMegatronModule and TransformerLayer.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123, use_cudagraphable_rng=True, force_reset_rng=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_mhc_layer(self, hidden_size=64, num_streams=4, **extra_config):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            **extra_config,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layer = HyperConnectionTransformerLayer(config, layer_spec.submodules)
        layer.cuda()
        return layer, config

    def test_get_layer_static_inputs_shape_for_mhc(self):
        """get_layer_static_inputs must return [s, b, n*C] for mHC layers.

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. If the shape is [s, b, C] instead of [s, b, n*C], the graph
        capture will produce a shape mismatch at the first hyper connection module.
        """
        layer, config = self._create_mhc_layer()
        seq_length = 32
        micro_batch_size = 2

        static_inputs = layer.get_layer_static_inputs(seq_length, micro_batch_size)
        hidden_states = static_inputs["hidden_states"]

        expected_hidden_dim = config.num_residual_streams * config.hidden_size
        assert hidden_states.shape[-1] == expected_hidden_dim, (
            f"get_layer_static_inputs returns hidden dim {hidden_states.shape[-1]} "
            f"but mHC expects {expected_hidden_dim} (n={config.num_residual_streams} * "
            f"C={config.hidden_size}). "
            f"HyperConnectionTransformerLayer must override get_layer_static_inputs."
        )

    def test_submodules_under_cudagraphs_includes_hyper_connection(self):
        """_get_submodules_under_cudagraphs must include hyper connection modules.

        CUDA graph manual hooks are set up for parameters of submodules returned
        by this method. Missing hyper connection modules means their parameters
        (mapping_proj, alpha_*, bias) will not get proper pre-forward hooks during
        graph replay, leading to stale parameter values.
        """
        layer, config = self._create_mhc_layer()

        submodules = layer._get_submodules_under_cudagraphs()

        hc_modules_found = any(
            hasattr(m, 'mapping_proj') for submod in submodules for m in submod.modules()
        )
        assert hc_modules_found, (
            "_get_submodules_under_cudagraphs does not include HyperConnectionModule. "
            "Parameters like mapping_proj, alpha_pre/post/res will not be updated "
            "during CUDA graph replay."
        )

    def test_forward_through_te_cuda_graph_capture_path(self):
        """_te_cuda_graph_capture must produce correct output shapes for mHC.

        TE CUDA graph capture calls _te_cuda_graph_capture() during warmup.
        For mHC layers, the input must be n-stream [s, b, n*C] and output must
        also be [s, b, n*C].
        """
        layer, config = self._create_mhc_layer()
        layer.eval()

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        hidden_states = torch.randn(seq_len, batch_size, n_channels, device='cuda')
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        with torch.no_grad():
            outputs = layer._te_cuda_graph_capture(
                hidden_states=hidden_states, attention_mask=attention_mask
            )

        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        assert output.shape == (seq_len, batch_size, n_channels), (
            f"_te_cuda_graph_capture output shape {output.shape} != "
            f"expected {(seq_len, batch_size, n_channels)}"
        )

    def test_cuda_graph_fwd_bwd_with_hyper_connection(self):
        """End-to-end CUDA graph capture and replay for forward+backward with mHC.

        Captures both the forward and backward pass of HyperConnectionTransformerLayer
        into a torch.cuda.CUDAGraph and replays it with fresh input data, verifying
        that the computation graph is fully static (capturable) and produces correct
        output shapes and non-trivial gradients.
        """
        layer, config = self._create_mhc_layer()
        layer.train()

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        static_input = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        # Warmup on side stream to trigger lazy allocations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out, _ = layer(hidden_states=static_input, attention_mask=attention_mask)
                out.sum().backward()
        torch.cuda.current_stream().wait_stream(s)

        # Set .grad to None so backward allocates fresh gradient tensors in the
        # graph's private memory pool during capture.
        layer.zero_grad(set_to_none=True)
        static_input.grad = None

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output, _ = layer(hidden_states=static_input, attention_mask=attention_mask)
            output.sum().backward()

        # Replay with new input data.
        # Use no_grad because backward inside the captured graph already
        # bumped the autograd version counter on static_input, making
        # in-place copy_ illegal without disabling grad tracking.
        with torch.no_grad():
            static_input.copy_(torch.randn_like(static_input))
        g.replay()

        assert output.shape == (
            seq_len,
            batch_size,
            n_channels,
        ), f"Output shape {output.shape} != expected {(seq_len, batch_size, n_channels)}"
        assert (
            static_input.grad is not None
        ), "Gradients should be computed for static_input after graph replay"
        assert static_input.grad.shape == static_input.shape
        assert static_input.grad.abs().sum() > 0, "Gradients should be non-trivial"

        # Verify numerical consistency: graph replay should match eager execution
        # with the same input and weights.
        test_data = torch.randn(seq_len, batch_size, n_channels, device='cuda')

        with torch.no_grad():
            static_input.copy_(test_data)
        g.replay()
        graph_out = output.detach().clone()
        graph_grad = static_input.grad.detach().clone()

        eager_input = test_data.clone().requires_grad_(True)
        eager_output, _ = layer(hidden_states=eager_input, attention_mask=attention_mask)
        eager_output.sum().backward()

        assert torch.allclose(graph_out, eager_output.detach(), atol=1e-5), (
            f"Graph vs eager output mismatch: "
            f"max diff = {(graph_out - eager_output.detach()).abs().max().item()}"
        )
        assert torch.allclose(graph_grad, eager_input.grad, atol=1e-5), (
            f"Graph vs eager gradient mismatch: "
            f"max diff = {(graph_grad - eager_input.grad).abs().max().item()}"
        )

    def test_cuda_graph_fwd_bwd_with_hyper_connection_and_recompute(self):
        """CUDA graph capture+replay for fwd+bwd with mHC and CheckpointManager.

        When a CheckpointManager is used, additional CheckpointWithoutOutput
        objects are created for layernorm and hyper-connection operations. The
        manager discards intermediate activations during forward (storage.resize_(0))
        and recomputes them during backward via a unified gradient hook.
        This test verifies the full capture+replay still works correctly.
        """
        layer, config = self._create_mhc_layer()
        layer.train()

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        static_input = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        # Warmup on side stream; fresh manager per iteration to avoid stale state.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                mgr = CheckpointManager()
                mgr.is_last_layer_in_recompute_block = True
                out, _ = layer(
                    hidden_states=static_input,
                    attention_mask=attention_mask,
                    mhc_recompute_manager=mgr,
                )
                mgr.discard_all_outputs_and_register_unified_recompute(out)
                out.sum().backward()
        torch.cuda.current_stream().wait_stream(s)

        layer.zero_grad(set_to_none=True)
        static_input.grad = None

        capture_mgr = CheckpointManager()
        capture_mgr.is_last_layer_in_recompute_block = True

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output, _ = layer(
                hidden_states=static_input,
                attention_mask=attention_mask,
                mhc_recompute_manager=capture_mgr,
            )
            capture_mgr.discard_all_outputs_and_register_unified_recompute(output)
            output.sum().backward()

        # Replay with new input data.
        with torch.no_grad():
            static_input.copy_(torch.randn_like(static_input))
        g.replay()

        assert output.shape == (
            seq_len,
            batch_size,
            n_channels,
        ), f"Output shape {output.shape} != expected {(seq_len, batch_size, n_channels)}"
        assert (
            static_input.grad is not None
        ), "Gradients should be computed for static_input after graph replay"
        assert static_input.grad.shape == static_input.shape
        assert static_input.grad.abs().sum() > 0, "Gradients should be non-trivial"

        # Numerical consistency: graph replay vs eager with the same input.
        test_data = torch.randn(seq_len, batch_size, n_channels, device='cuda')

        with torch.no_grad():
            static_input.copy_(test_data)
        g.replay()
        graph_out = output.detach().clone()
        graph_grad = static_input.grad.detach().clone()

        eager_mgr = CheckpointManager()
        eager_mgr.is_last_layer_in_recompute_block = True
        eager_input = test_data.clone().requires_grad_(True)
        eager_output, _ = layer(
            hidden_states=eager_input,
            attention_mask=attention_mask,
            mhc_recompute_manager=eager_mgr,
        )
        eager_mgr.discard_all_outputs_and_register_unified_recompute(eager_output)
        eager_output.sum().backward()

        assert torch.allclose(graph_out, eager_output.detach(), atol=1e-5), (
            f"Graph vs eager output mismatch: "
            f"max diff = {(graph_out - eager_output.detach()).abs().max().item()}"
        )
        assert torch.allclose(graph_grad, eager_input.grad, atol=1e-5), (
            f"Graph vs eager gradient mismatch: "
            f"max diff = {(graph_grad - eager_input.grad).abs().max().item()}"
        )

    def test_mcore_cudagraph_manager_with_mhc_recompute_manager(self):
        """MCore CudaGraphManager must not crash on mhc_recompute_manager kwarg.

        When cuda_graph_impl="local" is set, TransformerLayer.__call__ routes
        through MegatronModule.__call__ → CudaGraphManager.__call__, which
        iterates over all kwargs to check supported types. CheckpointManager
        (used by mhc_recompute_manager) is not a CUDA-graph-supported type.

        This test verifies that mhc_recompute_manager is properly extracted
        from kwargs before the CudaGraphManager sees them, preventing the
        AssertionError that would otherwise occur.
        """
        layer, config = self._create_mhc_layer(cuda_graph_impl="local", cuda_graph_scope="attn")
        layer.train()

        assert hasattr(
            layer, 'cudagraph_manager'
        ), "Layer should have cudagraph_manager with cuda_graph_impl='local'"

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        mgr = CheckpointManager()
        mgr.is_last_layer_in_recompute_block = True

        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask, mhc_recompute_manager=mgr
        )

        assert output.shape == (seq_len, batch_size, n_channels)

    def test_mcore_cudagraph_manager_without_mhc_recompute_manager(self):
        """MCore CudaGraphManager path works when mhc_recompute_manager is None."""
        layer, config = self._create_mhc_layer(cuda_graph_impl="local", cuda_graph_scope="attn")
        layer.train()

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        output, context = layer(hidden_states=hidden_states, attention_mask=attention_mask)

        assert output.shape == (seq_len, batch_size, n_channels)


class TestMHCWithOffloading:
    """Test HyperConnectionTransformerLayer with fine-grained activation offloading.

    Fine-grained activation offloading transfers specific activations (e.g., layernorm
    inputs) to CPU during forward and reloads them during backward. These tests verify
    that the mHC layer's multi-stream architecture works correctly with offloading.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_mhc_layer_with_offloading(
        self, hidden_size=64, num_streams=4, offload_modules=None
    ):
        if offload_modules is None:
            offload_modules = ["attn_norm", "mlp_norm"]

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layer = HyperConnectionTransformerLayer(config, layer_spec.submodules)
        layer.cuda()
        return layer, config

    def test_offloading_flags_set_correctly(self):
        """Verify offload_attn_norm and offload_mlp_norm are properly set for mHC."""
        layer, config = self._create_mhc_layer_with_offloading()

        assert layer.offload_attn_norm, (
            "offload_attn_norm should be True when fine_grained_activation_offloading=True "
            "and 'attn_norm' in offload_modules"
        )
        assert layer.offload_mlp_norm, (
            "offload_mlp_norm should be True when fine_grained_activation_offloading=True "
            "and 'mlp_norm' in offload_modules"
        )

    def test_forward_backward_with_offloading(self):
        """Forward+backward should work with activation offloading enabled.

        This exercises the off_interface context manager around layernorms in
        the mHC forward path, including the group_commit that commits the
        offloading group for the aggregated 1-stream layernorm input.
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        layer, config = self._create_mhc_layer_with_offloading()
        layer.train()

        seq_len = 8
        batch_size = 2
        n_channels = config.num_residual_streams * config.hidden_size

        hidden_states = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        mgr = PipelineOffloadManager.get_instance()
        mgr.init_model_chunk_offload_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=0)

        output, context = layer(hidden_states=hidden_states, attention_mask=attention_mask)

        assert output.shape == (
            seq_len,
            batch_size,
            n_channels,
        ), f"Output shape {output.shape} != expected {(seq_len, batch_size, n_channels)}"

        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Gradients should flow through offloaded path"
        assert hidden_states.grad.shape == hidden_states.shape
        assert hidden_states.grad.abs().sum() > 0, "Gradients should be non-trivial"

        PipelineOffloadManager.reset_instance()

    def test_offloading_numerical_equivalence(self):
        """Offloaded forward+backward must produce the same result as non-offloaded.

        Compares outputs and gradients between a layer with offloading disabled
        vs enabled to ensure the offloading path does not corrupt activations.
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        PipelineOffloadManager.reset_instance()

        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2
        n_channels = num_streams * hidden_size

        torch.manual_seed(42)
        input_data = torch.randn(seq_len, batch_size, n_channels, device='cuda')
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        common_config_kwargs = dict(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

        # Run without offloading
        config_no_offload = TransformerConfig(**common_config_kwargs)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layer_no_offload = HyperConnectionTransformerLayer(
            config_no_offload, layer_spec.submodules
        ).cuda()
        layer_no_offload.train()

        h1 = input_data.clone().detach().requires_grad_(True)
        out1, _ = layer_no_offload(hidden_states=h1, attention_mask=attention_mask)
        out1.sum().backward()
        grad_no_offload = h1.grad.clone()
        out1_detached = out1.detach().clone()

        # Run with offloading using the same weights
        config_offload = TransformerConfig(
            **common_config_kwargs,
            fine_grained_activation_offloading=True,
            offload_modules=["attn_norm", "mlp_norm"],
        )
        layer_offload = HyperConnectionTransformerLayer(
            config_offload, layer_spec.submodules
        ).cuda()
        layer_offload.load_state_dict(layer_no_offload.state_dict())
        layer_offload.train()

        mgr = PipelineOffloadManager.get_instance()
        mgr.init_model_chunk_offload_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=0)

        h2 = input_data.clone().detach().requires_grad_(True)
        out2, _ = layer_offload(hidden_states=h2, attention_mask=attention_mask)
        out2.sum().backward()
        grad_offload = h2.grad.clone()

        PipelineOffloadManager.reset_instance()

        assert torch.allclose(out1_detached, out2.detach(), atol=1e-5), (
            f"Forward outputs differ: max diff = "
            f"{(out1_detached - out2.detach()).abs().max().item()}"
        )
        assert torch.allclose(grad_no_offload, grad_offload, atol=1e-5), (
            f"Gradients differ: max diff = "
            f"{(grad_no_offload - grad_offload).abs().max().item()}"
        )
