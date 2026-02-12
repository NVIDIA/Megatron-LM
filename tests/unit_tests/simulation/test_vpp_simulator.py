# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for VppSimulator.run_global_step functionality.

Tests the simulation of Pipeline Parallel (PP) and Virtual Pipeline Parallel (VPP)
training under different configurations, using mocked execution functions to avoid
actual GPU computation.
"""

import os
import json
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from megatron.training.simulation.vpp_simulate import VppSimulator
from megatron.training.simulation.task import TaskType
from megatron.training.global_vars import set_args
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    destroy_num_microbatches_calculator
)
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestVppSimulatorBasic:
    """Test VppSimulator basic functionality - run_global_step under different PP/VPP configs"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Setup: Initialize minimal parallel environment
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1
        )
        yield
        # Teardown: Clean up
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Mock args object with all necessary simulation parameters"""
        args = MagicMock()

        # PP configuration
        args.pipeline_model_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = None
        # Create a simple pipeline layout - will be updated per test
        # Format: list of lists, each inner list is layers for one PP stage
        # For now, create a simple 2-stage layout with decoder layers
        args.pipeline_model_parallel_layout = None  # Will be set per test

        # Model configuration
        args.num_layers = 4
        args.hidden_size = 64
        args.num_attention_heads = 4
        args.seq_length = 128
        args.max_position_embeddings = 128

        # Training configuration
        args.micro_batch_size = 2
        args.global_batch_size = 8
        args.data_parallel_size = 1

        # Simulation configuration
        args.simulate_result_dir = str(tmp_path / "sim_results")
        args.execute_mode = 'router_balanced'
        args.skip_execute = False
        args.microbatch_group_size_per_vp_stage = None

        # Other required parameters
        args.rank = 0
        # MoE configuration - all layers are dense (no MoE)
        # moe_layer_freq is a list where 0=dense, 1=moe
        args.moe_layer_freq = [0] * args.num_layers  # All dense layers
        args.use_cpu_initialization = True
        args.perform_initialization = True
        args.fp16 = False
        args.bf16 = False

        # Data configuration - use mock data to avoid downloads
        args.mock_data = True
        args.tokenizer_type = 'NullTokenizer'
        args.vocab_size = 128256
        args.data_path = None

        # Create result directory
        os.makedirs(args.simulate_result_dir, exist_ok=True)

        return args

    @pytest.fixture
    def simple_model_provider(self):
        """Simple model provider returning minimal test model"""
        def provider(pre_process=True, post_process=True, **kwargs):
            """
            Model provider function for testing

            Accepts all standard model_provider arguments:
            - pre_process, post_process: bool flags
            - config: TransformerConfig (optional)
            - pg_collection: ProcessGroupCollection (optional)
            - vp_stage: int (optional, for VPP)

            Returns a list of models (VPP format)
            """
            class SimpleTestModel(torch.nn.Module):
                """Minimal model for testing"""
                def __init__(self, config):
                    super().__init__()
                    self.config = config  # Required by Megatron
                    self.linear = torch.nn.Linear(64, 64)

                def forward(self, hidden_states):
                    return self.linear(hidden_states)

            # Create a minimal TransformerConfig for the model
            # Use config from kwargs if provided, otherwise create a simple one
            config = kwargs.get('config', None)
            if config is None:
                config = TransformerConfig(
                    num_layers=4,
                    hidden_size=64,
                    num_attention_heads=4,
                    use_cpu_initialization=True
                )

            # Return single model object (get_model will wrap in list if needed for VPP)
            return SimpleTestModel(config)

        return provider

    @pytest.fixture
    def mock_data_iterator(self, mock_args):
        """Mock data iterator"""
        class MockDataIterator:
            def __init__(self, args):
                self.args = args

            def __next__(self):
                batch_size = self.args.micro_batch_size
                seq_length = self.args.seq_length
                return {
                    'tokens': torch.randint(0, 1000, (batch_size, seq_length)),
                    'labels': torch.randint(0, 1000, (batch_size, seq_length)),
                    'loss_mask': torch.ones(batch_size, seq_length),
                    'attention_mask': torch.ones(batch_size, 1, seq_length, seq_length)
                }

            def __iter__(self):
                return self

        return MockDataIterator(mock_args)

    @pytest.fixture
    def forward_step_func(self):
        """Simplified forward step function"""
        def forward_step(data_iterator, model, num_microbatches, input_tensor, **kwargs):
            """
            Simplified forward step for testing
            Returns output_tensor and loss_func
            """
            batch_size = 2
            seq_length = 128
            hidden_size = 64

            # Create fake output_tensor
            if input_tensor is None:
                # First stage - create new tensor
                output_tensor = torch.randn(batch_size, seq_length, hidden_size)
            else:
                # Middle stage - create output based on input
                output_tensor = torch.randn_like(input_tensor)

            # Simple loss_func
            def loss_func(output_tensor, non_loss_data):
                loss = torch.tensor(1.0)
                return loss, {'lm loss': loss}

            return output_tensor, loss_func

        return forward_step

    @pytest.fixture
    def mock_execute_functions(self, monkeypatch):
        """Mock forward and backward execution functions to return fixed duration"""

        def fake_execute_forward(*args, **kwargs):
            """
            Mock execute_forward_with_timing
            Returns (output_tensor, duration)
            """
            batch_size = 2
            seq_length = 128
            hidden_size = 64
            output_tensor = torch.randn(batch_size, seq_length, hidden_size)
            duration = 10.0  # Fixed 10ms
            return output_tensor, duration

        def fake_execute_backward(*args, **kwargs):
            """
            Mock execute_backward_with_timing
            Returns (input_tensor_grad, duration)
            """
            batch_size = 2
            seq_length = 128
            hidden_size = 64
            input_tensor_grad = torch.randn(batch_size, seq_length, hidden_size)
            duration = 15.0  # Fixed 15ms
            return input_tensor_grad, duration

        # Patch execution functions
        monkeypatch.setattr(
            'megatron.training.simulation.model_executor.execute_forward_with_timing',
            fake_execute_forward
        )
        monkeypatch.setattr(
            'megatron.training.simulation.model_executor.execute_backward_with_timing',
            fake_execute_backward
        )

    @pytest.mark.parametrize("pp_size,vpp_size,num_microbatches", [
        (1, None, 4),   # No Pipeline
        (2, None, 4),   # 2-stage Pipeline
        (4, None, 8),   # 4-stage Pipeline
        (2, 2, 8),      # PP=2, VPP=2
        (4, 2, 8),      # PP=4, VPP=2
    ])
    def test_run_global_step_completes(
        self,
        pp_size,
        vpp_size,
        num_microbatches,
        mock_args,
        simple_model_provider,
        mock_data_iterator,
        forward_step_func,
        mock_execute_functions,
        monkeypatch
    ):
        """Test that run_global_step completes successfully under different PP/VPP configs"""

        # 1. Configure args for this test
        mock_args.pipeline_model_parallel_size = pp_size
        mock_args.virtual_pipeline_model_parallel_size = vpp_size
        mock_args.global_batch_size = num_microbatches * mock_args.micro_batch_size

        # Create a simple pipeline layout based on pp_size and vpp_size
        # Format: "t" = decoder layer, "E" = embedding, "L" = loss
        # For simplicity, use a uniform layout with decoder layers
        num_model_chunks = vpp_size if vpp_size else 1
        layers_per_chunk = 2  # 2 decoder layers per model chunk

        if pp_size == 1:
            # Single PP stage
            if num_model_chunks == 1:
                # No VPP: "EttL" (Embedding + 2 decoder + Loss)
                layout_str = f"E{'t' * layers_per_chunk}L"
            else:
                # With VPP: "E(tt|)*N-1 tt L" where N is num_model_chunks
                # E.g., for vpp_size=2: "Ett|ttL"
                vpp_parts = []
                for i in range(num_model_chunks):
                    if i == 0:
                        vpp_parts.append("E" + "t" * layers_per_chunk)
                    elif i == num_model_chunks - 1:
                        vpp_parts.append("t" * layers_per_chunk + "L")
                    else:
                        vpp_parts.append("t" * layers_per_chunk)
                layout_str = "|".join(vpp_parts)
        else:
            # Multiple PP stages
            if num_model_chunks == 1:
                # No VPP: distribute layers across PP stages
                # E.g., for pp_size=2: "Ett|ttL"
                # E.g., for pp_size=4: "Et|t|t|tL"
                pp_parts = []
                for i in range(pp_size):
                    if i == 0:
                        pp_parts.append("E" + "t" * layers_per_chunk)
                    elif i == pp_size - 1:
                        pp_parts.append("t" * layers_per_chunk + "L")
                    else:
                        pp_parts.append("t" * layers_per_chunk)
                layout_str = "|".join(pp_parts)
            else:
                # With VPP: grouped layout
                # E.g., for pp_size=2, vpp_size=2: "E(tt|)*2L"
                layout_str = f"E({'t' * layers_per_chunk}|)*{num_model_chunks}L"

        mock_args.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
            layout=layout_str,
            pipeline_model_parallel_size=pp_size
        )

        # 2. Set global args and initialize num_microbatches_calculator
        set_args(mock_args)
        # Destroy any existing calculator first
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=mock_args.global_batch_size,
            micro_batch_size=mock_args.micro_batch_size,
            data_parallel_size=mock_args.data_parallel_size
        )

        # 3. Create VppSimulator
        simulator = VppSimulator(
            mock_data_iterator,
            simple_model_provider,
            forward_step_func
        )

        # 4. Execute run_global_step
        simulator.run_global_step()

        # 5. Verify: Task count is correct
        num_model_chunks = vpp_size if vpp_size else 1
        expected_task_count = pp_size * num_microbatches * num_model_chunks * 2
        assert simulator.task_num_count == expected_task_count, \
            f"Expected {expected_task_count} tasks, got {simulator.task_num_count}"

        # 6. Verify: All tasks are finished
        for task_key, task in simulator.pp_mbid_mcid_fb_task_dict.items():
            assert task.finished, \
                f"Task {task.task_id} not finished"
            assert task.duration is not None and task.duration > 0, \
                f"Task {task.task_id} has invalid duration: {task.duration}"

        # 7. Verify: Result files created (only rank 0)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            result_dir = Path(mock_args.simulate_result_dir)
            required_files = [
                'finished_tasks.json',
                'pp_task_orders.json',
                'static_memory_info.json',
                'all_pp_vpp_layout.json',
                'task_durations.json'
            ]

            for filename in required_files:
                filepath = result_dir / filename
                assert filepath.exists(), \
                    f"Expected result file not found: {filename}"
        elif not torch.distributed.is_initialized():
            # Single-process mode - should still create files
            result_dir = Path(mock_args.simulate_result_dir)
            required_files = [
                'finished_tasks.json',
                'pp_task_orders.json',
                'static_memory_info.json',
                'all_pp_vpp_layout.json',
                'task_durations.json'
            ]

            for filename in required_files:
                filepath = result_dir / filename
                assert filepath.exists(), \
                    f"Expected result file not found: {filename}"
