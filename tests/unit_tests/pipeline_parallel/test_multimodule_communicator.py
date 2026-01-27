# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
    _avg_params,
    _create_transformer_block,
    _get_pg_collection_from_grid,
    create_hypercomm_grid,
    get_transformer_block_and_grid,
)
from tests.unit_tests.test_utilities import Utils


class TestMultiModulePipelineCommunicator:

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(
                f"These tests require 8 GPUs, but only {world_size} are available.",
                allow_module_level=True,
            )

    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_multimodule_communicator_init(self):
        """Test MultiModulePipelineCommunicator initialization."""

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=2, dp=1)
        generator_grid = create_hypercomm_grid(offset=6, tp=2, cp=1, pp=1, dp=1)

        # Define module-grid mapping
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        # Define module computation topology
        topology = {
            'image_encoder': ['llm'],
            'audio_encoder': ['llm'],
            'llm': ['generator'],
            'generator': [],
        }
        config = ModelParallelConfig(bf16=True)
        # Initialize communicator
        mllm_comm = MultiModulePipelineCommunicator(module_to_grid_map, topology, config)
        # Test attributes match expectations
        assert mllm_comm.module_to_grid_map == module_to_grid_map
        assert mllm_comm.topology == topology
        assert mllm_comm.config == config
        assert mllm_comm.current_rank == dist.get_rank()

    def test_compute_total_pipeline_stages(self):
        """Test compute_total_pipeline_stages for overall chain and until specific ranks."""

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=2, dp=1)
        generator_grid = create_hypercomm_grid(offset=6, tp=1, cp=1, pp=1, dp=2)

        # Define module-grid mapping and topology
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        topology = {
            'image_encoder': ['llm'],
            'audio_encoder': ['llm'],
            'llm': ['generator'],
            'generator': [],
        }

        # Overall total pipeline stages: max(1,1) + 2 + 1 = 4
        total = MultiModulePipelineCommunicator.compute_total_pipeline_stages(
            topology, module_to_grid_map
        )
        assert total == 4

        llm_pp_rank = MultiModulePipelineCommunicator.compute_total_pipeline_stages(
            topology, module_to_grid_map, rank=2, module_name='llm'
        )
        assert llm_pp_rank == 2

    def test_send_forward_recv_forward(self):
        """Test send_forward and recv_forward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=2, dp=1)
        generator_grid = create_hypercomm_grid(offset=6, tp=1, cp=1, pp=1, dp=2)

        # Set up module-grid mapping and topology
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        topology = {
            'image_encoder': ['llm'],
            'audio_encoder': ['llm'],
            'llm': ['generator'],
            'generator': [],
        }
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        mllm_comm = MultiModulePipelineCommunicator(module_to_grid_map, topology, config)

        # Simulate forward communication for each module
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            # Image encoder sends output forward
            output_dict = {'image_encoder': torch.randn(2, 8, 128).cuda()}
            mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            # Audio encoder sends output forward
            output_dict = {'audio_encoder': torch.randn(2, 16, 128).cuda()}
            mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            output_dict = {'llm': torch.randn(2, 32, 128).cuda()}
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                # LLM stage receives both image and audio outputs
                input_dict = mllm_comm.recv_forward()
                assert input_dict['image_encoder'].shape == (2, 8, 128)
                assert input_dict['audio_encoder'].shape == (2, 16, 128)
                mllm_comm.send_forward(output_dict)
            else:
                # LLM stage receives concatenated LLM outputs
                input_dict = mllm_comm.recv_forward(tensor_shape=(2, 32, 128))
                assert input_dict['llm'].shape == (2, 32, 128)
                mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(generator_grid):
            # Generator module receives final LLM output
            input_dict = mllm_comm.recv_forward()
            assert input_dict['llm'].shape == (1, 32, 128)

    def test_send_forward_recv_forward_with_different_pp_size(self):
        """Test for the case when pp(image_encoder) != pp(audio_encoder)."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=2, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=4, tp=1, cp=1, pp=4, dp=1)

        # Set up module-grid mapping and topology
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
        }
        topology = {'image_encoder': ['llm'], 'audio_encoder': ['llm'], 'llm': []}
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        mllm_comm = MultiModulePipelineCommunicator(module_to_grid_map, topology, config)

        # Simulate forward communication for each module
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            output_dict = {'image_encoder': torch.randn(2, 8, 128).cuda()}
            if dist.get_rank() == 0:
                # Image encoder sends output forward
                mllm_comm.send_forward(output_dict)
            else:
                # Image stage receives image outputs
                input_dict = mllm_comm.recv_forward(tensor_shape=(2, 8, 128))
                assert input_dict['image_encoder'].shape == (2, 8, 128)
                mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            # Audio encoder sends output forward
            output_dict = {'audio_encoder': torch.randn(2, 16, 128).cuda()}
            mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            output_dict = {'llm': torch.randn(2, 32, 128).cuda()}
            if dist.get_rank() == 4:
                # LLM stage receives both image and audio outputs
                input_dict = mllm_comm.recv_forward()
                assert input_dict['image_encoder'].shape == (2, 8, 128)
                assert input_dict['audio_encoder'].shape == (2, 16, 128)
                mllm_comm.send_forward(output_dict)
            elif dist.get_rank() == 5 or dist.get_rank() == 6:
                # LLM stage receives concatenated LLM outputs
                input_dict = mllm_comm.recv_forward(tensor_shape=(2, 32, 128))
                assert input_dict['llm'].shape == (2, 32, 128)
                mllm_comm.send_forward(output_dict)
            elif dist.get_rank() == 7:
                # LLM stage receives concatenated LLM outputs
                input_dict = mllm_comm.recv_forward(tensor_shape=(2, 32, 128))
                assert input_dict['llm'].shape == (2, 32, 128)

    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=2, dp=1)
        generator_grid = create_hypercomm_grid(offset=6, tp=1, cp=1, pp=1, dp=2)

        # Set up module-grid mapping and topology
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        topology = {
            'image_encoder': ['llm'],
            'audio_encoder': ['llm'],
            'llm': ['generator'],
            'generator': [],
        }
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        mllm_comm = MultiModulePipelineCommunicator(module_to_grid_map, topology, config)

        # Simulate backward communication for each module
        if mllm_comm.is_current_rank_in_grid(generator_grid):
            # Generator sends gradient backward
            grad_dict = {'llm': torch.randn(1, 32, 128).cuda()}
            mllm_comm.send_backward(grad_dict)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 4 or dist.get_rank() == 5:
                # LLM receives expanded gradient and sends backward
                received_grad = mllm_comm.recv_backward()
                assert received_grad['llm'].shape == (2, 32, 128)
                grad_dict = {'llm': torch.randn(2, 32, 128).cuda()}
                mllm_comm.send_backward(grad_dict)
            else:
                # LLM receives gradient and sends backward to both image/audio encoders
                received_grad = mllm_comm.recv_backward(tensor_shape=(2, 32, 128))
                assert received_grad['llm'].shape == (2, 32, 128)
                grad_dict = {
                    'image_encoder': torch.randn(2, 8, 128).cuda(),
                    'audio_encoder': torch.randn(2, 16, 128).cuda(),
                }
                mllm_comm.send_backward(grad_dict)
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            # Image encoder receives its gradient
            received_grad = mllm_comm.recv_backward()
            assert received_grad['image_encoder'].shape == (2, 8, 128)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            # Audio encoder receives its gradient
            received_grad = mllm_comm.recv_backward()
            assert received_grad['audio_encoder'].shape == (2, 16, 128)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Feature requires PyTorch 2.3 or later",
    )
    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test send_forward_recv_backward and send_backward_recv_forward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create process group grids for each module
        image_encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
        audio_encoder_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=2, tp=2, cp=1, pp=2, dp=1)
        generator_grid = create_hypercomm_grid(offset=6, tp=1, cp=1, pp=1, dp=2)

        # Set up module-grid mapping and topology
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        topology = {
            'image_encoder': ['llm'],
            'audio_encoder': ['llm'],
            'llm': ['generator'],
            'generator': [],
        }
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        mllm_comm = MultiModulePipelineCommunicator(module_to_grid_map, topology, config)

        # Simulate bidirectional send/recv for forward and backward in pipeline

        # Encoder stages send forward to the first stage of LLM, and receive backward from the first stage of LLM
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            output_dict = {'image_encoder': torch.randn(2, 8, 128).cuda()}
            received_grad = mllm_comm.send_forward_recv_backward(output_dict)
            assert received_grad['image_encoder'].shape == (2, 8, 128)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            output_dict = {'audio_encoder': torch.randn(2, 16, 128).cuda()}
            received_grad = mllm_comm.send_forward_recv_backward(output_dict)
            assert received_grad['audio_encoder'].shape == (2, 16, 128)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                grad_dict = {
                    'image_encoder': torch.randn(2, 8, 128).cuda(),
                    'audio_encoder': torch.randn(2, 16, 128).cuda(),
                }
                input_dict = mllm_comm.send_backward_recv_forward(grad_dict)
                assert input_dict['image_encoder'].shape == (2, 8, 128)
                assert input_dict['audio_encoder'].shape == (2, 16, 128)

        # First stage of LLM sends forward to the second stage of LLM, and receive backward from the second stage of LLM
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                output_dict = {'llm': torch.randn(2, 32, 128).cuda()}
                received_grad = mllm_comm.send_forward_recv_backward(
                    output_dict, tensor_shape=(2, 32, 128)
                )
                assert received_grad['llm'].shape == (2, 32, 128)
            if dist.get_rank() == 4 or dist.get_rank() == 5:
                grad_dict = {'llm': torch.randn(2, 32, 128).cuda()}
                input_dict = mllm_comm.send_backward_recv_forward(
                    grad_dict, tensor_shape=(2, 32, 128)
                )
                assert input_dict['llm'].shape == (2, 32, 128)

        # Second stage of LLM sends forward to generator, and receive backward from generator
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 4 or dist.get_rank() == 5:
                output_dict = {'llm': torch.randn(2, 32, 128).cuda()}
                received_grad = mllm_comm.send_forward_recv_backward(output_dict)
                assert received_grad['llm'].shape == (2, 32, 128)
        if mllm_comm.is_current_rank_in_grid(generator_grid):
            grad_dict = {'llm': torch.randn(1, 32, 128).cuda()}
            input_dict = mllm_comm.send_backward_recv_forward(grad_dict)
            assert input_dict['llm'].shape == (1, 32, 128)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Feature requires PyTorch 2.3 or later",
    )
    def test_send_forward_recv_forward_with_transformer_blocks(self):
        """Test send_forward and recv_forward operations."""

        # Set model/test dimensions for easier debugging and output comparison
        hidden_size = 16
        sequence_length = 2
        micro_batch_size = 2

        # For reproducibility, set a fixed seed
        torch.manual_seed(12345)
        dtype = torch.float32

        # Create random input hidden states tensor
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, hidden_size), device="cuda"
        ).to(dtype)
        current_rank = dist.get_rank()

        # ========== Initialize tensor model-parallel environment ==========
        parallel_state_tp = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

        # ========== Build reference 1D grid and transformer block for weight sharing ==========
        ref_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
        ref_pg_collection = _get_pg_collection_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg_collection
        )
        _avg_params(
            ref_block, ref_grid.get_pg("dp")
        )  # Ensure parameters are averaged across data parallel (DP)

        # ========== Create different transformer blocks for each model stage ==========
        # Image encoder
        image_encoder_block, image_encoder_grid = get_transformer_block_and_grid(
            ref_block,
            tp_size=1,
            cp_size=1,
            pp_size=1,
            dp_size=1,
            grid_offset=0,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        # Audio encoder
        audio_encoder_block, audio_encoder_grid = get_transformer_block_and_grid(
            ref_block,
            tp_size=1,
            cp_size=1,
            pp_size=1,
            dp_size=1,
            grid_offset=1,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        # LLM (Large Language Model) block with tensor & pipeline parallelism
        llm_block, llm_grid = get_transformer_block_and_grid(
            ref_block,
            tp_size=2,
            cp_size=1,
            pp_size=2,
            dp_size=1,
            grid_offset=2,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        # Generator block (final stage) with DP=2
        generator_block, generator_grid = get_transformer_block_and_grid(
            ref_block,
            tp_size=1,
            cp_size=1,
            pp_size=1,
            dp_size=2,
            grid_offset=6,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        # ========== Define module-to-grid correspondence and pipeline topology ==========
        module_to_grid_map = {
            'image_encoder': image_encoder_grid,
            'audio_encoder': audio_encoder_grid,
            'llm': llm_grid,
            'generator': generator_grid,
        }
        topology = {
            'image_encoder': ['llm'],  # image_encoder sends output to llm
            'audio_encoder': ['llm'],  # audio_encoder sends output to llm
            'llm': ['generator'],  # llm sends output to generator
            'generator': [],  # generator is the final module
        }
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        # Define dimension mapping for sequence, batch, hidden
        dim_mapping = {'s': 0, 'h': 2, 'b': 1}
        seq_dim = dim_mapping['s']

        # Communication handler for multi-module pipeline (send/recv abstraction)
        mllm_comm = MultiModulePipelineCommunicator(
            module_to_grid_map, topology, config, dim_mapping=dim_mapping
        )

        # ========== Run actual distributed pipeline blocks (per process, depending on role) ==========
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            # Image encoder rank: run forward and send output
            image_encoder_output = image_encoder_block(
                hidden_states=hidden_states, attention_mask=None
            )
            output_dict = {'image_encoder': image_encoder_output}
            mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            # Audio encoder rank: run forward and send output
            audio_encoder_output = audio_encoder_block(
                hidden_states=hidden_states, attention_mask=None
            )
            output_dict = {'audio_encoder': audio_encoder_output}
            mllm_comm.send_forward(output_dict)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                # LLM stage 0 (receives both image and audio, concatenates along seq_dim)
                input_dict = mllm_comm.recv_forward()
                llm_output = llm_block(
                    hidden_states=torch.cat(
                        [input_dict['image_encoder'], input_dict['audio_encoder']], dim=seq_dim
                    ),
                    attention_mask=None,
                )
                output_dict = {'llm': llm_output}
                mllm_comm.send_forward(output_dict)
            else:
                # LLM stage 1 (receives output of previous LLM stage)
                input_dict = mllm_comm.recv_forward(
                    tensor_shape=(sequence_length * 2, micro_batch_size, hidden_size)
                )
                llm_output = llm_block(hidden_states=input_dict['llm'], attention_mask=None)
                output_dict = {'llm': llm_output}
                mllm_comm.send_forward(output_dict)

        if mllm_comm.is_current_rank_in_grid(generator_grid):
            # Generator block: only receives from llm and runs forward
            input_dict = mllm_comm.recv_forward()
            generator_output = generator_block(hidden_states=input_dict['llm'], attention_mask=None)

        # ========== Build a reference (serial/global) pipeline for correctness checking ==========
        global_image_encoder_block, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_audio_encoder_block, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_llm_block_pp_rank_0, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_llm_block_pp_rank_1, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_generator_block, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        # Run each stage sequentially as a global pipeline (for truth)
        global_image_encoder_output = global_image_encoder_block(
            hidden_states=hidden_states, attention_mask=None
        )
        global_audio_encoder_output = global_audio_encoder_block(
            hidden_states=hidden_states, attention_mask=None
        )
        # Compare output between global and distributed blocks for image/audio stage
        if current_rank == 0:
            torch.testing.assert_close(
                global_image_encoder_output, image_encoder_output, rtol=1e-3, atol=1e-3
            )
        if current_rank == 1:
            torch.testing.assert_close(
                global_audio_encoder_output, audio_encoder_output, rtol=1e-3, atol=1e-3
            )

        # Feed outputs to LLM stages (emulate pipeline cut with concatenation)
        global_llm_input = torch.cat(
            [global_image_encoder_output, global_audio_encoder_output], dim=seq_dim
        )
        global_llm_pp_rank_0_output = global_llm_block_pp_rank_0(
            hidden_states=global_llm_input, attention_mask=None
        )
        if current_rank == 2 or current_rank == 3:
            torch.testing.assert_close(
                global_llm_pp_rank_0_output, llm_output, rtol=1e-3, atol=1e-3
            )
        global_llm_pp_rank_1_output = global_llm_block_pp_rank_1(
            hidden_states=global_llm_pp_rank_0_output, attention_mask=None
        )
        if current_rank == 4 or current_rank == 5:
            torch.testing.assert_close(
                global_llm_pp_rank_1_output, llm_output, rtol=1e-3, atol=1e-3
            )

        # Generator output and comparison to distributed output (for each DP chunk)
        global_generator_block_output = global_generator_block(
            hidden_states=global_llm_pp_rank_1_output, attention_mask=None
        )
        global_generator_block_chunks = torch.split(
            global_generator_block_output, global_generator_block_output.shape[1] // 2, dim=1
        )
        if current_rank == 6:
            torch.testing.assert_close(
                global_generator_block_chunks[0], generator_output, rtol=1e-3, atol=1e-3
            )
        if current_rank == 7:
            torch.testing.assert_close(
                global_generator_block_chunks[1], generator_output, rtol=1e-3, atol=1e-3
            )

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "grid1_tp, grid1_pp, grid1_dp, grid2_tp, grid2_pp, grid2_dp, parallel_state_tp",
        [
            (2, 1, 1, 2, 1, 1, 2),  # TP2PP1DP1 to TP2PP1DP1
            (2, 1, 1, 2, 2, 1, 2),  # TP2PP1DP1 to TP2PP2DP1
            (2, 2, 1, 2, 2, 1, 2),  # TP2PP2DP1 to TP2PP2DP1
            (4, 1, 1, 4, 1, 1, 4),  # TP4DP1 to TP4DP1
            (2, 1, 2, 4, 1, 1, 2),  # TP2DP2 to TP4DP1
            (4, 1, 1, 2, 1, 2, 2),  # TP4DP1 to TP2DP2
            (2, 1, 2, 1, 1, 4, 2),  # TP2DP2 to TP1DP4
        ],
    )
    def test_send_forward_recv_forward_with_transformer_blocks_and_different_parallelisms(
        self, grid1_tp, grid1_pp, grid1_dp, grid2_tp, grid2_pp, grid2_dp, parallel_state_tp
    ):
        """Test bridge communicator with two transformer blocks having different process group configurations."""
        # Model and input configuration
        hidden_size = 16
        sequence_length = 2
        micro_batch_size = 8
        torch.manual_seed(12345)
        dtype = torch.float32

        # Create random input tensor on CUDA
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, hidden_size), device="cuda"
        ).to(dtype)
        hidden_states_ref = hidden_states.clone()
        current_rank = dist.get_rank()

        # Initialize model parallel with desired TP
        Utils.initialize_model_parallel(tensor_model_parallel_size=parallel_state_tp)

        # Build a reference grid and block for parameter sharing & DP averaging
        ref_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
        ref_pg_collection = _get_pg_collection_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg_collection
        )
        _avg_params(
            ref_block, ref_grid.get_pg("dp")
        )  # Synchronize parameters across DP for reproducibility

        # ====== Create two transformer block+grid pairs with different TP/DP settings ======
        block_grid_1, grid_1 = get_transformer_block_and_grid(
            ref_block,
            tp_size=grid1_tp,
            pp_size=grid1_pp,
            dp_size=grid1_dp,
            grid_offset=0,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        block_grid_2, grid_2 = get_transformer_block_and_grid(
            ref_block,
            tp_size=grid2_tp,
            pp_size=grid2_pp,
            dp_size=grid2_dp,
            grid_offset=grid_1.size,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        dist.barrier()  # Synchronize ranks before communication

        # Module-grid map and pipeline communication topology
        module_to_grid_map = {'image_encoder': grid_1, 'llm': grid_2}
        topology = {
            'image_encoder': ['llm'],  # image_encoder sends forward results to llm
            'llm': [],  # llm is the last stage here
        }
        config = ModelParallelConfig(pipeline_dtype=torch.float)
        mllm_comm = MultiModulePipelineCommunicator(
            module_to_grid_map, topology, config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
        )

        output_grid_2 = None
        # If current rank is in the first grid, run first block and send output
        if grid_1 is not None and mllm_comm.is_current_rank_in_grid(grid_1):
            rank_module_info = mllm_comm.rank_module_map['image_encoder']
            if rank_module_info.pp_rank == 0:
                hidden_states = block_grid_1(hidden_states=hidden_states, attention_mask=None)
                mllm_comm.send_forward({'image_encoder': hidden_states})
            else:
                input_dict = mllm_comm.recv_forward(
                    tensor_shape=(sequence_length, micro_batch_size, hidden_size)
                )
                hidden_states = input_dict['image_encoder']
                hidden_states = block_grid_1(hidden_states=hidden_states, attention_mask=None)
                mllm_comm.send_forward({'image_encoder': hidden_states})

        # If current rank is in second grid, receive and run the second block
        if grid_2 is not None and mllm_comm.is_current_rank_in_grid(grid_2):
            rank_module_info = mllm_comm.rank_module_map['llm']
            if rank_module_info.pp_rank == 0:
                input_dict = mllm_comm.recv_forward()
                hidden_states = input_dict['image_encoder']
                hidden_states = block_grid_2(hidden_states=hidden_states, attention_mask=None)
                if rank_module_info.pp_rank == rank_module_info.pp_size - 1:
                    output_grid_2 = hidden_states
                else:
                    mllm_comm.send_forward({'llm': hidden_states})
            elif rank_module_info.pp_rank < rank_module_info.pp_size - 1:
                input_dict = mllm_comm.recv_forward(
                    tensor_shape=(
                        sequence_length,
                        (grid1_dp * micro_batch_size) // grid2_dp,
                        hidden_size,
                    )
                )
                hidden_states = input_dict['llm']
                hidden_states = block_grid_2(hidden_states=hidden_states, attention_mask=None)
                mllm_comm.send_forward({'llm': hidden_states})
            else:
                input_dict = mllm_comm.recv_forward(
                    tensor_shape=(
                        sequence_length,
                        (grid1_dp * micro_batch_size) // grid2_dp,
                        hidden_size,
                    )
                )
                hidden_states = input_dict['llm']
                output_grid_2 = block_grid_2(hidden_states=hidden_states, attention_mask=None)

                # Compute expected output shape based on change in DP size (chunk/expand batch dimension appropriately)
                factor = max(grid1_dp, grid2_dp) // min(grid1_dp, grid2_dp)
                expected_output_shape = (
                    sequence_length,
                    (
                        micro_batch_size * factor
                        if grid1_dp > grid2_dp
                        else micro_batch_size // factor
                    ),
                    hidden_size,
                )
                assert (
                    output_grid_2.shape == expected_output_shape
                ), f"Output2 shape mismatch: {output_grid_2.shape}"

        # ====== Reference: global (replicated) pipeline forward for correctness checking ======
        global_block_1, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_block_2, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        for i in range(grid1_pp):
            hidden_states_ref = global_block_1(hidden_states=hidden_states_ref, attention_mask=None)

        for i in range(grid2_pp):
            hidden_states_ref = global_block_2(hidden_states=hidden_states_ref, attention_mask=None)

        # Output comparison under different DP compositions between grids
        if (
            grid_2 is not None
            and mllm_comm.is_current_rank_in_grid(grid_2)
            and rank_module_info.pp_rank == rank_module_info.pp_size - 1
        ):
            if grid1_dp == grid2_dp:
                # DP size matches: all outputs directly compared
                torch.testing.assert_close(hidden_states_ref, output_grid_2, rtol=1e-3, atol=1e-3)
            elif grid1_dp < grid2_dp:
                # If grid2 expands DP: each output_grid_2 chunk corresponds to a split of the reference output
                grid2_dp_ranks = grid_2._gen_rank_enum([x for x in grid_2.dim_names if x != "dp"])
                global_block_2_chunks = torch.split(
                    hidden_states_ref, hidden_states_ref.shape[1] // (grid2_dp // grid1_dp), dim=1
                )
                relevant_chunk = None
                for i, dp_ranks in enumerate(grid2_dp_ranks):
                    if current_rank in dp_ranks:
                        relevant_chunk = global_block_2_chunks[i % len(global_block_2_chunks)]
                torch.testing.assert_close(relevant_chunk, output_grid_2, rtol=1e-3, atol=1e-3)
            else:
                # If DP shrinks (grid1_dp > grid2_dp): just compare the relevant first chunk
                output_grid_2_first_chunk = torch.chunk(output_grid_2, grid1_dp // grid2_dp, dim=1)[
                    0
                ]
                torch.testing.assert_close(
                    hidden_states_ref, output_grid_2_first_chunk, rtol=1e-3, atol=1e-3
                )
