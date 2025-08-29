import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multi_module_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _create_transformer_block(
    dtype=torch.bfloat16, hidden_size=4096, model_comm_pgs=None
) -> TransformerBlock:
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(123)
    if model_comm_pgs is not None:
        cp_size = model_comm_pgs.cp.size()
    else:
        cp_size = get_context_parallel_group().size()
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        context_parallel_size=cp_size,
    )

    block = (
        TransformerBlock(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec(),
            model_comm_pgs=model_comm_pgs,
        )
        .cuda()
        .to(dtype)
    )
    with torch.no_grad():
        for mod in block.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.zero_()
    return block


def _shard_and_copy_(
    ref_block: TransformerBlock, tgt_block: TransformerBlock, tp_size: int, tp_rank: int
) -> None:
    """Copy weights from *ref_block* into a tensor-parallel *tgt_block*."""

    ref_sd = ref_block.state_dict()
    tgt_sd = tgt_block.state_dict()

    for name, tgt_param in tgt_sd.items():
        full_param = ref_sd[name]

        # Exact match – just copy.
        if full_param.shape == tgt_param.shape:
            tgt_param.copy_(full_param)
            continue

        # ColumnParallel: shard along dim-0.
        if tgt_param.shape[0] * tp_size == full_param.shape[0]:
            slice_ = torch.chunk(full_param, tp_size, dim=0)[tp_rank]
            tgt_param.copy_(slice_)
            continue

        # RowParallel: shard along dim-1.
        if tgt_param.shape[1] * tp_size == full_param.shape[1]:
            slice_ = torch.chunk(full_param, tp_size, dim=1)[tp_rank]
            tgt_param.copy_(slice_)
            continue

        raise RuntimeError(
            f"Unhandled TP sharding for {name}: ref {full_param.shape} tgt {tgt_param.shape}"
        )


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2."""
    # Set up environment for world size 8 if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"

    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    return grid


def _get_model_comm_pgs_from_grid(grid):
    model_comm_pgs = ModelCommProcessGroups()
    model_comm_pgs.tp = grid.get_pg("tp")
    model_comm_pgs.cp = grid.get_pg("cp")
    model_comm_pgs.pp = grid.get_pg("pp")
    return model_comm_pgs


def _avg_params(module: torch.nn.Module, group: dist.ProcessGroup = None) -> None:
    world = dist.get_world_size(group=group or dist.group.WORLD)
    for p in module.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM, group=group or dist.group.WORLD)
        p.data.div_(world)


def get_transformer_block_and_grid(
    ref_block,
    tp_size=1,
    cp_size=1,
    pp_size=1,
    dp_size=1,
    grid_offset: int = 0,
    use_global_parallel_state: bool = False,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
):
    """Utility to build a ``TransformerBlock`` for tests."""

    current_rank = dist.get_rank()
    if use_global_parallel_state:
        block = _create_transformer_block(dtype=dtype, hidden_size=hidden_size)
        _shard_and_copy_(ref_block, block, tp_size, get_tensor_model_parallel_rank())
        grid = None
    else:
        grid = create_hypercomm_grid(
            offset=grid_offset, tp=tp_size, cp=cp_size, pp=pp_size, dp=dp_size
        )
        if grid.rank_offset <= current_rank < grid.rank_offset + grid.size:
            model_comm_pgs = _get_model_comm_pgs_from_grid(grid)
            block = _create_transformer_block(
                dtype=dtype, hidden_size=hidden_size, model_comm_pgs=model_comm_pgs
            )
            _shard_and_copy_(ref_block, block, tp_size, model_comm_pgs.tp.rank())
        else:
            block = None

    return block, grid


class TestMultiModulePipelineCommunicator:
    """Test suite for MultiModulePipelineCommunicator usage."""

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    @classmethod
    def teardown_class(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_mllm_communicator_init(self):
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

    def test_compute_total_pipeline_stages_overall_and_till_rank(self):
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

    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""

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

    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test send_forward_recv_backward and send_backward_recv_forward operations."""

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

        # Encoder stages: send forward tensor, receive backward gradient
        if mllm_comm.is_current_rank_in_grid(image_encoder_grid):
            output_dict = {'image_encoder': torch.randn(16, 256, 512).cuda()}
            received_grad = mllm_comm.send_forward_recv_backward(output_dict)
            assert received_grad['image_encoder'].shape == (16, 256, 512)
        if mllm_comm.is_current_rank_in_grid(audio_encoder_grid):
            output_dict = {'audio_encoder': torch.randn(16, 128, 512).cuda()}
            received_grad = mllm_comm.send_forward_recv_backward(output_dict)
            assert received_grad['audio_encoder'].shape == (16, 128, 512)

        # LLM: receives backward (from generator) then immediately receives forward (from encoders)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                grad_dict = {
                    'image_encoder': torch.randn(16, 256, 512).cuda(),
                    'audio_encoder': torch.randn(16, 128, 512).cuda(),
                }
                input_dict = mllm_comm.send_backward_recv_forward(grad_dict)
                assert input_dict['image_encoder'].shape == (16, 256, 512)
                assert input_dict['audio_encoder'].shape == (16, 128, 512)

        # LLM: send forward (as LLM) and receive backward
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 2 or dist.get_rank() == 3:
                output_dict = {'llm': torch.randn(16, 128, 512).cuda()}
                received_grad = mllm_comm.send_forward_recv_backward(
                    output_dict, tensor_shape=(16, 128, 512)
                )
                assert received_grad['llm'].shape == (16, 128, 512)
            if dist.get_rank() == 4 or dist.get_rank() == 5:
                grad_dict = {'llm': torch.randn(16, 128, 512).cuda()}
                input_dict = mllm_comm.send_backward_recv_forward(
                    grad_dict, tensor_shape=(16, 128, 512)
                )
                assert input_dict['llm'].shape == (16, 128, 512)

        # LLM: send forward and get gradient (2nd stage)
        if mllm_comm.is_current_rank_in_grid(llm_grid):
            if dist.get_rank() == 4 or dist.get_rank() == 5:
                output_dict = {'llm': torch.randn(16, 128, 512).cuda()}
                received_grad = mllm_comm.send_forward_recv_backward(output_dict)
                assert received_grad['llm'].shape == (16, 128, 512)
        # Generator: send backward gradient, receive forward activation
        if mllm_comm.is_current_rank_in_grid(generator_grid):
            grad_dict = {'llm': torch.randn(8, 128, 512).cuda()}
            input_dict = mllm_comm.send_backward_recv_forward(grad_dict)
            assert input_dict['llm'].shape == (8, 128, 512)

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
        ref_model_comm_pgs = _get_model_comm_pgs_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=ref_model_comm_pgs
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
        global_llm_block_pp_stage_0, _ = get_transformer_block_and_grid(
            ref_block,
            tp_size=parallel_state_tp,
            use_global_parallel_state=True,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        global_llm_block_pp_stage_1, _ = get_transformer_block_and_grid(
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
        global_llm_pp_stage_0_output = global_llm_block_pp_stage_0(
            hidden_states=global_llm_input, attention_mask=None
        )
        if current_rank == 2 or current_rank == 3:
            torch.testing.assert_close(
                global_llm_pp_stage_0_output, llm_output, rtol=1e-3, atol=1e-3
            )
        global_llm_pp_stage_1_output = global_llm_block_pp_stage_1(
            hidden_states=global_llm_pp_stage_0_output, attention_mask=None
        )
        if current_rank == 4 or current_rank == 5:
            torch.testing.assert_close(
                global_llm_pp_stage_1_output, llm_output, rtol=1e-3, atol=1e-3
            )

        # Generator output and comparison to distributed output (for each DP chunk)
        global_generator_block_output = global_generator_block(
            hidden_states=global_llm_pp_stage_1_output, attention_mask=None
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

        # ========== Clean up model-parallel state ==========
        Utils.destroy_model_parallel()

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
        hidden_size = 1024
        sequence_length = 16
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
        ref_model_comm_pgs = _get_model_comm_pgs_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=ref_model_comm_pgs
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
            if rank_module_info.pp_stage == 0:
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
            if rank_module_info.pp_stage == 0:
                input_dict = mllm_comm.recv_forward()
                hidden_states = input_dict['image_encoder']
                hidden_states = block_grid_2(hidden_states=hidden_states, attention_mask=None)
                if rank_module_info.pp_stage == rank_module_info.pp_size - 1:
                    output_grid_2 = hidden_states
                else:
                    mllm_comm.send_forward({'llm': hidden_states})
            elif rank_module_info.pp_stage < rank_module_info.pp_size - 1:
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
            and rank_module_info.pp_stage == rank_module_info.pp_size - 1
        ):
            if grid1_dp == grid2_dp:
                # DP size matches: all outputs directly compared
                torch.testing.assert_close(hidden_states_ref, output_grid_2, rtol=1e-3, atol=1e-3)
            # elif grid1_dp < grid2_dp:
            #     # If grid2 expands DP: each output_grid_2 chunk corresponds to a split of the reference output
            #     grid2_dp_ranks = grid_2._gen_rank_enum([x for x in grid_2.dim_names if x != "dp"])
            #     global_block_2_chunks = torch.split(
            #         hidden_states_ref,
            #         hidden_states_ref.shape[1] // (grid2_dp // grid1_dp),
            #         dim=1,
            #     )
            #     relevant_chunk = None
            #     for i, dp_ranks in enumerate(grid2_dp_ranks):
            #         if current_rank in dp_ranks:
            #             relevant_chunk = global_block_2_chunks[i % len(global_block_2_chunks)]
            #     torch.testing.assert_close(relevant_chunk, output_grid_2, rtol=1e-3, atol=1e-3)
            # else:
            #     # If DP shrinks (grid1_dp > grid2_dp): just compare the relevant first chunk
            #     output_grid_2_first_chunk = torch.chunk(output_grid_2, grid1_dp // grid2_dp, dim=1)[
            #         0
            #     ]
            #     torch.testing.assert_close(
            #         hidden_states_ref, output_grid_2_first_chunk, rtol=1e-3, atol=1e-3
            #     )

        Utils.destroy_model_parallel()  # Clean up parallel context
