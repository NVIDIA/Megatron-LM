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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _create_transformer_block(
    dtype=torch.bfloat16, hidden_size=4096, pg_collection=None
) -> TransformerBlock:
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(
        123,
        tp_rank=(
            pg_collection.tp.rank()
            if pg_collection is not None
            else get_tensor_model_parallel_rank()
        ),
        ep_rank=torch.distributed.get_rank(),
        etp_rank=torch.distributed.get_rank(),
    )
    if pg_collection is not None:
        cp_size = pg_collection.cp.size()
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
            pg_collection=pg_collection,
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

        # Skip non-tensor entries (e.g., _metadata or other buffers stored as BytesIO).
        if not (torch.is_tensor(tgt_param) and torch.is_tensor(full_param)):
            logging.info(f'_shard_and_copy_ skipping non-tensor entry: {name}')
            continue

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
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized")

    #  tests below assume a world size of 8
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


def _get_pg_collection_from_grid(grid):
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    return pg_collection


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
            pg_collection = _get_pg_collection_from_grid(grid)
            block = _create_transformer_block(
                dtype=dtype, hidden_size=hidden_size, pg_collection=pg_collection
            )
            _shard_and_copy_(ref_block, block, tp_size, pg_collection.tp.rank())
        else:
            block = None

    return block, grid


class TestBridgeCommunicator:

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

    def test_bridge_communicator_init(self):

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=1, dp=2)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid is grid1
        assert bridge_communicator.dest_grid is grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None

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
