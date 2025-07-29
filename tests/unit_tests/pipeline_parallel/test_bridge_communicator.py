import os
import sys
import pytest
import torch
import torch.distributed as dist
from packaging import version
import logging
from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import get_tensor_model_parallel_rank, get_context_parallel_group
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from tests.unit_tests.test_utilities import Utils


logging.basicConfig(
    level=logging.INFO,              # emit INFO and above
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,               # send to stdout so it lands in stdout.log
    force=True,                      # override any existing handlers (Py ≥3.8)
)

def _create_transformer_block(hidden_size = 4096, model_comm_pgs=None) -> TransformerBlock:
    """Build a *non-sharded* TransformerBlock (tp=cp=dp=1).

    All ranks build an identical copy; parameters will later be sharded and
    copied into the real model-parallel blocks.
    """
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(123)
    if model_comm_pgs is not None:
        cp_size = model_comm_pgs.cp.size()
    else:
        cp_size = get_context_parallel_group().size()
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        context_parallel_size=cp_size,
    )

    block = TransformerBlock(transformer_config, get_gpt_layer_with_transformer_engine_spec(), model_comm_pgs=model_comm_pgs).cuda().bfloat16()
    return block


def _shard_and_copy_(ref_block: TransformerBlock, tgt_block: TransformerBlock, tp_size: int, tp_rank: int) -> None:
    """Copy weights from *ref_block* into a tensor-parallel *tgt_block*.

    The copy handles simple TP sharding by slicing the reference parameter on
    dim-0 (ColumnParallelLinear) or dim-1 (RowParallelLinear) depending on the
    target shape. Parameters whose shape already matches are copied verbatim.
    """

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

        # Fallback – dimensions do not match expected pattern.
        raise RuntimeError(f"Unhandled TP sharding for {name}: ref {full_param.shape} tgt {tgt_param.shape}")


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """
    Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2.

    Returns:
        HyperCommGrid: A grid configured with tp=2, cp=2, dp=2 (total size = 8).
    """
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

def get_model_comm_pgs_from_grid(grid):
    model_comm_pgs = ModelCommProcessGroups()
    model_comm_pgs.tp = grid.get_pg("tp")
    model_comm_pgs.cp = grid.get_pg("cp")
    model_comm_pgs.pp = grid.get_pg("pp")
    return model_comm_pgs

def _avg_params(module: torch.nn.Module, group: dist.ProcessGroup = None) -> None:
    """Average parameters across a (data-parallel) process group."""
    world = dist.get_world_size(group=group or dist.group.WORLD)
    for p in module.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM, group=group or dist.group.WORLD)
        p.data.div_(world)

def get_transformer_block_and_grid(tp_size, cp_size, pp_size, dp_size, grid_offset: int = 0, use_global_parallel_state: bool = False, hidden_size: int = 4096):
    """Utility to build a ``TransformerBlock`` for tests."""
    ref_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
    ref_model_comm_pgs = get_model_comm_pgs_from_grid(ref_grid)
    ref_block = _create_transformer_block(hidden_size, ref_model_comm_pgs)
    _avg_params(ref_block, ref_grid.get_pg("dp"))
    current_rank = dist.get_rank()
    if use_global_parallel_state:
        block = _create_transformer_block()
        _shard_and_copy_(ref_block, block, tp_size, get_tensor_model_parallel_rank())
        grid = None
        # block = ref_block
        # grid = ref_grid
    else:
        grid = create_hypercomm_grid(offset=grid_offset, tp=tp_size, cp=cp_size, pp=pp_size, dp=dp_size)
        if grid.rank_offset <= current_rank < grid.rank_offset + grid.size:
            model_comm_pgs = get_model_comm_pgs_from_grid(grid)
            block = _create_transformer_block(hidden_size, model_comm_pgs)
            _shard_and_copy_(ref_block, block, tp_size, model_comm_pgs.tp.rank())
        else:
            block = None

    return block, grid


class TestBridgeCommunicator:
    """Test suite for BridgeCommunicator usage."""

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
        Utils.initialize_model_parallel(tensor_model_parallel_size=4, context_parallel_size=1)

    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_bridge_communicator_init(self):

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None

    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test combined send_forward_recv_backward and send_backward_recv_forward operations."""

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)

        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = torch.randn(16, 128, 512).cuda()
            received_grad = bridge_communicator.send_forward_recv_backward(
                random_hidden_state, dtype=random_hidden_state.dtype
            )

            # Assert that the returned gradient tensor is valid
            assert received_grad is not None, "send_forward_recv_backward should return a gradient tensor"
            assert isinstance(received_grad, torch.Tensor), f"Expected torch.Tensor, got {type(received_grad)}"
            assert received_grad.shape == random_hidden_state.shape, f"Expected gradient shape {random_hidden_state.shape}, got {received_grad.shape}"
            assert received_grad.device == random_hidden_state.device, f"Expected device {random_hidden_state.device}, got {received_grad.device}"

        else:
            random_grad_state = torch.randn(16, 128, 512).cuda()
            received_activation = bridge_communicator.send_backward_recv_forward(
                random_grad_state, dtype=random_grad_state.dtype
            )

            # Assert that the returned activation tensor is valid
            assert received_activation is not None, "send_backward_recv_forward should return an activation tensor"
            assert isinstance(received_activation, torch.Tensor), f"Expected torch.Tensor, got {type(received_activation)}"
            assert received_activation.shape == random_grad_state.shape, f"Expected activation shape {random_grad_state.shape}, got {received_activation.shape}"
            assert received_activation.device == random_grad_state.device, f"Expected device {random_grad_state.device}, got {received_activation.device}"

    def test_send_forward_recv_forward(self):
        """Test send_forward and recv_forward operations."""
        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)

        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        random_hidden_state = torch.randn(16, 128, 512)
        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = random_hidden_state.cuda()
            bridge_communicator.send_forward(random_hidden_state)

        else:
            received_activation = bridge_communicator.receive_forward(
                dtype=random_hidden_state.dtype
            )
            assert received_activation.shape == (16, 128, 512), f"Expected activation shape {(16, 128, 512)}, got {received_activation.shape}"

    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)

        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        random_grad_state = torch.randn(16, 128, 512)
        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.dest_grid):
            # In backward pass, gradients flow from destination grid back to source grid
            random_grad_state = random_grad_state.cuda()
            bridge_communicator.send_backward(random_grad_state)

        else:
            received_gradient = bridge_communicator.receive_backward(dtype=random_grad_state.dtype)
            # Assert that the returned gradient tensor is valid
            assert received_gradient is not None, "recv_backward should return a gradient tensor"
            assert isinstance(received_gradient, torch.Tensor), f"Expected torch.Tensor, got {type(received_gradient)}"
            assert received_gradient.shape == (16, 128, 512), f"Expected gradient shape {(16, 128, 512)}, got {received_gradient.shape}"


    @pytest.mark.parametrize(
        "grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid2_tp, grid2_cp, grid2_pp, grid2_dp, mbs",
        [
             (4, 1, 1, 1, 4, 1, 1, 1, 2),
            # (1, 4, 1, 1, 4, 1, 1, 1, 2),  # Current setup: Grid1 cp=4, Grid2 tp=4,
            # (1, 4, 1, 1, 1, 1, 1, 4, 8),  # Fan-out test
            # (1, 1, 1, 4, 4, 1, 1, 1, 8),  # Fan-in test
            # (2, 1, 1, 2, 2, 1, 1, 2, 8),  # Multiple dp groups test
            # (1, 1, 1, 4, 2, 1, 1, 2, 8),  # Multiple dp groups test different dp sizes
        ],
    )
    def test_bridge_communicator_with_transformer_blocks(
        self, grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid2_tp, grid2_cp, grid2_pp, grid2_dp, mbs
    ):
        """Test bridge communicator with two transformer blocks having different process group configurations."""

        os.environ["NVTE_FLASH_ATTN"] = "1"
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

        hidden_size = 4096
        sequence_length = 2048
        micro_batch_size = mbs
        torch.manual_seed(12345)
        hidden_states = torch.randn((sequence_length, micro_batch_size, hidden_size), device="cuda").bfloat16()
        current_rank = dist.get_rank()

        block_grid_1, grid_1 = get_transformer_block_and_grid(
            grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid_offset=0, hidden_size=hidden_size
        )
        block_grid_2, grid_2 = get_transformer_block_and_grid(
            grid2_tp, grid2_cp, grid2_pp, grid2_dp, grid_offset=4, hidden_size=hidden_size
        )

        dist.barrier()

        # Create bridge communicator linking the two grids.
        bridge_communicator = BridgeCommunicator(
            grid_1, grid_2, dim_mapping={'s': 0, 'h': 2, 'b': 1}, requires_scatter_gather=False
        )
        print(f" rank {current_rank} Calling grid_1 send forward grid_1 {grid_1} grid_2 {grid_2}")
        if grid_1 is not None and bridge_communicator.is_current_rank_in_grid(grid_1):
            # Send forward activation to grid2
            print(f" rank {current_rank} Calling block_grid_1")
            output_grid_1 = block_grid_1(hidden_states=hidden_states, attention_mask=None)
            print(f" Grid 1 rank {dist.get_rank()}: Sending activation shape {output_grid_1.shape} sum {output_grid_1.sum()}")
            bridge_communicator.send_forward(output_grid_1)

        print(f" rank {current_rank} Calling grid_2 receive forward grid_1 {grid_1} grid_2 {grid_2}")
        # Grid 2 Forward receive
        if grid_2 is not None and bridge_communicator.is_current_rank_in_grid(grid_2):
            received_activation = bridge_communicator.receive_forward(dtype=torch.bfloat16)
            assert received_activation is not None, "Should receive activation from grid1"
            logging.info(f" Grid 2 rank {dist.get_rank()}: Received activation shape {received_activation.shape} sum {received_activation.sum()}")

            output_grid_2 = block_grid_2(hidden_states=received_activation, attention_mask=None)
            factor = max(grid1_dp, grid2_dp) // min(grid1_dp, grid2_dp)
            expected_output_shape = (
                sequence_length,
                micro_batch_size * factor if grid1_dp > grid2_dp else micro_batch_size // factor,
                hidden_size,
            )
           
            assert output_grid_2.shape == expected_output_shape, f"Output2 shape mismatch: {output_grid_2.shape}"

            logging.info(f" Grid 2 rank {dist.get_rank()}: forward pass with output shape {output_grid_2.shape} sum {output_grid_2.sum()}")

       
        
        global_block_1, _ = get_transformer_block_and_grid(4,1,1,2, use_global_parallel_state=True)
        global_block_2, _ = get_transformer_block_and_grid(4,1,1,2, use_global_parallel_state=True)

        print(f"rank {dist.get_rank()} ")
        if dist.get_rank() == 0 or dist.get_rank() == 3:
             print(f" [rank {dist.get_rank()}] block_grid_1 {block_grid_1.layers[0].mlp.linear_fc1.weight.data[0]} sum {block_grid_1.layers[0].mlp.linear_fc1.weight.data.sum()}")
             print(f" [rank {dist.get_rank()}] global_block_1 {global_block_1.layers[0].mlp.linear_fc1.weight.data[0]} sum {global_block_1.layers[0].mlp.linear_fc1.weight.data.sum()}")
        if dist.get_rank() == 4 or dist.get_rank() == 7:
             print(f" [rank {dist.get_rank()}] block_grid_2 {block_grid_2.layers[0].mlp.linear_fc1.weight.data[0]} sum {block_grid_2.layers[0].mlp.linear_fc1.weight.data.sum()}")
             print(f" [rank {dist.get_rank()}] global_block_2 {global_block_2.layers[0].mlp.linear_fc1.weight.data[0]} sum {global_block_2.layers[0].mlp.linear_fc1.weight.data.sum()}")
        # dist.barrier()
        global_block_1_output = global_block_1(hidden_states=hidden_states, attention_mask=None)
        print(f"rank {dist.get_rank()}: block 1 output sum {global_block_1_output.sum()}")
        # print(f" rank {dist.get_rank()}: input hidden states sum {hidden_states.sum()} output block 1 sum {block_1_output.sum()} shape {block_1_output.shape}")
        # dist.barrier()
        global_block_2_output = global_block_2(hidden_states=global_block_1_output, attention_mask=None)
        print(f"rank {dist.get_rank()}: full block output sum {global_block_2_output.sum()}")
        # print(f" rank {dist.get_rank()}: input block 1 output sum {block_1_output.sum()} output block 2 sum {full_block_output.sum()} shape {full_block_output.shape}")
        # dist.barrier()
        # if dist.get_rank() == 0:
        #     breakpoint()
        # dist.barrier()
        if bridge_communicator.is_current_rank_in_grid(grid_2):
            print(f"rank {dist.get_rank()}: output2 shape {output_grid_2.shape} full_block_output shape {global_block_2_output.shape}")
            torch.testing.assert_close(global_block_2_output, output_grid_2, rtol=1e-3, atol=1e-3)
