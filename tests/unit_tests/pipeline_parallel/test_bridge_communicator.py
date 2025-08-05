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
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)


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

    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_bridge_communicator_init(self):

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=1, dp=2)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None

    def test_send_forward_recv_forward(self):
        """Test send_forward and recv_forward operations."""

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        grid2 = create_hypercomm_grid(offset=4, tp=4, cp=1, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2, comm_dtype=torch.float32)

        random_hidden_state = torch.randn(16, 128, 512)
        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = random_hidden_state.cuda()
            bridge_communicator.send_forward(random_hidden_state)

        else:
            received_activation = bridge_communicator.recv_forward()
            # default assunes bsh
            assert received_activation.shape == (
                32,
                128,
                512,
            ), f"Expected activation shape {(32, 128, 512)}, got {received_activation.shape}"

    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=4)
        grid2 = create_hypercomm_grid(offset=4, tp=4, cp=1, pp=1, dp=1)
        random_grad_state = torch.randn(16, 128, 512)
        bridge_communicator = BridgeCommunicator(grid1, grid2, comm_dtype=torch.float32)

        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.dest_grid):
            random_grad_state = random_grad_state.cuda()
            bridge_communicator.send_backward(random_grad_state)

        else:
            received_gradient = bridge_communicator.recv_backward()
            assert received_gradient.shape == (
                4,
                128,
                512,
            ), f"Expected gradient shape {(4, 128, 512)}, got {received_gradient.shape}"

    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test combined send_forward_recv_backward and send_backward_recv_forward operations."""

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        grid2 = create_hypercomm_grid(offset=4, tp=4, cp=1, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2, comm_dtype=torch.float32)

        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = torch.randn(16, 128, 512).cuda()
            received_grad = bridge_communicator.send_forward_recv_backward(random_hidden_state)
            assert (
                received_grad.shape == random_hidden_state.shape
            ), f"Expected gradient shape {random_hidden_state.shape}, got {received_grad.shape}"

        else:
            random_grad_state = torch.randn(32, 128, 512).cuda()
            received_activation = bridge_communicator.send_backward_recv_forward(random_grad_state)

            assert received_activation.shape == (
                32,
                128,
                512,
            ), f"Expected activation shape {random_grad_state.shape}, got {received_activation.shape}"

    @pytest.mark.parametrize(
        "grid1_tp, grid1_dp, grid2_tp, grid2_dp, parallel_state_tp",
        [
            (2, 1, 2, 1, 2),  # TP2DP2 to TP4DP1
            (4, 1, 4, 1, 4),  # TP4DP1 to TP4DP1
            (2, 2, 4, 1, 2),  # TP2DP2 to TP4DP1
            (4, 1, 2, 2, 2),  # TP4DP1 to TP2DP2
        ],
    )
    def test_bridge_communicator_with_transformer_blocks(
        self, grid1_tp, grid1_dp, grid2_tp, grid2_dp, parallel_state_tp
    ):
        """Test bridge communicator with two transformer blocks having different process group configurations."""

        hidden_size = 1024
        sequence_length = 16
        micro_batch_size = 8
        torch.manual_seed(12345)
        dtype = torch.float32
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, hidden_size), device="cuda"
        ).to(dtype)
        current_rank = dist.get_rank()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=parallel_state_tp, create_gloo_process_groups=False
        )

        ref_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
        ref_model_comm_pgs = _get_model_comm_pgs_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=ref_model_comm_pgs
        )
        _avg_params(ref_block, ref_grid.get_pg("dp"))

        block_grid_1, grid_1 = get_transformer_block_and_grid(
            ref_block,
            tp_size=grid1_tp,
            dp_size=grid1_dp,
            grid_offset=0,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        block_grid_2, grid_2 = get_transformer_block_and_grid(
            ref_block,
            tp_size=grid2_tp,
            dp_size=grid2_dp,
            grid_offset=4,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        dist.barrier()
        bridge_communicator = BridgeCommunicator(
            grid_1, grid_2, dim_mapping={'s': 0, 'h': 2, 'b': 1}, comm_dtype=dtype
        )
        output_grid_2 = None
        if grid_1 is not None and bridge_communicator.is_current_rank_in_grid(grid_1):
            output_grid_1 = block_grid_1(hidden_states=hidden_states, attention_mask=None)
            bridge_communicator.send_forward(output_grid_1)

        if grid_2 is not None and bridge_communicator.is_current_rank_in_grid(grid_2):
            received_activation = bridge_communicator.recv_forward()
            output_grid_2 = block_grid_2(hidden_states=received_activation, attention_mask=None)
            factor = max(grid1_dp, grid2_dp) // min(grid1_dp, grid2_dp)
            expected_output_shape = (
                sequence_length,
                micro_batch_size * factor if grid1_dp > grid2_dp else micro_batch_size // factor,
                hidden_size,
            )

            assert (
                output_grid_2.shape == expected_output_shape
            ), f"Output2 shape mismatch: {output_grid_2.shape}"

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

        global_block_1_output = global_block_1(hidden_states=hidden_states, attention_mask=None)
        global_block_2_output = global_block_2(
            hidden_states=global_block_1_output, attention_mask=None
        )

        if grid_2 is not None and bridge_communicator.is_current_rank_in_grid(grid_2):
            if grid1_dp == grid2_dp:
                torch.testing.assert_close(
                    global_block_2_output, output_grid_2, rtol=1e-3, atol=1e-3
                )
            elif grid1_dp < grid2_dp:
                print(
                    f"output_grid_2 shape: {output_grid_2.shape} global_block_2_output shape: {global_block_2_output.shape}"
                )
                grid2_dp_ranks = grid_2._gen_rank_enum([x for x in grid_2.dim_names if x != "dp"])
                global_block_2_chunks = torch.split(
                    global_block_2_output, global_block_2_output.shape[1] // grid2_dp, dim=1
                )
                relevant_chunk = None
                for i, dp_ranks in enumerate(grid2_dp_ranks):
                    if current_rank in dp_ranks:
                        relevant_chunk = global_block_2_chunks[i]
                torch.testing.assert_close(relevant_chunk, output_grid_2, rtol=1e-3, atol=1e-3)
            else:
                output_grid_2_first_chunk = torch.chunk(output_grid_2, grid1_dp // grid2_dp, dim=1)[
                    0
                ]
                torch.testing.assert_close(
                    global_block_2_output, output_grid_2_first_chunk, rtol=1e-3, atol=1e-3
                )

        Utils.destroy_model_parallel()

    def test_tranformer_block_with_different_parallelisms(self):
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"

        torch.manual_seed(12345)

        # Initialize model parallel state (required for model_parallel_cuda_manual_seed)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, create_gloo_process_groups=False
        )

        hidden_size = 2048
        dtype = torch.float32

        sequence_length = 8192
        micro_batch_size = 2
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, hidden_size), device="cuda"
        ).to(dtype)

        ref_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=8)
        ref_model_comm_pgs = _get_model_comm_pgs_from_grid(ref_grid)
        ref_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=ref_model_comm_pgs
        )
        _avg_params(ref_block, ref_grid.get_pg("dp"))

        # tp8 dp 1 grid
        block1, grid_1 = get_transformer_block_and_grid(
            tp_size=8,
            cp_size=1,
            pp_size=1,
            dp_size=1,
            ref_block=ref_block,
            dtype=dtype,
            hidden_size=hidden_size,
        )

        # tp4 dp 2 grid
        block2, grid_2 = get_transformer_block_and_grid(
            tp_size=4,
            cp_size=1,
            pp_size=1,
            dp_size=2,
            ref_block=ref_block,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        dist.barrier()

        output_grid_1 = block1(hidden_states=hidden_states, attention_mask=None)

        output_grid_2 = block2(hidden_states=hidden_states, attention_mask=None)

        logging.debug(
            f"Rank {dist.get_rank()}: shapes - grid 1 {output_grid_1.shape} grid 2 {output_grid_2.shape}"
        )
        logging.debug(
            f"Rank {dist.get_rank()}: sum -  grid 1 {output_grid_1.sum()} grid 2 {output_grid_2.sum()}"
        )

        torch.testing.assert_close(output_grid_1, output_grid_2, rtol=1e-3, atol=1e-3)

        # Clean up model parallel state
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "tp, cp, pp, dp, expected_src_ranks, expected_dest_ranks",
        [
            # Test Case 1: tp=2, cp=1, pp=2, dp=2
            (2, 1, 2, 2, [[2, 3], [6, 7]], [[0, 1], [4, 5]]),
            # Test Case 2: tp=4, cp=1, pp=2, dp=1
            (4, 1, 2, 1, [[4, 5, 6, 7]], [[0, 1, 2, 3]]),
            # Test Case 3: tp=1, cp=1, pp=2, dp=4
            (1, 1, 2, 4, [[1], [3], [5], [7]], [[0], [2], [4], [6]]),
            # Test Case 4: tp=2, cp=1, pp=4, dp=1
            (2, 1, 4, 1, [[6, 7]], [[0, 1]]),
        ],
    )
    def test_get_boundary_pp_stage_ranks(
        self, tp, cp, pp, dp, expected_src_ranks, expected_dest_ranks
    ):
        """Test get_boundary_pp_stage_ranks function with different parallelism configurations."""

        # Create grid with specified parallelism dimensions
        grid = create_hypercomm_grid(offset=0, tp=tp, cp=cp, pp=pp, dp=dp)
        bridge_communicator = BridgeCommunicator(grid, grid)  # Using same grid for simplicity

        # For source grid (is_src=True), should return ranks from last pp stage
        src_boundary_ranks = bridge_communicator.get_boundary_pp_stage_ranks(grid, is_src=True)
        assert (
            src_boundary_ranks == expected_src_ranks
        ), f"Source: Expected {expected_src_ranks}, got {src_boundary_ranks}"

        # For destination grid (is_src=False), should return ranks from first pp stage
        dest_boundary_ranks = bridge_communicator.get_boundary_pp_stage_ranks(grid, is_src=False)
        assert (
            dest_boundary_ranks == expected_dest_ranks
        ), f"Dest: Expected {expected_dest_ranks}, got {dest_boundary_ranks}"

    @pytest.mark.parametrize(
        "tp, cp, pp, dp, expected_src_leaders, expected_dest_leaders",
        [
            # Test Case 1: tp=2, cp=1, pp=2, dp=2
            (2, 1, 2, 2, [3, 7], [0, 4]),
            # Test Case 2: tp=4, cp=1, pp=2, dp=1
            (4, 1, 2, 1, [7], [0]),
            # Test Case 3: tp=1, cp=1, pp=2, dp=4
            (1, 1, 2, 4, [1, 3, 5, 7], [0, 2, 4, 6]),
            # Test Case 4: tp=2, cp=1, pp=4, dp=1
            (2, 1, 4, 1, [7], [0]),
        ],
    )
    def test_get_leader_rank(self, tp, cp, pp, dp, expected_src_leaders, expected_dest_leaders):
        """Test get_leader_rank function with different parallelism configurations."""

        # Create grid with specified parallelism dimensions
        grid = create_hypercomm_grid(offset=0, tp=tp, cp=cp, pp=pp, dp=dp)
        bridge_communicator = BridgeCommunicator(grid, grid)  # Using same grid for simplicity

        # For source grid (is_src=True), should return leader ranks from last pp stage of each dp replica
        src_leaders, _ = bridge_communicator.get_leader_rank(grid, is_src=True)
        assert (
            src_leaders == expected_src_leaders
        ), f"Source leaders: Expected {expected_src_leaders}, got {src_leaders}"

        # For destination grid (is_src=False), should return leader ranks from first pp stage of each dp replica
        dest_leaders, _ = bridge_communicator.get_leader_rank(grid, is_src=False)
        assert (
            dest_leaders == expected_dest_leaders
        ), f"Dest leaders: Expected {expected_dest_leaders}, got {dest_leaders}"
