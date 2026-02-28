# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for multimodule pipeline schedules with heterogeneous parallelism."""

from contextlib import contextmanager
from typing import Dict, Optional

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# ============================================================================
# Helper Functions
# ============================================================================


def create_hypercomm_grid(offset=0, tp=1, pp=1, dp=1):
    """Create a HyperCommGrid with specified parallelism."""
    grid = HyperCommGrid(
        shape=[tp, 1, pp, dp, 1],  # [tp, cp, pp, dp, ep]
        dim_names=["tp", "cp", "pp", "dp", "ep"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["ep"])
    return grid


def get_pg_collection(grid):
    """Get ProcessGroupCollection from grid."""
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    pg_collection.dp = grid.get_pg("dp")
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    return pg_collection


def add_embedding_groups(pg_collection):
    """Add embedding groups to process group collection."""
    if not pg_collection.pp:
        return pg_collection

    pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    # Always set pos_embd and embd (to group or None)
    pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
    pg_collection.embd = (
        embd_pg
        if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
        else None
    )

    return pg_collection


def create_transformer_block(hidden_size, pg_collection, dtype=torch.bfloat16):
    """Create a transformer block for testing."""
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(
        123,
        tp_rank=pg_collection.tp.rank(),
        ep_rank=pg_collection.ep.rank() if hasattr(pg_collection, 'ep') else 0,
        etp_rank=dist.get_rank(),
    )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=(dtype == torch.bfloat16),
    )

    block = (
        TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec(), pg_collection=pg_collection
        )
        .cuda()
        .to(dtype)
    )

    with torch.no_grad():
        for mod in block.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.zero_()

    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    block = DistributedDataParallel(
        config=block.config, ddp_config=ddp_config, module=block, pg_collection=pg_collection
    )
    block.pre_process = False
    block.post_process = False
    block.share_embeddings_and_output_weights = False
    return block


def create_module_with_grid(tp, pp, dp, grid_offset, hidden_size):
    """Create a module (transformer block) with its grid."""
    rank = dist.get_rank()
    grid = create_hypercomm_grid(offset=grid_offset, tp=tp, pp=pp, dp=dp)

    if grid.rank_offset <= rank < grid.rank_offset + grid.size:
        pg_collection = add_embedding_groups(get_pg_collection(grid))
        module = create_transformer_block(hidden_size, pg_collection)
    else:
        module = None

    return module, grid


# ============================================================================
# Model Wrapper
# ============================================================================


class MultiModuleModel(torch.nn.Module):
    """Wrapper for testing multimodule schedules with multiple encoders + LLM."""

    def __init__(self, encoder_configs, llm_config, hidden_size):
        """
        Args:
            encoder_configs: List of dicts with keys: tp, pp, dp, grid_offset, name
            llm_config: Dict with keys: tp, pp, dp, grid_offset
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = dist.get_rank()

        # Create encoders
        self.encoders = {}
        self.encoder_grids = {}
        for enc_cfg in encoder_configs:
            name = enc_cfg['name']
            module, grid = create_module_with_grid(
                enc_cfg['tp'], enc_cfg['pp'], enc_cfg['dp'], enc_cfg['grid_offset'], hidden_size
            )
            self.encoders[name] = module
            self.encoder_grids[name] = grid

        # Create LLM
        self.llm, self.llm_grid = create_module_with_grid(
            llm_config['tp'],
            llm_config['pp'],
            llm_config['dp'],
            llm_config['grid_offset'],
            hidden_size,
        )

        # Track all modules for gradient sync
        self.modules_and_grids = []
        for name, module in self.encoders.items():
            self.modules_and_grids.append((module, self.encoder_grids[name]))
        self.modules_and_grids.append((self.llm, self.llm_grid))

        # Input tensors for pipeline stages
        self.input_tensors = {name: None for name in self.encoders.keys()}
        self.input_tensors['llm'] = None

    def is_rank_in_grid(self, grid):
        """Check if current rank is in grid."""
        return grid.rank_offset <= self.rank < grid.rank_offset + grid.size

    @contextmanager
    def no_sync(self):
        """No-sync context for all active modules."""
        contexts = []
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_rank_in_grid(grid):
                contexts.append(module.no_sync())

        for ctx in contexts:
            ctx.__enter__()
        try:
            yield
        finally:
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    @property
    def ddp_config(self):
        """Get DDP config from first active module."""
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_rank_in_grid(grid):
                return module.ddp_config
        raise AttributeError(f"No active modules on rank {self.rank}")

    def finalize_model_grads(self, *args, **kwargs):
        """Finalize gradients for all active modules."""
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_rank_in_grid(grid):
                pg_collection = add_embedding_groups(get_pg_collection(grid))
                finalize_model_grads([module], num_tokens=None, pg_collection=pg_collection)

    def set_input_tensor(self, input_tensor):
        """Set input tensors from previous pipeline stage."""
        if not input_tensor or not input_tensor[0]:
            return

        tensor_dict = input_tensor[0]

        # Set encoder inputs
        for name in self.encoders.keys():
            if name in tensor_dict:
                self.input_tensors[name] = (
                    tensor_dict[name][0]
                    if isinstance(tensor_dict[name], list)
                    else tensor_dict[name]
                )

        # Set LLM input (from either encoder outputs or previous LLM stage)
        # Only do this if we're on the LLM grid
        if self.is_rank_in_grid(self.llm_grid):
            if 'llm' in tensor_dict:
                self.input_tensors['llm'] = (
                    tensor_dict['llm'][0]
                    if isinstance(tensor_dict['llm'], list)
                    else tensor_dict['llm']
                )
            elif len(self.encoders) > 0:
                # Concatenate encoder outputs for LLM input (received via bridge)
                encoder_outputs = []
                for name in self.encoders.keys():
                    if name in tensor_dict:
                        tensor = tensor_dict[name]
                        # Extract tensor from list if needed (P2P sends as list)
                        if isinstance(tensor, list):
                            tensor = tensor[0]
                        encoder_outputs.append(tensor)
                if encoder_outputs:
                    self.input_tensors['llm'] = (
                        torch.cat(encoder_outputs, dim=0)
                        if len(encoder_outputs) > 1
                        else encoder_outputs[0]
                    )

    def forward(self, hidden_states):
        """Forward pass through active modules."""
        output_dict = {}

        # Forward through encoders
        for name, encoder in self.encoders.items():
            if encoder is not None and self.is_rank_in_grid(self.encoder_grids[name]):
                pp_group = self.encoder_grids[name].get_pg("pp")
                input_tensor = (
                    hidden_states if is_pp_first_stage(pp_group) else self.input_tensors[name]
                )
                output_dict[name] = encoder(input_tensor, attention_mask=None)

        # Forward through LLM
        if self.llm is not None and self.is_rank_in_grid(self.llm_grid):
            output_dict['llm'] = self.llm(self.input_tensors['llm'], attention_mask=None)

        return output_dict


# ============================================================================
# Data Iterator
# ============================================================================


class DataIterator:
    """Simple data iterator for testing."""

    def __init__(self, hidden_size, seq_length, micro_batch_size):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return torch.randn(
            self.seq_length,
            self.micro_batch_size,
            self.hidden_size,
            device='cuda',
            dtype=torch.bfloat16,
        )


# ============================================================================
# Test Runner
# ============================================================================


def run_multimodule_schedule_test(
    encoder_configs, llm_config, hidden_size, seq_length, micro_batch_size, num_microbatches
):
    """Run multimodule schedule test with given configuration.

    Args:
        encoder_configs: List of encoder configs
        llm_config: LLM config dict
        hidden_size: Hidden dimension
        seq_length: Sequence length
        micro_batch_size: Micro batch size
        num_microbatches: Number of microbatches
    """
    # Create model
    model = MultiModuleModel(encoder_configs, llm_config, hidden_size)
    model.model_type = 'unit-test'

    # Build module_to_grid_map and topology
    module_to_grid_map = {name: grid for name, grid in model.encoder_grids.items()}
    module_to_grid_map['llm'] = model.llm_grid

    topology = {name: ['llm'] for name in model.encoders.keys()}
    topology['llm'] = []

    # Configure
    config = ModelParallelConfig(pipeline_dtype=torch.bfloat16)
    config.variable_seq_lengths = True
    config.calculate_per_token_loss = False
    config.fine_grained_activation_offloading = False
    config.qk_layernorm = False
    config.sequence_parallel = False
    config.moe_router_enable_expert_bias = False
    config.moe_router_load_balancing_type = "aux_loss"
    config.no_sync_func = model.no_sync
    config.finalize_model_grads_func = model.finalize_model_grads
    config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )
    config.hidden_size = hidden_size
    model.config = config

    # Create communicator
    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
    )

    # Create data iterator (only on first encoder's first stage)
    data_iterator = None
    first_encoder_name = encoder_configs[0]['name']
    first_encoder_grid = model.encoder_grids[first_encoder_name]
    if model.is_rank_in_grid(first_encoder_grid):
        if is_pp_first_stage(first_encoder_grid.get_pg("pp")):
            data_iterator = DataIterator(hidden_size, seq_length, micro_batch_size)

    # Get process group collection for current rank
    rank = dist.get_rank()
    pg_collection = None
    for name, grid in model.encoder_grids.items():
        if grid.rank_offset <= rank < grid.rank_offset + grid.size:
            pg_collection = add_embedding_groups(get_pg_collection(grid))
            break
    if (
        pg_collection is None
        and model.llm_grid.rank_offset <= rank < model.llm_grid.rank_offset + model.llm_grid.size
    ):
        pg_collection = add_embedding_groups(get_pg_collection(model.llm_grid))

    # Define step function
    def step_func(data_iterator, model):
        def loss_func(output_tensor_dict: Dict[str, torch.Tensor]):
            assert 'llm' in output_tensor_dict, f"Expected 'llm' in output"
            loss = output_tensor_dict['llm'].sum()
            return loss, {'loss_reduced': loss}

        input_tensor = next(data_iterator) if data_iterator is not None else None
        model_output = model(input_tensor)
        return model_output, loss_func

    # Run schedule
    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=step_func,
        data_iterator=data_iterator,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        p2p_communicator=communicator,
        pg_collection=pg_collection,
    )

    # Verify results on last LLM stage
    if model.is_rank_in_grid(model.llm_grid):
        if is_pp_last_stage(model.llm_grid.get_pg("pp")):
            assert len(losses) > 0, "Expected losses on last LLM stage"

    return losses


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMultimoduleSchedules:
    """Test multimodule pipeline schedules."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_single_encoder_2gpu(self):
        """Test single encoder + LLM on 2 GPUs (no PP)."""
        if self.world_size != 2:
            pytest.skip(f"Requires 2 GPUs, got {self.world_size}")

        encoder_configs = [{'name': 'encoder', 'tp': 1, 'pp': 1, 'dp': 1, 'grid_offset': 0}]
        llm_config = {'tp': 1, 'pp': 1, 'dp': 1, 'grid_offset': 1}

        run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=512,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_dual_encoder_2gpu(self):
        """Test dual encoder + LLM on 2 GPUs (both encoders on rank 0)."""
        if self.world_size != 2:
            pytest.skip(f"Requires 2 GPUs, got {self.world_size}")

        encoder_configs = [
            {'name': 'encoder_1', 'tp': 1, 'pp': 1, 'dp': 1, 'grid_offset': 0},
            {'name': 'encoder_2', 'tp': 1, 'pp': 1, 'dp': 1, 'grid_offset': 0},
        ]
        llm_config = {'tp': 1, 'pp': 1, 'dp': 1, 'grid_offset': 1}

        run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=512,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )

    def test_single_encoder_8gpu(self):
        """Test single encoder + LLM on 8 GPUs (TP=2, PP=2 each)."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_configs = [{'name': 'encoder', 'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 0}]
        llm_config = {'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 4}

        run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=1024,
            seq_length=512,
            micro_batch_size=4,
            num_microbatches=16,
        )

    def test_dual_encoder_tp1_dp2_8gpu(self):
        """Test dual encoder + LLM on 8 GPUs (TP=1, DP=2, PP=2 for each encoder)."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_configs = [
            {'name': 'encoder_1', 'tp': 1, 'pp': 2, 'dp': 2, 'grid_offset': 0},
            {'name': 'encoder_2', 'tp': 1, 'pp': 2, 'dp': 2, 'grid_offset': 0},
        ]
        llm_config = {'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 4}

        run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=1024,
            seq_length=512,
            micro_batch_size=4,
            num_microbatches=16,
        )

    def test_dual_encoder_8gpu(self):
        """Test dual encoder + LLM on 8 GPUs (TP=2, PP=2 for each)."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_configs = [
            {'name': 'encoder_1', 'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 0},
            {'name': 'encoder_2', 'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 0},
        ]
        llm_config = {'tp': 2, 'pp': 2, 'dp': 1, 'grid_offset': 4}

        run_multimodule_schedule_test(
            encoder_configs,
            llm_config,
            hidden_size=1024,
            seq_length=512,
            micro_batch_size=4,
            num_microbatches=16,
        )
