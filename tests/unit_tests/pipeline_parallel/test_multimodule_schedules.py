import logging
import os
from typing import Dict, List

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel.multi_module_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import GradFinalizeProcessGroups, ModelCommProcessGroups, GradCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
rank = Utils.rank


class DataIterator:

    def __init__(self, hidden_size: int, seq_length: int, micro_batch_size: int):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return torch.randn(self.seq_length, self.micro_batch_size, self.hidden_size, device='cuda', dtype = torch.bfloat16)


class Model(torch.nn.Module):
    def __init__(self, hidden_size, encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset):

        super().__init__()

        self.encoder, self.encoder_grid = get_transformer_block_and_grid(
            tp_size=encoder_tp, cp_size=1, pp_size=encoder_pp, dp_size=encoder_dp, hidden_size=hidden_size
        )

        self.llm, self.llm_grid = get_transformer_block_and_grid(
            tp_size=llm_tp, cp_size=1, pp_size=llm_pp, dp_size=llm_dp, grid_offset=llm_grid_offset, hidden_size=hidden_size
        )

        self.current_rank = dist.get_rank()
        self.encoder_input_tensor = None
        self.llm_input_tensor = None

        self.pre_process = False
        self.post_process = False
        self.share_embeddings_and_output_weights = False

    def finish_grad_sync(self):
        """Finish gradient synchronization for all active modules on this rank."""
        if self.is_current_rank_in_grid(self.encoder_grid) and self.encoder is not None:
            self.encoder.finish_grad_sync()
        if self.is_current_rank_in_grid(self.llm_grid) and self.llm is not None:
            self.llm.finish_grad_sync()
    
    @property
    def ddp_config(self):
        # Try to get ddp_config from the first available module on this rank
        if self.is_current_rank_in_grid(self.encoder_grid) and self.encoder is not None:
            return self.encoder.ddp_config
        elif self.is_current_rank_in_grid(self.llm_grid) and self.llm is not None:
            return self.llm.ddp_config
        else:
            raise AttributeError(f"No active modules with ddp_config found on rank {self.current_rank}")

    def scale_gradients(self, scaling_factor: float):
        """Scale gradients for all active modules on this rank."""
        if self.is_current_rank_in_grid(self.encoder_grid) and self.encoder is not None:
            self.encoder.scale_gradients(scaling_factor)
        if self.is_current_rank_in_grid(self.llm_grid) and self.llm is not None:
            self.llm.scale_gradients(scaling_factor)
    
    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)
    
    def set_input_tensor(self, input_tensor: List[Dict[str, torch.Tensor]]):
        if self.is_current_rank_in_grid(self.encoder_grid) and 'encoder' in input_tensor[0]:
            logging.info(f"Current rank: {dist.get_rank()} setting encoder input tensor with shape {input_tensor[0]['encoder'][0].shape} dtype {input_tensor[0]['encoder'][0].dtype}")
            self.encoder_input_tensor = input_tensor[0]["encoder"][0]
        elif self.is_current_rank_in_grid(self.llm_grid):
            if 'llm' in input_tensor[0]:
                logging.info(
                    f"Current rank: {dist.get_rank()} setting llm input tensor with shape {input_tensor[0]['llm'][0].shape} dtype {input_tensor[0]['llm'][0].dtype}"
                )
                self.llm_input_tensor = input_tensor[0]["llm"][0]
            elif 'encoder' in input_tensor[0]:
                logging.info(
                    f"Current rank: {dist.get_rank()} setting encoder input tensor with shape {input_tensor[0]['encoder'].shape} dtype {input_tensor[0]['encoder'][0].dtype}"
                )
                self.llm_input_tensor = input_tensor[0]["encoder"]
            else:
                raise ValueError(f"Rank {dist.get_rank()} is not valid")

    def forward(self, hidden_states):

        current_rank = dist.get_rank()
        output_dict = {}
        if self.is_current_rank_in_grid(self.encoder_grid):
            # if pp rank > 0 in encoder pp group then we use self.encoder_input_tensor as input else we use hidden_states
            if is_pp_first_stage(self.encoder_grid.get_pg("pp")):
                input_tensor = hidden_states
            else:
                assert (
                    self.encoder_input_tensor is not None
                ), "Encoder input tensor is not provided for pp rank > 0"
                input_tensor = self.encoder_input_tensor
            logging.info(f"Current rank: {dist.get_rank()} encoder forward with input_tensor shape {input_tensor.shape} dtype {input_tensor.dtype}")
            output_dict["encoder"] = self.encoder(input_tensor, attention_mask=None)
        elif self.is_current_rank_in_grid(self.llm_grid):
            assert (
                self.llm_input_tensor is not None
            ), "LLM input tensor is not provided for pp rank > 0"
            input_tensor = self.llm_input_tensor
            logging.info(f"Current rank: {dist.get_rank()} llm forward with input_tensor shape {input_tensor.shape} dtype {input_tensor.dtype}")
            output_dict["llm"] = self.llm(input_tensor, attention_mask=None)
        else:
            raise ValueError(f"Rank {current_rank} is not valid")

        return output_dict


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


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2."""
    # Set up environment for world size 8 if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"

    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep"],
        rank_offset=offset,
        backend="nccl",
    )
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    _ = grid.create_pg(["dp", "cp"])
    _ = grid.create_pg(["ep"])
    return grid


def _get_model_comm_pgs_from_grid(grid):
    model_comm_pgs = ModelCommProcessGroups()
    model_comm_pgs.tp = grid.get_pg("tp")
    model_comm_pgs.cp = grid.get_pg("cp")
    model_comm_pgs.pp = grid.get_pg("pp")
    model_comm_pgs.ep = grid.get_pg("ep")
    return model_comm_pgs

def _get_grad_comm_pgs_from_grid(grid):
    grad_comm_pgs = GradCommProcessGroups()
    dp_group = grid.get_pg("dp")
    dp_cp_group = grid.get_pg(["dp", "cp"])
    grad_comm_pgs.dp = dp_group
    grad_comm_pgs.dp_cp = dp_cp_group
    return grad_comm_pgs


def get_transformer_block_and_grid(
    tp_size=1,
    cp_size=1,
    pp_size=1,
    dp_size=1,
    grid_offset: int = 0,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
):
    """Utility to build a ``TransformerBlock`` for tests."""

    current_rank = dist.get_rank()
    grid = create_hypercomm_grid(offset=grid_offset, tp=tp_size, cp=cp_size, pp=pp_size, dp=dp_size)
    if grid.rank_offset <= current_rank < grid.rank_offset + grid.size:
        model_comm_pgs = _get_model_comm_pgs_from_grid(grid)
        grad_comm_pgs = _get_grad_comm_pgs_from_grid(grid)
        block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=model_comm_pgs
        )
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
        block = DistributedDataParallel(
            config=block.config,
            ddp_config=ddp_config,
            module=block,
            grad_comm_pgs=grad_comm_pgs,
            model_comm_pgs=model_comm_pgs,
        )
    else:
        block = None

    return block, grid


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def _get_grad_finalize_pgs(grid):
    grad_finalize_pgs = GradFinalizeProcessGroups()
    grad_finalize_pgs.tp = grid.get_pg("tp")
    grad_finalize_pgs.pp = grid.get_pg("pp")
    grad_finalize_pgs.cp = grid.get_pg("cp")

    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(grad_finalize_pgs.pp)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(grad_finalize_pgs.pp) else None
    embd_pg = (
        embd_pg
        if (is_pp_last_stage(grad_finalize_pgs.pp) or is_pp_first_stage(grad_finalize_pgs.pp))
        else None
    )
    dp_cp_group = grid.get_pg(["dp", "cp"])
    grad_finalize_pgs.pos_embd = pos_embd_pg
    grad_finalize_pgs.dp_cp = dp_cp_group
    grad_finalize_pgs.embd = embd_pg

    return grad_finalize_pgs


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_multi_module(mocker):
    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor_dict: Dict[str, torch.Tensor]):
            assert 'llm' in output_tensor_dict, f'llm is not in output_tensor_dict: {output_tensor_dict}'
            loss = output_tensor_dict['llm'].sum()
            return loss, {'loss_reduced': loss}

        if data_iterator is not None:
            input_tensor = next(data_iterator)
        else:
            input_tensor = None

        model_output = model(input_tensor)

        return model_output, loss_func

    sequence_length = 512
    micro_batch_size = 1
    hidden_size = 1024

    encoder_tp, encoder_pp, encoder_dp = 2, 2, 1    
    llm_tp, llm_pp, llm_dp = 2, 2, 1
    llm_grid_offset = 4 

    # Create model
    model = Model(hidden_size=hidden_size, encoder_tp=encoder_tp, encoder_pp=encoder_pp, encoder_dp=encoder_dp, llm_tp=llm_tp, llm_pp=llm_pp, llm_dp=llm_dp, llm_grid_offset=llm_grid_offset)
    model.model_type = 'unit-test'   

    module_to_grid_map = {'encoder': model.encoder_grid, 'llm': model.llm_grid}
    topology = {
        'encoder': ['llm'],  # image_encoder sends forward results to llm
        'llm': [],  # llm is the last stage here
    }
    config = ModelParallelConfig(pipeline_dtype=torch.bfloat16)
    config.finalize_model_grads_func = finalize_model_grads
    config.calculate_per_token_loss = False
    config.qk_layernorm = False
    config.sequence_parallel = False
    config.moe_router_enable_expert_bias = False
    
    # Add grad scale function to convert float losses to tensors
    def grad_scale_func(loss):
        """Convert float loss to tensor by multiplying with unit tensor."""
        if isinstance(loss, (int, float)):
            return torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        else:
            return loss  # Already a tensor
    
    config.grad_scale_func = grad_scale_func
    model.config = config
    config.hidden_size = hidden_size

    multi_module_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
    )
    # ykarnati: remove this
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    data_iterator = None
    if model.is_current_rank_in_grid(model.encoder_grid) and is_pp_first_stage(model.encoder_grid.get_pg("pp")):
        data_iterator = DataIterator(hidden_size=hidden_size, seq_length=sequence_length, micro_batch_size=micro_batch_size)

    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': data_iterator,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': False,
    }

    if 0 <= dist.get_rank() < 4:
        grad_finalize_pgs = _get_grad_finalize_pgs(model.encoder_grid)
    elif 4 <= dist.get_rank() < 8:
        grad_finalize_pgs = _get_grad_finalize_pgs(model.llm_grid)
    else:
        raise ValueError(f"Rank {dist.get_rank()} is not valid")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multi_module_communicator,
        grad_finalize_pgs=grad_finalize_pgs,
        **common_args,
    )
    logging.info(f"Losses reduced explicit: {losses_reduced_explicit}")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    from unittest.mock import Mock

    # Create a mock object that mimics pytest-mock's mocker
    mock_mocker = Mock()

    test_forward_backward_pipelining_without_interleaving_multi_module(mock_mocker)
