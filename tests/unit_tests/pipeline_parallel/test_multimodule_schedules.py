import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker
from megatron.core.process_groups_config import ModelCommProcessGroups
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import GradFinalizeProcessGroups
from tests.unit_tests.test_utilities import Utils
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from typing import Dict
from megatron.core.pipeline_parallel.multi_module_communicator import (
    MultiModulePipelineCommunicator,
)

rank = Utils.rank

class DataIterator:
    def __iter__(self):
        return self
    def __next__(self):
        return torch.randn(8, 512, 256, device='cuda')

class Model(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.encoder, self.encoder_grid = get_transformer_block_and_grid(
            tp_size=2,
            cp_size=1,
            pp_size=2,
            dp_size=1,
        )

        self.llm, self.llm_grid = get_transformer_block_and_grid(
            tp_size=2,
            cp_size=1,
            pp_size=2,
            dp_size=1,
            grid_offset=4,
        )

        self.encoder_input_tensor = None
        self.llm_input_tensor = None
    
    def set_input_tensor(self, input_tensor: Dict[str, torch.Tensor]):
        if 0<=dist.get_rank()<4:
            assert "encoder" in input_tensor, "Encoder input tensor is not provided"
            self.encoder_input_tensor = input_tensor["encoder"]
        elif 4<=dist.get_rank()<8:
            assert "llm" in input_tensor, "LLM input tensor is not provided"
            self.llm_input_tensor = input_tensor["llm"]
        else:
            raise ValueError(f"Rank {dist.get_rank()} is not valid")

    def forward(self, hidden_states):

        current_rank = dist.get_rank()
        output_dict = {}
        if 0<=current_rank<4:
            # if pp rank > 0 in encoder pp group then we use self.encoder_input_tensor as input else we use hidden_states
            if is_pp_first_stage(self.encoder_grid.get_pg("pp")):
                input_tensor = hidden_states
            else:
                assert self.encoder_input_tensor is not None, "Encoder input tensor is not provided for pp rank > 0"
                input_tensor = self.encoder_input_tensor
            output_dict["encoder"] = self.encoder(input_tensor)
        elif 4<=current_rank<8:
            # if pp rank > 0 in llm pp group then we use self.llm_input_tensor as input else we use hidden_states
            if is_pp_first_stage(self.llm_grid.get_pg("pp")):
                input_tensor = hidden_states
            else:
                assert self.llm_input_tensor is not None, "LLM input tensor is not provided for pp rank > 0"
                input_tensor = self.llm_input_tensor
            output_dict["llm"] = self.llm(input_tensor)
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
    grid = create_hypercomm_grid(
        offset=grid_offset, tp=tp_size, cp=cp_size, pp=pp_size, dp=dp_size
    )
    if grid.rank_offset <= current_rank < grid.rank_offset + grid.size:
        model_comm_pgs = _get_model_comm_pgs_from_grid(grid)
        block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, model_comm_pgs=model_comm_pgs
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
    embd_pg = embd_pg if (is_pp_last_stage(grad_finalize_pgs.pp) or is_pp_first_stage(grad_finalize_pgs.pp)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])
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
            return rank, {'loss_reduced': rank}
        
        if data_iterator is not None:
            input_tensor = next(data_iterator)
        else:
            input_tensor = None
        
        model_output = model(input_tensor)

        return model_output, loss_func

    # Create model
    model = Model()
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256


    module_to_grid_map = {'encoder': model.encoder_grid, 'llm': model.llm_grid}
    topology = {
        'encoder': ['llm'],  # image_encoder sends forward results to llm
        'llm': [],  # llm is the last stage here
    }
    config = ModelParallelConfig(pipeline_dtype=torch.float)
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    multi_module_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
    )
    # ykarnati: remove this
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    data_iterator = None
    if dist.get_rank() == 0:
        data_iterator = DataIterator()

    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': data_iterator,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    if 0<=dist.get_rank()<4:
        grad_finalize_pgs = _get_grad_finalize_pgs(model.encoder_grid)
    elif 4<=dist.get_rank()<8:        
        grad_finalize_pgs = _get_grad_finalize_pgs(model.llm_grid)
    else:
        raise ValueError(f"Rank {dist.get_rank()} is not valid")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multi_module_communicator, grad_finalize_pgs=grad_finalize_pgs, **common_args
    )

    Utils.destroy_model_parallel()