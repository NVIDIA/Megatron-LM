import logging
import os
from typing import Dict, List
from contextlib import contextmanager
import pytest
import torch
import torch.distributed as dist
from packaging import version


import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig

from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)

from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
    _get_pg_collection_from_grid,
    get_transformer_block_and_grid,
)
from tests.unit_tests.pipeline_parallel.test_schedules import _populate_embedding_and_position_groups

rank = Utils.rank
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIterator:

    def __init__(self, hidden_size: int, seq_length: int, micro_batch_size: int):
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



class SingleEncoderModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        encoder_tp,
        encoder_pp,
        encoder_dp,
        llm_tp,
        llm_pp,
        llm_dp,
        llm_grid_offset,
    ):

        super().__init__()

        self.encoder, self.encoder_grid = get_transformer_block_and_grid(
            tp_size=encoder_tp,
            cp_size=1,
            pp_size=encoder_pp,
            dp_size=encoder_dp,
            hidden_size=hidden_size,
            wrap_with_ddp=True,
        )

        self.llm, self.llm_grid = get_transformer_block_and_grid(
            tp_size=llm_tp,
            cp_size=1,
            pp_size=llm_pp,
            dp_size=llm_dp,
            grid_offset=llm_grid_offset,
            hidden_size=hidden_size,
            wrap_with_ddp=True,
        )

        # Simple list for iteration
        self.modules_and_grids = [
            (self.encoder, self.encoder_grid),
            (self.llm, self.llm_grid)
        ]

        self.current_rank = dist.get_rank()
        self.encoder_input_tensor = None
        self.llm_input_tensor = None


    def finish_grad_sync(self):
        """Finish gradient synchronization for all active modules on this rank."""
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_current_rank_in_grid(grid):
                module.finish_grad_sync()
    @contextmanager
    def no_sync(self):
        contexts = []
        if self.is_current_rank_in_grid(self.encoder_grid):
            contexts.append(self.encoder.no_sync())
        if self.is_current_rank_in_grid(self.llm_grid):
            contexts.append(self.llm.no_sync())
        
        # Enter all contexts
        for ctx in contexts:
            ctx.__enter__()
        
        try:
            yield
        finally:
            # Exit all contexts in reverse order
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    @property
    def ddp_config(self):
        # Try to get ddp_config from the first available module on this rank
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_current_rank_in_grid(grid):
                return module.ddp_config
        raise AttributeError(
            f"No active modules with ddp_config found on rank {self.current_rank}"
        )

    def scale_gradients(self, scaling_factor: float):
        """Scale gradients for all active modules on this rank."""
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_current_rank_in_grid(grid):
                module.scale_gradients(scaling_factor)

    def is_current_rank_in_grid(self, grid: HyperCommGrid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= self.current_rank < (grid.rank_offset + grid.size)
    
    def finalize_model_grads(self, module=None, num_tokens=None, pg_collection=None):
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_current_rank_in_grid(grid):
                finalize_model_grads([module], num_tokens=None, pg_collection=_get_pg_collection_with_embedding_groups(grid))

    @contextmanager
    def no_sync(self):
        contexts = []
        for module, grid in self.modules_and_grids:
            if module is not None and self.is_current_rank_in_grid(grid):
                contexts.append(module.no_sync())
        
        # Enter all contexts
        for ctx in contexts:
            ctx.__enter__()
        
        try:
            yield
        finally:
            # Exit all contexts in reverse order
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    def set_input_tensor(self, input_tensor: List[Dict[str, torch.Tensor]]):
        if self.is_current_rank_in_grid(self.encoder_grid) and 'encoder' in input_tensor[0]:
            if isinstance(input_tensor[0]["encoder"], list):
                encoder_input_tensor = input_tensor[0]["encoder"][0]
            else:
                encoder_input_tensor = input_tensor[0]["encoder"]
            logging.debug(
                f"[Rank {dist.get_rank()} ][SingleEncoderModel] [set_input_tensor] [encoder] input tensor shape: {input_tensor[0]['encoder'][0].shape}"
            )
            self.encoder_input_tensor = encoder_input_tensor
        elif self.is_current_rank_in_grid(self.llm_grid):
            if 'llm' in input_tensor[0]:
                if isinstance(input_tensor[0]["llm"], list):
                    llm_input_tensor = input_tensor[0]["llm"][0]
                else:
                    llm_input_tensor = input_tensor[0]["llm"]
                logging.debug(
                    f"[Rank {dist.get_rank()} ][SingleEncoderModel] [set_input_tensor] [llm] input tensor shape: {llm_input_tensor.shape}"
                )
                self.llm_input_tensor = llm_input_tensor
            elif 'encoder' in input_tensor[0]:
                logging.debug(
                    f"[Rank {dist.get_rank()} ][SingleEncoderModel] [set_input_tensor] [encoder] input tensor shape: {input_tensor[0]['encoder'].shape}"
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
            logging.debug(
                f"[Rank {dist.get_rank()} ][SingleEncoderModel] [forward] [encoder] input tensor shape: {input_tensor.shape}"
            )
            output_dict["encoder"] = self.encoder(input_tensor, attention_mask=None)
        elif self.is_current_rank_in_grid(self.llm_grid):
            assert (
                self.llm_input_tensor is not None
            ), "LLM input tensor is not provided for pp rank > 0"
            input_tensor = self.llm_input_tensor
            logging.debug(
                f"[Rank {dist.get_rank()} ][SingleEncoderModel] [forward] [llm] input tensor shape: {input_tensor.shape}"
            )
            output_dict["llm"] = self.llm(input_tensor, attention_mask=None)
        else:
            raise ValueError(f"Rank {current_rank} is not valid")

        return output_dict



def _get_pg_collection_with_embedding_groups(grid):
    pg_collection = _get_pg_collection_from_grid(grid)
    if pg_collection.pp:
        pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pg_collection.pp)
        pos_embd_pg = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
        embd_pg = (
            embd_pg
            if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
            else None
        )
        pg_collection.pos_embd = pos_embd_pg
        pg_collection.embd = embd_pg

    return pg_collection


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.parametrize(
    "encoder_tp,encoder_pp,encoder_dp,llm_tp,llm_pp,llm_dp,llm_grid_offset",
    [
        # (2, 2, 1, 2, 2, 1, 4),
        # (4, 1, 1, 2, 2, 1, 4),
        (2, 1, 1, 1, 6, 1, 2),
    ],
)
def test_forward_backward_pipelining_without_interleaving_multi_module_single_encoder(
 encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset
):

    Utils.initialize_distributed()

    def step_func(data_iterator, model):

        def loss_func(output_tensor_dict: Dict[str, torch.Tensor]):
            assert (
                'llm' in output_tensor_dict
            ), f'llm is not in output_tensor_dict: {output_tensor_dict}'
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

    # Create model
    model = SingleEncoderModel(
        hidden_size=hidden_size,
        encoder_tp=encoder_tp,
        encoder_pp=encoder_pp,
        encoder_dp=encoder_dp,
        llm_tp=llm_tp,
        llm_pp=llm_pp,
        llm_dp=llm_dp,
        llm_grid_offset=llm_grid_offset,
    )
    model.model_type = 'unit-test'

    module_to_grid_map = {'encoder': model.encoder_grid, 'llm': model.llm_grid}
    topology = {
        'encoder': ['llm'],  # image_encoder sends forward results to llm
        'llm': [],  # llm is the last stage here
    }
    config = ModelParallelConfig(pipeline_dtype=torch.bfloat16)
    config.calculate_per_token_loss = False
    config.qk_layernorm = False
    config.sequence_parallel = False
    config.moe_router_enable_expert_bias = False
    config.moe_router_load_balancing_type = "aux_loss"
    config.variable_seq_lengths = True
    config.no_sync_func = model.no_sync
    config.finalize_model_grads_func = model.finalize_model_grads
    config.fine_grained_activation_offloading = False
    

    # Add grad scale function to convert float losses to tensors
    def grad_scale_func(loss):
        if isinstance(loss, (int, float)):
            return torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        else:
            return loss  # Already a tensor

    config.grad_scale_func = grad_scale_func
    model.config = config
    config.hidden_size = hidden_size

    multimodule_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, config, dim_mapping={'s': 0, 'h': 2, 'b': 1}
    )

    data_iterator = None
    if model.is_current_rank_in_grid(model.encoder_grid) and is_pp_first_stage(
        model.encoder_grid.get_pg("pp")
    ):
        data_iterator = DataIterator(
            hidden_size=hidden_size, seq_length=sequence_length, micro_batch_size=micro_batch_size
        )

    common_args = {
        'forward_step_func': step_func,
        'data_iterator': data_iterator,
        'model': [model],
        'num_microbatches': 16,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': False,
    }

    if 0 <= dist.get_rank() < llm_grid_offset:
        pg_collection = _get_pg_collection_with_embedding_groups(model.encoder_grid)
    elif llm_grid_offset <= dist.get_rank() < llm_grid_offset + model.llm_grid.size:
        pg_collection = _get_pg_collection_with_embedding_groups(model.llm_grid)
    else:
        raise ValueError(f"Rank {dist.get_rank()} is not valid")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multimodule_communicator, pg_collection=pg_collection, **common_args
    )
    logging.info(f"Losses reduced explicit: {losses_reduced_explicit}")



if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Use the same parameters as defined in the pytest.mark.parametrize decorator
    test_forward_backward_pipelining_without_interleaving_multi_module_single_encoder(
        encoder_tp=2, 
        encoder_pp=1, 
        encoder_dp=1, 
        llm_tp=1, 
        llm_pp=6, 
        llm_dp=1, 
        llm_grid_offset=2
    )