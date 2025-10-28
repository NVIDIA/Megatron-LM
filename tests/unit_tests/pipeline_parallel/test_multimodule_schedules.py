import logging
import os
from typing import Dict, List
from contextlib import contextmanager
import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

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
        )

        self.llm, self.llm_grid = get_transformer_block_and_grid(
            tp_size=llm_tp,
            cp_size=1,
            pp_size=llm_pp,
            dp_size=llm_dp,
            grid_offset=llm_grid_offset,
            hidden_size=hidden_size,
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

class DualEncoderModel(SingleEncoderModel):
    def __init__(self, hidden_size, encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset):
        super().__init__(hidden_size, encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset)

        self.encoder_1, self.encoder_1_grid = get_transformer_block_and_grid(
            tp_size=encoder_tp,
            cp_size=1,
            pp_size=encoder_pp,
            dp_size=encoder_dp,
            hidden_size=hidden_size,
        )

        self.encoder_2, self.encoder_2_grid = get_transformer_block_and_grid(
            tp_size=encoder_tp,
            cp_size=1,
            pp_size=encoder_pp,
            dp_size=encoder_dp,
            hidden_size=hidden_size,
        )

        self.llm, self.llm_grid = get_transformer_block_and_grid(
            tp_size=llm_tp,
            cp_size=1,
            pp_size=llm_pp,
            dp_size=llm_dp,
            grid_offset=llm_grid_offset,
            hidden_size=hidden_size,
        )

        self.modules_and_grids = [
            (self.encoder_1, self.encoder_1_grid),
            (self.encoder_2, self.encoder_2_grid),
            (self.llm, self.llm_grid)
        ]

        self.current_rank = dist.get_rank()
        self.encoder_1_input_tensor = None
        self.encoder_2_input_tensor = None
        self.llm_input_tensor = None

        self.pre_process = False
        self.post_process = False
        self.share_embeddings_and_output_weights = False


    def set_input_tensor(self, input_tensor: List[Dict[str, torch.Tensor]]):
        logging.debug(f" In DualEncoderModel set_input_tensor rank {dist.get_rank()} input_tensor keys: {input_tensor[0].keys()}")
        if self.is_current_rank_in_grid(self.encoder_1_grid) and 'encoder_1' in input_tensor[0]:
            if isinstance(input_tensor[0]["encoder_1"], list):
                self.encoder_1_input_tensor = input_tensor[0]["encoder_1"][0]
            else:
                self.encoder_1_input_tensor = input_tensor[0]["encoder_1"]
        if self.is_current_rank_in_grid(self.encoder_2_grid) and 'encoder_2' in input_tensor[0]:
            if isinstance(input_tensor[0]["encoder_2"], list):
                self.encoder_2_input_tensor = input_tensor[0]["encoder_2"][0]
            else:
                self.encoder_2_input_tensor = input_tensor[0]["encoder_2"]
        if self.is_current_rank_in_grid(self.llm_grid):
            if 'llm' in input_tensor[0]:
                if isinstance(input_tensor[0]["llm"], list):
                    self.llm_input_tensor = input_tensor[0]["llm"][0]
                else:
                    self.llm_input_tensor = input_tensor[0]["llm"]
            elif 'encoder_1' in input_tensor[0] and 'encoder_2' in input_tensor[0]:
                # concat across sequence dimension (s, b, h)
                logging.debug(f'In DualEncoderModel LLM set_input_tensor rank {dist.get_rank()} encoder_1 shape: {input_tensor[0]["encoder_1"].shape} encoder_2 shape: {input_tensor[0]["encoder_2"].shape}')
                self.llm_input_tensor = torch.concat([input_tensor[0]["encoder_1"], input_tensor[0]["encoder_2"]], dim=0)
                logging.debug(f" In DualEncoderModel LLM set_input_tensor rank {dist.get_rank()} llm_input_tensor shape: {self.llm_input_tensor.shape}")
            else:
                raise ValueError(f"Rank {dist.get_rank()} is not valid")
    
    def forward(self, hidden_states):
        current_rank = dist.get_rank()
        output_dict = {}
        logging.debug(f" In DualEncoderModel forward rank {dist.get_rank()}")
        if self.is_current_rank_in_grid(self.encoder_1_grid):
            if is_pp_first_stage(self.encoder_1_grid.get_pg("pp")):
                input_tensor = hidden_states
            else:
                assert (
                    self.encoder_1_input_tensor is not None
                ), "Encoder input tensor is not provided for pp rank > 0"
                input_tensor = self.encoder_1_input_tensor
            output_dict["encoder_1"] = self.encoder_1(input_tensor, attention_mask=None)
        if self.is_current_rank_in_grid(self.encoder_2_grid):
            if is_pp_first_stage(self.encoder_2_grid.get_pg("pp")):
                input_tensor = hidden_states
            else:
                assert (
                    self.encoder_2_input_tensor is not None
                ), "Encoder input tensor is not provided for pp rank > 0"
                input_tensor = self.encoder_2_input_tensor
            output_dict["encoder_2"] = self.encoder_2(input_tensor, attention_mask=None)
        if self.is_current_rank_in_grid(self.llm_grid):
            assert (
                self.llm_input_tensor is not None
            ), "LLM input tensor is not provided for pp rank > 0"
            input_tensor = self.llm_input_tensor
            output_dict["llm"] = self.llm(input_tensor, attention_mask=None)
        logging.debug(f"[DualEncoderModel] model fwd pass in rank {dist.get_rank()} output_dict keys: {output_dict.keys()}")
        return output_dict

def _create_transformer_block(
    dtype=torch.bfloat16, hidden_size=4096, pg_collection=None
) -> TransformerBlock:
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(
        123,
        tp_rank=pg_collection.tp.rank(),
        ep_rank=pg_collection.ep.rank(),
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


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1, ep=1, etp=1):
    """Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2."""
    # Set up environment for world size 8 if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"

    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, ep, etp], # 需要加上etp吗
        dim_names=["tp", "cp", "pp", "dp", "ep", "etp"],
        rank_offset=offset,
        backend="nccl",
    )
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    _ = grid.create_pg(["ep"])
    # _ = grid.create_pg(["etp"])
    # _ = grid.create_pg(["edp"])
    _ = grid.create_pg(["tp", "pp"])
    _ = grid.create_pg(["dp", "cp"])
    _ = grid.create_pg(["tp", "cp"])
    _ = grid.create_pg(["tp", "dp", "cp"])
    _ = grid.create_pg(["tp", "ep", "pp"])
    return grid


def _get_pg_collection_from_grid(grid):
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    dp_group = grid.get_pg("dp")
    dp_cp_group = grid.get_pg(["dp", "cp"])
    pg_collection.dp = dp_group
    pg_collection.dp_cp = dp_cp_group
    pg_collection.mp = grid.get_pg(["tp", "pp"])
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    pg_collection.tp_cp = grid.get_pg(["tp", "cp"])
    pg_collection.tp_dp_cp = grid.get_pg(["tp", "dp", "cp"])
    pg_collection.tp_ep_pp = grid.get_pg(["tp", "ep", "pp"])
    pg_collection.expt_tp = None
    pg_collection.expt_dp = None
    return pg_collection


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
        pg_collection = _get_pg_collection_from_grid(grid)
        block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=pg_collection
        )
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
        block = DistributedDataParallel(
            config=block.config, ddp_config=ddp_config, module=block, pg_collection=pg_collection
        )
        block.pre_process = False
        block.post_process = False
        block.share_embeddings_and_output_weights = False


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
    "encoder_tp,encoder_pp,encoder_dp,llm_tp,llm_pp,llm_dp,llm_grid_offset", [(2, 2, 1, 2, 2, 1, 4)]
)
def test_forward_backward_pipelining_without_interleaving_multi_module_single_encoder(
    mocker, encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset
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

    if 0 <= dist.get_rank() < 4:
        pg_collection = _get_pg_collection_with_embedding_groups(model.encoder_grid)
    elif 4 <= dist.get_rank() < 8:
        pg_collection = _get_pg_collection_with_embedding_groups(model.llm_grid)
    else:
        raise ValueError(f"Rank {dist.get_rank()} is not valid")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multimodule_communicator, pg_collection=pg_collection, **common_args
    )
    logging.info(f"Losses reduced explicit: {losses_reduced_explicit}")


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.parametrize(
    "encoder_tp,encoder_pp,encoder_dp,llm_tp,llm_pp,llm_dp,llm_grid_offset", [(2, 2, 1, 2, 2, 1, 4)]
)
def test_forward_backward_pipelining_without_interleaving_multi_module_dual_encoder(
    mocker, encoder_tp, encoder_pp, encoder_dp, llm_tp, llm_pp, llm_dp, llm_grid_offset
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
    model = DualEncoderModel(
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

    module_to_grid_map = {'encoder_1': model.encoder_1_grid, 'encoder_2': model.encoder_2_grid, 'llm': model.llm_grid}
    topology = {
        'encoder_1': ['llm'],  # encoder_1 sends forward results to llm
        'encoder_2': ['llm'],  # encoder_2 sends forward results to llm
        'llm': [],  # llm is the last stage here
    }
    config = ModelParallelConfig(pipeline_dtype=torch.bfloat16)
    config.finalize_model_grads_func = model.finalize_model_grads
    config.calculate_per_token_loss = False
    config.qk_layernorm = False
    config.sequence_parallel = False
    config.moe_router_enable_expert_bias = False
    config.moe_router_load_balancing_type = "aux_loss"
    config.variable_seq_lengths = True
    config.no_sync_func = model.no_sync

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
    if model.is_current_rank_in_grid(model.encoder_1_grid) and is_pp_first_stage(
        model.encoder_1_grid.get_pg("pp")
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

    if 0 <= dist.get_rank() < 4:
        pg_collection_encoder_1 = _get_pg_collection_with_embedding_groups(model.encoder_1_grid)
        pg_collection_encoder_2 = _get_pg_collection_with_embedding_groups(model.encoder_2_grid)
        pg_collection = [pg_collection_encoder_1, pg_collection_encoder_2]
    elif 4 <= dist.get_rank() < 8:
        pg_collection_llm = _get_pg_collection_with_embedding_groups(model.llm_grid)
        pg_collection = [pg_collection_llm]
    else:
        raise ValueError(f"Rank {dist.get_rank()} is not valid")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multimodule_communicator, pg_collection=pg_collection, **common_args
    )
    logging.info(f"Losses reduced explicit: {losses_reduced_explicit}")


if __name__ == "__main__":
    from unittest.mock import Mock
    
    # Set logging level to DEBUG
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a mock object that mimics pytest-mock's mocker
    mock_mocker = Mock()

    # Use the same parameters as defined in the pytest.mark.parametrize decorator
    test_forward_backward_pipelining_without_interleaving_multi_module_single_encoder(
        mock_mocker, 
        encoder_tp=2, 
        encoder_pp=2, 
        encoder_dp=1, 
        llm_tp=2, 
        llm_pp=2, 
        llm_dp=1, 
        llm_grid_offset=4
    )
