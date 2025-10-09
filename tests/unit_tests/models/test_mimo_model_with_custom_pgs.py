import torch.distributed as dist
from functools import partial
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel

from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups
import logging
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from examples.mimo.configs.llava_vlm import get_llava_projection_layer_spec, get_llava_projection_config
from examples.mimo.data.mock import MockVLMDataset
import megatron.core.pipeline_parallel.schedules as schedule
import torch
from typing import Any, Dict, Iterator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from contextlib import contextmanager
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('megatron.core.models.mimo.model.base').setLevel(logging.DEBUG)

from torch.utils.data import DataLoader

from typing import List

def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        batch: List of dictionaries from the dataset

    Returns:
        Dictionary of batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])

    bsz = input_ids.shape[0]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            "images": {
                "clip_encoder": {'hidden_states': torch.randn(1024, bsz, 1024, dtype=torch.bfloat16), 'attention_mask': None},
            }
        },
    }

def is_current_rank_in_grid(grid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= dist.get_rank() < (grid.rank_offset + grid.size)
    
def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor

    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {'lm loss': (reporting_loss)})


def get_data_iterator(encoder_grid, llm_grid):
    data_iterator = None

    # we initialize iterator on first pp stage of encoders and LLM

    encoder_1_condition =   is_current_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    

    llm_condition = is_current_rank_in_grid(llm_grid) and (is_pp_first_stage(
        llm_grid.get_pg("pp")
    ) or is_pp_last_stage(llm_grid.get_pg("pp")))

    if encoder_1_condition or llm_condition:
        dataset = MockVLMDataset(
            size=256,
            image_size=224,
            seq_len=4096,
            image_seq_length=1024,
            pad_token_id=0,
            image_token_id=32000
        )
        dataloader =  DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: _collate_fn(batch),
        )
        data_iterator = iter(dataloader)
    return data_iterator

def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    if data_iterator is not None:
        input_tensor = next(data_iterator)
        if input_tensor is not None:
            input_tensor = move_to_device(input_tensor, torch.device("cuda"))
    else:
        input_tensor = None

    return input_tensor

def move_to_device(data, device):
    """Recursively move tensors in nested dicts to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data

def get_module_to_grid_tuple(mimo_model, vision_module_grid, language_module_grid):
    return_tuple = [(mimo_model.modality_submodules['images'], vision_module_grid), (mimo_model.language_model, language_module_grid)]
    return return_tuple


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.

    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance

    Returns:
        tuple: (output_tensor, loss_function)
    """
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        data_batch = {'input_ids': None}
    output_tensor, loss_mask = model(**data_batch)
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)

@contextmanager
def multimodule_no_sync(module_to_grid_tuple):
    contexts = []
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
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

def finalize_model_grads(module_to_grid_tuple, module=None, num_tokens=None, pg_collection=None):
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            finalize_model_grads([module], num_tokens=None, pg_collection=_get_pg_collection_with_embedding_groups(grid))

def get_language_model_spec(hidden_size, vocab_size, seq_len, pg_collection):
    """Get the language model spec."""
    # Determine pre_process and post_process based on PP rank
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)
    
    print(f"[get_language_model_spec] Rank {dist.get_rank()}: PP rank={pp_rank}/{pp_size}, "
          f"pre_process={pre_process}, post_process={post_process}")

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1
    
    lm_config = TransformerConfig(
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='native',
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": pre_process,
            "post_process": post_process,
            "pg_collection": pg_collection,
        },
    )
    return language_model_spec


def get_vision_submodules_spec(hidden_size, pg_collection):
    """Get the submodule spec for the vision modality."""
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1

    vision_config = TransformerConfig(
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
    )
    vision_encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": vision_layer_spec,
            "pg_collection": pg_collection,
            "pre_process": True,
            "post_process": True
        },
    )

    # Create vision projection spec
    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": get_llava_projection_config(
                hidden_size=vision_config.hidden_size
            ),
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 1024,
            "tp_group": pg_collection.tp,
        },
    )

    # Create vision modality spec
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )

    return vision_submodule_spec


def get_vlm_mimo_model(
    hidden_size, vocab_size, seq_len, special_token_ids
):
    language_module_grid = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=2, dp=1)
    language_pg_collection = _get_pg_collection_with_embedding_groups(language_module_grid)

    vision_module_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=2, dp=1)
    vision_pg_collection = _get_pg_collection_with_embedding_groups(vision_module_grid)

    language_model_spec = get_language_model_spec(hidden_size, vocab_size, seq_len, language_pg_collection)
    vision_submodule_spec = get_vision_submodules_spec(hidden_size, vision_pg_collection)

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec,},
        special_token_ids=special_token_ids,
    )
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    module_to_grid_map = {'images': vision_module_grid, 'language_module': language_module_grid}
    topology = {
        'images': ['language_module'],  # images sends forward results to language_module
        'language_module': [],  # language_module is the last stage here
    }


    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
        config=mimo_model.language_model.config,
        ddp_config=ddp_config,
        module=mimo_model.language_model,
        pg_collection=language_pg_collection
        )
    submodule = mimo_model.modality_submodules['images']

    if submodule is not None:
        submodule = DistributedDataParallel(
            config=submodule.encoders['clip_encoder'].config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=vision_pg_collection
        )
    mimo_model.modality_submodules['images'] = submodule

    return mimo_model, module_to_grid_map, topology


def get_pg_collections_for_rank(module_to_grid_map):
    """Get pg_collections for modules that should be initialized on the current rank."""
    pg_collections = []
    for _ , grid_name in module_to_grid_map.items():
        if is_current_rank_in_grid(grid_name):
            pg_collections.append(_get_pg_collection_with_embedding_groups(grid_name))
    return pg_collections


def test_init_avlm_mimo_model_custom_pgs():
    Utils.initialize_distributed()
    hidden_size = 64
    batch_size = 2
    seq_len = 2048
    vocab_size = 48000
    special_token_ids = {"images": 32000}
    mimo_model, _, _ = get_vlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, special_token_ids=special_token_ids)
    assert mimo_model is not None
    dist.destroy_process_group()


def test_1f_1b_schedule_avlm_mimo_model_custom_pgs():

    Utils.initialize_distributed()
    hidden_size = 1024
    batch_size = 2
    seq_len = 4096
    vocab_size = 48000
    special_token_ids = {"images": 32000}

    mimo_model, module_to_grid_map, topology = get_vlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, special_token_ids=special_token_ids)

    module_to_grid_tuple = get_module_to_grid_tuple(mimo_model, module_to_grid_map['images'], module_to_grid_map['language_module'])
    mimo_model.config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)
    mimo_model.config.finalize_model_grads_func = partial(finalize_model_grads, module_to_grid_tuple=module_to_grid_tuple)

    multimodule_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config, dim_mapping={'b': 0,'s': 1, 'h': 2}
    )

    mimo_model.model_type = 'unit-test'
    common_args = {
        'forward_step_func': forward_step,
        'data_iterator':  get_data_iterator(encoder_grid=module_to_grid_map['images'], llm_grid=module_to_grid_map['language_module']),
        'model': [mimo_model],
        'num_microbatches': 16,
        'seq_length': seq_len,
        'micro_batch_size': batch_size,
        'forward_only': True,
    }

    # Get pg collections for modules that should be initialized on this rank
    pg_collection = get_pg_collections_for_rank(module_to_grid_map)

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=multimodule_communicator, pg_collection=pg_collection, **common_args
    )
    logging.info(f"Losses reduced explicit: {losses_reduced_explicit}")

     

if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    test_1f_1b_schedule_avlm_mimo_model_custom_pgs()
    # hidden_size = 64
    # batch_size = 2
    # seq_len = 2048
    # img_h = 224
    # img_w = 224
    # patch_dim = 16
    # vocab_size = 48000
    # special_token_ids = {"images": 50257}
    # mimo_model = get_vlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, special_token_ids=special_token_ids)
    # print(mimo_model)
    dist.destroy_process_group()
