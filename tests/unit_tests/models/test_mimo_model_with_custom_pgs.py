import torch.distributed as dist
from functools import partial
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups
import logging
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from examples.mimo.configs.llava_vlm import get_llava_projection_layer_spec, get_llava_projection_config
import megatron.core.pipeline_parallel.schedules as schedule

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('megatron.core.models.mimo.model.base').setLevel(logging.DEBUG)

def is_current_rank_in_grid(grid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= dist.get_rank() < (grid.rank_offset + grid.size)
    

def multimodule_no_sync(modules_and_grids):
    contexts = []
    for module, grid in modules_and_grids:
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

def finalize_model_grads(modules_and_grids, module=None, num_tokens=None, pg_collection=None):
    for module, grid in modules_and_grids:
        if module is not None and is_current_rank_in_grid(grid):
            finalize_model_grads([module], num_tokens=None, pg_collection=_get_pg_collection_with_embedding_groups(grid))

def get_language_model_spec(hidden_size, vocab_size, seq_len, pg_collection):
    """Get the language model spec."""
    lm_config = TransformerConfig(
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
            "pg_collection": pg_collection,
        },
    )
    return language_model_spec


def get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim, pg_collection):
    """Get the submodule spec for the vision modality."""
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    vision_config = TransformerConfig(
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True
    )
    vision_encoder_spec = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
            "pg_collection": pg_collection
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


def get_avlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
    language_module_grid = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=2, dp=1)
    language_pg_collection = _get_pg_collection_with_embedding_groups(language_module_grid)

    audio_module_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=2, dp=1)
    audio_pg_collection = _get_pg_collection_with_embedding_groups(audio_module_grid)

    vision_module_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=2, dp=1)
    vision_pg_collection = _get_pg_collection_with_embedding_groups(vision_module_grid)

    language_model_spec = get_language_model_spec(hidden_size, vocab_size, seq_len, language_pg_collection)
    vision_submodule_spec = get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim, vision_pg_collection)
    audio_submodule_spec = get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim, audio_pg_collection)
    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"vision_module": vision_submodule_spec, "audio_module": audio_submodule_spec},
        special_token_ids=special_token_ids,
    )
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    module_to_grid_map = {'vision_module': mimo_model.vision_module_grid, 'audio_module': mimo_model.audio_module_grid, 'language_module': mimo_model.language_module_grid}
    topology = {
        'vision_module': ['language_module'],  # vision_module sends forward results to language_module
        'audio_module': ['language_module'],  # audio_module sends forward results to language_module
        'language_module': [],  # language_module is the last stage here
    }
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
    img_h = 224
    img_w = 224
    patch_dim = 16
    vocab_size = 48000
    special_token_ids = {"images": 50257, "audio": 50258}
    mimo_model, _, _ = get_avlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, img_h=img_h, img_w=img_w, patch_dim=patch_dim, special_token_ids=special_token_ids)
    assert mimo_model is not None
    dist.destroy_process_group()


def test_1f_1b_schedule_avlm_mimo_model_custom_pgs():

    Utils.initialize_distributed()
    hidden_size = 1024
    batch_size = 2
    seq_len = 2048
    img_h = 224
    img_w = 224
    patch_dim = 16
    vocab_size = 48000
    special_token_ids = {"images": 50257, "audio": 50258}

    mimo_model, module_to_grid_map, topology = get_avlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, img_h=img_h, img_w=img_w, patch_dim=patch_dim, special_token_ids=special_token_ids)

    mimo_model.config.no_sync_func = partial(multimodule_no_sync, modules_and_grids=module_to_grid_map)
    mimo_model.config.finalize_model_grads_func = partial(finalize_model_grads, modules_and_grids=module_to_grid_map)

    mimo_model.config.no_sync_func()

    multimodule_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config, dim_mapping={'b': 0,'s': 1, 'h': 2}
    )

    # TODO: Add data iterator and step function
    data_iterator = None
    step_func = None

    common_args = {
        'forward_step_func': step_func,
        'data_iterator': data_iterator,
        'model': [mimo_model],
        'num_microbatches': 16,
        'seq_length': seq_len,
        'micro_batch_size': batch_size,
        'forward_only': False,
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
    
    hidden_size = 64
    batch_size = 2
    seq_len = 2048
    img_h = 224
    img_w = 224
    patch_dim = 16
    vocab_size = 48000
    special_token_ids = {"images": 50257, "audio": 50258}
    mimo_model = get_avlm_mimo_model(hidden_size=hidden_size, vocab_size=vocab_size, seq_len=seq_len, img_h=img_h, img_w=img_w, patch_dim=patch_dim, special_token_ids=special_token_ids)
    print(mimo_model)
    dist.destroy_process_group()
