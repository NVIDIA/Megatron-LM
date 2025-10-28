import torch.distributed as dist
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from examples.mimo.configs.llava_vlm import get_llava_projection_layer_spec, get_llava_projection_config
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups

def get_language_model_spec(num_layers, hidden_size, vocab_size, seq_len, pg_collection):
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
        num_layers=num_layers, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
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


def get_vision_submodules_spec(num_layers, hidden_size, language_hidden_size, pg_collection):
    """Get the submodule spec for the vision modality.
    
    Args:
        num_layers: Number of transformer layers in vision encoder
        hidden_size: Hidden size of vision encoder
        language_hidden_size: Hidden size of language model (for projection output)
        pg_collection: Process group collection
    """
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1

    vision_config = TransformerConfig(
        num_layers=num_layers, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
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

    # Create vision projection spec - projects from vision hidden size to language hidden size
    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": get_llava_projection_config(
                hidden_size=language_hidden_size  # Output size should match language model
            ),
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,  # Input size from vision encoder
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
    vision_num_layers, vision_hidden_size, language_num_layers, language_hidden_size, 
    vocab_size, seq_len, special_token_ids, 
    vision_tp, vision_pp, vision_dp,
    language_tp, language_pp, language_dp
):
    # Calculate offsets for grids to avoid overlap (CP and EP are hardcoded to 1)
    vision_grid_size = vision_tp * vision_pp * vision_dp
    language_module_grid = create_hypercomm_grid(offset=vision_grid_size, tp=language_tp, cp=1, pp=language_pp, dp=language_dp)
    language_pg_collection = _get_pg_collection_with_embedding_groups(language_module_grid)

    vision_module_grid = create_hypercomm_grid(offset=0, tp=vision_tp, cp=1, pp=vision_pp, dp=vision_dp)
    vision_pg_collection = _get_pg_collection_with_embedding_groups(vision_module_grid)

    language_model_spec = get_language_model_spec(language_num_layers, language_hidden_size, vocab_size, seq_len, language_pg_collection)
    vision_submodule_spec = get_vision_submodules_spec(vision_num_layers, vision_hidden_size, language_hidden_size, vision_pg_collection)

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
