import math

from megatron.core.model_parallel_config import ModelParallelConfig
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel

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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.extensions.transformer_engine import TELinear
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_from_grid, _get_pg_collection_with_embedding_groups
import logging
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from examples.mimo.configs.llava_vlm import get_llava_projection_layer_spec, get_llava_projection_config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('megatron.core.models.mimo.model.base').setLevel(logging.DEBUG)

def get_language_model_spec(hidden_size, vocab_size, seq_len, pg_collection):
    """Get the language model spec."""
    lm_config = TransformerConfig(
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
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
        num_layers=16, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
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
        modality_submodules_spec={"images": vision_submodule_spec, "audio": audio_submodule_spec},
        special_token_ids=special_token_ids,
    )
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    return mimo_model


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
