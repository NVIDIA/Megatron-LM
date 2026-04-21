import os
import sys

from schema_hf import get_language_model_schema, get_vision_model_schema
from saver_hf_llava import HFCheckpointSaverLLaVA
from saver_hf_moe import HFCheckpointSaverMoE


def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace LLaVA MoE saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')


class HFCheckpointSaverLLaVAMoE(HFCheckpointSaverLLaVA, HFCheckpointSaverMoE):
    def __init__(self, args, queue):
        HFCheckpointSaverLLaVA.__init__(self, args, queue)

    def receive_model(self):
        # Vision backbone
        vision_model_prefix = "vision_model.vision_model."
        vision_layer_prefix = "encoder.layers"
        if self.md.vision_model_type == "radio":
            vision_model_prefix = "vision_model.radio_model."
            vision_layer_prefix = "model.blocks"
        vision_schema = get_vision_model_schema(
            self.md.vision_model_type,
            prefix=vision_model_prefix,
            layer_prefix=vision_layer_prefix,
            use_swiglu=self.md.vision_swiglu,
        )
        self.receive_vision_backbone(vision_schema)
        self.receive_vision_projection()

        # Sound model (if present)
        # Note: Order must match loader_llava.py - projection comes before model
        if getattr(self.md, 'sound_model_type', None) is not None:
            self.receive_sound_projection()
            self.receive_sound_model()

        # Language model (MoE-aware)
        language_model_prefix = "language_model."
        if self.md.model_type == "hybrid":
            language_layer_prefix = "backbone.layers"
        else:
            language_layer_prefix = "model.layers"
        language_schema = get_language_model_schema(
            model_type=self.args.model_type,
            prefix=language_model_prefix,
            layer_prefix=language_layer_prefix,
            use_swiglu=self.md.swiglu,
        )
        print(f"language_schema: {language_schema}")
        HFCheckpointSaverMoE.receive_lm(self, language_schema)


def save_checkpoint(queue, args):
    saver = HFCheckpointSaverLLaVAMoE(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e


