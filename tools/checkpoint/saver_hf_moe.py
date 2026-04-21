import os
import sys

import torch

from schema_hf import get_language_model_schema
from saver_hf import HFCheckpointSaver

from huggingface_hub import save_torch_state_dict


def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace MoE Model saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')


class HFCheckpointSaverMoE(HFCheckpointSaver):
    def __init__(self, args, queue):
        super().__init__(args, queue)

    def _receive_moe_layer(self, message: dict) -> dict:
        """
        Convert loader MoE message into a params_dict consumable by the MoE-aware HF schema.
        Returns keys that HFHybridMoELMSchema understands, including stacked expert weights.
        """
        params_dict = {}

        # Norm: for hybrid schema we use "norm_weight"/"norm_bias" (already mapped by schema)
        params_dict["norm_weight"] = message["pre mlp norm weight"]
        if self.md.norm_has_bias:
            params_dict["norm_bias"] = message["pre mlp norm bias"]

        # Router and shared experts
        params_dict["router_weight"] = message["router weight"]
        params_dict["router_bias"] = message["router bias"].to(torch.float32)
        params_dict["shared_up_proj_weight"] = message["shared mlp l0 weight"]
        params_dict["shared_down_proj_weight"] = message["shared mlp l1 weight"]

        # Experts (stacked on dim 0)
        #TODO: maybe handle swiglu case?
        experts_up = message["mlp l0 weight"]  # [E, out, in]
        params_dict["experts_up_proj_weight"] = experts_up
        params_dict["experts_down_proj_weight"] = message["mlp l1 weight"]

        return params_dict

    def receive_lm(self, schema):
        # Embeddings
        embeddings_msg = self.queue_get("embeddings")
        params_dict = {}

        params_dict["word_embeddings"] = embeddings_msg["word embeddings"]
        if self.md.position_embedding_type == "learned_absolute":
            params_dict["position_embeddings"] = embeddings_msg["position embeddings"]
        schema.set(self.state_dict, params_dict)

        # Hybrid path: allocate layers and branch
        if self.md.model_type == "hybrid":
            from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
            from megatron.core.ssm.mamba_hybrid_layer_allocation import allocate_layers

            layer_type_list = allocate_layers(
                self.md.num_layers,
                self.md.hybrid_attention_ratio,
                self.md.hybrid_mlp_ratio,
                self.md.hybrid_override_pattern,
            )

            for i in range(self.md.num_layers):
                message = self.queue_get(f"transformer layer {i}")

                layer_type = layer_type_list[i]
                if layer_type == LayerSymbols.MAMBA:
                    params_dict = self._receive_mamba_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.ATTENTION:
                    params_dict = self._receive_attention_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.MLP:
                    params_dict = self._receive_mlp_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                elif layer_type == LayerSymbols.MOE:
                    params_dict = self._receive_moe_layer(message)
                    schema.set_layer(self.state_dict, i, params_dict)
                else:
                    raise ValueError(f"hybrid layer {i} is not one of MAMBA, ATTENTION, MLP, or MOE")
        else:
            raise ValueError("Non-hybrid model is not supported for MoE")
        # Final norms and output layer
        params_dict = {
            "final_norm": self.queue_get('final norm')['weight'],
            "output_layer": self.queue_get('output layer')['weight'],
        }
        schema.set(self.state_dict, params_dict)

        msg = self.queue_get()
        if msg != "done":
            print("ERROR: got some more data but was expecting to be done")


def save_checkpoint(queue, args):
    """
    Entry point for LM-only MoE and hybrid models.
    """
    saver = HFCheckpointSaverMoE(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e
