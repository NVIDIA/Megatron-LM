# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Huggingface model schemas."""

import typing as T

# Using this instead of starting from ModelSchema, since cannot make similar assumptions about for example 'layer' location and HF state dicts use fully spelled out keys instead of nested attributes so _set_deep_tensor will not work.
class HFSchema:
    def __init__(self, schema, layer_schema, prefix, layer_prefix):
        self.mapping = schema
        self.layer_prefix = f"{prefix}{layer_prefix}"
        self.layer_mapping = layer_schema

    def set(self, state_dict, params):
        for k, p in params.items():
            assert k in self.mapping, f"params_dict contains key {k} that isn't specified in the schema"
            state_dict[self.mapping[k]] = p.clone()

    def set_layer(self, state_dict, layer_idx, params):
        for k, p in params.items():
            assert k in self.layer_mapping, f"params_dict for layer {layer_idx} contains key {k} that isn't specified in the schema"
            state_dict[f"{self.layer_prefix}.{layer_idx}.{self.layer_mapping[k]}"] = p.clone()


class HFLMSchema(HFSchema):
    def __init__(self, prefix, layer_prefix, use_swiglu=False):
        schema = {
            "word_embeddings": f"{prefix}model.embed_tokens.weight",
            "position_embeddings": f"{prefix}embeddings.position_embedding",
            "final_norm": f"{prefix}model.norm.weight",
            "output_layer": f"{prefix}lm_head.weight",
        }

        layer_schema = {
            "input_norm_weight": "input_layernorm.weight",
            "input_norm_bias": "input_layernorm.bias",
            "post_norm_weight": "post_attention_layernorm.weight",
            "post_norm_bias": "post_attention_layernorm.bias",
            "mlp_l0_weight_W": "mlp.gate_proj.weight",
            "mlp_l0_weight_V": "mlp.up_proj.weight",
            "mlp_l0_weight": "mlp.fc1.weight",
            "mlp_l0_bias_W": "mlp.gate_proj.bias",
            "mlp_l0_bias_V": "mlp.up_proj.bias",
            "mlp_l0_bias": "mlp.fc1.bias",
            "q_proj_weight": "self_attn.q_proj.weight",
            "k_proj_weight": "self_attn.k_proj.weight",
            "v_proj_weight": "self_attn.v_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "dense_weight": "self_attn.o_proj.weight",
            "dense_bias": "self_attn.o_proj.bias",
            "mlp_l1_weight": "mlp.down_proj.weight" if use_swiglu else "mlp.fc2.weight",
            "mlp_l1_bias": "mlp.down_proj.bias" if use_swiglu else "mlp.fc2.bias",
        }
        super().__init__(schema=schema, layer_schema=layer_schema, prefix=prefix, layer_prefix=layer_prefix)

class HFInternViTSchema(HFSchema):
    def __init__(self, prefix, layer_prefix, use_swiglu=False):
        schema = {
            "patch_embedding_weight": f"{prefix}embeddings.patch_embedding.weight",
            "patch_embedding_bias": f"{prefix}embeddings.patch_embedding.bias",
            "position_embeddings": f"{prefix}embeddings.position_embedding",
            "class_token": f"{prefix}embeddings.class_embedding",
        }

        layer_schema = {
            "ls1": "ls1",
            "ls2": "ls2",
            "input_norm_weight": "norm1.weight",
            "input_norm_bias": "norm1.bias",
            "pre_mlp_norm_weight": "norm2.weight",
            "pre_mlp_norm_bias": "norm2.bias",
            "k_norm_weight": "attn.k_norm.weight",
            "q_norm_weight": "attn.q_norm.weight",
            "k_norm_bias": "attn.k_norm.bias",
            "q_norm_bias": "attn.q_norm.bias",
            "mlp_l0_weight_W": "mlp.gate_proj.weight",
            "mlp_l0_weight_V": "mlp.up_proj.weight",
            "mlp_l0_weight": "mlp.fc1.weight",
            "mlp_l0_bias_W": "mlp.gate_proj.bias",
            "mlp_l0_bias_V": "mlp.up_proj.bias",
            "mlp_l0_bias": "mlp.fc1.bias",
            "qkv_weight": "attn.qkv.weight",
            "qkv_bias": "attn.qkv.bias",
            "dense_weight": "attn.proj.weight",
            "dense_bias": "attn.proj.bias",
            "mlp_l1_weight": "mlp.down_proj.weight" if use_swiglu else "mlp.fc2.weight",
            "mlp_l1_bias": "mlp.down_proj.bias" if use_swiglu else "mlp.fc2.bias",
        }
        super().__init__(schema=schema, layer_schema=layer_schema, prefix=prefix, layer_prefix=layer_prefix)


class HFSiglipSchema(HFSchema):
    def __init__(self, prefix, layer_prefix, use_swiglu=False):
        schema = {
            "patch_embedding_weight": f"{prefix}embeddings.patch_embedding.weight",
            "patch_embedding_bias": f"{prefix}embeddings.patch_embedding.bias",
            "ln_post_weight": f"{prefix}post_layernorm.weight",
            "ln_post_bias": f"{prefix}post_layernorm.bias",
            "position_embeddings": f"{prefix}embeddings.position_embedding.weight",
        }

        layer_schema = {
            "input_norm_weight": f"layer_norm1.weight",
            "input_norm_bias": f"layer_norm1.bias",
            "pre_mlp_norm_weight": f"layer_norm2.weight",
            "pre_mlp_norm_bias": f"layer_norm2.bias",
            "mlp_l0_weight_W": "mlp.gate_proj.weight",
            "mlp_l0_weight_V": "mlp.up_proj.weight",
            "mlp_l0_weight": "mlp.fc1.weight",
            "mlp_l0_bias_W": "mlp.gate_proj.bias",
            "mlp_l0_bias_V": "mlp.up_proj.bias",
            "mlp_l0_bias": "mlp.fc1.bias",
            "q_proj_weight": "self_attn.q_proj.weight",
            "k_proj_weight": "self_attn.k_proj.weight",
            "v_proj_weight": "self_attn.v_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "dense_weight": "self_attn.out_proj.weight",
            "dense_bias": "self_attn.out_proj.bias",
            "mlp_l1_weight": "mlp.down_proj.weight" if use_swiglu else "mlp.fc2.weight",
            "mlp_l1_bias": "mlp.down_proj.bias" if use_swiglu else "mlp.fc2.bias",
        }
        super().__init__(schema=schema, layer_schema=layer_schema, prefix=prefix, layer_prefix=layer_prefix)


class HFRADIOSchema(HFSchema):
    def __init__(self, prefix, layer_prefix, use_swiglu=False):
        schema = {
            "embedder_weight": f"{prefix}model.patch_generator.embedder.weight",
            "class_token": f"{prefix}model.patch_generator.cls_token.token",
            "position_embeddings": f"{prefix}model.patch_generator.pos_embed",
            "input_conditioner_norm_mean": f"{prefix}input_conditioner.norm_mean",
            "input_conditioner_norm_std": f"{prefix}input_conditioner.norm_std",
        }

        layer_schema = {
            "input_norm_weight": "norm1.weight",
            "input_norm_bias": "norm1.bias",
            "pre_mlp_norm_weight": "norm2.weight",
            "pre_mlp_norm_bias": "norm2.bias",
            "mlp_l0_weight_W": "mlp.gate_proj.weight",
            "mlp_l0_weight_V": "mlp.up_proj.weight",
            "mlp_l0_weight": "mlp.fc1.weight",
            "mlp_l0_bias_W": "mlp.gate_proj.bias",
            "mlp_l0_bias_V": "mlp.up_proj.bias",
            "mlp_l0_bias": "mlp.fc1.bias",
            "qkv_weight": "attn.qkv.weight",
            "qkv_bias": "attn.qkv.bias",
            "dense_weight": "attn.proj.weight",
            "dense_bias": "attn.proj.bias",
            "mlp_l1_weight": "mlp.down_proj.weight" if use_swiglu else "mlp.fc2.weight",
            "mlp_l1_bias": "mlp.down_proj.bias" if use_swiglu else "mlp.fc2.bias",
        }
        super().__init__(schema=schema, layer_schema=layer_schema, prefix=prefix, layer_prefix=layer_prefix)


def get_vision_model_schema(
    vision_model_type,
    prefix: T.Optional[str] = "",
    layer_prefix: T.Optional[str] = "",
    use_swiglu=False,
) -> HFSchema:
    return {
        "internvit" : HFInternViTSchema,
        "siglip" : HFSiglipSchema,
        "radio" : HFRADIOSchema,
    }[vision_model_type](prefix, layer_prefix, use_swiglu=use_swiglu)


def get_language_model_schema(
    prefix: T.Optional[str] = "",
    layer_prefix: T.Optional[str] = "",
    use_swiglu=False,
) -> HFSchema:
    return HFLMSchema(prefix, layer_prefix, use_swiglu=use_swiglu)
