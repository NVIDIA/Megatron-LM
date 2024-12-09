# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import argparse
import json
import pprint
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file as safe_save_file

# The regex to extract layer names.
LAYER_RE = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

LOCAL_HF_FILES_PATH = Path(__file__).absolute().parent.parent / "local-hf-files"

try:
    from transformers.modeling_utils import shard_checkpoint


    def save_state_dict(path, state_dict, max_shard_size):
        shards_dict, shards_index = shard_checkpoint(
            state_dict, max_shard_size, weights_name='model.safetensors')

        # Save index.
        if shards_index:
            # Only save non-empty shards index.
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f_safe_index:
                content = json.dumps(shards_index, indent=2, sort_keys=True) + "\n"
                f_safe_index.write(content)

        # Save shards.
        for shard_file, shard in shards_dict.items():
            shard_filename = path / shard_file
            print(f'Saving to shard checkpoint {shard_filename} ...')
            safe_save_file(shard, shard_filename, metadata={"format": "pt"})

except ImportError:
    print('WARNING: Cannot import `transformers.modeling_utils.shard_checkpoint`, '
          'use `huggingface_hub.split_torch_state_dict_into_shards` instead.')

    from huggingface_hub import split_torch_state_dict_into_shards


    def save_state_dict(path, state_dict, max_shard_size):
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern="model{suffix}.safetensors", max_shard_size=max_shard_size)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            safe_save_file(
                shard,
                path / filename,
                metadata={"format": "pt"},
            )
        if state_dict_split.is_sharded:
            shards_index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            # Only save non-empty shards index.
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f_safe_index:
                content = json.dumps(shards_index, indent=2, sort_keys=True) + "\n"
                f_safe_index.write(content)


def save_hf_checkpoint(
        path: Path,
        state_dict: dict,
        max_shard_size: Optional[str],
):
    path.mkdir(exist_ok=True, parents=True)

    if max_shard_size is None:
        ckpt_filename = path / 'pytorch_model.bin'
        print(f'Saving to no-shard checkpoint ...')
        torch.save(state_dict, ckpt_filename)
    else:
        save_state_dict(path, state_dict, max_shard_size)
    print(f'Successful saved checkpoint to {path}')


def add_arguments(parser):
    group = parser.add_argument_group(title='Hf qwen saver')
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help='Sharded size of converted HF checkpoint, e.g. "2GB", "8GB", default to None (no shards)',
    )
    return group


def split_qkv(
        param, num_heads, hidden_size, num_key_value_heads
):
    input_shape = param.size()
    channels = hidden_size // num_heads
    saved_shape = [num_key_value_heads, (num_heads // num_key_value_heads + 2) * channels] + list(input_shape[1:])
    qkv_weight = param.view(*saved_shape)
    query, key, value = qkv_weight.split(
        [num_heads // num_key_value_heads * channels, channels, channels], dim=1)

    query, key, value = query.contiguous().view([-1] + list(input_shape[1:])), \
        key.contiguous().view([-1] + list(input_shape[1:])), \
        value.contiguous().view([-1] + list(input_shape[1:]))

    return query, key, value


def construct_qwen2moe_config(
        megatron_cfg: argparse.Namespace,
        num_query_groups: int = None,
):
    assert getattr(megatron_cfg, 'num_experts', 1) > 1, 'Not a MoE model'

    try:
        from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
    except ImportError:
        raise('Cannot import Qwen2MoeForCausalLM from transformers.')

    print("Converting from megatron to qwen2-moe ...")

    if megatron_cfg.moe_shared_expert_intermediate_size is not None:
        moe_shared_expert_intermediate_size = megatron_cfg.moe_shared_expert_intermediate_size
    else:
        moe_shared_expert_intermediate_size = 0

    # Spell out all parameters.
    qwen2_moe_cfg = Qwen2MoeConfig(
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_size=megatron_cfg.hidden_size,
        intermediate_size=megatron_cfg.ffn_hidden_size * megatron_cfg.moe_router_topk,
        max_position_embeddings=megatron_cfg.max_position_embeddings,
        num_attention_heads=megatron_cfg.num_attention_heads,
        num_key_value_heads=megatron_cfg.num_attention_heads,
        num_hidden_layers=getattr(megatron_cfg, "num_layers_without_padding", megatron_cfg.num_layers),
        rms_norm_eps=megatron_cfg.norm_epsilon,
        rope_theta=megatron_cfg.rotary_base,
        torch_dtype="bfloat16",
        vocab_size=megatron_cfg.padded_vocab_size,
        tie_word_embeddings=not megatron_cfg.untie_embeddings_and_output_weights,
        decoder_sparse_step=1,
        moe_intermediate_size=megatron_cfg.ffn_hidden_size,
        shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        num_experts_per_tok=megatron_cfg.moe_router_topk,
        num_experts=megatron_cfg.num_experts,
        output_router_logits=False,
        router_aux_loss_coef=0.001
    )
    if num_query_groups is not None:
        qwen2_moe_cfg.num_key_value_heads = num_query_groups

    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen2_moe_cfg.num_key_value_heads = megatron_cfg.num_query_groups

    qwen2_moe_cfg.architectures = ["Qwen2MoeForCausalLM"]
    print('Qwen2-MoE config:', qwen2_moe_cfg)
    return qwen2_moe_cfg

def construct_qwen2_config(
        megatron_cfg: argparse.Namespace,
        num_query_groups: int = None,
):
    try:
        from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    except ImportError as e:
        print('Cannot import Qwen2Model, please check your transformers install.')
        exit(1)

    print("Converting from megatron to qwen2 ...")

    config_dict = dict(
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_size=megatron_cfg.hidden_size,
        intermediate_size=megatron_cfg.ffn_hidden_size,
        max_position_embeddings=megatron_cfg.max_position_embeddings,
        num_attention_heads=megatron_cfg.num_attention_heads,
        num_key_value_heads=megatron_cfg.num_attention_heads,
        num_hidden_layers=megatron_cfg.num_layers,
        rms_norm_eps=megatron_cfg.norm_epsilon,
        rope_theta=megatron_cfg.rotary_base,
        torch_dtype='bfloat16',
        vocab_size=megatron_cfg.padded_vocab_size,
        tie_word_embeddings=not megatron_cfg.untie_embeddings_and_output_weights,
    )
    qwen2_cfg = Qwen2Config(**config_dict)

    if num_query_groups is not None:
        qwen2_cfg.num_key_value_heads = num_query_groups
    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen2_cfg.num_key_value_heads = megatron_cfg.num_query_groups

    qwen2_cfg.architectures = ["Qwen2ForCausalLM"]
    print('Qwen2 config:', qwen2_cfg)
    return qwen2_cfg

def set_dense_mlp(qwen2_hf, prefix, msg):
    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l1_weight = msg.pop("mlp l1 weight")
    qwen2_hf[f"{prefix}.mlp.gate_proj.weight"] = mlp_l0_weight_W
    qwen2_hf[f"{prefix}.mlp.up_proj.weight"] = mlp_l0_weight_V
    qwen2_hf[f"{prefix}.mlp.down_proj.weight"] = mlp_l1_weight


def set_moe_mlp(qwen2_hf, prefix, msg, md):
    shared_expert_mlp_l0_weight_W = msg.pop("shared mlp l0 weight W")
    shared_expert_mlp_l0_weight_V = msg.pop("shared mlp l0 weight V")
    shared_expert_mlp_l1_weight = msg.pop("shared mlp l1 weight")
    shared_expert_gate_weight = msg.pop("shared gate weight")
    qwen2_hf[f'{prefix}.mlp.shared_expert_gate.weight'] = shared_expert_gate_weight
    qwen2_hf[f'{prefix}.mlp.shared_expert.gate_proj.weight'] = shared_expert_mlp_l0_weight_W
    qwen2_hf[f'{prefix}.mlp.shared_expert.up_proj.weight'] = shared_expert_mlp_l0_weight_V
    qwen2_hf[f'{prefix}.mlp.shared_expert.down_proj.weight'] = shared_expert_mlp_l1_weight

    router_weight = msg.pop("router weight")
    qwen2_hf[f'{prefix}.mlp.gate.weight'] = router_weight

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l1_weight = msg.pop("mlp l1 weight")

    assert len(mlp_l0_weight_W) == md.num_experts
    for expert_idx in range(md.num_experts):
        qwen2_hf[prefix + f".mlp.experts.{expert_idx}.gate_proj.weight"] = mlp_l0_weight_W[expert_idx]
        qwen2_hf[prefix + f".mlp.experts.{expert_idx}.up_proj.weight"] = mlp_l0_weight_V[expert_idx]
        qwen2_hf[prefix + f".mlp.experts.{expert_idx}.down_proj.weight"] = mlp_l1_weight[expert_idx]


def save_checkpoint(queue, args):
    import os
    import sys
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    md = queue_get()
    qwen2_hf = {}
    # Embeddings
    # -----------
    embeddings_msg = queue_get("embeddings")
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = embeddings_msg['word embeddings'][: md.padded_vocab_size, :]
    qwen2_hf["model.embed_tokens.weight"] = word_embeddings

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for layer_idx in range(md.num_layers):
        layer_name = f"model.layers.{layer_idx}"

        msg = queue_get(f"transformer layer {total_layer_num}")

        input_norm_weight = msg.pop("input norm weight")
        post_norm_weight = msg.pop("post norm weight")
        qwen2_hf[layer_name + ".input_layernorm.weight"] = input_norm_weight
        qwen2_hf[layer_name + ".post_attention_layernorm.weight"] = post_norm_weight

        # attention
        qkv_weight = msg.pop("qkv weight")
        dense_weight = msg.pop("dense weight")

        hidden_size = md.hidden_size
        heads = md.num_attention_heads
        num_key_value_heads = md.num_query_groups
        q, k, v = split_qkv(
            qkv_weight, heads, hidden_size, num_key_value_heads
        )
        qwen2_hf[layer_name + ".self_attn.q_proj.weight"] = q
        qwen2_hf[layer_name + ".self_attn.k_proj.weight"] = k
        qwen2_hf[layer_name + ".self_attn.v_proj.weight"] = v
        # Transpose the bias.
        if md.qkv_bias:
            qkv_bias = msg.pop("qkv bias")
            q_b, k_b, v_b = split_qkv(
                qkv_bias, heads, hidden_size, num_key_value_heads
            )
            qwen2_hf[layer_name + ".self_attn.q_proj.bias"] = q_b
            qwen2_hf[layer_name + ".self_attn.k_proj.bias"] = k_b
            qwen2_hf[layer_name + ".self_attn.v_proj.bias"] = v_b
        qwen2_hf[layer_name + ".self_attn.o_proj.weight"] = dense_weight

        # mlp
        if md.num_experts:
            set_moe_mlp(qwen2_hf, layer_name, msg, md)
        else:
            set_dense_mlp(qwen2_hf, layer_name, msg)

        total_layer_num = total_layer_num + 1

    msg = queue_get("final norm")
    final_norm_weight = msg.pop("weight")
    qwen2_hf["model.norm.weight"] = final_norm_weight

    if md.output_layer:
        msg = queue_get("output layer")
        # LM head
        if md.untie_embeddings_and_output_weights:
            qwen2_hf["lm_head.weight"] = msg.pop("weight")
        else:
            qwen2_hf["lm_head.weight"] = word_embeddings

    if md.num_experts:
        qwen2_cfg = construct_qwen2moe_config(md, num_query_groups=md.num_query_groups)
    else:
        qwen2_cfg = construct_qwen2_config(md, num_query_groups=md.num_query_groups)
    save_hf_checkpoint(Path(args.save_dir), qwen2_hf, args.shard)
    qwen2_cfg.save_pretrained(Path(args.save_dir))
    print("Done!")
