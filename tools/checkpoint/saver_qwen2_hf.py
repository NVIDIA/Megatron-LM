# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import argparse
import collections
import json
import pprint
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file as safe_save_file
from transformers.pytorch_utils import id_tensor_storage

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

KNOWN_TIED_WEIGHT_KEYS = ["lm_head.weight"]


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
        # Remove tensor aliases before sharding.
        # See `transformers.modeling_utils.PreTrainedModel.save_pretrained` for more details.

        # Safetensors does not allow tensor aliasing.
        # We're going to remove aliases before saving
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            # Sometimes in the state_dict we have non-tensor objects.
            # e.g. in bitsandbytes we have some `str` objects in the state_dict
            if isinstance(tensor, torch.Tensor):
                ptrs[id_tensor_storage(tensor)].append(name)
            else:
                # In the non-tensor case, fall back to the pointer of the object itself
                ptrs[id(tensor)].append(name)

        # These are all the pointers of shared tensors.
        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

        # [NOTE] (suyang.fy):
        #  To simplify the implementation, this function only allowed tensor aliases in `KNOWN_TIED_WEIGHT_KEYS`.
        warn_names = set()
        for names in shared_ptrs.values():
            found = 0
            for name in sorted(names):
                matches_pattern = any(re.search(pat, name) for pat in KNOWN_TIED_WEIGHT_KEYS)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        print(f'Remove known tensor alias {name} from the state dict.')
                        del state_dict[name]

            found = 0
            for name in names:
                if name in state_dict:
                    found += 1
                    if found > 1:
                        del state_dict[name]
                        warn_names.add(name)

            if len(warn_names) > 0:
                raise RuntimeError(
                    f"Unexpected tensor aliases {warn_names} detected while saving. Please check the checkpoint again.",
                )

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


def fix_query_key_value_ordering_qwen2(
    param, num_heads, hidden_size, num_key_value_heads
):
    input_shape = param.size()
    channels = hidden_size // num_heads
    saved_shape = [num_key_value_heads, (num_heads // num_key_value_heads + 2) * channels] + list(input_shape[1:])
    qkv_weight = param.view(*saved_shape)  # 8+1+1, 4, 128, hid
    query, key, value = qkv_weight.split(
        [num_heads // num_key_value_heads * channels, channels, channels], dim=1)

    query, key, value = query.contiguous().view([-1] + list(input_shape[1:])), \
        key.contiguous().view([-1] + list(input_shape[1:])), \
        value.contiguous().view([-1] + list(input_shape[1:]))

    return query, key, value

def convert_megatron_to_qwen2_moe(
    megatron_cfg: argparse.Namespace,
    megatron_sd,
    num_query_groups: int = None,
):
    assert getattr(megatron_cfg, 'num_experts', 1) > 1, 'Not a MoE model'

    try:
        from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
    except ImportError:
        print('Cannot import Qwen2MoeModel from transformers, use local version.')
        from support.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
        moe_in_repo = False
    else:
        moe_in_repo = True

    print("Converting from megatron to qwen2-moe ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

    if megatron_cfg.moe_shared_expert_intermediate_size is not None:
        moe_shared_expert_intermediate_size = megatron_cfg.moe_shared_expert_intermediate_size
    else:
        moe_shared_expert_intermediate_size = 0

    # Spell out all parameters in case the defaults change.
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
        decoder_sparse_step=1,   #megatron_cfg.moe_expert_interval,
        moe_intermediate_size=megatron_cfg.ffn_hidden_size,
        shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        num_experts_per_tok=megatron_cfg.moe_router_topk,
        num_experts=megatron_cfg.num_experts,
        output_router_logits=False,
        router_aux_loss_coef=0.001 #megatron_cfg.moe_loss_weight,
    )
    if num_query_groups is not None:
        qwen2_moe_cfg.num_key_value_heads = num_query_groups

    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen2_moe_cfg.num_key_value_heads = megatron_cfg.num_query_groups

    qwen2_moe_cfg.architectures = ["Qwen2MoeForCausalLM"]
    if not moe_in_repo:
        qwen2_moe_cfg.auto_map = {
            'AutoConfig': 'configuration_qwen2_moe.Qwen2MoeConfig',
            'AutoModelForCausalLM': 'modeling_qwen2_moe.Qwen2MoeForCausalLM',
        }
    print('Qwen2-MoE config:', qwen2_moe_cfg)

    def _name(*parts) -> str:
        """Make parameter name from parts."""
        name = '.'.join(parts)
        return name

    # The converted output model.
    qwen2_moe_sd = {}
    # The hidden_size
    hidden_size = qwen2_moe_cfg.hidden_size
    # The number of heads.
    heads = qwen2_moe_cfg.num_attention_heads
    num_key_value_heads = qwen2_moe_cfg.num_key_value_heads
    # Number of routed experts
    num_experts = qwen2_moe_cfg.num_experts

    print(f"megatron_sd: {megatron_sd.keys()}")
    prefix_name = "model."

    # The word embeddings.
    word_embeddings = megatron_sd["embedding.word_embeddings.weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: qwen2_moe_cfg.vocab_size, :]
    qwen2_moe_sd[f"{prefix_name}embed_tokens.weight"] = word_embeddings
    print(f"word_embeddings: {word_embeddings.size()}. vocab_size: {qwen2_moe_cfg.vocab_size}.")

    # The transformer. now encoder
    transformer = megatron_sd

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "self_attention.linear_proj": "self_attn.o_proj",
        "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate",
        "mlp.router": "mlp.gate",
        "mlp.shared_experts.linear_fc2": "mlp.shared_expert.down_proj",
    }

    # Extract the layers.
    for key, val in transformer.items():
        m = LAYER_RE.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"{prefix_name}layers.{layer_idx}"
        # For layernorm(s), simply store the layer norm.
        # print(op_name)
        if op_name == 'mlp.shared_experts.linear_fc1':
            w1, w2 = val.chunk(2, dim=0)
            w1 = w1.clone().detach()
            w2 = w2.clone().detach()
            w1_name = '.'.join([layer_name + f".mlp.shared_expert.gate_proj.weight"])
            w2_name = '.'.join([layer_name + f".mlp.shared_expert.up_proj.weight"])
            qwen2_moe_sd[w1_name] = w1
            qwen2_moe_sd[w2_name] = w2

        # For layernorm(s), simply store the layer norm.
        elif op_name.endswith("layernorm"):

            if op_name.startswith("self_attention."):
                qwen2_moe_sd[
                    _name(layer_name, 'input_layernorm', weight_or_bias)
                ] = val.clone()
            elif op_name.startswith("pre_mlp_layernorm"):
                qwen2_moe_sd[
                    _name(layer_name, 'post_attention_layernorm', weight_or_bias)
                ] = val.clone()

        elif op_name == "mlp.experts.local_experts.linear_fc1":
            experts = val
            assert len(experts) == num_experts
            for expert_idx in range(num_experts):
                w1, w2 = experts[expert_idx].chunk(2, dim=0)
                w1 = w1.clone().detach()
                w2 = w2.clone().detach()
                qwen2_moe_sd[layer_name + f".mlp.experts.{expert_idx}.gate_proj.weight"] = w1
                qwen2_moe_sd[layer_name + f".mlp.experts.{expert_idx}.up_proj.weight"] = w2
        elif op_name == "mlp.experts.local_experts.linear_fc2":
            experts = val
            assert len(experts) == num_experts
            for expert_idx in range(num_experts):
                qwen2_moe_sd[layer_name + f".mlp.experts.{expert_idx}.down_proj.weight"] = experts[expert_idx].clone()
        # Transpose the QKV matrix.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "weight":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )

            qwen2_moe_sd[layer_name + ".self_attn.q_proj.weight"] = q
            qwen2_moe_sd[layer_name + ".self_attn.k_proj.weight"] = k
            qwen2_moe_sd[layer_name + ".self_attn.v_proj.weight"] = v
        # Transpose the bias.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "bias":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )
            qwen2_moe_sd[layer_name + ".self_attn.q_proj.bias"] = q
            qwen2_moe_sd[layer_name + ".self_attn.k_proj.bias"] = k
            qwen2_moe_sd[layer_name + ".self_attn.v_proj.bias"] = v
        # Transpose the weights. Add clone() to avoid shared data.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            qwen2_moe_sd[_name(layer_name, out_name, "weight")] = val.clone()
        # Copy the bias. Add clone() to avoid shared data.
        elif weight_or_bias == "bias":
            assert False
            out_name = megatron_to_transformers[op_name]
            qwen2_moe_sd[_name(layer_name, out_name, "bias")] = val.clone()
        else:
            assert False

    # DEBUG.
    assert qwen2_moe_cfg.num_hidden_layers == layer_idx + 1

    # The final layernorm.
    qwen2_moe_sd[f"{prefix_name}norm.weight"] = transformer[
        "decoder.final_layernorm.weight"
    ]

    # LM head
    if not qwen2_moe_cfg.tie_word_embeddings:
        output_layer = transformer["output_layer.weight"]
        qwen2_moe_sd["lm_head.weight"] = output_layer
    else:
        qwen2_moe_sd["lm_head.weight"] = word_embeddings
    # It should be done!
    return qwen2_moe_cfg, qwen2_moe_sd

def convert_megatron_to_qwen2(
    megatron_cfg: argparse.Namespace,
    megatron_sd,
    num_query_groups: int = None,
):
    try:
        from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    except ImportError as e:
        print('Cannot import Qwen2Model, please check your transformers install.')
        exit(1)

    print("Converting from megatron to qwen2 ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

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

    def _name(*parts) -> str:
        """Make parameter name from parts."""
        name = '.'.join(parts)
        return name

    # The converted output model.
    qwen2_sd = {}

    # The number of heads.
    heads = qwen2_cfg.num_attention_heads
    num_key_value_heads = qwen2_cfg.num_key_value_heads
    # The hidden_size .
    hidden_size = qwen2_cfg.hidden_size
    print(megatron_sd.keys())
    # The word embeddings.
    word_embeddings = megatron_sd["embedding.word_embeddings.weight"]

    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: qwen2_cfg.vocab_size, :]
    qwen2_sd[_name("model", "embed_tokens", "weight")] = word_embeddings
    print(f"word_embeddings: {word_embeddings.size()}. vocab_size: {qwen2_cfg.vocab_size}.")

    # The transformer. now encoder
    transformer = megatron_sd

    # The simple map of names for "automated" rules.

    megatron_to_transformers = {
        "mlp.linear_fc2": "mlp.down_proj",
        "self_attention.linear_proj": "self_attn.o_proj"
    }

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = LAYER_RE.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = m.group(1)
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        if op_name == 'mlp.linear_fc1':
            w1, w2 = val.chunk(2, dim=0)
            w1 = w1.clone().detach()
            w2 = w2.clone().detach()
            w1_name = _name('model', 'layers', layer_idx, 'mlp', 'gate_proj', 'weight')
            w2_name = _name('model', 'layers', layer_idx, 'mlp', 'up_proj', 'weight')
            qwen2_sd[w1_name] = w1
            qwen2_sd[w2_name] = w2

        # For layernorm(s), simply store the layer norm.
        elif op_name.endswith("layernorm"):

            if op_name.startswith("self_attention."):
                qwen2_sd[
                    _name('model', 'layers', layer_idx, 'input_layernorm', weight_or_bias)
                ] = val
            elif op_name.startswith("mlp"):
                qwen2_sd[
                    _name('model', 'layers', layer_idx, 'post_attention_layernorm', weight_or_bias)
                ] = val

        # Transpose the QKV matrix.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "weight":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'weight')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'weight')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'weight')
            qwen2_sd[q_name] = q
            qwen2_sd[k_name] = k
            qwen2_sd[v_name] = v

        # Transpose the bias.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "bias":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'bias')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'bias')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'bias')
            qwen2_sd[q_name] = q
            qwen2_sd[k_name] = k
            qwen2_sd[v_name] = v

        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            qwen2_key = _name('model', 'layers', layer_idx, out_name, 'weight')
            qwen2_sd[qwen2_key] = val
        else:
            raise ValueError(f'param {key} failed')

    # The final layernorm.
    qwen2_sd[_name("model", "norm", "weight")] = transformer["decoder.final_layernorm.weight"]

    # LM head
    if not qwen2_cfg.tie_word_embeddings:
        output_layer = transformer["output_layer.weight"]
        qwen2_sd[_name("lm_head", "weight")] = output_layer
    else:
        qwen2_sd["lm_head.weight"] = word_embeddings

    # print(qwen2_sd.keys())
    return qwen2_cfg, qwen2_sd

def set_dense_mlp(mcore_state, prefix, msg):
    post_norm_weight = msg.pop("post norm weight")

    mlp_l1_weight = msg.pop("mlp l1 weight")

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
    mcore_state[f'{prefix}.mlp.linear_fc1.layernorm.weight'] = post_norm_weight
    mcore_state[f'{prefix}.mlp.linear_fc1.weight'] = mlp_l0_weight
    mcore_state[f'{prefix}.mlp.linear_fc2.weight'] = mlp_l1_weight

def set_moe_mlp(mcore_state, prefix, msg):
    post_norm_weight = msg.pop("post norm weight")
    mcore_state[f'{prefix}.pre_mlp_layernorm.weight'] = post_norm_weight

    shared_expert_mlp_l0_weight_W = msg.pop("shared mlp l0 weight W")
    shared_expert_mlp_l0_weight_V = msg.pop("shared mlp l0 weight V")
    shared_expert_mlp_l0_weight = torch.cat((shared_expert_mlp_l0_weight_W, shared_expert_mlp_l0_weight_V), dim=-2)
    shared_expert_mlp_l1_weight = msg.pop("shared mlp l1 weight")
    shared_expert_gate_weight = msg.pop("shared gate weight")
    mcore_state[f'{prefix}.mlp.shared_experts.gate_weight.weight'] = shared_expert_gate_weight
    mcore_state[f'{prefix}.mlp.shared_experts.linear_fc1.weight'] = shared_expert_mlp_l0_weight
    mcore_state[f'{prefix}.mlp.shared_experts.linear_fc2.weight'] = shared_expert_mlp_l1_weight

    router_weight = msg.pop("router weight")
    mcore_state[f'{prefix}.mlp.router.weight'] = router_weight

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
    mcore_state[f'{prefix}.mlp.experts.local_experts.linear_fc1.weight'] = mlp_l0_weight
    mlp_l1_weight = msg.pop("mlp l1 weight")
    mcore_state[f'{prefix}.mlp.experts.local_experts.linear_fc2.weight'] = mlp_l1_weight

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
    mcore_state = {}
    # Embeddings
    # -----------
    embeddings_msg = queue_get("embeddings")
    mcore_state['embedding.word_embeddings.weight'] = embeddings_msg['word embeddings']

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for layer_idx in range(md.num_layers):
        prefix = f'layers.{layer_idx}'

        msg = queue_get(f"transformer layer {total_layer_num}")

        input_norm_weight = msg.pop("input norm weight")

        qkv_weight = msg.pop("qkv weight")
        dense_weight = msg.pop("dense weight")

        # Save them to the model state
        # attention
        mcore_state[f'{prefix}.self_attention.linear_proj.weight'] = dense_weight
        mcore_state[f'{prefix}.self_attention.linear_qkv.weight'] = qkv_weight
        mcore_state[f'{prefix}.self_attention.linear_qkv.layernorm.weight'] = input_norm_weight

        if md.add_qkv_bias:
            qkv_bias = msg.pop("qkv bias")
            mcore_state[f'{prefix}.self_attention.linear_qkv.bias'] = qkv_bias

        # mlp
        if md.num_experts:
            set_moe_mlp(mcore_state, prefix, msg)
        else:
            set_dense_mlp(mcore_state, prefix, msg)

        total_layer_num = total_layer_num + 1

    msg = queue_get("final norm")
    final_norm_weight = msg.pop("weight")
    mcore_state[f'decoder.final_layernorm.weight'] = final_norm_weight

    if md.output_layer:
        msg = queue_get("output layer")
        output_layer_weight = msg.pop("weight")
        mcore_state['output_layer.weight'] = output_layer_weight

    if md.num_experts:
        qwen2_cfg, output_state_dict = convert_megatron_to_qwen2_moe(md, mcore_state, num_query_groups=md.num_query_groups)
    else:
        qwen2_cfg, output_state_dict = convert_megatron_to_qwen2(md, mcore_state, num_query_groups=md.num_query_groups)
    save_hf_checkpoint(Path(args.save_dir), output_state_dict, args.shard)
    qwen2_cfg.save_pretrained(Path(args.save_dir))
    print("Done!")
