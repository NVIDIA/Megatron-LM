# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import json
import os
import re
import sys
from typing import Optional

try:
    import habana_frameworks.torch
except ImportError:
    pass

import torch
from huggingface_hub import save_torch_state_dict
from transformers import (
    GenerationConfig,
    LlamaConfig,
    MixtralConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter

device = "cpu"

# Sourced from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L108
DEFAULT_LLAMA_SPECIAL_TOKENS = {
    "llama3": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)],
    "llama3.1": [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|reserved_special_token_2|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    + [f"<|reserved_special_token_{i}|>" for i in range(3, 256 - 8)],
}

CONTEXT_LENGTH_FOR_VERSION = {"llama3.1": 131072, "llama3": 8192, "llama2": 4096}


def add_checkpoint_conversion_args(parser):
    parser.add_argument(
        "--target-params-dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help=("The dtype of the converted checkpoint."),
    )
    parser.add_argument(
        "--ckpt-dir-name",
        type=str,
        required=False,
        help="Directory (`iter_0000008`) to load specific checkpoint for conversion.",
    )
    parser.add_argument(
        "--source-model-type",
        type=str,
        choices=["llama2", "llama3", "llama3.1", "mixtral"],
        default="llama3.1",
        help="Type of source model to be converted.",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to the megatron checkpoint to convert.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--instruct",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether the recipe is an instruct model or not. Will affect special tokens for llama 3.1.",
    )

    return parser

# Sourced from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L432
class Llama3Converter(TikTokenConverter):
    def __init__(
        self, vocab_file, special_tokens=None, instruct=False, model_max_length=None, **kwargs
    ):
        super().__init__(vocab_file, additional_special_tokens=special_tokens, **kwargs)
        tokenizer = self.converted()
        chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>" if not instruct else "<|eot_id|>",
            chat_template=chat_template if instruct else None,
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
        )


def save_tokenizer(
    tokenizer_path, input_tokenizer_path, source_model_type="llama3.1", special_tokens=None, instruct=False
):

    if source_model_type in ["llama3", "llama3.1"]:
        tokenizer = Llama3Converter(
            input_tokenizer_path,
            special_tokens,
            instruct,
            model_max_length=CONTEXT_LENGTH_FOR_VERSION[source_model_type],
        ).tokenizer
    else:
        tokenizer_class = LlamaTokenizerFast if LlamaTokenizerFast is not None else LlamaTokenizer
        if source_model_type == 'mixtral':
            tokenizer = tokenizer_class(input_tokenizer_path, add_bos_token=False, add_eos_token=False, add_prefix_space=True)
        else:
            tokenizer = tokenizer_class(input_tokenizer_path)
        print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer.save_pretrained(tokenizer_path)
    return tokenizer


def convert_dtype_to_str(obj):
    try:
        return str(obj)
    except TypeError:
        print(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    return None


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.linear_proj": ".self_attn.o_proj.",
    "mlp.linear_fc2": ".mlp.down_proj.",
    "mlp.router": ".block_sparse_moe.gate.",
}

tensor_parallel_params = [
    "self_attention.linear_qkv.weight",
    "self_attention.linear_proj.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
    "linear_fc1.weight",
    "linear_fc2.weight",
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "mlp.up_proj.weight",
    "mlp.gate_proj.weight",
]


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        print("loading", i, ":", num_checkpoints + 1)
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"pytorch_model-{i}-of-{num_checkpoints}.bin")
            assert os.path.exists(checkpoint_path), f"Cannot find checkpoint {checkpoint_path}"
        current_chunk = torch.load(checkpoint_path, map_location=device)
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(
    load_path: str,
    tp_size: int,
    pp_size: int,
    pp_rank: int,
    ep_size: Optional[int] = None,
    ep_rank: Optional[int] = None,
):
    """
    Get sharded checkpoints from Megatron-LM checkpoint based on the provided
    tensor, pipeline, expert parallel sizes and pipeline, expert parallel ranks.

    Args:
        load_path (str): the directory containing the checkpoints.
        tp_size (int): the tensor parallel size.
        pp_size (int): the pipeline parallel size.
        pp_rank (int): the pipeline parallel rank.
        ep_size (int): the expert parallel size.
        ep_rank (int): the expert parallel rank.
    """
    tp_state_dicts = []

    if pp_size == 1 and (ep_size is None or ep_size == 1):
        sub_dir_template = "mp_rank_{:02d}"
    elif pp_size == 1 and ep_size > 1:
        sub_dir_template = f"mp_rank_{{:02d}}_{ep_rank:03d}"
    elif pp_size > 1 and (ep_size is None or ep_size == 1):
        sub_dir_template = f"mp_rank_{{:02d}}_{pp_rank:03d}"
    elif pp_size > 1 and ep_size > 1:
        sub_dir_template = f"mp_rank_{{:02d}}_{pp_rank:03d}_{ep_rank:03d}"
    else:
        raise ValueError("Incorrect pipeline parallel and expert parallel sizes.")

    for i in range(tp_size):

        sub_dir_name = sub_dir_template.format(i)
        sub_dir_path = os.path.join(load_path, sub_dir_name)
        checkpoint_path = os.path.join(sub_dir_path, "model_optim_rng.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device)

        tp_state_dicts.append(state_dict)

        # Clear state_dict variable to free memory if not needed in this scope
        del state_dict

    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def convert_layers(
    source_model_type: str,
    tp_state_dicts: list,
    moe_tp_state_dicts: list,
    output_state_dict: dict,
    capacity_bins_state_dict: dict,
    layer_re: re.Pattern,
    moe_op_name_re: re.Pattern,
    tp_size: int,
    pp_rank: int,
    ep_rank: Optional[int],
    moe_tp: bool,
    num_layers_per_pp_stage: int,
    num_layers_per_ep_stage: Optional[int],
    qkv_total_dim: int,
    hidden_size: int,
    heads_per_group: int,
    hidden_size_per_head: int,
    num_query_groups: int,
    rotary_base: float,
    dtype: torch.dtype,
):
    # Extract the layers.
    for key, val in tp_state_dicts[0]["model"].items():
        try:
            # Match the name.
            match = layer_re.match(key)

            is_expert_layer = "experts.local_experts" in key

            if source_model_type == 'mixtral':
                # For ep ranks > 0 we convert only MoE layers.
                if ep_rank > 0 and not is_expert_layer:
                    continue

            # Stop if that's not a layer.
            if match is None or str(key).endswith("_extra_state"):
                print("skipping: ", key)
                continue

            print("processing: ", key)

            if "capacity_bins" in key:
                capacity_bins_state_dict[key] = val
                continue

            # The index of the layer.
            layer_idx = int(match.group(2)) + pp_rank * num_layers_per_pp_stage
            moe_layer_idx = None

            # The name of the operation.
            op_name = match.group(3)

            # Is it a weight or a bias?
            weight_or_bias = match.group(4)

            # Extract the layer index and operation name of the MoE layer.
            if is_expert_layer:
                moe_match = moe_op_name_re.match(op_name)
                moe_layer_idx = int(moe_match.group(1)) + ep_rank * num_layers_per_ep_stage
                op_name = moe_match.group(2)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            param_key = op_name + "." + weight_or_bias
            params = val.to(dtype)

            if param_key in tensor_parallel_params:
                dim = 1 if op_name in ["self_attention.linear_proj", "linear_fc2"] else 0
                for tp_rank in range(0, tp_size):
                    if tp_rank > 0:
                        tp_tensor = tp_state_dicts[tp_rank]["model"][key].to(dtype)
                        params = torch.cat([params, tp_tensor], dim=dim)

                    if moe_tp and is_expert_layer:
                        for moe_tp_state_dict in moe_tp_state_dicts:
                            moe_tp_tensor = moe_tp_state_dict[tp_rank]["model"][key].to(dtype)
                            params = torch.cat([params, moe_tp_tensor], dim=dim)

            params.to(device=device)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):
                ln_name = (
                    "input_layernorm"
                    if op_name.startswith("input")
                    else "post_attention_layernorm"
                )
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Split QKV packed weights.
            elif op_name == "self_attention.linear_qkv" and weight_or_bias == "weight":

                qkv_weights = params.reshape([qkv_total_dim, -1, hidden_size])

                q_slice = torch.cat(
                    [
                        torch.arange(
                            (heads_per_group + 2) * i,
                            (heads_per_group + 2) * i + heads_per_group,
                        )
                        for i in range(num_query_groups)
                    ]
                ).to(device)
                k_slice = torch.arange(
                    heads_per_group, qkv_total_dim, (heads_per_group + 2)
                    ).to(device)
                v_slice = torch.arange(
                    heads_per_group + 1, qkv_total_dim, (heads_per_group + 2)
                ).to(device)

                q_weights_base_name = f"{layer_name}.self_attn.q_proj.weight"
                k_weights_base_name = f"{layer_name}.self_attn.k_proj.weight"
                v_weights_base_name = f"{layer_name}.self_attn.v_proj.weight"

                output_state_dict[q_weights_base_name] = (
                    qkv_weights[q_slice].reshape(-1, hidden_size).to(dtype)
                )
                output_state_dict[k_weights_base_name] = (
                    qkv_weights[k_slice].reshape(-1, hidden_size).to(dtype)
                )
                output_state_dict[v_weights_base_name] = (
                    qkv_weights[v_slice].reshape(-1, hidden_size).to(dtype)
                )

            elif op_name == "mlp.linear_fc1" and weight_or_bias == "weight":
                    # Converts Llama mlp.linear_fc1 weights.

                    # Split params across tensor parallel size and process chunks simultaneously
                    gate_chunks, up_chunks = zip(*[t.chunk(2) for t in params.chunk(dim=0, chunks=megatron_args.tensor_model_parallel_size)])

                    # concat chunks at once, reducing memory allocations
                    gate = torch.cat(gate_chunks).to(dtype).contiguous()
                    up = torch.cat(up_chunks).to(dtype).contiguous()

                    # Assign to output state dict directly
                    output_state_dict[f"{layer_name}.mlp.gate_proj.weight"] = gate
                    output_state_dict[f"{layer_name}.mlp.up_proj.weight"] = up

            elif (op_name == "linear_fc1" or op_name == "linear_fc2") and weight_or_bias == "weight":
                # Converts Mixtral linear_fc1 and linear_fc2 weights.

                base_name = f"{layer_name}.block_sparse_moe.experts.{moe_layer_idx}."
                if op_name == "linear_fc1":
                    split_size = params.size(0) // 2
                    w1, w3 = torch.split(params, split_size, dim=0)
                    output_state_dict[base_name + "w1.weight"] = w1.clone().to(dtype)
                    output_state_dict[base_name + "w3.weight"] = w3.clone().to(dtype)
                else: # op_name == "linear_fc2"
                    output_state_dict[base_name + "w2.weight"] = params

            # Map the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "weight"] = params

            # Copy the bias.
            elif weight_or_bias == "bias":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "bias"] = params

            if source_model_type.startswith('llama') or (source_model_type == 'mixtral' and "self_attention." in op_name): 
                rotary_base = float(rotary_base)
                inv_freq = 1.0 / (
                    rotary_base
                    ** (torch.arange(0, hidden_size_per_head, 2).float() / hidden_size_per_head)
                )
                output_state_dict[layer_name + ".self_attn.rotary_emb.inv_freq"] = inv_freq

        except Exception as e:
            print(str(e))
            print(e)
    
    return layer_idx


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert Megatron-LM to HuggingFace Transformers checkpoint.
    Handles checkpoint with varying tensor parallel and pipeline parallel sizes.
    It saves the converted checkpoint into shard splits
    using HuggingFace Transformers checkpoint sharding functionality.
    """

    # Traverse the directory and Load Megatron-LM checkpoint
    sub_dirs = os.listdir(args.load_path)
    release = False
    if args.ckpt_dir_name:
        ckpt_path = os.path.join(args.load_path, args.ckpt_dir_name)
        if not os.path.exists(ckpt_path):
            raise NotADirectoryError(f"Checkpoint directory {args.ckpt_dir_name} not found!")
        latest_ckpt = args.ckpt_dir_name
    elif "latest_checkpointed_iteration.txt" in sub_dirs:
        with open(os.path.join(args.load_path, "latest_checkpointed_iteration.txt")) as f:
            latest_ckpt = f.readline().strip()
            print(f"latest checkpoint: {latest_ckpt}")
            if isinstance(latest_ckpt, bytearray):
                latest_ckpt = latest_ckpt.decode("utf-8")
            try:
                _ = int(latest_ckpt)
            except ValueError:
                release = latest_ckpt == "release"
                if not release:
                    raise ValueError(f"Invalid latest checkpoint: {latest_ckpt}")
            for sub_dir in sub_dirs:
                if latest_ckpt in sub_dir:
                    latest_ckpt = sub_dir
                    break
    else:
        raise ValueError("Cannot find latest ckpt!")

    os.makedirs(args.save_path, exist_ok=True)
    possible_state_paths = [
        os.path.join(args.load_path, latest_ckpt),
        os.path.join(
            args.load_path,
            latest_ckpt,
            "release",
        ),
    ]
    state_path = None
    for p in possible_state_paths:
        if os.path.exists(p):
            state_path = p
            print(f"Loading Megatron-LM checkpoint arguments from: {state_path}")
            break
    assert state_path is not None, f"Cannot find state path in {possible_state_paths}"
    possible_sub_dirs = [
        "mp_rank_00",
        "mp_rank_00_000",
        "mp_rank_00_000_000",
    ]
    state_dirs = os.listdir(state_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            rank0_ckpt_path = os.path.join(state_path, sub_dir, "model_optim_rng.pt")
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_ckpt_path}")
    state_dict = torch.load(rank0_ckpt_path, map_location=device)
    megatron_args = state_dict.get("args", None)
    del state_dict
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to recipe,"
            " the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers config from Megatron-LM arguments
    assert megatron_args.vocab_size is not None, "vocab_size must not be None"
    vocab_size = megatron_args.vocab_size

    # params dtype
    dtype_mapping = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_mapping.get(args.target_params_dtype, torch.bfloat16)

    # Load tokenizer metadata from checkpoint and save in HF supported format.

    tokenizer = save_tokenizer(
        args.save_path,
        megatron_args.tokenizer_model,
        source_model_type=args.source_model_type,
        special_tokens=DEFAULT_LLAMA_SPECIAL_TOKENS.get(str(args.source_model_type), []),
    )

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    assert vocab_size == megatron_args.vocab_size, "Vocab size mismatch"
    print("Saved Tokenizer")

    if args.source_model_type == "mixtral":
        max_position_embeddings = megatron_args.max_position_embeddings
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        router_jitter_noise=0.0 if megatron_args.moe_input_jitter_eps is None else megatron_args.moe_input_jitter_eps

        hidden_act = "silu" if megatron_args.swiglu else "gelu"
    else:
        # Sourced from
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L207
        if float(megatron_args.rotary_base) > 10000.0 and args.source_model_type == 'llama2':
            max_position_embeddings = 16384
        else:
            max_position_embeddings = CONTEXT_LENGTH_FOR_VERSION[args.source_model_type]

        if args.source_model_type in ["llama3", "llama3.1"]:
            bos_token_id = 128000

            if args.instruct:
                eos_token_id = [128001, 128008, 128009]
            else:
                eos_token_id = 128001
        else:
            bos_token_id = 1
            eos_token_id = 2

        rope_scaling = None
        if args.source_model_type in ["llama3", "llama3.1"]:
            rope_scaling = {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": megatron_args.max_position_embeddings,
            }

    if args.source_model_type == 'mixtral':
        # Mixtral Config
        config = MixtralConfig(
            vocab_size=megatron_args.vocab_size,
            hidden_size=megatron_args.hidden_size,
            intermediate_size=megatron_args.ffn_hidden_size,
            num_hidden_layers=megatron_args.num_layers,
            num_attention_heads=megatron_args.num_attention_heads,
            num_key_value_heads=megatron_args.num_query_groups,
            hidden_act=hidden_act,
            max_position_embeddings=megatron_args.max_position_embeddings,
            rms_norm_eps=megatron_args.norm_epsilon,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=(not megatron_args.untie_embeddings_and_output_weights),
            rope_theta=float(megatron_args.rotary_base),
            attention_dropout=megatron_args.attention_dropout,
            num_experts_per_tok=megatron_args.moe_router_topk,
            num_local_experts=megatron_args.num_experts,
            router_aux_loss_coef=megatron_args.moe_aux_loss_coeff,
            router_jitter_noise=router_jitter_noise,
            architectures=["MixtralForCausalLM"],
            torch_dtype=dtype,
        )
    else:
        # Llama Config
        config = LlamaConfig(
            vocab_size=megatron_args.vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=megatron_args.hidden_size,
            num_hidden_layers=megatron_args.num_layers,
            num_attention_heads=megatron_args.num_attention_heads,
            intermediate_size=megatron_args.ffn_hidden_size,
            num_key_value_heads=megatron_args.num_query_groups,
            rms_norm_eps=megatron_args.norm_epsilon,
            rope_theta=float(megatron_args.rotary_base),
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            architectures=["LlamaForCausalLM"],
            torch_dtype=dtype,
            rope_scaling=rope_scaling,
        )

    # Save config to file.
    config.save_pretrained(args.save_path)
    print("Saved config")

    if args.instruct and args.source_model_type.startswith('llama'):
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        generation_config.save_pretrained(args.save_path)
        print("Saved Generation config")

    # `source_megatron_args.json`` file, comprises of all global states from source
    # checkpoint required to load during the HF -> MLM flow.
    # This is done to maintain target MLM checkpoint state consistent with the source MLM checkpoint.
    with open(os.path.join(args.save_path, "source_megatron_args.json"), "w") as f:
        json.dump(vars(megatron_args), f, default=convert_dtype_to_str, indent=2)

    output_state_dict = {}
    capacity_bins_state_dict = {}

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    if args.source_model_type == 'mixtral':
        ep_size = megatron_args.expert_model_parallel_size
        ep_rank = 0
        moe_tp = megatron_args.moe_extended_tp
    else:
        ep_size = None
        ep_rank = None
        moe_tp = False

    # The regex to extract layer names.
    # example: "model.layers.0.self_attention.linear_proj.weight"
    layer_re = re.compile(r"([a-z0-9_.]+)\.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # example: "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight"
    moe_op_name_re = re.compile(r"mlp\.experts\.local_experts\.(\d+)\.([a-z0-9_.]+)")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, 0, ep_size, ep_rank)

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            tp_state_dicts[tp_rank]["model"]["embedding.word_embeddings.weight"]
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )

    output_state_dict["model.embed_tokens.weight"] = word_embeddings[:vocab_size].clone().to(dtype)

    # Transformer Layers
    print("Converting transformer layers")

    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers_per_pp_stage = config.num_hidden_layers // pp_size
    num_layers_per_ep_stage = config.num_local_experts // ep_size if \
                                args.source_model_type == 'mixtral' else None
    hidden_size = megatron_args.hidden_size
    num_heads = megatron_args.num_attention_heads
    num_query_groups = megatron_args.num_query_groups

    if num_query_groups is None:
        num_query_groups = num_heads
    heads_per_group = num_heads // num_query_groups
    qkv_total_dim = num_heads + 2 * num_query_groups

    # Initialize layer_idx to avoid UnboundLocalError
    layer_idx = -1

    for pp_rank in range(pp_size):
        print(f"Converting pipeline parallel rank {pp_rank}")
        tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, pp_rank, ep_size, ep_rank)
        moe_tp_state_dicts = []

        if moe_tp:
            for ep_rank in range(1, ep_size):
                moe_tp_state_dicts.append(get_megatron_sharded_states(state_path, tp_size, pp_size, pp_rank, ep_size, ep_rank))
            ep_rank = 0

        layer_idx = convert_layers(
            args.source_model_type, tp_state_dicts, moe_tp_state_dicts,
            output_state_dict, capacity_bins_state_dict, layer_re,
            moe_op_name_re, tp_size, pp_rank, ep_rank, moe_tp,
            num_layers_per_pp_stage, num_layers_per_ep_stage, qkv_total_dim,
            hidden_size, heads_per_group, hidden_size_per_head, num_query_groups,
            float(megatron_args.rotary_base), dtype
        )

    if not moe_tp and args.source_model_type == "mixtral":
        for ep_rank in range(1, ep_size):
            for pp_rank in range(pp_size):
                print(f"Converting expert parallel rank {ep_rank} on pp_rank {pp_rank}")
                tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, pp_rank, ep_size, ep_rank)
                moe_tp_state_dicts = []

                layer_idx = convert_layers(
                    args.source_model_type, tp_state_dicts, moe_tp_state_dicts,
                    output_state_dict, capacity_bins_state_dict, layer_re,
                    moe_op_name_re, tp_size, pp_rank, ep_rank, False,
                    num_layers_per_pp_stage, num_layers_per_ep_stage, qkv_total_dim,
                    hidden_size, heads_per_group, hidden_size_per_head,
                    num_query_groups, float(megatron_args.rotary_base), dtype
                )


    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final norm.
    print("Converting final norm")
    output_state_dict["model.norm.weight"] = tp_state_dicts[0]["model"][
        "decoder.final_layernorm.weight"
    ].to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    output_state_dict["lm_head.weight"] = (
        torch.cat(
            [tp_state_dicts[i]["model"]["output_layer.weight"] for i in range(len(tp_state_dicts))],
            dim=0,
        )[:vocab_size]
        .clone()
        .to(dtype)
    )

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Release Memory
    del tp_state_dicts
    del word_embeddings
    gc.collect()

    # Store the state_dict to file.
    max_shard_size = (
        int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    )

    save_torch_state_dict(
        state_dict=output_state_dict, save_directory=args.save_path, max_shard_size=max_shard_size
    )

    if len(capacity_bins_state_dict) > 0:
        # Save capacity bins parameters.    
        torch.save(capacity_bins_state_dict, os.path.join(args.save_path, "capacity_bins.pt"))

    print(f"Converted checkpoint saved in {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpoint_conversion_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()

    # Add megatron_path to sys path.
    megatron_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
    sys.path.insert(0, megatron_path)

    convert_checkpoint_from_megatron_to_transformers(args)


if __name__ == "__main__":
    main()
