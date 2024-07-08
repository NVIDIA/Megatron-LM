"""
Convert megatron checkpoints to huggingface weights.

This script will also convert the tokenizer configured.
Set the `--input_dir` to the megatron checkpoint root (i.e. where the
`latest_checkpointed_iteration.txt` file is located) and  `--output_dir` to
the directory where the huggingface weights should be stored.
"""

# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import gc
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).parent.parent.absolute()))  # megatron is importable

import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def convert_wqkv(
    llama_mega, layer_idx=0, n_heads=32, n_heads_kv=8, head_size: int = None
):
    qkv_w = llama_mega["transformer"][
        f"layers.{layer_idx}.attention.query_key_value.weight"
    ]
    n_hidden = qkv_w.size(1)
    if head_size is None:
        hidden_dim = n_hidden // n_heads
    else:
        hidden_dim = head_size

    n_qs_per_kv = n_heads // n_heads_kv
    n_groups = qkv_w.size(0) // hidden_dim // (n_qs_per_kv + 2)
    qkv_w = list(torch.split(qkv_w, hidden_dim, dim=0))

    wq, wk, wv = [], [], []
    for group in range(n_groups):
        for qs in range(n_qs_per_kv):
            wq.append(qkv_w[0])
            del qkv_w[0]
        wk.append(qkv_w[0])
        del qkv_w[0]
        wv.append(qkv_w[0])
        del qkv_w[0]
    print('\n\n\n\n')
    assert len(qkv_w) == 0

    wq = torch.concat(wq, dim=0)
    wk = torch.concat(wk, dim=0)
    wv = torch.concat(wv, dim=0)
    return wq, wk, wv


def convert_ffn(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega["transformer"][f"layers.{layer_idx}.mlp.dense_h_to_4h.weight"]
    ffn_w1, ffn_w3 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def write_llama_model(
    model_path,
    input_base_path,
    num_output_shards: int = 2,
    norm_eps: float = 1e-05,
    rope_theta: float = 1e4,
    iteration: str = "0"
):
    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    os.makedirs(model_path, exist_ok=True)

    if iteration == 0:
        with open(os.path.join(input_base_path, "latest_checkpointed_iteration.txt")) as f:
            iteration = f.read()
    if iteration != "release":
        iteration = f"iter_{int(iteration):07d}"

    # Load weights
    base_path = Path(input_base_path) / iteration
    assert (
        len(list(base_path.glob("mp_rank_*"))) == 1
    ), "Unshard your model with checkpoint_util.py first!"
    loaded = torch.load(
        base_path / "mp_rank_00" / "model_optim_rng.pt", map_location="cpu"
    )
    args = loaded["args"]

    loaded = loaded["model"]["language_model"]
    if "transformer" not in loaded:  # normalize key names
        loaded["transformer"] = loaded.pop("encoder")
        for key in list(loaded["transformer"].keys()):
            loaded["transformer"][key.replace("self_attention", "attention")] = loaded[
                "transformer"
            ].pop(key)
        loaded["embedding"]["word_embeddings.weight"] = loaded["embedding"].pop(
            "word_embeddings"
        )["weight"]
        args.num_layers = args.encoder_num_layers

    # Load arguments
    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    n_heads_kv = getattr(args, "num_query_groups", n_heads)
    n_dense = args.ffn_hidden_size
    n_hidden = args.hidden_size
    hidden_per_head = n_hidden // n_heads
    intermediate_size = args.ffn_hidden_size
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head)
    )
    print("Llama-Megatron Loaded!")
    print(loaded.keys())
    print(loaded['transformer'].keys())
    print(loaded['output_layer'].keys())
    print(loaded['embedding'].keys())
    param_count = 0
    index_dict = {"weight_map": {}}
    # Start conversion

    with TemporaryDirectory() as tmp_model_path:
        print(f"Weighted Converting for {n_layers} layers...")
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            wq_proj, wk_proj, wv_proj = convert_wqkv(
                llama_mega=loaded,
                layer_idx=layer_i,
                n_heads=n_heads,
                n_heads_kv=n_heads_kv,
            )
            ffn_w1, ffn_w3 = convert_ffn(
                llama_mega=loaded, layer_idx=layer_i, n_dense=n_dense
            )
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": wq_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": wk_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                    "transformer"
                ][f"layers.{layer_i}.attention.dense.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded["transformer"][
                    f"layers.{layer_i}.mlp.dense_4h_to_h.weight"
                ],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded["transformer"][
                    f"layers.{layer_i}.input_norm.weight"
                ],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                    "transformer"
                ][f"layers.{layer_i}.post_attention_norm.weight"],
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq,
            }

            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f"Sharded file saved to {filename}")

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            "model.norm.weight": loaded["transformer"]["final_norm.weight"],
            "lm_head.weight": loaded["output_layer"]["weight"],
            "model.embed_tokens.weight": loaded["embedding"]["word_embeddings.weight"],
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch_dtype = state_dict["lm_head.weight"].dtype
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f"Sharded file saved to {filename}")

        print("total_size", param_count)
        # Write configs and save
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(
            index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json")
        )
        config = LlamaConfig(
            vocab_size=args.padded_vocab_size,
            hidden_size=n_hidden,
            intermediate_size=intermediate_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=norm_eps,
            num_key_value_heads=n_heads_kv,
            max_position_embeddings=args.seq_length,
        )
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        print("Loading the checkpoint in a Llama model...")
        model = LlamaForCausalLM.from_pretrained(
            tmp_model_path, torch_dtype=torch_dtype
        )
        # Avoid saving this as part of the config.
        del model.config._name_or_path

    print("Saving in the Transformers format.")
    max_num_params_per_shard = param_count * 2 // max(1, (num_output_shards - 1))
    model.save_pretrained(
        model_path, max_shard_size=max_num_params_per_shard, safe_serialization=False
    )

def write_tokenizer(args: Namespace):
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    hf_tokenizer.save_pretrained(args.output_dir)


def main():
    # make sure megatron is importable

    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", help="Location of Megatron weights", required=True
    )
    parser.add_argument("--num_output_shards", type=int, default=1)
    parser.add_argument(
        "--output_dir", help="Location to write HF model and tokenizer", required=True
    )
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer")
    parser.add_argument("--iteration", type=str, default="0")
    

    args = parser.parse_args()
    eps =  1e-5
    rope_theta = 1e4
    write_llama_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_output_shards=args.num_output_shards,
        norm_eps=eps,
        rope_theta=rope_theta,
        iteration=args.iteration
    )

    write_tokenizer(args)


if __name__ == "__main__":
    main()