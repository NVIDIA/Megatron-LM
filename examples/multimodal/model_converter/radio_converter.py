# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch

def convert_radio_h(output_path, tensor_parallel_size, use_te, version):
    device = "cuda"

    version = version if version is not None else 'radio_v2.5-h'
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=version, progress=True)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 80 
    hidden_dim = 1280 
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():
        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "summary_idxs" in name:
            continue
        elif "patch_generator" in name:
            if "embedder" in name:
                new_name = "embedder.weight"
                chunk_dim = 0
            elif "cls_token" in name:
                new_name = "class_token"
            elif "pos_embed" in name:
                new_name = "position_embeddings"
        elif "input_conditioner" in name:
            continue
        elif "blocks" in name:
            layer_idx = name.split(".")[2]
            base = f"decoder.layers.{layer_idx}"

            if "attn.qkv.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.qkv.bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "attn.proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "norm1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "norm1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif "mlp.fc1.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.fc1.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.fc2.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.fc2.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "norm2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif "norm2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

        assert new_name != "", f"unexpected layer name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)
    with open(os.path.join(output_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1") 

def convert_radio_g(output_path, tensor_parallel_size, use_te, version):
    device = "cuda"

    version = version if version is not None else 'radio_v2.5-g'
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=version, progress=True)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 64
    hidden_dim = 1536
    num_heads = 24
    ffn_hidden_dim = 4096
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    mlp_indices = []
    step = ffn_hidden_dim // tensor_parallel_size
    for i in range(tensor_parallel_size):
        mlp_indices.append(torch.arange(i * step, (i + 1) * step, dtype=torch.int))
        mlp_indices.append(torch.arange(ffn_hidden_dim + i * step, ffn_hidden_dim + (i + 1) * step, dtype=torch.int))

    mlp_indices = torch.cat(mlp_indices)

    for name, tensor in state_dict.items():
        # Map parameter names to ones used in megatron.
        new_names = []
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)
        new_tensors = [new_tensor]

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "model" not in name:
            continue;
        elif "patch_generator" in name:
            if "embedder.weight" in name:
                new_names.append("embedder.weight")
                chunk_dim = 0
            elif "embedder.bias" in name:
                new_names.append("embedder.bias")
                chunk_dim = 0
            elif "cls_token" in name:
                new_names.append("class_token")
            elif "pos_embed" in name:
                new_names.append("position_embeddings")
        elif "input_conditioner" in name:
            continue;
        elif "mask_token" in name:
            new_names.append("mask_token")
        elif "inner.norm" in name:
            if "norm.weight" in name:
                new_names.append("ln_post.weight")
            elif "norm.bias" in name:
                new_names.append("ln_post.bias")
        elif "blocks" in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"

            if "attn.qkv.weight" in name:
                new_names.append(f"{base}.self_attention.linear_qkv.weight")
                new_tensors[0] = new_tensors[0][indices]
                chunk_dim = 0
            elif "attn.qkv.bias" in name:
                new_names.append(f"{base}.self_attention.linear_qkv.bias")
                new_tensors[0] = new_tensors[0][indices]
                chunk_dim = 0
            elif "attn.proj.weight" in name:
                new_names.append(f"{base}.self_attention.linear_proj.weight")
                chunk_dim = 1
            elif "attn.proj.bias" in name:
                new_names.append(f"{base}.self_attention.linear_proj.bias")
            elif "norm1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
                new_names.append(new_name)
            elif "norm1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
                new_names.append(new_name)
            elif "mlp.w12.weight" in name: 
                new_names.append(f"{base}.mlp.linear_fc1.weight")
                new_tensors[0] = new_tensors[0][mlp_indices]
                chunk_dim = 0
            elif "mlp.w12.bias" in name: 
                new_names.append(f"{base}.mlp.linear_fc1.bias")
                new_tensors[0] = new_tensors[0][mlp_indices]
                chunk_dim = 0
            elif "mlp.w3.weight" in name: 
                new_names.append(f"{base}.mlp.linear_fc2.weight")
                chunk_dim = 1
            elif "mlp.w3.bias" in name: 
                new_names.append(f"{base}.mlp.linear_fc2.bias")
            elif "norm2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
                new_names.append(new_name)
            elif "norm2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"
                new_names.append(new_name)
            elif "ls1.grandma" in name:
                new_names.append(f"{base}.ls1")
            elif "ls2.grandma" in name:
                new_names.append(f"{base}.ls2")

        assert len(new_names) == len(new_tensors), f"{new_names} {new_tensors}"

        for new_name, new_tensor in zip(new_names, new_tensors):
            if chunk_dim is None:
                tp_new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
            else:
                tp_new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

            for i in range(tensor_parallel_size):
                # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
                new_state_dicts[i]["model"][new_name] = tp_new_tensors[i].clone()

                # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
                extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
                is_extra_state_layer = any([l in new_name for l in extra_state_layers])
                if use_te and is_extra_state_layer:
                    layer = new_name.split(".")[-2]
                    if layer in extra_state_layers:
                        extra_state_name = (
                            new_name[: new_name.rfind(".") + 1] + "_extra_state"
                        )  # Replace the weight name.
                        new_state_dicts[i]["model"][extra_state_name] = None

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)
        with open(os.path.join(output_path, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write("1") 


def convert(output_path, tensor_parallel_size, use_te, model_type, version):
    if model_type == "radio_v2.5-h":
        convert_radio_h(output_path, tensor_parallel_size, use_te, version)
    elif model_type == "radio_v2.5-g":
        convert_radio_g(output_path, tensor_parallel_size, use_te, version)
    else:
        raise NotImplementedError(f"Converter doesn't support model type {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert RADIO weights to megatron format.


Example usage:
python radio_converter.py --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")
    parser.add_argument("--model-type", required=True, type=str, choices=['radio_v2.5-h', 'radio_v2.5-g'], help="Type of radio to load for conversion")
    parser.add_argument("--version", type=str, default=None, help="Version to pass to torch.hub.load. Can be a local path or a version RADIO on torch hub. By default use the version from the model type.")

    args = parser.parse_args()

    convert(args.output, args.tensor_parallel_size, args.use_te, args.model_type, args.version)

    print("done.")
