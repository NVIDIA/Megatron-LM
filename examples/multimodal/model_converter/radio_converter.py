# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch


def print_state_dict_keys(state_dict, name_filter=None):
    """Print all keys and shapes from a state dict."""
    print(f"\n{'Key':<80} {'Shape':>25} {'Dtype':>15}")
    print("=" * 120)
    for name, tensor in sorted(state_dict.items()):
        if name_filter and name_filter not in name:
            continue
        if hasattr(tensor, 'shape'):
            shape_str = str(tuple(tensor.shape))
            dtype_str = str(tensor.dtype).replace('torch.', '')
            print(f"{name:<80} {shape_str:>25} {dtype_str:>15}")
        else:
            print(f"{name:<80} {'(non-tensor)':>25} {str(type(tensor).__name__):>15}")
    print()


def generate_qkv_indices(num_heads, hidden_dim):
    """
    Generate indices to convert QKV attention weights from PyTorch format to Megatron format.

    PyTorch format: [Q_all, K_all, V_all] - all queries, then all keys, then all values
    Megatron format: [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]

    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension size

    Returns:
        torch.Tensor: Indices for reordering the QKV tensor
    """
    kv_channels = hidden_dim // num_heads
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    return torch.cat(indices)


def get_block_and_local_layer(global_layer_idx: int, downscaling_levels: list) -> tuple:
    """
    Convert a global layer index to (block_idx, local_layer_idx) based on downscaling levels.

    Args:
        global_layer_idx: The global layer index (0-based)
        downscaling_levels: List of layer indices where downscaling happens

    Returns:
        (block_idx, local_layer_idx): Block index and layer index within that block

    Example:
        With downscaling_levels=[8, 16] and 32 total layers:
        - global_layer_idx=5 -> (0, 5)   # Block 0, layer 5
        - global_layer_idx=10 -> (1, 2)  # Block 1, layer 2 (10 - 8 = 2)
        - global_layer_idx=20 -> (2, 4)  # Block 2, layer 4 (20 - 16 = 4)
    """
    block_idx = 0
    local_layer_idx = global_layer_idx

    for level in downscaling_levels:
        if global_layer_idx >= level:
            block_idx += 1
            local_layer_idx = global_layer_idx - level
        else:
            break

    return block_idx, local_layer_idx


def convert_radio(output_path, tensor_parallel_size, use_te, version, torchhub_version, torchhub_source, model_type, downscaling_levels=None):
    hidden_dim = HIDDEN_DIM_MAP[model_type]
    num_heads = NUM_HEADS_MAP[model_type]

    model = torch.hub.load(torchhub_version, 'radio_model', version=version, source=torchhub_source, progress=True)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Base indices for initial hidden_dim (before any downscaling)
    # We'll compute indices dynamically for each QKV layer based on actual tensor shape
    base_num_heads = num_heads
    base_hidden_dim = hidden_dim

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
        elif "decoder" in name:
            # Ignore all decoder-related parameters (they are used in RADIO 1D during training.)
            continue
        elif "downscale_blocks" in name:
            layer_idx = name.split(".")[2]
            base = f"downscale_blocks.{layer_idx}"
            # Downscale operators used in RADIO 1D.
            if "norm.weight" in name:
                new_name = f"{base}.norm.weight"
            elif "norm.bias" in name:
                new_name = f"{base}.norm.bias"
            elif "reduction.weight" in name:
                new_name = f"{base}.reduction.weight"
                chunk_dim = 0
            elif "reduction.bias" in name:
                new_name = f"{base}.reduction.bias"
                chunk_dim = 0
        elif "prefix_proj_blocks" in name:
            layer_idx = name.split(".")[2]
            base = f"prefix_proj_blocks.{layer_idx}"
            # Projection layers for prefix tokens during downscaling in RADIO 1D.
            if "weight" in name:
                new_name = f"{base}.weight"
                chunk_dim = 0
            elif "bias" in name:
                new_name = f"{base}.bias"
                chunk_dim = 0
        elif "blocks" in name:
            global_layer_idx = int(name.split(".")[2])

            # Determine the base path based on whether downscaling is enabled
            if downscaling_levels:
                block_idx, local_layer_idx = get_block_and_local_layer(global_layer_idx, downscaling_levels)
                base = f"decoder_blocks.{block_idx}.layers.{local_layer_idx}"
            else:
                # Legacy format: single decoder block
                base = f"decoder.layers.{global_layer_idx}"

            if "attn.qkv.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                # Dynamically compute indices based on actual tensor shape
                # QKV weight shape is [3 * hidden_dim, ...], so infer hidden_dim
                actual_hidden_dim = new_tensor.shape[0] // 3
                actual_num_heads = base_num_heads * actual_hidden_dim // base_hidden_dim
                indices = generate_qkv_indices(actual_num_heads, actual_hidden_dim)
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.qkv.bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                # Dynamically compute indices based on actual tensor shape
                # QKV bias shape is [3 * hidden_dim], so infer hidden_dim
                actual_hidden_dim = new_tensor.shape[0] // 3
                actual_num_heads = base_num_heads * actual_hidden_dim // base_hidden_dim
                indices = generate_qkv_indices(actual_num_heads, actual_hidden_dim)
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

def convert_radio_g(output_path, tensor_parallel_size, use_te, version, torchhub_version, torchhub_source, model_type):
    version = version if version is not None else 'radio_v2.5-g'
    model = torch.hub.load(torchhub_version, 'radio_model', version=version, source=torchhub_source, progress=True)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    hidden_dim = 1536
    num_heads = 24
    ffn_hidden_dim = 4096
    indices = generate_qkv_indices(num_heads, hidden_dim)

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
            continue
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
            continue
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


def convert(output_path, tensor_parallel_size, use_te, model_type, version, torchhub_version, downscaling_levels=None):
    if torchhub_version.startswith("/lustre/") and not os.path.exists(torchhub_version):
        raise ValueError(f"Torchhub version is a lustre path that does not exist: {torchhub_version}")
    elif os.path.exists(torchhub_version):
        torchhub_source = "local"
    else:
        torchhub_source = "github"

    # Use provided downscaling_levels or fall back to model-specific defaults
    if downscaling_levels is None:
        downscaling_levels = DOWNSCALING_LEVELS_MAP.get(model_type, None)

    if model_type in CONVERT_FN_MAP:
        CONVERT_FN_MAP[model_type](output_path, tensor_parallel_size, use_te, version, torchhub_version, torchhub_source, model_type, downscaling_levels)
    else:
        raise NotImplementedError(f"Converter doesn't support model type {model_type}")

HIDDEN_DIM_MAP = {
    'radio_v4-h-1d': 1280,
    'radio_v4-h': 1280,
    'radio_v3-h': 1280,
    'radio_v2.5-h': 1280,
    'c-radio_v2-vlm-h': 1280,
    'radio-so400m': 1152,
}
NUM_HEADS_MAP = {
    'radio_v4-h-1d': 16,
    'radio_v4-h': 16,
    'radio_v3-h': 16,
    'radio_v2.5-h': 16,
    'c-radio_v2-vlm-h': 16,
    'radio-so400m': 16,
}
# Default downscaling levels for each model type (None means no downscaling / legacy format)
DOWNSCALING_LEVELS_MAP = {
    'radio_v4-h-1d': [24],  # Downscale at layer 24: block 0 = layers 0-23, block 1 = layers 24-31
    'radio_v4-h': None,
    'radio_v3-h': None,
    'radio_v2.5-h': None,
    'c-radio_v2-vlm-h': None,
    'radio-so400m': None,
}

CONVERT_FN_MAP = {
    'radio_v4-h-1d': convert_radio,
    'radio_v4-h': convert_radio,
    'radio_v3-h': convert_radio,
    'radio_v2.5-h': convert_radio,
    'c-radio_v2-vlm-h': convert_radio,
    'radio-so400m': convert_radio,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert RADIO weights to megatron format.


Example usage:
python radio_converter.py --output /some/output/folder --tensor-parallel-size 4

To inspect keys and shapes without converting, add `--print-keys`
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", type=str, default=None, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")
    parser.add_argument(
        "--model-type",
        required=True,
        type=str,
        choices=['radio_v2.5-h', 'radio_v2.5-g', 'c-radio_v2-vlm-h', 'radio_v4-h', 'radio_v3-h', 'radio-so400m', 'radio_v4-h-1d'],
        help="Type of radio to load for conversion"
    )
    parser.add_argument("--version", type=str, default=None, help="Version to pass to torch.hub.load. Can be a local path or a version RADIO on torch hub. By default use the version from the model type.")
    parser.add_argument("--torchhub-version", type=str, default="NVlabs/RADIO", help="TorchHub repo. Can be a local path or a Github repo. By default use NVlabs/RADIO.")
    parser.add_argument("--radio-downscaling-levels", nargs='*', type=int, default=None, help="Block indices where downscaling happens. If not set, uses model-specific defaults. Use empty list for legacy format.")
    parser.add_argument(
        "--print-keys", action="store_true",
        help="Print all tensor keys and shapes from the model, then exit without converting"
    )
    parser.add_argument(
        "--key-filter", type=str, default=None,
        help="Optional filter to only show keys containing this substring (used with --print-keys)"
    )

    args = parser.parse_args()

    # Handle --print-keys mode
    if args.print_keys:
        torchhub_version = args.torchhub_version
        if torchhub_version.startswith("/lustre/") and not os.path.exists(torchhub_version):
            raise ValueError(f"Torchhub version is a lustre path that does not exist: {torchhub_version}")
        torchhub_source = "local" if os.path.exists(torchhub_version) else "github"

        version = args.version if args.version else args.model_type
        print(f"Loading model: {version} from {torchhub_version} ({torchhub_source})")
        model = torch.hub.load(torchhub_version, 'radio_model', version=version, source=torchhub_source, progress=True)

        state_dict = model.state_dict()
        print(f"\nTotal parameters: {len(state_dict)}")
        print_state_dict_keys(state_dict, args.key_filter)
        print("Exiting after printing keys.")
        exit(0)

    # Existing conversion logic requires --output
    if args.output is None:
        parser.error("--output is required when not using --print-keys")

    assert args.radio_downscaling_levels is None or len(args.radio_downscaling_levels) > 0, \
        "If using --radio-downscaling-levels, it must be a non-empty list of block indices"

    convert(args.output, args.tensor_parallel_size, args.use_te, args.model_type, args.version, args.torchhub_version, args.radio_downscaling_levels)

    print("Finished model conversion.")
