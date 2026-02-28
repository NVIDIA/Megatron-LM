# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Utility helpers for mimo models.
"""
import re
import os
import torch
from megatron.core import dist_checkpointing


def load_submodule_ckpt(module: torch.nn.Module, ckpt_dir: str):
    """Load *ckpt_dir* into *module* using Megatron distributed-checkpointing."""

    # 1) Ask for tensors using a `module.` prefix so they match checkpoint keys.
    sharded_sd_with_prefix = module.sharded_state_dict(prefix="module.")

    # Remove fp8 extra_state tensors – they may not exist in older checkpoints.
    for k in list(sharded_sd_with_prefix.keys()):
        if "extra_state" in k:
            del sharded_sd_with_prefix[k]

    # 2) Wrap it under a root key just as in user snippet; this becomes the state
    #    dict returned by `load` so we can easily strip the prefix afterwards.
    wrapper_sd = dict(state_dict=sharded_sd_with_prefix)
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
    )
    # 3) Remove the prefix and push into the module.
    cleaned = {k.removeprefix("module."): v for k, v in loaded["state_dict"].items()}

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load_state_dict had unexpected mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )


def load_submodule_ckpt_for_mot(module: torch.nn.Module, ckpt_dir: str, init_gen_from_und: bool = True):
    """Load checkpoint into MoT model, handling missing _gen parameters.

    This function supports two checkpoint formats:
    1. Standard Megatron Core format: decoder.layers.N.xxx (with layer indices)
    2. ModelOpt/PAI fused-layer format: decoder.layers.xxx with shard_N_M (layers merged)

    Args:
        module: The model module to load checkpoint into
        ckpt_dir: Path to checkpoint directory
        init_gen_from_und: If True, initialize _gen parameters from corresponding non-_gen params
    """
    # Check if this is a ModelOpt/PAI format checkpoint
    metadata_path = os.path.join(ckpt_dir, "modelopt_run_config.yaml")
    is_modelopt_format = os.path.exists(metadata_path)

    if is_modelopt_format:
        print("[load_submodule_ckpt_for_mot] Detected ModelOpt/PAI checkpoint format (fused layers)")
        _load_fused_layer_checkpoint_pytorch(module, ckpt_dir, init_gen_from_und)
    else:
        print("[load_submodule_ckpt_for_mot] Using standard Megatron Core checkpoint format")
        _load_standard_checkpoint(module, ckpt_dir, init_gen_from_und)


def _load_fused_layer_checkpoint_pytorch(module: torch.nn.Module, ckpt_dir: str, init_gen_from_und: bool = True):
    """Load ModelOpt/PAI format checkpoint using PyTorch distributed checkpoint.

    This checkpoint format stores all layers merged into single tensors.
    We load them and then split them back into per-layer tensors.
    """
    import torch.distributed.checkpoint as dcp
    import pickle
    import glob

    # Load checkpoint metadata to understand structure
    with open(os.path.join(ckpt_dir, ".metadata"), 'rb') as f:
        ckpt_metadata = pickle.load(f)

    ckpt_keys = set(ckpt_metadata.state_dict_metadata.keys())
    ckpt_keys_no_extra = {k for k in ckpt_keys if "extra_state" not in k}
    print(f"[_load_fused_layer_checkpoint_pytorch] Checkpoint has {len(ckpt_keys_no_extra)} weight keys")

    # Get model structure
    model_sd = module.state_dict()

    # Count layers in model
    layer_indices = set()
    for k in model_sd.keys():
        match = re.match(r'decoder\.layers\.(\d+)\.', k)
        if match:
            layer_indices.add(int(match.group(1)))
    num_model_layers = len(layer_indices) if layer_indices else 24

    # Check checkpoint layer count from config
    try:
        import yaml
        config_path = os.path.join(ckpt_dir, "modelopt_run_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_ckpt_layers = config.get('num_layers', 24)
    except:
        num_ckpt_layers = 24

    print(f"[_load_fused_layer_checkpoint_pytorch] Model has {num_model_layers} layers, checkpoint has {num_ckpt_layers} layers")

    pt_files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if False: #pt_files:
        print(f"[_load_fused_layer_checkpoint_pytorch] Found {len(pt_files)} .pt files, loading directly...")
        load_state_dict = {}
        for pt_file in pt_files:
            shard = torch.load(pt_file, map_location='cpu')
            load_state_dict.update(shard)
        print("[_load_fused_layer_checkpoint_pytorch] Checkpoint loaded successfully")
    else:
        # Create empty tensors matching checkpoint structure for loading
        load_state_dict = {}
        for ckpt_key in ckpt_keys_no_extra:
            meta = ckpt_metadata.state_dict_metadata[ckpt_key]
            if hasattr(meta, 'size'):
                load_state_dict[ckpt_key] = torch.zeros(meta.size, dtype=torch.bfloat16)

        # Load checkpoint using PyTorch distributed checkpoint
        print(f"[_load_fused_layer_checkpoint_pytorch] Loading {len(load_state_dict)} tensors from checkpoint...")
        dcp.load(
            state_dict=load_state_dict,
            checkpoint_id=ckpt_dir,
        )
        print("[_load_fused_layer_checkpoint_pytorch] Checkpoint loaded successfully")

    for k, v in load_state_dict.items():
        print(f"[_load_fused_layer_checkpoint_pytorch] {k}: {v.shape}, sum={v.sum().item()}")
        if 'bias' in k and v.sum().item() == 0:
            print(f"[_load_fused_layer_checkpoint_pytorch] WARNING: {k} has zero sum!")

    # Now map loaded weights to model format
    new_state_dict = {}

    for model_key in model_sd.keys():
        if "extra_state" in model_key or "_gen" in model_key:
            continue

        # Handle non-layer keys directly
        if not model_key.startswith("decoder.layers."):
            if model_key in load_state_dict:
                new_state_dict[model_key] = load_state_dict[model_key]
            continue

        # Handle layer keys: decoder.layers.N.xxx -> decoder.layers.xxx
        match = re.match(r'decoder\.layers\.(\d+)(\..*)', model_key)
        if not match:
            continue

        layer_idx = int(match.group(1))
        suffix = match.group(2)  # e.g., .self_attention.linear_qkv.weight

        # Checkpoint key without layer index
        ckpt_key = f"decoder.layers{suffix}"

        # Key mapping: model key -> checkpoint key
        # Some checkpoint formats use different naming conventions
        key_mapping = {
            "decoder.layers.pre_mlp_layernorm.weight": "decoder.layers.mlp.linear_fc1.layer_norm_weight",
            "decoder.layers.input_layernorm.weight": "decoder.layers.self_attention.linear_qkv.layer_norm_weight",
        }

        # Apply key mapping if needed
        if ckpt_key in key_mapping:
            mapped_key = key_mapping[ckpt_key]
            if mapped_key in load_state_dict:
                ckpt_key = mapped_key

        if ckpt_key not in load_state_dict:
            print(f"[_load_fused_layer_checkpoint_pytorch] WARNING: {ckpt_key} not in checkpoint")
            continue

        # Get the fused tensor from checkpoint
        fused_tensor = load_state_dict[ckpt_key]
        model_tensor = model_sd[model_key]

        # Calculate slice for this layer
        # Handle different fused tensor formats:

        # Case 1: 3D tensor with layer as first dimension [num_layers, ...]
        # e.g., ckpt [24, 1152, 896] -> model [1152, 896]
        if (len(fused_tensor.shape) == len(model_tensor.shape) + 1 and
            fused_tensor.shape[0] == num_ckpt_layers and
            fused_tensor.shape[1:] == model_tensor.shape):
            new_state_dict[model_key] = fused_tensor[layer_idx].clone()

        # Case 2: 2D tensor with layers concatenated along dim 0
        # e.g., ckpt [24*1152, 896] -> model [1152, 896]
        elif fused_tensor.shape[0] == num_ckpt_layers * model_tensor.shape[0]:
            layer_size = model_tensor.shape[0]
            start_idx = layer_idx * layer_size
            end_idx = start_idx + layer_size
            new_state_dict[model_key] = fused_tensor[start_idx:end_idx].clone()

        # Case 3: Same shape, just copy (non-layer-specific weight like embedding)
        elif fused_tensor.shape == model_tensor.shape:
            new_state_dict[model_key] = fused_tensor.clone()

        # Case 4: 1D tensor with layers concatenated
        # e.g., ckpt [24*896] -> model [896] (for biases, layernorm weights)
        elif len(fused_tensor.shape) == 1 and fused_tensor.shape[0] == num_ckpt_layers * model_tensor.shape[0]:
            layer_size = model_tensor.shape[0]
            start_idx = layer_idx * layer_size
            end_idx = start_idx + layer_size
            new_state_dict[model_key] = fused_tensor[start_idx:end_idx].clone()

        # Case 5: 2D tensor [num_layers, dim] for 1D model tensor [dim]
        # e.g., ckpt [24, 896] -> model [896] (for layernorm weights)
        elif (len(fused_tensor.shape) == 2 and
              len(model_tensor.shape) == 1 and
              fused_tensor.shape[0] == num_ckpt_layers and
              fused_tensor.shape[1] == model_tensor.shape[0]):
            new_state_dict[model_key] = fused_tensor[layer_idx].clone()
            print(f"[_load_fused_layer_checkpoint_pytorch] model_key {model_key}: {model_sd[model_key].shape}")
        else:
            print(f"[_load_fused_layer_checkpoint_pytorch] Shape mismatch for {model_key}: "
                  f"model {model_tensor.shape} vs ckpt {fused_tensor.shape}")

    print(f"[_load_fused_layer_checkpoint_pytorch] Mapped {len(new_state_dict)} weights")

    # Load into model
    incompatible = module.load_state_dict(new_state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k and "_gen" not in k]
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]

    print(f"[_load_fused_layer_checkpoint_pytorch] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Sample missing keys: {missing}")

    # Initialize _gen parameters
    if init_gen_from_und:
        _init_gen_from_und(module)


def _load_standard_checkpoint(module: torch.nn.Module, ckpt_dir: str, init_gen_from_und: bool = True):
    """Load standard Megatron Core format checkpoint."""
    # 1) Ask for tensors using a `module.` prefix so they match checkpoint keys.
    sharded_sd_with_prefix = module.sharded_state_dict(prefix="module.")

    # Remove fp8 extra_state tensors – they may not exist in older checkpoints.
    for k in list(sharded_sd_with_prefix.keys()):
        if "extra_state" in k:
            del sharded_sd_with_prefix[k]

    # Create a version without _gen keys for loading from non-MoT checkpoint
    sharded_sd_no_gen = {k: v for k, v in sharded_sd_with_prefix.items() if "_gen" not in k}

    # 2) Wrap and load
    wrapper_sd = dict(state_dict=sharded_sd_no_gen)
    try:
        loaded = dist_checkpointing.load(
            sharded_state_dict=wrapper_sd,
            checkpoint_dir=ckpt_dir,
        )
    except (KeyError, RuntimeError) as e:
        print(f"[_load_standard_checkpoint] ERROR: Could not load checkpoint: {e}")
        raise

    # 3) Remove the prefix
    cleaned = {k.removeprefix("module."): v for k, v in loaded["state_dict"].items()}

    # 4) Load with strict=False to allow missing _gen keys
    incompatible = module.load_state_dict(cleaned, strict=False)

    # Filter out _gen keys from missing keys (these are expected to be missing)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing_non_gen = [k for k in incompatible.missing_keys
                       if "extra_state" not in k and "_gen" not in k]

    if unexpected or missing_non_gen:
        print(f"[_load_standard_checkpoint] WARNING: load_state_dict had mismatch.")
        print(f"  Missing (non-gen): {len(missing_non_gen)} keys")
        print(f"  Unexpected: {len(unexpected)} keys")

    # 5) Initialize _gen parameters from corresponding non-_gen parameters
    if init_gen_from_und:
        _init_gen_from_und(module)


def _init_gen_from_und(module: torch.nn.Module):
    """Initialize _gen parameters from corresponding non-_gen parameters."""
    state_dict = module.state_dict()
    gen_keys = [k for k in state_dict.keys() if "_gen" in k and "extra_state" not in k]

    print(f"[_init_gen_from_und] Initializing {len(gen_keys)} _gen parameters from corresponding und parameters")

    copied_count = 0
    for gen_key in gen_keys:
        # Convert _gen key to corresponding non-_gen key
        und_key = re.sub(r'_gen(?=\.|$)', '', gen_key)

        if und_key in state_dict:
            with torch.no_grad():
                gen_param = _get_nested_attr(module, gen_key)
                und_param = _get_nested_attr(module, und_key)
                if gen_param is not None and und_param is not None:
                    gen_param.data.copy_(und_param.data)
                    copied_count += 1

    print(f"[_init_gen_from_und] Successfully copied {copied_count} parameters")


def _get_nested_attr(module: torch.nn.Module, key: str):
    """Get a nested attribute from a module using dot notation."""
    parts = key.split('.')
    current = module
    try:
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current
    except (AttributeError, IndexError, KeyError):
        return None
