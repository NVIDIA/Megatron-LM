# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import re

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import default_planner

logger = logging.getLogger(__name__)

try:
    from torch.distributed import DeviceMesh
    from torch.distributed._tensor import DTensor
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.tensor.placement_types import Replicate, Shard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        make_fsdp_dtensor,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        gather_uneven_dtensor_to_full_tensor,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
        get_mcore_tensor_parallel_partition_dim,
        is_mcore_tensor_model_parallel,
    )

    HAVE_MEGATRON_FSDP = True
except ImportError:
    HAVE_MEGATRON_FSDP = False

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from megatron.core.transformer.transformer_layer import TransformerLayer


def get_ep_layer_offset():
    """
    Get the expert layer offset for the current model.
    """
    from megatron.training.global_vars import get_args

    args = get_args()
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_local_experts = args.num_experts // ep_size if args.num_experts else 0
    local_expert_offset = ep_rank * num_local_experts

    return local_expert_offset


def get_total_num_experts():
    """
    Get the total number of experts for the current model.
    """
    from megatron.training.global_vars import get_args

    args = get_args()
    return args.num_experts if args.num_experts else 0


def get_expert_index_from_key(key):
    """Extract expert index from various expert key formats.

    Supported formats:
    - GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
    - SequentialMLP: 'mlp.experts.local_experts.0.linear_fc1.weight',
        'mlp.experts.local_experts.0.linear_fc2.weight'

    Returns:
        int: Expert index if found, None otherwise.
    """
    # GroupedMLP: index is at the end after 'weight'
    if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
        m = re.search(r'^.*\.mlp\.experts\.linear_fc\d\.weight(\d+)', key)
        assert m, f"Failed to parse expert index from key: {key}"
        return int(m.group(1))
    # SequentialMLP: index is between 'local_experts.' and next '.'
    elif 'mlp.experts.local_experts' in key:
        m = re.search(r'^.*\.mlp\.experts\.local_experts\.(\d+)', key)
        assert m, f"Failed to parse expert index from key: {key}"
        return int(m.group(1))
    return None


def handle_experts_in_state_dict(state_dict):
    """
    Rewrite expert keys in state dict.
    """
    local_expert_start = get_ep_layer_offset()
    local_expert_end = get_total_num_experts()

    def should_keep_expert_key(expert_index):
        """Determine if this rank should keep this expert key based on expert index"""
        if expert_index is None:
            # If we can't determine expert index, keep the key (non-expert weights)
            return True

        # Check if this expert belongs to this rank
        return local_expert_start <= expert_index < local_expert_end

    def replace_expert_index_in_key(key, expert_index, state_dict):
        """Replace expert index in key with new index corresponding to the current rank"""
        new_expert_index = expert_index + local_expert_start
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            # Handle SwiGLU weight{idx}_w and weight{idx}_v format
            if key.endswith('_w') or key.endswith('_v'):
                suffix = key[-2:]  # '_w' or '_v'
                new_key = key.replace(
                    f'weight{expert_index}{suffix}', f'weight{new_expert_index}{suffix}'
                )
            # Handle regular weight{idx} format
            else:
                new_key = key.replace(f'weight{expert_index}', f'weight{new_expert_index}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(
                f'local_experts.{expert_index}.', f'local_experts.{new_expert_index}.'
            )
        else:
            raise ValueError(f"Unexpected expert key format: {key}")

        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    # Process model state dict
    state_dict = state_dict.copy()
    for key in list(state_dict.keys()):
        expert_index = get_expert_index_from_key(key)
        if not should_keep_expert_key(expert_index):
            replace_expert_index_in_key(key, expert_index, state_dict)

    return state_dict


def expert_param_local_key(key):
    """Get the module parameter corresponding to the key."""
    local_expert_offset = get_ep_layer_offset()
    expert_index = get_expert_index_from_key(key)
    if expert_index is not None:
        new_expert_index = expert_index - local_expert_offset
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            new_key = key.replace(f'weight{expert_index}', f'weight{new_expert_index}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(
                f'local_experts.{expert_index}.', f'local_experts.{new_expert_index}.'
            )
        else:
            raise ValueError(f"Unexpected expert key format: {key}")
        key = new_key

    return key


def handle_swiglu_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """
    Handle SWiGLU in model and optimizer state dicts.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    def intersection(s1, s2):
        # Only works for step=1
        start = max(s1.start, s2.start)
        stop = min(s1.stop, s2.stop)
        if start >= stop:
            return slice(0, 0)  # Empty slice if no intersection
        return slice(start, stop)

    def offset_slice(s, offset):
        return slice(s.start + offset, s.stop + offset)

    def is_swiglu_key(key):
        """
        Check if this key should be handled as SwiGLU linear_fc1 weight or bias.
        """
        # Non-expert MLP: 'mlp.linear_fc1.weight', 'mlp.linear_fc1.bias'
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc1.bias0'
        # SequentialMLP: 'mlp.experts.local_experts.0.linear_fc1.weight',
        #   'mlp.experts.local_experts.0.linear_fc1.bias'
        return any(
            re.search(pat, key)
            for pat in [
                r"(.*)\.mlp\.linear_fc1\.weight$",
                r"(.*)\.mlp\.linear_fc1\.bias$",
                r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
                r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
                r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
                r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
                r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight$",
                r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias$",
            ]
        )

    def split_swiglu_linear_fc1(data, dist_param, swiglu_shard_axis, is_expert_param):
        """
        Split the SWiGLU linear_fc1 parameter into two parts: weight_w and weight_v.
        """
        assert data.shape[swiglu_shard_axis] % 2 == 0, (
            f"SWiGLU weights must have an even size along the shard axis {swiglu_shard_axis}, "
            f"got {data.shape[swiglu_shard_axis]}"
        )

        fsdp_slice = dist_param.megatron_fsdp_slice
        megatron_fsdp_dist_index = dist_param.megatron_fsdp_dist_index

        tp_mesh = megatron_fsdp_dist_index.get_submesh(
            [megatron_fsdp_dist_index.tp_dim], is_expert_parallel=is_expert_param
        )
        data_size = data.numel() // tp_mesh.mesh.numel()
        w_slice = slice(0, data_size // 2)
        v_slice = slice(data_size // 2, data_size)

        view_shape = list(data.shape)
        view_shape[swiglu_shard_axis] = -1
        local_tensor = data.to_local()
        weight_w = local_tensor.view(-1)[
            offset_slice(intersection(fsdp_slice, w_slice), -fsdp_slice.start)
        ]
        weight_v = local_tensor.view(-1)[
            offset_slice(intersection(fsdp_slice, v_slice), -fsdp_slice.start)
        ]
        weight_w = weight_w.reshape(view_shape)
        weight_v = weight_v.reshape(view_shape)

        # Fake parameters w and v are used to provide the correct parameter
        # shape and Tensor-Parallelism information.
        per_tp_rank_shape = list(data.shape)
        if is_mcore_tensor_model_parallel(dist_param):
            tp_dim = get_mcore_tensor_parallel_partition_dim(dist_param)
            assert tp_dim is not None, "Tensor model parallel dimension not found"
            per_tp_rank_shape[tp_dim] //= tp_mesh.mesh.numel()
        linear_fc1_meta = torch.empty(*per_tp_rank_shape, device="meta")
        w_meta, v_meta = torch.chunk(linear_fc1_meta, 2, dim=swiglu_shard_axis)
        copy_tensor_model_parallel_attributes(w_meta, dist_param)
        copy_tensor_model_parallel_attributes(v_meta, dist_param)

        weight_w = make_fsdp_dtensor(
            weight_w.data,
            w_meta,
            dist_index=megatron_fsdp_dist_index,
            is_expert_param=is_expert_param,
            run_check=True,
            update_uneven_dtensor_chunk_meta=True,
        )
        weight_v = make_fsdp_dtensor(
            weight_v.data,
            v_meta,
            dist_index=megatron_fsdp_dist_index,
            is_expert_param=is_expert_param,
            run_check=True,
            update_uneven_dtensor_chunk_meta=True,
        )
        return weight_w, weight_v

    model_state_dict = model_state_dict.copy()
    for key in list(model_state_dict.keys()):
        if is_swiglu_key(key):
            dist_param = model.get_parameter(f"module.{key}")
            weight_w, weight_v = split_swiglu_linear_fc1(
                model_state_dict[key],
                dist_param,
                swiglu_shard_axis=0,
                is_expert_param='mlp.experts' in key,
            )

            # Update the model state dict with the new keys
            model_state_dict[f"{key}_w"] = weight_w
            model_state_dict[f"{key}_v"] = weight_v
            del model_state_dict[key]

    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict["state"]) != 0:
            opt_state_dict = optimizer_state_dict["state"]
            new_opt_state_dict = {}
            for key in list(opt_state_dict.keys()):
                # Only process SWIGLU keys
                if not is_swiglu_key(key):
                    new_opt_state_dict[key] = opt_state_dict[key]
                    continue
                new_opt_state_dict[f"{key}_w"] = opt_state_dict[key].copy()
                new_opt_state_dict[f"{key}_v"] = opt_state_dict[key].copy()
                for subkey in ["exp_avg", "exp_avg_sq"]:
                    dist_param = model.get_parameter(expert_param_local_key(key[len("module.") :]))
                    weight_w, weight_v = split_swiglu_linear_fc1(
                        opt_state_dict[key][subkey],
                        dist_param,
                        swiglu_shard_axis=0,
                        is_expert_param="mlp.experts" in key,
                    )
                    # Update the optimizer state dict with the new keys
                    new_opt_state_dict[f"{key}_w"][subkey] = weight_w
                    new_opt_state_dict[f"{key}_v"][subkey] = weight_v
            optimizer_state_dict["state"] = new_opt_state_dict

    return model_state_dict, optimizer_state_dict


def handle_fp8_extra_state_case(model_state_dict):
    """
    Handle the case where FP8 extra state is present in the model state dict.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    for key in list(model_state_dict.keys()):
        if key.endswith('._extra_state'):
            del model_state_dict[key]


def flatten_state_dict(obj, parent_key="", sep="."):
    """
    Recursively flattens a nested state dict into a single-level dict with keys
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_state_dict(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_state_dict(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items


def print_diff_in_state_dicts(state_dict_metadata, load_state_dict, limit=100):
    """
    Print the differences between two state dicts: metadata state dict and load state dict.
    This function compares the keys and shapes of the tensors in both dicts.
    """
    state_dict_metadata = flatten_state_dict(state_dict_metadata)
    load_state_dict = flatten_state_dict(load_state_dict)
    meta_keys = set(state_dict_metadata.keys())
    load_keys = set(load_state_dict.keys())

    only_in_meta = list(meta_keys - load_keys)
    only_in_load = list(load_keys - meta_keys)
    in_both = list(meta_keys & load_keys)

    logger.info(f"Keys only in checkpoint metadata_state_dict(first {limit}):")
    for k in sorted(only_in_meta[:limit]):
        logger.info(f"  {k}")

    logger.info(f"\nKeys only in load_state_dict(first {limit}):")
    for k in sorted(only_in_load[:limit]):
        logger.info(f"  {k}")

    logger.info(f"\nKeys in both but with different shapes(first {limit}):")
    for k in sorted(in_both[:limit]):
        v_meta = state_dict_metadata[k]
        v_load = load_state_dict[k]
        # If tensors, compare shape; else, compare type/values
        meta_shape = v_meta.size if hasattr(v_meta, "size") else type(v_meta)
        load_shape = v_load.shape if hasattr(v_load, "shape") else type(v_load)
        if meta_shape != load_shape:
            logger.info(f"  {k}: meta shape={meta_shape}, load shape={load_shape}")


def validate_loaded_state_dict(state_dict, checkpoint_path):
    """
    Validate the loaded state dict against the expected structure and types.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    # Initialize reader
    reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    flat_state_dict = flatten_state_dict(state_dict)

    for key, value in flat_state_dict.items():
        tensor_metadata = metadata.state_dict_metadata[key]

        if not isinstance(tensor_metadata, TensorStorageMetadata):
            continue
        if not isinstance(value, DTensor):
            load_item_dict = {key: torch.empty_like(value)}
        else:
            load_item_dict = {
                key: torch.distributed.tensor.empty(
                    tensor_metadata.size,
                    dtype=tensor_metadata.properties.dtype,
                    device_mesh=DeviceMesh.from_group(
                        group=dist.group.WORLD,
                        device_type="cuda",
                        mesh=torch.arange(dist.get_world_size()),
                        mesh_dim_names=("world",),
                    ),
                    placements=[Shard(0)],
                )
            }
        torch.distributed.checkpoint.load(
            load_item_dict, storage_reader=reader, planner=default_planner.DefaultLoadPlanner()
        )
        if isinstance(value, DTensor):
            full_value = gather_uneven_dtensor_to_full_tensor(value)
            loaded_tensor = load_item_dict[key].redistribute(
                placements=[Replicate()] * len(value.placements)
            )
            assert torch.allclose(
                loaded_tensor._local_tensor, full_value._local_tensor, atol=1e-8, rtol=1e-5
            ), f"key: {key}; {loaded_tensor} {full_value}"
        else:
            assert torch.allclose(
                value, load_item_dict[key]
            ), f"key: {key}; {value} {load_item_dict[key]}"


def get_global_unique_param_name(model_chunks, param):
    """
    Get the global unique parameter name for a given model and parameter.
    """
    param_name = None
    for model in model_chunks:
        for name, p in model.named_parameters():
            if p is param:
                param_name = name
                break
    if param_name is None:
        raise ValueError("Parameter not found in model chunks")

    # Get PP unique parameter name
    if re.search(r"layers\.(\d+)", param_name) and "mtp" not in param_name:
        tf_layer_number = -1
        for module in model.modules():
            if not isinstance(module, TransformerLayer):
                continue
            for p in module.parameters():
                if p is param:
                    tf_layer_number = module.layer_number
                    break
        if tf_layer_number != -1:
            param_name = re.sub(r"layers\.(\d+)", f"layers.{tf_layer_number - 1}", param_name)

    # Get EP unique parameter name
    param_name = list(handle_experts_in_state_dict({param_name: None}).keys())[0]

    return param_name
