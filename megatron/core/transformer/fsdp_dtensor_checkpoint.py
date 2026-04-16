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
        uneven_dtensor_to_full_tensor,
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
from megatron.core.utils import get_attr_wrapped_model


def get_ep_layer_offset(num_experts: int | None = None) -> int:
    """
    Get the expert layer offset for the current model.

    Args:
        num_experts: Total number of experts in the model. If None, returns 0.

    Returns:
        The expert layer offset for the current EP rank.
    """
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_local_experts = num_experts // ep_size if num_experts else 0
    local_expert_offset = ep_rank * num_local_experts

    return local_expert_offset


def get_total_num_experts(num_experts: int | None = None) -> int:
    """
    Get the total number of experts for the current model.

    Args:
        num_experts: Total number of experts in the model. If None, returns 0.

    Returns:
        The total number of experts.
    """
    return num_experts if num_experts else 0


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


def handle_experts_in_state_dict(state_dict, num_experts: int | None = None):
    """
    Rewrite expert keys in state dict.

    Args:
        state_dict: The state dictionary to process.
        num_experts: Total number of experts in the model. If None, no expert processing occurs.

    Returns:
        The processed state dictionary with rewritten expert keys.
    """
    local_expert_start = get_ep_layer_offset(num_experts)
    local_expert_end = get_total_num_experts(num_experts)

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


def expert_param_local_key(key: str, num_experts: int | None = None) -> str:
    """Get the module parameter corresponding to the key.

    Args:
        key: The parameter key to process.
        num_experts: Total number of experts in the model. If None, no expert processing occurs.

    Returns:
        The local parameter key with adjusted expert indices.
    """
    local_expert_offset = get_ep_layer_offset(num_experts)
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

    Only splits linear_fc1 parameters whose parent TransformerLayer has
    ``config.gated_linear_unit == True``.  This is critical for heterogeneous
    models (e.g. VLMs) where the vision encoder uses GELU while the language
    decoder uses SWiGLU — splitting non-SWiGLU fc1 weights would create _w/_v
    keys that don't exist in the checkpoint, causing a load-time mismatch.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    # Extract num_experts from model config for expert parameter processing
    model_config = get_attr_wrapped_model(model, "config", allow_none=True)
    num_experts = (
        getattr(model_config, 'num_moe_experts', None) if model_config is not None else None
    )

    # ------------------------------------------------------------------
    # Build per-TransformerLayer gated_linear_unit map.
    # For homogeneous LLMs every layer agrees; for VLMs the vision encoder
    # layers have gated_linear_unit=False while language decoder layers
    # have gated_linear_unit=True.
    # ------------------------------------------------------------------
    def _strip_wrappers(path):
        """Strip DDP/FSDP wrapper prefixes (module., model.) from a path."""
        parts = path.split('.')
        while parts and parts[0] in ('module', 'model'):
            parts = parts[1:]
        return '.'.join(parts)

    _layer_glu = {}
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            _layer_glu[_strip_wrappers(name)] = getattr(module.config, 'gated_linear_unit', False)

    def _key_in_glu_layer(key):
        """Return True if *key* belongs to a TransformerLayer with gated_linear_unit=True."""
        norm_key = _strip_wrappers(key)
        best_glu, best_len = None, -1
        for layer_path, uses_glu in _layer_glu.items():
            if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
                best_glu, best_len = uses_glu, len(layer_path)
        if best_glu is None:
            return True  # no TransformerLayer found — assume GLU for backward compat
        return best_glu

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
        local_tensor = data.to_local() if isinstance(data, DTensor) else data
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
    _swiglu_split_count = 0
    _swiglu_skip_count = 0
    for key in list(model_state_dict.keys()):
        if is_swiglu_key(key):
            if not _key_in_glu_layer(key):
                _swiglu_skip_count += 1
                continue
            dist_param = model.get_parameter(key)
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
            _swiglu_split_count += 1

    if _swiglu_skip_count > 0:
        logger.info(
            f"[SWiGLU] Split {_swiglu_split_count} fc1 keys; "
            f"skipped {_swiglu_skip_count} keys in non-GLU layers (e.g. vision encoder)."
        )

    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict["state"]) != 0:
            opt_state_dict = optimizer_state_dict["state"]
            new_opt_state_dict = {}
            for key in list(opt_state_dict.keys()):
                # Note: unwrap_model returns the original (unwrapped) model.
                # However, keys in opt_state_dict contain three "module." prefixes due to nested wrapping,
                # whereas model.state_dict keys follow the native (unwrapped) format.
                #
                # opt_state_dict key example:
                #   "module.module.module.language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0"
                # model.state_dict key example:
                #   "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0"
                if not is_swiglu_key(key) or not _key_in_glu_layer(key):
                    new_opt_state_dict[key] = opt_state_dict[key]
                    continue
                new_opt_state_dict[f"{key}_w"] = opt_state_dict[key].copy()
                new_opt_state_dict[f"{key}_v"] = opt_state_dict[key].copy()
                for subkey in ["exp_avg", "exp_avg_sq"]:
                    dist_param = model.get_parameter(
                        expert_param_local_key(_strip_wrappers(key), num_experts)
                    )
                    weight_w, weight_v = split_swiglu_linear_fc1(
                        opt_state_dict[key][subkey],
                        dist_param,
                        swiglu_shard_axis=0,
                        is_expert_param="mlp.experts" in key,
                    )
                    new_opt_state_dict[f"{key}_w"][subkey] = weight_w
                    new_opt_state_dict[f"{key}_v"][subkey] = weight_v
            optimizer_state_dict["state"] = new_opt_state_dict

    return model_state_dict, optimizer_state_dict


def handle_gdn_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """Handle GDN (Gated DeltaNet) fused projections in model and optimizer state dicts.

    GDN layers fuse query/key/value/gate/beta/alpha projections into a single
    ``in_proj.weight`` ColumnParallelLinear, and query/key/value into ``conv1d``
    (weight + optional bias).  For FSDP checkpoints these fused tensors must be
    split back into their constituent sub-tensors so that each can be
    independently TP-sharded — otherwise loading a checkpoint written at TP=M
    into TP=N would slice across logical component boundaries.

    This is analogous to :func:`handle_swiglu_in_state_dict` which splits
    ``linear_fc1`` into ``weight_w`` / ``weight_v``.

    Sub-key naming follows ``GatedDeltaNet.sharded_state_dict()``::

        in_proj.weight  → .query / .key / .value / .z / .beta / .alpha   (6-way)
        conv1d.weight   → .query / .key / .value                         (3-way)
        conv1d.bias     → .query / .key / .value                         (3-way)
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    GDN_IN_PROJ_NAMES = ["query", "key", "value", "z", "beta", "alpha"]
    GDN_CONV1D_NAMES = ["query", "key", "value"]

    def _strip_wrappers(path):
        """Strip DDP/FSDP wrapper prefixes (module., model.) from a path."""
        parts = path.split('.')
        while parts and parts[0] in ('module', 'model'):
            parts = parts[1:]
        return '.'.join(parts)

    # ------------------------------------------------------------------
    # Build per-GDN-module split-size map by walking the model tree.
    # GDN modules are identified by the presence of qk_dim / v_dim /
    # in_proj_dim attributes (set in GatedDeltaNet.__init__).
    # ------------------------------------------------------------------
    _gdn_info = {}  # normalized_path → {in_proj_sizes, conv1d_sizes}
    for name, mod in model.named_modules():
        if not (hasattr(mod, 'qk_dim') and hasattr(mod, 'v_dim') and hasattr(mod, 'in_proj_dim')):
            continue
        tp = getattr(mod, 'tp_size', 1)
        qk = mod.qk_dim // tp
        v = mod.v_dim // tp
        nvh = mod.num_value_heads // tp
        _gdn_info[_strip_wrappers(name)] = {
            'in_proj_sizes': [qk, qk, v, v, nvh, nvh],
            'conv1d_sizes': [qk, qk, v],
        }

    if not _gdn_info:
        return model_state_dict, optimizer_state_dict

    def _match_gdn_key(key):
        """Return (split_sizes, sub_names, split_dim) if *key* is a GDN fused
        parameter that needs splitting, else ``None``."""
        norm = _strip_wrappers(key)
        for gdn_path, info in _gdn_info.items():
            if not norm.startswith(gdn_path + '.'):
                continue
            rel = norm[len(gdn_path) + 1 :]
            if rel == 'in_proj.weight':
                return info['in_proj_sizes'], GDN_IN_PROJ_NAMES, 0
            if rel in ('conv1d.weight', 'conv1d.bias'):
                return info['conv1d_sizes'], GDN_CONV1D_NAMES, 0
        return None

    def intersection(s1, s2):
        start = max(s1.start, s2.start)
        stop = min(s1.stop, s2.stop)
        return slice(0, 0) if start >= stop else slice(start, stop)

    def offset_slice(s, offset):
        return slice(s.start + offset, s.stop + offset)

    def split_gdn_fused(data, dist_param, split_sizes, split_dim):
        """Split a fused GDN projection DTensor into per-component DTensors."""
        fsdp_slice = dist_param.megatron_fsdp_slice
        dist_index = dist_param.megatron_fsdp_dist_index
        tp_mesh = dist_index.get_submesh([dist_index.tp_dim], is_expert_parallel=False)

        data_size = data.numel() // tp_mesh.mesh.numel()
        total_split = sum(split_sizes)
        elems_per_unit = data_size // total_split

        local_tensor = data.to_local()
        view_shape = list(data.shape)

        per_tp_rank_shape = list(data.shape)
        if is_mcore_tensor_model_parallel(dist_param):
            tp_dim = get_mcore_tensor_parallel_partition_dim(dist_param)
            assert tp_dim is not None, "Tensor model parallel dimension not found"
            per_tp_rank_shape[tp_dim] //= tp_mesh.mesh.numel()

        results = []
        flat_offset = 0
        for s in split_sizes:
            comp_flat = s * elems_per_unit
            comp_slice = slice(flat_offset, flat_offset + comp_flat)

            shard = intersection(fsdp_slice, comp_slice)
            comp_data = local_tensor.view(-1)[offset_slice(shard, -fsdp_slice.start)]

            comp_view = list(view_shape)
            comp_view[split_dim] = -1
            comp_data = comp_data.reshape(comp_view)

            meta_shape = list(per_tp_rank_shape)
            meta_shape[split_dim] = s
            meta = torch.empty(*meta_shape, device="meta")
            copy_tensor_model_parallel_attributes(meta, dist_param)

            dtensor = make_fsdp_dtensor(
                comp_data.data,
                meta,
                dist_index=dist_index,
                is_expert_param=False,
                run_check=True,
                update_uneven_dtensor_chunk_meta=True,
            )
            results.append(dtensor)
            flat_offset += comp_flat

        return results

    # ---- Model state dict ------------------------------------------------
    model_state_dict = model_state_dict.copy()
    _gdn_split_count = 0
    for key in list(model_state_dict.keys()):
        match = _match_gdn_key(key)
        if match is None:
            continue
        sizes, names, dim = match
        dist_param = model.get_parameter(f"{key}")
        sub_tensors = split_gdn_fused(model_state_dict[key], dist_param, sizes, dim)
        for sub_name, tensor in zip(names, sub_tensors):
            model_state_dict[f"{key}.{sub_name}"] = tensor
        del model_state_dict[key]
        _gdn_split_count += 1

    if _gdn_split_count > 0:
        logger.info(
            f"[GDN] Split {_gdn_split_count} fused keys into sub-tensors "
            f"(in_proj/conv1d → query/key/value/...)."
        )

    # ---- Optimizer state dict --------------------------------------------
    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict["state"]) != 0:
            opt_state = optimizer_state_dict["state"]
            new_opt_state = {}
            for key in list(opt_state.keys()):
                match = _match_gdn_key(key)
                if match is None:
                    new_opt_state[key] = opt_state[key]
                    continue
                sizes, names, dim = match
                for sub_name in names:
                    new_opt_state[f"{key}.{sub_name}"] = opt_state[key].copy()
                for subkey in ["exp_avg", "exp_avg_sq"]:
                    dist_param = model.get_parameter(key)
                    sub_tensors = split_gdn_fused(opt_state[key][subkey], dist_param, sizes, dim)
                    for sub_name, tensor in zip(names, sub_tensors):
                        new_opt_state[f"{key}.{sub_name}"][subkey] = tensor
            optimizer_state_dict["state"] = new_opt_state

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
            full_tensor_v = uneven_dtensor_to_full_tensor(value)
            loaded_tensor = load_item_dict[key].redistribute(
                placements=[Replicate()] * len(value.placements)
            )
            assert torch.allclose(
                loaded_tensor._local_tensor, full_tensor_v, atol=1e-8, rtol=1e-5
            ), f"key: {key}; {loaded_tensor} {full_tensor_v}"
        else:
            assert torch.allclose(
                value, load_item_dict[key]
            ), f"key: {key}; {value} {load_item_dict[key]}"


def get_global_unique_param_name(model_chunks, param):
    """
    Get the global unique parameter name for a given model and parameter.

    Args:
        model_chunks: List of model chunks to search for the parameter.
        param: The parameter to find the name for.

    Returns:
        The global unique parameter name.
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
    num_experts = model_chunks[0].config.num_moe_experts if model_chunks else None
    param_name = next(iter(handle_experts_in_state_dict({param_name: None}, num_experts).keys()))

    return param_name
