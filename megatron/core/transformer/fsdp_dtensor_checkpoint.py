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
        split_dtensor,
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
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.validation import StrictHandling, parse_strict_flag
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import get_attr_wrapped_model


def _strip_wrapper_prefixes(path):
    """Strip DDP/FSDP wrapper prefixes (module., model.) from a module or state dict path."""
    parts = path.split('.')
    while parts and parts[0] in ('module', 'model'):
        parts = parts[1:]
    return '.'.join(parts)


def _intersect_slice(s1, s2):
    """Intersection of two step-1 slices, or an empty slice when they do not overlap."""
    start = max(s1.start, s2.start)
    stop = min(s1.stop, s2.stop)
    return slice(0, 0) if start >= stop else slice(start, stop)


def _shift_slice(s, offset):
    """Move a step-1 slice by ``offset``, e.g. to rebase it onto a shard's local storage."""
    return slice(s.start + offset, s.stop + offset)


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
    _layer_glu = {}
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            _layer_glu[_strip_wrapper_prefixes(name)] = getattr(
                module.config, 'gated_linear_unit', False
            )

    def _key_in_glu_layer(key):
        """Return True if *key* belongs to a TransformerLayer with gated_linear_unit=True."""
        norm_key = _strip_wrapper_prefixes(key)
        best_glu, best_len = None, -1
        for layer_path, uses_glu in _layer_glu.items():
            if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
                best_glu, best_len = uses_glu, len(layer_path)
        if best_glu is None:
            return True  # no TransformerLayer found — assume GLU for backward compat
        return best_glu

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
            _shift_slice(_intersect_slice(fsdp_slice, w_slice), -fsdp_slice.start)
        ]
        weight_v = local_tensor.view(-1)[
            _shift_slice(_intersect_slice(fsdp_slice, v_slice), -fsdp_slice.start)
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
                if not is_swiglu_key(key) or not _key_in_glu_layer(key):
                    new_opt_state_dict[key] = opt_state_dict[key]
                    continue
                new_opt_state_dict[f"{key}_w"] = opt_state_dict[key].copy()
                new_opt_state_dict[f"{key}_v"] = opt_state_dict[key].copy()
                for subkey in ["exp_avg", "exp_avg_sq"]:
                    dist_param = model.get_parameter(
                        expert_param_local_key(key[len("module.") :], num_experts)
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
    """Handle GDN-family fused projections in model and optimizer state dicts.

    GDN layers fuse query/key/value/gate/beta/alpha projections into a single
    ``in_proj.weight`` ColumnParallelLinear, while KDA uses its own five-way
    query/key/value/g/gate split and a separate beta projection. Both variants
    share the query/key/value ``conv1d`` split. For FSDP checkpoints these fused
    tensors must be split back into their constituent sub-tensors so that each
    can be independently TP-sharded — otherwise loading a checkpoint written at
    TP=M into TP=N would slice across logical component boundaries.

    This is analogous to :func:`handle_swiglu_in_state_dict` which splits
    ``linear_fc1`` into ``weight_w`` / ``weight_v``.

    Sub-key naming follows each module's ``in_proj_split_names`` and
    ``in_proj_split_sections`` metadata::

        GDN in_proj.weight → .query / .key / .value / .z / .beta / .alpha (6-way)
        KDA in_proj.weight → .query / .key / .value / .g / .gate (5-way)
        conv1d.weight   → .query / .key / .value                         (3-way)
        conv1d.bias     → .query / .key / .value                         (3-way)
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    GDN_CONV1D_NAMES = ["query", "key", "value"]

    # ------------------------------------------------------------------
    # Build the per-GDN-family-module split-size map by walking the model tree.
    # Both GDN and KDA expose qk_dim / v_dim / in_proj_dim; the split metadata
    # is authoritative because their in_proj layouts are different.
    # ------------------------------------------------------------------
    projection_info = {}  # normalized path → split sizes and names
    for name, module in model.named_modules():
        if not all(hasattr(module, attr) for attr in ('qk_dim', 'v_dim', 'in_proj_dim')):
            continue
        has_names = hasattr(module, 'in_proj_split_names')
        has_sections = hasattr(module, 'in_proj_split_sections')
        if not has_names and not has_sections:
            continue
        if not (has_names and has_sections):
            raise ValueError(
                f"GDN-family module {name!r} must define both in_proj_split_names and "
                "in_proj_split_sections."
            )

        in_proj_names = tuple(module.in_proj_split_names)
        in_proj_sizes = tuple(module.in_proj_split_sections)
        if not in_proj_names or len(in_proj_sizes) != len(in_proj_names):
            raise ValueError(
                f"GDN-family module {name!r} has mismatched in_proj split metadata: "
                f"{len(in_proj_sizes)} sizes for {len(in_proj_names)} names."
            )
        if any(not isinstance(split_name, str) or not split_name for split_name in in_proj_names):
            raise ValueError(
                f"GDN-family module {name!r} has invalid in_proj split names: {in_proj_names}."
            )
        if len(set(in_proj_names)) != len(in_proj_names):
            raise ValueError(
                f"GDN-family module {name!r} has duplicate in_proj split names: "
                f"{in_proj_names}."
            )
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in in_proj_sizes
        ):
            raise ValueError(
                f"GDN-family module {name!r} has invalid in_proj split sizes: {in_proj_sizes}."
            )

        tp_size = getattr(module, 'tp_size', 1)
        if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size <= 0:
            raise ValueError(f"GDN-family module {name!r} has invalid tp_size={tp_size!r}.")
        if module.in_proj_dim % tp_size != 0:
            raise ValueError(
                f"GDN-family module {name!r} has in_proj_dim={module.in_proj_dim} "
                f"which is not divisible by tp_size={tp_size}."
            )
        expected_size = module.in_proj_dim // tp_size
        if sum(in_proj_sizes) != expected_size:
            raise ValueError(
                f"GDN-family module {name!r} has in_proj split sizes totaling "
                f"{sum(in_proj_sizes)}, expected {expected_size}."
            )

        qk_size = getattr(module, 'qk_dim_local_tp', None)
        value_size = getattr(module, 'v_dim_local_tp', None)
        if qk_size is None or value_size is None:
            if module.qk_dim % tp_size != 0 or module.v_dim % tp_size != 0:
                raise ValueError(
                    f"GDN-family module {name!r} qk_dim/v_dim must be divisible by "
                    f"tp_size={tp_size}."
                )
            qk_size = module.qk_dim // tp_size
            value_size = module.v_dim // tp_size

        normalized_name = _strip_wrapper_prefixes(name)
        if normalized_name in projection_info:
            raise ValueError(f"Multiple GDN-family modules normalize to path {normalized_name!r}.")
        projection_info[normalized_name] = {
            'in_proj_sizes': in_proj_sizes,
            'in_proj_names': in_proj_names,
            'conv1d_sizes': (qk_size, qk_size, value_size),
        }

    if not projection_info:
        return model_state_dict, optimizer_state_dict

    parameter_map = {}
    for name, parameter in model.named_parameters():
        normalized_name = _strip_wrapper_prefixes(name)
        if normalized_name in parameter_map and parameter_map[normalized_name] is not parameter:
            raise ValueError(
                f"Multiple parameters normalize to GDN-family key {normalized_name!r}."
            )
        parameter_map[normalized_name] = parameter

    def _get_parameter(key):
        normalized_key = _strip_wrapper_prefixes(key)
        try:
            return parameter_map[normalized_key]
        except KeyError as error:
            raise KeyError(
                f"No model parameter matches GDN-family state-dict key {key!r} "
                f"(normalized as {normalized_key!r})."
            ) from error

    def _match_gdn_key(key):
        """Return split metadata when ``key`` names a fused GDN-family parameter."""
        norm = _strip_wrapper_prefixes(key)
        for module_path, info in projection_info.items():
            if module_path:
                if not norm.startswith(module_path + '.'):
                    continue
                rel = norm[len(module_path) + 1 :]
            else:
                rel = norm
            if rel == 'in_proj.weight':
                return info['in_proj_sizes'], info['in_proj_names'], 0
            if rel in ('conv1d.weight', 'conv1d.bias'):
                return info['conv1d_sizes'], GDN_CONV1D_NAMES, 0
        return None

    def split_gdn_fused(data, dist_param, split_sizes, split_dim):
        """Split a fused GDN-family projection DTensor into component DTensors.

        The implementation handles both model-state DTensors and optimizer-state
        tensors while preserving FSDP and tensor-parallel metadata.
        """
        total_split = sum(split_sizes)
        if isinstance(data, DTensor) and data.shape[split_dim] == total_split:
            return list(
                split_dtensor(
                    data, split_sizes, dim=split_dim, update_uneven_dtensor_chunk_meta=True
                )
            )

        fsdp_slice = dist_param.megatron_fsdp_slice
        dist_index = dist_param.megatron_fsdp_dist_index
        tp_mesh = dist_index.get_submesh([dist_index.tp_dim], is_expert_parallel=False)

        data_size = data.numel() // tp_mesh.mesh.numel()
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

            shard = _intersect_slice(fsdp_slice, comp_slice)
            comp_data = local_tensor.view(-1)[_shift_slice(shard, -fsdp_slice.start)]

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
        dist_param = _get_parameter(key)
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
                    dist_param = _get_parameter(key)
                    sub_tensors = split_gdn_fused(opt_state[key][subkey], dist_param, sizes, dim)
                    for sub_name, tensor in zip(names, sub_tensors):
                        new_opt_state[f"{key}.{sub_name}"][subkey] = tensor
            optimizer_state_dict["state"] = new_opt_state

    return model_state_dict, optimizer_state_dict


def split_fused_fsdp_param(data, dist_param, split_sizes, is_expert_param=False, split_dim=0):
    """Split a fused Megatron-FSDP parameter along ``split_dim`` into per-section DTensors.

    ``split_sizes`` are given in full, tensor-parallel-unsharded units along ``split_dim``
    (so they can be taken straight from the module config) and are scaled down to this
    rank's TP shard internally.

    The returned DTensors alias the fused parameter's storage, so a save reads the fused
    values and an in-place load writes back into the fused parameter.

    Same flat-slice arithmetic as :func:`handle_swiglu_in_state_dict`, generalized from an
    even two-way split to arbitrary section sizes.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    fsdp_slice = dist_param.megatron_fsdp_slice
    dist_index = dist_param.megatron_fsdp_dist_index
    tp_mesh = dist_index.get_submesh([dist_index.tp_dim], is_expert_parallel=is_expert_param)

    per_tp_rank_shape = list(data.shape)
    if is_mcore_tensor_model_parallel(dist_param):
        tp_dim = get_mcore_tensor_parallel_partition_dim(dist_param)
        assert tp_dim is not None, "Tensor model parallel dimension not found"
        per_tp_rank_shape[tp_dim] //= tp_mesh.mesh.numel()

    total_full = sum(split_sizes)
    assert data.shape[split_dim] == total_full, (
        f"Fused parameter is {data.shape[split_dim]} wide along dim {split_dim}, "
        f"but the requested sections sum to {total_full}"
    )

    local_total = per_tp_rank_shape[split_dim]
    local_sizes = []
    for size in split_sizes:
        assert (size * local_total) % total_full == 0, (
            f"Section of size {size} does not divide evenly across the tensor-parallel "
            f"group (fused dim {total_full} -> {local_total} on this rank)"
        )
        local_sizes.append(size * local_total // total_full)

    data_size = 1
    for dim_size in per_tp_rank_shape:
        data_size *= dim_size
    elems_per_unit = data_size // local_total

    local_tensor = data.to_local()
    view_shape = list(per_tp_rank_shape)
    view_shape[split_dim] = -1

    results = []
    flat_offset = 0
    for local_size in local_sizes:
        section_numel = local_size * elems_per_unit
        section_slice = slice(flat_offset, flat_offset + section_numel)

        shard = _intersect_slice(fsdp_slice, section_slice)
        section_data = local_tensor.view(-1)[_shift_slice(shard, -fsdp_slice.start)]
        section_data = section_data.reshape(view_shape)

        # A meta tensor carries the unsharded shape and TP attributes of the section.
        meta_shape = list(per_tp_rank_shape)
        meta_shape[split_dim] = local_size
        section_meta = torch.empty(*meta_shape, device="meta")
        copy_tensor_model_parallel_attributes(section_meta, dist_param)

        results.append(
            make_fsdp_dtensor(
                section_data.data,
                section_meta,
                dist_index=dist_index,
                is_expert_param=is_expert_param,
                run_check=True,
                update_uneven_dtensor_chunk_meta=True,
            )
        )
        flat_offset += section_numel

    return results


# The single down-projection registered by FusedMLASelfAttention, and the two separate
# projections registered by the unfused MLASelfAttention it replaces.
MLA_FUSED_DOWN_PROJ = 'linear_qkv_down_proj'
MLA_UNFUSED_DOWN_PROJS = ('linear_q_down_proj', 'linear_kv_down_proj')


def get_mla_fused_down_proj_splits(model):
    """Map each FusedMLASelfAttention module path to its ``[q, kv]`` down-proj split sizes.

    Sizes match ``FusedMLASelfAttention.sharded_state_dict`` and are expressed in full,
    tensor-parallel-unsharded units. Returns an empty dict for unfused models.
    """
    splits = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, MLA_FUSED_DOWN_PROJ):
            continue
        config = mod.config
        splits[_strip_wrapper_prefixes(name)] = [
            config.q_lora_rank,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
        ]
    return splits


def match_mla_fused_down_proj_key(key, fused_splits):
    """Match a state dict key against the fused MLA down-projections in ``fused_splits``.

    Returns ``(wrapper, attention_path, split_sizes, leaf)`` where ``wrapper`` is the
    stripped ``module.``/``model.`` prefix and ``leaf`` is the part after the fused module
    name, or ``None`` when the key is not a fused down-projection.
    """
    norm = _strip_wrapper_prefixes(key)
    for attention_path, split_sizes in fused_splits.items():
        fused_prefix = f'{attention_path}.{MLA_FUSED_DOWN_PROJ}.'
        if norm.startswith(fused_prefix):
            wrapper = key[: len(key) - len(norm)]
            return wrapper, attention_path, split_sizes, norm[len(fused_prefix) :]
    return None


def absorbed_input_layernorm_key(wrapper, attention_path, leaf):
    """Rewrite a ``layer_norm_*`` leaf absorbed by the fused down-proj to ``input_layernorm.*``.

    With ``fuse_input_layernorm``, the fused module owns the layer's input layernorm as
    ``layer_norm_weight``/``layer_norm_bias``; on disk it lives one level up under
    ``input_layernorm``, per the layer spec's ``sharded_state_dict_keys_map``.
    """
    assert '.' in attention_path, (
        f"Cannot locate the transformer layer owning {attention_path}.{leaf}; expected the "
        f"attention module to be nested inside it."
    )
    layer_path = attention_path.rsplit('.', 1)[0]
    return f'{wrapper}{layer_path}.input_layernorm.{leaf[len("layer_norm_"):]}'


def handle_mla_down_proj_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """Rewrite a fused MLA down-projection into the unfused layout used on disk.

    ``mla_down_proj_fusion=True`` swaps ``MLASelfAttention`` for ``FusedMLASelfAttention``,
    replacing ``linear_q_down_proj`` and ``linear_kv_down_proj`` with a single
    ``linear_qkv_down_proj`` holding their row-concatenation, and on the Transformer Engine
    backend also absorbing the layer's ``input_layernorm`` into it as ``layer_norm_*``.

    ``FusedMLASelfAttention.sharded_state_dict`` hides both changes from ``torch_dist``
    checkpoints so fused and unfused runs share one on-disk format. The Megatron-FSDP path
    builds its keys from ``named_parameters()`` and never calls ``sharded_state_dict``, so
    the same remapping is applied here.

    No-op for unfused models.
    """
    assert HAVE_MEGATRON_FSDP, "This function requires Megatron-FSDP to be installed."

    fused_splits = get_mla_fused_down_proj_splits(model)
    if not fused_splits:
        return model_state_dict, optimizer_state_dict

    rewritten = 0

    model_state_dict = model_state_dict.copy()
    for key in list(model_state_dict.keys()):
        match = match_mla_fused_down_proj_key(key, fused_splits)
        if match is None:
            continue
        wrapper, attention_path, split_sizes, leaf = match

        if leaf.endswith('_extra_state'):
            # Not a tensor; handle_fp8_extra_state_case drops these beforehand.
            continue

        if leaf.startswith('layer_norm_'):
            new_key = absorbed_input_layernorm_key(wrapper, attention_path, leaf)
            model_state_dict[new_key] = model_state_dict.pop(key)
            rewritten += 1
            continue

        sections = split_fused_fsdp_param(
            model_state_dict[key], model.get_parameter(f'module.{key}'), split_sizes
        )
        for proj_name, section in zip(MLA_UNFUSED_DOWN_PROJS, sections):
            model_state_dict[f'{wrapper}{attention_path}.{proj_name}.{leaf}'] = section
        del model_state_dict[key]
        rewritten += 1

    if rewritten:
        logger.info(
            f"[MLA] Rewrote {rewritten} fused {MLA_FUSED_DOWN_PROJ} key(s) across "
            f"{len(fused_splits)} attention module(s) into the unfused layout."
        )

    if optimizer_state_dict is not None and len(optimizer_state_dict.get("state", {})) != 0:
        optimizer_state_dict = optimizer_state_dict.copy()
        optimizer_state = optimizer_state_dict["state"]
        new_optimizer_state = {}
        for key in list(optimizer_state.keys()):
            match = match_mla_fused_down_proj_key(key, fused_splits)
            if match is None:
                new_optimizer_state[key] = optimizer_state[key]
                continue
            wrapper, attention_path, split_sizes, leaf = match

            if leaf.startswith('layer_norm_'):
                # Absorbed input layernorm: moved, not split.
                new_key = absorbed_input_layernorm_key(wrapper, attention_path, leaf)
                new_optimizer_state[new_key] = optimizer_state[key]
                continue

            new_keys = [
                f'{wrapper}{attention_path}.{proj_name}.{leaf}'
                for proj_name in MLA_UNFUSED_DOWN_PROJS
            ]
            for new_key in new_keys:
                new_optimizer_state[new_key] = optimizer_state[key].copy()
            dist_param = model.get_parameter(key[len("module.") :])
            for subkey in ["exp_avg", "exp_avg_sq"]:
                sections = split_fused_fsdp_param(
                    optimizer_state[key][subkey], dist_param, split_sizes
                )
                for new_key, section in zip(new_keys, sections):
                    new_optimizer_state[new_key][subkey] = section
        optimizer_state_dict["state"] = new_optimizer_state

    return model_state_dict, optimizer_state_dict


# MCore renamed MultiTokenPredictionLayer's inner transformer layer; checkpoints keep the
# original name.
MTP_INNER_LAYER = 'mtp_model_layer'
MTP_INNER_LAYER_CHECKPOINT_NAME = 'transformer_layer'


def get_mtp_inner_layer_paths(model):
    """Paths of MTP layers whose inner layer is renamed on disk.

    Mamba MTP layers (``mtp_layer_pattern`` set) are excluded, matching
    ``MultiTokenPredictionLayer.sharded_state_dict``: for them ``mtp_model_layer`` is
    already the native checkpoint name.
    """
    return [
        _strip_wrapper_prefixes(name)
        for name, mod in model.named_modules()
        if hasattr(mod, MTP_INNER_LAYER) and getattr(mod, 'mtp_layer_pattern', None) is None
    ]


def rename_mtp_inner_layer_keys(state_dict, mtp_layer_paths):
    """Rename ``mtp_model_layer.*`` keys to ``transformer_layer.*`` under ``mtp_layer_paths``."""
    state_dict = state_dict.copy()
    for key in list(state_dict.keys()):
        norm = _strip_wrapper_prefixes(key)
        for mtp_path in mtp_layer_paths:
            current_prefix = f'{mtp_path}.{MTP_INNER_LAYER}.'
            if not norm.startswith(current_prefix):
                continue
            wrapper = key[: len(key) - len(norm)]
            leaf = norm[len(current_prefix) :]
            new_key = f'{wrapper}{mtp_path}.{MTP_INNER_LAYER_CHECKPOINT_NAME}.{leaf}'
            state_dict[new_key] = state_dict.pop(key)
            break
    return state_dict


def handle_mtp_in_state_dict(model, model_state_dict, optimizer_state_dict):
    """Rename MTP inner-layer keys to the ``transformer_layer`` name used on disk.

    ``MultiTokenPredictionLayer.sharded_state_dict`` remaps ``mtp_model_layer.*`` back to
    ``transformer_layer.*`` so GPT MTP checkpoints written before the rename keep loading.
    The Megatron-FSDP path never calls ``sharded_state_dict``, so the same mapping is
    applied here.

    No-op for models without MTP layers.
    """
    mtp_layer_paths = get_mtp_inner_layer_paths(model)
    if not mtp_layer_paths:
        return model_state_dict, optimizer_state_dict

    model_state_dict = rename_mtp_inner_layer_keys(model_state_dict, mtp_layer_paths)
    logger.info(
        f"[MTP] Renamed {MTP_INNER_LAYER} -> {MTP_INNER_LAYER_CHECKPOINT_NAME} for "
        f"{len(mtp_layer_paths)} MTP layer(s)."
    )

    if optimizer_state_dict is not None and len(optimizer_state_dict.get("state", {})) != 0:
        optimizer_state_dict = optimizer_state_dict.copy()
        optimizer_state_dict["state"] = rename_mtp_inner_layer_keys(
            optimizer_state_dict["state"], mtp_layer_paths
        )

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


# Top-level sections holding model weights: "model" for a single chunk, "model0", "model1",
# ... with virtual pipeline parallelism.
_MODEL_SECTION_PATTERN = re.compile(r'^model\d*\.')


def get_unexpected_model_keys(state_dict_metadata, load_state_dict):
    """Model weights this rank requests that the checkpoint does not contain.

    "Unexpected" follows :class:`~megatron.core.dist_checkpointing.validation.StrictHandling`:
    keys accessed locally but absent from the checkpoint. The opposite direction (keys the
    checkpoint holds that this rank does not request) is *not* reported, because that is
    normal under pipeline, tensor and expert parallelism where every rank loads a subset of
    the global checkpoint.

    Only the ``model`` sections are inspected, so optimizer and RNG state that is
    intentionally absent does not count. Transformer Engine ``_extra_state`` entries are
    skipped because they are dropped from both sides before saving and loading.
    """
    metadata_keys = set(flatten_state_dict(state_dict_metadata).keys())
    return {
        key
        for key in flatten_state_dict(load_state_dict)
        if _MODEL_SECTION_PATTERN.match(key)
        and not key.endswith('._extra_state')
        and key not in metadata_keys
    }


def validate_fsdp_dtensor_model_load(
    state_dict_metadata,
    load_state_dict,
    checkpoint_path,
    strict=StrictHandling.RAISE_UNEXPECTED,
    max_reported=20,
):
    """Report model weights that an ``fsdp_dtensor`` load cannot supply.

    Torch DCP loads with ``allow_partial_load=True`` skip such keys silently, leaving those
    parameters at their initialized values. That does not crash; it surfaces much later as
    a convergence regression, so by default fail here instead.

    Only "unexpected" keys are determined (see :func:`get_unexpected_model_keys`), so the
    ``*_ALL`` variants of ``strict`` behave like their ``*_UNEXPECTED`` counterparts;
    identifying genuinely "missing" keys would require exchanging key sets across ranks.

    ``ASSUME_OK_UNEXPECTED`` is treated as ``RAISE_UNEXPECTED`` here. It means "rely on the
    underlying strategy to raise", which a partial DCP load never does, and it exists to skip
    the extra disk access an explicit check normally costs — but the caller already holds the
    checkpoint metadata, so this check is free. Use ``IGNORE_ALL`` to actually skip it.

    Args:
        state_dict_metadata: ``state_dict_metadata`` from the checkpoint's DCP metadata.
        load_state_dict: state dict this rank is about to load into.
        checkpoint_path: checkpoint location, for the error message.
        strict (StrictHandling): how to handle a mismatch. Defaults to raising.
        max_reported (int): cap on the number of keys named in the message.

    Returns:
        Set[str]: the unexpected model keys, empty when ``strict`` disables the check.

    Raises:
        CheckpointingException: if ``strict`` is a ``RAISE_*`` value and keys are missing.
    """
    strict = parse_strict_flag(strict)
    if strict is StrictHandling.IGNORE_ALL:
        return set()

    unexpected_keys = get_unexpected_model_keys(state_dict_metadata, load_state_dict)
    if not unexpected_keys or strict in (
        StrictHandling.RETURN_UNEXPECTED,
        StrictHandling.RETURN_ALL,
    ):
        return unexpected_keys

    reported = sorted(unexpected_keys)
    truncated = (
        f'\n  ... and {len(reported) - max_reported} more' if len(reported) > max_reported else ''
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    message = (
        f"Megatron-FSDP checkpoint load is incomplete: rank {rank} requests "
        f"{len(reported)} model parameter(s) not present in the checkpoint at "
        f"{checkpoint_path}, which would silently keep their initialized values.\n"
        + '\n'.join(f'  {key}' for key in reported[:max_reported])
        + truncated
        + "\nThe model's parameter names have diverged from the checkpoint's, typically "
        "because of a fused vs. unfused module variant or a renamed submodule. "
        "fsdp_dtensor checkpoints bypass sharded_state_dict(), so key remapping "
        "implemented there does not apply; add the mapping to "
        "preprocess_fsdp_dtensor_state_dict instead."
    )

    if strict in (
        StrictHandling.ASSUME_OK_UNEXPECTED,
        StrictHandling.RAISE_UNEXPECTED,
        StrictHandling.RAISE_ALL,
    ):
        raise CheckpointingException(message)
    logger.warning(message)
    return unexpected_keys


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
