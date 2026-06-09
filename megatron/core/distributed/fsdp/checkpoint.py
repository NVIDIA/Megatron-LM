# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Checkpoint save/load helpers for Megatron FSDP v2.

Provides:

- ``MegatronFSDPStateful`` — ``Stateful`` wrapper that integrates with
  PyTorch DCP, handles uneven DTensor chunk metadata via ``get_state_dict``,
  and applies MCore post-processing.
- Post-processing functions for Megatron FSDP v2 state dicts:
  ``handle_swiglu_in_state_dict_v2``, ``handle_gdn_in_state_dict_v2``,
  ``handle_mamba_in_state_dict_v2``, ``handle_experts_in_state_dict``,
  ``handle_fp8_extra_state_case``.
"""

import logging
import re
from collections.abc import Mapping
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import set_state_dict as _set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Shard

import megatron.core.parallel_state as mpu
from megatron.core import dist_checkpointing
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import copy_chunk_metadata
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import get_chunk_meta_source
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict as _get_state_dict,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    make_uneven_dtensor,
    preprocess_state_dict_for_uneven_dtensor,
    redistribute_uneven_dtensor_to_replicated,
    split_dtensor,
)
from megatron.core.transformer.transformer_layer import TransformerLayer

try:
    from megatron.core.ssm.mamba_mixer import MambaMixer
except ImportError:
    MambaMixer = None

logger = logging.getLogger(__name__)

__all__ = [
    "MegatronFSDPStateful",
    "add_module_prefix",
    "strip_module_prefix",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "_build_dtensor_optim_sd",
    "handle_fp8_extra_state_case",
    "handle_experts_in_state_dict",
    "handle_swiglu_in_state_dict_v2",
    "handle_gdn_in_state_dict_v2",
    "handle_mamba_in_state_dict_v2",
]

_MODULE_PREFIX = "module."


# ------------------------------------------------------------------
# Stateful wrapper
# ------------------------------------------------------------------


class MegatronFSDPStateful(Stateful):
    """Stateful wrapper for Megatron FSDP v2 model + optimizer checkpointing.

    Implements DCP's ``Stateful`` protocol.  ``state_dict()`` uses
    ``get_state_dict`` from ``uneven_dtensor`` (which preprocesses DTensors
    for uneven sharding and produces FQN-keyed optimizer states), then applies
    MCore post-processing (SwiGLU, GDN, MambaMixer, FP8, expert remapping — see
    individual ``handle_*_v2`` functions).  ``load_state_dict()`` uses PyTorch's
    ``set_state_dict``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        args: Optional[object] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def state_dict(self):
        if self.optimizer is not None:
            model_sd, optim_sd = _get_state_dict(self.model, self.optimizer)
        else:
            model_sd = self.model.state_dict()
            preprocess_state_dict_for_uneven_dtensor(model_sd)
            optim_sd = None

        state_dict = {"model": model_sd}
        if optim_sd is not None:
            state_dict["optimizer"] = optim_sd

        if self.args is not None:
            state_dict = _apply_mcore_postprocess(state_dict, self.args, self.model)
        return state_dict

    def load_state_dict(self, state_dict):
        _set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict.get("model"),
            optim_state_dict=state_dict.get("optimizer"),
        )


# ------------------------------------------------------------------
# FP8 artifact cleanup (format-agnostic)
# ------------------------------------------------------------------


def handle_fp8_extra_state_case(model_state_dict: dict) -> None:
    """Remove ``._extra_state`` keys (artifact of FP8 training)."""
    keys_to_remove = [k for k in model_state_dict if k.endswith("._extra_state")]
    for k in keys_to_remove:
        del model_state_dict[k]


# ------------------------------------------------------------------
# Expert key remapping (format-agnostic)
# ------------------------------------------------------------------


def handle_experts_in_state_dict(model_state_dict: dict, num_experts: int) -> dict:
    """Rename expert keys to reflect expert-parallel sharding.

    Standalone implementation (no dependency on ``fsdp_dtensor_checkpoint``).
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    local_expert_start = ep_rank * (num_experts // ep_size) if num_experts else 0
    local_expert_end = local_expert_start + (num_experts // ep_size) if num_experts else 0

    def _should_keep(expert_index):
        if expert_index is None:
            return True
        return local_expert_start <= expert_index < local_expert_end

    def _replace(key, expert_index, sd):
        new_idx = expert_index + local_expert_start
        if new_idx == expert_index and not _should_keep(expert_index):
            logger.warning(
                "Identity transform for non-local expert key '%s' (expert_index=%d, "
                "local_expert_start=%d). This expert does not belong to EP rank %d "
                "but survived in the state dict. Consider removing it instead.",
                key, expert_index, local_expert_start, ep_rank,
            )
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            if key.endswith('_w') or key.endswith('_v'):
                new_key = key.replace(
                    f'weight{expert_index}{key[-2:]}', f'weight{new_idx}{key[-2:]}'
                )
            else:
                new_key = key.replace(f'weight{expert_index}', f'weight{new_idx}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(f'local_experts.{expert_index}.', f'local_experts.{new_idx}.')
        else:
            raise ValueError(f"Unexpected expert key format: {key}")
        sd[new_key] = sd.pop(key)

    model_state_dict = model_state_dict.copy()
    for key in list(model_state_dict.keys()):
        ei = _get_expert_index_from_key(key)
        if not _should_keep(ei):
            _replace(key, ei, model_state_dict)

    return model_state_dict


def _get_expert_index_from_key(key: str):
    """Extract expert index from key (``weight{N}`` or ``local_experts.N.``)."""
    m = re.search(r'(?:weight(\d+)$)|(?:local_experts\.(\d+)\.)', key)
    if m:
        return int(m.group(1) or m.group(2))
    return None


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _strip_wrappers(path: str) -> str:
    """Strip DDP/FSDP wrapper prefixes (``module.``, ``model.``) from a path."""
    parts = path.split('.')
    while parts and parts[0] in ('module', 'model'):
        parts = parts[1:]
    return '.'.join(parts)


def _get_dist_param(model: nn.Module, key: str) -> nn.Parameter:
    """Get a model parameter by state dict key, handling ``module.`` prefix.

    Tries multiple key variants to handle both wrapped (``FullyShardedDataParallel``
    with ``self.module``) and unwrapped (raw ``FSDPModule``) models:
      1. Key as-is.
      2. Key with ``module.`` prefix added.
      3. Key with ``module.`` prefix stripped (if present).
    """
    try:
        return model.get_parameter(key)
    except AttributeError:
        pass
    try:
        return model.get_parameter(f"module.{key}")
    except AttributeError:
        pass
    if key.startswith("module."):
        try:
            return model.get_parameter(key[len("module.") :])
        except AttributeError:
            pass
    raise AttributeError(
        f"Parameter '{key}' not found in model "
        f"(tried as-is, with 'module.' prefix, and without 'module.' prefix)"
    )


def _detect_glu_layers(model: nn.Module) -> dict:
    """Return ``{layer_path: uses_gated_linear_unit}`` for TransformerLayers."""
    _layer_glu = {}
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            _layer_glu[_strip_wrappers(name)] = getattr(module.config, 'gated_linear_unit', False)
    return _layer_glu


def _key_in_glu_layer(key: str, layer_glu: dict) -> bool:
    """Return True if *key* belongs to a TransformerLayer with GLU enabled."""
    norm_key = _strip_wrappers(key)
    best_glu, best_len = None, -1
    for layer_path, uses_glu in layer_glu.items():
        if norm_key.startswith(layer_path + '.') and len(layer_path) > best_len:
            best_glu, best_len = uses_glu, len(layer_path)
    if best_glu is None:
        return True  # no TransformerLayer found — assume GLU for backward compat
    return best_glu


# ------------------------------------------------------------------
# Fused-parameter splitting — shared skeleton (Megatron FSDP v2)
# ------------------------------------------------------------------
#
# SwiGLU and GDN both require splitting fused parameter tensors into
# per-component pieces (SwiGLU: fc1 → _w/_v; GDN: linear_qkv → query/key/value).
# This function handles both.
#
# The caller provides a *detector* that returns ``(sizes, names, dim)`` for
# parameters that need splitting, or ``None`` otherwise, and a *key formatter*
# to build sub-parameter names.

_DEFAULT_STATE_KEYS = ("exp_avg", "exp_avg_sq")


def _split_fused_params_v2(
    model: nn.Module,
    model_state_dict: dict,
    optimizer_state_dict: Optional[dict],
    detector,
    key_fmt,
    tag: str,
    state_keys: Tuple[str, ...] = _DEFAULT_STATE_KEYS,
) -> Tuple[dict, Optional[dict]]:
    """Split fused DTensor parameters in model and optimizer state dicts.

    Args:
        model: The model (for ``_get_dist_param``).
        model_state_dict: Model state dict (a copy is returned).
        optimizer_state_dict: Optional optimizer state dict (a copy is returned).
        detector(key, dtensor, model) -> ``(sizes, names, dim)`` or ``None``.
        key_fmt(key, sub_name) -> new_key.
        tag: Log tag (e.g. ``"SwiGLU"``).
        state_keys: Optimizer state keys to split (default: ``exp_avg``, ``exp_avg_sq``).

    Returns:
        ``(model_state_dict, optimizer_state_dict)`` — modified copies.
    """
    # ---- Model state dict ----
    model_state_dict = model_state_dict.copy()
    split_count = 0
    for key in list(model_state_dict.keys()):
        value = model_state_dict[key]
        match = detector(key, value, model)
        if match is None:
            continue
        sizes, names, dim = match

        dist_param = _get_dist_param(model, key)
        assert isinstance(
            dist_param, DTensor
        ), f"Expected DTensor for {key}, got {type(dist_param).__name__}"
        sub_tensors = split_dtensor(value, sizes, dim)
        for sub_name, tensor in zip(names, sub_tensors):
            model_state_dict[key_fmt(key, sub_name)] = tensor
        del model_state_dict[key]
        split_count += 1

    if split_count > 0:
        logger.info(f"[{tag}] Split {split_count} fused keys.")

    # ---- Optimizer state dict ----
    if optimizer_state_dict is not None:
        optimizer_state_dict = optimizer_state_dict.copy()
        if len(optimizer_state_dict.get("state", {})) != 0:
            opt_state = optimizer_state_dict["state"]
            new_opt_state = {}
            for key in list(opt_state.keys()):
                param_key = key
                if param_key.startswith("module."):
                    param_key = param_key[len("module.") :]
                dist_param = _get_dist_param(model, param_key)

                match = detector(key, dist_param, model)
                if match is None:
                    new_opt_state[key] = opt_state[key]
                    continue
                sizes, names, dim = match

                for sub_name in names:
                    new_opt_state[key_fmt(key, sub_name)] = opt_state[key].copy()
                for sk in state_keys:
                    sub_tensors = split_dtensor(opt_state[key][sk], sizes, dim)
                    for sub_name, tensor in zip(names, sub_tensors):
                        new_opt_state[key_fmt(key, sub_name)][sk] = tensor
            optimizer_state_dict["state"] = new_opt_state

    return model_state_dict, optimizer_state_dict


# ------------------------------------------------------------------
# SwiGLU key patterns and helpers
# ------------------------------------------------------------------

_SWIGLU_KEY_PATTERNS = [
    r"(.*)\.mlp\.linear_fc1\.weight$",
    r"(.*)\.mlp\.linear_fc1\.bias$",
    r"(.*)\.mlp\.experts\.linear_fc1\.weight(\d+)$",
    r"(.*)\.mlp\.experts\.linear_fc1\.bias(\d+)$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight$",
    r"(.*)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.bias$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.weight$",
    r"(.*)\.mlp\.shared_experts\.linear_fc1\.bias$",
]


def _is_swiglu_key(key: str) -> bool:
    return any(re.search(pat, key) for pat in _SWIGLU_KEY_PATTERNS)


def _swiglu_detector(key, dtensor, model, layer_glu):
    if not _is_swiglu_key(key):
        return None
    if not _key_in_glu_layer(key, layer_glu):
        return None
    dim = 0
    assert (
        dtensor.shape[dim] % 2 == 0
    ), f"Expected SwiGLU fc1 weight size divisible by 2, got {dtensor.shape[dim]}"
    half = dtensor.shape[dim] // 2
    return ([half, half], ["w", "v"], dim)


def handle_swiglu_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split SwiGLU fc1 parameters in model and optimizer state dicts.

    Megatron FSDP v2 version — only processes layers with
    ``gated_linear_unit=True``.  Delegates to :func:`_split_fused_params_v2`.
    """
    layer_glu = _detect_glu_layers(model)

    def detector(key, dtensor, _model):
        return _swiglu_detector(key, dtensor, _model, layer_glu)

    return _split_fused_params_v2(
        model,
        model_state_dict,
        optimizer_state_dict,
        detector,
        lambda k, s: f"{k}_{s}",
        "SwiGLU",
    )


# ------------------------------------------------------------------
# GDN key patterns and helpers
# ------------------------------------------------------------------

GDN_CONV1D_NAMES = ["query", "key", "value"]

_GDN_KEY_PATTERNS = [
    r"(.*)\.self_attention\.linear_proj\.weight$",
    r"(.*)\.self_attention\.linear_qkv\.weight$",
    r"(.*)\.self_attention\.linear_qkv\.bias$",
]


def _match_gdn_key(key: str, dtensor: DTensor):
    for pat in _GDN_KEY_PATTERNS:
        m = re.match(pat, key)
        if m:
            dim = 0
            size = dtensor[dim]
            assert (
                size % 3 == 0
            ), f"Expected GDN projection size divisible by 3, got {size} for key {key}"
            qkv_size = size // 3
            return [qkv_size, qkv_size, qkv_size], GDN_CONV1D_NAMES, dim
    return None


def handle_gdn_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split fused GDN projection parameters into per-component DTensors.

    Megatron FSDP v2 version.  Delegates to :func:`_split_fused_params_v2`.
    """
    return _split_fused_params_v2(
        model,
        model_state_dict,
        optimizer_state_dict,
        _match_gdn_key,
        lambda k, s: f"{k}.{s}",
        "GDN v2",
    )


# ------------------------------------------------------------------
# MambaMixer key patterns and helpers
# ------------------------------------------------------------------

_MAMBA_MIXER_IN_PROJ_NAMES = ["z", "x", "B", "C", "dt"]
_MAMBA_MIXER_CONV1D_NAMES = ["x", "B", "C"]

_MAMBA_MIXER_KEY_PATTERNS = [
    r"(.*)\.mixer\.in_proj\.weight$",
    r"(.*)\.mixer\.conv1d\.weight$",
    r"(.*)\.mixer\.conv1d\.bias$",
]


def _detect_mamba_mixers(model: nn.Module) -> dict:
    """Return ``{layer_path: MambaMixer_module}`` for MambaMixer layers."""
    if MambaMixer is None:
        return {}
    _mixers = {}
    for name, module in model.named_modules():
        if isinstance(module, MambaMixer):
            _mixers[_strip_wrappers(name)] = module
    return _mixers


def _mamba_mixer_detector(key, dtensor, model, mixer_map):
    """Detector for MambaMixer fused parameters.

    Returns ``(sizes, names, dim)`` for keys matching ``mixer.in_proj.*``
    or ``mixer.conv1d.*``, using the TP-local dimensions from the owning
    ``MambaMixer`` module.  Returns ``None`` for non-mamba keys.
    """
    if not _MAMBA_MIXER_KEY_PATTERNS or MambaMixer is None:
        return None
    dim = 0
    for pat in _MAMBA_MIXER_KEY_PATTERNS:
        m = re.match(pat, key)
        if not m:
            continue
        prefix = m.group(1)
        mixer_module = mixer_map.get(prefix)
        if mixer_module is None:
            # Strip module. prefix and retry
            alt = prefix
            while alt.startswith(_MODULE_PREFIX):
                alt = alt[len(_MODULE_PREFIX):]
                mixer_module = mixer_map.get(alt)
                if mixer_module is not None:
                    break
        if mixer_module is None:
            return None
        if "in_proj.weight" in key:
            sizes = [
                mixer_module.d_inner_local_tp,
                mixer_module.d_inner_local_tp,
                mixer_module.ngroups_local_tp * mixer_module.d_state,
                mixer_module.ngroups_local_tp * mixer_module.d_state,
                mixer_module.nheads_local_tp,
            ]
            return (sizes, _MAMBA_MIXER_IN_PROJ_NAMES, dim)
        if "conv1d.weight" in key or "conv1d.bias" in key:
            sizes = [
                mixer_module.d_inner_local_tp,
                mixer_module.ngroups_local_tp * mixer_module.d_state,
                mixer_module.ngroups_local_tp * mixer_module.d_state,
            ]
            return (sizes, _MAMBA_MIXER_CONV1D_NAMES, dim)
    return None


def handle_mamba_in_state_dict_v2(
    model: nn.Module, model_state_dict: dict, optimizer_state_dict: Optional[dict]
) -> Tuple[dict, Optional[dict]]:
    """Split fused MambaMixer parameters into per-component DTensors.

    Splits ``mixer.in_proj.weight`` → ``.z`` / ``.x`` / ``.B`` / ``.C`` / ``.dt``
    and ``mixer.conv1d.{weight,bias}`` → ``.x`` / ``.B`` / ``.C``, using
    TP-local dimensions read from each ``MambaMixer`` module.
    Delegates to :func:`_split_fused_params_v2`.
    """
    if MambaMixer is None:
        return model_state_dict, optimizer_state_dict
    mixer_map = _detect_mamba_mixers(model)
    if not mixer_map:
        return model_state_dict, optimizer_state_dict

    def detector(key, dtensor, _model):
        return _mamba_mixer_detector(key, dtensor, _model, mixer_map)

    return _split_fused_params_v2(
        model,
        model_state_dict,
        optimizer_state_dict,
        detector,
        lambda k, s: f"{k}.{s}",
        "MambaMixer",
    )


# ------------------------------------------------------------------
# Unified post-processing
# ------------------------------------------------------------------


def _find_param_in_map(key: str, param_map: dict) -> Optional[DTensor]:
    """Look up *key* in *param_map*, trying ``module.`` prefix variants."""
    param = param_map.get(key)
    if param is not None:
        return param
    stripped = key
    while stripped.startswith(_MODULE_PREFIX):
        stripped = stripped[len(_MODULE_PREFIX) :]
        param = param_map.get(stripped)
        if param is not None:
            return param
    return param_map.get(f"{_MODULE_PREFIX}{key}")


def _propagate_chunk_metadata_to_state_dict(model: nn.Module, state_dict: dict) -> None:
    """Copy chunk metadata from model parameters to state dict DTensors.

    ``model.state_dict()`` returns fresh DTensor objects that lack
    ``__create_chunk_list__`` / ``__create_write_items__``.  The model
    parameters (from ``named_parameters()``) already have them.  This
    function copies the closures locally — no collectives.
    """
    param_map = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor) and hasattr(param._local_tensor, "__create_chunk_list__"):
            param_map[name] = param

    missing_src_metadata = []
    for key, value in state_dict.items():
        if not isinstance(value, DTensor):
            continue
        param = _find_param_in_map(key, param_map)
        if param is not None:
            copy_chunk_metadata(param, value)
        else:
            has_meta = hasattr(value._local_tensor, "__create_chunk_list__")
            missing_src_metadata.append((key, value.shape, value._local_tensor.shape, has_meta))

    if missing_src_metadata:
        if torch.distributed.get_rank() == 0:
            logger.warning(
                "[chunk_metadata_diag] _propagate_chunk_metadata_to_state_dict: "
                f"{len(missing_src_metadata)} DTensor(s) in state_dict could NOT be matched to "
                "a model parameter with __create_chunk_list__. Their metadata (if any) comes "
                "from preprocess_state_dict_for_uneven_dtensor only."
            )
            for key, global_shape, local_shape, has_meta in missing_src_metadata:
                logger.warning(
                    f"  key={key} global_shape={tuple(global_shape)} "
                    f"local_shape={tuple(local_shape)} has_create_chunk_list={has_meta}"
                )


def _apply_mcore_postprocess(raw_state_dict, args, model):
    """Apply MCore-specific state dict post-processing.

    Copies *raw_state_dict*, wraps optimizer states as DTensors, then
    applies FP8 cleanup, SwiGLU/GDN/MambaMixer split, and expert key remapping.
    The original *raw_state_dict* is not mutated.
    """
    state_dict = raw_state_dict.copy()
    _propagate_chunk_metadata_to_state_dict(model, state_dict["model"])

    if "optimizer" in state_dict:
        state_dict["optimizer"] = _build_dtensor_optim_sd(state_dict["optimizer"], model)
    handle_fp8_extra_state_case(state_dict["model"])

    if getattr(args, "swiglu", False):
        opt_sd = state_dict.get("optimizer")
        model_sd, new_opt = handle_swiglu_in_state_dict_v2(model, state_dict["model"], opt_sd)
        state_dict["model"] = model_sd
        if new_opt is not None:
            state_dict["optimizer"] = new_opt

    if getattr(args, "gdn", False):
        opt_sd = state_dict.get("optimizer")
        model_sd, new_opt = handle_gdn_in_state_dict_v2(model, state_dict["model"], opt_sd)
        state_dict["model"] = model_sd
        if new_opt is not None:
            state_dict["optimizer"] = new_opt

    opt_sd = state_dict.get("optimizer")
    model_sd, new_opt = handle_mamba_in_state_dict_v2(model, state_dict["model"], opt_sd)
    state_dict["model"] = model_sd
    if new_opt is not None:
        state_dict["optimizer"] = new_opt

    num_experts = getattr(args, "num_experts", None)
    if num_experts:
        state_dict["model"] = handle_experts_in_state_dict(state_dict["model"], num_experts)
        if "optimizer" in state_dict:
            optim_sd = state_dict["optimizer"]
            optim_sd["state"] = handle_experts_in_state_dict(optim_sd["state"], num_experts)
            optim_sd["param_to_group_meta"] = handle_experts_in_state_dict(
                optim_sd["param_to_group_meta"], num_experts
            )

    flattened_sd = flatten_state_dict_keys(state_dict)
    _verify_chunk_metadata(flattened_sd)

    return state_dict


def _numel(shape: tuple) -> int:
    """Return the product of all dimensions in *shape*."""
    n = 1
    for d in shape:
        n *= d
    return n


def _verify_chunk_metadata(flattened_sd: dict) -> None:
    """Verify every DTensor has correct ``__create_chunk_list__`` metadata.

    Checks both existence AND consistency (chunks total numel must match
    local tensor numel).  On failure, prints diagnostic information
    including the metadata source tag and shape details.
    """
    failures = []
    for key, value in flattened_sd.items():
        if not isinstance(value, DTensor):
            continue
        lt = value._local_tensor
        if not hasattr(lt, "__create_chunk_list__"):
            failures.append(
                f"MISSING metadata: key={key} global_shape={tuple(value.shape)} "
                f"local_shape={tuple(lt.shape)} source={get_chunk_meta_source(value)}"
            )
            continue

        cl = lt.__create_chunk_list__()
        cl_total = sum(_numel(c.sizes) for c in cl)
        local_numel = lt.numel()
        if cl_total != local_numel:
            cl_detail = [(tuple(c.offsets), tuple(c.sizes)) for c in cl]
            failures.append(
                f"NUMEL MISMATCH: key={key} "
                f"global_shape={tuple(value.shape)} "
                f"local_shape={tuple(lt.shape)} "
                f"local_numel={local_numel} "
                f"chunks_total={cl_total} "
                f"chunk_list={cl_detail} "
                f"source={get_chunk_meta_source(value)} "
                f"device_mesh={value.device_mesh}"
            )

    if failures:
        logger.error(
            "[chunk_metadata_verify] %d DTensor(s) have invalid chunk metadata:",
            len(failures),
        )
        for msg in failures:
            logger.error("  %s", msg)
        raise AssertionError(
            f"{len(failures)} DTensor(s) have invalid chunk metadata. "
            "See log above for details."
        )


def flatten_state_dict_keys(state_dict, parent_key="", sep="."):
    """
    Recursively flatten nested mappings inside a state_dict.

    Returns a dict: { "flat.key": value }
    """
    items = {}
    for k, v in state_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            # Recurse into nested dicts (e.g., when you saved extra metadata)
            items.update(flatten_state_dict_keys(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ------------------------------------------------------------------
# Model state dict prefix alignment
# ------------------------------------------------------------------


def add_module_prefix(state_dict: dict) -> dict:
    """Add ``module.`` prefix to all keys in a state dict.

    Megatron FSDP v2 applies ``fully_shard`` directly to the model without
    a ``MegatronFSDP`` wrapper, so ``model.state_dict()`` produces keys
    without the ``module.`` prefix (e.g., ``embedding.word_embeddings.weight``).
    Megatron's checkpoint format expects the prefix to be present. This
    function adds it for alignment.
    """
    return {f"{_MODULE_PREFIX}{k}": v for k, v in state_dict.items()}


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove ``module.`` prefix from all keys in a state dict.

    Inverse of :func:`add_module_prefix`. Used when loading a Megatron
    checkpoint (which stores keys with ``module.`` prefix) back into a
    Megatron FSDP v2 model that does not use the prefix.
    """
    return {
        k[len(_MODULE_PREFIX) :] if k.startswith(_MODULE_PREFIX) else k: v
        for k, v in state_dict.items()
    }


def _model_has_module_prefix(model: nn.Module) -> bool:
    """Detect whether the model's state dict keys already carry ``module.`` prefix."""
    for name, _ in model.named_parameters():
        return name.startswith(_MODULE_PREFIX)
    return False


# ------------------------------------------------------------------
# Model state dict (Megatron FSDP v2)
# ------------------------------------------------------------------


def get_model_state_dict(model: nn.Module) -> dict:
    """Get model state dict with ``module.`` prefix for Megatron FSDP v2.

    If the model's parameters already carry the ``module.`` prefix (e.g.,
    when the model is accessed through the ``FullyShardedDataParallel``
    adapter), the state dict is returned as-is. Otherwise the prefix is
    added to align with Megatron's checkpoint key convention.

    Returns:
        Model state dict with ``module.``-prefixed keys containing DTensors.
    """
    model_sd = model.state_dict()
    if not _model_has_module_prefix(model):
        model_sd = add_module_prefix(model_sd)
    return model_sd


# ------------------------------------------------------------------
# Optimizer state dict — Path A (sharded_param_state_fsdp_dtensor)
# ------------------------------------------------------------------


def get_optimizer_state_dict(optimizer, is_loading: bool = False) -> Optional[dict]:
    """Get optimizer state dict following Path A (``sharded_param_state_fsdp_dtensor``).

    Delegates to ``optimizer.sharded_state_dict()`` with the ``fsdp_dtensor``
    sharding type, which internally calls ``sharded_param_state_fsdp_dtensor``.

    For loading, this also initializes optimizer states with dummy values so
    that the returned dict has correctly-sized tensors for DCP in-place load.

    Args:
        optimizer: A ``DistributedOptimizer`` (or compatible) instance.
        is_loading: If True, pre-allocates optimizer state tensors.

    Returns:
        Optimizer state dict with structure::

            {
                "state": {<param_name>: {"exp_avg": DTensor, "exp_avg_sq": DTensor, ...}},
                "param_to_group_meta": {<param_name>: {"lr": ..., "weight_decay": ...}},
            }

        Returns None if ``optimizer`` is None or is a stub optimizer.
    """
    if optimizer is None:
        return None
    if getattr(optimizer, "is_stub_optimizer", False):
        return None
    return optimizer.sharded_state_dict(
        model_sharded_state_dict={},
        is_loading=is_loading,
        metadata={"distrib_optim_sharding_type": "fsdp_dtensor"},
    )


# ------------------------------------------------------------------
# Optimizer state DTensor wrapping
# ------------------------------------------------------------------


def _maybe_wrap_as_uneven_dtensor(tensor, dist_param: DTensor):
    """Wrap a plain tensor as an uneven DTensor if it matches the param's local shard.

    FusedAdam stores ``exp_avg`` / ``exp_avg_sq`` as plain tensors; DCP needs
    DTensors.  Returns the original tensor unchanged if it is already a DTensor
    or its shape does not match the parameter's local shard.
    """
    if isinstance(tensor, DTensor):
        dt = tensor
    else:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.shape != dist_param._local_tensor.shape:
            return tensor
        dt = make_uneven_dtensor(
            tensor,
            shape=dist_param.size(),
            dp_mesh=dist_param.device_mesh,
            placements=dist_param.placements,
        )
    copy_chunk_metadata(dist_param, dt)
    return dt


def _build_dtensor_optim_sd(raw_opt_state_dict: dict, model: nn.Module) -> dict:
    """Return a copy of *raw_opt_state_dict* with optimizer state tensors converted to DTensors.

    FusedAdam stores ``exp_avg`` / ``exp_avg_sq`` as plain tensors matching
    the parameter's local DTensor shard.  DCP requires all sharded data to
    be DTensors.  This returns a new dict where those plain tensors are
    converted to uneven DTensors using the model parameter's mesh/placements.
    *raw_opt_state_dict* is **not** mutated.
    """
    opt_state_dict = raw_opt_state_dict.copy()
    if "state" not in opt_state_dict:
        return opt_state_dict

    param_map = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            param_map[name] = param

    if not param_map:
        return opt_state_dict

    old_state = opt_state_dict["state"]
    new_state = {}
    for key, param_states in old_state.items():
        if not isinstance(param_states, dict):
            new_state[key] = param_states
            continue
        new_param_states = dict(param_states)
        dist_param = _find_param_in_map(key, param_map)
        if dist_param is not None:
            for state_key, state_val in new_param_states.items():
                new_param_states[state_key] = _maybe_wrap_as_uneven_dtensor(state_val, dist_param)
        new_state[key] = new_param_states

    opt_state_dict["state"] = new_state
    return opt_state_dict


def _preprocess_and_verify_v2_state_dict(v2_state_dict, model=None):
    """Preprocess and verify the Megatron FSDP v2 state dict before DCP loading.

    DCP requires DTensors for distributed load, but ``optimizer.load_state_dict``
    only accepts plain tensors.  This function builds a *shadow* optimizer state
    dict (``v2_optim_state``) where plain tensors are wrapped as uneven DTensors
    sharing storage with the originals.  DCP loads into this shadow dict, and the
    data lands in the original plain tensors via shared storage.  The original
    ``v2_state_dict`` is **not** mutated.

    Also verifies that every model and shadow optimizer DTensor has
    ``__create_chunk_list__`` and ``__create_write_items__`` metadata.

    Args:
        v2_state_dict: The v2 state dict (from ``_build_megatron_fsdp_v2_state_dict``).
        model: The FSDP model (for chunk metadata propagation).  If ``None``,
            metadata propagation is skipped.

    Returns:
        v2_by_canonical: ``{canonical_name: model_DTensor}`` mapping.
        v2_optim_state: ``{canonical_name: {state_key: DTensor}}`` shadow dict.
    """
    v2_model = v2_state_dict["model"]
    v2_optim_state_raw = v2_state_dict.get("optimizer", {}).get("state", {})

    # ---- Propagate chunk metadata from model to state dict DTensors ----
    if model is not None:
        _propagate_chunk_metadata_to_state_dict(model, v2_model)

    # ---- Strip ``module.`` prefix from optimizer state keys ----
    # Copy inner dicts so DTensor wrapping below does not mutate
    # the original state dict (optimizer.load_state_dict needs plain tensors).
    v2_optim_state = {}
    for k, v in v2_optim_state_raw.items():
        canonical = k
        while canonical.startswith("module."):
            canonical = canonical[len("module.") :]
        v2_optim_state[canonical] = dict(v)

    # ---- Build canonical model param map + param lookup ----
    v2_by_canonical = {}
    param_map = {}
    for v2_key, v2_val in v2_model.items():
        canonical = v2_key
        while canonical.startswith("module."):
            canonical = canonical[len("module.") :]
        v2_by_canonical[canonical] = v2_val
        if isinstance(v2_val, DTensor):
            param_map[canonical] = v2_val

    # ---- Wrap plain tensors as uneven DTensors ----
    for param_name, states in dict(v2_optim_state).items():
        dist_param = param_map.get(param_name)
        if dist_param is None:
            continue
        for sk, sv in dict(states).items():
            v2_optim_state[param_name][sk] = _maybe_wrap_as_uneven_dtensor(sv, dist_param)

    # ---- Verify model DTensors have chunk metadata ----
    for canonical, v2_val in v2_by_canonical.items():
        if hasattr(v2_val, "_local_tensor"):
            lt = v2_val._local_tensor
            assert hasattr(lt, "__create_chunk_list__"), (
                f"Expected v2 model DTensor '{canonical}' to have "
                f"__create_chunk_list__ metadata"
            )
            assert hasattr(lt, "__create_write_items__"), (
                f"Expected v2 model DTensor '{canonical}' to have "
                f"__create_write_items__ metadata"
            )

    # ---- Verify optimizer state DTensors have chunk metadata ----
    # Also propagate metadata for states that are already DTensors
    # (_maybe_wrap_as_uneven_dtensor only copies metadata when wrapping
    # plain tensors, not for pre-existing DTensors).
    for param_name, states in v2_optim_state.items():
        dist_param = param_map.get(param_name)
        for sk, sv in states.items():
            if (
                isinstance(sv, DTensor)
                and dist_param is not None
                and hasattr(dist_param._local_tensor, "__create_chunk_list__")
            ):
                copy_chunk_metadata(dist_param, sv)
            if hasattr(sv, "_local_tensor"):
                lt = sv._local_tensor
                assert hasattr(lt, "__create_chunk_list__"), (
                    f"Expected optimizer state DTensor '{param_name}.{sk}' "
                    f"to have __create_chunk_list__ metadata"
                )
                assert hasattr(lt, "__create_write_items__"), (
                    f"Expected optimizer state DTensor '{param_name}.{sk}' "
                    f"to have __create_write_items__ metadata"
                )

    return v2_by_canonical, v2_optim_state


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 checkpoint key normalization
# ------------------------------------------------------------------


def normalize_torch_dist_key(key: str) -> str:
    """Normalize a torch_dist checkpoint key to v2 canonical form.

    Maps structural naming differences:
    - ``transformer_layer`` → ``mtp_model_layer``
    """
    if ".transformer_layer." in key:
        key = key.replace(".transformer_layer.", ".mtp_model_layer.")
    return key


def reverse_normalize_torch_dist_key(key: str) -> str:
    """Reverse the v2 canonical key back to torch_dist naming.

    Inverse of :func:`normalize_torch_dist_key`:
    - ``mtp_model_layer`` → ``transformer_layer``
    """
    key = key.replace(".mtp_model_layer", ".transformer_layer")
    return key


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 name mapping
# ------------------------------------------------------------------


# Regex patterns for detecting fused vs per-layer/per-expert keys.
# Torch_dist fuses across layers: <prefix>.layers.{field}  (no layer index)
# v2 has per-layer:               <prefix>.layers.{N}.{field}
# Torch_dist fuses experts:       <prefix>.layers.{field}.mlp.experts.experts.{fc}.weight
# v2 has per-expert:              <prefix>.layers.{N}.{field}.mlp.experts.{fc}.weight{M}
_LAYERED_KEY_RE = re.compile(r'^(.+\.)?layers\.(\d+)\.(.+)$')
_EXPERT_V2_KEY_RE = re.compile(r'(.+\.)?mlp\.experts\.linear_fc([12])\.weight(\d+)$')
_SEQ_EXPERT_FIELD_RE = re.compile(
    r'(?:.+\.)?mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight$'
)


def _strip_module_prefix(key: str) -> str:
    """Strip leading 'module.' from a key string."""
    return key[len("module.") :] if key.startswith("module.") else key


def _build_torch_dist_to_v2_map(metadata, v2_by_canonical, v2_optim_state):
    """Build maps from torch_dist checkpoint keys to Megatron FSDP v2 DTensors.

    Automatically handles fused layer tensors by matching
    ``decoder.layers.{N}.{field}`` (v2) against
    ``decoder.layers.{field}`` (torch_dist) — no hardcoded field paths.

    Returns:
        regular_model: ``{td_key: v2_DTensor}`` for 1:1 params.
        fused_layer_groups: ``{td_key: [(v2_key, v2_val, layer_idx), ...]}``
            for fused layer params to be loaded/sliced in Phase 4.
        hi_prec_model: ``{td_param_name: v2_DTensor}`` for high-precision copies.
        optim_keys: ``{td_key: v2_state_DTensor}`` for 1:1 optimizer states.
        optim_matched: ``set`` of canonical param names with matched optimizer states.
    """
    regular_model = {}
    fused_layer_groups = {}
    hi_prec_td_keys = set()
    hi_prec_model = {}
    optim_keys = {}
    optim_matched = set()

    # ------------------------------------------------------------------
    # Helper: given a v2 canonical key with layer index, derive the
    # torch_dist fused key (if the fused key exists in metadata).
    # ------------------------------------------------------------------
    def _match_fused_key(v2_canonical, value=None):
        """Return match info for a v2 key whose fused counterpart exists in metadata.

        Returns:
            ``(td_key, layer_idx)`` for regular fused-layer keys and GroupedMLP keys.
            ``(td_key, local_expert_idx, "seq_mlp")`` for SequentialMLP expert keys.
            ``None`` if no fused counterpart was found.
        """
        m = _LAYERED_KEY_RE.match(v2_canonical)
        if not m:
            return None
        prefix = m.group(1) or ""  # e.g. "decoder." or ""
        layer_idx = int(m.group(2))
        field = m.group(3)

        # ---- Check SequentialMLP expert format first ----
        # v2:   ``{sub_layer}mlp.experts.local_experts.{N}.linear_fc{1|2}.weight``
        #       (sub_layer may be empty or e.g. ``mtp_model_layer.`` for MTP)
        # td:   ``{prefix}layers.{layer_idx}.{sub_layer}mlp.experts.experts.linear_fc{1|2}.weight``
        seq_exp_m = _SEQ_EXPERT_FIELD_RE.match(field)
        if seq_exp_m:
            local_expert_idx = int(seq_exp_m.group(1))
            fc_type = seq_exp_m.group(2)
            # Extract sub-layer prefix (if any) before ``mlp.experts.local_experts``
            sub_layer_prefix = ""
            if not field.startswith("mlp.experts.local_experts."):
                idx = field.find("mlp.experts.local_experts.")
                if idx > 0:
                    sub_layer_prefix = field[:idx]  # e.g. "mtp_model_layer."
            td_base = (
                f"{prefix}layers.{layer_idx}."
                f"{sub_layer_prefix}mlp.experts.experts.{fc_type}.weight"
            )
            td_base = reverse_normalize_torch_dist_key(td_base)
            candidates = [td_base]
            candidates.append(f"model.{td_base}")
            candidates.append(f"module.{td_base}")
            candidates.append(f"model.module.{td_base}")
            for c in candidates:
                if c in metadata:
                    return c, local_expert_idx, "seq_mlp"
            return None

        # ---- Build fused key (strip layer index) ----
        # v2:   ``{prefix}layers.{N}.{field}``
        # td:   ``{prefix}layers.{field}`` (no layer index)
        fused_base = f"{prefix}layers.{field}"

        candidates = [fused_base]
        # Torch_dist metadata may have "model." prefix
        candidates.append(f"model.{fused_base}")
        candidates.append(f"module.{fused_base}")
        candidates.append(f"model.module.{fused_base}")

        # For GroupedMLP expert params: v2 uses "experts.linear_fc1.weight{N}"
        # torch_dist uses "experts.experts.linear_fc1.weight"
        exp_m = _EXPERT_V2_KEY_RE.search(field)
        if exp_m:
            fc_type = exp_m.group(2)
            field_exp = (
                _EXPERT_V2_KEY_RE.sub(rf"mlp.experts.experts.linear_fc\2.weight", field).rstrip(
                    ".weight"
                )
                + ".weight"
            )
            fused_exp = f"{prefix}layers.{field_exp}"
            candidates.append(fused_exp)
            candidates.append(f"model.{fused_exp}")

        for c in candidates:
            if c in metadata:
                return c, layer_idx

        # Debug: log first failure per unique field
        if not hasattr(_match_fused_key, '_logged'):
            _match_fused_key._logged = set()
        if field not in _match_fused_key._logged:
            _match_fused_key._logged.add(field)
            if not v2_canonical.endswith("._extra_state"):
                logger.warning(
                    "Fused key not found in metadata for v2_key='%s' (field='%s'). "
                    "Tried candidates: %s",
                    v2_canonical,
                    field,
                    candidates,
                )
        return None

    # ------------------------------------------------------------------
    # Phase A: Model weights
    # ------------------------------------------------------------------
    # Debug: log sample metadata keys on first call
    _sample_keys = sorted(k for k in metadata if not k.startswith("optimizer."))
    logger.debug(
        "DCP metadata has %d non-optimizer keys. Sample: %s", len(_sample_keys), _sample_keys[:10]
    )

    for v2_key, v2_val in v2_by_canonical.items():
        result = _match_fused_key(v2_key, v2_val)
        if result:
            if len(result) == 3 and result[2] == "seq_mlp":
                # SequentialMLP: (td_key, local_expert_idx, "seq_mlp")
                td_key, local_expert_idx, _ = result
                fused_layer_groups.setdefault(td_key, []).append(
                    (v2_key, v2_val, local_expert_idx, "seq_mlp")
                )
            else:
                # Fused layer or GroupedMLP: (td_key, layer_idx)
                td_key, layer_idx = result
                exp_m = _EXPERT_V2_KEY_RE.match(v2_key)
                if exp_m:
                    expert_idx = int(exp_m.group(3))
                    fused_layer_groups.setdefault(td_key, []).append(
                        (v2_key, v2_val, layer_idx, expert_idx)
                    )
                else:
                    fused_layer_groups.setdefault(td_key, []).append((v2_key, v2_val, layer_idx))
            continue
        # Not fused — check 1:1
        if v2_key in metadata:
            regular_model[v2_key] = v2_val
            continue
        m_key = f"module.{v2_key}"
        if m_key in metadata:
            regular_model[m_key] = v2_val
            continue

    # ------------------------------------------------------------------
    # Phase B: Optimizer states (exp_avg / exp_avg_sq) & hi-prec (param)
    # ------------------------------------------------------------------

    # Phase B1: Build fused_layer_groups for optimizer state params
    # that have fused counterparts in the torch_dist checkpoint.
    # Handles three formats: regular fused layer, GroupedMLP, SequentialMLP.
    for param_name, states in v2_optim_state.items():
        m = _LAYERED_KEY_RE.match(param_name)
        if not m:
            continue
        prefix = m.group(1) or ""
        layer_idx = int(m.group(2))
        field = m.group(3)

        # Check SequentialMLP expert format first
        seq_exp_m = _SEQ_EXPERT_FIELD_RE.match(field)
        if seq_exp_m:
            local_expert_idx = int(seq_exp_m.group(1))
            fc_type = seq_exp_m.group(2)
            # Extract sub-layer prefix (if any) before ``mlp.experts.local_experts``
            sub_layer_prefix = ""
            if not field.startswith("mlp.experts.local_experts."):
                idx = field.find("mlp.experts.local_experts.")
                if idx > 0:
                    sub_layer_prefix = field[:idx]  # e.g. "mtp_model_layer."
            for sk, sv in states.items():
                if sk not in ("exp_avg", "exp_avg_sq"):
                    continue
                fused_base = (
                    f"optimizer.state.{sk}."
                    f"{prefix}layers.{layer_idx}."
                    f"{sub_layer_prefix}mlp.experts.experts.{fc_type}.weight"
                )
                fused_base = reverse_normalize_torch_dist_key(fused_base)
                td_opt_key = None
                for candidate in [fused_base, f"model.{fused_base}"]:
                    if candidate in metadata:
                        td_opt_key = candidate
                        break
                if td_opt_key is None:
                    continue
                fused_layer_groups.setdefault(td_opt_key, []).append(
                    (param_name, sv, local_expert_idx, "seq_mlp")
                )
                optim_matched.add(param_name)
            continue

        # Check GroupedMLP expert format
        fused_base = f"{prefix}layers.{field}"
        exp_m = _EXPERT_V2_KEY_RE.search(field)
        if exp_m:
            fc_type = exp_m.group(2)
            field_exp = (
                _EXPERT_V2_KEY_RE.sub(rf"mlp.experts.experts.linear_fc\2.weight", field).rstrip(
                    ".weight"
                )
                + ".weight"
            )
            fused_base = f"{prefix}layers.{field_exp}"
            expert_idx = int(exp_m.group(3))
        else:
            expert_idx = None

        for sk, sv in states.items():
            if sk not in ("exp_avg", "exp_avg_sq"):
                continue
            td_opt_key = None
            for candidate in [
                f"optimizer.state.{sk}.{fused_base}",
                f"optimizer.state.{sk}.model.{fused_base}",
            ]:
                if candidate in metadata:
                    td_opt_key = candidate
                    break
            if td_opt_key is None:
                continue
            if expert_idx is not None:
                fused_layer_groups.setdefault(td_opt_key, []).append(
                    (param_name, sv, layer_idx, expert_idx)
                )
            else:
                fused_layer_groups.setdefault(td_opt_key, []).append((param_name, sv, layer_idx))
            optim_matched.add(param_name)

    # Phase B2: Handle remaining metadata entries (1:1 optimizer keys + hi-prec)
    for td_key in metadata:
        if not td_key.startswith("optimizer.state."):
            continue
        rest = td_key[len("optimizer.state.") :]
        parts = rest.split(".", 1)

        if rest.startswith("param."):
            param_name_td = rest[len("param.") :]
            param_canonical = _canonicalize_td_key(param_name_td, strip_model_prefix=False)
            param_canonical = _strip_module_prefix(param_canonical)
            if param_canonical in v2_by_canonical:
                hi_prec_model[param_name_td] = v2_by_canonical[param_canonical]
                hi_prec_td_keys.add(param_canonical)
            continue

        if len(parts) != 2:
            continue
        state_key, param_name_td = parts
        param_canonical = _canonicalize_td_key(param_name_td, strip_model_prefix=False)
        param_canonical = _strip_module_prefix(param_canonical)

        # Try 1:1 match. Fused-layer matching is handled by Phase B1 above.
        if param_canonical in v2_optim_state and state_key in v2_optim_state[param_canonical]:
            optim_keys[td_key] = v2_optim_state[param_canonical][state_key]
            optim_matched.add(param_canonical)

    if logger.isEnabledFor(logging.DEBUG):
        total_fused = sum(len(v) for v in fused_layer_groups.values())
        logger.debug(
            "torch_dist → v2: %d metadata keys → %d regular + %d fused (%d groups) "
            "+ %d hi-prec + %d optim matched",
            len(metadata),
            len(regular_model),
            total_fused,
            len(fused_layer_groups),
            len(hi_prec_model),
            len(optim_matched),
        )
    return regular_model, fused_layer_groups, hi_prec_model, optim_keys, optim_matched


# ------------------------------------------------------------------
# Torch_dist → FSDP v2 online checkpoint conversion
# ------------------------------------------------------------------


def _assert_dcp_keys_in_metadata(mapped_sd: dict, metadata: dict) -> None:
    """Verify every DCP load key exists in the torch_dist metadata."""
    missing = []
    for top_key, subtree in mapped_sd.items():
        _collect_missing_keys(f"{top_key}.", subtree, metadata, missing)
    if missing:
        raise RuntimeError(
            f"{len(missing)} DCP load keys not found in torch_dist metadata. "
            f"Missing: {sorted(missing)}"
        )


def _collect_missing_keys(prefix: str, subtree, metadata: dict, missing: list) -> None:
    """Recursively check that every leaf key in *subtree* exists in *metadata*."""
    if isinstance(subtree, dict):
        for k, v in subtree.items():
            _collect_missing_keys(f"{prefix}{k}.", v, metadata, missing)
    elif isinstance(subtree, torch.Tensor):
        key = prefix.rstrip(".")
        if key not in metadata:
            missing.append(key)


def _canonicalize_td_key(td_key, *, strip_model_prefix=True):
    """Strip the top-level DCP prefix, shard suffix, and normalize naming."""
    if strip_model_prefix and td_key.startswith("model."):
        key = td_key[len("model.") :]
    else:
        key = td_key
    key = normalize_torch_dist_key(key)
    return key


def _load_torch_dist_into_megatron_fsdp_v2(
    args,
    checkpoint_name,
    model,
    v2_state_dict,
    strict=True,
):
    """Load a torch_dist checkpoint into a Megatron FSDP v2 skeleton via DCP.

    This is the entry point for online checkpoint conversion from
    legacy ``torch_dist`` format (ND-parallel) to ``fsdp_dtensor``
    (Megatron FSDP v2).

    The conversion proceeds in five phases:

    1. **Preprocess & verify** — wrap optimizer states as uneven DTensors
       and verify ``__create_chunk_list__`` / ``__create_write_items__`` metadata.
       When ``args.mamba`` is True, split fused MambaMixer params
       (``in_proj.weight`` → ``.z`` / ``.x`` / ``.B`` / ``.C`` / ``.dt``,
       ``conv1d.*`` → ``.x`` / ``.B`` / ``.C``) so they match the torch_dist
       split keys.
    2. **Build name mapping** — match torch_dist metadata keys to v2 DTensors,
       handling fused layers, GroupedMLP, and SequentialMLP expert formats.
    3. **DCP load 1:1 entries** — load regular model weights and optimizer
       states that have a direct 1:1 correspondence in the torch_dist checkpoint.
    4. **Load fused entries** — load fused (multi-layer and/or multi-expert)
       tensors and slice them into per-layer/per-expert v2 DTensors.
    5. **Verify** — when ``strict=True``, ensure every v2 model param and
       optimizer state was loaded from the torch_dist checkpoint.  When
       ``strict=False``, unmatched entries are logged as warnings instead
       of raising errors.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    # ---- Phase 1: Preprocess & verify v2 state dict ----

    # Split fused MambaMixer params so that the sub-keys match the
    # torch_dist checkpoint (which stores e.g. ``in_proj.weight.z``).
    if model is not None:
        if isinstance(model, (list, tuple)):
            assert len(model) == 1
            model = model[0]
        model_sd, new_opt = handle_mamba_in_state_dict_v2(
            model, v2_state_dict["model"], v2_state_dict.get("optimizer")
        )
        v2_state_dict["model"] = model_sd
        if new_opt is not None:
            v2_state_dict["optimizer"] = new_opt

    v2_by_canonical, v2_optim_state = _preprocess_and_verify_v2_state_dict(v2_state_dict, model)

    # ---- Phase 2: Read torch_dist metadata & build name mapping ----
    reader = FileSystemReader(checkpoint_name)
    metadata = reader.read_metadata().state_dict_metadata
    regular_model, fused_layer_groups, hi_prec_model, optim_keys, optim_matched = (
        _build_torch_dist_to_v2_map(metadata, v2_by_canonical, v2_optim_state)
    )

    # ---- Phase 3: Build & load mapped state dict via DCP ----
    mapped_sd = {}
    if regular_model:
        mapped_sd.update(regular_model)

    opt_state = {}
    for td_key, v2_val in optim_keys.items():
        rest = td_key[len("optimizer.state.") :]
        state_key, param_name_td = rest.split(".", 1)
        opt_state.setdefault(state_key, {})[param_name_td] = v2_val
    if hi_prec_model:
        opt_state["param"] = hi_prec_model
    if opt_state:
        mapped_sd["optimizer"] = {"state": opt_state}

    _assert_dcp_keys_in_metadata(mapped_sd, metadata)
    if mapped_sd:
        dcp.load(
            state_dict=mapped_sd,
            checkpoint_id=checkpoint_name,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

    # Merge hi-precision model params back so the model subtree is complete.
    if hi_prec_model:
        mapped_sd.update({k: v for k, v in hi_prec_model.items() if k not in mapped_sd})

    # ---- Phase 4: Load fused layer / expert params by slicing ----
    # Handles three tensor formats:
    #   Regular fused:  shape (num_layers, ...)            → flat[layer_idx]
    #   GroupedMLP:     shape (num_layers, num_experts, ...) → flat[layer_idx, expert_idx]
    #   SequentialMLP:  shape (num_global_experts, ...)     → flat[global_expert_idx]
    #
    # Entry types by tag:
    #   (v2_key, v2_val, layer_idx)                      → regular fused layer
    #   (v2_key, v2_val, layer_idx, expert_idx)           → GroupedMLP expert
    #   (v2_key_or_param_name, v2_val, local_expert_idx, "seq_mlp") → SequentialMLP expert
    for td_key, entries in fused_layer_groups.items():
        td_meta = metadata.get(td_key)
        assert td_meta is not None, f"Missing metadata for fused key '{td_key}'"
        first_entry = entries[0]
        is_seq_mlp = len(first_entry) >= 4 and first_entry[-1] == "seq_mlp"
        is_grouped_mlp = len(first_entry) >= 4 and not is_seq_mlp

        entries.sort(key=lambda x: x[2])  # sort by layer_idx / local_expert_idx
        example_val = entries[0][1]  # v2_val (model) or optimizer state tensor

        # Build the full fused tensor shape and compute slicing
        fused_shape = list(example_val.shape)
        if is_seq_mlp:
            num_total_experts = td_meta.size[0]
            fused_shape.insert(0, num_total_experts)
            num_local = len(entries)
            ep_rank = mpu.get_expert_model_parallel_rank()
        elif is_grouped_mlp:
            num_layers = td_meta.size[0]
            num_experts = td_meta.size[1]
            fused_shape.insert(0, num_experts)
            fused_shape.insert(0, num_layers)
        else:
            num_layers = td_meta.size[0]
            fused_shape.insert(0, num_layers)

        # Use a Shard(0) DTensor so the large fused tensor is distributed
        # across DP ranks during DCP load, avoiding OOM on a single rank
        # (e.g. for GroupedMLP with many layers × experts).
        device_mesh = example_val.device_mesh if isinstance(example_val, DTensor) else None
        if device_mesh is not None:
            flat = torch.distributed.tensor.empty(
                fused_shape, dtype=example_val.dtype, device_mesh=device_mesh, placements=[Shard(0)]
            )
        else:
            flat = torch.empty(fused_shape, dtype=example_val.dtype, device=example_val.device)

        dcp.load(
            state_dict={td_key: flat},
            checkpoint_id=checkpoint_name,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        # Gather shards back into a full local tensor for per-layer slicing.
        if device_mesh is not None and isinstance(flat, DTensor):
            flat = redistribute_uneven_dtensor_to_replicated(flat).to_local()

        for i, entry in enumerate(entries):
            if is_seq_mlp:
                v2_key_or_param, v2_val, local_expert_idx = entry[:3]
                chunk = flat[ep_rank * num_local + local_expert_idx]
            elif is_grouped_mlp:
                v2_key_or_param, v2_val, layer_idx, expert_idx = entry[:4]
                chunk = flat[layer_idx, expert_idx]
            else:
                v2_key_or_param, v2_val, layer_idx = entry[:3]
                chunk = flat[layer_idx]
            if isinstance(v2_val, DTensor):
                lt = v2_val._local_tensor
                if lt.numel() != 0 and hasattr(lt, "__create_chunk_list__"):
                    local_off = 0
                    for c in lt.__create_chunk_list__():
                        off = c.offsets
                        sz = c.sizes
                        src_idx = tuple(slice(o, o + s) for o, s in zip(off, sz))
                        src = chunk[src_idx]
                        dst_idx = (slice(local_off, local_off + sz[0]),) + tuple(
                            slice(0, s) for s in sz[1:]
                        )
                        lt[dst_idx].copy_(src)
                        local_off += sz[0]
            else:
                v2_val.copy_(chunk)
            mapped_sd[v2_key_or_param] = v2_val

    # ---- Phase 5: Verify all v2 params & optimizer states were loaded ----
    loaded = set(_canonicalize_td_key(k) for k in mapped_sd)
    all_v2 = set(v2_by_canonical.keys())
    unloaded = all_v2 - loaded - {k for k in all_v2 if k.endswith("._extra_state")}

    v2_optim_unmatched = set(v2_optim_state.keys()) - optim_matched

    if strict:
        if unloaded:
            raise RuntimeError(
                f"{len(unloaded)} v2 model parameters were not loaded from the "
                f"torch_dist checkpoint. Unloaded params: {sorted(unloaded)}"
                f" loaded: {sorted(loaded)}"
            )
        if v2_optim_unmatched:
            raise RuntimeError(
                f"{len(v2_optim_unmatched)} v2 optimizer state entries were "
                f"not matched to torch_dist data: {sorted(v2_optim_unmatched)}"
            )
    else:
        if unloaded:
            logger.warning(
                "%d v2 model parameters were not loaded from the torch_dist "
                "checkpoint (strict=False): %s",
                len(unloaded),
                sorted(unloaded),
            )
        if v2_optim_unmatched:
            logger.warning(
                "%d v2 optimizer state entries were not matched to "
                "torch_dist data (strict=False): %s",
                len(v2_optim_unmatched),
                sorted(v2_optim_unmatched),
            )

    # ---- Load common state dict ----
    common_info = dist_checkpointing.load_common_state_dict(checkpoint_name)
    if "optimizer" in common_info:
        opt_common = common_info.pop("optimizer")
        if isinstance(opt_common, dict):
            if "optimizer" in opt_common:
                opt_common = opt_common["optimizer"]
    v2_state_dict.update(common_info)

    return v2_state_dict
