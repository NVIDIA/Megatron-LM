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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/utils.py
# Source license: Apache-2.0.

import copy
import functools
import pickle
import re
import types
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import torch
from rich.table import Table
from transformers.configuration_utils import PretrainedConfig

from megatron.core.transformer.module import MegatronModule


def mcore_to_hf_window_size(window_size: int | list[int] | tuple[int, int] | None) -> int | None:
    """Convert an MCore inclusive attention window to the Hugging Face token count.

    MCore represents a causal sliding window as ``(left, right)``, where the
    current token is included in the Hugging Face window size. Checkpoint YAML
    reloads tuples as lists, so both sequence types are accepted. Providers that
    already store the Hugging Face scalar representation pass through unchanged.
    """
    if isinstance(window_size, (list, tuple)):
        if len(window_size) != 2:
            raise ValueError(f"Expected a two-element MCore window, got {window_size!r}")
        return window_size[0] + 1
    return window_size


def unwrap_model(model, module_instances=None):
    """Unwrap a model (or list of models) to the underlying module.
    Extends ``megatron.core.utils.unwrap_model`` with awareness of ``MegatronFSDP``.
    """
    if module_instances is None:
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP
        from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
            FullyShardedDataParallelV1,
            FullyShardedDataParallelV2,
        )
        from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
        from megatron.core.transformer.module import Float16Module

        module_instances = (
            DDP,
            torch_FSDP,
            FullyShardedDataParallelV1,
            FullyShardedDataParallelV2,
            Float16Module,
            MegatronFSDP,
        )

    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def weights_verification_table(bridge, megatron_model) -> Table:
    """
    Returns a table comparing weights between a Hugging Face model and a Megatron-LM model.

    Args:
        bridge (AutoBridge): The bridge object containing model information.
        megatron_model: The Megatron-LM model instance.

    Returns:
        Table: A rich Table object with the comparison.
    """
    table = Table(title="Hugging Face Weights Verification")
    table.add_column("Weight Name", style="cyan")
    table.add_column("Shape")
    table.add_column("DType")
    table.add_column("Device")
    table.add_column("Matches Original", justify="center")

    # Check each weight against the original HF-model
    for name, param in bridge.export_hf_weights(megatron_model, show_progress=True):
        original_param = bridge.hf_pretrained.state[name]
        table.add_row(
            name,
            str(tuple(param.shape)),
            str(param.dtype).replace("torch.", ""),
            str(param.device),
            "✅" if torch.allclose(param, original_param.to(param.device), atol=1e-6) else "❌",
        )

    return table


def get_module_and_param_from_name(
    models: MegatronModule | List[MegatronModule], param_name: str, vp_stage: Optional[int] = None
) -> Tuple[torch.nn.Module, torch.Tensor] | Tuple[torch.nn.Module, torch.Tensor, Tuple]:
    """
    Get parameter from specific VP stage, ensuring that parameter
    attributes are preserved. Supports both absolute and relative parameter names.

    Args:
        models: List of Megatron model instances or a submodule
        param_name: Dot-separated parameter name (can be absolute or relative to models)
        vp_stage: Virtual pipeline stage index (None for single stage)

    Returns:
        Tuple of (module, parameter) where module owns the parameter

    Raises:
        ValueError: If vp_stage is out of range or parameter doesn't exist

    Examples:
        Basic usage with full model:
        >>> module, param = get_module_and_param_from_name(
        ...     models=full_model,
        ...     param_name="transformer.layers.0.attention.query.weight"
        ... )

        Usage with model list and VP stage:
        >>> module, param = get_module_and_param_from_name(
        ...     models=[model1, model2, model3],
        ...     param_name="layers.0.mlp.dense.bias",
        ...     vp_stage=1
        ... )

        Usage with submodule and relative path:
        >>> linear_module = model.transformer.layers[0].mlp.dense
        >>> module, param = get_module_and_param_from_name(
        ...     models=linear_module,
        ...     param_name="weight"
        ... )

        Usage with submodule and absolute path (automatic suffix matching):
        >>> linear_module = model.transformer.layers[0].mlp.dense
        >>> module, param = get_module_and_param_from_name(
        ...     models=linear_module,
        ...     param_name="transformer.layers.0.mlp.dense.weight"
        ... )
        # Automatically matches "weight" suffix and returns the parameter

        Edge case with partial path matching:
        >>> attention_module = model.transformer.layers[0].attention
        >>> module, param = get_module_and_param_from_name(
        ...     models=attention_module,
        ...     param_name="layers.0.attention.query.weight"
        ... )
        # Matches "query.weight" suffix within the attention module
    """

    if isinstance(models, list):
        if vp_stage is None:
            model = models[0]
        else:
            if vp_stage >= len(models):
                raise ValueError(f"VP stage {vp_stage} out of range (max: {len(models) - 1})")
            model = models[vp_stage]
    else:
        model = models

    module = unwrap_model(model)
    splitted_name = param_name.split(".")

    # Try to find the parameter using the given parts
    def try_get_param(parts):
        param = module
        temp_module = module

        for i, part in enumerate(parts):
            if not hasattr(param, part):
                return None
            param = getattr(param, part)
            if i < len(parts) - 1:
                temp_module = getattr(temp_module, part)

        return temp_module, param

    # First try the full parameter name (current behavior)
    result = try_get_param(splitted_name)
    if result is not None:
        return result

    # If full name doesn't work, try suffixes of the parameter name
    # This handles cases where models is a submodule but param_name is absolute
    for start_idx in range(1, len(splitted_name)):
        suffix_parts = splitted_name[start_idx:]
        result = try_get_param(suffix_parts)
        if result is not None:
            return result

    # If no approach works, raise an error
    raise ValueError(f"Parameter '{param_name}' not found in model at VP stage {vp_stage}")


def remove_non_pickleables(obj, max_depth: int = 3, current_depth: int = 0):
    """Remove non-pickleable objects from a configuration object recursively.

    This utility function identifies and removes objects that cannot be pickled for
    inter-process communication, including functions, bound methods, partial
    functions, and other problematic callables.

    Args:
        obj: The object to clean
        max_depth: Maximum recursion depth (default: 3)
        current_depth: Current recursion depth (internal use)

    Returns:
        The cleaned object with non-pickleables removed

    Raises:
        TypeError: Metadata at the recursion limit cannot be serialized. Unknown
            metadata is never silently discarded to make serialization succeed.
    """

    # Handle None
    if obj is None:
        return obj

    # Explicitly drop process group objects without importing their classes directly.
    cls = obj if isinstance(obj, type) else type(obj)
    cls_module = getattr(cls, "__module__", "")
    cls_name = getattr(cls, "__qualname__", getattr(cls, "__name__", ""))
    if (cls_module, cls_name) in {
        ("megatron.core.process_groups_config", "ProcessGroupCollection"),
        ("torch._C._distributed_c10d", "ProcessGroup"),
    }:
        return None

    # Check if object is a problematic callable
    if callable(obj):
        # Allow classes/types but remove function objects, methods, partials
        if isinstance(obj, type):
            return obj
        elif hasattr(obj, "__call__") and (
            isinstance(obj, (types.FunctionType, types.MethodType, functools.partial))
            or hasattr(obj, "__self__")
        ):  # bound methods
            return None

    # Remove known runtime leaves even at the boundary, but do not discard an
    # entire metadata subtree merely because one of its descendants is unsafe.
    if current_depth >= max_depth:
        try:
            pickle.dumps(obj)
        except (pickle.PicklingError, TypeError, AttributeError, RecursionError) as exc:
            raise TypeError(
                f"Cannot serialize conversion metadata of type {type(obj).__name__} "
                f"at max_depth={max_depth}; separate runtime state from conversion metadata."
            ) from exc
        return obj

    # Check containers before objects: OrderedDict has a __dict__, but its
    # hook values are mapping entries rather than object attributes.
    if isinstance(obj, list):
        cleaned_obj = copy.copy(obj)
        for index, item in enumerate(obj):
            cleaned_obj[index] = remove_non_pickleables(item, max_depth, current_depth + 1)
    elif isinstance(obj, tuple):
        # Build the tuple storage directly: named tuples and other subclasses
        # may have constructors that take positional fields, not an iterable.
        items = [remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj]
        if all(cleaned is original for cleaned, original in zip(items, obj)):
            # Keep native tuple types such as torch.Size on their own copy path.
            cleaned_obj = copy.copy(obj)
        else:
            cleaned_obj = tuple.__new__(type(obj), items)
        if hasattr(obj, "__dict__"):
            vars(cleaned_obj).update(vars(obj))
    elif isinstance(obj, dict):
        # A shallow copy retains OrderedDict/defaultdict behavior and attributes.
        cleaned_obj = copy.copy(obj)
        if isinstance(obj, defaultdict):
            cleaned_obj.default_factory = remove_non_pickleables(
                obj.default_factory, max_depth, current_depth + 1
            )
        for key, value in obj.items():
            cleaned_obj[key] = remove_non_pickleables(value, max_depth, current_depth + 1)
    else:
        cleaned_obj = None

    # Handle dataclass/object with attributes
    if hasattr(obj, "__dict__"):
        # Create a copy to avoid modifying the original
        if cleaned_obj is None:
            cleaned_obj = copy.copy(obj)

        # Read stored attributes from ``__dict__`` directly. Configuration classes may
        # deliberately reject dynamic attribute access for values whose meaning is
        # layer-dependent, even though the raw value still needs to be copied for IPC.
        for attr_name, attr_value in list(vars(cleaned_obj).items()):
            # Setup recreates these registries when constructing a model. They
            # are not conversion metadata; keep them off the detached IPC copy
            # without unregistering any hooks on the live training provider.
            if attr_name in {"_pre_wrap_hooks", "_megatron_bridge_setup_pre_wrap_hooks"}:
                delattr(cleaned_obj, attr_name)
                continue
            # Recursively clean attribute
            try:
                cleaned_value = remove_non_pickleables(attr_value, max_depth, current_depth + 1)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot clean conversion metadata attribute {attr_name!r}: {exc}"
                ) from exc

            # Set the cleaned value (or None if it was removed)
            setattr(cleaned_obj, attr_name, cleaned_value)

        return cleaned_obj

    if cleaned_obj is not None:
        return cleaned_obj

    # For primitive types and other safe objects, return as-is
    return obj


def extract_sort_key(param_name: str):
    """Extract sorting key based on layer and expert numbers."""

    # Extract at most 2 numbers: layer number and expert number
    # Pattern: *layers.d+.*d+ (layer number and potentially expert number)
    numbers = []
    # Find layer number
    layer_match = re.search(r"layers\.(\d+)", param_name)
    if layer_match:
        numbers.append(int(layer_match.group(1)))
    # Find expert number after bias or weight for grouped experts.
    expert_match = re.search(r"(?:bias|weight)(\d+)", param_name)
    if expert_match:
        numbers.append(int(expert_match.group(1)))
    else:
        # SequentialMLP names use local_experts.<N> instead.
        local_experts_match = re.search(r"local_experts\.(\d+)", param_name)
        if local_experts_match:
            numbers.append(int(local_experts_match.group(1)))
    # Pad to ensure consistent comparison (max 2 numbers)
    while len(numbers) < 2:
        numbers.append(-1)
    numbers = numbers[:2]  # Keep at most 2 numbers
    return numbers, param_name


def moe_experts_stored_packed(hf_pretrained, layers_prefix: str, default: bool = False) -> bool:
    """Whether a checkpoint stores routed MoE experts *fused* rather than *per-expert*.

    Fused layout keys look like ``{layers_prefix}<L>.mlp.experts.gate_up_proj`` / ``.down_proj`` (a
    single stacked tensor per projection), while the per-expert layout (what ``save_pretrained``
    writes) uses ``{layers_prefix}<L>.mlp.experts.<i>.gate_proj|up_proj|down_proj.weight``. Which one a
    checkpoint uses depends on the transformers version, so the bridge selects mappings accordingly.

    Args:
        hf_pretrained: The loaded HF model wrapper (its ``state.source`` provides the checkpoint keys).
        layers_prefix: HF prefix up to and including ``layers.`` (e.g. ``"model.layers."`` for an LLM,
            ``"model.language_model.layers."`` for a VLM).
        default: Value returned when the checkpoint keys are unavailable (e.g. building mappings
            without a loaded checkpoint, as in unit tests) or no routed-expert keys are found.

    Returns:
        ``True`` if experts are stored fused, ``False`` if per-expert, ``default`` if undetermined.

    Raises:
        ValueError: if the checkpoint mixes fused and per-expert layouts.
    """
    source = getattr(getattr(hf_pretrained, "state", None), "source", None)
    if source is None or not hasattr(source, "get_all_keys"):
        return default
    prefix = re.escape(layers_prefix)
    per_expert_re = re.compile(rf"^{prefix}\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$")
    fused_re = re.compile(rf"^{prefix}\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)$")
    keys = list(source.get_all_keys())
    has_per_expert = any(per_expert_re.match(k) for k in keys)
    has_fused = any(fused_re.match(k) for k in keys)
    if has_per_expert and has_fused:
        raise ValueError(
            f"Checkpoint mixes fused and per-expert MoE expert layouts under {layers_prefix}*.mlp.experts; "
            "expected a single consistent layout."
        )
    return has_fused if (has_fused or has_per_expert) else default


def get_causal_lm_class_name_via_auto_map(config: PretrainedConfig) -> str | None:
    """Return CausalLM class name via config.auto_map if available; otherwise None.

    If auto_map["AutoModelForCausalLM"] is present in the config, returns the class
    name string extracted from the mapping value by splitting on '.' and taking the
    last segment. Returns None if auto_map is not set.
    """
    auto_map = getattr(config, "auto_map", None)
    if auto_map and "AutoModelForCausalLM" in auto_map:
        auto_map_class = auto_map["AutoModelForCausalLM"]
        return str(auto_map_class).split(".")[-1]

    return None


def conform_config_to_reference(
    hf_config_dict: dict[str, object], reference_config: dict[str, object]
) -> dict[str, object]:
    """Return hf_config_dict projected onto reference keys, filling missing values from reference_config."""
    reference_config_keys = set(reference_config.keys())
    filtered_config_dict = {}
    for key, value in hf_config_dict.items():
        if key not in reference_config_keys:
            continue

        reference_value = reference_config[key]
        if isinstance(value, dict) and isinstance(reference_value, dict):
            value = conform_config_to_reference(value, reference_value)
        filtered_config_dict[key] = value

    for key, value in reference_config.items():
        if key not in filtered_config_dict:
            filtered_config_dict[key] = value
    return filtered_config_dict


def persistent_buffers(model: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    """Return an iterator over persistent module buffers, yielding both the name of the buffer as well as the buffer itself."""

    for mod_prefix, mod in model.named_modules():
        # only local buffers; we'll add the prefix ourselves
        for local_name, buffer in mod.named_buffers(recurse=False):
            if local_name not in getattr(mod, "_non_persistent_buffers_set", set()):
                full_name = f"{mod_prefix + '.' if mod_prefix else ''}{local_name}"
                yield full_name, buffer


def is_modelopt_dynamic_module(module):
    """Check if a module is a modelopt dynamic module."""
    try:
        from modelopt.torch.opt.dynamic import DynamicModule

        return isinstance(module, DynamicModule)
    except ImportError:
        return False
