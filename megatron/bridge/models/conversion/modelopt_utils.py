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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/modelopt_utils.py
# Source license: Apache-2.0.

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    ConcatenatedQKVMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    GDNLinearMapping,
    GDNLinearMappingSeparate,
    MambaInProjMapping,
    MegatronParamMapping,
    QKVGMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    _split_gdn_grouped_to_separate,
    split_gdn_linear_weights,
    split_qkv_weights,
    split_qkvg_weights,
)
from megatron.bridge.models.conversion.quant_mapping import AmaxMapping
from megatron.core.utils import get_pg_size

HFExportHook = Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]


class ModelOptExportPlan(NamedTuple):
    """Prepared conversion tasks and stable canonical ModelOpt metadata."""

    conversion_tasks: list[WeightConversionTask]
    quantization_config: dict[str, Any]
    quantized_params: frozenset[str]
    local_quantization_specs: dict[str, object | None]
    traced_rows: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {}


@dataclass(frozen=True)
class _SourceState:
    state: object | None
    weight_shape: tuple[int, ...]
    parallelism: str | None = None
    qkv_layout: tuple[int, int, int, int, bool] | None = None
    weight_dtype: torch.dtype | None = None
    gdn_layout: tuple[int, int, int, int] | None = None
    mamba_layout: tuple[int, int, int, int, int] | None = None
    traced_rows: tuple[tuple[str, tuple[int, ...]], ...] | None = None


# Recipes keep GatedDeltaNet's beta / alpha gates in BF16, but Megatron fuses them into the
# quantized in_proj, so a quantizer pattern can't exclude them; they're exported unquantized.
_GDN_UNQUANTIZED_OUTPUTS = ("b", "a")


def _same_storage(left: object, right: object) -> bool:
    if left is right:
        return True
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    return (
        left.device == right.device
        and left.numel() == right.numel()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
    )


def _direct_weight_name(module: torch.nn.Module, weight: torch.Tensor) -> str | None:
    for name, parameter in module.named_parameters(recurse=False):
        if _same_storage(parameter, weight):
            return name
    return None


# Mappings whose HF rows follow a known layout, unless a subclass overrides that layout.
_ROW_LAYOUT_MAPPINGS = (
    QKVMapping,
    ConcatenatedQKVMapping,
    QKVGMapping,
    MambaInProjMapping,
    GDNLinearMappingSeparate,
)
_PLAIN_MAPPINGS = (
    AutoMapping,
    ColumnParallelMapping,
    RowParallelMapping,
    ReplicatedMapping,
    GatedMLPMapping,
)


def _traces_rows(mapping: Any) -> bool:
    """Whether the HF rows of a quantized weight come from tracing the mapping's own ``megatron_to_hf``."""
    if mapping.is_expert:
        return False
    for base in _ROW_LAYOUT_MAPPINGS:
        if isinstance(mapping, base):
            return any(
                getattr(type(mapping), method) is not getattr(base, method)
                for method in ("megatron_to_hf", "_megatron_to_hf")
                if hasattr(base, method)
            )
    return not isinstance(mapping, _PLAIN_MAPPINGS)


def _trace_rows(
    task: WeightConversionTask, parallelism: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Rows of the TP-merged weight that land in each HF output, from the mapping's own conversion.

    Each element is labeled with its (row, column) in the merged weight, so the converted labels show
    where every element went. Only whole-row selections can reuse the weight's quantization state.
    """
    mapping, weight = task.mapping, task.param_weight
    if weight.ndim != 2 or parallelism not in ("column", "row", "replicated"):
        raise NotImplementedError(
            f"ModelOpt export cannot trace {type(mapping).__name__} rows for {task.global_param_name}"
        )
    rows, cols = weight.shape
    full_rows = rows * mapping.tp_size if parallelism == "column" else rows
    full_cols = cols * mapping.tp_size if parallelism == "row" else cols
    dtype = torch.int32 if full_rows * full_cols <= torch.iinfo(torch.int32).max else torch.int64
    row_ids = torch.arange(rows, dtype=dtype, device=weight.device)
    col_ids = torch.arange(cols, dtype=dtype, device=weight.device)
    row_ids += mapping.tp_rank * rows if parallelism == "column" else 0
    col_ids += mapping.tp_rank * cols if parallelism == "row" else 0
    with MegatronParamMapping.skip_pp_broadcast():
        outputs = mapping.megatron_to_hf(
            row_ids[:, None] * full_cols + col_ids, task.megatron_module
        )
    if not outputs:
        raise NotImplementedError(
            f"ModelOpt export requires {type(mapping).__name__} to convert {task.global_param_name} on its own"
        )
    traced = []
    for hf_name, labels in outputs.items():
        hf_rows = labels[:, 0] // full_cols if labels.ndim == 2 else None
        if (
            hf_rows is None
            or labels.dtype != dtype
            or not torch.equal(
                labels,
                hf_rows[:, None] * full_cols
                + torch.arange(full_cols, dtype=dtype, device=labels.device),
            )
        ):
            raise NotImplementedError(
                f"ModelOpt export requires {type(mapping).__name__} to copy whole rows of "
                f"{task.global_param_name}, but {hf_name} does not"
            )
        traced.append((hf_name, tuple(hf_rows.tolist())))
    return tuple(traced)


def _mapping_parallelism(mapping: Any, module: torch.nn.Module) -> str:
    if getattr(mapping, "is_grouped_export", False):
        raise NotImplementedError(
            "ModelOpt real-quant export does not yet support grouped HF expert weights"
        )
    if isinstance(mapping, GatedMLPMapping):
        return "gated"
    if isinstance(mapping, MambaInProjMapping):
        return "column"
    if isinstance(
        mapping, (QKVMapping, ConcatenatedQKVMapping, QKVGMapping, GDNLinearMappingSeparate)
    ):
        return mapping._tp_mapping._detect_parallelism_type(module)
    if isinstance(mapping, AutoMapping):
        return mapping._detect_parallelism_type(module)
    if isinstance(mapping, ColumnParallelMapping):
        return "column"
    if isinstance(mapping, RowParallelMapping):
        return "row"
    if isinstance(mapping, ReplicatedMapping):
        return "replicated"
    if _traces_rows(mapping):
        tp_mapping = getattr(mapping, "_tp_mapping", None)
        if not isinstance(
            tp_mapping, (AutoMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping)
        ):
            tp_mapping = AutoMapping(mapping.megatron_param, mapping.megatron_param)
        return _mapping_parallelism(tp_mapping, module)
    raise NotImplementedError(
        f"ModelOpt real-quant export does not support {type(mapping).__name__} topology"
    )


def _qkv_layout(mapping: Any, module: torch.nn.Module) -> tuple[int, int, int, int, bool] | None:
    if not isinstance(mapping, (QKVMapping, ConcatenatedQKVMapping, QKVGMapping)):
        return None
    config = mapping._get_config(module)
    return (
        int(config.num_attention_heads),
        int(config.num_query_groups),
        int(config.kv_channels or 0),
        int(config.hidden_size),
        bool(getattr(config, "attention_output_gate", False)),
    )


def _gdn_layout(mapping: Any, module: torch.nn.Module) -> tuple[int, int, int, int] | None:
    if not isinstance(mapping, GDNLinearMappingSeparate):
        return None
    config = mapping._get_config(module)
    return (
        int(config.linear_num_key_heads),
        int(config.linear_num_value_heads),
        int(config.linear_key_head_dim),
        int(config.linear_value_head_dim),
    )


def _gdn_row_indices(
    layout: tuple[int, int, int, int], rows: int, tp_size: int
) -> dict[str, torch.Tensor]:
    """Rows of the gathered in_proj that land in each HF projection, from the mapping's own split."""
    num_key_heads, num_value_heads, key_head_dim, value_head_dim = layout
    config = SimpleNamespace(
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        linear_key_head_dim=key_head_dim,
        linear_value_head_dim=value_head_dim,
    )
    qkvz, ba = split_gdn_linear_weights(
        config, torch.arange(rows).unsqueeze(1), tp_size=tp_size, feature_dim=1
    )
    indices = _split_gdn_grouped_to_separate(config, qkvz, ba, feature_dim=1)
    return {
        name: index.squeeze(1) for name, index in zip(("qkv", "z", "b", "a"), indices, strict=True)
    }


def _unquantized_outputs(mapping: Any) -> set[str]:
    if isinstance(mapping, GDNLinearMappingSeparate):
        return {mapping.hf_param[key] for key in _GDN_UNQUANTIZED_OUTPUTS}
    if isinstance(mapping, GDNLinearMapping):
        return {mapping.hf_param["ba"]}
    return set()


def _mamba_layout(mapping: Any, module: torch.nn.Module) -> tuple[int, int, int, int, int] | None:
    if not isinstance(mapping, MambaInProjMapping):
        return None
    config = mapping._get_config(module)
    inner = int(config.mamba_num_heads * config.mamba_head_dim)
    ssm = int(config.mamba_state_dim * config.mamba_num_groups)
    return inner, inner, ssm, ssm, int(config.mamba_num_heads)


def _mamba_row_indices(shards: list[_SourceState], tp_size: int) -> torch.Tensor:
    """Follow MambaInProjMapping's component-wise TP gather for quantization state."""
    layout = shards[0].mamba_layout
    if (
        layout is None
        or len(shards) != tp_size
        or any(shard.mamba_layout != layout for shard in shards)
    ):
        raise ValueError("Inconsistent Mamba projection geometry across TP")
    if any(size <= 0 or size % tp_size for size in layout):
        raise ValueError("Mamba projection components must be divisible by TP")
    local_sizes = [size // tp_size for size in layout]
    if any(
        len(shard.weight_shape) != 2 or shard.weight_shape[0] != sum(local_sizes)
        for shard in shards
    ):
        raise ValueError("Mamba projection rows do not match its component geometry")
    rows = torch.arange(sum(layout)).reshape(tp_size, -1)
    return torch.cat([component.reshape(-1) for component in torch.split(rows, local_sizes, dim=1)])


def _validate_quantized_task(task: WeightConversionTask) -> None:
    if isinstance(task.mapping, AutoMapping) and task.mapping.permute_dims is not None:
        raise NotImplementedError("ModelOpt export does not support dimension-permuting mappings")


def _capture_source_state(task: WeightConversionTask) -> _SourceState | None:
    if task.megatron_module is None or task.param_weight is None:
        return None
    if not isinstance(task.param_weight, torch.Tensor):
        raise NotImplementedError("ModelOpt real-quant export requires a tensor source weight")

    weight_name = _direct_weight_name(task.megatron_module, task.param_weight)
    if weight_name is None:
        return _SourceState(
            None, tuple(task.param_weight.shape), weight_dtype=task.param_weight.dtype
        )

    from modelopt.torch.export.quantized_weight_export import capture_quantized_weight_export_state

    return _SourceState(
        capture_quantized_weight_export_state(task.megatron_module, weight_name, cpu=False),
        tuple(task.param_weight.shape),
        _mapping_parallelism(task.mapping, task.megatron_module),
        _qkv_layout(task.mapping, task.megatron_module),
        task.param_weight.dtype,
        _gdn_layout(task.mapping, task.megatron_module),
        _mamba_layout(task.mapping, task.megatron_module),
    )


def _capture_source_spec(task: WeightConversionTask) -> _SourceState | None:
    if task.megatron_module is None or task.param_weight is None:
        return None
    if not isinstance(task.param_weight, torch.Tensor):
        raise NotImplementedError("ModelOpt real-quant export requires a tensor source weight")

    weight_name = _direct_weight_name(task.megatron_module, task.param_weight)
    if weight_name is None:
        return _SourceState(
            None, tuple(task.param_weight.shape), weight_dtype=task.param_weight.dtype
        )

    from modelopt.torch.export.quantized_weight_export import get_quantized_weight_export_spec

    spec = get_quantized_weight_export_spec(task.megatron_module, weight_name)
    if spec is not None:
        _validate_quantized_task(task)
    return _SourceState(
        spec,
        tuple(task.param_weight.shape),
        _mapping_parallelism(task.mapping, task.megatron_module) if spec is not None else None,
        _qkv_layout(task.mapping, task.megatron_module) if spec is not None else None,
        task.param_weight.dtype,
        _gdn_layout(task.mapping, task.megatron_module) if spec is not None else None,
        _mamba_layout(task.mapping, task.megatron_module) if spec is not None else None,
    )


def _all_gather_objects(value: Any, group: Any) -> list[Any]:
    world_size = get_pg_size(group)
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, value, group=group)
    return gathered


def _raise_distributed_errors(error: str | None, group: Any, context: str) -> None:
    world_size = get_pg_size(group)
    if world_size > 1:
        device = torch.device("cpu")
        if str(torch.distributed.get_backend(group)).lower() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL ModelOpt error synchronization requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        failed = torch.tensor(error is not None, dtype=torch.uint8, device=device)
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=group)
        if not failed.item():
            return

    errors = [item for item in _all_gather_objects(error, group) if item is not None]
    if errors:
        raise RuntimeError(f"{context}: {'; '.join(dict.fromkeys(errors))}")


def _world_group() -> Any:
    if not torch.distributed.is_initialized():
        return None
    return torch.distributed.group.WORLD


def _sync_source_specs(
    local_states: dict[str, _SourceState], group: Any
) -> dict[str, _SourceState]:
    synchronized: dict[str, _SourceState] = {}
    for rank_states in _all_gather_objects(local_states, group):
        synchronized.update(rank_states)
    return synchronized


def _transform_source_spec(
    task: WeightConversionTask, source: _SourceState
) -> dict[str, object | None]:
    names = _mapping_names(task.mapping, source.traced_rows)
    shards: list[_SourceState] = _all_gather_objects(source, task.mapping.tp_group)
    quantized = [shard.state is not None for shard in shards]
    if any(quantized) and not all(quantized):
        raise RuntimeError(
            f"Inconsistent ModelOpt quantization across TP for {task.global_param_name}"
        )
    if not any(quantized):
        if len(source.weight_shape) < 2:
            # Norms and other vectors are not deployment layers, so they stay out of the exclude list.
            return {}
        return {name: None for name in names.values() if name.endswith(".weight")}

    reference = shards[0].state
    if any(shard.state != reference for shard in shards[1:]):
        raise RuntimeError(
            f"Inconsistent ModelOpt export format across TP for {task.global_param_name}"
        )
    if isinstance(task.mapping, MambaInProjMapping) and source.traced_rows is None:
        _mamba_row_indices(shards, task.mapping.tp_size)
    unquantized = _unquantized_outputs(task.mapping)
    # Deployments read quantized linear layers from `<module>.weight`; anything else would be packed unannounced.
    if other := sorted(
        name for name in names.values() if not name.endswith(".weight") and name not in unquantized
    ):
        raise NotImplementedError(
            f"ModelOpt export requires quantized {task.global_param_name} to map to HF '.weight' tensors, not {other}"
        )
    return {
        name: None if name in unquantized else reference
        for name in names.values()
        if name.endswith(".weight")
    }


def _mapping_names(
    mapping: Any, traced_rows: tuple[tuple[str, tuple[int, ...]], ...] | None = None
) -> dict[str, str]:
    if traced_rows is not None:
        return {name: name for name, _ in traced_rows}
    if isinstance(mapping.hf_param, str):
        return {"": mapping.hf_param}
    return dict(mapping.hf_param)


def _merge_states(states: list[object], weight_dim: int) -> object:
    from modelopt.torch.export.quantized_weight_export import merge_quantized_weight_export_states

    if len(states) == 1:
        return states[0]
    return merge_quantized_weight_export_states(states, weight_dim)


def _select_state(state: object, weight_dim: int, indices: torch.Tensor) -> object:
    from modelopt.torch.export.quantized_weight_export import select_quantized_weight_export_state

    return select_quantized_weight_export_state(state, weight_dim, indices)


def _gather_source_states(
    source: _SourceState | None, mapping: Any, capture_error: str | None = None
) -> list[_SourceState]:
    from modelopt.torch.export.quantized_weight_export import (
        restore_quantized_weight_export_state,
        split_quantized_weight_export_state,
    )

    state_metadata = None
    state_tensors: tuple[torch.Tensor, ...] = ()
    tensor_specs = ()
    if source is not None and source.state is not None:
        state_metadata, state_tensors = split_quantized_weight_export_state(source.state)
        tensor_specs = tuple((tuple(tensor.shape), tensor.dtype) for tensor in state_tensors)
        state_metadata = replace(source, state=state_metadata)
    payloads = _all_gather_objects((state_metadata, tensor_specs, capture_error), mapping.tp_group)
    errors = [error for _, _, error in payloads if error is not None]
    if errors:
        raise RuntimeError("ModelOpt state capture failed: " + "; ".join(dict.fromkeys(errors)))
    source_payloads = [(metadata, specs) for metadata, specs, _ in payloads if metadata is not None]
    if len(source_payloads) != len(payloads):
        raise RuntimeError("ModelOpt source state is missing on a tensor-parallel rank")
    reference_metadata, reference_specs = source_payloads[0]
    if any(specs != reference_specs for _, specs in source_payloads[1:]):
        raise RuntimeError(
            f"Inconsistent ModelOpt state tensors for {reference_metadata.weight_shape}"
        )

    gathered_tensors = [
        mapping.gather_from_tp_ranks(
            _stage_tensor_for_collective(tensor.contiguous(), mapping.tp_group)
        )
        for tensor in state_tensors
    ]
    return [
        replace(
            source_metadata,
            state=restore_quantized_weight_export_state(
                source_metadata.state, [values[rank] for values in gathered_tensors]
            ),
        )
        for rank, (source_metadata, _) in enumerate(source_payloads)
    ]


def _transform_source_state(
    task: WeightConversionTask,
    source: _SourceState | None,
    capture_error: str | None = None,
    traced_rows: tuple[tuple[str, tuple[int, ...]], ...] | None = None,
) -> dict[str, object | None]:
    names = _mapping_names(task.mapping, traced_rows)
    shards = (
        _gather_source_states(source, task.mapping, capture_error)
        if source is None or capture_error is not None or source.state is not None
        else _all_gather_objects(source, task.mapping.tp_group)
    )
    quantized = [shard.state is not None for shard in shards]
    if any(quantized) and not all(quantized):
        raise RuntimeError(
            f"Inconsistent ModelOpt quantization across TP for {task.global_param_name}"
        )
    if not any(quantized):
        return {name: None for name in names.values() if name.endswith(".weight")}

    states = [shard.state for shard in shards]
    assert all(state is not None for state in states)
    concrete_states = [state for state in states if state is not None]
    parallelism = source.parallelism

    if parallelism == "gated":
        if set(names) != {"gate", "up"}:
            raise ValueError(f"Expected gate/up HF names for {task.global_param_name}")
        gate_states = []
        up_states = []
        for shard, state in zip(shards, concrete_states, strict=True):
            fused_size = shard.weight_shape[0]
            if fused_size % 2:
                raise ValueError(f"Expected an even gated dimension for {task.global_param_name}")
            midpoint = fused_size // 2
            gate_states.append(_select_state(state, 0, torch.arange(midpoint)))
            up_states.append(_select_state(state, 0, torch.arange(midpoint, fused_size)))
        transformed = {
            names["gate"]: _merge_states(gate_states, 0),
            names["up"]: _merge_states(up_states, 0),
        }
    else:
        if parallelism == "column":
            transformed_state = _merge_states(concrete_states, 0)
            full_shape = (sum(shard.weight_shape[0] for shard in shards), *source.weight_shape[1:])
        elif parallelism == "row":
            transformed_state = (
                concrete_states[0]
                if len(source.weight_shape) == 1
                else _merge_states(concrete_states, 1)
            )
            full_shape = (
                source.weight_shape
                if len(source.weight_shape) == 1
                else (
                    source.weight_shape[0],
                    sum(shard.weight_shape[1] for shard in shards),
                    *source.weight_shape[2:],
                )
            )
        elif parallelism == "replicated":
            transformed_state = concrete_states[0]
            full_shape = source.weight_shape
        else:
            raise NotImplementedError(
                f"Unsupported ModelOpt topology {parallelism!r} for {task.global_param_name}"
            )

        if traced_rows is not None:
            unquantized = _unquantized_outputs(task.mapping)
            transformed = {
                name: (
                    None
                    if name in unquantized
                    else _select_state(transformed_state, 0, torch.tensor(rows))
                )
                for name, rows in traced_rows
            }
        elif isinstance(task.mapping, (QKVMapping, ConcatenatedQKVMapping, QKVGMapping)):
            if source.qkv_layout is None:
                raise RuntimeError(f"Missing QKV layout for {task.global_param_name}")
            heads, groups, channels, hidden, output_gate = source.qkv_layout
            config = SimpleNamespace(
                num_attention_heads=heads,
                num_query_groups=groups,
                kv_channels=channels or None,
                hidden_size=hidden,
                attention_output_gate=output_gate,
            )
            rows = torch.arange(full_shape[0])
            if isinstance(task.mapping, QKVGMapping):
                # The mapping's own split; its gate takes one row per head.
                indices = split_qkvg_weights(config, rows.unsqueeze(1), feature_dim=1)
                transformed = {
                    names[key]: _select_state(transformed_state, 0, index.squeeze(1))
                    for key, index in zip(("q", "k", "v", "g"), indices, strict=True)
                }
            elif isinstance(task.mapping, ConcatenatedQKVMapping):
                # One HF tensor holding Q, K and V in that order, as the mapping writes it.
                transformed = {
                    task.mapping.hf_param: _select_state(
                        transformed_state, 0, torch.cat(split_qkv_weights(config, rows))
                    )
                }
            else:
                q_indices, k_indices, v_indices = split_qkv_weights(config, rows)
                transformed = {
                    names["q"]: _select_state(transformed_state, 0, q_indices),
                    names["k"]: _select_state(transformed_state, 0, k_indices),
                    names["v"]: _select_state(transformed_state, 0, v_indices),
                }
        elif isinstance(task.mapping, GDNLinearMappingSeparate):
            if source.gdn_layout is None:
                raise RuntimeError(f"Missing GatedDeltaNet layout for {task.global_param_name}")
            row_indices = _gdn_row_indices(source.gdn_layout, full_shape[0], task.mapping.tp_size)
            transformed = {
                names[key]: (
                    None
                    if key in _GDN_UNQUANTIZED_OUTPUTS
                    else _select_state(transformed_state, 0, row_indices[key])
                )
                for key in names
            }
        elif isinstance(task.mapping, MambaInProjMapping):
            transformed = {
                task.mapping.hf_param: _select_state(
                    transformed_state, 0, _mamba_row_indices(shards, task.mapping.tp_size)
                )
            }
        else:
            if len(names) != 1:
                raise ValueError(f"Expected one HF name for {task.global_param_name}")
            transformed = {next(iter(names.values())): transformed_state}

    return transformed


class _LocalExpertMappingMixin:
    @property
    def tp_group(self):
        return self._etp_group

    @property
    def is_expert(self) -> bool:
        return False


class _LocalExpertColumnMapping(_LocalExpertMappingMixin, ColumnParallelMapping):
    pass


class _LocalExpertRowMapping(_LocalExpertMappingMixin, RowParallelMapping):
    pass


class _LocalExpertReplicatedMapping(_LocalExpertMappingMixin, ReplicatedMapping):
    pass


class _LocalExpertAutoMapping(_LocalExpertMappingMixin, AutoMapping):
    def _get_or_create_mapping(self, parallelism_type: str):
        mapping_types = {
            "column": _LocalExpertColumnMapping,
            "row": _LocalExpertRowMapping,
            "replicated": _LocalExpertReplicatedMapping,
        }
        try:
            mapping_type = mapping_types[parallelism_type]
        except KeyError as error:
            raise ValueError(f"Unknown parallelism type: {parallelism_type}") from error
        mapping = mapping_type(self.megatron_param, self.hf_param)
        mapping.set_process_groups_from_pg_collection(self._pg_collection)
        return mapping


class _LocalExpertGatedMLPMapping(_LocalExpertMappingMixin, GatedMLPMapping):
    pass


def _local_expert_mapping(mapping: Any, global_param_name: str, pg_collection: Any) -> Any | None:
    specs = mapping.local_hf_param_specs(global_param_name)
    if isinstance(mapping, FusedGatedExpertMapping) or type(mapping) is GatedMLPMapping:
        if (
            len(specs) != 2
            or {spec.split_index for spec in specs} != {0, 1}
            or any(spec.split_count != 2 for spec in specs)
            or len({spec.split_dim for spec in specs}) != 1
        ):
            return None
        gate, up = sorted(specs, key=lambda spec: spec.split_index)
        replacement = _LocalExpertGatedMLPMapping(mapping.megatron_param, gate.name, up.name)
    elif isinstance(mapping, FusedExpertMapping) or type(mapping) is AutoMapping:
        if len(specs) != 1 or specs[0].split_count != 1:
            return None
        replacement = _LocalExpertAutoMapping(
            mapping.megatron_param, specs[0].name, getattr(mapping, "permute_dims", None)
        )
    elif type(mapping) in (ColumnParallelMapping, RowParallelMapping):
        if len(specs) != 1 or specs[0].split_count != 1:
            return None
        local_type = (
            _LocalExpertColumnMapping
            if type(mapping) is ColumnParallelMapping
            else _LocalExpertRowMapping
        )
        replacement = local_type(mapping.megatron_param, specs[0].name)
    else:
        return None
    replacement.set_process_groups_from_pg_collection(pg_collection)
    return replacement


def _stage_tensor_for_collective(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    if not torch.distributed.is_initialized() or get_pg_size(group) == 1:
        return tensor
    backend = str(torch.distributed.get_backend(group)).lower()
    if backend != "nccl" or tensor.device.type != "cpu":
        return tensor
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL ModelOpt expert gather requires CUDA")
    return tensor.to(torch.device("cuda", torch.cuda.current_device()))


def _capture_current_source_state(
    task: WeightConversionTask,
) -> tuple[_SourceState | None, str | None]:
    from modelopt.torch.export.quantized_weight_export import (
        restore_quantized_weight_export_state,
        split_quantized_weight_export_state,
    )

    local_source = None
    local_tensors: tuple[torch.Tensor, ...] = ()
    capture_error = None
    try:
        local_source = _capture_source_state(task)
        if local_source is not None:
            if local_source.state is None:
                raise RuntimeError(f"{task.global_param_name} is no longer quantized")
            state_metadata, local_tensors = split_quantized_weight_export_state(local_source.state)
            tensor_specs = tuple((tuple(tensor.shape), tensor.dtype) for tensor in local_tensors)
            local_source = replace(local_source, state=state_metadata)
            local_payload = (local_source, tensor_specs)
        else:
            local_payload = None
    except Exception as error:
        local_payload = None
        capture_error = f"{type(error).__name__}: {error}"
    if capture_error is not None:
        local_payload = (None, (), capture_error)
    elif local_payload is not None:
        local_payload = (*local_payload, None)

    # Carry failures through the existing PP metadata path; the following TP
    # metadata gather propagates them without a per-weight WORLD collective.
    try:
        source_metadata, tensor_specs, capture_error = task.mapping.broadcast_obj_from_pp_rank(
            local_payload
        )
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if capture_error is not None:
        return None, capture_error

    cache_prefix = f"modelopt:{task.global_param_name}"
    values = []
    for index, _ in enumerate(tensor_specs):
        local_tensor = local_tensors[index] if local_source is not None else None
        if local_tensor is not None:
            local_tensor = _stage_tensor_for_collective(
                local_tensor.contiguous(), task.mapping.pp_group
            )
        values.append(
            task.mapping.broadcast_from_pp_rank(
                local_tensor, cache_key=f"{cache_prefix}:tensor:{index}"
            )
        )
    return (
        replace(
            source_metadata,
            state=restore_quantized_weight_export_state(source_metadata.state, values),
        ),
        None,
    )


def _gather_expert_outputs(outputs: Iterable[HFWeightTuple], group: Any) -> Iterable[HFWeightTuple]:
    error = None
    # Materialize inside the error envelope so every EP rank reaches the same
    # metadata gather even when one local exporter fails.
    try:
        outputs = tuple(outputs)
        metadata = tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in outputs)
    except Exception as exception:
        outputs = ()
        metadata = ()
        error = f"{type(exception).__name__}: {exception}"
    gathered = _all_gather_objects((metadata, error), group)
    errors = [rank_error for _, rank_error in gathered if rank_error is not None]
    if errors:
        raise RuntimeError("ModelOpt expert export failed: " + "; ".join(dict.fromkeys(errors)))
    gathered_metadata = [rank_metadata for rank_metadata, _ in gathered]
    if len(gathered_metadata) == 1:
        yield from outputs
        return

    reference_layout = tuple((shape, dtype) for _, shape, dtype in gathered_metadata[0])
    if any(
        tuple((shape, dtype) for _, shape, dtype in rank_metadata) != reference_layout
        for rank_metadata in gathered_metadata[1:]
    ):
        raise RuntimeError("Inconsistent ModelOpt expert output tensors across EP")

    for index, (_, tensor) in enumerate(outputs):
        tensor = _stage_tensor_for_collective(tensor.contiguous(), group)
        local_bytes = tensor.reshape(-1).view(torch.uint8)
        gathered = [torch.empty_like(local_bytes) for _ in gathered_metadata]
        torch.distributed.all_gather(gathered, local_bytes, group=group)
        for rank_metadata, value in zip(gathered_metadata, gathered, strict=True):
            name, shape, dtype = rank_metadata[index]
            yield HFWeightTuple(name, value.view(dtype).reshape(shape))


def _compose_export_hooks(
    exporter: HFExportHook,
    finalizer: Callable[[Iterable[HFWeightTuple]], Iterable[HFWeightTuple]] | None,
) -> HFExportHook:
    if finalizer is None:
        return exporter

    def export_and_finalize(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        yield from finalizer(exporter(hf_name, tensor))

    return export_and_finalize


def _make_export_hook(
    task: WeightConversionTask, traced_rows: tuple[tuple[str, tuple[int, ...]], ...] | None = None
) -> HFExportHook:
    from modelopt.torch.export.quantized_weight_export import export_quantized_weight_tensors

    expected_names = {
        name
        for name in _mapping_names(task.mapping, traced_rows).values()
        if name.endswith(".weight")
    }
    transformed_states = None
    remaining_names = set(expected_names)

    def export_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        nonlocal remaining_names, transformed_states
        if transformed_states is None:
            source, capture_error = _capture_current_source_state(task)
            transformed_states = _transform_source_state(task, source, capture_error, traced_rows)
            remaining_names = set(expected_names)

        if hf_name not in transformed_states:
            raise RuntimeError(f"Missing ModelOpt export state for {hf_name}")
        state = transformed_states[hf_name]
        remaining_names.discard(hf_name)
        if not remaining_names:
            transformed_states = None
        if state is None:
            yield HFWeightTuple(hf_name, tensor)
            return
        prefix = hf_name.removesuffix(".weight")
        # TODO: Drop the float32 cast once nvidia-modelopt includes NVIDIA/Model-Optimizer#3005: earlier
        # versions round a per-tensor FP8 weight in its own dtype, unlike fake quantization.
        dtype = (
            torch.float32 if getattr(state, "quantization_format", None) == "fp8" else tensor.dtype
        )
        exported = export_quantized_weight_tensors(tensor, state, dtype)

        for relative_name, exported_tensor in exported.items():
            exported_name = hf_name if relative_name == "weight" else f"{prefix}.{relative_name}"
            yield HFWeightTuple(exported_name, exported_tensor)

    return export_weight


def build_modelopt_export_plan(
    conversion_tasks: list[WeightConversionTask | None],
    *,
    model: list[torch.nn.Module],
    tied_weight_aliases: dict[str, str] | None = None,
) -> ModelOptExportPlan:
    """Prepare canonical ModelOpt export state without encoding a quantization format.

    Args:
        conversion_tasks: Weight conversion tasks, including remote pipeline-stage tasks.
        model: Megatron model chunks used to capture quantization state.
        tied_weight_aliases: HF alias-to-source weight names that share export specifications
            even when the alias has no separate conversion task.
    """
    # ModelOpt converts quantizer state into deployment scales from the owning
    # weight task. Amax mappings are only part of the fake-quant refit path.
    concrete_tasks = [
        task
        for task in conversion_tasks
        if task is not None and not isinstance(getattr(task, "mapping", None), AmaxMapping)
    ]
    pg_collection = model_bridge_utils._get_pg_collection_from_model(model)
    local_expert_mappings = {}
    planning_tasks = concrete_tasks
    local_specs = {}
    capture_error = None
    try:
        for task in concrete_tasks:
            mapping = getattr(task, "mapping", None)
            if mapping is None or not mapping.is_expert:
                continue
            local_mapping = _local_expert_mapping(mapping, task.global_param_name, pg_collection)
            if local_mapping is not None:
                local_expert_mappings[task.global_param_name] = local_mapping
        planning_tasks = [
            (
                replace(task, mapping=local_expert_mappings[task.global_param_name])
                if task.global_param_name in local_expert_mappings
                else task
            )
            for task in concrete_tasks
        ]
        local_specs = {
            task.global_param_name: source
            for task in planning_tasks
            if (source := _capture_source_spec(task)) is not None
        }
    except Exception as error:
        capture_error = f"{type(error).__name__}: {error}"
    _raise_distributed_errors(
        capture_error, _world_group(), "ModelOpt export metadata capture failed"
    )

    # TP ranks of a pipeline stage hold the same tasks, but they must also agree on which ones are
    # quantized before entering the mappings' TP collectives.
    trace_error = None
    traced_tasks = [
        task
        for task in planning_tasks
        if task.global_param_name in local_specs and _traces_rows(task.mapping)
    ]
    quantized = {
        task.global_param_name
        for task in traced_tasks
        if local_specs[task.global_param_name].state is not None
    }
    if traced_tasks:
        quantized_by_rank = _all_gather_objects(quantized, traced_tasks[0].mapping.tp_group)
        quantized = set.intersection(*quantized_by_rank)
        if mismatched := set.union(*quantized_by_rank) - quantized:
            trace_error = f"Inconsistent ModelOpt quantization across TP for {min(mismatched)}"
    for task in traced_tasks:
        source = local_specs[task.global_param_name]
        if trace_error is not None or task.global_param_name not in quantized:
            continue
        try:
            local_specs[task.global_param_name] = replace(
                source, traced_rows=_trace_rows(task, source.parallelism)
            )
        except Exception as error:
            trace_error = trace_error or f"{type(error).__name__}: {error}"
    _raise_distributed_errors(trace_error, _world_group(), "ModelOpt export row tracing failed")

    pp_group = (
        model_bridge_utils._get_pp_group(model) if torch.distributed.is_initialized() else None
    )
    source_specs = _sync_source_specs(local_specs, pp_group)

    expert_mapping_error = None
    try:
        for task in concrete_tasks:
            source = source_specs.get(task.global_param_name)
            if source is None or source.state is None or not task.mapping.is_expert:
                continue
            local_mapping = local_expert_mappings.get(task.global_param_name)
            if local_mapping is None:
                raise NotImplementedError(
                    "ModelOpt real-quant export cannot pack expert mapping "
                    f"{type(task.mapping).__name__} before its EP gather"
                )
    except Exception as error:
        expert_mapping_error = f"{type(error).__name__}: {error}"
    _raise_distributed_errors(
        expert_mapping_error, _world_group(), "ModelOpt expert mapping validation failed"
    )

    named_specs: dict[str, object | None] = {}
    fused_expert_specs: dict[str, list[tuple[str, object | None]]] = {}
    for task, planning_task in zip(concrete_tasks, planning_tasks, strict=True):
        source = source_specs.get(task.global_param_name)
        if source is None:
            continue
        transformed = None
        transform_error = None
        try:
            transformed = _transform_source_spec(planning_task, source)
        except Exception as error:
            transform_error = f"{type(error).__name__}: {error}"

        if task.mapping.is_expert:
            expert_specs: dict[str, object | None] = {}
            gathered = _all_gather_objects((transformed, transform_error), task.mapping.ep_group)
            errors = [error for _, error in gathered if error is not None]
            if errors:
                raise RuntimeError(
                    "ModelOpt expert metadata transform failed: " + "; ".join(dict.fromkeys(errors))
                )
            for rank_specs, _ in gathered:
                assert rank_specs is not None
                overlap = expert_specs.keys() & rank_specs.keys()
                if overlap:
                    raise RuntimeError(f"Duplicate ModelOpt expert specs: {sorted(overlap)}")
                expert_specs.update(rank_specs)
            transformed = expert_specs
        elif transform_error is not None:
            raise RuntimeError(
                f"ModelOpt metadata transform failed for {task.global_param_name}: {transform_error}"
            )

        assert transformed is not None
        if task.global_param_name in local_expert_mappings and getattr(
            task.mapping, "is_grouped_export", False
        ):
            module_names = {name.rsplit(".", 3)[0] for name in transformed}
            if len(module_names) != 1:
                raise RuntimeError(
                    f"Unable to identify one fused expert module for {task.global_param_name}: {sorted(transformed)}"
                )
            module_name = module_names.pop()
            fused_expert_specs.setdefault(module_name, []).extend(transformed.items())
        for name, spec in transformed.items():
            if name in named_specs and named_specs[name] != spec:
                raise RuntimeError(f"Duplicate ModelOpt spec for {name}")
            named_specs[name] = spec

    for module_name, named_module_specs in fused_expert_specs.items():
        reference = named_module_specs[0][1]
        if any(spec != reference for _, spec in named_module_specs[1:]):
            names = sorted(name for name, _ in named_module_specs)
            raise RuntimeError(
                f"Inconsistent ModelOpt export specs for fused expert module {module_name}: {names}"
            )

    if not any(spec is not None for spec in named_specs.values()):
        raise RuntimeError("No supported ModelOpt quantized weights were found")

    from modelopt.torch.export.quantized_weight_export import build_hf_quantization_config

    # Tied output weights may have no task, but deployment still needs their spec.
    # Copy explicit None too: a BF16 embedding must keep its tied head unquantized.
    for alias, source_name in (tied_weight_aliases or {}).items():
        if source_name not in named_specs:
            raise ValueError(f"Missing ModelOpt export spec for tied source {source_name}")
        source_spec = named_specs[source_name]
        if alias in named_specs and named_specs[alias] != source_spec:
            raise ValueError(f"Inconsistent ModelOpt export spec for tied weight {alias}")
        named_specs[alias] = source_spec

    quantization_config = build_hf_quantization_config(named_specs)

    export_tasks = []
    for task in concrete_tasks:
        source = source_specs.get(task.global_param_name)
        mapping = task.mapping
        if task.mapping.is_expert:
            eligibility = _all_gather_objects(
                source is not None and source.state is not None, task.mapping.ep_group
            )
            if any(eligibility) and not all(eligibility):
                raise RuntimeError(
                    f"Inconsistent ModelOpt quantization across EP for {task.global_param_name}"
                )
        else:
            eligibility = [False]
        if all(eligibility):
            local_mapping = local_expert_mappings.get(task.global_param_name)
            if local_mapping is not None:
                mapping = local_mapping
        # Keep source-checkpoint hooks for key/padding fixes, but opt out of their
        # requantization: ModelOpt packs the logical weight after HF conversion.
        weight_dtype = task.weight_dtype
        if weight_dtype is None and source is not None:
            weight_dtype = source.weight_dtype
        export_tasks.append(
            replace(task, mapping=mapping, export_hook=None, weight_dtype=weight_dtype)
        )

    quantized_params = frozenset(
        name for name, source in source_specs.items() if source.state is not None
    )
    return ModelOptExportPlan(
        export_tasks,
        quantization_config,
        quantized_params,
        {name: source.state for name, source in local_specs.items()},
        {
            name: source.traced_rows
            for name, source in source_specs.items()
            if source.traced_rows is not None
        },
    )


def prepare_modelopt_export_tasks(plan: ModelOptExportPlan) -> list[WeightConversionTask]:
    """Attach fresh per-stream state capture hooks to a metadata-only plan."""
    validation_error = None
    try:
        for task in plan.conversion_tasks:
            if task.global_param_name not in plan.local_quantization_specs:
                continue
            source = _capture_source_spec(task)
            if (
                source is None
                or source.state != plan.local_quantization_specs[task.global_param_name]
            ):
                raise ValueError(
                    f"ModelOpt quantization configuration changed for {task.global_param_name}"
                )
    except Exception as error:
        validation_error = f"{type(error).__name__}: {error}"
    # Validate once per stream, including newly enabled weights. Every rank must
    # reject a stale deployment config before entering weight-conversion collectives.
    _raise_distributed_errors(
        validation_error,
        _world_group(),
        "ModelOpt export plan validation failed; rebuild the plan and reinitialize the consumer",
    )

    export_tasks = []
    for task in plan.conversion_tasks:
        if task.global_param_name not in plan.quantized_params:
            export_tasks.append(task)
            continue
        _validate_quantized_task(task)

        finalizer = None
        if isinstance(task.mapping, _LocalExpertMappingMixin):
            group = task.mapping.ep_group

            def finalize(outputs: Iterable[HFWeightTuple], group=group):
                yield from _gather_expert_outputs(outputs, group)

            finalizer = finalize
        export_tasks.append(
            replace(
                task,
                export_hook=_compose_export_hooks(
                    _make_export_hook(task, plan.traced_rows.get(task.global_param_name)), finalizer
                ),
            )
        )
    return export_tasks


# HF projections that vLLM / TRT-LLM fuse into one GEMM, which takes a single scale per tensor.
_FUSED_SIBLING_PROJECTIONS = (
    frozenset({"q_proj", "k_proj", "v_proj"}),
    frozenset({"gate_proj", "up_proj"}),
    frozenset({"q_a_proj", "kv_a_proj_with_mqa"}),
)
_SHARED_SIBLING_SCALES = ("weight_scale_2", "input_scale")
_EXPORTED_QUANTIZER = re.compile(r"(?:^|\.)(?:\w*weight_quantizer|input_quantizer)(?:\.\d+)?$")
_WEIGHT_QUANTIZER = re.compile(r"(?:^|\.)\w*weight_quantizer(?:\.\d+)?$")


def check_fused_sibling_scales(tensors: Mapping[str, torch.Tensor]) -> None:
    """Reject per-tensor scales that differ within a group of projections deployments fuse.

    Args:
        tensors: Exported HF tensors; only the ``weight_scale_2`` and ``input_scale`` entries are read.

    Raises:
        ValueError: If siblings such as ``q_proj`` / ``k_proj`` / ``v_proj`` carry different scales.
    """
    groups: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        module, _, scale = name.rpartition(".")
        parent, _, projection = module.rpartition(".")
        if scale not in _SHARED_SIBLING_SCALES:
            continue
        for index, siblings in enumerate(_FUSED_SIBLING_PROJECTIONS):
            if projection in siblings:
                groups.setdefault((parent, index, scale), {})[name] = tensor
    mismatched = [
        sorted(members)
        for members in groups.values()
        if any(not torch.equal(tensor, next(iter(members.values()))) for tensor in members.values())
    ]
    if mismatched:
        raise ValueError(
            f"Fused projections need equal per-tensor scales, but these differ: {mismatched}"
        )


def check_exportable_quantizers(model: list[torch.nn.Module]) -> None:
    """Fail on every rank when any rank holds an enabled quantizer the export would drop.

    Only weight and input quantizers become checkpoint tensors; e.g. KV-cache (``k_bmm_quantizer``)
    and output quantizers have no deployment format here yet, and an input quantizer is exported only
    with its layer's weight quantizer.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    unsupported = set()
    for chunk in model:
        enabled = [
            name
            for name, module in chunk.named_modules()
            if isinstance(module, TensorQuantizer) and module.is_enabled
        ]
        unsupported.update(name for name in enabled if not _EXPORTED_QUANTIZER.search(name))
        weight_layers = {
            _WEIGHT_QUANTIZER.sub("", name) for name in enabled if _WEIGHT_QUANTIZER.search(name)
        }
        unsupported.update(
            name
            for name in enabled
            if name.endswith("input_quantizer") and name.rpartition(".")[0] not in weight_layers
        )
    unsupported = sorted(unsupported)
    error = None
    if unsupported:
        error = f"ModelOpt HF export cannot carry these enabled quantizers: {unsupported[:4]}"
    _raise_distributed_errors(error, _world_group(), "ModelOpt export failed")


def save_safetensors_shards(
    tensors: Iterable[HFWeightTuple], path: Path, *, max_shard_size: int
) -> set[str]:
    """Stream tensors into HF safetensors shards of at most ``max_shard_size`` bytes each.

    Unlike the source-checkpoint layout, this accepts tensors the source does not have, such as
    quantization scales. A single shard is written as ``model.safetensors`` without an index.

    Returns:
        Names of the files written, relative to ``path``.
    """
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    shards: list[tuple[Path, list[str]]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = total_size = 0

    def flush() -> None:
        temporary = path / f".modelopt-shard-{len(shards):05d}.safetensors.tmp"
        save_file(current, temporary, metadata={"format": "pt"})
        shards.append((temporary, list(current)))

    seen: set[str] = set()
    for name, tensor in tensors:
        if name in seen:
            raise ValueError(f"Duplicate exported tensor {name}")
        seen.add(name)
        size = tensor.numel() * tensor.element_size()
        if current and current_size + size > max_shard_size:
            flush()
            current, current_size = {}, 0
        current[name] = tensor.contiguous()
        current_size += size
        total_size += size
    if current or not shards:
        flush()

    if len(shards) == 1:
        shards[0][0].replace(path / "model.safetensors")
        return {"model.safetensors"}
    weight_map = {}
    for index, (temporary, names) in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        temporary.replace(path / filename)
        weight_map.update(dict.fromkeys(names, filename))
    index_file = "model.safetensors.index.json"
    (path / index_file).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        + "\n"
    )
    return {*weight_map.values(), index_file}
