# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in Dynamic Context Parallel plugin for the MLite runtime.

The plugin is installed on one runtime instance.  It leaves the ordinary
``build_model`` and ``forward_backward`` implementations untouched when it is
disabled.  When enabled, callers observe logical DP=1 while the model keeps its
physical DP group for gradient synchronization.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from functools import wraps
from typing import Any

import torch
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context

_SAMPLE_IDS = "_mlite_dcp_sample_ids"
_GROUP_LEADER = "_mlite_dcp_group_leader"
_LOCAL_CP_SIZE = "_mlite_dcp_local_cp_size"
_GLOBAL_NUM_TOKENS = "_mlite_dcp_global_num_tokens"
_LOSS_SCALE_CORRECTION = "_mlite_dcp_loss_scale_correction"


def _positive_power_of_two(value: int, name: str) -> int:
    if type(value) is not int or value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value!r}.")
    return value


def _create_groups(ps: Any, minimum: int, parallel: Any) -> dict[int, Any]:
    try:
        from megatron.core.parallel_state import create_dynamic_dp_cp_groups
    except ImportError as exc:
        raise RuntimeError(
            "Dynamic CP requires Megatron-Core dynamic group APIs."
        ) from exc

    rank = torch.distributed.get_rank()
    logical_dp_group = None
    for member in range(torch.distributed.get_world_size()):
        group = torch.distributed.new_group([member])
        if member == rank:
            logical_dp_group = group
    tp, cp, pp = int(parallel.tp), int(parallel.cp), int(parallel.pp)
    dp = int(ps.dp_size)
    pool_size = dp * cp

    def global_rank(tp_rank: int, cp_rank: int, dp_rank: int, pp_rank: int) -> int:
        return ((pp_rank * dp + dp_rank) * cp + cp_rank) * tp + tp_rank

    local: dict[int, Any] = {}
    # Every rank makes the same ordered calls; this is required by new_group.
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            ranks = [
                global_rank(tp_rank, cp_rank, dp_rank, pp_rank)
                for dp_rank in range(dp)
                for cp_rank in range(cp)
            ]
            groups = create_dynamic_dp_cp_groups(
                rank, ranks, pg_options=None, min_cp_size=minimum
            )
            if rank in ranks:
                local.update(groups)
                local[pool_size] = ps.dp_cp_group
    if not local:
        raise RuntimeError("Dynamic CP group initialization omitted the local rank.")
    if logical_dp_group is None:
        raise RuntimeError("Dynamic CP logical DP group omitted the local rank.")
    local[1] = logical_dp_group
    return local


def _batch_samples(batch: PackedBatch) -> list[dict[str, Any]]:
    if not isinstance(batch, PackedBatch):
        raise TypeError("Dynamic CP accepts only text-only PackedBatch inputs.")
    lengths = batch.seq_lens
    if lengths.ndim != 1 or lengths.numel() == 0 or int(lengths.min().item()) < 1:
        raise ValueError("Dynamic CP requires non-empty positive 1-D sequence lengths.")
    total = int(lengths.sum().item())
    for name in ("input_ids", "labels"):
        value = getattr(batch, name)
        if value.ndim != 1 or value.numel() != total:
            raise ValueError(f"PackedBatch.{name} must contain {total} packed tokens.")
    for name in ("loss_mask", "position_ids"):
        value = getattr(batch, name)
        if value is not None and (value.ndim != 1 or value.numel() != total):
            raise ValueError(f"PackedBatch.{name} must contain {total} packed tokens.")
    routes = batch.routed_experts
    replay_mask = batch.r3_replay_mask
    if (routes is None) != (replay_mask is None):
        raise ValueError(
            "Dynamic CP router replay requires routed_experts and r3_replay_mask together."
        )
    route_rows = None
    if routes is not None:
        if not getattr(routes, "is_nested", False):
            raise TypeError("Dynamic CP router replay requires jagged routed_experts.")
        route_rows = list(routes.unbind())
        if len(route_rows) != len(lengths):
            raise ValueError(
                "Dynamic CP routed_experts and seq_lens must have the same sample count."
            )

    positions = batch.make_position_ids()
    mask = batch.loss_mask
    samples = []
    start = 0
    for raw_length in lengths.tolist():
        length = int(raw_length)
        end = start + length
        samples.append(
            {
                "input_ids": batch.input_ids[start:end],
                "labels": batch.labels[start:end],
                "loss_mask": None if mask is None else mask[start:end],
                "position_ids": positions[start:end],
                "routed_experts": (
                    None if route_rows is None else route_rows[len(samples)]
                ),
                "r3_replay_mask": (
                    None if replay_mask is None else replay_mask[start:end]
                ),
            }
        )
        start = end
    return samples


def _detach_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    move = getattr(value, "to", None)
    if callable(move):
        try:
            return move("cpu")
        except (RuntimeError, TypeError, ValueError):
            pass
    return value


@dataclass(frozen=True)
class _NestedSourceSample:
    value: Any
    nested_keys: tuple[Any, ...]


def _nested_source_keys(source: Any) -> tuple[Any, ...]:
    keys = getattr(source, "keys", None)
    get = getattr(source, "get", None)
    if not callable(keys) or not callable(get) or not hasattr(source, "batch_size"):
        return ()
    return tuple(key for key in keys() if getattr(get(key), "is_nested", False))


def _split_source(source: Any, count: int) -> list[Any]:
    if source is None:
        return [None] * count
    try:
        if len(source) != count:
            raise ValueError(
                "LossContext.source_batch and PackedBatch must have the same sample count."
            )
    except TypeError as exc:
        raise TypeError(
            "Dynamic CP requires indexable LossContext.source_batch."
        ) from exc
    if isinstance(source, list):
        return [[_detach_cpu(item)] for item in source]
    nested_keys = _nested_source_keys(source)
    if nested_keys:
        return [
            _NestedSourceSample(_detach_cpu(source[index]), nested_keys)
            for index in range(count)
        ]
    return [_detach_cpu(source[index : index + 1]) for index in range(count)]


def _same_policy(left: LossContext, right: LossContext) -> bool:
    for item in fields(LossContext):
        if item.name == "source_batch":
            continue
        lhs, rhs = getattr(left, item.name), getattr(right, item.name)
        if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
            if not torch.equal(torch.as_tensor(lhs), torch.as_tensor(rhs)):
                return False
        elif lhs != rhs:
            return False
    return True


def _source_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_source_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_source_value_equal(lhs, rhs) for lhs, rhs in zip(left, right))
        )
    result = left == right
    return result if isinstance(result, bool) else False


def _merge_source(parts: list[Any], device: torch.device) -> Any:
    if not parts or parts[0] is None:
        return None
    if isinstance(parts[0], list):
        return [item for part in parts for item in part]
    if isinstance(parts[0], _NestedSourceSample):
        if any(not isinstance(part, _NestedSourceSample) for part in parts):
            raise TypeError("Dynamic CP cannot mix nested and ordinary source samples.")
        nested_keys = parts[0].nested_keys
        if any(part.nested_keys != nested_keys for part in parts[1:]):
            raise ValueError(
                "Dynamic CP cannot merge incompatible nested source schemas."
            )
        samples = [part.value for part in parts]
        keys = tuple(samples[0].keys())
        expected_keys = set(keys)
        if any(set(sample.keys()) != expected_keys for sample in samples[1:]):
            raise ValueError("Dynamic CP cannot merge incompatible nested source keys.")
        data = {}
        for key in keys:
            values = [sample.get(key) for sample in samples]
            if key in nested_keys:
                data[key] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            elif all(isinstance(value, torch.Tensor) for value in values):
                data[key] = torch.stack(values)
            else:
                # Integer indexing turns TensorDict non-tensor fields into
                # scalar wrappers.  Re-stacking equal broadcast metadata would
                # change it into a per-sample stack (for example, a mode enum),
                # so retain the scalar representation when every sample agrees.
                plain = [
                    (
                        value.data
                        if hasattr(value, "batch_size")
                        and hasattr(value, "data")
                        and not isinstance(value, torch.Tensor)
                        else value
                    )
                    for value in values
                ]
                data[key] = (
                    values[0]
                    if all(_source_value_equal(value, plain[0]) for value in plain[1:])
                    else plain
                )
        constructor = getattr(type(samples[0]), "from_dict", None)
        if not callable(constructor):
            raise TypeError(
                "Dynamic CP nested source container cannot be reconstructed."
            )
        return constructor(data, batch_size=[len(samples)]).to(device)
    try:
        merged = torch.cat(parts, dim=0)
    except (RuntimeError, TypeError) as exc:
        raise TypeError("Dynamic CP cannot merge source_batch samples.") from exc
    move = getattr(merged, "to", None)
    return move(device) if callable(move) else merged


def _select_context(
    contexts: list[tuple[LossContext | None, Any]], ids: list[int], device: torch.device
) -> LossContext | None:
    selected = [contexts[index] for index in ids]
    policies = [context for context, _ in selected if context is not None]
    if not policies:
        return None
    if len(policies) != len(selected):
        raise RuntimeError(
            "Dynamic CP cannot mix samples with and without LossContext."
        )
    first = policies[0]
    if any(not _same_policy(first, other) for other in policies[1:]):
        raise ValueError(
            "Dynamic CP cannot merge incompatible per-sample loss policies."
        )
    return replace(
        first,
        source_batch=_merge_source([source for _, source in selected], device),
    )


def _select_batch(
    samples: list[dict[str, Any]],
    ids: list[int],
    *,
    cp_size: int,
    leader: bool,
) -> PackedBatch:
    selected = [samples[index] for index in ids]

    def combine(name: str) -> torch.Tensor | None:
        values = [sample[name] for sample in selected]
        if values[0] is None:
            if any(value is not None for value in values):
                raise ValueError(f"Dynamic CP cannot mix missing and present {name}.")
            return None
        return torch.cat(values)

    def combine_routes() -> torch.Tensor | None:
        values = [sample["routed_experts"] for sample in selected]
        if values[0] is None:
            if any(value is not None for value in values):
                raise ValueError(
                    "Dynamic CP cannot mix missing and present routed_experts."
                )
            return None
        if any(value is None for value in values):
            raise ValueError(
                "Dynamic CP cannot mix missing and present routed_experts."
            )
        return torch.nested.as_nested_tensor(values, layout=torch.jagged)

    device = selected[0]["input_ids"].device
    return PackedBatch(
        input_ids=combine("input_ids"),
        labels=combine("labels"),
        seq_lens=torch.tensor(
            [sample["input_ids"].numel() for sample in selected],
            dtype=torch.int64,
            device=device,
        ),
        loss_mask=combine("loss_mask"),
        position_ids=combine("position_ids"),
        routed_experts=combine_routes(),
        r3_replay_mask=combine("r3_replay_mask"),
        extras={_SAMPLE_IDS: ids, _LOCAL_CP_SIZE: cp_size, _GROUP_LEADER: leader},
    )


def _context_parallel_modules(model: Any) -> list[Any]:
    roots = model if isinstance(model, (list, tuple)) else [model]
    found = []
    seen = set()
    for root in roots:
        modules = getattr(root, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            if id(module) in seen:
                continue
            setter = getattr(module, "set_context_parallel_group", None)
            if callable(setter) and all(
                hasattr(module, name)
                for name in (
                    "cp_group",
                    "cp_global_ranks",
                    "cp_stream",
                    "cp_comm_type",
                )
            ):
                found.append(module)
                seen.add(id(module))
    return found


class _Binding:
    def __init__(
        self, ps: Any, groups: dict[int, Any], context_parallel_modules: list[Any]
    ):
        self.ps = ps
        self.groups = groups
        self.static = (ps.cp_size, ps.cp_rank, ps.cp_group)
        self.had_global_ranks = hasattr(ps, "cp_global_ranks")
        self.static_global_ranks = getattr(ps, "cp_global_ranks", None)
        self.context_parallel_modules = [
            (
                module,
                module.cp_group,
                module.cp_global_ranks,
                module.cp_stream,
                module.cp_comm_type,
            )
            for module in context_parallel_modules
        ]
        self.local_cp_size: int | None = None

    def bind(self, size: int) -> None:
        self.restore()
        group = self.groups.get(size)
        if group is None or group.size() != size:
            raise RuntimeError(f"No Dynamic CP group exists for size={size}.")
        global_ranks = None
        if self.had_global_ranks or self.context_parallel_modules:
            global_ranks = list(torch.distributed.get_process_group_ranks(group))
            self.ps.cp_global_ranks = global_ranks
        self.ps.cp_size, self.ps.cp_rank, self.ps.cp_group = size, group.rank(), group
        for module, _group, _ranks, stream, comm_type in self.context_parallel_modules:
            module.set_context_parallel_group(group, global_ranks, stream, comm_type)
        self.local_cp_size = size

    def restore(self) -> None:
        if self.local_cp_size is not None:
            self.ps.cp_size, self.ps.cp_rank, self.ps.cp_group = self.static
            if self.had_global_ranks:
                self.ps.cp_global_ranks = self.static_global_ranks
            elif hasattr(self.ps, "cp_global_ranks"):
                del self.ps.cp_global_ranks
            for (
                module,
                group,
                ranks,
                stream,
                comm_type,
            ) in self.context_parallel_modules:
                module.set_context_parallel_group(group, ranks, stream, comm_type)
            self.local_cp_size = None


class _BoundIterator:
    def __init__(self, items: Iterator[Any], binding: _Binding):
        self.items = items
        self.binding = binding
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.binding.restore()
        item = next(self.items)
        batch, _ = split_loss_context(item)
        self.binding.bind(int(batch.extras[_LOCAL_CP_SIZE]))
        self.consumed += 1
        return item


class _ReplayProtocolBinding:
    """Bind the scheduled CP group before router replay packs one microbatch."""

    def __init__(self, protocol: Any, binding: _Binding):
        self._protocol = protocol
        self._binding = binding

    def __getattr__(self, name: str) -> Any:
        if self._protocol is None:
            raise AttributeError(name)
        return getattr(self._protocol, name)

    def _bind(self, batch: PackedBatch) -> None:
        self._binding.bind(int(batch.extras[_LOCAL_CP_SIZE]))

    def pack_routed_experts(self, model: Any, batch: PackedBatch, routed: Any):
        self._bind(batch)
        if self._protocol is not None:
            pack = getattr(self._protocol, "pack_routed_experts", None)
            if callable(pack):
                return pack(model, batch, routed)
        from megatron.lite.model import protocol_utils

        return protocol_utils.pack_routed_experts(model, batch, routed)

    def pack_r3_replay_mask(self, model: Any, batch: PackedBatch):
        self._bind(batch)
        if self._protocol is not None:
            pack = getattr(self._protocol, "pack_r3_replay_mask", None)
            if callable(pack):
                return pack(model, batch)
        from megatron.lite.model import protocol_utils

        return protocol_utils.pack_r3_replay_mask(model, batch)


def _router_replay_requested(spec: Any) -> bool:
    if not spec:
        return False
    action = spec.get("action") if isinstance(spec, Mapping) else spec
    return action not in (None, "disabled")


def _reject_fused_router_replay(model: Any) -> None:
    roots = model if isinstance(model, (list, tuple)) else [model]
    for root in roots:
        modules = getattr(root, "modules", None)
        candidates = modules() if callable(modules) else (root,)
        if any(
            bool(getattr(module, "moe_router_fusion", False)) for module in candidates
        ):
            raise NotImplementedError(
                "Dynamic CP router replay requires moe_router_fusion=False because "
                "the fused router path bypasses the replay hook."
            )


def _record_output(
    output: dict[str, Any],
    batch: PackedBatch,
    loss: torch.Tensor,
    metrics: dict[str, Any],
    extractor: Callable[[dict[str, Any]], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    ids = list(batch.extras[_SAMPLE_IDS])
    parts_by_key: dict[str, list[torch.Tensor]] = {}
    for key, value in extractor(output).items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Runtime-collected output {key!r} must be a tensor.")
        parts = list(value.unbind())
        if len(parts) != len(ids):
            raise RuntimeError(f"Output {key!r} has {len(parts)} rows for ids {ids}.")
        parts_by_key[key] = [part.detach().cpu() for part in parts]
    return {
        "sample_ids": ids,
        "model_output": parts_by_key,
        "loss": float(loss.detach().item()),
        "metrics": _detach_cpu(metrics),
    }


def _merge_metric_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge leader metrics without importing an application metric type."""
    merged: dict[str, Any] = {}
    keys = {key for record in records for key in record["metrics"]}
    for key in keys:
        values = [
            record["metrics"][key] for record in records if key in record["metrics"]
        ]
        first = values[0]
        init_list = getattr(first, "init_list", None)
        if callable(init_list):
            aggregate = init_list()
            for value in values:
                aggregate.append(value)
            merged[key] = aggregate
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        elif any(isinstance(value, list) for value in values):
            raise TypeError(
                f"Dynamic CP metric {key!r} mixes list and scalar leader values."
            )
        else:
            merged[key] = values
    return merged


def _restore_outputs(
    collector: list[dict[str, Any]] | None,
    records: list[dict[str, Any]],
    *,
    pool: Any,
    input_groups: list[list[int]],
    device: torch.device,
) -> None:
    gathered: list[list[dict[str, Any]] | None] = [None] * pool.size()
    torch.distributed.all_gather_object(gathered, records, group=pool)
    if collector is None:
        return
    by_id: dict[int, tuple[dict[str, Any], int]] = {}
    for rank_records in gathered:
        for record in rank_records or []:
            for part, sample_id in enumerate(record["sample_ids"]):
                if sample_id in by_id:
                    raise RuntimeError(
                        f"Dynamic CP collected sample {sample_id} twice."
                    )
                by_id[sample_id] = (record, part)

    for ids in input_groups:
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise RuntimeError(f"Dynamic CP output is missing sample ids {missing}.")
        key_sets = [set(by_id[sample_id][0]["model_output"]) for sample_id in ids]
        keys = key_sets[0]
        if any(candidate != keys for candidate in key_sets[1:]):
            raise RuntimeError(
                f"Dynamic CP samples {ids} have incompatible model output keys."
            )
        model_output = {}
        for key in keys:
            rows = [
                by_id[sample_id][0]["model_output"][key][by_id[sample_id][1]]
                for sample_id in ids
            ]
            model_output[key] = torch.nested.as_nested_tensor(
                rows, layout=torch.jagged, device=device
            )
        contributing = {
            id(by_id[sample_id][0]): by_id[sample_id][0] for sample_id in ids
        }
        values = list(contributing.values())
        collector.append(
            {
                "model_output": model_output,
                "loss": sum(item["loss"] for item in values),
                "metrics": _merge_metric_rows(values),
            }
        )


@dataclass(slots=True)
class _Prepared:
    data: Iterator[Any]
    count: int
    forward: Callable
    loss: Callable | None
    pre_forward: Callable | None
    binding: _Binding
    consumed: _BoundIterator
    records: list[dict[str, Any]]
    collector: list[dict[str, Any]] | None
    collect_outputs: bool
    pool: Any
    input_groups: list[list[int]]
    device: torch.device

    def finish(self, require_complete: bool) -> None:
        self.binding.restore()
        if not require_complete:
            return
        if self.consumed.consumed != self.count:
            raise RuntimeError(
                f"Dynamic CP prepared {self.count} microbatches but consumed "
                f"{self.consumed.consumed}."
            )
        if self.collect_outputs:
            _restore_outputs(
                self.collector,
                self.records,
                pool=self.pool,
                input_groups=self.input_groups,
                device=self.device,
            )


class DynamicCPPlugin:
    """Runtime-instance sidecar implementing logical-DP=1 Dynamic CP."""

    def __init__(
        self,
        config: Mapping[str, Any],
        create_groups: Callable | None = None,
    ):
        if not isinstance(config, Mapping):
            raise TypeError("dynamic_context_parallel plugin config must be a mapping.")
        max_length = config.get("max_seqlen_per_dp_cp_rank")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("Dynamic CP requires max_seqlen_per_dp_cp_rank >= 1.")
        self.max_length = max_length
        self.minimum = _positive_power_of_two(
            int(config.get("min_context_parallel_size", 1)),
            "min_context_parallel_size",
        )
        require_coverage = config.get("require_full_cp_size_coverage", False)
        if type(require_coverage) is not bool:
            raise TypeError("require_full_cp_size_coverage must be a bool.")
        self.require_full_coverage = require_coverage
        self._create_groups = create_groups or _create_groups
        self._groups: dict[int, Any] | None = None
        self._pool: Any = None
        self._context_parallel_modules: list[Any] = []
        self._step = 0

    def initialize(self, handle: Any) -> Any:
        ps = handle._parallel_state
        parallel = handle.config.parallel
        if not bool(handle.config.impl_cfg.get("use_thd", True)):
            raise ValueError("Dynamic CP requires packed THD inputs.")
        if int(parallel.vpp) != 1:
            raise NotImplementedError("Dynamic CP requires VPP=1.")
        pool = ps.dp_cp_group
        pool_size = int(ps.dp_size) * int(ps.cp_size)
        if pool is None or pool.size() != pool_size or pool_size < 2 or pool_size & 1:
            raise ValueError("Dynamic CP requires an even initialized DPxCP pool.")
        if self.minimum > pool_size:
            raise ValueError("Dynamic CP minimum group size exceeds the DPxCP pool.")
        groups = self._create_groups(ps, self.minimum, parallel)
        cp_size_space = []
        size = self.minimum
        while size <= pool_size:
            cp_size_space.append(size)
            size *= 2
        missing_groups = [size for size in cp_size_space if size not in groups]
        if missing_groups:
            raise RuntimeError(
                f"Dynamic CP groups omit cp_size values {missing_groups}."
            )
        logical = groups.get(1)
        if logical is None or logical.size() != 1:
            raise RuntimeError("Dynamic CP requires a singleton logical DP group.")
        # Transformer Engine defaults to unbatched CP P2P for cp_size > 2.
        # That creates pair communicators lazily, whose creation order differs
        # when ranks enter dynamic subgroups in different plans.  The batched
        # ring call initializes the complete ProcessGroup collectively.
        os.environ["NVTE_BATCH_MHA_P2P_COMM"] = "1"
        self._groups, self._pool = groups, pool
        self._context_parallel_modules = _context_parallel_modules(handle._model)
        self._cp_size_space = cp_size_space
        handle._extras.update(
            logical_dp_size=1,
            logical_dp_rank=0,
            logical_dp_group=logical,
            metric_group=pool,
            cp_range=(self.minimum, pool_size),
        )
        return handle

    def _prepare(
        self,
        handle: Any,
        data: Any,
        loss_fn: Callable | None,
        num_microbatches: int,
    ) -> _Prepared:
        if self._groups is None or self._pool is None:
            raise RuntimeError(
                "Dynamic CP plugin was not initialized by build_model()."
            )
        if hasattr(data, "__next__"):
            iterator = data
        elif hasattr(data, "__iter__"):
            iterator = iter(data)
        else:
            iterator = iter([data])
        samples: list[dict[str, torch.Tensor]] = []
        contexts: list[tuple[LossContext | None, Any]] = []
        input_groups: list[list[int]] = []
        device = None
        for _ in range(num_microbatches):
            batch, context = split_loss_context(next(iterator))
            batch_samples = _batch_samples(batch)
            device = batch.input_ids.device
            start = len(samples)
            samples.extend(batch_samples)
            input_groups.append(list(range(start, len(samples))))
            sources = _split_source(
                None if context is None else context.source_batch, len(batch_samples)
            )
            contexts.extend((context, source) for source in sources)

        try:
            from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Dynamic CP requires MCore DefaultDynamicCPScheduler."
            ) from exc
        scheduler = DefaultDynamicCPScheduler(
            max_seqlen_per_dp_cp_rank=self.max_length,
            cp_size=self._pool.size(),
            dp_size=1,
            microbatch_group_size_per_vp_stage=None,
            min_cp_size=self.minimum,
        )
        plans = scheduler.get_groups_and_subsamples(
            [
                (index, sample["input_ids"].numel())
                for index, sample in enumerate(samples)
            ]
        )
        covered = []
        plan_groups = []
        cp_size_histogram: dict[int, int] = {}
        for assignments in plans:
            if len(assignments) != self._pool.size() or any(
                not ids for ids in assignments
            ):
                raise RuntimeError("Dynamic CP scheduler must assign every pool rank.")
            seen = set()
            for ids in assignments:
                key = tuple(ids)
                if key not in seen:
                    members = [
                        rank
                        for rank, rank_ids in enumerate(assignments)
                        if rank_ids == ids
                    ]
                    if members != list(range(members[0], members[0] + len(members))):
                        raise RuntimeError(
                            "Dynamic CP subgroup members must be contiguous."
                        )
                    sample_ids = [int(index) for index in ids]
                    cp_size = len(members)
                    covered.extend(sample_ids)
                    plan_groups.append(
                        {"cp_size": cp_size, "ranks": members, "sample_ids": sample_ids}
                    )
                    cp_size_histogram[cp_size] = cp_size_histogram.get(
                        cp_size, 0
                    ) + len(sample_ids)
                    seen.add(key)
        if sorted(covered) != list(range(len(samples))):
            raise RuntimeError(
                "Dynamic CP plan must cover every input sample exactly once."
            )
        missing_cp_sizes = sorted(set(self._cp_size_space) - set(cp_size_histogram))
        if loss_fn is not None:
            missing_masks = [
                sample_id
                for sample_id, sample in enumerate(samples)
                if sample["loss_mask"] is None
            ]
            if missing_masks:
                raise ValueError(
                    "Dynamic CP requires loss_mask on every sample for pool-global "
                    f"loss normalization; missing sample ids {missing_masks}."
                )
        global_num_tokens = None
        if all(sample["loss_mask"] is not None for sample in samples):
            owned_ids = [
                sample_id
                for group in plan_groups
                if group["ranks"][0] == self._pool.rank()
                for sample_id in group["sample_ids"]
            ]
            owned_num_tokens = torch.zeros((), dtype=torch.int64, device=device)
            for sample_id in owned_ids:
                owned_num_tokens += samples[sample_id]["loss_mask"].sum(
                    dtype=torch.int64
                )
            torch.distributed.all_reduce(owned_num_tokens, group=self._pool)
            global_num_tokens = int(owned_num_tokens.item())
            if global_num_tokens < 1:
                raise ValueError("Dynamic CP requires a positive global token count.")
        if self._pool.rank() == 0:
            histogram = {
                str(size): cp_size_histogram[size] for size in sorted(cp_size_histogram)
            }
            print(
                f"MLITE_DYNAMIC_CP_PLAN step={self._step} "
                f"cp_size_space={json.dumps(self._cp_size_space, separators=(',', ':'))} "
                f"cp_size_histogram={json.dumps(histogram, separators=(',', ':'))} "
                f"groups={json.dumps(plan_groups, separators=(',', ':'))} "
                f"global_num_tokens={global_num_tokens}",
                flush=True,
            )
        self._step += 1
        if self.require_full_coverage and missing_cp_sizes:
            raise RuntimeError(
                "Dynamic CP plan did not cover required cp_size values "
                f"{missing_cp_sizes}; expected {self._cp_size_space}."
            )
        rank = self._pool.rank()
        scheduled = []
        for assignments in plans:
            ids = [int(index) for index in assignments[rank]]
            if not ids:
                raise RuntimeError(
                    "Dynamic CP scheduler returned an empty rank assignment."
                )
            members = [
                peer
                for peer, peer_ids in enumerate(assignments)
                if peer_ids == assignments[rank]
            ]
            cp_size = len(members)
            if members != list(range(members[0], members[0] + cp_size)):
                raise RuntimeError("Dynamic CP subgroup members must be contiguous.")
            batch = _select_batch(
                samples, ids, cp_size=cp_size, leader=rank == members[0]
            )
            context = _select_context(contexts, ids, batch.input_ids.device)
            correction = None
            if context is not None:
                if global_num_tokens is None:
                    raise RuntimeError(
                        "Dynamic CP loss normalization has no pool-global token count."
                    )
                if context.loss_scale <= 0:
                    raise ValueError(
                        f"Dynamic CP requires positive loss_scale, got {context.loss_scale}."
                    )
                correct_scale = num_microbatches / global_num_tokens
                correction = correct_scale / context.loss_scale
            batch.extras[_GLOBAL_NUM_TOKENS] = global_num_tokens
            batch.extras[_LOSS_SCALE_CORRECTION] = correction
            scheduled.append((batch, context))

        binding = _Binding(
            handle._parallel_state, self._groups, self._context_parallel_modules
        )
        bound = _BoundIterator(iter(scheduled), binding)

        pending_pre_forward_scale: torch.Tensor | None = None
        original_pre_forward = handle._extras.get("_dcp_original_pre_forward")

        def forward(model: Any, batch: PackedBatch):
            nonlocal pending_pre_forward_scale
            binding.bind(int(batch.extras[_LOCAL_CP_SIZE]))
            if original_pre_forward is not None:
                if pending_pre_forward_scale is None:
                    raise RuntimeError(
                        "Dynamic CP forward is missing its pre-forward scale."
                    )
                original_pre_forward(
                    pending_pre_forward_scale * int(binding.local_cp_size or 1)
                )
                pending_pre_forward_scale = None
            return handle._extras["_dcp_original_forward"](model, batch)

        pre_forward = None
        if original_pre_forward is not None:

            def pre_forward(scale: torch.Tensor) -> None:
                nonlocal pending_pre_forward_scale
                pending_pre_forward_scale = scale

        records: list[dict[str, Any]] = []
        collector = getattr(loss_fn, "runtime_output_collector", None)
        collect_outputs = loss_fn is not None and hasattr(
            loss_fn, "runtime_output_collector"
        )
        wrapped_loss = loss_fn
        if loss_fn is not None:
            extractor = getattr(
                loss_fn, "runtime_output_extractor", lambda output: output
            )
            output_loss_scale = float(
                getattr(loss_fn, "runtime_output_loss_scale", 1.0)
            )
            loss_fn.runtime_collects_outputs = collect_outputs
            schedule_scale = len(scheduled) / num_microbatches

            @wraps(loss_fn)
            def wrapped_loss(output, batch, *args, **kwargs):
                loss, metrics = loss_fn(output, batch, *args, **kwargs)
                if collect_outputs and bool(batch.extras[_GROUP_LEADER]):
                    records.append(
                        _record_output(
                            output,
                            batch,
                            loss * output_loss_scale,
                            metrics,
                            extractor,
                        )
                    )
                # DDP still reduces over the physical pool. A sample computed by
                # K CP ranks is represented K times; N/K preserves logical-DP=1
                # global-loss weighting across mixed subgroup sizes.
                replica_scale = self._pool.size() / int(batch.extras[_LOCAL_CP_SIZE])
                normalization = batch.extras[_LOSS_SCALE_CORRECTION]
                if normalization is not None:
                    loss = loss * float(normalization)
                scaled_loss = loss * schedule_scale * replica_scale
                return scaled_loss, metrics

        return _Prepared(
            data=bound,
            count=len(scheduled),
            forward=forward,
            loss=wrapped_loss,
            pre_forward=pre_forward,
            binding=binding,
            consumed=bound,
            records=records,
            collector=collector,
            collect_outputs=collect_outputs,
            pool=self._pool,
            input_groups=input_groups,
            device=device or torch.device("cpu"),
        )

    def wrap_build_model(self, build_model: Callable) -> Callable:
        @wraps(build_model)
        def wrapped(*args, **kwargs):
            return self.initialize(build_model(*args, **kwargs))

        return wrapped

    def wrap_forward_backward(self, forward_backward: Callable) -> Callable:
        @wraps(forward_backward)
        def wrapped(
            handle,
            data,
            loss_fn,
            *,
            num_microbatches=1,
            forward_only=False,
            router_replay=None,
        ):
            original_forward = handle._extras["forward_step"]
            original_hook = handle._extras.get("pre_forward_hook")
            original_protocol = handle._extras.get("protocol")
            handle._extras["_dcp_original_forward"] = original_forward
            handle._extras["_dcp_original_pre_forward"] = original_hook
            prepared = None
            try:
                replay_requested = _router_replay_requested(router_replay)
                if replay_requested:
                    _reject_fused_router_replay(handle._model)
                prepared = self._prepare(handle, data, loss_fn, num_microbatches)
                handle._extras["forward_step"] = prepared.forward
                handle._extras["pre_forward_hook"] = prepared.pre_forward
                if replay_requested:
                    handle._extras["protocol"] = _ReplayProtocolBinding(
                        original_protocol, prepared.binding
                    )
                return forward_backward(
                    handle,
                    prepared.data,
                    prepared.loss,
                    num_microbatches=prepared.count,
                    forward_only=forward_only,
                    router_replay=router_replay,
                )
            finally:
                try:
                    if prepared is not None:
                        prepared.finish(require_complete=sys.exc_info()[0] is None)
                finally:
                    handle._extras.pop("_dcp_original_forward", None)
                    handle._extras.pop("_dcp_original_pre_forward", None)
                    handle._extras["forward_step"] = original_forward
                    if original_protocol is None:
                        handle._extras.pop("protocol", None)
                    else:
                        handle._extras["protocol"] = original_protocol
                    if original_hook is None:
                        handle._extras.pop("pre_forward_hook", None)
                    else:
                        handle._extras["pre_forward_hook"] = original_hook

        return wrapped


def install(runtime: Any, config: Mapping[str, Any]) -> None:
    """Install the sidecar on exactly one runtime object."""
    if not isinstance(config, Mapping):
        raise TypeError("dynamic_context_parallel plugin config must be a mapping.")
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("dynamic_context_parallel.enabled must be a bool.")
    if not enabled:
        return
    plugin = DynamicCPPlugin(config)
    runtime.build_model = plugin.wrap_build_model(runtime.build_model)
    runtime.forward_backward = plugin.wrap_forward_backward(runtime.forward_backward)
    runtime._dynamic_cp_plugin = plugin


__all__ = ["DynamicCPPlugin", "install"]
