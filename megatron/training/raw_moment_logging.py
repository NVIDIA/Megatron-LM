# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Activation, dgrad, residual-stream, and residual-dgrad raw-moment logging."""

from __future__ import annotations

import re
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.per_parameter_stats import (
    RAW_MOMENT_FIELDS,
    raw_moment_row,
    raw_moment_row_to_dict,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.layer_boundary_observer import (
    LayerBoundary,
    observe_transformer_layer_boundaries,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.utils import unwrap_model

from .activation_logging import LINEAR_TYPES

_LAYER_NAME_PATTERN = re.compile(r"layers\.(\d+)")
_COLUMN_PARALLEL_CLASS_NAMES = {
    "TEColumnParallelLinear",
    "TELayerNormColumnParallelLinear",
    "TEColumnParallelGroupedLinear",
}
_ROW_PARALLEL_CLASS_NAMES = {"TERowParallelLinear", "TERowParallelGroupedLinear"}
_SEQUENCE_PARALLEL_CLASS_NAMES = {"TENorm"}


@dataclass(frozen=True)
class _SitePolicy:
    reduce_groups: tuple[torch.distributed.ProcessGroup, ...]
    owner_groups: tuple[torch.distributed.ProcessGroup, ...]


@dataclass
class _RawMomentSite:
    name: str
    policy: _SitePolicy
    moments: torch.Tensor | None = None


@dataclass
class _RawMomentCollection:
    sites: OrderedDict[str | tuple[int, int, LayerBoundary], _RawMomentSite] = field(
        default_factory=OrderedDict
    )
    latest: list[tuple[str, dict[str, float]]] | None = None


class RawMomentLogger:
    """Collect activation, dgrad, residual-stream, and residual-dgrad raw moments."""

    def __init__(self):
        self._activation_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._dgrad_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._residual_dgrad_hooks: list[
            tuple[torch.utils.hooks.RemovableHandle, torch.autograd.graph.GradientEdge]
        ] = []
        self._activations = _RawMomentCollection()
        self._dgrads = _RawMomentCollection()
        self._residuals = _RawMomentCollection()
        self._residual_dgrads = _RawMomentCollection()

    def register_activation_hooks(self, model: Iterable[nn.Module] | nn.Module) -> None:
        """Register forward hooks for activation raw moments."""
        assert not self._activation_hooks
        self._activations.sites.clear()
        for module, input_site, output_site in _iter_hook_modules(model):
            input_site = self._activations.sites.setdefault(input_site.name, input_site)
            if output_site is not None:
                output_site = self._activations.sites.setdefault(output_site.name, output_site)

            def hook(_, args, __kwargs, output, input_site=input_site, output_site=output_site):
                if not torch.is_grad_enabled():
                    return
                self._add_tensor(input_site, _first_item(args))
                if output_site is not None:
                    self._add_tensor(output_site, _first_item(output))

            self._activation_hooks.append(module.register_forward_hook(hook, with_kwargs=True))

    def register_dgrad_hooks(self, model: Iterable[nn.Module] | nn.Module) -> None:
        """Register backward hooks for dgrad raw moments."""
        assert not self._dgrad_hooks
        self._dgrads.sites.clear()
        for module, input_site, output_site in _iter_hook_modules(model):
            input_site = self._dgrads.sites.setdefault(input_site.name, input_site)
            if output_site is not None:
                output_site = self._dgrads.sites.setdefault(output_site.name, output_site)

            def hook(_, grad_input, grad_output, input_site=input_site, output_site=output_site):
                if output_site is not None:
                    self._add_tensor(output_site, _first_item(grad_output))
                self._add_tensor(input_site, _first_item(grad_input))

            self._dgrad_hooks.append(module.register_full_backward_hook(hook))

    def prepare_residual_logging(
        self,
        model: Iterable[nn.Module] | nn.Module,
        capture_residuals: bool = True,
        capture_dgrads: bool = False,
    ) -> None:
        """Create stable residual and residual-dgrad sites for the decoder layers in ``model``."""
        assert not self._residual_dgrad_hooks
        self._residuals.sites.clear()
        self._residual_dgrads.sites.clear()
        residual_sites_by_name: dict[str, _RawMomentSite] = {}
        residual_dgrad_sites_by_name: dict[str, _RawMomentSite] = {}
        for key, name, policy in _iter_residual_sites(model):
            if capture_residuals:
                self._residuals.sites[key] = residual_sites_by_name.setdefault(
                    name, _RawMomentSite(name, policy)
                )
            if capture_dgrads:
                self._residual_dgrads.sites[key] = residual_dgrad_sites_by_name.setdefault(
                    name, _RawMomentSite(name, policy)
                )

    def record_residual_boundary(
        self, stack: nn.Module, layer: nn.Module, boundary: LayerBoundary, tensor: torch.Tensor
    ) -> None:
        """Accumulate one observed residual-stream boundary."""
        # Activation checkpointing reaches this path again under grad during recomputation.
        if not torch.is_grad_enabled():
            return
        key = (id(stack), id(layer), boundary)
        site = self._residuals.sites.get(key)
        if site is not None:
            self._add_tensor(site, tensor)
        dgrad_site = self._residual_dgrads.sites.get(key)
        if dgrad_site is not None and tensor.requires_grad:
            # CUDA graph replay may reuse this Tensor object with a new grad_fn.
            edge = torch.autograd.graph.get_gradient_edge(tensor)

            def record_dgrad(
                grad_outputs: tuple[torch.Tensor | None, ...],
                output_nr: int = edge.output_nr,
                site: _RawMomentSite = dgrad_site,
            ) -> None:
                self._add_tensor(site, grad_outputs[output_nr])

            handle = edge.node.register_prehook(record_dgrad)
            # Retain the edge's ownership token until backward and remove the hook afterward.
            self._residual_dgrad_hooks.append((handle, edge))

    def finalize_activation_raw_moments_by_layer(self) -> None:
        """Reduce and cache activation raw moments for later logging."""
        self._finalize_collection(self._activations)

    def finalize_dgrad_raw_moments_by_layer(self) -> None:
        """Reduce and cache dgrad raw moments for later logging."""
        self._finalize_collection(self._dgrads)

    def finalize_residual_raw_moments_by_layer(self) -> None:
        """Reduce and cache residual-stream raw moments for later logging."""
        self._finalize_collection(self._residuals)

    def finalize_residual_dgrad_raw_moments_by_layer(self) -> None:
        """Reduce and cache residual-stream dgrad raw moments for later logging."""
        self._finalize_collection(self._residual_dgrads)
        for hook, _ in self._residual_dgrad_hooks:
            hook.remove()
        self._residual_dgrad_hooks.clear()

    def consume_activation_raw_moments_by_layer(self) -> list[tuple[str, dict[str, float]]] | None:
        """Return and clear the latest activation raw moments."""
        return self._consume_collection(self._activations)

    def consume_dgrad_raw_moments_by_layer(self) -> list[tuple[str, dict[str, float]]] | None:
        """Return and clear the latest dgrad raw moments."""
        return self._consume_collection(self._dgrads)

    def consume_residual_raw_moments_by_layer(self) -> list[tuple[str, dict[str, float]]] | None:
        """Return and clear the latest residual-stream raw moments."""
        return self._consume_collection(self._residuals)

    def consume_residual_dgrad_raw_moments_by_layer(
        self,
    ) -> list[tuple[str, dict[str, float]]] | None:
        """Return and clear the latest residual-stream dgrad raw moments."""
        return self._consume_collection(self._residual_dgrads)

    def remove_activation_hooks(self) -> None:
        """Remove activation raw-moment hooks."""
        for hook in self._activation_hooks:
            hook.remove()
        self._activation_hooks.clear()

    def remove_dgrad_hooks(self) -> None:
        """Remove dgrad raw-moment hooks."""
        for hook in self._dgrad_hooks:
            hook.remove()
        self._dgrad_hooks.clear()

    def _finalize_collection(self, collection: _RawMomentCollection) -> None:
        unique_sites = OrderedDict((site.name, site) for site in collection.sites.values())
        collection.latest = self._finalize_sites(unique_sites.values())
        collection.sites.clear()

    @staticmethod
    def _consume_collection(
        collection: _RawMomentCollection,
    ) -> list[tuple[str, dict[str, float]]] | None:
        values = collection.latest
        collection.latest = None
        return values

    @torch.no_grad()
    def _add_tensor(self, site: _RawMomentSite, tensor: torch.Tensor | None) -> None:
        if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return
        if tensor.numel() == 0:
            return

        row = raw_moment_row(tensor)
        if site.moments is None:
            site.moments = row
        else:
            site.moments.add_(row.to(device=site.moments.device))

    def _finalize_sites(
        self, sites: Iterable[_RawMomentSite]
    ) -> list[tuple[str, dict[str, float]]] | None:
        sites = list(sites)
        if not sites:
            return None

        device = _select_device(sites)
        reduced_rows: dict[str, torch.Tensor] = {}
        sites_by_reduce_key = OrderedDict()
        for site in sites:
            key = tuple(id(group) for group in site.policy.reduce_groups)
            if key not in sites_by_reduce_key:
                sites_by_reduce_key[key] = (site.policy.reduce_groups, [])
            sites_by_reduce_key[key][1].append(site)

        for reduce_groups, group_sites in sites_by_reduce_key.values():
            rows = [
                site.moments.to(device=device)
                if site.moments is not None
                else torch.zeros(len(RAW_MOMENT_FIELDS), dtype=torch.float32, device=device)
                for site in group_sites
            ]
            moments = torch.stack(rows)
            if _distributed_is_initialized():
                for group in reduce_groups:
                    torch.distributed.all_reduce(
                        moments, op=torch.distributed.ReduceOp.SUM, group=group
                    )
            for index, site in enumerate(group_sites):
                reduced_rows[site.name] = moments[index]

        writer_sites = [site for site in sites if _is_writer(site.policy)]
        if not writer_sites:
            return []

        rows = torch.stack([reduced_rows[site.name] for site in writer_sites]).detach().cpu().tolist()
        values = []
        for site, row in zip(writer_sites, rows):
            if row[0] == 0:
                continue
            values.append((site.name, raw_moment_row_to_dict(row)))
        return values


def _iter_hook_modules(
    model: Iterable[nn.Module] | nn.Module,
) -> Iterable[tuple[nn.Module, _RawMomentSite, _RawMomentSite | None]]:
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        unwrapped = unwrap_model(model_chunk)
        for module_name, module in unwrapped.named_modules():
            if not isinstance(module, LINEAR_TYPES):
                continue
            canonical_module_name = _canonical_module_name(unwrapped, module_name, module)
            input_site_name = f"{canonical_module_name}/input0"
            output_site_name = f"{canonical_module_name}/output0"
            output_site = None
            if not _is_output_layer_logits_site(canonical_module_name):
                output_site = _RawMomentSite(
                    output_site_name, _site_policy(module_name, module, "output0")
                )
            yield (
                module,
                _RawMomentSite(input_site_name, _site_policy(module_name, module, "input0")),
                output_site,
            )


def _iter_residual_sites(
    model: Iterable[nn.Module] | nn.Module,
) -> Iterable[tuple[tuple[int, int, LayerBoundary], str, _SitePolicy]]:
    for stack_name, stack in _iter_residual_stacks(model):
        layer_prefix = f"{stack_name}." if stack_name else ""
        for layer in getattr(stack, "layers"):
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                continue
            policy = _residual_site_policy(layer)
            if layer_number == 1:
                input_name = f"{stack_name}/input0" if stack_name else "input0"
                yield (id(stack), id(layer), "input"), input_name, policy
            output_name = f"{layer_prefix}layers.{layer_number - 1}/output0"
            yield (id(stack), id(layer), "output"), output_name, policy


def _iter_residual_stacks(
    model: Iterable[nn.Module] | nn.Module,
) -> Iterable[tuple[str, nn.Module]]:
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.transformer_block import TransformerBlock

    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        unwrapped = unwrap_model(model_chunk)
        for module_name, module in unwrapped.named_modules():
            if not isinstance(module, (TransformerBlock, HybridStack)):
                continue
            if getattr(module, "is_mtp_layer", False) or "mtp" in module_name.split("."):
                continue
            yield module_name, module


def _is_output_layer_logits_site(module_name: str) -> bool:
    return module_name.rsplit(".", maxsplit=1)[-1] == "output_layer"


def _canonical_module_name(model_chunk: nn.Module, module_name: str, module: nn.Module) -> str:
    if "mtp" in module_name or _LAYER_NAME_PATTERN.search(module_name) is None:
        return module_name

    from megatron.core.transformer.transformer_layer import TransformerLayer

    for transformer_layer in model_chunk.modules():
        if not isinstance(transformer_layer, TransformerLayer):
            continue
        for child in transformer_layer.modules():
            if child is module:
                return _LAYER_NAME_PATTERN.sub(
                    f"layers.{transformer_layer.layer_number - 1}", module_name
                )
    return module_name


def _site_policy(module_name: str, module: nn.Module, field: str) -> _SitePolicy:
    reduce_groups = []
    owner_groups = []
    is_expert_site = _is_expert_site(module_name, module)

    if is_expert_site:
        expert_data_parallel_group = _expert_data_parallel_group()
        if expert_data_parallel_group is not None:
            reduce_groups.append(expert_data_parallel_group)
            owner_groups.append(expert_data_parallel_group)

        context_parallel_group = _context_parallel_group()
        if context_parallel_group is not None:
            reduce_groups.append(context_parallel_group)
            owner_groups.append(context_parallel_group)
    else:
        dp_cp_group = _data_parallel_with_context_group()
        if dp_cp_group is not None:
            reduce_groups.append(dp_cp_group)
            owner_groups.append(dp_cp_group)

    tp_group = _module_tensor_parallel_group(module)
    if tp_group is not None:
        owner_groups.append(tp_group)
        if _field_is_tensor_parallel_shard(module, field):
            reduce_groups.append(tp_group)

    if is_expert_site:
        expert_group = _expert_model_parallel_group()
        if expert_group is not None:
            reduce_groups.append(expert_group)
            owner_groups.append(expert_group)

    return _SitePolicy(tuple(reduce_groups), tuple(owner_groups))


def _residual_site_policy(layer: nn.Module) -> _SitePolicy:
    reduce_groups = []
    owner_groups = []

    dp_cp_group = _data_parallel_with_context_group()
    if dp_cp_group is not None:
        reduce_groups.append(dp_cp_group)
        owner_groups.append(dp_cp_group)

    tp_group = _module_tensor_parallel_group(layer)
    if tp_group is not None:
        owner_groups.append(tp_group)
        if _sequence_parallel_enabled(layer):
            reduce_groups.append(tp_group)

    return _SitePolicy(tuple(reduce_groups), tuple(owner_groups))


def _field_is_tensor_parallel_shard(module: nn.Module, field: str) -> bool:
    class_name = module.__class__.__name__
    parallel_mode = getattr(module, "parallel_mode", None)
    sequence_parallel = _sequence_parallel_enabled(module)

    if isinstance(module, ColumnParallelLinear) or class_name in _COLUMN_PARALLEL_CLASS_NAMES:
        if field == "output0":
            return not bool(getattr(module, "gather_output", False))
        return sequence_parallel

    if isinstance(module, RowParallelLinear) or class_name in _ROW_PARALLEL_CLASS_NAMES:
        if field == "input0":
            return bool(getattr(module, "input_is_parallel", False)) or sequence_parallel
        return sequence_parallel or bool(getattr(module, "explicit_expert_comm", False))

    if parallel_mode == "column":
        return field == "output0" or sequence_parallel
    if parallel_mode == "row":
        return field == "input0" or sequence_parallel

    if isinstance(module, Router) or class_name in _SEQUENCE_PARALLEL_CLASS_NAMES:
        return sequence_parallel

    return False


def _sequence_parallel_enabled(module: nn.Module) -> bool:
    config = getattr(module, "config", None)
    return bool(getattr(module, "sequence_parallel", False) or getattr(config, "sequence_parallel", False))


def _is_expert_site(module_name: str, module: nn.Module) -> bool:
    return bool(
        getattr(module, "explicit_expert_comm", False)
        or getattr(module, "is_expert", False)
        or ".experts." in module_name
    )


def _select_device(sites: list[_RawMomentSite]) -> torch.device:
    for site in sites:
        if site.moments is not None:
            return site.moments.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _first_item(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _model_parallel_is_initialized() -> bool:
    return _distributed_is_initialized() and parallel_state.is_initialized()


def _active_group(group: torch.distributed.ProcessGroup | None):
    if group is None:
        return None
    try:
        return group if group.size() > 1 else None
    except RuntimeError:
        return None


def _data_parallel_with_context_group():
    if not _model_parallel_is_initialized():
        return None
    return _active_group(parallel_state.get_data_parallel_group(with_context_parallel=True))


def _expert_data_parallel_group():
    if not _model_parallel_is_initialized():
        return None
    return _active_group(parallel_state.get_expert_data_parallel_group())


def _context_parallel_group():
    if not _model_parallel_is_initialized():
        return None
    return _active_group(parallel_state.get_context_parallel_group())


def _module_tensor_parallel_group(module: nn.Module):
    if not _model_parallel_is_initialized():
        return None
    group = getattr(module, "tp_group", None) or getattr(module, "_tp_group", None)
    if group is None:
        group = parallel_state.get_tensor_model_parallel_group()
    return _active_group(group)


def _expert_model_parallel_group():
    if not _model_parallel_is_initialized():
        return None
    return _active_group(parallel_state.get_expert_model_parallel_group())


def _is_writer(policy: _SitePolicy) -> bool:
    if not _distributed_is_initialized():
        return True
    for group in policy.owner_groups:
        if torch.distributed.get_rank(group=group) != 0:
            return False
    return True


_LOGGER: RawMomentLogger | None = None


def _get_logger() -> RawMomentLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = RawMomentLogger()
    return _LOGGER


def _require_logger() -> RawMomentLogger:
    assert _LOGGER is not None, "No RawMomentLogger has been initialised"
    return _LOGGER


def enable_activation_raw_moment_logging(model: Iterable[nn.Module] | nn.Module) -> None:
    """Enable activation raw-moment logging on ``model``."""
    _get_logger().register_activation_hooks(model)


def finalize_activation_raw_moments_by_layer() -> None:
    """Reduce and cache activation raw moments."""
    _require_logger().finalize_activation_raw_moments_by_layer()


def consume_activation_raw_moments_by_layer() -> list[tuple[str, dict[str, float]]] | None:
    """Return and clear the latest activation raw moments."""
    return _require_logger().consume_activation_raw_moments_by_layer()


def disable_activation_raw_moment_logging() -> None:
    """Disable activation raw-moment logging."""
    _require_logger().remove_activation_hooks()


def enable_dgrad_raw_moment_logging(model: Iterable[nn.Module] | nn.Module) -> None:
    """Enable dgrad raw-moment logging on ``model``."""
    _get_logger().register_dgrad_hooks(model)


def finalize_dgrad_raw_moments_by_layer() -> None:
    """Reduce and cache dgrad raw moments."""
    _require_logger().finalize_dgrad_raw_moments_by_layer()


def consume_dgrad_raw_moments_by_layer() -> list[tuple[str, dict[str, float]]] | None:
    """Return and clear the latest dgrad raw moments."""
    return _require_logger().consume_dgrad_raw_moments_by_layer()


def disable_dgrad_raw_moment_logging() -> None:
    """Disable dgrad raw-moment logging."""
    _require_logger().remove_dgrad_hooks()


@contextmanager
def capture_residual_raw_moments(
    model: Iterable[nn.Module] | nn.Module,
    capture_residuals: bool = True,
    capture_dgrads: bool = False,
) -> Iterator[None]:
    """Capture decoder residual-stream and residual-dgrad raw moments within this context."""
    logger = _get_logger()
    logger.prepare_residual_logging(
        model, capture_residuals=capture_residuals, capture_dgrads=capture_dgrads
    )
    with observe_transformer_layer_boundaries(logger.record_residual_boundary):
        yield


def finalize_residual_raw_moments_by_layer() -> None:
    """Reduce and cache residual-stream raw moments."""
    _require_logger().finalize_residual_raw_moments_by_layer()


def consume_residual_raw_moments_by_layer() -> list[tuple[str, dict[str, float]]] | None:
    """Return and clear the latest residual-stream raw moments."""
    return _require_logger().consume_residual_raw_moments_by_layer()


def finalize_residual_dgrad_raw_moments_by_layer() -> None:
    """Reduce and cache residual-stream dgrad raw moments."""
    _require_logger().finalize_residual_dgrad_raw_moments_by_layer()


def consume_residual_dgrad_raw_moments_by_layer() -> list[tuple[str, dict[str, float]]] | None:
    """Return and clear the latest residual-stream dgrad raw moments."""
    return _require_logger().consume_residual_dgrad_raw_moments_by_layer()
