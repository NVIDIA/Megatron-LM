# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Portions adapted from NVIDIA-NeMo/RL under Apache-2.0.
# https://github.com/NVIDIA-NeMo/RL/tree/34dd6b3b4f8f108554ec0c9312d83ba7e639da9d
"""Plan and publish tensor sharing between co-resident Megatron workers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace

import torch

from megatron.core.resharding.utils import ParameterMetadata

_FOREACH_COPY_MAX_BYTES = 512 * 1024
_FLOAT_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
_GROUP_FIELDS = (
    "tensor_parallel_group_ranks",
    "expert_parallel_group_ranks",
    "data_parallel_group_ranks",
    "pipeline_parallel_group_ranks",
)


@dataclass(frozen=True)
class RefitSlices:
    """Local learner views that populate one private inference weight."""

    shape: tuple[int, ...]
    parts: tuple[tuple[torch.Tensor, tuple[slice, ...]], ...]


RefitSource = torch.Tensor | RefitSlices


@dataclass(frozen=True)
class LocalSourcePlan:
    """Engine-local names mapped to local views or residual destination metadata."""

    local_sources: dict[str, RefitSource]
    residual: tuple["ParameterMetadata", ...]
    # Only remote tensors needing a cast have a separate transport buffer.
    cast_destination_dtypes: dict[str, torch.dtype] = field(default_factory=dict)
    _cast_receives: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def current_tensors(self, core: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Resolve live engine tensors because inference may repack weights after binding."""
        from megatron.core.resharding.utils import named_refit_tensors

        tensors = dict(named_refit_tensors(core))
        if tensors.keys() != self.local_sources.keys() | {item.name for item in self.residual}:
            raise RuntimeError("co-resident inference tensor names changed after planning")
        return tensors

    def refresh(self, core: torch.nn.Module) -> dict[str, float]:
        """Update private tensors and rebuild the destination module caches."""
        from megatron.core.resharding.execution import refresh_module_caches

        tensors = self.current_tensors(core)
        sources = self.local_sources
        if self.cast_destination_dtypes:
            if self._cast_receives.keys() != self.cast_destination_dtypes.keys():
                raise RuntimeError("Export remote cast buffers before publishing inference weights")
            sources = sources | self._cast_receives
        stats = copy_refit(sources, ((name, tensors[name]) for name in sources))
        with torch.inference_mode():
            refresh_module_caches(core)
        return stats

    def export_destinations(self, core: torch.nn.Module, previous_fingerprint: str | None):
        """Export new receive handles only when destination storage has changed."""
        tensors = self.current_tensors(core)
        targets = {}
        for metadata in self.residual:
            name = metadata.name
            tensor = tensors[name]
            if (
                type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.shape != metadata.shape
                or tensor.dtype != self.cast_destination_dtypes.get(name, metadata.dtype)
            ):
                raise RuntimeError(f"co-resident receive storage changed layout or dtype: {name}")
            if name in self.cast_destination_dtypes:
                # Receive at learner precision, then cast into the live model tensor during refresh.
                if name not in self._cast_receives:
                    self._cast_receives[name] = torch.empty(
                        tensor.shape, dtype=metadata.dtype, device=tensor.device
                    )
                tensor = self._cast_receives[name]
                if tensor.device != tensors[name].device:
                    raise RuntimeError(f"Co-resident receive buffer changed device: {name}")
            targets[name] = tensor
        fingerprint = storage_fingerprint(targets.items())
        handles = None
        if fingerprint != previous_fingerprint:
            handles = export_refit_handles(targets.items())
        return fingerprint, handles


def _validate_metadata(metadata: "ParameterMetadata") -> str:
    resolved = metadata.resolved_name
    if not metadata.name or not resolved:
        raise ValueError("Refit metadata needs explicit local and resolved names")
    if type(metadata.owner_rank) is not int or metadata.owner_rank < 0:
        raise ValueError(f"Invalid owner rank for {metadata.name}")
    if metadata.dtype not in _FLOAT_BYTES or metadata.element_size != _FLOAT_BYTES[metadata.dtype]:
        raise ValueError(f"Unsupported dtype or element size for {metadata.name}")
    if metadata.is_gtp or metadata.gtp_pad_length or metadata.gtp_remat_group_ranks:
        raise ValueError("Co-resident residual refit does not support GTP weights")
    if any(type(extent) is not int or extent <= 0 for extent in metadata.shape):
        raise ValueError(f"Invalid shape for {metadata.name}: {metadata.shape}")
    if type(metadata.partition_stride) is not int or metadata.partition_stride < 1:
        raise ValueError(f"Invalid partition stride for {metadata.name}")
    if metadata.is_tp and not 0 <= metadata.partition_dim < len(metadata.shape):
        raise ValueError(f"Invalid TP partition dimension for {metadata.name}")
    if metadata.partition_sizes is not None:
        if not 0 <= metadata.partition_dim < len(metadata.shape) or any(
            type(size) is not int or size <= 0 for size in metadata.partition_sizes
        ):
            raise ValueError(f"Invalid packed partition sizes for {metadata.name}")
        if sum(metadata.partition_sizes) != metadata.shape[metadata.partition_dim]:
            raise ValueError(f"Packed partition sizes do not cover {metadata.name}")
    for field in _GROUP_FIELDS:
        ranks = getattr(metadata, field)
        if ranks is not None and (
            not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)
            or metadata.owner_rank not in ranks
        ):
            raise ValueError(f"Invalid {field} for {metadata.name}")
    if metadata.is_tp and metadata.tensor_parallel_group_ranks is None:
        raise ValueError(f"Missing TP group for {metadata.name}")
    if metadata.is_ep and (
        metadata.expert_parallel_group_ranks is None or metadata.global_expert_index is None
    ):
        raise ValueError(f"Expert refit needs canonical per-expert metadata: {metadata.name}")

    return resolved


def remap_metadata(
    metadata: list["ParameterMetadata"],
    engine_to_learner: dict[int, int],
    *,
    learner_world_size: int,
) -> list["ParameterMetadata"]:
    """Copy engine metadata into learner-world ranks, retaining group order and names."""
    world = set(range(len(engine_to_learner)))
    if (
        not world
        or type(learner_world_size) is not int
        or any(type(rank) is not int for pair in engine_to_learner.items() for rank in pair)
        or set(engine_to_learner) != world
        or len(set(engine_to_learner.values())) != len(world)
        or any(rank not in range(learner_world_size) for rank in engine_to_learner.values())
    ):
        raise ValueError(
            "Engine-to-learner mapping must cover all engine ranks with distinct learner ranks"
        )
    remapped = deepcopy(metadata)
    for item in remapped:
        _validate_metadata(item)
        try:
            item.owner_rank = engine_to_learner[item.owner_rank]
            for field in _GROUP_FIELDS:
                group = getattr(item, field)
                if group is not None:
                    setattr(item, field, [engine_to_learner[rank] for rank in group])
        except KeyError as exc:
            raise ValueError(f"Metadata rank is outside the placement map: {item.name}") from exc
    return remapped


def plan_local_sources(
    src: dict[str, torch.Tensor],
    all_src_metadata: list["ParameterMetadata"],
    dst_metadata: list["ParameterMetadata"],
    *,
    learner_rank: int,
) -> LocalSourcePlan:
    """Prove complete native coverage, then separate local views from residual copies.

    Source tensors are keyed by resolved name on the paired learner GPU.
    Destination metadata must already be remapped into learner-world ranks.
    A destination needing any remote fragment remains wholly private.
    """
    from megatron.core.resharding.planner import _find_source_metadata
    from megatron.core.resharding.shard_planner import plan_sharded_transfer
    from megatron.core.resharding.utils import select_src_metadata_balanced

    roster: dict[str, list[ParameterMetadata]] = defaultdict(list)
    local: dict[str, ParameterMetadata] = {}
    owner_names: set[tuple[int, str]] = set()
    owner_resolved: set[tuple[int, str]] = set()
    for item in all_src_metadata:
        resolved = _validate_metadata(item)
        name_key, resolved_key = ((item.owner_rank, item.name), (item.owner_rank, resolved))
        if name_key in owner_names or resolved_key in owner_resolved:
            raise ValueError("Duplicate source metadata on one learner rank")
        owner_names.add(name_key)
        owner_resolved.add(resolved_key)
        roster[resolved].append(item)
        if item.owner_rank == learner_rank:
            local[resolved] = item
    if src.keys() != local.keys():
        raise ValueError("Local source tensors and owned metadata must match exactly")
    for name, tensor in src.items():
        item = local[name]
        if (
            type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tuple(tensor.shape) != tuple(item.shape)
            or tensor.dtype != item.dtype
            or tensor.layout != torch.strided
        ):
            raise ValueError(f"Unsupported tensor storage or stale local metadata for {name}")
    mapped: dict[str, RefitSource] = {}
    residual = []
    cast_destination_dtypes = {}
    names, resolved_names = set(), set()
    for destination in dst_metadata:
        resolved = _validate_metadata(destination)
        if destination.owner_rank != learner_rank:
            raise ValueError("Destination metadata must use its paired learner owner rank")
        if destination.name in names or resolved in resolved_names:
            raise ValueError("Duplicate destination metadata")
        names.add(destination.name)
        resolved_names.add(resolved)
        candidates = _find_source_metadata(roster, resolved)
        if not candidates:
            raise ValueError(f"No learner owns {destination.resolved_name}")
        source_dtypes = {item.dtype for item in candidates}
        if len(source_dtypes) != 1:
            raise ValueError(f"Source replicas have inconsistent dtypes for {resolved}")
        source_dtype = next(iter(source_dtypes))
        transport_destination = replace(
            destination, dtype=source_dtype, element_size=_FLOAT_BYTES[source_dtype]
        )
        selected = select_src_metadata_balanced(candidates, transport_destination, learner_rank)
        operations = plan_sharded_transfer(resolved, candidates, selected, transport_destination)
        if any(owner != learner_rank for owner, _, _ in operations):
            if (
                not selected.is_tp
                and not destination.is_tp
                and (
                    selected.data_parallel_group_ranks is None
                    or destination.data_parallel_group_ranks is None
                )
            ):
                raise ValueError(f"Residual replica transfer needs DP groups for {resolved}")
            residual.append(deepcopy(transport_destination))
            if destination.dtype != source_dtype:
                cast_destination_dtypes[destination.name] = destination.dtype
            continue
        source = src[_validate_metadata(selected)]
        parts = tuple(
            (source[source_slice], target_slice) for _, source_slice, target_slice in operations
        )
        if len(parts) == 1 and all(
            part.start == 0 and part.stop == extent
            for part, extent in zip(parts[0][1], destination.shape)
        ):
            mapped[destination.name] = parts[0][0]
        else:
            mapped[destination.name] = RefitSlices(tuple(destination.shape), parts)
    return LocalSourcePlan(mapped, tuple(residual), cast_destination_dtypes)


def export_refit_handles(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, tuple]:
    """Learner side, once: picklable CUDA IPC handles of its local refit tensors, by name."""
    from torch.multiprocessing.reductions import reduce_tensor

    return {name: reduce_tensor(t.detach()) for name, t in named}


def storage_fingerprint(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Identify exported tensor addresses and layouts without reading GPU memory."""
    import hashlib

    h = hashlib.sha1()
    for name, t in named:
        h.update(
            f"{name}:{t.data_ptr()}:{tuple(t.shape)}:{t.stride()}:{t.dtype}:{t.device};".encode()
        )
    return h.hexdigest()


def refit_metadata(core: torch.nn.Module) -> list[ParameterMetadata]:
    """Extract native logical shard coordinates for this rank's refit tensors."""
    from megatron.core.resharding.planner import _extract_module_metadata

    return _extract_module_metadata(
        core,
        torch.distributed.get_rank(),
        getattr(getattr(core, "config", None), "num_moe_experts", None),
        0,
        {},
    )


def open_refit_handles(handles: dict[str, tuple]) -> dict[str, torch.Tensor]:
    """Map the exported tensors into this process and retain their logical names."""
    return {name: rebuild(*args) for name, (rebuild, args) in handles.items()}


def bind_shared(
    src: dict[str, RefitSource], named_dst: Iterable[tuple[str, torch.Tensor]], leaf_ids: set[int]
) -> dict[str, float]:
    """Bind compatible inference parameters to learner CUDA IPC storage.

    Only matching contiguous parameters can alias learner storage. Persistent buffers may
    change during backward. A later engine repack is handled when the next publication
    resolves its live destination tensors.
    """
    dst = dict(named_dst)
    _check_names(src, dst)
    bound = 0
    for name, d in dst.items():
        s = src[name]
        if s.shape != d.shape:
            raise ValueError(
                f"co-resident refit: {name} has shape {tuple(s.shape)} (learner) "
                f"vs {tuple(d.shape)} (engine)"
            )
        # TE GEMMs require dense weights and can misread strided views.
        # Persistent buffers can change during backward, before the optimizer pause.
        if (
            isinstance(s, torch.Tensor)
            and isinstance(d, torch.nn.Parameter)
            and s.dtype == d.dtype
            and s.is_contiguous()
            and id(d) in leaf_ids
        ):
            d.data = s
            bound += 1
    return {"tensors_bound": bound, "tensors": len(dst)}


def _check_names(src: dict[str, RefitSource], dst: dict[str, torch.Tensor]) -> None:
    if dst.keys() != src.keys():
        missing, extra = sorted(set(dst) - set(src)), sorted(set(src) - set(dst))
        raise ValueError(
            "co-resident refit: engine tensors differ by name from the planned local sources: "
            f"engine-only {missing[:8]} ({len(missing)}), learner-only {extra[:8]} ({len(extra)})"
        )


def _aliases(d: torch.Tensor, s: torch.Tensor) -> bool:
    return (
        d.data_ptr() == s.data_ptr()
        and d.shape == s.shape
        and d.stride() == s.stride()
        and d.dtype == s.dtype
    )


def copy_refit(
    src: dict[str, RefitSource], named_dst: Iterable[tuple[str, torch.Tensor]]
) -> dict[str, float]:
    """Copy private inference tensors and account for shared storage.

    Call with the engine paused and learner writes complete. Resolve destinations at each
    publication so inference can move or repack its storage between calls.
    """
    dst = dict(named_dst)
    _check_names(src, dst)
    copied = shared = 0
    copy_groups = defaultdict(lambda: ([], []))
    # Lazily allocated inference tensors require writes under inference mode.
    with torch.inference_mode():
        for name, d in dst.items():
            s = src[name]
            if s.shape != d.shape:
                raise ValueError(
                    f"co-resident refit: {name} has shape {tuple(s.shape)} (learner) "
                    f"vs {tuple(d.shape)} (engine)"
                )
            if isinstance(s, RefitSlices):
                for source_view, target_slice in s.parts:
                    d[target_slice].copy_(source_view)
                    copied += source_view.numel() * source_view.element_size()
                continue
            nbytes = s.numel() * s.element_size()
            if _aliases(d, s):
                shared += nbytes
                continue
            if nbytes <= _FOREACH_COPY_MAX_BYTES and s.is_contiguous() and d.is_contiguous():
                # Mixed dtype lists force foreach back to one copy per tensor.
                key = (s.device, d.device, s.dtype, d.dtype)
                sources, destinations = copy_groups[key]
                sources.append(s)
                destinations.append(d)
            else:
                # Enqueue large copies while the host prepares subsequent tensors.
                d.copy_(s)
            copied += nbytes
        for copy_src, copy_dst in copy_groups.values():
            torch._foreach_copy_(copy_dst, copy_src)
    torch.cuda.current_stream().synchronize()
    return {"bytes": float(copied), "bytes_shared": float(shared)}


class ResidualRefit:
    """Transfer private destination shards through the colocated learner process group.

    Metadata uses ranks in the supplied group. The inference process owns the
    destination storage and keeps it alive until outstanding copies finish.
    """

    def __init__(self, source_metadata, residual_metadata, group, *, execution_batch_bytes=None):
        from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
        from megatron.core.resharding.planner import build_plan_from_rosters, index_metadata_rosters

        self.group = group
        self.receive_tensors = None
        self.fingerprint = None
        world = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        rosters = [([], []) for _ in range(world)]
        for side, records in enumerate((source_metadata, residual_metadata)):
            for record in records:
                metadata = ParameterMetadata(**record)
                if not 0 <= metadata.owner_rank < world:
                    raise ValueError("co-resident metadata owner is outside the learner world")
                rosters[metadata.owner_rank][side].append(metadata)
        destinations, sources = index_metadata_rosters(rosters)
        self.receive_metadata = {item.name: item for item in rosters[rank][1]}
        self.plan = build_plan_from_rosters(
            destinations, sources, rank, execution_batch_bytes=execution_batch_bytes
        )
        self.service = NCCLCopyService(group=group)

    @torch.inference_mode()
    def execute(self, model: torch.nn.Module, *, fingerprint: str | None, handles: dict | None):
        """Participate on every source rank, including ranks without an inference process."""
        from megatron.core.resharding.execution import _execute_reshard_plan_with_tensors
        from megatron.core.resharding.utils import get_refit_tensor_dict

        receive_tensors = {}
        if fingerprint is None:
            if self.receive_metadata != {}:
                raise ValueError("Missing inference buffers for planned residual receives")
        elif handles is not None:
            tensors = open_refit_handles(handles)
            receive_metadata = self.receive_metadata
            if tensors.keys() != receive_metadata.keys():
                raise ValueError(
                    "co-resident receive handles do not cover the planned destinations"
                )
            for name, tensor in tensors.items():
                metadata = receive_metadata[name]
                if tuple(tensor.shape) != tuple(metadata.shape) or tensor.dtype != metadata.dtype:
                    raise ValueError(
                        f"co-resident receive handle has a different layout or dtype: {name}"
                    )
            if any(
                tensor.device != torch.device("cuda", torch.cuda.current_device())
                for tensor in tensors.values()
            ):
                raise ValueError(
                    "co-resident receive buffers must reside on the paired learner GPU"
                )
            self.receive_tensors = tensors
            self.fingerprint = fingerprint
        if fingerprint is not None:
            if self.receive_tensors is None or fingerprint != self.fingerprint:
                raise RuntimeError(
                    "co-resident receive storage changed without replacement IPC handles"
                )
            receive_tensors = self.receive_tensors
        _execute_reshard_plan_with_tensors(
            self.plan, get_refit_tensor_dict(model), receive_tensors, self.service, group=self.group
        )


def bind_inference_weights(
    core: torch.nn.Module,
    local_handles: dict,
    source_metadata: list[ParameterMetadata],
    engine_to_learner: dict[int, int],
    learner_world_size: int,
    learner_rank: int,
    *,
    share: bool,
) -> tuple[LocalSourcePlan, dict[str, float]]:
    """Bind the local tensors before starting inference and identify private destinations."""
    from megatron.core.resharding.utils import named_refit_tensors

    plan = plan_local_sources(
        open_refit_handles(local_handles),
        source_metadata,
        remap_metadata(
            refit_metadata(core), engine_to_learner, learner_world_size=learner_world_size
        ),
        learner_rank=learner_rank,
    )
    local_tensors = [
        (name, tensor) for name, tensor in named_refit_tensors(core) if name in plan.local_sources
    ]
    stats: dict[str, float] = {"tensors_bound": 0, "tensors": len(local_tensors)}
    if share:
        stats = bind_shared(plan.local_sources, local_tensors, {id(p) for p in core.parameters()})
    stats["residual_tensors"] = len(plan.residual)
    stats["tensors"] += len(plan.residual)
    return plan, stats
