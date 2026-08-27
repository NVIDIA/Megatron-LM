# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import groupby
from math import prod
from typing import Any, Mapping

import torch
import torch.distributed as dist

try:
    import nccl
    import nccl.core as nccl_core
    import nccl.m2n as m2n

    HAVE_NCCL_M2N = True
except ImportError:
    HAVE_NCCL_M2N = False

from ..transforms import MXFP8ReshardTransform, _ensure_sendable
from ..utils import ReshardPlan, TensorReshardSpec
from .base import CopyService

logger = logging.getLogger(__name__)

# nccl-extensions' public M2N header requires NCCL 2.30.5 or newer.
# _validate_nccl_version checks the loaded libnccl before M2N is initialized.
_MINIMUM_NCCL_VERSION = (2, 30, 5)

# M2N's grouped PACK path amortizes fixed per-call work across tensors. Bound
# each submission so receiver-side MXFP8 conversion does not stage an entire
# model in BF16 at once. A single parameter may exceed this soft limit.
_DEFAULT_MAX_GROUP_BYTES = 256 * 1024 * 1024
_MAX_GROUP_BYTES_ENV = "MEGATRON_NCCL_M2N_MAX_GROUP_BYTES"
_MAX_INT64 = (1 << 63) - 1

_StagePair = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _M2NTopology:
    """Contiguous source/destination rank intervals required by NCCL M2N."""

    src_ranks: tuple[int, ...]
    dst_ranks: tuple[int, ...]


@dataclass
class _M2NChannel:
    """NCCL4Py communicator dedicated to one PP-stage pair."""

    comm: Any
    stream: torch.cuda.Stream
    src_mesh: Any
    dst_mesh: Any


@dataclass
class _LocalTransfer:
    """Local buffers for one M2N call and an optional destination writeback."""

    src: torch.Tensor | None
    dst: torch.Tensor | None
    dst_view: torch.Tensor | None = None


@dataclass
class _PendingParameter:
    """Buffers retained until a grouped M2N submission is enqueued."""

    local_transfers: list[_LocalTransfer]
    transform_args: tuple[str, tuple[slice, ...], list[torch.Tensor]] | None = None


def _validate_role_roster(roles: list[tuple[bool, bool]]) -> _M2NTopology:
    """Validate NCCL M2N's disjoint, source-first rank topology."""
    overlapping = [rank for rank, (is_src, is_dst) in enumerate(roles) if is_src and is_dst]
    idle = [rank for rank, (is_src, is_dst) in enumerate(roles) if not is_src and not is_dst]
    if overlapping:
        raise RuntimeError(
            "NCCL M2N refit requires non-collocated source and destination ranks; "
            f"ranks {overlapping} participate on both sides"
        )
    if idle:
        raise RuntimeError(
            "NCCL M2N refit requires the process group to contain exactly the source and "
            f"destination meshes; idle ranks are not supported (idle ranks: {idle})"
        )

    src_ranks = tuple(rank for rank, (is_src, _is_dst) in enumerate(roles) if is_src)
    dst_ranks = tuple(rank for rank, (_is_src, is_dst) in enumerate(roles) if is_dst)
    if not src_ranks or not dst_ranks:
        raise RuntimeError("NCCL M2N refit requires at least one source and one destination rank")

    expected_src = tuple(range(len(src_ranks)))
    expected_dst = tuple(range(len(src_ranks), len(roles)))
    if src_ranks != expected_src or dst_ranks != expected_dst:
        raise RuntimeError(
            "NCCL M2N refit requires one source-first contiguous rank interval followed by "
            f"one destination interval; got source ranks {src_ranks} and destination ranks "
            f"{dst_ranks}"
        )
    return _M2NTopology(src_ranks=src_ranks, dst_ranks=dst_ranks)


def _validate_nccl_version(nccl_module: Any) -> None:
    """Ensure the loaded NCCL library supports the current M2N API."""
    try:
        version = nccl_module.get_version().nccl.version
        release = tuple(version.release)
    except AttributeError as exc:
        raise RuntimeError("NCCL M2N requires the current NCCL4Py package") from exc

    if release < _MINIMUM_NCCL_VERSION:
        required = ".".join(str(value) for value in _MINIMUM_NCCL_VERSION)
        raise RuntimeError(f"NCCL M2N requires NCCL >= {required}, found {version}")


def _has_nccl_cuda_backend(group: Any) -> bool:
    """Return whether CUDA collectives on *group* dispatch through NCCL."""
    if dist.get_backend(group) == dist.Backend.NCCL:
        return True
    if group is None:
        return False
    try:
        cuda_backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    except RuntimeError:
        return False
    return cuda_backend._get_backend_name() == dist.Backend.NCCL


def _stage_pairs(specs: list[TensorReshardSpec]) -> tuple[_StagePair, ...]:
    """Return the deterministic communicator roster used by every rank."""
    return tuple(sorted({(spec.src_ranks, spec.dst_ranks) for spec in specs}))


def _parameter_spec_key(spec: TensorReshardSpec) -> str:
    return spec.resolved_name


def _max_group_bytes_from_env() -> int:
    value = os.environ.get(_MAX_GROUP_BYTES_ENV)
    if value is None:
        return _DEFAULT_MAX_GROUP_BYTES
    try:
        max_group_bytes = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{_MAX_GROUP_BYTES_ENV} must be a positive integer") from exc
    if max_group_bytes <= 0:
        raise RuntimeError(f"{_MAX_GROUP_BYTES_ENV} must be a positive integer")
    return max_group_bytes


def _parameter_groups(
    specs: list[TensorReshardSpec], max_group_bytes: int
) -> list[list[list[TensorReshardSpec]]]:
    """Batch complete parameters without exceeding the soft byte limit."""
    parameters: list[list[TensorReshardSpec]] = []
    seen: set[str] = set()
    for key, grouped_specs in groupby(specs, key=_parameter_spec_key):
        if key in seen:
            raise RuntimeError(f"NCCL M2N plan contains non-contiguous specs for {key}")
        seen.add(key)
        parameters.append(list(grouped_specs))

    batches: list[list[list[TensorReshardSpec]]] = []
    batch: list[list[TensorReshardSpec]] = []
    batch_bytes = 0
    for parameter_specs in parameters:
        parameter_bytes = sum(
            prod(spec.dst_local_shape) * spec.dtype.itemsize for spec in parameter_specs
        )
        if batch and batch_bytes + parameter_bytes > max_group_bytes:
            batches.append(batch)
            batch = []
            batch_bytes = 0
        batch.append(parameter_specs)
        batch_bytes += parameter_bytes
    if batch:
        batches.append(batch)
    return batches


class NCCLM2NCopyService(CopyService):
    """Non-collocated ReFIT transport backed by native NCCL M2N resharding.

    M2N consumes TP-local parameter shards together with their source and
    destination mesh placements. Each pipeline-stage pair uses an exact,
    source-first communicator so native calls contain no inactive ranks. Stage
    pairs run on independent streams, matching the native M2N execution model.

    Grouped submissions amortize M2N's fixed per-call work. Receiver-side
    MXFP8 conversion keeps only a bounded batch of full-parameter BF16 buffers
    on each destination rank, then quantizes that batch on the same stream.

    Packed Mamba projections are split into their independently-sharded
    components; they are never lowered into a padded point-to-point tensor.

    Args:
        group: NCCL process group containing exactly the source and destination ranks.
        max_group_bytes: Soft limit for the destination bytes in one grouped
            submission. Defaults to ``MEGATRON_NCCL_M2N_MAX_GROUP_BYTES`` or
            256 MiB. A single parameter may exceed the limit.
    """

    requires_process_group_barrier = False
    supports_idle_ranks = False

    def __init__(self, group=None, *, max_group_bytes: int | None = None):
        if not HAVE_NCCL_M2N:
            raise ImportError("NCCL M2N refit requires NVIDIA/nccl-extensions and NCCL4Py")
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before NCCLM2NCopyService()")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCLM2NCopyService requires CUDA")
        super().__init__(group=group)

        self._device = torch.device("cuda", torch.cuda.current_device())
        if not _has_nccl_cuda_backend(group):
            raise RuntimeError("NCCLM2NCopyService requires an NCCL process group")

        _validate_nccl_version(nccl)
        if not callable(getattr(m2n, "group", None)):
            raise RuntimeError("NCCL M2N refit requires the grouped submission API")
        if max_group_bytes is None:
            max_group_bytes = _max_group_bytes_from_env()
        if not isinstance(max_group_bytes, int) or not 0 < max_group_bytes <= _MAX_INT64:
            raise ValueError("max_group_bytes must be a positive int64 value")
        self._max_group_bytes = max_group_bytes
        self._handle = m2n.init()
        self._is_source: bool | None = None
        self._is_destination: bool | None = None
        self._topology: _M2NTopology | None = None
        self._stage_pair_roster: tuple[_StagePair, ...] | None = None
        self._channels: dict[_StagePair, _M2NChannel | None] = {}
        self._closed = False
        self._poisoned = False
        logger.info("NCCLM2NCopyService initialized on rank %d/%d", self.rank, self.world_size)

    def set_model_roles(self, *, is_source: bool, is_destination: bool) -> None:
        """Set this rank's fixed source/destination participation."""
        current = (self._is_source, self._is_destination)
        requested = (is_source, is_destination)
        if current != (None, None) and current != requested:
            raise RuntimeError(
                "NCCL M2N model roles cannot change during the service lifetime; construct the "
                "service for exactly one source mesh and one destination mesh"
            )
        self._is_source = is_source
        self._is_destination = is_destination

    def set_plan(self, plan: object, *, transform: object | None = None) -> None:
        """Adopt the plan's cross-rank-coordinated grouped-submission limit."""
        if not isinstance(plan, ReshardPlan):
            raise TypeError("NCCL M2N requires a ReshardPlan")
        max_group_bytes = plan.execution_batch_bytes
        if max_group_bytes is None:
            return
        if self._topology is not None and max_group_bytes != self._max_group_bytes:
            raise RuntimeError(
                "NCCL M2N grouped-submission limit changed while reusing a service; "
                "close the service before using a plan with a different limit"
            )
        self._max_group_bytes = max_group_bytes

    def submit_send(
        self, src_tensor: torch.Tensor, dest_rank: int, task_id: int | None = None
    ) -> None:
        raise RuntimeError(
            "NCCL M2N requires a whole-tensor ReshardPlan; use execute_reshard_plan()"
        )

    def submit_recv(
        self, dest_tensor: torch.Tensor, src_rank: int, task_id: int | None = None
    ) -> None:
        raise RuntimeError(
            "NCCL M2N requires a whole-tensor ReshardPlan; use execute_reshard_plan()"
        )

    def run(self) -> None:
        raise RuntimeError(
            "NCCL M2N requires a whole-tensor ReshardPlan; use execute_reshard_plan()"
        )

    def _get_topology(self) -> _M2NTopology:
        if self._topology is not None:
            return self._topology
        if self._is_source is None or self._is_destination is None:
            raise RuntimeError(
                "NCCL M2N model roles were not configured; call set_model_roles() "
                "or use swap_model_weights()"
            )
        roles = torch.tensor(
            [int(self._is_source), int(self._is_destination), self._max_group_bytes],
            dtype=torch.int64,
            device=self._device,
        )
        gathered = [torch.empty_like(roles) for _ in range(self.world_size)]
        dist.all_gather(gathered, roles, group=self.group)
        host_values = [item.cpu().tolist() for item in gathered]
        group_byte_limits = {values[2] for values in host_values}
        if len(group_byte_limits) != 1:
            raise RuntimeError(
                f"NCCL M2N max_group_bytes must match on every rank, got "
                f"{sorted(group_byte_limits)}"
            )
        host_roles = [tuple(bool(value) for value in values[:2]) for values in host_values]
        self._topology = _validate_role_roster(host_roles)
        return self._topology

    def _validate_specs(self, topology: _M2NTopology, specs: list[TensorReshardSpec]) -> None:
        src_roster = set(topology.src_ranks)
        dst_roster = set(topology.dst_ranks)
        for spec in specs:
            if not spec.src_ranks or not spec.dst_ranks:
                raise RuntimeError(f"NCCL M2N plan for {spec.resolved_name} has an empty mesh")
            if not set(spec.src_ranks) <= src_roster:
                raise RuntimeError(
                    f"NCCL M2N source mesh for {spec.resolved_name} contains non-source ranks"
                )
            if not set(spec.dst_ranks) <= dst_roster:
                raise RuntimeError(
                    f"NCCL M2N destination mesh for {spec.resolved_name} contains "
                    "non-destination ranks"
                )
            if set(spec.src_ranks) & set(spec.dst_ranks):
                raise RuntimeError(f"NCCL M2N meshes overlap for {spec.resolved_name}")

    def _broadcast_unique_id(self, root_rank: int) -> Any:
        unique_id = bytes(nccl_core.get_unique_id(empty=self.rank != root_rank))
        if self.rank == root_rank and not unique_id:
            raise RuntimeError("NCCL4Py returned an empty NCCL unique ID")
        unique_id_tensor = torch.tensor(list(unique_id), dtype=torch.uint8, device=self._device)
        if self.rank != root_rank and unique_id_tensor.numel() == 0:
            unique_id_tensor = torch.empty(128, dtype=torch.uint8, device=self._device)
        src_rank = root_rank if self.group is None else dist.get_global_rank(self.group, root_rank)
        dist.broadcast(unique_id_tensor, src=src_rank, group=self.group)
        return nccl_core.UniqueId.from_bytes(bytes(unique_id_tensor.cpu().tolist()))

    def _prepare_channels(self, pairs: tuple[_StagePair, ...]) -> None:
        """Collectively bootstrap one exact communicator per stage pair."""
        if self._stage_pair_roster is not None and self._stage_pair_roster != pairs:
            raise RuntimeError(
                "NCCL M2N stage-pair topology changed while reusing a service; close the "
                "service before using a different reshard plan"
            )
        self._stage_pair_roster = pairs
        for pair in pairs:
            if pair in self._channels:
                continue
            src_ranks, dst_ranks = pair
            members = src_ranks + dst_ranks
            unique_id = self._broadcast_unique_id(members[0])
            if self.rank in members:
                channel_rank = members.index(self.rank)
                comm = nccl_core.Communicator.init(len(members), channel_rank, unique_id)
                stream = torch.cuda.Stream(device=self._device)
                self._channels[pair] = _M2NChannel(
                    comm=comm,
                    stream=stream,
                    src_mesh=m2n.Mesh((len(src_ranks),), start_rank=0),
                    dst_mesh=m2n.Mesh((len(dst_ranks),), start_rank=len(src_ranks)),
                )
            else:
                self._channels[pair] = None

    @staticmethod
    def _local_tensor(
        spec: TensorReshardSpec,
        rank: int,
        src_tensors: Mapping[str, torch.Tensor],
        dst_tensors: Mapping[str, torch.Tensor],
        *,
        src_override: torch.Tensor | None = None,
        dst_override: torch.Tensor | None = None,
    ) -> _LocalTransfer:
        """Select contiguous local component buffers for one native call."""
        src = None
        dst = None
        dst_view = None
        if rank in spec.src_ranks:
            if src_override is None:
                if spec.src_param_name is None or spec.src_param_name not in src_tensors:
                    raise RuntimeError(
                        f"NCCL M2N source tensor for {spec.resolved_name} is unavailable on "
                        f"rank {rank}"
                    )
                src_override = _ensure_sendable(src_tensors[spec.src_param_name]).detach()
            expected_param_shape = spec.src_param_shape or spec.src_local_shape
            if tuple(src_override.shape) != expected_param_shape:
                raise RuntimeError(
                    f"NCCL M2N source parameter shape changed for {spec.resolved_name}: "
                    f"expected {expected_param_shape}, got {tuple(src_override.shape)}"
                )
            src = src_override if spec.src_slice is None else src_override[spec.src_slice]
            if tuple(src.shape) != spec.src_local_shape or src.dtype != spec.dtype:
                raise RuntimeError(
                    f"NCCL M2N source tensor metadata changed for {spec.resolved_name}: "
                    f"expected {spec.src_local_shape}/{spec.dtype}, got "
                    f"{tuple(src.shape)}/{src.dtype}"
                )
            if not src.is_contiguous():
                src = src.contiguous()
        if rank in spec.dst_ranks:
            if dst_override is None:
                if spec.dst_param_name is None or spec.dst_param_name not in dst_tensors:
                    raise RuntimeError(
                        f"NCCL M2N destination tensor for {spec.resolved_name} is unavailable "
                        f"on rank {rank}"
                    )
                dst_override = dst_tensors[spec.dst_param_name].detach()
            expected_param_shape = spec.dst_param_shape or spec.dst_local_shape
            if tuple(dst_override.shape) != expected_param_shape:
                raise RuntimeError(
                    f"NCCL M2N destination parameter shape changed for {spec.resolved_name}: "
                    f"expected {expected_param_shape}, got {tuple(dst_override.shape)}"
                )
            dst_view = dst_override if spec.dst_slice is None else dst_override[spec.dst_slice]
            if tuple(dst_view.shape) != spec.dst_local_shape or dst_view.dtype != spec.dtype:
                raise RuntimeError(
                    f"NCCL M2N destination tensor metadata changed for {spec.resolved_name}: "
                    f"expected {spec.dst_local_shape}/{spec.dtype}, got "
                    f"{tuple(dst_view.shape)}/{dst_view.dtype}"
                )
            if dst_view.is_contiguous():
                dst = dst_view
                dst_view = None
            else:
                dst = torch.empty(spec.dst_local_shape, dtype=spec.dtype, device=dst_view.device)
        return _LocalTransfer(src=src, dst=dst, dst_view=dst_view)

    @staticmethod
    def _placement(shard_dim: int | None) -> tuple[Any]:
        return (m2n.Replicate() if shard_dim is None else m2n.Shard(shard_dim),)

    def _enqueue_parameter(
        self,
        parameter_specs: list[TensorReshardSpec],
        channel: _M2NChannel,
        src_tensors: Mapping[str, torch.Tensor],
        dst_tensors: Mapping[str, torch.Tensor],
        transform: MXFP8ReshardTransform | None,
    ) -> _PendingParameter:
        """Prepare one parameter and record all of its M2N component calls."""
        first_spec = parameter_specs[0]
        if [spec.part_index for spec in parameter_specs] != list(
            range(first_spec.part_count)
        ) or any(spec.part_count != first_spec.part_count for spec in parameter_specs):
            raise RuntimeError(
                f"NCCL M2N plan for {first_spec.resolved_name} has incomplete "
                "packed component metadata"
            )

        src_override = None
        if self.rank in first_spec.src_ranks:
            if first_spec.src_param_name is None or first_spec.src_param_name not in src_tensors:
                raise RuntimeError(
                    f"NCCL M2N source tensor for {first_spec.resolved_name} is unavailable "
                    f"on rank {self.rank}"
                )
            src_override = _ensure_sendable(src_tensors[first_spec.src_param_name]).detach()

        transform_args = None
        dst_override = None
        should_transform = (
            transform is not None
            and self.rank in first_spec.dst_ranks
            and first_spec.dst_param_name is not None
            and transform.should_transform(first_spec.dst_param_name)
        )
        if should_transform:
            assert transform is not None
            assert first_spec.dst_param_name is not None
            dst_param_shape = first_spec.dst_param_shape or first_spec.dst_local_shape
            full_slice = tuple(slice(None) for _ in dst_param_shape)
            recv_buffers = transform.prepare_recv(first_spec.dst_param_name, full_slice)
            if len(recv_buffers) != 1:
                raise RuntimeError(
                    "NCCL M2N receiver-side MXFP8 conversion requires exactly "
                    f"one BF16 buffer for {first_spec.resolved_name}"
                )
            dst_override = recv_buffers[0]
            transform_args = (first_spec.dst_param_name, full_slice, recv_buffers)

        local_transfers = []
        for spec in parameter_specs:
            local = self._local_tensor(
                spec,
                self.rank,
                src_tensors,
                dst_tensors,
                src_override=src_override,
                dst_override=dst_override,
            )
            src = m2n.DistTensor(
                local.src,
                local_shape=spec.src_local_shape,
                dtype=spec.dtype,
                mesh=channel.src_mesh,
                placements=self._placement(spec.src_shard_dim),
            )
            dst = m2n.DistTensor(
                local.dst,
                local_shape=spec.dst_local_shape,
                dtype=spec.dtype,
                mesh=channel.dst_mesh,
                placements=self._placement(spec.dst_shard_dim),
            )
            self._handle.reshard(channel.comm, src, dst, stream=channel.stream)
            local_transfers.append(local)
        return _PendingParameter(local_transfers=local_transfers, transform_args=transform_args)

    @staticmethod
    def _complete_parameter(
        pending: _PendingParameter, channel: _M2NChannel, transform: MXFP8ReshardTransform | None
    ) -> None:
        """Schedule writebacks and conversion after a group has been submitted."""
        for local in pending.local_transfers:
            if local.src is not None:
                local.src.record_stream(channel.stream)
            if local.dst is not None:
                local.dst.record_stream(channel.stream)
            if local.dst_view is not None:
                assert local.dst is not None
                local.dst_view.copy_(local.dst)

        if pending.transform_args is not None:
            assert transform is not None
            transform.finalize_recv(*pending.transform_args)

    def execute_plan(
        self,
        plan: object,
        src_tensors: Mapping[str, torch.Tensor],
        dst_tensors: Mapping[str, torch.Tensor],
        *,
        transform: object | None = None,
    ) -> bool:
        """Execute TP/PP parameter shards directly through ``m2n.reshard``."""
        if self._closed:
            raise RuntimeError("NCCLM2NCopyService is closed")
        if self._poisoned:
            raise RuntimeError(
                "NCCLM2NCopyService is unusable after an M2N submission failure; "
                "close it and initialize a new service"
            )
        if not isinstance(plan, ReshardPlan):
            raise TypeError("NCCL M2N requires a ReshardPlan")
        if transform is not None and not isinstance(transform, MXFP8ReshardTransform):
            raise RuntimeError(
                "NCCL M2N native resharding only supports the MXFP8 receiver-side transform"
            )
        mxfp8_transform = transform
        if mxfp8_transform is not None and mxfp8_transform.convert_on_send:
            raise RuntimeError(
                "NCCL M2N native resharding does not support MXFP8 sender-side conversion"
            )
        if plan.tensor_reshard_specs is None:
            reason = plan.tensor_reshard_error or "whole-tensor metadata is unavailable"
            raise RuntimeError(f"NCCL M2N cannot execute this reshard plan: {reason}")

        specs = plan.tensor_reshard_specs
        topology = self._get_topology()
        self._validate_specs(topology, specs)
        pairs = _stage_pairs(specs)
        self._prepare_channels(pairs)

        current_stream = torch.cuda.current_stream(self._device)
        ready = torch.cuda.Event()
        ready.record(current_stream)
        try:
            for pair in pairs:
                channel = self._channels[pair]
                if channel is None:
                    continue
                channel.stream.wait_event(ready)
                pair_specs = [spec for spec in specs if (spec.src_ranks, spec.dst_ranks) == pair]
                with torch.cuda.stream(channel.stream):
                    for batch in _parameter_groups(pair_specs, self._max_group_bytes):
                        pending_parameters = []
                        with m2n.group():
                            for parameter_specs in batch:
                                pending_parameters.append(
                                    self._enqueue_parameter(
                                        parameter_specs,
                                        channel,
                                        src_tensors,
                                        dst_tensors,
                                        mxfp8_transform,
                                    )
                                )
                        for pending in pending_parameters:
                            self._complete_parameter(pending, channel, mxfp8_transform)
            for pair in pairs:
                channel = self._channels[pair]
                if channel is not None:
                    current_stream.wait_stream(channel.stream)
        except BaseException:
            self._poisoned = True
            raise
        return True

    def close(self) -> None:
        """Wait for local work and release M2N resources; this is not collective."""
        if self._closed:
            return
        self._closed = True
        handle, self._handle = self._handle, None
        channels, self._channels = self._channels, {}
        errors: list[BaseException] = []
        try:
            torch.cuda.synchronize(self._device)
        except BaseException as exc:
            errors.append(exc)
        if handle is not None:
            try:
                handle.destroy()
            except BaseException as exc:
                errors.append(exc)
        for channel in channels.values():
            if channel is None:
                continue
            try:
                channel.comm.destroy()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
