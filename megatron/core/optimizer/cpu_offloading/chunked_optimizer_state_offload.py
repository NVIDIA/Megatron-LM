# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunked GPU optimizer-state execution backed by CPU canonical storage.

The wrapped optimizer continues to own its normal param_groups and state mappings.
Parameters selected for offload keep their master weights and tensor states in pinned CPU
memory between updates. Immediately before an update, master weights are restored as one
window and optimizer tensor states are restored one chunk at a time. The external optimizer
therefore only ever observes regular CUDA tensors and does not need an offload-specific API.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import torch

from megatron.core.utils import log_single_rank

_MASTER_PARAM_KEY = "master_param"
_NON_OFFLOADABLE_STATE_KEYS = frozenset({_MASTER_PARAM_KEY, "step", "found_inf"})

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OptimizerStateChunk:
    """A group of optimizer parameters updated as one GPU-resident window.

    Attributes:
        params: Parameters whose optimizer tensor states are staged together.
    """

    params: tuple[torch.Tensor, ...]


@dataclasses.dataclass
class _StatePrefetch:
    """Prefetch token whose existence also records a scheduled no-copy chunk."""

    event: torch.cuda.Event | None


@dataclasses.dataclass
class _StateStagingSlot:
    """Reusable GPU storage for one optimizer-state pipeline window."""

    buffers: Dict[tuple[torch.device, torch.dtype], torch.Tensor] = dataclasses.field(
        default_factory=dict
    )


class ChunkedOptimizerStateOffloader:
    """Manage CPU canonical optimizer state and chunked GPU optimizer updates.

    chunk_size_bytes is a soft state-size limit. Parameters are atomic units: a single
    parameter whose state exceeds the limit forms an oversized chunk. This is required by
    matrix optimizers such as Muon and also avoids fabricating optimizer parameters for slices
    of a DistributedOptimizer shard.

    Master weights are always offloaded for selected mixed-precision parameters. They use a
    full-window fallback: all selected masters are restored before the first state chunk and
    copied back after the optimizer/model-parameter staging phase. This keeps the external
    optimizer and FP8 parameter-gather paths unchanged while still removing masters from the
    forward/backward residency set.

    Args:
        optimizer: The external torch-compatible optimizer to execute.
        master_params: MCore-owned FP32 master parameters. These objects also appear in
            optimizer.param_groups and retain their identity while their data moves between
            CPU and CUDA.
        chunk_size_bytes: Target bytes per optimizer-state chunk. Zero means one full chunk.
        offload_fraction: Fraction of optimizer parameter bundles selected for offload.
        state_dtypes: Dtypes of the supported full-size optimizer tensor states. For example,
            Adam supplies its two moment dtypes and Muon supplies its momentum dtype. These are
            used only for deterministic byte planning; the external optimizer remains the
            authority for the actual state schema.
        optimizer_owned_master_dtypes: Dtype of each optimizer-owned ``master_param`` entry.
            Passing the exact parameters that materialize this entry avoids guessing and
            accounts for compact int16 parameter remainders.
        d2h_stream: Optional transfer stream shared by related optimizer wrappers.
        h2d_stream: Optional transfer stream shared by related optimizer wrappers.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        master_params: Sequence[torch.Tensor],
        chunk_size_bytes: int,
        offload_fraction: float,
        state_dtypes: Sequence[torch.dtype],
        optimizer_owned_master_dtypes: Mapping[torch.Tensor, torch.dtype] | None = None,
        d2h_stream: torch.cuda.Stream | None = None,
        h2d_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if chunk_size_bytes < 0:
            raise ValueError(f"chunk_size_bytes must be non-negative, got {chunk_size_bytes}")
        if not 0.0 <= offload_fraction <= 1.0:
            raise ValueError(f"offload_fraction must be in [0, 1], got {offload_fraction}")
        if not state_dtypes:
            raise ValueError("state_dtypes must contain at least one optimizer-state dtype")

        self.optimizer = optimizer
        self.chunk_size_bytes = chunk_size_bytes
        self.offload_fraction = offload_fraction
        self.state_dtypes = tuple(state_dtypes)
        self._state_bytes_per_param = sum(dtype.itemsize for dtype in self.state_dtypes)

        self._params = self._unique_optimizer_params(optimizer.param_groups)
        # Parameter-group membership is fixed for the optimizer lifetime. Cache it once so each
        # state chunk only walks its own parameters instead of rescanning every optimizer group.
        self._param_group_index_by_id = {
            id(param): group_index
            for group_index, group in enumerate(optimizer.param_groups)
            for param in group["params"]
        }
        self._param_devices = {param: param.device for param in self._params}
        self._explicit_master_param_ids = {id(param) for param in master_params}
        self._master_in_optimizer_state = bool(getattr(optimizer, "master_weights", False))
        if self._explicit_master_param_ids and self._master_in_optimizer_state:
            raise ValueError(
                "chunked optimizer state offload cannot manage both MCore-owned master "
                "parameters and optimizer state['master_param'] entries in one optimizer"
            )
        if optimizer_owned_master_dtypes is None:
            # Keep direct construction backward compatible. Production DistributedOptimizer
            # callers pass the exact optimizer-owned subset and its real storage dtype.
            self._optimizer_owned_master_dtypes = (
                {id(param): torch.float32 for param in self._params}
                if self._master_in_optimizer_state
                else {}
            )
        else:
            self._optimizer_owned_master_dtypes = {
                id(param): dtype for param, dtype in optimizer_owned_master_dtypes.items()
            }
            unknown_param_ids = set(self._optimizer_owned_master_dtypes) - {
                id(param) for param in self._params
            }
            if unknown_param_ids:
                raise ValueError(
                    "optimizer_owned_master_dtypes contains parameters outside "
                    "optimizer.param_groups"
                )
            if self._optimizer_owned_master_dtypes and not self._master_in_optimizer_state:
                raise ValueError(
                    "optimizer_owned_master_dtypes requires an optimizer with master_weights"
                )

        self._selected_params = self._select_params_for_offload()
        self._selected_param_ids = {id(param) for param in self._selected_params}
        self._resident_params = tuple(
            param for param in self._params if id(param) not in self._selected_param_ids
        )
        self._chunks = self._build_chunks(self._selected_params)

        self._cpu_state: Dict[torch.Tensor, Dict[str, torch.Tensor]] = collections.defaultdict(dict)
        # This map can be keyed by parameter alone because the constructor rejects combining
        # MCore-owned ``param.data`` masters with optimizer-owned ``state['master_param']`` in
        # one manager, and ``_validate_master_storage`` rechecks that invariant after lazy init.
        self._cpu_master: Dict[torch.Tensor, torch.Tensor] = {}

        self._d2h_stream = d2h_stream if d2h_stream is not None else torch.cuda.Stream()
        self._h2d_stream = h2d_stream if h2d_stream is not None else torch.cuda.Stream()
        self._master_h2d_event: torch.cuda.Event | None = None
        self._master_weights_resident = True
        # Two reusable state windows allow H2D(N+1), step(N), and D2H(N-1) to overlap
        # without letting host run-ahead allocate one CUDA tensor set per chunk.
        self._state_staging_slots = (_StateStagingSlot(), _StateStagingSlot())
        self._next_state_staging_slot = 0
        self._first_prefetch: _StatePrefetch | None = None

    @staticmethod
    def _unique_optimizer_params(param_groups: Sequence[dict]) -> tuple[torch.Tensor, ...]:
        params = []
        seen = set()
        for group in param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return tuple(params)

    def _estimated_bundle_bytes(self, param: torch.Tensor) -> int:
        state_bytes = self._estimated_state_bytes(param)
        if id(param) in self._explicit_master_param_ids:
            master_bytes = param.numel() * param.element_size()
        elif id(param) in self._optimizer_owned_master_dtypes:
            master_bytes = param.numel() * self._optimizer_owned_master_dtypes[id(param)].itemsize
        else:
            master_bytes = 0
        return state_bytes + master_bytes

    def _param_has_master(self, param: torch.Tensor) -> bool:
        return (
            id(param) in self._explicit_master_param_ids
            or id(param) in self._optimizer_owned_master_dtypes
        )

    def _select_params_for_offload(self) -> tuple[torch.Tensor, ...]:
        if not self._params or self.offload_fraction == 0.0:
            return ()
        if self.offload_fraction == 1.0:
            return self._params

        target_bytes = math.ceil(
            sum(self._estimated_bundle_bytes(param) for param in self._params)
            * self.offload_fraction
        )
        # Prefer bundles that actually have a separate master, so a partial byte budget removes
        # both state and master residency before selecting state-only bundles. Python's sort is
        # stable, preserving optimizer order within the two classes.
        candidates = sorted(self._params, key=lambda param: not self._param_has_master(param))
        selected = []
        selected_bytes = 0
        for param in candidates:
            if selected_bytes >= target_bytes:
                break
            selected.append(param)
            selected_bytes += self._estimated_bundle_bytes(param)
        return tuple(selected)

    def _estimated_state_bytes(self, param: torch.Tensor) -> int:
        return param.numel() * self._state_bytes_per_param

    def _build_chunks(self, params: Sequence[torch.Tensor]) -> tuple[OptimizerStateChunk, ...]:
        if not params:
            return ()
        if self.chunk_size_bytes == 0:
            return (OptimizerStateChunk(tuple(params)),)

        chunks = []
        current_params = []
        current_bytes = 0
        oversized_param_count = 0
        largest_oversized_param_bytes = 0
        for param in params:
            param_bytes = self._estimated_state_bytes(param)
            if param_bytes > self.chunk_size_bytes:
                oversized_param_count += 1
                largest_oversized_param_bytes = max(largest_oversized_param_bytes, param_bytes)
            if current_params and current_bytes + param_bytes > self.chunk_size_bytes:
                chunks.append(OptimizerStateChunk(tuple(current_params)))
                current_params = []
                current_bytes = 0
            current_params.append(param)
            current_bytes += param_bytes
        if current_params:
            chunks.append(OptimizerStateChunk(tuple(current_params)))
        if oversized_param_count:
            mib = 1024**2
            log_single_rank(
                logger,
                logging.WARNING,
                "Optimizer-state offload target is %.2f MiB, but %d selected parameter(s) "
                "have an atomic tensor-state payload larger than the target; the largest is "
                "%.2f MiB for %s. These parameters cannot be split, so this manager's "
                "two-slot GPU tensor-state window can exceed roughly twice the configured "
                "target.",
                self.chunk_size_bytes / mib,
                oversized_param_count,
                largest_oversized_param_bytes / mib,
                type(self.optimizer).__name__,
            )
        return tuple(chunks)

    @property
    def chunks(self) -> tuple[OptimizerStateChunk, ...]:
        """Return the immutable optimizer-state chunk plan."""

        return self._chunks

    @property
    def selected_params(self) -> tuple[torch.Tensor, ...]:
        """Return optimizer parameters whose state/master bundle is offloaded."""

        return self._selected_params

    def is_param_offloaded(self, param: torch.Tensor) -> bool:
        """Return whether param belongs to the offloaded fraction."""

        return id(param) in self._selected_param_ids

    @property
    def transfer_streams(self) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """Return the D2H/H2D stream pair used by this manager."""

        return self._d2h_stream, self._h2d_stream

    def use_transfer_streams(
        self, d2h_stream: torch.cuda.Stream, h2d_stream: torch.cuda.Stream
    ) -> None:
        """Adopt streams shared by sibling optimizers before any transfer is scheduled."""

        if d2h_stream is self._d2h_stream and h2d_stream is self._h2d_stream:
            return
        if (
            self._master_h2d_event is not None
            or self._first_prefetch is not None
            or self._cpu_state
            or self._cpu_master
        ):
            raise RuntimeError("optimizer-state transfer streams cannot change after first use")
        self._d2h_stream = d2h_stream
        self._h2d_stream = h2d_stream

    @staticmethod
    def _new_cpu_buffer(tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            tensor.size(), dtype=tensor.dtype, layout=tensor.layout, device="cpu", pin_memory=True
        )

    def _cpu_buffer_for_state(
        self, param: torch.Tensor, key: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        current = self._cpu_state[param].get(key)
        if current is None or current.shape != tensor.shape or current.dtype != tensor.dtype:
            current = self._new_cpu_buffer(tensor)
            self._cpu_state[param][key] = current
        return current

    def _cpu_buffer_for_master(self, param: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        current = self._cpu_master.get(param)
        if current is None or current.shape != tensor.shape or current.dtype != tensor.dtype:
            current = self._new_cpu_buffer(tensor)
            self._cpu_master[param] = current
        return current

    @staticmethod
    def _is_offloadable_state(param: torch.Tensor, key: str, value: object) -> bool:
        """Return whether a tensor belongs to the supported chunked state schema."""

        # Shape is sufficient only for the currently admitted fp32/bf16 Adam moments and the
        # exact Muon momentum schema. FP8 moments carry one-element scale metadata that can match
        # a one-element DistOpt shard; supporting them requires explicit state-key classification.
        return (
            key not in _NON_OFFLOADABLE_STATE_KEYS
            and isinstance(value, torch.Tensor)
            and value.shape == param.shape
        )

    def _state_items_to_offload(self, param: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
        for key, value in self.optimizer.state.get(param, {}).items():
            if self._is_offloadable_state(param, key, value) and value.is_cuda:
                yield key, value

    def _has_unregistered_cuda_state(self, params: Sequence[torch.Tensor]) -> bool:
        """Return whether a step lazily created tensor state outside the staging pool."""

        return any(
            key not in self._cpu_state.get(param, {})
            for param in params
            for key, _ in self._state_items_to_offload(param)
        )

    def _schedule_state_d2h(self, params: Sequence[torch.Tensor]) -> None:
        transfers = [
            (param, key, gpu_tensor)
            for param in params
            for key, gpu_tensor in self._state_items_to_offload(param)
        ]
        if not transfers:
            return

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for param, key, gpu_tensor in transfers:
                cpu_tensor = self._cpu_buffer_for_state(param, key, gpu_tensor)
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                gpu_tensor.record_stream(self._d2h_stream)
                self.optimizer.state[param][key] = cpu_tensor

    def adopt_cpu_optimizer_state(self, params: Sequence[torch.Tensor] | None = None) -> None:
        """Register CPU tensor states created or loaded outside the manager.

        Naturally CPU scalar state is left resident. Tensor state matching an optimizer
        parameter shape is treated as offloaded state; this covers Adam moments and Muon
        momentum while avoiding ordinary CPU step counters. Tracked buffers that are no longer
        part of the optimizer's canonical state are released so checkpoint schema changes do
        not retain stale pinned host memory.
        """

        params = self._selected_params if params is None else params
        for param in params:
            if not self.is_param_offloaded(param):
                continue
            adopted_keys = set()
            for key, value in tuple(self.optimizer.state.get(param, {}).items()):
                if self._is_offloadable_state(param, key, value) and value.device.type == "cpu":
                    if not value.is_pinned():
                        pinned_value = self._new_cpu_buffer(value)
                        pinned_value.copy_(value)
                        self.optimizer.state[param][key] = pinned_value
                        value = pinned_value
                    self._cpu_state[param][key] = value
                    adopted_keys.add(key)

            tracked_state = self._cpu_state.get(param)
            if tracked_state is None:
                continue
            for key in tuple(tracked_state):
                if key not in adopted_keys:
                    del tracked_state[key]
            if not tracked_state:
                del self._cpu_state[param]

    def initialize_state_for_loading(self, init_state_fn: Callable | None, config: object) -> None:
        """Initialize checkpoint state without materializing all selected state on CUDA.

        The optimizer's own initializer remains authoritative. This matters for precision-aware
        FusedAdam, whose state can include scaling metadata and parameter remainders, and for
        Muon, whose state schema is owned by Emerging Optimizers. Resident parameters initialize
        normally; selected parameters initialize one state chunk at a time and are immediately
        returned to CPU canonical storage. ``init_state_fn`` must only initialize per-parameter
        optimizer state; parameter-group metadata writes target temporary subset-group clones and
        are intentionally not committed to the original groups.
        """

        if init_state_fn is None:
            raise RuntimeError(
                "chunked optimizer state offload requires an optimizer state initializer "
                "when preparing a distributed-checkpoint load"
            )

        # Checkpoint preparation may have put explicit MCore masters on CPU. Restore the
        # selected master window so third-party initializers always observe CUDA parameters.
        self._schedule_master_h2d()
        self._wait_master_h2d()

        original_groups = list(self.optimizer.param_groups)
        base_group_metadata = self._snapshot_group_metadata(original_groups)

        def initialize_subset(params: Sequence[torch.Tensor]) -> None:
            groups, _ = self._make_subset_groups(params, base_group_metadata)
            if not groups:
                return
            self.optimizer.param_groups = groups
            init_state_fn(self.optimizer, config)

        try:
            if self._resident_params:
                initialize_subset(self._resident_params)

            for chunk in self._chunks:
                initialize_subset(chunk.params)
                self._schedule_state_d2h(chunk.params)
                # Initialization is an infrequent checkpoint operation. Synchronizing one
                # chunk here bounds its CUDA allocation and makes the CPU destination stable
                # for distributed-checkpoint sharded-state construction.
                self._d2h_stream.synchronize()
                self.adopt_cpu_optimizer_state(chunk.params)

            # Precision-aware FusedAdam creates masters in optimizer.state during init.
            # Scan and offload the full master window once, after all chunks exist.
            self._schedule_master_d2h()
            self._d2h_stream.synchronize()
            self.adopt_cpu_optimizer_state()
        finally:
            self.optimizer.param_groups = original_groups

    def load_state_dict_without_device_cast(self, state_dict: dict) -> None:
        """Load optimizer metadata while preserving each state's staging device.

        ``torch.optim.Optimizer.load_state_dict`` casts every tensor state to the
        corresponding parameter device. The distributed-checkpoint template initializes
        selected states in pinned CPU memory, although a loading strategy may preserve or
        replace those destinations. Casting the resulting CPU state would temporarily
        reconstruct the full state on CUDA. Reproduce the id remapping and parameter-group
        restoration without that cast.
        """

        saved_groups = state_dict["param_groups"]
        current_groups = self.optimizer.param_groups
        if len(saved_groups) != len(current_groups):
            raise ValueError(
                "loaded optimizer has a different number of parameter groups: "
                f"{len(saved_groups)} != {len(current_groups)}"
            )

        saved_param_ids = [param_id for group in saved_groups for param_id in group["params"]]
        current_params = [param for group in current_groups for param in group["params"]]
        if len(saved_param_ids) != len(current_params):
            raise ValueError(
                "loaded optimizer has a different number of parameters: "
                f"{len(saved_param_ids)} != {len(current_params)}"
            )
        id_map = dict(zip(saved_param_ids, current_params))
        current_param_ids = {id(param) for param in current_params}

        restored_groups = []
        for saved_group, current_group in zip(saved_groups, current_groups):
            if len(saved_group["params"]) != len(current_group["params"]):
                raise ValueError("loaded optimizer parameter group has a different size")
            restored_group = dict(saved_group)
            restored_group["params"] = current_group["params"]
            restored_groups.append(restored_group)

        set_scaled_state = getattr(self.optimizer, "set_scaled_state", None)
        state_dtype_map = getattr(self.optimizer, "name_to_dtype_map", None)
        if callable(set_scaled_state) and isinstance(state_dtype_map, dict):
            # TE FusedAdam serializes precision-aware state in its unscaled representation.
            # Invoke TE's public scaling API on CUDA in bounded state chunks, then return selected
            # destinations to CPU instead of installing serialized tensors verbatim.
            # ``initialize_state_for_loading`` has already populated every state slot in the
            # supported schema, so preserving ``current_state`` gives TE valid setter
            # destinations. Unlike torch Optimizer.load_state_dict, a checkpoint that omits a
            # parameter state retains that initialized value; supported checkpoints are complete.
            current_state = self.optimizer.state
            self.optimizer.__setstate__({"state": current_state, "param_groups": restored_groups})
            mapped_states = {}
            for key, saved_state in state_dict["state"].items():
                param = id_map.get(key, key)
                if id(param) not in current_param_ids:
                    self.optimizer.state[param] = saved_state
                    continue

                mapped_states[param] = saved_state

            def restore_scaled_state(param: torch.Tensor, saved_state: dict) -> None:
                for state_name, value in saved_state.items():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and state_name in state_dtype_map:
                        store_remainder = (
                            bool(getattr(self.optimizer, "store_param_remainders", False))
                            and state_name == _MASTER_PARAM_KEY
                            and param.dtype == torch.bfloat16
                        )
                        target_dtype = value.dtype if store_remainder else torch.float32
                        scaled_value = value.to(
                            device=self._param_devices[param],
                            dtype=target_dtype,
                            non_blocking=value.device.type == "cpu" and value.is_pinned(),
                        )
                        set_scaled_state(param, state_name, scaled_value)
                    else:
                        self.optimizer.state[param][state_name] = value

            for param, saved_state in mapped_states.items():
                if not self.is_param_offloaded(param):
                    restore_scaled_state(param, saved_state)

            self._schedule_master_h2d()
            self._wait_master_h2d()
            for chunk in self._chunks:
                for param in chunk.params:
                    saved_state = mapped_states.get(param)
                    if saved_state is not None:
                        restore_scaled_state(param, saved_state)
                # ``initialize_state_for_loading`` already installed CPU destinations. TE's
                # setter scales on CUDA and copies into those existing tensors, so there is no
                # CUDA optimizer state to schedule back through the D2H staging path here.
            self._release_state_staging_slots()
            self._schedule_master_d2h()
            # The TE setter's CUDA-to-CPU copies run on the current stream. This explicit edge
            # also covers precision-aware optimizers without an MCore-owned master D2H transfer.
            self._d2h_stream.wait_stream(torch.cuda.current_stream())
            self._d2h_stream.synchronize()
        else:
            restored_state = collections.defaultdict(dict)
            for key, value in state_dict["state"].items():
                param = id_map.get(key, key)
                if id(param) in current_param_ids and isinstance(value, dict):
                    value = dict(value)
                    if not self.is_param_offloaded(param):
                        for state_name, state_value in tuple(value.items()):
                            if (
                                self._is_offloadable_state(param, state_name, state_value)
                                and state_value.device.type == "cpu"
                            ):
                                value[state_name] = state_value.to(
                                    device=self._param_devices[param],
                                    non_blocking=state_value.is_pinned(),
                                )
                restored_state[param] = value
            self.optimizer.__setstate__({"state": restored_state, "param_groups": restored_groups})
        self.adopt_cpu_optimizer_state()

    def _validate_master_storage(self) -> None:
        for param in self._selected_params:
            if id(param) not in self._explicit_master_param_ids:
                continue
            state_master = self.optimizer.state.get(param, {}).get(_MASTER_PARAM_KEY)
            if isinstance(state_master, torch.Tensor):
                raise RuntimeError(
                    "optimizer created state['master_param'] for a parameter that already uses "
                    "an MCore-owned master parameter"
                )

    def _master_bindings_are_cuda(self) -> bool:
        """Return whether every materialized selected master is CUDA-bound."""

        for param in self._selected_params:
            if id(param) in self._explicit_master_param_ids and not param.data.is_cuda:
                return False
            state_master = self.optimizer.state.get(param, {}).get(_MASTER_PARAM_KEY)
            if isinstance(state_master, torch.Tensor) and not state_master.is_cuda:
                return False
        return True

    def assert_master_weights_resident(self, operation: str) -> None:
        """Reject an external master reader while D2H/H2D changes its canonical binding."""

        if self._master_weights_resident and self._master_bindings_are_cuda():
            return
        raise RuntimeError(
            f"{operation} requires CUDA-resident optimizer master weights, but chunked "
            "optimizer state offload has rebound at least one selected master to CPU or an "
            "H2D restore is still pending"
        )

    def _schedule_master_d2h(self) -> None:
        if not self._selected_params:
            return
        self._validate_master_storage()
        # A rerun or checkpoint can request offload after prefetch but before step().
        # Make the current stream depend on the H2D copy before the D2H stream follows it.
        self._wait_master_h2d()

        transfers = []
        for param in self._selected_params:
            if id(param) in self._explicit_master_param_ids and param.data.is_cuda:
                transfers.append((param, None, param.data))
            state_master = self.optimizer.state.get(param, {}).get(_MASTER_PARAM_KEY)
            if isinstance(state_master, torch.Tensor) and state_master.is_cuda:
                transfers.append((param, _MASTER_PARAM_KEY, state_master))
        if not transfers:
            return

        self._master_weights_resident = False
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for param, state_key, gpu_tensor in transfers:
                cpu_tensor = self._cpu_buffer_for_master(param, gpu_tensor)
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                gpu_tensor.record_stream(self._d2h_stream)
                # CPU is the canonical binding as soon as the copy is enqueued. CUDA's
                # allocator keeps the source alive through record_stream(), and any H2D
                # consumer waits on the D2H stream before reading this buffer.
                if state_key is None:
                    param.data = cpu_tensor
                else:
                    self.optimizer.state[param][state_key] = cpu_tensor

    def _schedule_master_h2d(self) -> None:
        if not self._selected_params or self._master_h2d_event is not None:
            return
        self._validate_master_storage()

        transfers = []
        for param in self._selected_params:
            if id(param) in self._explicit_master_param_ids and not param.data.is_cuda:
                transfers.append((param, None, param.data))
            state_master = self.optimizer.state.get(param, {}).get(_MASTER_PARAM_KEY)
            if isinstance(state_master, torch.Tensor) and not state_master.is_cuda:
                transfers.append((param, _MASTER_PARAM_KEY, state_master))
        if not transfers:
            self._master_weights_resident = self._master_bindings_are_cuda()
            return

        self._master_weights_resident = False
        gpu_transfers = [
            (
                param,
                state_key,
                cpu_tensor,
                torch.empty_like(cpu_tensor, device=self._param_devices[param]),
            )
            for param, state_key, cpu_tensor in transfers
        ]

        self._order_h2d_after_source_streams()
        with torch.cuda.stream(self._h2d_stream):
            for param, state_key, cpu_tensor, gpu_tensor in gpu_transfers:
                gpu_tensor.copy_(cpu_tensor, non_blocking=cpu_tensor.is_pinned())
                gpu_tensor.record_stream(self._h2d_stream)
                if state_key is None:
                    param.data = gpu_tensor
                else:
                    self.optimizer.state[param][state_key] = gpu_tensor
                self._cpu_master[param] = cpu_tensor

            event = torch.cuda.Event()
            event.record(self._h2d_stream)

        self._master_h2d_event = event

    def _wait_master_h2d(self) -> None:
        if self._master_h2d_event is None:
            return
        torch.cuda.current_stream().wait_event(self._master_h2d_event)
        self._master_h2d_event = None
        self._master_weights_resident = self._master_bindings_are_cuda()

    def _order_h2d_after_source_streams(self) -> None:
        """Order H2D after CPU production and compute-stream GPU allocation/use.

        CPU canonical buffers may still be D2H destinations. GPU staging buffers are
        deliberately allocated on the current compute stream so the caching allocator can
        recycle them into later forward/backward allocations. The H2D stream must therefore
        wait for both producer/owner streams before writing either side.
        """

        self._h2d_stream.wait_stream(self._d2h_stream)
        self._h2d_stream.wait_stream(torch.cuda.current_stream())

    def _state_staging_views(
        self, chunk: OptimizerStateChunk
    ) -> List[tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]]:
        """Return state views backed by the next reusable CUDA staging slot."""

        entries = []
        required_numel = collections.defaultdict(int)
        for param in chunk.params:
            state = self.optimizer.state.get(param, {})
            for key, cpu_tensor in self._cpu_state.get(param, {}).items():
                current = state.get(key)
                if not isinstance(current, torch.Tensor) or current.is_cuda:
                    continue
                pool_key = (self._param_devices[param], cpu_tensor.dtype)
                entries.append((param, key, cpu_tensor, pool_key))
                alignment_numel = max(1, 256 // cpu_tensor.element_size())
                required_numel[pool_key] = (
                    (required_numel[pool_key] + alignment_numel - 1) // alignment_numel
                ) * alignment_numel
                required_numel[pool_key] += cpu_tensor.numel()

        if not entries:
            return []

        slot = self._state_staging_slots[self._next_state_staging_slot]
        self._next_state_staging_slot = (self._next_state_staging_slot + 1) % len(
            self._state_staging_slots
        )
        for (device, dtype), numel in required_numel.items():
            buffer = slot.buffers.get((device, dtype))
            if buffer is None or buffer.numel() < numel:
                slot.buffers[(device, dtype)] = torch.empty(numel, dtype=dtype, device=device)

        offsets = collections.defaultdict(int)
        views = []
        for param, key, cpu_tensor, pool_key in entries:
            alignment_numel = max(1, 256 // cpu_tensor.element_size())
            offset = (
                (offsets[pool_key] + alignment_numel - 1) // alignment_numel
            ) * alignment_numel
            buffer = slot.buffers[pool_key]
            gpu_tensor = buffer.narrow(0, offset, cpu_tensor.numel()).view(cpu_tensor.shape)
            offsets[pool_key] = offset + cpu_tensor.numel()
            views.append((param, key, cpu_tensor, gpu_tensor))
        return views

    def _release_state_staging_slots(self) -> None:
        """Drop GPU pool ownership after all live views have been queued for D2H."""

        for slot in self._state_staging_slots:
            slot.buffers.clear()
        self._next_state_staging_slot = 0

    def _prefetch_state(self, chunk: OptimizerStateChunk) -> _StatePrefetch:
        staging_views = self._state_staging_views(chunk)
        if not staging_views:
            return _StatePrefetch(None)

        self._order_h2d_after_source_streams()
        with torch.cuda.stream(self._h2d_stream):
            for param, key, cpu_tensor, gpu_tensor in staging_views:
                gpu_tensor.copy_(cpu_tensor, non_blocking=cpu_tensor.is_pinned())
                gpu_tensor.record_stream(self._h2d_stream)
                self.optimizer.state[param][key] = gpu_tensor
            event = torch.cuda.Event()
            event.record(self._h2d_stream)
        return _StatePrefetch(event)

    @staticmethod
    def _wait_state_prefetch(prefetch: _StatePrefetch) -> None:
        if prefetch.event is not None:
            torch.cuda.current_stream().wait_event(prefetch.event)

    def prefetch_for_step(self) -> None:
        """Asynchronously restore masters and the first state chunk for the next update."""

        if not self._selected_params:
            return
        self._schedule_master_h2d()
        if self._chunks and self._first_prefetch is None:
            self._first_prefetch = self._prefetch_state(self._chunks[0])

    def prefetch_master_for_step(self) -> None:
        """Asynchronously restore only masters, leaving tensor state CPU-resident."""

        if self._selected_params:
            self._schedule_master_h2d()

    def ensure_master_for_param_sync(self) -> None:
        """Ensure selected master weights are available for pre-forward FP8 staging."""

        if self._master_weights_resident and self._master_h2d_event is None:
            return
        self._schedule_master_h2d()
        self._wait_master_h2d()

    def offload_for_forward(self, offload_master: bool = True) -> None:
        """Start D2H for tensors still resident after an update or load.

        Selected master bindings become CPU tensors as soon as their asynchronous D2H copy is
        enqueued. CUDA consumers must restore them through this manager, and host consumers must
        first call :meth:`synchronize_for_checkpoint`. The training loop deliberately calls this
        only at the optimizer-to-forward lifecycle boundary, where no other master reader runs.

        Args:
            offload_master: If false, stage only optimizer state. The training loop uses this
                while an MXFP8 parameter buffer still needs to read updated masters.
        """

        if not self._selected_params:
            return
        if self._first_prefetch is not None:
            self._wait_state_prefetch(self._first_prefetch)
            self._first_prefetch = None
        for chunk in self._chunks:
            self._schedule_state_d2h(chunk.params)
        self._release_state_staging_slots()
        if offload_master:
            self._schedule_master_d2h()

    def synchronize_for_checkpoint(self) -> None:
        """Make CPU canonical state stable and visible to distributed checkpointing."""

        self.offload_for_forward()
        self._d2h_stream.synchronize()
        self.adopt_cpu_optimizer_state()

    @classmethod
    def _clone_group_value(cls, value: object) -> object:
        """Clone mutable group metadata so each chunk starts from one logical step."""

        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: cls._clone_group_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._clone_group_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_group_value(item) for item in value)
        if isinstance(value, set):
            return {cls._clone_group_value(item) for item in value}
        return value

    @classmethod
    def _group_values_equal(cls, left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            compatible = (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
                and left.device == right.device
            )
            if not compatible:
                return False
            # CUDA value comparison would synchronize the host once per chunk. Each subset
            # starts from its own clone of the same metadata; commit the first result and only
            # validate structural compatibility for CUDA tensor-valued fields. Snapshot/commit
            # replaces tensor-valued group fields with clones, so identity-sensitive consumers
            # must not retain references to those tensors; optimizer CUDA graphs are rejected.
            return left.is_cuda or torch.equal(left, right)
        if isinstance(left, dict) or isinstance(right, dict):
            return (
                isinstance(left, dict)
                and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(cls._group_values_equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (
                isinstance(left, (list, tuple))
                and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(cls._group_values_equal(x, y) for x, y in zip(left, right))
            )
        try:
            result = left == right
        except (TypeError, ValueError):
            return left is right
        try:
            return bool(result)
        except (TypeError, ValueError):
            return left is right

    @classmethod
    def _snapshot_group_metadata(cls, groups: Sequence[dict]) -> List[dict]:
        return [
            {key: cls._clone_group_value(value) for key, value in group.items() if key != "params"}
            for group in groups
        ]

    def _make_subset_groups(
        self, params: Sequence[torch.Tensor], base_group_metadata: Sequence[dict]
    ) -> tuple[List[dict], List[tuple[int, dict]]]:
        params_by_group = collections.defaultdict(list)
        for param in params:
            group_index = self._param_group_index_by_id.get(id(param))
            if group_index is None:
                raise RuntimeError(
                    "optimizer parameter-group membership changed after offload setup"
                )
            params_by_group[group_index].append(param)

        groups = []
        indexed_groups = []
        for group_index in sorted(params_by_group):
            group = {
                key: self._clone_group_value(value)
                for key, value in base_group_metadata[group_index].items()
            }
            group["params"] = params_by_group[group_index]
            groups.append(group)
            indexed_groups.append((group_index, group))
        return groups, indexed_groups

    def _step_subset(
        self,
        params: Sequence[torch.Tensor],
        original_groups: Sequence[dict],
        base_group_metadata: Sequence[dict],
        resulting_group_metadata: Dict[int, dict],
    ) -> None:
        groups, indexed_groups = self._make_subset_groups(params, base_group_metadata)
        if not groups:
            return
        self.optimizer.param_groups = groups
        try:
            self.optimizer.step()
            for group_index, group in indexed_groups:
                metadata = self._snapshot_group_metadata([group])[0]
                previous = resulting_group_metadata.get(group_index)
                if previous is not None and not self._group_values_equal(previous, metadata):
                    raise RuntimeError(
                        "optimizer parameter-group updates differ across state chunks for "
                        "metadata comparable without CUDA value synchronization; chunked "
                        "optimizer state offload requires group-level updates to be independent "
                        "of the parameter subset. CUDA tensor-valued fields are validated only "
                        "by shape, dtype, and device"
                    )
                if previous is None:
                    resulting_group_metadata[group_index] = metadata
        finally:
            self.optimizer.param_groups = original_groups

    @torch.no_grad()
    def step(self) -> None:
        """Run the external optimizer over resident parameters and staged state chunks."""

        if not self._selected_params:
            self.optimizer.step()
            return

        self.prefetch_for_step()
        self._wait_master_h2d()

        original_groups = list(self.optimizer.param_groups)
        base_group_metadata = self._snapshot_group_metadata(original_groups)
        resulting_group_metadata: Dict[int, dict] = {}

        try:
            # Resident state needs no transfers and provides useful compute while the first
            # selected chunk is being prefetched.
            if self._resident_params:
                self._step_subset(
                    self._resident_params,
                    original_groups,
                    base_group_metadata,
                    resulting_group_metadata,
                )

            current_prefetch = self._first_prefetch
            self._first_prefetch = None
            for chunk_index, chunk in enumerate(self._chunks):
                if current_prefetch is None:
                    current_prefetch = self._prefetch_state(chunk)
                self._wait_state_prefetch(current_prefetch)

                next_prefetch = None
                if chunk_index + 1 < len(self._chunks):
                    next_prefetch = self._prefetch_state(self._chunks[chunk_index + 1])

                self._step_subset(
                    chunk.params, original_groups, base_group_metadata, resulting_group_metadata
                )
                created_state = self._has_unregistered_cuda_state(chunk.params)
                self._schedule_state_d2h(chunk.params)
                if created_state:
                    # External optimizers commonly allocate moments lazily on their first step.
                    # That storage is not backed by the reusable pool yet, so drain this one-time
                    # initialization copy before the next chunk to preserve the peak-memory bound.
                    self._d2h_stream.synchronize()
                current_prefetch = next_prefetch
        finally:
            self.optimizer.param_groups = original_groups
            self._release_state_staging_slots()

        for group_index, metadata in resulting_group_metadata.items():
            original_group = original_groups[group_index]
            for key in tuple(original_group):
                if key != "params" and key not in metadata:
                    del original_group[key]
            original_group.update(metadata)
