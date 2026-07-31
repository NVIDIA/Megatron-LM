# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Optimizer state offloading class."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class OptimizerStateOffloader:
    """
    Manages offloading of optimizer states and master weights to CPU.
    Used with DistributedOptimizer to reduce GPU memory usage.

    Full-reload mode overlaps D2H/H2D transfers via CUDA streams; the chunked
    path is synchronous per chunk.

    Master weights can be stored in two locations:
    - In Adam optimizer state (when FusedAdam was constructed with master_weights=True)
    - In mcore's shard_fp32_from_float16_groups
    """

    OPTIMIZER_STATE_KEYS = ('exp_avg', 'exp_avg_sq')
    MASTER_WEIGHT_KEY = 'master_param'

    def __init__(self, distrib_optimizer: "DistributedOptimizer"):
        """
        Args:
            distrib_optimizer: The DistributedOptimizer to offload states and master weights from.
        """
        self.dist_optimizer = distrib_optimizer
        self.adam_optimizer = distrib_optimizer.optimizer

        # Only support TE FusedAdam optimizer for now.
        try:
            from transformer_engine.pytorch.optimizers import FusedAdam

            assert isinstance(self.adam_optimizer, FusedAdam), (
                f"OptimizerStateOffloader requires TE FusedAdam optimizer, "
                f"but got {type(self.adam_optimizer).__name__}"
            )
        except ImportError:
            raise ImportError(
                "OptimizerStateOffloader requires transformer_engine.pytorch.optimizers.FusedAdam"
            )

        # Check if master weights are stored in adam optimizer state
        self.optimizer_contains_master_weights = self.adam_optimizer.master_weights

        # CUDA streams for async transfers
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()

        # CPU buffers for optimizer states: {param: {key: cpu_tensor}}
        self._opt_state_cpu_buffers: Dict[torch.Tensor, Dict[str, torch.Tensor]] = {}

        # CPU buffers for mcore master weights, matching the structure of source groups
        # List[List[cpu_tensor]]
        self._shard_fp32_from_float16_cpu_buffers: List[List[torch.Tensor]] = []

        # Lazily built {id(gpu master shard): cpu buffer} map for checkpoint-save reads
        self._master_cpu_by_param_id: Optional[Dict[int, torch.Tensor]] = None

        # State tracking
        self._offloaded = False
        self._offloaded_state_keys: Tuple[str, ...] = ()
        self._offloaded_mcore_master_weights = False
        self._d2h_inflight = False
        self._h2d_pending_state_keys: Tuple[str, ...] = ()
        self._h2d_pending_mcore_master_weights = False

        # Track whether optimizer states (exp_avg, exp_avg_sq) have been initialized.
        # These are lazily initialized by FusedAdam during the first optimizer.step().
        # Master weights (shard_fp32_from_float16_groups) are available from the start.
        self._optimizer_states_initialized = False

    def initialize_offloaded_mcore_master_weights(self, cpu_buffers: List[List[torch.Tensor]]):
        """Register mcore master weights that were constructed directly on CPU."""
        assert not self.optimizer_contains_master_weights
        assert len(cpu_buffers) == len(self.dist_optimizer.shard_fp32_from_float16_groups)
        for cpu_group, gpu_group in zip(
            cpu_buffers, self.dist_optimizer.shard_fp32_from_float16_groups
        ):
            assert len(cpu_group) == len(gpu_group)
            for cpu_tensor, gpu_tensor in zip(cpu_group, gpu_group):
                assert cpu_tensor.device.type == "cpu"
                assert cpu_tensor.shape == gpu_tensor.shape
                assert cpu_tensor.dtype == gpu_tensor.dtype
                assert gpu_tensor.is_cuda
                assert gpu_tensor.untyped_storage().size() == 0
        self._shard_fp32_from_float16_cpu_buffers = cpu_buffers
        self._offloaded_mcore_master_weights = True
        self._offloaded = True

    def mark_optimizer_states_initialized(self):
        """
        Mark that optimizer states (exp_avg, exp_avg_sq) are now available.
        Should be called after the first optimizer.step() completes.
        """
        self._optimizer_states_initialized = True

    @staticmethod
    def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
        if not isinstance(tensor, torch.Tensor):
            return 0
        try:
            return int(tensor.untyped_storage().size())
        except RuntimeError:
            return 0

    def _collect_memory_state(self) -> Dict[str, int]:
        """Collect GPU/CPU residency byte counts. Used by unit tests to assert residency."""
        state_gpu_bytes = {key: 0 for key in (*self.OPTIMIZER_STATE_KEYS, self.MASTER_WEIGHT_KEY)}
        state_cpu_bytes = {key: 0 for key in (*self.OPTIMIZER_STATE_KEYS, self.MASTER_WEIGHT_KEY)}
        for param, param_state in self.adam_optimizer.state.items():
            for key in state_gpu_bytes:
                tensor = param_state.get(key, None)
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    state_gpu_bytes[key] += self._tensor_storage_bytes(tensor)
            for key, tensor in self._opt_state_cpu_buffers.get(param, {}).items():
                if key in state_cpu_bytes:
                    state_cpu_bytes[key] += self._tensor_storage_bytes(tensor)

        mcore_master_gpu_bytes = 0
        for group in self.dist_optimizer.shard_fp32_from_float16_groups:
            for tensor in group:
                mcore_master_gpu_bytes += self._tensor_storage_bytes(tensor)

        mcore_master_cpu_bytes = 0
        for group in self._shard_fp32_from_float16_cpu_buffers:
            for tensor in group:
                mcore_master_cpu_bytes += self._tensor_storage_bytes(tensor)

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "state_gpu_exp_avg": state_gpu_bytes['exp_avg'],
            "state_gpu_exp_avg_sq": state_gpu_bytes['exp_avg_sq'],
            "state_gpu_master": state_gpu_bytes[self.MASTER_WEIGHT_KEY],
            "state_cpu_exp_avg": state_cpu_bytes['exp_avg'],
            "state_cpu_exp_avg_sq": state_cpu_bytes['exp_avg_sq'],
            "state_cpu_master": state_cpu_bytes[self.MASTER_WEIGHT_KEY],
            "mcore_master_gpu": mcore_master_gpu_bytes,
            "mcore_master_cpu": mcore_master_cpu_bytes,
        }

    def _get_state_keys_to_offload(
        self, offload_optimizer_states: bool, offload_master_weights: bool
    ) -> Tuple[str, ...]:
        """Get the state keys in FusedAdam to offload based on configuration."""
        keys = []
        # Skip optimizer states offloading if they haven't been initialized yet.
        # Optimizer states are lazily initialized by FusedAdam during the first optimizer.step().
        if self._optimizer_states_initialized:
            if offload_optimizer_states:
                keys.extend(self.OPTIMIZER_STATE_KEYS)
            if offload_master_weights and self.optimizer_contains_master_weights:
                keys.append(self.MASTER_WEIGHT_KEY)
        return tuple(keys)

    def _get_initialized_state_keys_to_offload(
        self, offload_optimizer_states: bool, offload_master_weights: bool
    ) -> Tuple[str, ...]:
        """Get state keys that may already exist for an incrementally stepped param subset."""
        keys = []
        if offload_optimizer_states:
            keys.extend(self.OPTIMIZER_STATE_KEYS)
        if offload_master_weights and self.optimizer_contains_master_weights:
            keys.append(self.MASTER_WEIGHT_KEY)
        return tuple(keys)

    def _ensure_state_cpu_buffer(
        self, param: torch.Tensor, state_key: str, gpu_tensor: torch.Tensor, pin_memory: bool = True
    ) -> torch.Tensor:
        """Get or create a CPU buffer for a state tensor."""
        if param not in self._opt_state_cpu_buffers:
            self._opt_state_cpu_buffers[param] = {}

        if state_key not in self._opt_state_cpu_buffers[param]:
            cpu_buffer = torch.empty(
                gpu_tensor.size(),
                dtype=gpu_tensor.dtype,
                layout=gpu_tensor.layout,
                device='cpu',
                pin_memory=pin_memory,
            )
            self._opt_state_cpu_buffers[param][state_key] = cpu_buffer

        return self._opt_state_cpu_buffers[param][state_key]

    def _offload_shard_groups(
        self,
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
        pin_memory: bool = True,
    ):
        """Offload a shard group to CPU buffers."""
        self._ensure_shard_group_cpu_buffers(shard_groups, cpu_buffers, pin_memory)

        # Copy D2H
        for group_idx, group in enumerate(shard_groups):
            for param_idx, gpu_tensor in enumerate(group):
                if (
                    not isinstance(gpu_tensor, torch.Tensor)
                    or not gpu_tensor.is_cuda
                    or gpu_tensor.untyped_storage().size() == 0
                ):
                    continue
                cpu_buffer = cpu_buffers[group_idx][param_idx]
                cpu_buffer.copy_(gpu_tensor, non_blocking=pin_memory)
                gpu_tensor.record_stream(self._d2h_stream)

    def _ensure_shard_group_cpu_buffers(
        self,
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
        pin_memory: bool = True,
    ):
        """Initialize CPU buffers matching shard groups on first use."""
        if len(cpu_buffers) != 0:
            return

        for group in shard_groups:
            group_buffers = []
            for gpu_tensor in group:
                cpu_buffer = torch.empty(
                    gpu_tensor.size(),
                    dtype=gpu_tensor.dtype,
                    layout=gpu_tensor.layout,
                    device='cpu',
                    pin_memory=pin_memory,
                )
                group_buffers.append(cpu_buffer)
            cpu_buffers.append(group_buffers)

    def _offload_states(
        self,
        offload_optimizer_states: bool,
        offload_master_weights: bool,
        use_pin_memory: bool = True,
    ):
        """Offload optimizer states and/or master weights to CPU."""
        # Offload states from adam optimizer
        self._offloaded_state_keys = self._get_state_keys_to_offload(
            offload_optimizer_states, offload_master_weights
        )
        states = self.adam_optimizer.state

        for param, param_state in states.items():
            for state_key in self._offloaded_state_keys:
                if state_key not in param_state:
                    continue

                gpu_tensor = param_state[state_key]
                if (
                    not isinstance(gpu_tensor, torch.Tensor)
                    or not gpu_tensor.is_cuda
                    or gpu_tensor.untyped_storage().size() == 0
                ):
                    continue

                cpu_buffer = self._ensure_state_cpu_buffer(
                    param, state_key, gpu_tensor, use_pin_memory
                )
                cpu_buffer.copy_(gpu_tensor, non_blocking=use_pin_memory)
                gpu_tensor.record_stream(self._d2h_stream)

        # Offload mcore master weights if not in optimizer state
        if offload_master_weights and not self.optimizer_contains_master_weights:
            self._offload_shard_groups(
                self.dist_optimizer.shard_fp32_from_float16_groups,
                self._shard_fp32_from_float16_cpu_buffers,
                use_pin_memory,
            )
            self._offloaded_mcore_master_weights = True

    def _release_states(self):
        """Replace optimizer state GPU tensors with CPU tensors to free GPU memory."""
        states = self.adam_optimizer.state

        for param, param_state in states.items():
            if param not in self._opt_state_cpu_buffers:
                continue

            for state_key in self._offloaded_state_keys:
                if state_key not in self._opt_state_cpu_buffers[param]:
                    continue

                param_state[state_key].untyped_storage().resize_(0)

        if self._offloaded_mcore_master_weights:
            for group in self.dist_optimizer.shard_fp32_from_float16_groups:
                for gpu_tensor in group:
                    gpu_tensor.untyped_storage().resize_(0)

    def _release_states_for_params(self, params: List[torch.Tensor], state_keys: Tuple[str, ...]):
        """Release selected optimizer state GPU tensors after their CPU copies complete."""
        states = self.adam_optimizer.state

        for param in params:
            param_state = states.get(param, None)
            if param_state is None:
                continue
            for state_key in state_keys:
                state_tensor = param_state.get(state_key, None)
                if isinstance(state_tensor, torch.Tensor) and state_tensor.is_cuda:
                    state_tensor.untyped_storage().resize_(0)

    def _reload_states_for_params(
        self, params: List[torch.Tensor], state_keys: Tuple[str, ...], is_allocate_stage: bool
    ):
        """Reload selected optimizer state tensors from CPU buffers for a subset of params."""
        states = self.adam_optimizer.state

        for param in params:
            param_state = states.get(param, None)
            if param_state is None or param not in self._opt_state_cpu_buffers:
                continue
            for state_key in state_keys:
                if state_key not in self._opt_state_cpu_buffers[param]:
                    continue
                cpu_buffer = self._opt_state_cpu_buffers[param][state_key]
                if is_allocate_stage:
                    param_state[state_key].untyped_storage().resize_(
                        cpu_buffer.untyped_storage().size()
                    )
                else:
                    param_state[state_key].copy_(cpu_buffer, non_blocking=cpu_buffer.is_pinned())

    def _reload_shard_groups(
        self,
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
        is_allocate_stage: bool,
    ):
        """Reload shard groups from CPU to GPU."""
        for group_idx, group in enumerate(shard_groups):
            for param_idx, _ in enumerate(group):
                cpu_buffer = cpu_buffers[group_idx][param_idx]
                if is_allocate_stage:
                    shard_groups[group_idx][param_idx].untyped_storage().resize_(
                        cpu_buffer.untyped_storage().size()
                    )
                else:
                    shard_groups[group_idx][param_idx].copy_(
                        cpu_buffer, non_blocking=cpu_buffer.is_pinned()
                    )

    def _iter_shard_group_entries_for_params(
        self,
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
        params: List[torch.Tensor],
    ):
        """Yield shard group entries whose GPU tensor is in params."""
        param_ids = {id(param) for param in params}
        for group_idx, group in enumerate(shard_groups):
            for param_idx, gpu_tensor in enumerate(group):
                if id(gpu_tensor) in param_ids:
                    yield gpu_tensor, cpu_buffers[group_idx][param_idx]

    def _reload_shard_groups_for_params(
        self,
        params: List[torch.Tensor],
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
        is_allocate_stage: bool,
    ):
        """Reload selected shard tensors from CPU to GPU."""
        for gpu_tensor, cpu_buffer in self._iter_shard_group_entries_for_params(
            shard_groups, cpu_buffers, params
        ):
            if is_allocate_stage:
                gpu_tensor.untyped_storage().resize_(cpu_buffer.untyped_storage().size())
            else:
                gpu_tensor.copy_(cpu_buffer, non_blocking=cpu_buffer.is_pinned())

    def _offload_shard_groups_for_params(
        self,
        params: List[torch.Tensor],
        shard_groups: List[List[torch.Tensor]],
        cpu_buffers: List[List[torch.Tensor]],
    ):
        """Offload selected shard tensors to their existing CPU buffers."""
        self._ensure_shard_group_cpu_buffers(shard_groups, cpu_buffers)
        for gpu_tensor, cpu_buffer in self._iter_shard_group_entries_for_params(
            shard_groups, cpu_buffers, params
        ):
            cpu_buffer.copy_(gpu_tensor, non_blocking=cpu_buffer.is_pinned())
            gpu_tensor.record_stream(self._d2h_stream)

    def _release_shard_groups_for_params(
        self, params: List[torch.Tensor], shard_groups: List[List[torch.Tensor]]
    ):
        """Release selected shard tensors after D2H copy completes."""
        param_ids = {id(param) for param in params}
        for group in shard_groups:
            for gpu_tensor in group:
                if id(gpu_tensor) in param_ids:
                    gpu_tensor.untyped_storage().resize_(0)

    def _reload_states(self, is_allocate_stage: bool):
        """
        Reload optimizer states and/or master weights from CPU to GPU.

        If is_allocate_stage is True, only allocate GPU memory for the states and master weights,
        but do not copy the data from CPU to GPU. Otherwise, copy the data from CPU to GPU.
        The two processes are separated to make sure that the GPU memory is allocated on the
        default stream to avoid fragmentation.
        """
        # Reload states to adam optimizer
        states = self.adam_optimizer.state

        for param, param_state in states.items():
            if param not in self._opt_state_cpu_buffers:
                continue

            for state_key in self._offloaded_state_keys:
                if state_key not in self._opt_state_cpu_buffers[param]:
                    continue

                cpu_buffer = self._opt_state_cpu_buffers[param][state_key]
                if is_allocate_stage:
                    param_state[state_key].untyped_storage().resize_(
                        cpu_buffer.untyped_storage().size()
                    )
                else:
                    param_state[state_key].copy_(cpu_buffer, non_blocking=cpu_buffer.is_pinned())

        # Reload mcore master weights if not in optimizer state
        if self._offloaded_mcore_master_weights:
            self._reload_shard_groups(
                self.dist_optimizer.shard_fp32_from_float16_groups,
                self._shard_fp32_from_float16_cpu_buffers,
                is_allocate_stage,
            )

    def _has_offloaded_work(self) -> bool:
        return bool(self._offloaded_state_keys) or self._offloaded_mcore_master_weights

    def _has_h2d_pending_work(self) -> bool:
        return bool(self._h2d_pending_state_keys) or self._h2d_pending_mcore_master_weights

    def _mark_h2d_pending(self, state_keys: Tuple[str, ...], reload_mcore_master_weights: bool):
        self._h2d_pending_state_keys = tuple(
            dict.fromkeys((*self._h2d_pending_state_keys, *state_keys))
        )
        self._h2d_pending_mcore_master_weights = (
            self._h2d_pending_mcore_master_weights or reload_mcore_master_weights
        )

    def _clear_h2d_pending(self):
        self._h2d_pending_state_keys = ()
        self._h2d_pending_mcore_master_weights = False

    def sync_pending_h2d(self):
        """Synchronize pending H2D reloads before CPU-side optimizer state reads."""
        if not self._has_h2d_pending_work():
            return
        self._h2d_stream.synchronize()
        self._clear_h2d_pending()

    def _mcore_master_cpu_buffers_by_param_id(self) -> Dict[int, torch.Tensor]:
        if not self._shard_fp32_from_float16_cpu_buffers:
            return {}
        if self._master_cpu_by_param_id is None:
            self._master_cpu_by_param_id = {
                id(gpu_tensor): cpu_buffer
                for group, cpu_group in zip(
                    self.dist_optimizer.shard_fp32_from_float16_groups,
                    self._shard_fp32_from_float16_cpu_buffers,
                )
                for gpu_tensor, cpu_buffer in zip(group, cpu_group)
            }
        return self._master_cpu_by_param_id

    def get_offloaded_states_for_read(self, main_param: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return this param's offloaded states as their CPU copies, for checkpoint save.

        Only states whose GPU storage was actually released are returned (keyed by the
        FusedAdam state name, plus MASTER_WEIGHT_KEY for mcore-managed master shards).
        CPU bytes are the true values only for unscaled fp32/bf16 states; that is
        enforced at config validation. Requires sync save: an async writer would race
        the next step's chunk offloads overwriting these same CPU buffers.
        """
        # An enqueued reload() flips _offloaded off before its H2D copies finish;
        # callers then fall back to raw GPU reads, so those copies must complete
        # first (also lets skipped-read paths retire the pending-work flags).
        self.sync_pending_h2d()
        if not self._offloaded:
            return {}
        if self._d2h_inflight:
            self._d2h_stream.synchronize()
            self._d2h_inflight = False

        tensors: Dict[str, torch.Tensor] = {}
        param_state = self.adam_optimizer.state.get(main_param, {})
        cpu_state = self._opt_state_cpu_buffers.get(main_param, {})
        for state_key in self._offloaded_state_keys:
            gpu_tensor = param_state.get(state_key, None)
            if (
                isinstance(gpu_tensor, torch.Tensor)
                and gpu_tensor.untyped_storage().size() == 0
                and state_key in cpu_state
            ):
                tensors[state_key] = cpu_state[state_key]
        if self._offloaded_mcore_master_weights and main_param.untyped_storage().size() == 0:
            cpu_master = self._mcore_master_cpu_buffers_by_param_id().get(id(main_param), None)
            if cpu_master is not None:
                tensors[self.MASTER_WEIGHT_KEY] = cpu_master
        return tensors

    def offload(self, offload_optimizer_states: bool = True, offload_master_weights: bool = True):
        """
        Offload optimizer states and/or master weights to CPU.
        Starts async D2H transfer that can overlap with other operations.

        Args:
            offload_optimizer_states: Whether to offload exp_avg, exp_avg_sq.
            offload_master_weights: Whether to offload master weights.
        """
        if not offload_optimizer_states and not offload_master_weights:
            return

        self.sync_pending_h2d()

        # Wait for current stream finishing updating the optimizer states.
        self._d2h_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._d2h_stream):
            self._offload_states(offload_optimizer_states, offload_master_weights)

        self._offloaded = self._has_offloaded_work()
        self._d2h_inflight = self._offloaded

    def reload_master_weights_for_params(self, params: List[torch.Tensor]):
        """Synchronously reload master weights for the params needed by one chunk."""
        if not self._offloaded:
            return

        reload_master_state = self.MASTER_WEIGHT_KEY in self._offloaded_state_keys
        reload_mcore_master_weights = self._offloaded_mcore_master_weights
        if not reload_master_state and not reload_mcore_master_weights:
            return
        mcore_params_to_reload = [
            param
            for param in params
            if isinstance(param, torch.Tensor) and param.untyped_storage().size() == 0
        ]

        self._h2d_stream.wait_stream(self._d2h_stream)

        if reload_master_state:
            self._reload_states_for_params(
                params, (self.MASTER_WEIGHT_KEY,), is_allocate_stage=True
            )
        if reload_mcore_master_weights and mcore_params_to_reload:
            self._reload_shard_groups_for_params(
                mcore_params_to_reload,
                self.dist_optimizer.shard_fp32_from_float16_groups,
                self._shard_fp32_from_float16_cpu_buffers,
                is_allocate_stage=True,
            )

        self._h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._h2d_stream):
            if reload_master_state:
                self._reload_states_for_params(
                    params, (self.MASTER_WEIGHT_KEY,), is_allocate_stage=False
                )
            if reload_mcore_master_weights and mcore_params_to_reload:
                self._reload_shard_groups_for_params(
                    mcore_params_to_reload,
                    self.dist_optimizer.shard_fp32_from_float16_groups,
                    self._shard_fp32_from_float16_cpu_buffers,
                    is_allocate_stage=False,
                )

        torch.cuda.current_stream().wait_stream(self._h2d_stream)

    def reload_optimizer_states_for_params(self, params: List[torch.Tensor]):
        """Synchronously reload Adam states for the params needed by the next chunk update."""
        state_keys = tuple(
            key for key in self.OPTIMIZER_STATE_KEYS if key in self._offloaded_state_keys
        )
        if not state_keys:
            return

        self._h2d_stream.wait_stream(self._d2h_stream)

        self._reload_states_for_params(params, state_keys, is_allocate_stage=True)

        self._h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._h2d_stream):
            self._reload_states_for_params(params, state_keys, is_allocate_stage=False)

        torch.cuda.current_stream().wait_stream(self._h2d_stream)

    def offload_initialized_states_for_params(
        self,
        params: List[torch.Tensor],
        offload_optimizer_states: bool = True,
        offload_master_weights: bool = True,
    ):
        """Offload optimizer states that were just initialized for a subset of params."""
        state_keys = self._get_initialized_state_keys_to_offload(
            offload_optimizer_states, offload_master_weights
        )
        offload_mcore_master_weights = (
            offload_master_weights and not self.optimizer_contains_master_weights
        )
        if not state_keys and not offload_mcore_master_weights:
            return

        self._d2h_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._d2h_stream):
            states = self.adam_optimizer.state
            for param in params:
                param_state = states.get(param, None)
                if param_state is None:
                    continue
                for state_key in state_keys:
                    gpu_tensor = param_state.get(state_key, None)
                    if not isinstance(gpu_tensor, torch.Tensor) or not gpu_tensor.is_cuda:
                        continue
                    cpu_buffer = self._ensure_state_cpu_buffer(param, state_key, gpu_tensor)
                    cpu_buffer.copy_(gpu_tensor, non_blocking=True)
                    gpu_tensor.record_stream(self._d2h_stream)
            if offload_mcore_master_weights:
                self._offload_shard_groups_for_params(
                    params,
                    self.dist_optimizer.shard_fp32_from_float16_groups,
                    self._shard_fp32_from_float16_cpu_buffers,
                )

        self._d2h_stream.synchronize()
        self._release_states_for_params(params, state_keys)
        if offload_mcore_master_weights:
            self._release_shard_groups_for_params(
                params, self.dist_optimizer.shard_fp32_from_float16_groups
            )
        self._offloaded_state_keys = tuple(
            dict.fromkeys((*self._offloaded_state_keys, *state_keys))
        )
        self._offloaded_mcore_master_weights = (
            self._offloaded_mcore_master_weights or offload_mcore_master_weights
        )
        self._offloaded = True

    def release_gpu_memory(self):
        """
        Release GPU memory for optimizer states and master weights after D2H copy completes.

        This is separated from offload() to allow delayed GPU memory release,
        which is needed for mxfp8 + overlap_param_gather case where master weights
        must remain on GPU until after _copy_main_params_to_param_buffer() is called.
        """
        if not self._offloaded:
            return

        if self._d2h_inflight:
            self._d2h_stream.synchronize()
            self._d2h_inflight = False
        self._release_states()

    def reload(self):
        """
        Reload optimizer states and/or master weights from CPU to GPU.
        Call before optimizer.step() to ensure states are on GPU.
        """
        if not self._offloaded:
            return

        reload_state_keys = self._offloaded_state_keys
        reload_mcore_master_weights = self._offloaded_mcore_master_weights

        # Allocate GPU memory on the current stream to avoid fragmentation.
        self._reload_states(is_allocate_stage=True)

        self._h2d_stream.wait_stream(self._d2h_stream)
        self._h2d_stream.wait_stream(torch.cuda.current_stream())

        # Reload states on the h2d stream to overlap with other operations.
        with torch.cuda.stream(self._h2d_stream):
            self._reload_states(is_allocate_stage=False)

        self._mark_h2d_pending(reload_state_keys, reload_mcore_master_weights)
        self._offloaded_state_keys = ()
        self._offloaded_mcore_master_weights = False
        self._offloaded = False

    def sync_before_step(self):
        """
        Wait for H2D reload to complete before optimizer.step().
        Must be called to ensure states are on GPU before optimizer uses them.

        This is separated from reload() to make it possible to move the reload ahead of time.
        """
        torch.cuda.current_stream().wait_stream(self._h2d_stream)
        self._clear_h2d_pending()
