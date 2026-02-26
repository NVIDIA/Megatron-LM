# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Optimizer state offloading class."""

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class OptimizerStateOffloader:
    """
    Manages offloading of optimizer states and master weights to CPU.
    Used with DistributedOptimizer to reduce GPU memory usage.

    Supports overlapped D2H/H2D transfers using CUDA streams.

    Master weights can be stored in two locations:
    - In adam optimizer state (when use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True)
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

        # State tracking
        self._offloaded = False
        self._offloaded_state_keys: Tuple[str, ...] = ()
        self._offloaded_mcore_master_weights = False

        # Track whether optimizer states (exp_avg, exp_avg_sq) have been initialized.
        # These are lazily initialized by FusedAdam during the first optimizer.step().
        # Master weights (shard_fp32_from_float16_groups) are available from the start.
        self._optimizer_states_initialized = False

    def mark_optimizer_states_initialized(self):
        """
        Mark that optimizer states (exp_avg, exp_avg_sq) are now available.
        Should be called after the first optimizer.step() completes.
        """
        self._optimizer_states_initialized = True

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
        # Initialize CPU buffers on first call
        if len(cpu_buffers) == 0:
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

        # Copy D2H
        for group_idx, group in enumerate(shard_groups):
            for param_idx, gpu_tensor in enumerate(group):
                cpu_buffer = cpu_buffers[group_idx][param_idx]
                cpu_buffer.copy_(gpu_tensor, non_blocking=pin_memory)
                gpu_tensor.record_stream(self._d2h_stream)

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
                if not isinstance(gpu_tensor, torch.Tensor) or not gpu_tensor.is_cuda:
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

        # Wait for current stream finishing updating the optimizer states.
        self._d2h_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._d2h_stream):
            self._offload_states(offload_optimizer_states, offload_master_weights)

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

        self._release_states()

    def reload(self):
        """
        Reload optimizer states and/or master weights from CPU to GPU.
        Call before optimizer.step() to ensure states are on GPU.
        """
        if not self._offloaded:
            return

        # Allocate GPU memory on the current stream to avoid fragmentation.
        self._reload_states(is_allocate_stage=True)

        self._h2d_stream.wait_stream(self._d2h_stream)
        self._h2d_stream.wait_stream(torch.cuda.current_stream())

        # Reload states on the h2d stream to overlap with other operations.
        with torch.cuda.stream(self._h2d_stream):
            self._reload_states(is_allocate_stage=False)

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
