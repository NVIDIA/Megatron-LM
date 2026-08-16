# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

from typing import Callable, List, Optional, override

import torch
from torch.distributed.tensor import DTensor

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    sync_model_weights_from_main_weights,
)
from ..transformer.module import MegatronModule
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer
from .optimizer_config import OptimizerConfig


def count_replication(tensor: DTensor) -> int:
    """Return how many ranks hold an identical copy of ``tensor``'s local shard.

    A sharded mesh axis holds disjoint pieces that must all be counted; a replicated
    axis holds identical copies that must be counted once, so a gradient statistic
    summed over the grad-stats group has to divide by this.

    MFSDP v2 gradients are always DTensors, so this takes one rather than accepting
    a plain tensor and guessing a layout for it.
    """
    replication = 1
    for axis, placement in enumerate(tensor.placements):
        if placement.is_replicate():
            replication *= tensor.device_mesh.size(axis)
        elif placement.is_partial():
            raise RuntimeError(
                "MFSDP v2 gradient is still Partial when gradient statistics are taken; "
                "the reduction must be finalized first."
            )
    return replication


class FullyShardedOptimizer(MixedPrecisionOptimizer):
    """MCore optimizer wrapper for MFSDP-owned sharded parameters and gradients.

    MFSDP v2 owns the optimizer-facing parameter and gradient shards directly.
    Unlike :class:`DistributedOptimizer`, this wrapper does not build DDP
    param-and-grad-buffer range maps or allocate separate main-parameter shards.
    It preserves MCore's mixed-precision optimizer step contract while making
    MFSDP-specific storage operations explicit.
    """

    @override
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
        model_chunks: List[MegatronModule],
    ) -> None:
        """Initialize the MFSDP optimizer wrapper.

        Args:
            optimizer: Base optimizer such as Adam or SGD.
            config: Optimizer configuration.
            grad_scaler: Optional loss scaler. Currently unsupported for MFSDP v2,
                but accepted to match the MCore optimizer construction contract.
            init_state_fn: Function used to initialize optimizer state.
            model_chunks: MFSDP v2 model chunks optimized by this wrapper.
        """
        FullyShardedOptimizer._validate_config(config, model_chunks)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)
        if grad_scaler is not None:
            raise ValueError("MFSDP v2 does not currently support loss scaling.")

        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        self.model_chunks = model_chunks
        self.ddp_config = self.model_chunks[0].ddp_config
        for model_chunk in self.model_chunks:
            if self.ddp_config != model_chunk.ddp_config:
                raise ValueError("All MFSDP v2 model chunks must share the same ddp_config.")
        self.is_stub_optimizer = optimizer is None
        self._casted_grads = []

    @staticmethod
    def _validate_config(config: OptimizerConfig, model_chunks: List[MegatronModule]) -> None:
        """Validate the MFSDP v2 optimizer support contract."""
        if not model_chunks:
            raise ValueError("MFSDP v2 requires at least one model chunk.")
        if config.use_distributed_optimizer:
            raise ValueError("MFSDP v2 currently requires use_distributed_optimizer=False.")
        if config.loss_scale is not None:
            raise ValueError("MFSDP v2 does not currently support loss scaling.")
        if config.fp16:
            raise ValueError(
                "MFSDP v2 does not currently support FP16 training because FP16 triggers "
                "loss unscale."
            )
        if config.overlap_param_gather_with_optimizer_step:
            raise ValueError("MFSDP v2 does not support optimizer-step parameter-gather overlap.")
        if config.optimizer_cpu_offload:
            raise ValueError("MFSDP v2 does not currently support optimizer CPU offload.")
        if config.use_layer_wise_distributed_optimizer:
            raise ValueError(
                "MFSDP v2 does not currently support layer-wise distributed optimizer."
            )

    @override
    def state_dict(self):
        """Return optimizer state.

        MFSDP v2 optimizer checkpointing needs an FSDP-native DTensor state
        contract. Keep this intentionally unsupported for the prototype instead
        of falling back to DDP-buffer assumptions.
        """
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def get_grad_norm(self):
        """Compute the global gradient L2 norm from each gradient's own DTensor layout.

        MFSDP v2 gradients are DTensors that record how they are distributed, and the
        dense and expert gradients do not share a device mesh: with EP=2 over eight
        ranks the dense gradients live on all eight while the expert gradients live on
        the four-rank expert-DP stripe. Reading the layout off each gradient keeps the
        norm correct without assuming a single mesh for all of them.

        Each rank contributes ``||local||^2`` divided by the product of its replicated
        mesh-axis sizes. A sharded axis holds disjoint pieces that must all be added; a
        replicated axis holds identical copies that must be counted once. Summing that
        over the grad-stats group is then exact, because every shard is held by exactly
        one rank in that group.

        ``get_grad_norm_fp32`` cannot do this: ``get_main_grads_for_grad_norm``
        replaces each DTensor with ``grad._local_tensor`` before it runs, so
        ``get_data_parallel_group_if_dtensor`` always sees plain tensors, returns None,
        and the layout is gone by the time the norm is taken.
        """
        total_norm_squared = torch.zeros(
            (), dtype=torch.float32, device=torch.cuda.current_device()
        )
        for parameter in self.get_parameters():
            # MFSDP v2 reduces into parameter.grad; it never populates decoupled_grad,
            # which is a v1 param-and-grad-buffer concept.
            grad = parameter.grad
            if grad is None:
                continue
            replication = count_replication(grad)
            local_grad = grad.to_local()
            if local_grad.numel() > 0:
                total_norm_squared += local_grad.float().pow(2).sum() / replication

        torch.distributed.all_reduce(
            total_norm_squared,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_grad_stats_parallel_group(),
        )
        return total_norm_squared.sqrt()

    @override
    def count_zeros(self) -> float:
        """Count zero gradient entries from each gradient's own DTensor layout.

        ``count_zeros_fp32`` has the same single-mesh assumption as the grad-norm path,
        and additionally rejects the combination of a Megatron-FSDP parameter with a
        DTensor-derived data-parallel group. Counting here keeps MFSDP v2 off that path,
        and matches how ``get_grad_norm`` reduces: each rank contributes its own shard,
        divided by the size of any replicated mesh axis, summed over the grad-stats group.
        """
        total_zeros = torch.zeros((), dtype=torch.float32, device=torch.cuda.current_device())
        for parameter in self.get_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            replication = count_replication(grad)
            local_grad = grad.to_local()
            if local_grad.numel() > 0:
                zeros = local_grad.numel() - torch.count_nonzero(local_grad)
                total_zeros += zeros.float() / replication

        torch.distributed.all_reduce(
            total_zeros,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_grad_stats_parallel_group(),
        )
        return total_zeros.item()

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear optimizer-visible sharded grads and any grads filtered from local groups."""
        if not self.is_stub_optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)

        # Empty local DTensor shards are filtered out of optimizer param groups
        # as a TE FusedAdam workaround. A rank with no local optimizer params
        # can still have stale module grads to clear.
        for model_chunk in self.model_chunks:
            model_chunk.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        """Step the optimizer, then mark the FSDP execution-trace boundary."""
        result = super().step()
        for model_chunk in self.model_chunks:
            complete_fsdp_trace = getattr(model_chunk, "complete_fsdp_trace", None)
            if complete_fsdp_trace is not None:
                complete_fsdp_trace()
        return result

    def _copy_model_grads_to_main_grads(self) -> None:
        """Install optimizer-compatible gradients for non-precision-aware optimizers."""
        if self.config.use_precision_aware_optimizer:
            return

        assert not self._casted_grads
        for parameter in self.get_parameters():
            if parameter.grad is None:
                continue
            if parameter.grad.dtype == parameter.data.dtype:
                continue

            original_grad = parameter.grad
            parameter.grad = None
            parameter.grad_dtype = parameter.data.dtype
            parameter.grad = original_grad.to(dtype=parameter.data.dtype)
            self._casted_grads.append((parameter, original_grad))

    @override
    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer and restore MFSDP gradient dtypes."""
        success = super().step_with_ready_grads()
        for parameter, original_grad in self._casted_grads:
            parameter.grad = None
            parameter.grad_dtype = original_grad.dtype
            parameter.grad = original_grad
        self._casted_grads.clear()
        return success

    def _copy_main_params_to_model_params(self) -> None:
        """Refresh MFSDP V2 compute weights after updating optimizer weights."""
        sync_model_weights_from_main_weights(self.get_parameters())

    def _copy_model_params_to_main_params(self, state_dict=None) -> None:
        """No-op: model loads already write into MFSDP v2's main weights."""
