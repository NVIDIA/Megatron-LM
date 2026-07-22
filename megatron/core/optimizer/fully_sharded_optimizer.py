# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

from typing import Callable, List, Optional

import torch

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..transformer.module import MegatronModule
from ..utils import to_local_if_dtensor
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer
from .optimizer_config import OptimizerConfig


class FullyShardedOptimizer(MixedPrecisionOptimizer):
    """MCore optimizer wrapper for MFSDP-owned sharded parameters and gradients.

    MFSDP v2 owns the optimizer-facing parameter and gradient shards directly.
    Unlike :class:`DistributedOptimizer`, this wrapper does not build DDP
    param-and-grad-buffer range maps or allocate separate main-parameter shards.
    It preserves MCore's mixed-precision optimizer step contract while making
    MFSDP-specific storage operations explicit.
    """

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
        self._validate_config(config, model_chunks)
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

    @staticmethod
    def _validate_config(config: OptimizerConfig, model_chunks: List[MegatronModule]) -> None:
        """Validate the MFSDP v2 optimizer support contract."""
        if len(model_chunks) != 1:
            raise ValueError("MFSDP v2 currently supports exactly one model chunk.")
        if not config.use_distributed_optimizer:
            raise ValueError("MFSDP v2 currently requires use_distributed_optimizer=True.")
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
        if config.use_precision_aware_optimizer:
            raise ValueError("MFSDP v2 does not currently support precision-aware optimizer.")
        if config.use_layer_wise_distributed_optimizer:
            raise ValueError(
                "MFSDP v2 does not currently support layer-wise distributed optimizer."
            )
        if config.optimizer_cuda_graph:
            raise ValueError("MFSDP v2 does not currently support optimizer CUDA graphs.")

    def state_dict(self):
        """Return optimizer state.

        MFSDP v2 optimizer checkpointing needs an FSDP-native DTensor state
        contract. Keep this intentionally unsupported for the prototype instead
        of falling back to DDP-buffer assumptions.
        """
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear optimizer-visible sharded grads and any grads filtered from local groups."""
        if not self.is_stub_optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)

        # Empty local DTensor shards are filtered out of optimizer param groups
        # as a TE FusedAdam workaround. A rank with no local optimizer params
        # can still have stale module grads to clear.
        for model_chunk in self.model_chunks:
            model_chunk.zero_grad(set_to_none=set_to_none)

    def _copy_model_grads_to_main_grads(self) -> None:
        """No-op: MFSDP v2 reduces directly into optimizer-visible sharded grads."""

    def _copy_main_params_to_model_params(self) -> None:
        """No-op: MFSDP v2 currently syncs compute weights in its forward pre-hook."""

    def _copy_model_params_to_main_params(self, state_dict=None) -> None:
        """No-op: model loads already write into MFSDP v2's main weights."""

    @torch.no_grad()
    def get_grad_norm(self) -> float:
        """Return the L2 norm over uniquely owned dense and EP-local gradient shards.

        MFSDP v2 + EP can place optimizer-visible DTensors on two different meshes:
        dense parameters use ``dp_cp`` while routed experts use singleton ``expt_dp``.
        Under the currently supported topology (TP=PP=CP=1 and expert-DP=1), every
        local gradient shard is uniquely owned and one world sum produces the global
        dense-plus-expert norm without double counting either mesh.
        """
        local_squared_norm = torch.zeros(1, dtype=torch.float32, device="cuda")
        for grad in self.get_grads_for_grad_norm():
            local_grad = to_local_if_dtensor(grad).detach().float()
            local_squared_norm += torch.sum(local_grad * local_grad)
        torch.distributed.all_reduce(local_squared_norm, op=torch.distributed.ReduceOp.SUM)
        return local_squared_norm.sqrt().item()

    @torch.no_grad()
    def clip_grad_norm(self, clip_grad: float) -> float:
        """Clip mixed-mesh MFSDP gradients using their global unique-shard norm."""
        grad_norm = self.get_grad_norm()
        if clip_grad > 0.0:
            clip_coefficient = min(clip_grad / (grad_norm + 1.0e-6), 1.0)
            if clip_coefficient < 1.0:
                for parameter in self.get_parameters():
                    if parameter.grad is not None:
                        to_local_if_dtensor(parameter.grad).mul_(clip_coefficient)
        return grad_norm
