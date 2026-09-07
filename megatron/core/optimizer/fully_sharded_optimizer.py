# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

from collections.abc import Iterator
from typing import Any, Callable, List, Optional, override

import torch
from torch.distributed.tensor import DTensor

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..distributed.fsdp.src.megatron_fsdp.experimental import init_optimizer_state
from ..distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    sync_model_weights_from_main_weights,
)
from ..transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
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
        # Each parameter's globally unique checkpoint name, mirroring
        # :class:`DistributedOptimizer`'s ``param_to_name``. Keyed by parameter because that
        # is the direction the checkpoint path resolves, and because the optimizer names each
        # parameter once: a tied parameter is one ``nn.Parameter`` with one state entry, and
        # its several FQNs are the model state dict's business, not this map's.
        self._param_to_fqn: dict[torch.nn.Parameter, str] = {
            param: get_global_unique_param_name(self.model_chunks, param)
            for param in self._trainable_parameters()
        }

    @staticmethod
    def _validate_config(config: OptimizerConfig, model_chunks: List[MegatronModule]) -> None:
        """Validate the MFSDP v2 optimizer support contract."""
        if len(model_chunks) != 1:
            raise ValueError("MFSDP v2 currently supports exactly one model chunk.")
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

        Deliberately unsupported: MFSDP v2 checkpoints through :meth:`sharded_state_dict`,
        and Megatron-FSDP always runs with ``--ckpt-format fsdp_dtensor`` (see
        ``validate_args``), so nothing reaches this accessor. Forwarding the wrapped
        optimizer's state here would only produce a checkpoint that silently drops the
        cross-rank resharding metadata.
        """
        raise NotImplementedError(
            "MFSDP v2 optimizer checkpointing goes through sharded_state_dict "
            "(--ckpt-format fsdp_dtensor)."
        )

    def _trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield the parameters the base optimizer owns, in model order.

        This is the same set ``_get_param_groups`` builds, read off the model so that it is
        available before the optimizer exists and on a rank whose optimizer is a stub.
        """
        for model_chunk in self.model_chunks:
            for param in model_chunk.parameters():
                if param.requires_grad:
                    yield param

    def _raise_if_parameters_span_multiple_meshes(self) -> None:
        """Reject a model whose parameters do not all share one device mesh.

        MFSDP v2 shards expert parameters over the expert-DP mesh and everything else over
        the DP mesh (see :class:`FullyShardedDataParallelV2`), and expert parallelism gives
        each rank a different set of expert FQNs. The ``fsdp_dtensor`` format needs a
        rank-identical DTensor keyspace, because ``preprocess_state_dict_for_uneven_dtensor``
        walks the state dict's DTensors in sorted key order and gathers over each one; a
        keyspace that differs across ranks desynchronizes those collectives.

        Raises:
            NotImplementedError: If the parameters span more than one device mesh, which is
                what expert parallelism produces. Not yet supported rather than refused on
                principle -- describing such a model needs a keyspace built per mesh -- and
                training under expert parallelism is unaffected, only checkpointing.
        """
        meshes = {
            param.device_mesh
            for param in self._trainable_parameters()
            if isinstance(param, DTensor)
        }
        if len(meshes) > 1:
            raise NotImplementedError(
                "MFSDP v2 optimizer checkpointing does not support expert parallelism yet: "
                "its parameters span more than one device mesh."
            )

    def _param_to_group_meta(self) -> dict[str, Any]:
        """Map each parameter's FQN to its param-group hyperparameters.

        The base optimizer (TE FusedAdam) tracks ``step`` per group rather than per
        parameter, so ``step`` round-trips here rather than in the per-parameter state.
        Keying by parameter rather than by group index means a load matches groups by the
        parameters in them, so a checkpoint still applies when the groups are ordered
        differently.
        """
        return {
            self._param_to_fqn[param]: {
                key: value for key, value in group.items() if key != "params"
            }
            for group in self.optimizer.param_groups
            for param in group["params"]
        }

    def _param_groups_from_group_meta(
        self, param_to_group_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Rebuild FQN-keyed torch param groups matching this rank's current optimizer.

        Iterating ``self.optimizer.param_groups`` preserves the current parameter order
        within each group, which is what :meth:`torch.optim.Optimizer.load_state_dict` uses
        to map the checkpoint's FQN-keyed state onto parameter tensors.
        """
        param_groups = []
        for group in self.optimizer.param_groups:
            fqns = [self._param_to_fqn[param] for param in group["params"]]
            missing = [fqn for fqn in fqns if fqn not in param_to_group_meta]
            if missing:
                raise ValueError(
                    f"Parameters {missing} are missing from the checkpoint's "
                    "param_to_group_meta; the checkpoint's optimizer param groups do not "
                    "match this model."
                )
            # Every parameter of a group carries that group's hyperparameters, so read them
            # from the first one.
            hyperparameters = param_to_group_meta[fqns[0]] if fqns else {}
            param_groups.append({"params": fqns, **hyperparameters})
        return param_groups

    @override
    def load_state_dict(self, state_dict: ShardedStateDict) -> None:
        """Load optimizer state produced by :meth:`sharded_state_dict`.

        By the time this runs DCP has already written the checkpoint's tensors into the
        resting optimizer-state DTensors in place, since ``sharded_state_dict(is_loading=True)``
        exposed them as the load destinations. What is left is to restore the param-group
        hyperparameters (including the group-level ``step``) and re-bind the FQN-keyed state
        to the current parameters, mirroring the Megatron-FSDP branch of
        :meth:`DistributedOptimizer.load_state_dict`.
        """
        self.optimizer.load_state_dict(
            {
                "state": state_dict["state"],
                "param_groups": self._param_groups_from_group_meta(
                    state_dict["param_to_group_meta"]
                ),
            }
        )

    @override
    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build the ``fsdp_dtensor`` sharded optimizer state dict.

        The layout mirrors :meth:`DistributedOptimizer.sharded_param_state_fsdp_dtensor`, so
        v1 and v2 write the same on-disk format::

            {"state": {fqn: {"exp_avg": DTensor, "exp_avg_sq": DTensor}},
             "param_to_group_meta": {fqn: {...group hyperparameters...}}}

        That is also why the FQNs come from ``get_global_unique_param_name`` instead of
        torch's :func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`,
        whose names are the plain ``named_parameters`` ones rather than MCore's PP/EP-unique
        checkpoint names.

        Rank consistency is the load-bearing invariant here: every rank's optimizer holds
        every trainable parameter, including the ones whose local shard is empty, so the
        emitted DTensor keyspace is the same on all of them. That is what
        ``preprocess_state_dict_for_uneven_dtensor`` needs, since it walks the DTensors in
        sorted key order and gathers over each one.

        Args:
            model_sharded_state_dict: Accepted for interface parity; the optimizer state is
                read from the wrapped optimizer directly.
            is_loading: Whether the state dict will be filled by a load. If so, the optimizer
                state is materialized first so the load has DTensors to write into.
            metadata: Accepted for interface parity; the ``fsdp_dtensor`` format takes no
                sharding options.

        Returns:
            The optimizer state dict, in the ``fsdp_dtensor`` format described above.
        """
        self._raise_if_parameters_span_multiple_meshes()
        if is_loading:
            init_optimizer_state(self.optimizer)

        packed_state = {
            self._param_to_fqn[param]: param_state
            for param, param_state in self.optimizer.state.items()
        }
        return {"state": packed_state, "param_to_group_meta": self._param_to_group_meta()}

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
