# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

from collections.abc import Iterator
from typing import Any, Callable, List, Optional

import torch
from torch.distributed.tensor import DTensor, Shard

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
        # Lazily built by _param_fqn.
        self._param_to_fqn: dict[torch.nn.Parameter, str] | None = None

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

    def _param_fqn(self, param: torch.nn.Parameter) -> str:
        """Return a model parameter's globally unique checkpoint name.

        ``get_global_unique_param_name`` rescans every model chunk on each call and the
        checkpoint path resolves each name several times, so the mapping is built once and
        cached, mirroring :class:`DistributedOptimizer`'s ``param_to_name``.
        """
        if self._param_to_fqn is None:
            self._param_to_fqn = {
                candidate: get_global_unique_param_name(self.model_chunks, candidate)
                for candidate in self._trainable_parameters()
            }
        return self._param_to_fqn[param]

    def _trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield the parameters the base optimizer owns before the empty-shard filter.

        ``_get_param_groups`` keeps every ``requires_grad`` parameter, but MFSDP v2 then
        drops the ones whose *local* shard is empty, working around a TE FusedAdam bug (see
        :func:`get_megatron_optimizer`). That filter is applied per rank, so the surviving
        set differs across ranks; this iterator reconstructs the rank-invariant superset.
        """
        for model_chunk in self.model_chunks:
            for param in model_chunk.parameters():
                if param.requires_grad:
                    yield param

    def _shard_process_groups(self) -> list[torch.distributed.ProcessGroup]:
        """Return the process groups across which the parameters are sharded.

        Every sharded dimension of the parameter device mesh contributes one group.
        Replicated dimensions are skipped: replicas hold identical shards, hence an
        identical keyspace. These are the groups ``preprocess_state_dict_for_uneven_dtensor``
        gathers over, so a collective run here reaches the same ranks the save and load
        collectives do.

        Raises:
            NotImplementedError: If the parameters do not all share one device mesh, which
                is what expert parallelism produces.
        """
        sharded_parameters = [
            param for param in self._trainable_parameters() if isinstance(param, DTensor)
        ]
        if len({param.device_mesh for param in sharded_parameters}) > 1:
            # MFSDP v2 shards expert parameters over the expert-DP mesh and everything else
            # over the DP mesh (see FullyShardedDataParallelV2). Expert parallelism also gives
            # each rank a different set of expert FQNs, so the DTensor keyspace this method
            # exists to equalize is rank-dependent for a reason no gather can repair.
            raise NotImplementedError(
                "MFSDP v2 optimizer checkpointing does not support expert parallelism: its "
                "parameters span more than one device mesh."
            )
        if not sharded_parameters:
            return []

        mesh = sharded_parameters[0].device_mesh
        return [
            mesh.get_group(mesh_dim)
            for mesh_dim, placement in enumerate(sharded_parameters[0].placements)
            if isinstance(placement, Shard) and mesh.size(mesh_dim) > 1
        ]

    def _gather_state_keys_by_fqn(self) -> dict[str, list[str]]:
        """Map every parameter's FQN to the keys of its ``DTensor`` optimizer state entries.

        The keys are read off live state (``exp_avg``/``exp_avg_sq`` for Adam) rather than
        hard-coded, so any base optimizer works. They are gathered because a parameter that
        this rank's optimizer filtered out is only described by the rank that owns a
        non-empty shard of it, which is what lets every rank synthesize a placeholder with
        exactly the owning rank's keys. A parameter that has no state anywhere (nothing has
        stepped yet) is simply absent.
        """
        state_keys_by_fqn = {
            self._param_fqn(param): sorted(
                key for key, value in state.items() if isinstance(value, DTensor)
            )
            for param, state in self.optimizer.state.items()
        }
        for group in self._shard_process_groups():
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, state_keys_by_fqn, group=group)
            state_keys_by_fqn = {fqn: keys for rank in gathered for fqn, keys in rank.items()}
        return state_keys_by_fqn

    def _param_to_group_meta(self) -> dict[str, Any]:
        """Map each locally-owned parameter's FQN to its param-group hyperparameters.

        Only the parameters this rank's optimizer owns appear, which is exactly what both
        ends need: on save DCP unions the (non-tensor) entries written by all ranks, and on
        load each rank reads back the entries of the parameters it owns. The base optimizer
        (TE FusedAdam) tracks ``step`` per group rather than per parameter, so ``step``
        round-trips here rather than in the per-parameter state.
        """
        return {
            self._param_fqn(param): {key: value for key, value in group.items() if key != "params"}
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
            fqns = [self._param_fqn(param) for param in group["params"]]
            missing = [fqn for fqn in fqns if fqn not in param_to_group_meta]
            if missing:
                raise ValueError(
                    f"Parameters {missing} are missing from the checkpoint's "
                    "param_to_group_meta; the checkpoint's optimizer param groups do not "
                    "match this model."
                )
            # Every parameter of a group carries that group's hyperparameters, so read them
            # from the first one. A group left empty by the empty-shard filter has none to
            # read and contributes no state either.
            hyperparameters = param_to_group_meta[fqns[0]] if fqns else {}
            param_groups.append({"params": fqns, **hyperparameters})
        return param_groups

    def load_state_dict(self, state_dict: ShardedStateDict) -> None:
        """Load optimizer state produced by :meth:`sharded_state_dict`.

        By the time this runs DCP has already written the checkpoint's tensors into the
        resting optimizer-state DTensors in place, since ``sharded_state_dict(is_loading=True)``
        exposed them as the load destinations. What is left is to restore the param-group
        hyperparameters (including the group-level ``step``) and re-bind the FQN-keyed state
        to the current parameters, mirroring the Megatron-FSDP branch of
        :meth:`DistributedOptimizer.load_state_dict`.
        """
        param_groups = self._param_groups_from_group_meta(state_dict["param_to_group_meta"])
        owned_fqns = {fqn for group in param_groups for fqn in group["params"]}
        self.optimizer.load_state_dict(
            {
                # Drop the empty-local placeholders synthesized for save-time rank
                # consistency; keep only what this rank's optimizer owns.
                "state": {
                    fqn: param_state
                    for fqn, param_state in state_dict["state"].items()
                    if fqn in owned_fqns
                },
                "param_groups": param_groups,
            }
        )

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

        Rank consistency is the load-bearing invariant here. The empty-shard filter (see
        :meth:`_trainable_parameters`) leaves each rank with a different set of parameters,
        while ``preprocess_state_dict_for_uneven_dtensor`` runs one ``all_gather_object``
        *per DTensor* in sorted key order, so a rank-dependent DTensor keyspace deadlocks.
        Every trainable parameter is therefore emitted: the parameters this rank's optimizer
        holds contribute their real state, and the rest contribute empty-local placeholders.

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
        if is_loading:
            init_optimizer_state(self.optimizer)

        state_keys_by_fqn = self._gather_state_keys_by_fqn()
        state_by_fqn = {
            self._param_fqn(param): param_state
            for param, param_state in self.optimizer.state.items()
        }

        packed_state: dict[str, Any] = {}
        for param in self._trainable_parameters():
            fqn = self._param_fqn(param)
            if fqn in state_by_fqn:
                packed_state[fqn] = state_by_fqn[fqn]
            else:
                # Filtered out of this rank's optimizer because its local shard is empty. The
                # rank owning a non-empty shard saves the real state; this placeholder has the
                # same global shape and dtype but no local rows, so it contributes no data and
                # only keeps the DTensor keyspace identical on every rank.
                packed_state[fqn] = {
                    key: torch.zeros_like(param) for key in state_keys_by_fqn.get(fqn, ())
                }

        return {"state": packed_state, "param_to_group_meta": self._param_to_group_meta()}

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
        """Refresh MFSDP V2 compute weights after updating optimizer weights."""
        sync_model_weights_from_main_weights(self.get_parameters())

    def _copy_model_params_to_main_params(self, state_dict=None) -> None:
        """No-op: model loads already write into MFSDP v2's main weights."""
