# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from torch.distributed.tensor import DTensor as _DTensor

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        gather_uneven_dtensor_to_full_tensor,
        update_uneven_dtensor_chunk_metadata,
    )

    _HAVE_DTENSOR = True
except ImportError:
    _DTensor = None  # type: ignore[assignment,misc]
    gather_uneven_dtensor_to_full_tensor = None  # type: ignore[assignment]
    update_uneven_dtensor_chunk_metadata = None  # type: ignore[assignment]
    _HAVE_DTENSOR = False

try:
    from emerging_optimizers import registry
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class FSDPTensorParallelMuon(TensorParallelMuon):
    """TensorParallelMuon for Megatron-FSDP ZeRO-1/2/3.

    M-FSDP shards parameters unevenly across DP ranks; params split at rank
    boundaries must be gathered before Newton-Schulz orthogonalization. Fully
    local params are orthogonalized without any collective.
    """

    def __init__(
        self,
        params: ParamsT,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs: Any,
    ) -> None:
        assert _HAVE_DTENSOR, (
            "[Megatron-FSDP] torch.distributed.tensor.DTensor "
            f"is required to use {type(self).__name__}."
        )
        self.dp_group = dp_group
        super().__init__(params, **kwargs)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Muon step for Megatron-FSDP ZeRO-1/2/3.

        Separates collective (AG) and local (NS) work into three phases so that
        no rank is blocked waiting on another rank computing NS to reach AG:
          1. Compute momentum updates locally for all params.
          2. All-gather boundary params — all collectives, no NS interleaved.
          3. Newton-Schulz + weight update locally for all params.
        """
        loss = None if closure is None else closure()

        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            for group in self.param_groups:
                self._init_group(group, skip_non_grad_params=False)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._local_muon_update(p, p.grad, group)
            return loss

        # Track all parameters to update ordered by parameter group index.
        # (param, pre_ns_grad, is_gathered, lr, group_kwargs)
        all_updates: list = []

        # Phase 1: Compute momentum updates (fully local).
        for group in self.param_groups:
            self._init_group(group, skip_non_grad_params=False)
            gather_param_indices = self._get_boundary_gather_param_indices(group)
            lr = group["lr"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            for param_idx, p in enumerate(group["params"]):
                p_local = p.to_local()
                needs_gather = param_idx in gather_param_indices
                if p_local.numel() == 0 and not needs_gather:
                    # If this parameter is not split by Megatron-FSDP,
                    # and is empty on this DP rank, then we can skip this
                    # update for all TP ranks, as tensor parallelism uses
                    # even sharding, so empty implies that FSDP did not
                    # assign any fraction of the parameter to this DP rank.
                    continue

                state = self.state[p]
                mom_local = state["momentum_buffer"].to_local()

                grad = p.grad
                local_grad = grad.to_local() if grad is not None else torch.zeros_like(mom_local)

                self._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
                mom_local.lerp_(local_grad, 1 - group["momentum"])
                if self.nesterov:
                    pre_ns_grad = local_grad.lerp(mom_local, group["momentum"])
                else:
                    pre_ns_grad = mom_local

                all_updates.append((p, pre_ns_grad, needs_gather, lr, group_kwargs))

        # Phase 2: AG all boundary gradients.
        for i, (p, pre_ns_grad, needs_gather, lr, group_kwargs) in enumerate(all_updates):
            if not needs_gather:
                continue
            # Un-shard the un-evenly sharded gradient.
            if gather_uneven_dtensor_to_full_tensor is None:
                raise RuntimeError(
                    "Megatron-FSDP `gather_uneven_dtensor_to_full_tensor` is required "
                    "to gather un-evenly sharded parameters for Muon step()."
                )
            pre_ns_grad_dtensor = self._dtensor_from_local_like(p, pre_ns_grad.contiguous())
            # Compute the global shape and offset for re-sharding the gradient.
            if not hasattr(pre_ns_grad_dtensor._local_tensor, "__create_chunk_list__"):
                update_uneven_dtensor_chunk_metadata(pre_ns_grad_dtensor)
            # Unsharded Gradient DTensor
            full_pre_ns_grad = gather_uneven_dtensor_to_full_tensor(pre_ns_grad_dtensor).to_local()
            # Mirror the uneven sharding metadata to Megatron-FSDP DTensor parameters.
            # By doing this, we can avoid unnecessary AG, as the parameters and gradients
            # are globally and locally symmetrical in shape and offset.
            self._copy_dtensor_chunk_metadata(p, pre_ns_grad_dtensor)
            all_updates[i] = (p, full_pre_ns_grad, True, lr, group_kwargs)

        # Phase 3: NS orthogonalization and weight update (fully local).
        from emerging_optimizers import utils

        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for p, pre_ns_grad, is_gathered, lr, group_kwargs in all_updates:
                if is_gathered:
                    orth_update = super(FSDPTensorParallelMuon, self).orthogonalize(
                        p, pre_ns_grad, **group_kwargs
                    )
                    sharded_update = self._reshard_full_update_like(p, orth_update)
                    self.pre_weight_update_fn_inplace(p._local_tensor, sharded_update._local_tensor)
                    p.add_(sharded_update, alpha=-lr)  # DTensors
                    self.post_weight_update_fn_inplace(p._local_tensor)
                else:
                    orth_update = (
                        super(FSDPTensorParallelMuon, self)
                        .orthogonalize(p, pre_ns_grad, **group_kwargs)
                        .to(dtype=p._local_tensor.dtype)
                    )
                    # Apply a Tensor step.
                    self.pre_weight_update_fn_inplace(p._local_tensor, orth_update)
                    p._local_tensor.add_(orth_update, alpha=-lr)
                    self.post_weight_update_fn_inplace(p._local_tensor)

        return loss

    def _needs_boundary_gather(self, dtensor: torch.Tensor) -> bool:
        assert isinstance(
            dtensor, _DTensor
        ), f"Detected non-DTensor during {type(self).__name__}: {dtensor}"
        local_tensor = dtensor.to_local()
        return local_tensor.numel() > 0 and tuple(dtensor.shape) != tuple(local_tensor.shape)

    def _get_boundary_gather_param_indices(self, group: Dict[str, Any]) -> set[int]:
        """Return globally-agreed parameter indices that need a boundary all-gather."""
        params = group["params"]
        local_boundary_indices = [
            idx for idx, param in enumerate(params) if self._needs_boundary_gather(param)
        ]

        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            return set(local_boundary_indices)

        gathered_indices: list[list[int] | None] = [None] * get_pg_size(self.dp_group)
        torch.distributed.all_gather_object(
            gathered_indices, local_boundary_indices, group=self.dp_group
        )
        return {
            idx
            for rank_indices in gathered_indices
            if rank_indices is not None
            for idx in rank_indices
        }

    def _copy_dtensor_chunk_metadata(self, dst, src) -> None:
        if hasattr(src._local_tensor, "__create_chunk_list__"):
            dst._local_tensor.__create_chunk_list__ = src._local_tensor.__create_chunk_list__
        if hasattr(src._local_tensor, "__create_write_items__"):
            dst._local_tensor.__create_write_items__ = src._local_tensor.__create_write_items__

    def _dtensor_from_local_like(self, dtensor_ref, local_tensor: torch.Tensor):
        dtensor = _DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=dtensor_ref.device_mesh,
            placements=dtensor_ref.placements,
            shape=dtensor_ref.shape,
            stride=dtensor_ref.stride(),
        )
        self._copy_dtensor_chunk_metadata(dtensor, dtensor_ref)
        return dtensor

    def _reshard_full_update_like(self, dtensor_ref, full_update: torch.Tensor):
        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            raise ValueError(
                f"{dtensor_ref} is not a Megatron-FSDP DTensor parameter "
                "with DTensor._local_tensor.__create_chunk_list__. "
                "Verify that `update_uneven_dtensor_chunk_metadata` "
                "has been called on this uneven DTensor."
            )
        shard_metadata = dtensor_ref._local_tensor.__create_chunk_list__()[0]
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(shard_metadata.offsets, shard_metadata.sizes)
        )
        local_update = full_update[slices].contiguous().to(dtype=dtensor_ref._local_tensor.dtype)
        return self._dtensor_from_local_like(dtensor_ref, local_update)

    @torch.no_grad()  # type: ignore[misc]
    def _local_muon_update(
        self, p: torch.Tensor, grad: torch.Tensor, group: Dict[str, Any]
    ) -> None:
        """Local (non-DP) Muon update – identical to OrthogonalizedOptimizer.step body."""
        from emerging_optimizers import utils

        state = self.state[p]
        self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
        if self.nesterov:
            grad = grad.lerp(state["momentum_buffer"], group["momentum"])
        else:
            grad = state["momentum_buffer"]
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            orth_grad = self.orthogonalize(p, grad, **group_kwargs)
        self.pre_weight_update_fn_inplace(p, orth_grad)
        p.add_(orth_grad, alpha=-group["lr"])
        self.post_weight_update_fn_inplace(p)


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
