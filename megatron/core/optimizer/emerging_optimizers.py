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
from typing import Any, Callable, Dict, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

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


def _get_qkv_split_shapes(model_cfg, split_qkv_per_head: bool = False) -> list[int]:
    """Compute fused QKV split shapes from logical attention layout metadata.

    Args:
        model_cfg: Transformer config or an owning attention layer's ``QKVLayout`` metadata.
        split_qkv_per_head: Return one split size per physical attention head. When false,
            return the per-query-group Q, gate (if present), K, and V projection widths.
    """
    if hasattr(model_cfg, 'projection_split_shapes'):
        if split_qkv_per_head:
            return list(model_cfg.per_head_split_shapes) * model_cfg.num_groups
        return list(model_cfg.projection_split_shapes)

    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if split_qkv_per_head:
        num_query_heads_per_group = model_cfg.num_attention_heads // model_cfg.num_query_groups
        per_group_shapes = [model_cfg.kv_channels] * num_query_heads_per_group
        if getattr(model_cfg, 'attention_output_gate', False):
            per_group_shapes += [model_cfg.kv_channels] * num_query_heads_per_group
        per_group_shapes += [model_cfg.kv_channels, model_cfg.kv_channels]
        return per_group_shapes * model_cfg.num_query_groups
    if getattr(model_cfg, 'attention_output_gate', False):
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


def _localize_qkv_split_shapes(
    global_split_shapes: list[int], local_start: int, local_rows: int
) -> tuple[list[int], bool]:
    """Intersect global per-head split sizes with a rank-local contiguous row range.

    Returns:
        The physical rank-local split sizes and whether every intersected head is complete.
    """
    local_stop = local_start + local_rows
    local_split_shapes = []
    all_heads_complete = True
    head_start = 0
    for head_rows in global_split_shapes:
        head_stop = head_start + head_rows
        overlap_start = max(head_start, local_start)
        overlap_stop = min(head_stop, local_stop)
        if overlap_start < overlap_stop:
            overlap_rows = overlap_stop - overlap_start
            local_split_shapes.append(overlap_rows)
            all_heads_complete &= overlap_rows == head_rows
        head_start = head_stop

    if sum(local_split_shapes) != local_rows:
        raise RuntimeError(
            f"Muon per-head QKV local range [{local_start}, {local_stop}) is outside "
            f"the global split shape with {sum(global_split_shapes)} rows"
        )
    return local_split_shapes, all_heads_complete


def _qkv_split_groups_are_complete(
    split_shapes: list[int], local_start: int, local_rows: int
) -> bool:
    """Return whether a local row range contains only complete fused QKV groups."""
    split_width = sum(split_shapes)
    if split_width <= 0:
        raise ValueError(f"Muon QKV split shapes must sum to a positive size: {split_shapes}")
    return local_start % split_width == 0 and local_rows % split_width == 0


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
        split_qkv_per_head: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
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
        if split_qkv_per_head and not split_qkv:
            raise ValueError("split_qkv_per_head requires split_qkv=True")

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
        self.split_qkv_per_head = split_qkv_per_head
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

    def _get_gtp_remat_group(self, p):
        """Return the GTP-remat process group for a parameter, if configured."""
        is_expert = getattr(p, 'expert_tp', False)
        return (
            (self.pg_collection.expt_gtp_remat if is_expert else self.pg_collection.gtp_remat)
            if self.pg_collection
            else None
        )

    def _gather_qkv_grad(self, p, grad, tp_group, expected_rows, gather_gtp=True):
        """Reconstruct a fused QKV gradient and record how to restore its local shard."""
        gathered_grad = grad
        gtp_slice = None
        gtp_remat_group = self._get_gtp_remat_group(p)
        gtp_pad_length = int(getattr(p, "qkv_gtp_pad_length", 0))
        if gtp_pad_length < 0:
            raise RuntimeError(f"Muon QKV GTP padding must be non-negative: {gtp_pad_length}")
        if (
            gather_gtp
            and gtp_remat_group is not None
            and get_pg_size(gtp_remat_group) > 1
            and getattr(p, 'is_gtp_weight_remat', False)
        ):
            gtp_size = get_pg_size(gtp_remat_group)
            gtp_rank = get_pg_rank(gtp_remat_group)
            gtp_local_rows = gathered_grad.shape[0]
            shards = [torch.empty_like(gathered_grad) for _ in range(gtp_size)]
            torch.distributed.all_gather(shards, gathered_grad, gtp_remat_group)
            gathered_grad = torch.cat(shards, dim=0)
            if gtp_pad_length >= gathered_grad.shape[0]:
                raise RuntimeError(
                    "Invalid Muon QKV GTP padding after gathering: "
                    f"pad_length={gtp_pad_length}, gathered_rows={gathered_grad.shape[0]}"
                )
            if gtp_pad_length > 0:
                gathered_grad = gathered_grad[:-gtp_pad_length]
            gtp_slice = (gtp_rank, gtp_local_rows, gtp_pad_length)
        elif gtp_pad_length > 0:
            raise RuntimeError(
                "Muon QKV has GTP padding but its GTP-remat shards were not gathered"
            )

        tp_slice = None
        if gathered_grad.shape[0] != expected_rows:
            partition_dim = getattr(p, "partition_dim", None)
            if partition_dim != 0 or tp_group is None:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(gathered_grad.shape)}, "
                    f"expected_rows={expected_rows}, partition_dim={partition_dim}"
                )
            tp_size = get_pg_size(tp_group)
            if gathered_grad.shape[0] * tp_size != expected_rows:
                raise RuntimeError(
                    "Muon QKV split cannot reconstruct the global tensor: "
                    f"local_grad_shape={tuple(gathered_grad.shape)}, tp_size={tp_size}, "
                    f"expected_rows={expected_rows}"
                )
            tp_rank = get_pg_rank(tp_group)
            tp_local_rows = gathered_grad.shape[0]
            shards = [torch.empty_like(gathered_grad) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, gathered_grad, tp_group)
            gathered_grad = torch.cat(shards, dim=0)
            tp_slice = (tp_rank, tp_local_rows)

        if gathered_grad.shape[0] != expected_rows:
            raise RuntimeError(
                "Muon QKV split shape mismatch after gathering: "
                f"grad_shape={tuple(gathered_grad.shape)}, expected_rows={expected_rows}"
            )
        return gathered_grad, tp_slice, gtp_slice

    @staticmethod
    def _restore_local_qkv_grad(gathered_grad, tp_slice, gtp_slice):
        """Restore the TP and GTP-remat shards recorded by ``_gather_qkv_grad``."""
        if tp_slice is not None:
            tp_rank, tp_local_rows = tp_slice
            gathered_grad = gathered_grad[tp_rank * tp_local_rows : (tp_rank + 1) * tp_local_rows]
        if gtp_slice is not None:
            gtp_rank, gtp_local_rows, gtp_pad_length = gtp_slice
            if gtp_pad_length > 0:
                gathered_grad = torch.nn.functional.pad(gathered_grad, (0, 0, 0, gtp_pad_length))
            gathered_grad = gathered_grad[
                gtp_rank * gtp_local_rows : (gtp_rank + 1) * gtp_local_rows
            ]
        return gathered_grad.contiguous()

    def _orthogonalize_split_qkv(self, grad, split_shapes, orthogonalize_fn):
        """Split and reconstruct Megatron's interleaved fused QKV update."""
        if grad.ndim != 2:
            raise RuntimeError(f"Muon QKV gradient must be 2D, got {grad.ndim}D")
        if not split_shapes or any(size <= 0 for size in split_shapes):
            raise RuntimeError(f"Muon QKV split shapes must be positive: {split_shapes}")
        split_width = sum(split_shapes)
        if self.split_qkv_per_head:
            if grad.shape[0] != split_width:
                raise RuntimeError(
                    f"Muon per-head QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                    f"split_shapes={split_shapes}"
                )
            if len(set(split_shapes)) == 1:
                # A 3D input selects Emerging-Optimizers' batched Newton-Schulz path.
                head_rows = split_shapes[0]
                return orthogonalize_fn(grad.view(len(split_shapes), head_rows, -1)).view_as(grad)
            return torch.cat(
                [orthogonalize_fn(head) for head in torch.split(grad, split_shapes, dim=0)], dim=0
            )

        if grad.shape[0] % split_width != 0:
            raise RuntimeError(
                f"Muon QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                f"split_shapes={split_shapes}"
            )
        num_query_groups = grad.shape[0] // split_width
        grouped_grad = grad.view(num_query_groups, split_width, -1)
        projection_grads = torch.split(grouped_grad, split_shapes, dim=1)
        projection_grads = [
            projection.reshape(-1, grad.shape[-1]) for projection in projection_grads
        ]
        projection_grads = [
            orthogonalize_fn(projection).view(num_query_groups, -1, grad.shape[-1])
            for projection in projection_grads
        ]
        return torch.cat(projection_grads, dim=1).view_as(grad)

    def scaled_orthogonalize_fn_with_gtp_remat(self, p, grad, tp_group, partition_dim):
        """All-gather grad along GTP_remat/EGTP_remat dim 0, orthogonalize, then slice back.

        GTP_remat shards weights along dim 0 independently of TP's partition_dim. Newton-Schulz
        needs the full weight matrix, so we reconstruct the GTP_remat dimension before running
        the TP-aware orthogonalization, then extract the local GTP_remat shard from the result.
        When GTP_remat is inactive this is a plain passthrough to scaled_orthogonalize_fn.
        """
        # TODO: Clean up code that determines if parameter is a MoE layer and which TP group to use
        gtp_remat_group = self._get_gtp_remat_group(p)

        if gtp_remat_group is None or get_pg_size(gtp_remat_group) <= 1:
            return self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)

        # Parameters with is_gtp_weight_remat=False are not sharded along the
        # GTP process group, and do not require all-gathering prior to
        # orthogonalization.
        if not getattr(p, 'is_gtp_weight_remat', False):
            return self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)

        gtp_remat_size = get_pg_size(gtp_remat_group)
        gtp_rank = get_pg_rank(gtp_remat_group)
        shards = [torch.empty_like(grad) for _ in range(gtp_remat_size)]
        torch.distributed.all_gather(shards, grad, gtp_remat_group)
        gathered_grad = torch.cat(shards, dim=0)

        gathered_grad = self.scaled_orthogonalize_fn(gathered_grad, tp_group, partition_dim)

        shard_size = gathered_grad.shape[0] // gtp_remat_size
        return gathered_grad[gtp_rank * shard_size : (gtp_rank + 1) * shard_size].contiguous()

    def _orthogonalize_qkv_per_head(self, p, grad, tp_group):
        """Orthogonalize every Q, gate, K, and V head independently.

        Split sizes may describe complete heads in the local tensor or the global fused
        QKV tensor. For a global layout, reconstruct GTP-remat and TP dimension 0 before
        splitting so heads crossing rank boundaries remain complete.
        """
        local_split_shapes = getattr(p, "qkv_split_shapes", None)
        heads_are_complete = getattr(p, "qkv_split_heads_are_complete", None)
        has_gtp_padding = int(getattr(p, "qkv_gtp_pad_length", 0)) > 0
        use_local_layout = not has_gtp_padding and (
            heads_are_complete is True
            or (
                heads_are_complete is None
                and local_split_shapes is not None
                and sum(local_split_shapes) == grad.shape[0]
            )
        )
        if use_local_layout:
            qkv_split_shapes = local_split_shapes
        else:
            qkv_split_shapes = getattr(p, "qkv_split_shapes_global", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
        if qkv_split_shapes is None:
            raise RuntimeError("Muon per-head QKV split requested but qkv_split_shapes is not set")
        if not qkv_split_shapes or any(size <= 0 for size in qkv_split_shapes):
            raise RuntimeError(
                f"Muon per-head QKV split shapes must be positive: {qkv_split_shapes}"
            )

        expected_rows = sum(qkv_split_shapes)
        gathered_grad, tp_slice, gtp_slice = self._gather_qkv_grad(
            p, grad, tp_group, expected_rows, gather_gtp=not use_local_layout
        )

        gathered_grad = self._orthogonalize_split_qkv(
            gathered_grad,
            qkv_split_shapes,
            lambda head_grad: self.scaled_orthogonalize_fn(
                head_grad, tp_group=None, partition_dim=None
            ),
        )

        return self._restore_local_qkv_grad(gathered_grad, tp_slice, gtp_slice)

    def _orthogonalize_fragmented_qkv(self, p, grad, tp_group, split_shapes):
        """Orthogonalize projections after reconstructing fragmented query-group blocks."""
        global_split_shapes = getattr(p, "qkv_split_shapes_global", None)
        if global_split_shapes is None:
            raise RuntimeError("Muon fragmented QKV split requires global split shapes")
        expected_rows = sum(global_split_shapes)
        if expected_rows % sum(split_shapes) != 0:
            raise RuntimeError(
                f"Muon global QKV layout does not contain complete query groups: "
                f"global_split_shapes={global_split_shapes}, split_shapes={split_shapes}"
            )

        gathered_grad, tp_slice, gtp_slice = self._gather_qkv_grad(p, grad, tp_group, expected_rows)
        gathered_grad = self._orthogonalize_split_qkv(
            gathered_grad,
            split_shapes,
            lambda projection_grad: self.scaled_orthogonalize_fn(
                projection_grad, tp_group=None, partition_dim=None
            ),
        )
        return self._restore_local_qkv_grad(gathered_grad, tp_slice, gtp_slice)

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
            if self.split_qkv_per_head:
                return self._orthogonalize_qkv_per_head(p, grad, tp_group)

            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if qkv_split_shapes is None:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
            if (
                getattr(p, "qkv_split_groups_are_complete", None) is False
                or int(getattr(p, "qkv_gtp_pad_length", 0)) > 0
            ) and getattr(p, "qkv_split_shapes_global", None) is not None:
                return self._orthogonalize_fragmented_qkv(p, grad, tp_group, qkv_split_shapes)
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad.shape}, split shapes {qkv_split_shapes}',
            )
            grad = self._orthogonalize_split_qkv(
                grad,
                qkv_split_shapes,
                lambda projection_grad: self.scaled_orthogonalize_fn_with_gtp_remat(
                    p, projection_grad, tp_group, partition_dim
                ),
            )
        else:
            grad = self.scaled_orthogonalize_fn_with_gtp_remat(p, grad, tp_group, partition_dim)
        return grad


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
        split_qkv_per_head: Whether to orthogonalize individual Q, gate, K, and V heads.
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
        split_qkv_per_head: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
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
            split_qkv_per_head=split_qkv_per_head,
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
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(
        model_chunks[0].config, split_qkv_per_head=kwargs.get("split_qkv_per_head", False)
    )
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
