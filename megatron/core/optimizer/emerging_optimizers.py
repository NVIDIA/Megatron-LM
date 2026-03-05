# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False


logger = logging.getLogger(__name__)


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


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
    init_state_fn: Callable
    config_to_kwargs: Callable
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(default_factory=dict)


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
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

if HAVE_EMERGING_OPTIMIZERS:

    class TensorParallelMuon(OrthogonalizedOptimizer):
        """Tensor Parallel Muon optimizer."""

        def __init__(
            self,
            params: ParamsT,
            lr: float = 3e-4,
            momentum_beta: float = 0.95,
            use_nesterov: bool = True,
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
            mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
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
                    mode="duplicated" if mode == "blockwise" else mode,
                )
                scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
                return orth_grad * scale_factor * extra_scale_factor

            self.pg_collection = pg_collection
            self.mode = mode
            self.split_qkv = split_qkv
            self.is_qkv_fn = is_qkv_fn
            self.qkv_split_shapes = qkv_split_shapes

            weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
            super().__init__(
                params,
                lr,
                momentum_beta,
                use_nesterov=use_nesterov,
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
            partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
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

    def _muon_init_state_fn(opt, config=None):
        """Initialize Muon optimizer state for torch_dist checkpoint format."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
        """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
        return {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "momentum_beta": config.muon_momentum,
            "use_nesterov": config.muon_use_nesterov,
            "fp32_matmul_prec": config.muon_fp32_matmul_prec,
            "num_ns_steps": config.muon_num_ns_steps,
            "scale_mode": config.muon_scale_mode,
            "extra_scale_factor": config.muon_extra_scale_factor,
            "mode": config.muon_tp_mode,
            "split_qkv": config.muon_split_qkv,
            "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
            "qkv_split_shapes": _get_qkv_split_shapes(model_chunks[0].config),
            "pg_collection": pg_collection,
        }

    # -----------------------------------------------------------------------
    # Register Muon
    # -----------------------------------------------------------------------
    _EMERGING_OPTIMIZERS['muon'] = EmergingOptimizerEntry(
        optimizer_cls=TensorParallelMuon,
        init_state_fn=_muon_init_state_fn,
        config_to_kwargs=_muon_config_to_kwargs,
        default_param_overrides={
            ParamKey(
                predicate=ParamPredicate(
                    name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                )
            ): {'optimizer': 'adam'}
        },
    )
