# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
from functools import partial
from typing import Callable, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


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

        orthogonalize_fn = partial(
            newton_schulz_tp,
            steps=num_ns_steps,
            coefficient_type=coefficient_type,
            mode="duplicated" if mode == "blockwise" else mode,
        )
        scale_factor_fn = partial(
            get_muon_scale_factor, mode=scale_mode, extra_scale_factor=extra_scale_factor
        )

        def orthogonalize_fn_tp(
            x: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            return orthogonalize_fn(x, tp_group=tp_group, partition_dim=partition_dim)

        def scale_factor_fn_tp(
            size_out: int, size_in: int, partition_dim: int | None = None
        ) -> float:
            if partition_dim is None:
                return scale_factor_fn(size_out, size_in)

            size = [size_out, size_in]
            size[partition_dim] *= get_pg_size(pg_collection.tp) if pg_collection else 1
            return scale_factor_fn(*size)

        self.pg_collection = pg_collection
        self.mode = mode

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov,
            weight_decay,
            use_decoupled_weight_decay,
            split_qkv,
            is_qkv_fn,
            qkv_split_shapes,
            fp32_matmul_prec,
            orthogonalize_fn_tp,
            scale_factor_fn_tp,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
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
            # llm-shower use different default value for partition_dim than TE.
            # Because -1 is a valid index for ndarray, we decided to not overload it.
            partition_dim = None
        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            qkv_grads = torch.split(grad, self.qkv_split_shapes, dim=0)

            # Apply Newton-Schulz to each component
            qkv_whitened = [
                self.orthogonalize_fn(g, tp_group=tp_group, partition_dim=partition_dim)
                for g in qkv_grads
            ]
            qkv_scales = [
                self.scale_factor_fn(g.size(0), g.size(1), partition_dim) for g in qkv_grads
            ]

            # Apply individual scales to each component and concatenate
            grad = torch.cat(
                [whitened * scale for whitened, scale in zip(qkv_whitened, qkv_scales)]
            )
        else:
            grad = self.orthogonalize_fn(
                grad, tp_group=tp_group, partition_dim=partition_dim
            ) * self.scale_factor_fn(grad.size(0), grad.size(1), partition_dim)
        return grad


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."

    # dist-optim is not supported due to strong coupling with how DDP init grad buffer
    # in thoery we can put some weight to use non-dist-muon and rest to dist-adam
    # but there are strong dependency and assumption in DDP that prevent it
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): might not be sufficient for future algorithm. revisit this conditioning
            if not getattr(param, 'is_embedding_or_output_parameter', False) and not (
                len(param.shape) == 1
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    # TODO(deyuf): support qkv split
    optimizer = TensorParallelMuon(
        linear_param_groups,
        lr=config.lr,
        momentum_beta=config.muon_momentum,
        use_nesterov=config.muon_use_nesterov,
        weight_decay=config.weight_decay,
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        num_ns_steps=config.muon_num_ns_steps,
        scale_mode=config.muon_scale_mode,
        split_qkv=False,
        qkv_split_shapes=None,
        extra_scale_factor=config.muon_extra_scale_factor,
        pg_collection=pg_collection,
        mode=config.muon_tp_mode,
    )

    # set config here to:
    # 1. get adam for rest of layer
    # 2. avoid ChainedOptimizer check fail that assert all optimizers are same kind
    # side effect is muon optimizer will have wrong name str, i.e. config.optimizer == 'adam'
    # TODO(deyuf): allow user to select optimizer mix and relax ChainedOptimizer design
    config.optimizer = 'adam'

    # need to wrap into megatron mix precision optimizer. (only support bf16 w/o loss scale now)
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')
    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight  so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(optimizer, config, None, None)
    else:
        optimizer = FP32Optimizer(optimizer, config, None)

    optimizers.append(optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(optimizers, config, pg_collection)
    return ChainedOptimizer(optimizers)
