# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    ##############
    # General
    ##############
    optimizer: str = 'adam'
    """Optimizer to use (one of Adam or SGD)."""

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    decoupled_lr: Optional[float] = None
    """Separate learning rate for the input and output layer."""

    decoupled_min_lr: Optional[float] = None
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Precision
    ##############
    fp16: bool = False
    """If true, train with fp16 mixed precision training. Defaults to False."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training. Defaults to False."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights. Defaults to torch.float32."""

    use_precision_aware_optimizer: bool = False
    """If true, allows optimizer-related tensors (master_param, gradients and optimizer states)
    to be set to lower precision. Defaults to False.
    """

    main_grads_dtype: torch.dtype = torch.float32
    """dtype of main grads when enabling precision-aware-optimizer"""

    main_params_dtype: torch.dtype = torch.float32
    """dtype of main params when enabling precision-aware-optimizer"""

    exp_avg_dtype: torch.dtype = torch.float32
    """dtype of exp_avg when enabling precision-aware-optimizer"""

    exp_avg_sq_dtype: torch.dtype = torch.float32
    """dtype of exp_avg_sq when enabling precision-aware-optimizer"""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2**32
    """Initial loss-scale for dynamic loss scaling."""

    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    ##############
    # Optimizer
    ##############
    # Adam
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    # SGD.
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_param_gather_with_optimizer_step: bool = False
    """If true, overlap param all-gather of first bucket with optimizer step."""

    #######################
    # Optimizer Offload
    #######################

    optimizer_cpu_offload: bool = False
    """If True, offload optimizer states tensor and compute to CPU."""

    optimizer_offload_fraction: float = 0.0
    """Specifies the fraction of optimizer states to offload from GPU memory to CPU."""

    use_torch_optimizer_for_cpu_offload: bool = False
    """If True, use torch.optim.Optimizer for CPU offload."""

    overlap_cpu_optimizer_d2h_h2d: bool = False
    """
    When set to `True`, this flag enables overlapping of the CPU optimizer
    update process with the data transfer operations. This can help improve
    overall training efficiency by reducing idle time during data movement,
    allowing the optimizer to perform updates while gradients and parameters
    are being transferred between devices.
    """

    pin_cpu_grads: bool = True
    """If True, pin the optimizer gradients to CPU memory."""

    pin_cpu_params: bool = True
    """If True, pin the optimizer parameters to CPU memory."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    barrier_with_L1_time: bool = False
    """If true, use barrier with level 1 time measurements."""

    timers: Callable = None
    """Function to get timers."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    def __post_init__(self):
        """Check the validity of the config."""
        if self.use_precision_aware_optimizer:
            assert (
                self.optimizer == 'adam'
            ), '--use-precision-aware-optimizer only supported with adam'
            assert (
                self.use_distributed_optimizer
            ), '--use-precision-aware-optimizer only supported with distributed optimizer'

            # Only the FusedAdam in TE and HybridDeviceOptimizer supports
            # --use-precision-aware-optimizer.
            # TODO: Remove this check when apex's FusedAdam is no longer used.
            if self.optimizer_cpu_offload:
                return
            try:
                import inspect

                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                adam_args = inspect.signature(Adam).parameters
                arg_names = [
                    'master_weight_dtype',
                    'exp_avg_dtype',
                    'exp_avg_sq_dtype',
                    'use_decoupled_grad',
                ]
                for name in arg_names:
                    assert name in adam_args, (
                        "Current FusedAdam of TE doesn't support --use-precision-aware-optimizer, "
                        "please update TE version."
                    )
            except ImportError:
                raise RuntimeError(
                    '--use-precision-aware-optimizer requires FusedAdam from TransformerEngine, '
                    'but not found.'
                )
        else:
            assert (
                self.main_grads_dtype == torch.float32
            ), "main_grads_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.main_params_dtype == torch.float32
            ), "main_params_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_dtype == torch.float32
            ), "exp_avg_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_sq_dtype == torch.float32
            ), "exp_avg_sq_dtype can only be fp32 when not using precision-aware optimizer"
