# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer.


    Precision
    ---------

    fp16 (bool): If true, train with fp16 mixed precision training. Defaults to False.

    bf16 (bool): If true, train with bf16 mixed precision training. Defaults to False.

    params_dtype (torch.dtype): dtype used when intializing the weights. Defaults to torch.float32.


    General Optimizer
    -----------------

    optimizer (str): Optimizer to use (one of Adam or SGD).

    lr (float, optional): Initial learning rate. Depending on decay style and initial warmup, the learning
                          rate at each iteration would be different.


    Loss Scaler
    -----------

    loss_scale (float, optional): Static loss scaling, positive power of 2 values can improve fp16 convergence.
                                  If None, dynamic loss scaling is used.

    initial_loss_scale (float): Initial loss-scale for dynamic loss scaling.

    min_loss_scale (float): Minimum loss scale for dynamic loss scaling.

    loss_scale_window (float): Window over which to raise/lower dynamic scale.

    hysteresis (int): Hysteresis for dynamic loss scaling.


    Weight Decay
    ------------

    weight_decay (float): Weight decay coefficient for L2 regularization.


    Base Optimizer
    --------------

    adam_beta1 (float): First coefficient for computing running averages of gradient and its square in Adam optimizer.

    adam_beta2 (float): Second coefficient for computing running averages of gradient and its square in Adam optimizer.

    adam_eps (float): Term added to the denominator to improve numerical stability in Adam optimizer.

    sgd_momentum (float): Momentum factor for SGD optimizer.


    Distributed Optimizer
    ---------------------

    use_distributed_optimizer (bool): Distribute optimizer state over data-parallel replicas.

    overlap_param_gather (bool): If true, overlap param all-gather with forward compute in distributed optimizer.


    Miscellaneous
    -------------

    clip_grad (float): Gradient clipping based on global L2 norm.

    log_num_zeros_in_grad (bool): If true, calculate and log the number of zeros in gradient.
    """

    # Precision.
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32

    optimizer: str = 'adam'
    lr: Optional[float] = None

    # Loss scaling.
    loss_scale: Optional[float] = None
    initial_loss_scale: float = 2 ** 32
    min_loss_scale: float = 1.0
    loss_scale_window: float = 1000
    hysteresis: int = 2

    weight_decay: float = 0.01

    # Adam.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-08
    # SGD.
    sgd_momentum: float = 0.9

    # Distributed optimizer.
    use_distributed_optimizer: bool = False
    overlap_param_gather: bool = False

    # Miscellaneous.
    clip_grad: float = 1.0
    log_num_zeros_in_grad: bool = False
