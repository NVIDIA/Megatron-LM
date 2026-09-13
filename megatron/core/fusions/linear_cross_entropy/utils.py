# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import typing
from enum import Enum


class EntropyReductionEnum(Enum):
    """
    Enum for the reduction method of cross entropy.
    """

    kNone = 0
    kSum = 1
    kMean = 2


def str_to_reduction_enum(reduction: typing.Literal["none", "sum", "mean"]) -> EntropyReductionEnum:
    """
    str -> EntropyReductionEnum
    """
    _enum = EntropyReductionEnum.kNone
    if reduction == "none":
        _enum = EntropyReductionEnum.kNone
    elif reduction == "sum":
        _enum = EntropyReductionEnum.kSum
    elif reduction == "mean":
        _enum = EntropyReductionEnum.kMean
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _enum


class BackwardMethodEnum(Enum):
    """
    Enum for the backward method of linear cross entropy.
    """

    # two separate kernels for d_hidden and d_weight, respectively
    kTwoKernels = 0
    # calculate partial d_logits along its N dimension
    kDlogitsSplitN = 1
    # fuse d_hidden and d_weight into a single kernel
    kFused = 2
