import typing
from dataclasses import dataclass

@dataclass
class EntropyReductionEnum:
    """
    Enum for the reduction method of cross entropy.
    """
    kNone = 0
    kSum = 1
    kMean = 2

def str_to_reduction_enum(reduction: str) -> EntropyReductionEnum:
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

@dataclass
class BackwardMethodEnum:
    # two separate kernels for d_hidden and d_weight, respectively
    kTwoKernels = 0
    # calculate partial d_logits along its N dimension
    kDlogitsSplitN = 1
    # fuse d_hidden and d_weight into a single kernel
    kFused = 2
