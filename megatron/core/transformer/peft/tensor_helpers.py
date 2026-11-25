# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

"""Initialization and sequence padding for low-rank projections."""

from typing import Callable, Tuple

import torch
import torch.nn as nn


def init_method_normal(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on normal distribution N(0, sigma).

    Args:
        sigma: Standard deviation for the normal distribution.

    Returns:
        Initialization function that applies normal distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on Kaiming uniform distribution.

    Args:
        val: The 'a' parameter for Kaiming uniform initialization.

    Returns:
        Initialization function that applies Kaiming uniform distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_uniform(bound: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on uniform distribution U(-bound, bound).

    Args:
        bound: Half-width of the uniform distribution.

    Returns:
        Initialization function that applies the uniform distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.uniform_(tensor, -bound, bound)

    return init_


def init_method_const(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method that sets all values to a constant.

    Args:
        val: Constant value to initialize the tensor with.

    Returns:
        Initialization function that sets tensor to constant value.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.constant_(tensor, val)

    return init_


def pad_seq_to_mult(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, int]:
    """Pad sequence length to be a multiple of mult.

    This function pads the first dimension of the tensor to ensure it's divisible by mult.
    Used primarily for MoE (Mixture of Experts) operations that require specific sequence lengths.

    Args:
        x: Input tensor to pad.
        mult: Multiple that the sequence length should be divisible by.

    Returns:
        A tuple containing:
            - Padded tensor
            - Number of padding elements added
    """
    if x.shape[0] % mult == 0:
        return x, 0
    pad_len = mult - (x.shape[0] % mult)
    x = nn.functional.pad(x, (0, 0, 0, pad_len))
    return x, pad_len


def unpad_seq_to_mult(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Remove sequence padding that was added by pad_seq_to_mult.

    Args:
        x: Padded tensor to unpad.
        pad_len: Number of padding elements to remove from the end.

    Returns:
        Unpadded tensor with pad_len elements removed from the first dimension.
    """
    if pad_len <= 0:
        return x
    return x[:-pad_len, :]
