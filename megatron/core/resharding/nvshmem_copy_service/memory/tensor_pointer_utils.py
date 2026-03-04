# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Utilities for extracting data pointers from different tensor types.

Supports PyTorch tensors, CuPy arrays, and raw integer pointers.
"""

from typing import Any

import torch


class TensorPointerExtractor:
    """Extract memory pointers from various tensor types."""

    @staticmethod
    def get_pointer(tensor: Any) -> int:
        """
        Extract the data pointer from a tensor.

        Args:
            tensor: Can be torch.Tensor, CuPy array, or raw int pointer

        Returns:
            int: Memory address of the tensor data

        Examples:

            >>> import torch

            >>> t = torch.zeros(100, device='cuda')

            >>> ptr = TensorPointerExtractor.get_pointer(t)

            >>> isinstance(ptr, int)

            True
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.data_ptr()
        elif hasattr(tensor, "data"):  # CuPy array
            return tensor.data.ptr
        else:  # Assume raw integer pointer
            return tensor
