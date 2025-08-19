from .eig import *

from contextlib import contextmanager
from typing import Generator
import torch

__all__ = [
    "fp32_matmul_precision",
]


@contextmanager
def fp32_matmul_precision(precision: str = "highest") -> Generator[None, None, None]:
    """Context manager for setting the precision of matmuls.

    Args:
        precision: Precision of matmuls (defaults to "highest")
    """
    prev_val = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_val)
