# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import warnings
from typing import Callable, Any

import torch


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
STRICT_JIT_FUSION = os.environ.get('STRICT_JIT_FUSION', 'false').lower() == 'true'


def safe_jit_fuser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Safely apply JIT fusion to a function, with fallback for newer PyTorch versions.

    This function attempts to apply JIT fusion to the input function. For PyTorch
    versions 2.2 and above, it uses `torch.compile`. If compilation fails, it
    either falls back to the original function and logs a warning, or raises an
    exception if STRICT_JIT_FUSION is set to True. For older PyTorch versions,
    it uses `torch.jit.script`.

    Args:
        func (Callable[..., Any]): The function to be JIT fused.

    Returns:
        Callable[..., Any]: A wrapped function that applies JIT fusion when possible,
        or falls back to the original function if compilation fails (unless in strict mode).

    Raises:
        RuntimeError: If compilation fails and STRICT_JIT_FUSION is set to True.
    """
    if (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 2):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return torch.compile(func)(*args, **kwargs)
            except RuntimeError as e:
                if STRICT_JIT_FUSION:
                    raise RuntimeError(f"torch.compile failed in strict mode: {str(e)}")
                warnings.warn(f"torch.compile failed: {str(e)}. Falling back to original function.")
                return func(*args, **kwargs)
        return wrapper
    else:
        return torch.jit.script(func)

jit_fuser = safe_jit_fuser
