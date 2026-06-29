# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import io
import pickle
from argparse import Namespace
from io import BytesIO
from pathlib import PosixPath
from signal import Signals
from types import SimpleNamespace

import torch
from numpy import dtype, ndarray
from numpy.core.multiarray import _reconstruct
from numpy.dtypes import UInt32DType

from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig
from megatron.core.rerun_state_machine import RerunDiagnostic, RerunMode, RerunState
from megatron.core.transformer.enums import (
    AttnBackend,
    CudaGraphModule,
    CudaGraphScope,
    InferenceCudaGraphScope,
)

SAFE_GLOBALS = [
    SimpleNamespace,
    PosixPath,
    _reconstruct,
    ndarray,
    dtype,
    UInt32DType,
    Namespace,
    AttnBackend,
    CudaGraphModule,
    CudaGraphScope,
    InferenceCudaGraphScope,
    ModelType,
    OptimizerConfig,
    RerunDiagnostic,
    RerunMode,
    RerunState,
    BytesIO,
    Signals,
    torch._C.Generator,  # Needed for torch ckpt format loading after weights_only default change
]


def register_safe_globals():
    """Register megatron-core safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])


def safe_load_from_bytes(b):
    """Safe version (weights_only=True) of `torch.storage._load_from_bytes`."""
    return torch.load(io.BytesIO(b), weights_only=True)


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for FP8 extra-state checkpoints.
    Only allows the narrow set of types that ``_encode_extra_state`` can
    produce: plain Python containers, numeric scalars, and the PyTorch
    tensor/storage primitives used by `pickle.dumps(tensor)`.  Any attempt
    to instantiate a class outside this allowlist raises
    `pickle.UnpicklingError`, preventing arbitrary code execution via a
    crafted checkpoint.
    """

    _SAFE_CLASSES: frozenset = frozenset(
        {
            ("builtins", "dict"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "bytes"),
            ("builtins", "str"),
            ("collections", "OrderedDict"),
            ("torch", "Size"),
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._tensor", "_rebuild_from_type_v2"),
            ("torch.storage", "UntypedStorage"),
            ("torch.storage", "_load_from_bytes"),
            ("transformer_engine.common.recipe", "DelayedScaling"),
            ("transformer_engine.common.recipe", "Float8CurrentScaling"),
            ("transformer_engine.common.recipe", "Float8BlockScaling"),
            ("transformer_engine.common.recipe", "MXFP8BlockScaling"),
            ("transformer_engine.common.recipe", "NVFP4BlockScaling"),
            ("transformer_engine.common.recipe", "Format"),
            ("transformer_engine.common.recipe", "_FormatHelper"),
            ("transformer_engine.common.recipe", "MMParams"),
            ("transformer_engine.common.recipe", "QParams"),
            ("megatron.core.extensions.transformer_engine", "TEDelayedScaling"),
            ("megatron.core.safe_globals", "safe_load_from_bytes"),
        }
    )

    def find_class(self, module: str, name: str):
        if (module, name) not in self._SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}' "
                "in FP8 extra-state checkpoint."
            )
        return super().find_class(module, name)
