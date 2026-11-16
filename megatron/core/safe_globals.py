# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import io
import pickle
import threading
from argparse import Namespace
from io import BytesIO
from pathlib import PosixPath
from signal import Signals
from types import SimpleNamespace
from unittest.mock import patch

import numpy
import torch
import torch.distributed.checkpoint.filesystem as _dcp_filesystem
import types
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
    # Allow frozenset values in legacy checkpoint metadata when loading with weights_only=True.
    frozenset,
    Signals,
    torch._C.Generator,  # Needed for torch ckpt format loading after weights_only default change
]

SAFE_OBJECTS = {
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
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}

_pickle_patch_lock = threading.Lock()


def register_safe_globals():
    """Register megatron-core safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])


def safe_load_from_bytes(b):
    """Safe version (weights_only=True) of `torch.storage._load_from_bytes`."""
    return torch.load(io.BytesIO(b), weights_only=True)


def _safe_pickle_load(file, **kwargs):
    """Safe version of `pickle.load`."""
    return SafeUnpickler(file, **kwargs).load()


def safe_numpy_load(path, **kwargs):
    """Safe version of `numpy.load` which calls `pickle.load`."""
    with _pickle_patch_lock:
        with patch('pickle.load', _safe_pickle_load):
            return numpy.load(path, **kwargs)


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for FP8 extra-state checkpoints.
    Only allows the narrow set of types that ``_encode_extra_state`` can
    produce: plain Python containers, numeric scalars, and the PyTorch
    tensor/storage primitives used by `pickle.dumps(tensor)`.  Any attempt
    to instantiate a class outside this allowlist raises
    `pickle.UnpicklingError`, preventing arbitrary code execution via a
    crafted checkpoint.
    """

    _SAFE_CLASSES: frozenset = frozenset(SAFE_OBJECTS)

    def find_class(self, module: str, name: str):
        if (module, name) not in self._SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}' "
            )
        return super().find_class(module, name)


class SafeDCPUnpickler(SafeUnpickler):
    """Restricted unpickler for the torch distributed checkpoint ``.metadata`` file.

    Only allows the dataclasses and primitives DCP writes into ``Metadata``.
    Anything else raises ``pickle.UnpicklingError``, so a tampered ``.metadata``
    cannot execute code at load time.
    """

    _SAFE_CLASSES: frozenset = frozenset(
        {
            **SAFE_OBJECTS,
            ("pathlib", "PosixPath"),
            ("torch.serialization", "_get_layout"),
            ("torch.distributed.checkpoint.filesystem", "_StorageInfo"),
            ("torch.distributed.checkpoint.metadata", "Metadata"),
            ("torch.distributed.checkpoint.metadata", "MetadataIndex"),
            ("torch.distributed.checkpoint.metadata", "StorageMeta"),
            ("torch.distributed.checkpoint.metadata", "TensorProperties"),
            ("torch.distributed.checkpoint.metadata", "TensorStorageMetadata"),
            ("torch.distributed.checkpoint.metadata", "ChunkStorageMetadata"),
            ("torch.distributed.checkpoint.metadata", "BytesStorageMetadata"),
            ("torch.distributed.checkpoint.metadata", "_MEM_FORMAT_ENCODING"),
        }
    )

    def find_class(self, module, name):
        if module == "torch" and isinstance(getattr(torch, name, None), torch.dtype):
            return getattr(torch, name)
        return super().find_class(module, name)


class _SafeDCPPickleModule(types.ModuleType):
    """Stand-in for the ``pickle`` module inside ``torch.distributed.checkpoint.filesystem``.

    ``load`` is restricted; everything else (e.g. ``dump``) is forwarded to real ``pickle``.
    """

    def __init__(self):
        super().__init__("pickle")

    def __getattr__(self, name):
        return getattr(pickle, name)

    @staticmethod
    def load(file, **kwargs):
        return SafeDCPUnpickler(file, **kwargs).load()


def patch_dcp_metadata_loading():
    """Make ``FileSystemReader.read_metadata`` use a restricted unpickler for ``.metadata``."""
    if getattr(_dcp_filesystem, "pickle", None) is pickle:
        _dcp_filesystem.pickle = _SafeDCPPickleModule()
