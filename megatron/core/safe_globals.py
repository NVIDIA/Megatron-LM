# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from argparse import Namespace
from io import BytesIO
from pathlib import PosixPath
from types import SimpleNamespace

import torch
from numpy import dtype, ndarray
from numpy.core.multiarray import _reconstruct
from numpy.dtypes import UInt32DType

from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import RerunDiagnostic, RerunMode, RerunState
from megatron.core.transformer.enums import AttnBackend

SAFE_GLOBALS = [
    SimpleNamespace,
    PosixPath,
    _reconstruct,
    ndarray,
    dtype,
    UInt32DType,
    Namespace,
    AttnBackend,
    ModelType,
    RerunDiagnostic,
    RerunMode,
    RerunState,
    BytesIO,
]


def register_safe_globals():
    """Register megatron-core safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])
