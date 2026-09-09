# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Initialize the environment for bit-exact correctness tests.

Each determinism test imports this module before its other dependencies so
cuBLAS, Transformer Engine, and NCCL capture the settings at first use.
The launcher must set them before CUDA initialization if another collected
test has already initialized CUDA.

``CUDA_DEVICE_MAX_CONNECTIONS=1`` is also needed for pre-Blackwell async-TP
correctness. Set it before importing the training helper, which imports
PyTorch. Launcher overrides such as ``=32`` for Blackwell MoE overlap are
preserved by ``setdefault``.
"""

import os

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from megatron.training.determinism import apply_determinism_env  # noqa: E402

apply_determinism_env(os.environ)
