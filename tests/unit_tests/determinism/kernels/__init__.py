# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Operator-level bit-exact determinism tests for the kernels Megatron-LM dispatches.

``manifest.py`` registers every kernel and the test that covers it; ``harness.py`` holds the
replay helpers; the ``test_*.py`` modules exercise one kernel family each. The determinism
env vars are pinned at import like ``correctness/`` so cuBLAS / TE / NCCL capture them on
first use.
"""

import os

import torch

from megatron.training.determinism import apply_determinism_env

# The SSM Triton autotuners read the determinism policy when their modules are imported
# (``megatron/core/ssm/ops/common/determinism.py``); pin it before any kernel module loads
# so the tests measure the kernels, not Triton's timing-based config search.
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")
os.environ.setdefault("CAUSAL_CONV1D_DETERMINISTIC", "1")
apply_determinism_env(os.environ)

# Most kernel tests never initialise a process group, so pin each ``torch.distributed.run``
# rank to its own GPU here (after the env is set, before any CUDA context exists). This
# lives in the package ``__init__`` rather than a ``conftest.py`` on purpose: pytest loads a
# directory's conftest even when every test file in it is ``--ignore``d, which would leak
# the env pinning above into the generic unit-test bucket.
if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
