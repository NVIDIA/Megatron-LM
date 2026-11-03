# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys

# Inductor's Triton compile sidecar re-forks its worker pool after an idle quiesce; a worker
# forked mid-shutdown can inherit a held lock and never return its job, hanging the rank in
# async_compile.wait (pytorch/pytorch#200848). CPython fixed the lock in 3.12.8 and 3.13.1.
if sys.version_info < (3, 12, 8) or sys.version_info[:3] == (3, 13, 0):
    os.environ.setdefault("TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL", "0")

import torch._dynamo  # after the env default: importing dynamo loads torch._inductor.config

torch._dynamo.config.suppress_errors = True
