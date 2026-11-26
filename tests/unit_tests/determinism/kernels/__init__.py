# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Operator-level bit-exact determinism tests for the kernels Megatron-LM dispatches.

``manifest.py`` registers every kernel and the test that covers it; ``harness.py`` holds the
replay helpers; the ``test_*.py`` modules exercise one kernel family each. The determinism
env vars are pinned at import like ``correctness/`` so cuBLAS / TE / NCCL capture them on
first use.
"""

import os
import tempfile

import torch

from megatron.core.tuning import AutotunePolicy, install, set_deterministic_mode
from megatron.core.utils import configure_nvtx_profiling
from megatron.training.determinism import apply_determinism_env

# Perf breakdown (tests/performance_tests/shell_test_utils/determinism/kernel_breakdown.sh):
# ``run_nsys_breakdown.sh`` runs these tests twice and sets DETERMINISM_PERF_MODE. Either
# value tags every replay with an NVTX range (``harness._perf_range``). "nondet" also skips
# the pinning below, so the same tests time the default paths; their replay assertions may
# then fail, which is the expected sign that a kernel needs the deterministic path.
PERF_MODE = os.environ.get("DETERMINISM_PERF_MODE")
if PERF_MODE:
    configure_nvtx_profiling(True)
    # Fresh compile caches per pass: a kernel that torch.compile or Triton built in the other
    # mode's process must not be reused, or both passes time the code of whichever ran first.
    for _var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        os.environ[_var] = tempfile.mkdtemp(prefix=f"{_var.lower()}_{PERF_MODE}_")

if PERF_MODE != "nondet":
    # Pin workspaces and future decorations, and cover autotuners that training imports
    # already loaded. Replay must not include Triton's timing-based configuration search.
    # Use the default scope, as --deterministic-mode does, so replays exercise the same
    # pinning (Transformer Engine and in-tree fusions included) that training runs under.
    set_deterministic_mode(True)
    install(AutotunePolicy(mode="pinned"))
    # External Mamba and causal-conv libraries retain their own environment controls.
    os.environ.setdefault("MAMBA_DETERMINISTIC", "1")
    os.environ.setdefault("CAUSAL_CONV1D_DETERMINISTIC", "1")
    apply_determinism_env(os.environ)

if PERF_MODE == "det":
    # The rest of what --deterministic-mode does (megatron/training/determinism.py), so the
    # det pass also times PyTorch's deterministic implementations.
    torch.use_deterministic_algorithms(True)
if PERF_MODE:
    # --deterministic-mode clears this, and it only acts while deterministic algorithms are on.
    # Clear it in both passes: tests that turn deterministic algorithms on themselves would
    # otherwise pay NaN-filling of new tensors in the nondet pass only.
    torch.utils.deterministic.fill_uninitialized_memory = False

# Most kernel tests never initialise a process group, so pin each ``torch.distributed.run``
# rank to its own GPU here (after the env is set, before any CUDA context exists). This
# lives in the package ``__init__`` rather than a ``conftest.py`` on purpose: pytest loads a
# directory's conftest even when every test file in it is ``--ignore``d, which would leak
# the env pinning above into the generic unit-test bucket.
if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
