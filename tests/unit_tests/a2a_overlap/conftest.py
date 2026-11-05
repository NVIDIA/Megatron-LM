# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Share one model-parallel state across the A2A overlap tests of a module.

destroy_model_parallel() releases the fused A2A (DeepEP and HybridEP) buffers together with
the process groups they communicate over, and building a HybridEP buffer takes tens of
seconds. Initializing and destroying model parallelism around every test would rebuild the
buffer for every HybridEP case. Tests request their layout through ``shared_model_parallel``
instead. It keeps the current state while tests request the same layout, and destroys it when
a test requests a different layout, when a test fails on any rank, and at the end of the
module.
"""

import pytest
import torch

import megatron.core.parallel_state as ps
from megatron.core.transformer.moe.token_dispatcher import nccl_ep_release_context
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

# (layout, model-parallel group) of the state initialized through this module, or None.
_shared = None


def _destroy_shared():
    global _shared
    _shared = None
    Utils.destroy_model_parallel()


def _use_layout(**layout):
    """Initialize model parallelism with ``layout``, reusing the current state if it matches."""
    global _shared
    clear_nvte_env_vars()
    if (
        _shared is not None
        and _shared[0] == layout
        and ps.model_parallel_is_initialized()
        and ps.get_model_parallel_group(check_initialized=False) is _shared[1]
    ):
        return
    if _shared is not None:
        _destroy_shared()
    Utils.initialize_model_parallel(**layout)
    _shared = (layout, ps.get_model_parallel_group())


@pytest.fixture(scope="module")
def shared_model_parallel():
    """Return a function that initializes model parallelism with the given layout.

    Its keyword arguments are those of Utils.initialize_model_parallel. The state persists
    across the module's tests while they request the same layout.
    """
    yield _use_layout
    if _shared is not None:
        _destroy_shared()


@pytest.fixture(autouse=True)
def _reset_shared_model_parallel(request):
    """Reset per-test state between tests that share model parallelism."""
    failures_before = request.session.testsfailed
    yield
    if _shared is None:
        return
    # All ranks must agree on whether to destroy the groups, and a test can fail on a
    # subset of ranks. Destroy them after any failure, so later tests start fresh.
    torch.cuda.synchronize()
    failed = torch.tensor(
        [int(request.session.testsfailed > failures_before)], device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        _destroy_shared()
        return
    # The NCCL EP context is sized by the first dispatch that bootstraps it, so each test
    # bootstraps its own, as it did when every test destroyed model parallelism.
    nccl_ep_release_context()
    torch.cuda.empty_cache()
