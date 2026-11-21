# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FlashInfer is a hard requirement of every oracle in this package.

The CI image ships FlashInfer. Without it a test skips with a reason, or fails when the run
targets a known node (``MCORE_ORACLE_TARGET_SM``), so a runner that lost the package cannot pass
by skipping.
"""

import pytest

from tests.unit_tests.inference.oracles import oracle_utils as ou

try:
    import flashinfer  # noqa: F401

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False


@pytest.fixture(autouse=True)
def require_flashinfer():
    """Skip without FlashInfer, or fail when the run targets a node that must have it."""
    if HAVE_FLASHINFER:
        return
    reason = "FlashInfer is not installed (the CI image ships it)"
    if ou.session_target_sm() is not None:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)
