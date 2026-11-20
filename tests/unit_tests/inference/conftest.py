# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.tuning import AutotunePolicy, install
from megatron.core.tuning.policy import DEFAULT_MODULES
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="package", autouse=True)
def no_gloo_process_groups():
    """Inference tests never use Gloo groups, so their model-parallel setup skips them."""
    default = Utils.create_gloo_process_groups
    Utils.create_gloo_process_groups = False
    yield
    Utils.create_gloo_process_groups = default


@pytest.fixture(scope="package", autouse=True)
def pinned_triton_autotune():
    """Pin each autotuned Triton kernel to one config, as deterministic runs do.

    In the default mode every fresh process compiles and times every candidate config of each
    autotuned kernel at its first shape, which dominated the hybrid-model tests (FLA's gated
    delta rule among them). The timed winner depends on the machine, so no test can rely on it.
    """
    install(AutotunePolicy(mode="pinned", modules=DEFAULT_MODULES + ("fla",)))
    yield
    install()
