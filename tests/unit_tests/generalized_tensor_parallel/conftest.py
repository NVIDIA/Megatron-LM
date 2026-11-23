# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixtures applied to every GTP unit test.

The GTP bucket runs every module in one process, so each test must leave no GTP state behind.
"""

import dataclasses

import pytest
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    """Initialize the torchrun-managed dist group once per module."""
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def reset_gtp_state_after_test():
    """Restore the process-global GTP and FP8 state a test may have changed.

    Beyond the production reset (chain cursors and wgrad rings), this drops the buffers and side
    streams later tests would otherwise reuse, every GTP param ever built, the CUDA-graph scope and
    GTP_CONFIG.
    """
    config = dataclasses.replace(gtp.GTP_CONFIG)
    yield
    gtp.reset_gtp_state()
    gtp._GTP_PARAMS.clear()
    gtp.get_global_GTP_cache().clear()
    gtp._wgrad_buf_pool.clear()
    gtp._inflight_comm_params.clear()
    # Keyed by id(group): a later group reusing a destroyed group's id would inherit its streams.
    gtp._AG_STREAMS.clear()
    gtp._RS_STREAMS.clear()
    gtp.set_cuda_graph_modules(None)
    for field in dataclasses.fields(config):
        setattr(gtp.GTP_CONFIG, field.name, getattr(config, field.name))
    FP8GlobalStateManager.reset()
