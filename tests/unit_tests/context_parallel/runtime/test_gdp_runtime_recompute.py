# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed runtime-CP GDP projection recomputation and saved-gate storage."""

import pytest

from tests.unit_tests.ssm.gdp_test_utils import (
    HAVE_GDP_DEPS,
    assert_in_proj_recompute_preserves_norm_gate_view,
)

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def deterministic_conv1d(monkeypatch):
    """Hold the conv backward to its deterministic reduction for every test in this module.

    Without it, `causal_conv1d_fn` accumulates its weight gradient across blocks with atomicAdd
    on the channel-last layout the mixer feeds it, so two runs of the *same* mixer disagree on
    `conv1d.weight` and a parity test has to tolerate a drifting gradient. Set per-test rather
    than process-wide because the env var is read on each backward call.
    """
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize("recompute_qkv", [False, True])
@pytest.mark.parametrize("fp8_recipe", [None, "mxfp8"])
def test_runtime_cp_in_proj_recompute_preserves_norm_gate_view(
    monkeypatch, recompute_qkv, fp8_recipe
):
    """Runtime CP2 keeps and restores the packed projection's saved gate view."""
    assert_in_proj_recompute_preserves_norm_gate_view(
        monkeypatch, recompute_qkv, fp8_recipe, cp_size=1, runtime_cp_size=2
    )
