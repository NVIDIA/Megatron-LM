# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for GatedDeltaProduct selective recompute.

`gdp_in_proj` recompute is meant to be an exact replay of the input projection and its
preprocessing: the forward pass is unaffected, and the backward pass rebuilds the same
subgraph from the same saved input. So enabling it must not move the numerics at all.
Any drift would mean the replay observed a different weight, a different RNG state, or
a stale storage after the output-discarding checkpoint shared the recomputed one back.

`conv1d.weight` used to be exempt, because the conv backward reduced it with atomicAdd. The
fixture below turns on the kernel's deterministic reduction instead, so it is held to the same
bitwise bar as everything else.
"""

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gdp_test_utils import (
    HAVE_GDP_DEPS,
    HIDDEN_SIZE,
    MAMBA_NUM_HEADS,
    _assert_replay_matches,
    _build_mixer,
    _make_config,
    _run,
    assert_in_proj_recompute_preserves_norm_gate_view,
)
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support

    fp8_available, reason_for_no_fp8 = check_fp8_support()
except ImportError:
    fp8_available, reason_for_no_fp8 = False, "transformer-engine is not installed"


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
def test_in_proj_recompute_parity(recompute_qkv):
    """`gdp_in_proj` recompute must be an exact replay, alone and with `gdp_qkv`.

    The two are stacked deliberately: `gdp_qkv` saves VKQ, which `gdp_in_proj` discards,
    so this pins the ordering contract between the two recompute hooks -- the in_proj
    replay on the mixer output has to rematerialize VKQ before the QKV replay on
    core_attn_out consumes it.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        modules = ["gdp_in_proj"] + (["gdp_qkv"] if recompute_qkv else [])
        baseline = _build_mixer(_make_config([]))
        recomputed = _build_mixer(_make_config(modules), state_dict=baseline.state_dict())
        assert recomputed.recompute_in_proj and not baseline.recompute_in_proj
        assert recomputed.recompute_qkv == recompute_qkv

        sequence_length, micro_batch_size = 256, 2
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)

        _assert_replay_matches(
            _run(baseline, hidden_states, grads), _run(recomputed, hidden_states, grads)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_recipe", ["tensorwise"])
def test_fp8_in_proj_recompute_parity(fp8_recipe):
    """The replay must stay exact under FP8, where in_proj is the quantized op.

    CheckpointWithoutOutput records the forward recipe and amax state and replays under
    the recorded fp8_autocast, so the replayed GEMM has to see the same scales as the
    original. Drift here would mean the recompute re-derived them from an amax history
    that had already moved on.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        baseline = _build_mixer(_make_config([], fp8_recipe=fp8_recipe))
        recomputed = _build_mixer(
            _make_config(["gdp_in_proj"], fp8_recipe=fp8_recipe), state_dict=baseline.state_dict()
        )
        assert recomputed.recompute_in_proj and not baseline.recompute_in_proj

        # FP8 GEMMs need the token dimension aligned; keep it a multiple of 128.
        sequence_length, micro_batch_size = 256, 2
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)

        _assert_replay_matches(
            _run(baseline, hidden_states, grads), _run(recomputed, hidden_states, grads)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize("recompute_qkv", [False, True])
@pytest.mark.parametrize("fp8_recipe", [None, "mxfp8"])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_in_proj_recompute_preserves_norm_gate_view(
    monkeypatch, recompute_qkv, fp8_recipe, cp_size
):
    """At MB=1, CP1 and chunkwise CP must discard and restore the norm's saved gate."""
    assert_in_proj_recompute_preserves_norm_gate_view(
        monkeypatch, recompute_qkv, fp8_recipe, cp_size
    )


def test_in_proj_recompute_rejects_qkv_offload():
    """gdp_in_proj leaves nothing for the gdp_qkv offload group, so the pair is rejected."""
    with pytest.raises(ValueError, match="gdp_qkv cannot be set in offload_modules"):
        TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=MAMBA_NUM_HEADS,
            recompute_granularity="selective",
            recompute_modules=["gdp_in_proj"],
            fine_grained_activation_offloading=True,
            offload_modules=["gdp_qkv"],
        )
