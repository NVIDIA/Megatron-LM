# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for GTP runtime configuration and buffer ownership."""

import pytest
import torch

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module


@pytest.fixture(autouse=True)
def _restore_gtp_config():
    original = {
        "pad_for_alignment": gtp_module.GTP_CONFIG.pad_for_alignment,
        "check_param_states": gtp_module.GTP_CONFIG.check_param_states,
        "weight_prefetch": gtp_module.GTP_CONFIG.weight_prefetch,
        "async_reduction": gtp_module.GTP_CONFIG.async_reduction,
        "calculate_per_token_loss": gtp_module.GTP_CONFIG.calculate_per_token_loss,
    }
    try:
        yield
    finally:
        gtp_module.update_gtp_config(**original)


def test_sync_batched_rs_returns_original_inputs_to_wgrad_pool(monkeypatch):
    """Sync RS outputs are shard buffers; only its full-wgrad inputs are pool-owned."""

    input_bufs = [torch.randn(4, 3), torch.randn(4, 3)]
    reduced_outputs = [torch.randn(2, 3), torch.randn(2, 3)]

    class _FakeWeight:
        def __init__(self, reduced):
            self.main_grad = torch.zeros_like(reduced)

    class _FakeSyncWgradOwner:
        def __init__(self):
            self._weights = [_FakeWeight(reduced) for reduced in reduced_outputs]
            self.chain_id = gtp_module.GTPChain.UNGRAPHED.value
            self.prev_w = None
            self.next_w = None
            # No reduction in flight from an earlier use (the MTP entry drain checks this).
            self._wgrad_rs_handle = None
            self.rs_calls = []

        def _reduce_scatter(self, wgrads, async_op, nvtx_label=None):
            del nvtx_label
            self.rs_calls.append((list(wgrads), async_op))
            return reduced_outputs, None

        @staticmethod
        def _handle_megatron_grad_accum(param):
            return torch.empty_like(param.main_grad)

    returned_to_pool = []
    monkeypatch.setattr(gtp_module, "_wgrad_pool_put", returned_to_pool.append)
    monkeypatch.setattr(gtp_module, "nvtx_range_push", lambda *_args: None)
    monkeypatch.setattr(gtp_module, "nvtx_range_pop", lambda *_args: None)

    owner = _FakeSyncWgradOwner()
    result = gtp_module.GTPShardedParam.wgrad_reduce_scatter(
        owner, input_bufs, nvtx_label="test.sync_batched"
    )

    assert len(result) == len(input_bufs)
    assert len(owner.rs_calls) == 1
    actual_inputs, async_op = owner.rs_calls[0]
    assert async_op is False
    assert all(actual is expected for actual, expected in zip(actual_inputs, input_bufs))
    assert len(returned_to_pool) == len(input_bufs)
    assert all(actual is expected for actual, expected in zip(returned_to_pool, input_bufs))
    assert not any(actual is output for actual in returned_to_pool for output in reduced_outputs)
    for weight, reduced in zip(owner._weights, reduced_outputs):
        torch.testing.assert_close(weight.main_grad, reduced)


@pytest.mark.parametrize(
    "recipe_kwargs,expected_pad",
    [
        pytest.param({"fp4": True}, 16, id="fp4"),
        pytest.param({"fp8_recipe": "mxfp8"}, 32, id="mxfp8"),
        pytest.param({"fp8": True}, 16, id="fp8"),
    ],
)
def test_recipe_defaults_set_pad_for_alignment(recipe_kwargs, expected_pad):
    """With no explicit value, each quantized recipe keeps its historical alignment pad."""
    gtp_module.configure_gtp_remat_from_recipe(**recipe_kwargs)
    assert gtp_module.GTP_CONFIG.pad_for_alignment == expected_pad


def test_bf16_recipe_leaves_pad_for_alignment_untouched():
    """No quantized branch fires for bf16; the pre-existing value must survive."""
    gtp_module.update_gtp_config(pad_for_alignment=48)
    gtp_module.configure_gtp_remat_from_recipe()
    assert gtp_module.GTP_CONFIG.pad_for_alignment == 48


@pytest.mark.parametrize(
    "recipe_kwargs",
    [
        pytest.param({}, id="bf16"),
        pytest.param({"fp4": True}, id="fp4"),
        pytest.param({"fp8_recipe": "mxfp8"}, id="mxfp8"),
    ],
)
def test_explicit_pad_for_alignment_wins_over_every_recipe(recipe_kwargs):
    """--gtp-remat-pad-for-alignment pins the pad; 0 disables padding entirely."""
    gtp_module.configure_gtp_remat_from_recipe(pad_for_alignment=0, **recipe_kwargs)
    assert gtp_module.GTP_CONFIG.pad_for_alignment == 0
