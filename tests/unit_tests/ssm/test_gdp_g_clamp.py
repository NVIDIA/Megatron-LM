# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional GDP decay bounds, gradients, and checkpoint arguments."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm import gated_delta_product as gdp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_network_size_args
from megatron.training.checkpointing import CheckpointType, load_args_from_checkpoint


def _mixer(clamp_value, *, cutedsl=False, chunkwise_cp=True):
    # Exercise gate math without constructing CUDA kernels or distributed layers.
    mixer = gdp.GatedDeltaProductMixer.__new__(gdp.GatedDeltaProductMixer)
    torch.nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(gdp_g_clamp_value=clamp_value, gdp_cutedsl_kernel=cutedsl)
    mixer.nheads_local_cp = 2
    mixer.num_householder = 3
    mixer.chunkwise_context_parallel = chunkwise_cp
    mixer.A_log = torch.nn.Parameter(torch.zeros(2))
    mixer.dt_bias = torch.nn.Parameter(torch.zeros(2))
    mixer.cp = SimpleNamespace(get_A_log=lambda: mixer.A_log, get_dt_bias=lambda: mixer.dt_bias)
    # A CP-1 geometry, returned for every microbatch in place of the process-group lookup.
    mixer.runtime_cp = gdp._GDPRuntimeCP(
        group=SimpleNamespace(size=lambda: 1),
        helper=None if chunkwise_cp else mixer.cp,
        chunkwise=chunkwise_cp,
        d_inner=8,
        nheads=2,
        ngroups=1,
    )
    mixer._resolve_runtime_cp = lambda packed_seq_params: mixer.runtime_cp
    return mixer


def test_g_clamp_is_opt_in_and_preserves_zero_decay_copies():
    # Decode interleaves one gate and two zero-decay Householder copies.
    raw = torch.tensor([-10000.0, 0.0, 0.0, -1.0, 0.0, 0.0], requires_grad=True)
    assert _mixer(None)._clamp_g(raw) is raw
    bounded = _mixer(50.0)._clamp_g(raw)
    torch.testing.assert_close(
        bounded, torch.tensor([-50.0, 0.0, 0.0, -1.0, 0.0, 0.0]), rtol=0, atol=0
    )
    bounded.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert raw[0].item() == -10000.0


@pytest.mark.parametrize("clamp_value", [None, 50.0])
@pytest.mark.parametrize("dynamic_slots", [False, True])
def test_decode_forwards_bounded_g_to_recurrence(clamp_value, dynamic_slots):
    mixer = _mixer(clamp_value)
    mixer.ngroups_local_cp = 1
    mixer.headdim = 4
    mixer.d_state = 4
    raw_g = torch.tensor(
        [[[-10000.0, -1.0], [0.0, 0.0], [0.0, 0.0]], [[-100.0, -0.25], [0.0, 0.0], [0.0, 0.0]]]
    )
    original_g = raw_g.clone()
    qkv = torch.zeros(2, 3, 2, 4)
    beta = torch.full((2, 3, 2), 0.5)
    state = torch.zeros(3, 2, 4, 4)
    slots = torch.tensor([2, 0], dtype=torch.int32) if dynamic_slots else None
    # Compact output: only the last Householder step per token.
    recurrent_output = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4)
    projected_output = torch.zeros(1, 2, 8)

    with (
        patch.object(
            mixer,
            "_preprocess",
            return_value=(torch.zeros(2, 1, 8), torch.zeros(2, 1, 24), torch.zeros(2, 1, 8)),
        ),
        patch.object(mixer, "_decode_conv", return_value=torch.zeros(2, 1, 24)),
        patch.object(gdp, "gdp_decode_prepare", return_value=(qkv, qkv, qkv, beta, raw_g)),
        patch.object(
            gdp, "fused_recurrent_gated_delta_rule_update", return_value=(recurrent_output, state)
        ) as recurrence,
        patch.object(mixer, "_postprocess", return_value=projected_output) as postprocess,
    ):
        output = mixer.ssm_decode(torch.zeros(2, 1, 40), torch.zeros(2, 24, 4), state, slots)

    recurrence.assert_called_once()
    forwarded = recurrence.call_args.kwargs
    expected = original_g if clamp_value is None else original_g.clamp_min(-clamp_value)
    torch.testing.assert_close(forwarded["g"], expected, rtol=0, atol=0)
    if clamp_value is None:
        assert forwarded["g"] is raw_g
    else:
        assert forwarded["g"] is not raw_g
    torch.testing.assert_close(raw_g, original_g, rtol=0, atol=0)
    assert torch.count_nonzero(forwarded["g"][:, 1:]) == 0
    assert forwarded["state"] is state
    assert forwarded["state_indices"] is slots
    assert forwarded["beta"] is beta
    torch.testing.assert_close(postprocess.call_args.args[0], recurrent_output)
    torch.testing.assert_close(output, projected_output.transpose(0, 1))


@pytest.mark.parametrize("clamp_value", [None, 50.0])
@pytest.mark.parametrize("cutedsl", [False, True])
@pytest.mark.parametrize("chunkwise_cp", [False, True])
def test_gating_clamps_decay_with_correct_gradients(clamp_value, cutedsl, chunkwise_cp):
    mixer = _mixer(clamp_value, cutedsl=cutedsl, chunkwise_cp=chunkwise_cp)
    ba = torch.tensor([[[0.0] * 6 + [10000.0, 0.0]]], requires_grad=True)

    beta, gate = mixer._compute_gating(ba, runtime_cp=mixer.runtime_cp)

    raw = -F.softplus(ba.detach()[..., 6:])
    expected = raw if clamp_value is None else raw.clamp_min(-clamp_value)
    torch.testing.assert_close(gate, expected.reshape_as(gate), rtol=0, atol=0)
    assert gate.shape == ((1, 2) if cutedsl else (1, 1, 2))
    assert gate.dtype == torch.float32
    torch.testing.assert_close(beta, torch.full_like(beta, 0.5), rtol=0, atol=0)

    gate.sum().backward()
    active = torch.ones_like(raw) if clamp_value is None else (raw > -clamp_value)
    expected_da = -ba.detach()[..., 6:].sigmoid() * active
    torch.testing.assert_close(ba.grad[..., 6:], expected_da)
    torch.testing.assert_close(ba.grad[..., :6], torch.zeros_like(ba.grad[..., :6]))
    torch.testing.assert_close(mixer.A_log.grad, (raw * active).sum(dim=(0, 1)))
    torch.testing.assert_close(mixer.dt_bias.grad, expected_da.sum(dim=(0, 1)))


def test_g_clamp_cli_and_default():
    parser = _add_network_size_args(argparse.ArgumentParser())
    assert parser.parse_args([]).gdp_g_clamp_value is None
    assert parser.parse_args(["--gdp-g-clamp-value", "50"]).gdp_g_clamp_value == 50.0
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2)
    assert config.gdp_g_clamp_value is None


@pytest.mark.parametrize("value", [0.0, -50.0, float("inf"), float("-inf"), float("nan")])
def test_g_clamp_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="gdp_g_clamp_value must be finite and positive"):
        TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=2, gdp_g_clamp_value=value
        )


@pytest.mark.parametrize(
    "saved,current,expected",
    [
        ("missing", None, None),
        ("missing", 50.0, 50.0),
        (None, 50.0, 50.0),
        (80.0, None, 80.0),
        (80.0, 50.0, 50.0),
    ],
)
def test_checkpoint_g_clamp_restore_respects_explicit_cli(saved, current, expected):
    checkpoint_args = SimpleNamespace()
    if saved != "missing":
        checkpoint_args.gdp_g_clamp_value = saved
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        gdp_g_clamp_value=current,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    with patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(
            {"args": checkpoint_args, "iteration": 12},
            "checkpoint",
            False,
            CheckpointType.LEGACY,
        ),
    ):
        restored_args, _ = load_args_from_checkpoint(args)
    assert restored_args.gdp_g_clamp_value == expected
