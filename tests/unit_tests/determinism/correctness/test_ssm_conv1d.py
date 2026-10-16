# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism checks for the SSM causal convolution.

Operator-level rather than model-level: the conv's channel-last backward reduces its weight
gradient over ``micro_batch * ceil(seq_len / 128)`` blocks with ``atomicAdd``, and the model
cells in this directory run 32x2, which is two blocks -- too few to expose a regression.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import causal_conv1d as causal_conv1d_module
from megatron.core.ssm.causal_conv1d import assert_causal_conv1d_deterministic
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.determinism.configs import hybrid_base
from tests.unit_tests.determinism.utils import (
    assert_bit_exact,
    capture_rng_state,
    collect_grads,
    restore_rng_state,
    zero_grads,
)
from tests.unit_tests.test_utilities import Utils

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False

# The deterministic conv backward landed in causal_conv1d 1.6.0; older builds ignore the flag.
DETERMINISTIC_CONV1D_MIN = "1.6.0"
HAVE_DETERMINISTIC_CAUSAL_CONV1D = (
    HAVE_CAUSAL_CONV1D
    and causal_conv1d_module.is_causal_conv1d_min_version(DETERMINISTIC_CONV1D_MIN)
)

requires_deterministic_conv1d = pytest.mark.skipif(
    not (HAVE_DETERMINISTIC_CAUSAL_CONV1D and torch.cuda.is_available()),
    reason=f"needs a GPU and causal_conv1d >= {DETERMINISTIC_CONV1D_MIN}",
)

GRAD_NAMES = ("dx", "dweight", "dbias")

# 4 * ceil(1024 / 128) = 32 contending blocks for the mixer, 2 * ceil(4096 / 128) = 64 for the
# kernel, against a measurement where 1 contributor never drifts and 32 always does.
_SEQ_LEN = 1024
_MICRO_BATCH = 4
_KERNEL_SHAPE = dict(batch=2, dim=256, seq_len=4096, width=4)


def _channel_last_conv_inputs(batch, dim, seq_len, width):
    """Build a channel-last [B, D, L] conv input plus weight, bias and an output gradient."""
    torch.manual_seed(7)
    x = (
        torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .detach()
        .requires_grad_()
    )
    assert x.stride(1) == 1, "these tests are about the channel-last kernel"
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    return x, weight, bias, grad


def _conv_backward(x, weight, bias, grad):
    out = causal_conv1d_fn(x=x, weight=weight, bias=bias, activation="silu")
    return torch.autograd.grad(out, (x, weight, bias), grad_outputs=grad)


def _build_mixer(deterministic_mode=True):
    """A bare MambaMixer on the suite's shared hybrid config."""
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        **(hybrid_base() | {"num_layers": 1, "deterministic_mode": deterministic_mode})
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp']),
    )
    return mixer.cuda()


def _fake_min_version(monkeypatch, supported):
    monkeypatch.setattr(
        causal_conv1d_module, "is_causal_conv1d_min_version", lambda *_a, **_kw: supported
    )


@requires_deterministic_conv1d
@pytest.mark.parametrize("deterministic", [False, True])
def test_kernel_replays_bitwise(monkeypatch, deterministic):
    """The channel-last backward replays bit-for-bit only under the deterministic reduction.

    The ``False`` arm asserts that the hardware is nondeterministic, so it is sized with
    margin: 64 contending blocks, against 19 of 19 replays differing on GB300 at 16. ``dx`` is
    written by disjoint stores, not the reduction, so it is reproducible either way. The env
    var is read on each backward call.
    """
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1" if deterministic else "0")
    replays = 8
    inputs = _channel_last_conv_inputs(**_KERNEL_SHAPE)
    first, differing = None, dict.fromkeys(GRAD_NAMES, 0)
    for _ in range(replays):
        got = _conv_backward(*inputs)
        if first is None:
            first = got
            continue
        for name, ref, cur in zip(GRAD_NAMES, first, got):
            differing[name] += not torch.equal(ref, cur)

    assert differing["dx"] == 0
    if deterministic:
        assert differing["dweight"] == 0 and differing["dbias"] == 0
    else:
        assert differing["dweight"] > 0, (
            f"the default channel-last backward was bit-reproducible over {replays} replays. "
            "Either upstream made it deterministic -- in which case drop this control -- or "
            "this GPU serialized the contending blocks, which the shape above is meant to "
            "prevent."
        )


@requires_deterministic_conv1d
def test_deterministic_reduction_agrees_with_the_default_one(monkeypatch):
    """The deterministic path must reorder the same sum, not compute a different one.

    Two summation orders cannot be bitwise equal in fp32, so the bar is the gap relative to
    each tensor's own magnitude: measured ~6e-07, a few ulps, against a 1e-5 bound. ``rtol=0``
    with the magnitude in ``atol`` because per element the gap reaches ~1e-2, where a
    ``dweight`` entry near zero cancels much larger products. ``dx`` skips the reduction.
    """
    inputs = _channel_last_conv_inputs(**_KERNEL_SHAPE)

    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
    default = _conv_backward(*inputs)
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    deterministic = _conv_backward(*inputs)

    for name, ref, got in zip(GRAD_NAMES, default, deterministic):
        if name == "dx":
            assert torch.equal(ref, got), "dx does not go through the reduction"
            continue
        torch.testing.assert_close(
            got,
            ref,
            rtol=0,
            atol=float(ref.abs().max()) * 1e-5,
            msg=f"{name} moved by more than fp32 accumulation error",
        )


class TestMambaMixerDeterminism:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @requires_deterministic_conv1d
    def test_mixer_replays_bit_exactly(self, monkeypatch):
        """Two runs of one mixer agree bitwise under the deterministic conv reduction.

        ``MAMBA_DETERMINISTIC`` pins the SSD scan, whose nondeterminism would otherwise reach
        the conv weight gradient, so a failure points at the convolution.
        """
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
        mixer = _build_mixer()
        hidden_size = mixer.config.hidden_size
        torch.manual_seed(7)
        hidden_states = torch.randn(_SEQ_LEN, _MICRO_BATCH, hidden_size, device="cuda")
        grad = torch.randn(_SEQ_LEN, _MICRO_BATCH, hidden_size, device="cuda")

        def fwd_bwd():
            output, _ = mixer(hidden_states)
            output.backward(grad)
            return output.detach().clone(), collect_grads([mixer])

        state = capture_rng_state()
        out_a, grads_a = fwd_bwd()
        # Drain run A's collectives and autograd post-hooks, as BitExactRunner._two_runs does;
        # otherwise B overlaps A's tail.
        torch.cuda.synchronize()
        restore_rng_state(state)
        zero_grads(mixer)
        out_b, grads_b = fwd_bwd()

        assert_bit_exact(out_a, grads_a, out_b, grads_b)

    @pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal_conv1d is not installed")
    def test_deterministic_mode_requires_a_deterministic_conv(self, monkeypatch):
        """``MambaMixer.__init__`` actually calls the guard, so a disabled reduction raises."""
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
        with pytest.raises(AssertionError, match="deterministic causal_conv1d backward"):
            _build_mixer(deterministic_mode=True)


@pytest.mark.parametrize("env", ["1", None])
def test_guard_accepts_an_enabled_kernel(monkeypatch, env):
    """Both ways the kernel can be enabled: the env var, and the torch flag it falls back to.

    ``None`` is what ``--deterministic-mode`` produces: env unset plus
    ``torch.use_deterministic_algorithms(True)``.
    """
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    if env is None:
        monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", env)
    _fake_min_version(monkeypatch, True)
    assert_causal_conv1d_deterministic(deterministic_mode=True)


def test_guard_rejects_what_cannot_deliver_it(monkeypatch):
    """Both failure modes raise; neither is inherited from torch's flag alone, which unrelated
    tests set globally and never restore."""
    _fake_min_version(monkeypatch, True)
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
    with pytest.raises(AssertionError, match="deterministic causal_conv1d backward"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)

    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    _fake_min_version(monkeypatch, False)
    with pytest.raises(AssertionError, match="is required for deterministic_mode"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)

    # Same state, but the run never asked: must return rather than raise.
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    assert_causal_conv1d_deterministic(deterministic_mode=False)
