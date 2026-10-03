# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the ``jit_fuser`` / ``torch.compile`` fused elementwise kernels.

Covers every compiled function under ``megatron/core/fusions/`` that is not a Triton or CUDA
kernel, plus the compiled activations and helpers scattered through ``megatron/core``
(``activations.py``, ``transformer/utils.py``, ``transformer/torch_norm.py``,
``transformer/attention.py``). Inductor picks reduction tilings per shape; the row
reductions in the weighted backward passes (``torch.sum(weights_grad, dim=-1)``) are the
only non-elementwise math, so shapes are sized to make those reductions wide.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.attention as attention_module
from megatron.core import activations, parallel_state
from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.transformer.attention import Attention, SelfAttention
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.utils import erf_gelu, gelu_impl
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TOKENS = CONTENTION_TOKENS
FFN = 8192
DTYPE = torch.bfloat16


def _act(shape, dtype=DTYPE, grad=True):
    return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=grad)


def _weights(rows):
    return torch.rand(rows, 1, device="cuda", dtype=torch.float32, requires_grad=True)


# --- gated / weighted MLP fusions ----------------------------------------------------------

GATED_CASES = {
    "bias_swiglu": lambda: (bias_swiglu_impl, (_act((TOKENS, 2 * FFN)), _act((2 * FFN,)))),
    "swiglu_no_bias": lambda: (bias_swiglu_impl, (_act((TOKENS, 2 * FFN)), None)),
    "clamped_swiglu": lambda: (
        lambda x, b: bias_swiglu_impl(x, b, clamp_value=7.0),
        (_act((TOKENS, 2 * FFN)), _act((2 * FFN,))),
    ),
    "situ_glu": lambda: (
        lambda x, b: bias_swiglu_impl(x, b, gate_clamp_scale=10.0, linear_clamp_scale=3.0),
        (_act((TOKENS, 2 * FFN)), None),
    ),
    "weighted_swiglu": lambda: (
        weighted_bias_swiglu_impl,
        (_act((TOKENS, 2 * FFN)), None, _weights(TOKENS)),
    ),
    "weighted_clamped_swiglu": lambda: (
        lambda x, b, w: weighted_bias_swiglu_impl(x, b, w, clamp_value=7.0),
        (_act((TOKENS, 2 * FFN)), None, _weights(TOKENS)),
    ),
    "weighted_situ_glu": lambda: (
        lambda x, b, w: weighted_bias_swiglu_impl(
            x, b, w, gate_clamp_scale=10.0, linear_clamp_scale=3.0
        ),
        (_act((TOKENS, 2 * FFN)), None, _weights(TOKENS)),
    ),
    "bias_geglu": lambda: (bias_geglu_impl, (_act((TOKENS, 2 * FFN)), _act((2 * FFN,)))),
    "geglu_no_bias": lambda: (bias_geglu_impl, (_act((TOKENS, 2 * FFN)), None)),
    "weighted_quick_geglu": lambda: (
        lambda x, b, w: weighted_bias_quick_geglu_impl(x, b, w, linear_offset=1.0),
        (_act((TOKENS, 2 * FFN)), None, _weights(TOKENS)),
    ),
    "weighted_clamped_quick_geglu": lambda: (
        lambda x, b, w: weighted_bias_quick_geglu_impl(x, b, w, clamp_value=10.0),
        (_act((TOKENS, 2 * FFN)), None, _weights(TOKENS)),
    ),
    "bias_gelu": lambda: (bias_gelu_impl, (_act((TOKENS, FFN)), _act((FFN,)))),
    "weighted_squared_relu": lambda: (
        weighted_squared_relu_impl,
        (_act((TOKENS, FFN)), _weights(TOKENS)),
    ),
    "weighted_clamped_squared_relu": lambda: (
        lambda x, w: weighted_squared_relu_impl(x, w, clamp_scale=10.0),
        (_act((TOKENS, FFN)), _weights(TOKENS)),
    ),
    # Unweighted clamped path (weights=None) used by the dense MLP; saves only x for backward.
    "clamped_squared_relu": lambda: (
        lambda x: weighted_squared_relu_impl(x, None, clamp_scale=10.0),
        (_act((TOKENS, FFN)),),
    ),
}


@pytest.mark.parametrize("case", sorted(GATED_CASES))
def test_mlp_activation_fusions_replay_bit_exactly(case):
    seeded()
    fn, inputs = GATED_CASES[case]()
    assert_replays_bit_exact(fn, inputs, replays=3, what=case)


@pytest.mark.parametrize("weighted", [False, True])
def test_clamped_swiglu_boundaries_replay_bit_exactly(weighted):
    seeded()
    # Include both endpoints and their adjacent BF16 values in each clamp branch.
    bounds = torch.tensor(
        [-10.0625, -10.0, -9.9375, 9.9375, 10.0, 10.0625], device="cuda", dtype=DTYPE
    )
    x = torch.cat((bounds, bounds)).repeat(32, 1).requires_grad_(True)
    if weighted:
        fn = lambda x, w: weighted_bias_swiglu_impl(x, None, w, clamp_value=10.0)
        inputs = (x, _weights(32) + 1.0)
    else:
        fn = lambda x: bias_swiglu_impl(x, None, clamp_value=10.0)
        inputs = (x,)
    _, grads = assert_replays_bit_exact(fn, inputs, replays=3, what="clamped SwiGLU boundaries")

    # Replay alone would also accept a consistently wrong endpoint derivative.
    # Compare its zero mask with the installed PyTorch scalar-clamp contract.
    reference_x = x.detach().clone().requires_grad_(True)
    gate, linear = reference_x.float().chunk(2, dim=-1)
    reference = (F.silu(gate.clamp(max=10.0)) * linear.clamp(-10.0, 10.0)).to(DTYPE)
    if weighted:
        reference = (reference * inputs[1]).to(DTYPE)
    reference_grad = torch.autograd.grad(reference.sum(), reference_x)[0]
    assert torch.equal(grads["in[0]"] == 0, reference_grad == 0)


# --- plain compiled activations -----------------------------------------------------------

ACTIVATION_CASES = {
    "squared_relu": lambda x: activations.squared_relu(x),
    "quick_gelu": lambda x: activations.quick_gelu(x),
    "fast_gelu": lambda x: activations.fast_gelu(x),
    "tanh_soft_clamp": lambda x: activations.tanh_soft_clamp(x, 5.0),
    "situ": lambda x: activations.situ(x, 5.0),
    "situ_glu": lambda x: activations.situ_glu(x, 5.0, 3.0),
    "openai_gelu": gelu_impl,
    "erf_gelu": erf_gelu,
}


@pytest.mark.parametrize("case", sorted(ACTIVATION_CASES))
def test_compiled_activations_replay_bit_exactly(case):
    seeded()
    x = _act((TOKENS, FFN)) * 5.0
    x = x.detach().requires_grad_(True)
    assert_replays_bit_exact(ACTIVATION_CASES[case], (x,), replays=3, what=case)


def test_attention_output_gate_replays_bit_exactly():
    """``Attention._apply_output_gate`` is a compiled method; ``self`` is unused."""
    seeded()
    x = _act((2048, 4, 4096))
    gate = _act((2048, 4, 4096))
    assert_replays_bit_exact(
        lambda x, gate: Attention._apply_output_gate(None, x, gate),
        (x, gate),
        replays=3,
        what="attention output gate",
    )


def test_l2norm_replays_bit_exactly():
    """QK L2 norm: compiled row reduction (``pow(2).mean(-1)``) over head_dim."""
    seeded()
    norm = L2Norm(hidden_size=128).cuda()
    x = _act((4096, 2, 32, 128))
    assert_replays_bit_exact(norm, (x,), replays=3, what="L2Norm")


# --- bias + dropout + residual --------------------------------------------------------------


@pytest.mark.parametrize("residual_dtype", [torch.bfloat16, torch.float32])
def test_bias_dropout_add_fused_train_replays_under_restored_rng(residual_dtype):
    """Dropout consumes the CUDA RNG: identical mask, output and grads when the RNG is restored."""
    seeded()
    x = _act((2048, 4, 4096))
    bias = _act((4096,))
    residual = _act((2048, 4, 4096), dtype=residual_dtype, grad=False)

    def fn(x, bias, residual):
        return bias_dropout_add_fused_train((x, bias), residual, 0.1)

    assert_replays_bit_exact(
        fn, (x, bias, residual), replays=3, restore_rng=True, what="bias_dropout_add_fused_train"
    )


def test_bias_dropout_add_fused_inference_replays_bit_exactly():
    seeded()
    x = _act((2048, 4, 4096), grad=False)
    bias = _act((4096,), grad=False)
    residual = _act((2048, 4, 4096), grad=False)
    assert_replays_bit_exact(
        lambda x, b, r: bias_dropout_add_fused_inference((x, b), r, 0.1),
        (x, bias, residual),
        replays=3,
        backward=False,
        what="bias_dropout_add_fused_inference",
    )


# --- fused vocab-parallel cross entropy ------------------------------------------------------


class TestFusedCrossEntropy:
    """``--cross-entropy-loss-fusion`` is rejected by ``--deterministic-mode``; this records
    whether the compiled kernel itself replays bit-exactly (op catalog: non-deterministic)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @pytest.mark.xfail(
        strict=False,
        reason="op-catalog lists the fused cross entropy as non-deterministic; recorded, not gated",
    )
    def test_fused_vocab_parallel_cross_entropy_replays(self, dtype):
        """Both logits dtypes: fp32 logits skip the internal upcast, and the backward must
        hand back a gradient in the logits dtype rather than a hardcoded one."""
        seeded()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        logits = _act((TOKENS, 32768), dtype=dtype)
        target = torch.randint(0, 32768 * tp_group.size(), (TOKENS,), device="cuda")
        _, grads = assert_replays_bit_exact(
            lambda l, t: fused_vocab_parallel_cross_entropy(l, t, tp_group),
            (logits, target),
            replays=4,
            what="fused_vocab_parallel_cross_entropy",
        )
        assert grads["in[0]"].dtype == dtype


# --- DeepSeek-V4 hybrid attention: compiled query RMS norm -----------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_dsv4_q_rms_norm_replays(dtype, layout):
    """``_q_rms_norm`` (weightless RMS norm, ``torch.compile``) on a [s, b, heads, dim] query.

    Inductor reduces ``q.square().mean(-1)`` per row; the transposed layout exercises the
    strided-input specialisation the compiler guards on separately.
    """
    try:
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            _q_rms_norm,
        )
    except ImportError as e:  # pragma: no cover - depends on optional dependencies
        pytest.skip(f"DeepSeek-V4 hybrid attention unavailable: {e}")

    seeded()
    seq, batch, heads, dim = 1024, 4, 32, 128
    if layout == "contiguous":
        q = _act((seq, batch, heads, dim), dtype=dtype)
    else:
        q = _act((batch, seq, heads, dim), dtype=dtype).transpose(0, 1)
    assert_replays_bit_exact(
        lambda q: _q_rms_norm(q, 1e-6), (q,), replays=3, contention=True, what="_q_rms_norm"
    )


@pytest.mark.parametrize("is_decode_only", [False, True])
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize(
    "flash_attention_version,head_dim,page_size",
    [pytest.param(4, 64, 128, id="fa4"), pytest.param(None, 8, 256, id="auto-hopper-small-head")],
)
def test_paged_attention_dispatch_replays_bit_exactly(
    monkeypatch, is_decode_only, has_sink, flash_attention_version, head_dim, page_size
):
    if not attention_module.HAVE_FA4:
        pytest.skip("requires FlashAttention-4")
    capability = torch.cuda.get_device_capability()[0]
    if capability not in (9, 10, 11):
        pytest.skip("requires FA4 paged attention on Hopper, Blackwell, or Rubin")
    if flash_attention_version is None and capability != 9:
        pytest.skip("requires Hopper to exercise the small-head automatic fallback")

    seeded()
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
    attention.layer_number = 1
    attention.batch_invariant_mode = False
    attention.flash_attention_version = flash_attention_version
    attention.train(False)

    # Observe dispatch arguments while still running the real FA4 kernel.
    fa4 = attention_module.flash_attn4_varlen_func
    fa4_calls = []

    def checked_fa4(*args, **kwargs):
        assert kwargs["num_splits"] == (1 if capability == 9 else 0)
        assert kwargs["return_lse"] is has_sink
        fa4_calls.append(True)
        return fa4(*args, **kwargs)

    monkeypatch.setattr(attention_module, "flash_attn4_varlen_func", checked_fa4)
    q = torch.randn(2, 1, 4, head_dim, device="cuda", dtype=DTYPE)
    num_pages = 2048 // page_size
    k = torch.randn(num_pages, page_size, 4, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int32)
    seqlens_k = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).reshape(2, -1)
    offset = torch.arange(4, device="cuda", dtype=torch.float32) if has_sink else None

    @torch.inference_mode()
    def forward(query, key, value):
        output = attention.flash_decode_and_prefill(
            q=query,
            k=key,
            v=value,
            max_seqlen_q=1,
            max_seqlen_k=1024,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqlens_k=seqlens_k,
            block_table=block_table,
            is_decode_only=is_decode_only,
            softmax_offset=offset,
        )
        assert torch.isfinite(output).all()
        return output

    assert_replays_bit_exact(
        forward, (q, k, v), replays=3, backward=False, what="paged attention dispatch"
    )
    assert len(fa4_calls) == (3 if flash_attention_version == 4 else 0)
