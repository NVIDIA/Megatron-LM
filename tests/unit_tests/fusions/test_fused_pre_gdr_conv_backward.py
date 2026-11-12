# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the channel-last conv + SiLU backward of the fused pre-gated-delta-rule path.

The Triton backward is compared against the causal-conv1d backward it replaces (values of
every gradient the fused function returns, with packed document boundaries and a
chunkwise-CP left boundary), against a float64 autograd reference that does not share any
code with either kernel, and after a CUDA graph replay whose packing differs from the
capture.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

try:
    import triton  # pylint: disable=unused-import

    from megatron.core.fusions import fused_pre_gated_delta_rule as pre_gdr

    HAVE_FUSION = True
except ImportError:
    HAVE_FUSION = False

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

    HAVE_CAUSAL_CONV1D = callable(causal_conv1d_bwd_function)
except ImportError:
    HAVE_CAUSAL_CONV1D = False

pytestmark = [
    pytest.mark.skipif(not HAVE_FUSION, reason="triton is not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]
needs_causal_conv1d = pytest.mark.skipif(
    not HAVE_CAUSAL_CONV1D, reason="causal-conv1d is not installed"
)

NUM_KEY_HEADS = 2
NUM_VALUE_HEADS = 4
KEY_HEAD_DIM = 16
VALUE_HEAD_DIM = 16
QK_CHANNELS = NUM_KEY_HEADS * KEY_HEAD_DIM
V_CHANNELS = NUM_VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * QK_CHANNELS + V_CHANNELS
TOTAL_CHANNELS = CONV_DIM + V_CHANNELS + 2 * NUM_VALUE_HEADS
HEAD_KWARGS = dict(
    num_key_heads=NUM_KEY_HEADS,
    num_value_heads=NUM_VALUE_HEADS,
    key_head_dim=KEY_HEAD_DIM,
    value_head_dim=VALUE_HEAD_DIM,
)

# Document boundaries that fall inside, at the start of and at the end of the kernel's
# token blocks, including one-token documents and documents shorter than the conv window.
PACKINGS = {
    "one_doc": lambda t: [t],
    "two_docs": lambda t: [t // 2, t - t // 2],
    "short_docs": lambda t: [1, 2, 3, 61, 1, t - 68],
    "block_edges": lambda t: [64, 64, 1, 127, t - 256],
}


def _cu_seqlens(lengths, num_entries=None):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    while num_entries is not None and len(cu) < num_entries:
        cu.append(cu[-1])
    return torch.tensor(cu, device="cuda", dtype=torch.int32)


def _seq_idx(cu_seqlens, total_tokens):
    positions = torch.arange(total_tokens, device="cuda", dtype=torch.int32)
    return (
        torch.searchsorted(cu_seqlens[1:].contiguous(), positions, right=True)
        .to(torch.int32)
        .unsqueeze(0)
    )


def _make_inputs(seq_len, batch, k_w, dtype, weight_dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0, out_dtype=torch.float32):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(out_dtype)

    return dict(
        qkvzba=randn(seq_len, batch, TOTAL_CHANNELS, out_dtype=dtype).requires_grad_(True),
        conv1d_weight=randn(CONV_DIM, 1, k_w, scale=0.5, out_dtype=weight_dtype).requires_grad_(
            True
        ),
        A_log=randn(NUM_VALUE_HEADS, scale=0.1).requires_grad_(True),
        dt_bias=randn(NUM_VALUE_HEADS, scale=0.1).requires_grad_(True),
    )


def _loss_weights(outputs, seed=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(out.shape, device="cuda", generator=generator, dtype=torch.float32)
        for out in outputs
    ]


def _run_fused(inputs, impl, cu_seqlens=None, seq_idx=None, loss_weights=None, num_outputs=6):
    """Run the fused path and return ``(outputs, grads)`` for a fixed linear loss."""

    params = [inputs[name] for name in ("qkvzba", "conv1d_weight", "A_log", "dt_bias")]
    outputs = pre_gdr.fused_streamed_pre_gated_delta_rule(
        inputs["qkvzba"],
        inputs["conv1d_weight"],
        None,
        inputs["A_log"],
        inputs["dt_bias"],
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        conv_backward_impl=impl,
        **HEAD_KWARGS,
    )
    if loss_weights is None:
        loss_weights = _loss_weights(outputs)
    loss = sum(
        (out.float() * weight).sum()
        for out, weight in list(zip(outputs, loss_weights))[:num_outputs]
    )
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return outputs, grads


def _assert_close(actual, expected, rtol, name):
    """Compare against the scale of the reference so near-zero entries do not dominate."""

    # A silent broadcast would compare unrelated entries.
    assert actual.shape == expected.shape, f"{name}: {actual.shape} vs {expected.shape}"
    actual = actual.double()
    expected = expected.double()
    scale = expected.abs().max().clamp_min(1e-12)
    error = (actual - expected).abs().max()
    assert error <= rtol * scale, f"{name}: max abs error {error:.3e} vs scale {scale:.3e}"


def _reference_qkv(qkvzba, conv1d_weight, lengths):
    """Float64 conv + SiLU + l2norm + head repeat with per-document causal convs."""

    k_w = conv1d_weight.shape[-1]
    repeat = NUM_VALUE_HEADS // NUM_KEY_HEADS
    weight = conv1d_weight.double()
    outs = []
    for batch_id in range(qkvzba.shape[1]):
        x = qkvzba[:, batch_id, :CONV_DIM].double()
        docs = []
        start = 0
        for length in lengths:
            doc = x[start : start + length].t().unsqueeze(0)
            conv = F.conv1d(doc, weight, padding=k_w - 1, groups=CONV_DIM)[..., :length]
            docs.append(F.silu(conv).squeeze(0).t())
            start += length
        outs.append(torch.cat(docs, dim=0))
    y = torch.stack(outs, dim=0)  # (batch, seq_len, conv_dim)
    batch, seq_len, _ = y.shape

    def l2norm_repeat(part):
        part = part.reshape(batch, seq_len, NUM_KEY_HEADS, KEY_HEAD_DIM)
        part = part * torch.rsqrt(part.pow(2).sum(-1, keepdim=True) + 1e-6)
        return part.repeat_interleave(repeat, dim=2)

    query = l2norm_repeat(y[..., :QK_CHANNELS])
    key = l2norm_repeat(y[..., QK_CHANNELS : 2 * QK_CHANNELS])
    value = y[..., 2 * QK_CHANNELS :].reshape(batch, seq_len, NUM_VALUE_HEADS, VALUE_HEAD_DIM)
    return query, key, value


@contextlib.contextmanager
def _redirect_stale_autograd_streams():
    """Let a capture run backward through leaves whose autograd nodes predate it.

    The warmup call creates the leaves' ``AccumulateGrad`` nodes on the default stream;
    PyTorch refuses to capture through them unless they are redirected to the capturing
    stream. The full-iteration capture sets the same switch (``full_cuda_graph.py``).
    Older PyTorch builds have neither the check nor the switch.
    """

    set_override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
    if set_override is None:
        yield
        return
    set_override(True)
    try:
        yield
    finally:
        set_override(False)


@needs_causal_conv1d
@pytest.mark.parametrize("packing", [None, *PACKINGS])
@pytest.mark.parametrize("k_w", [2, 3, 4])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_matches_causal_conv1d_backward(packing, k_w, weight_dtype):
    """Same forward values; every gradient within bf16 rounding of the replaced path."""

    seq_len = 333
    batch = 1 if packing is not None else 2
    inputs = _make_inputs(seq_len, batch, k_w, torch.bfloat16, weight_dtype)
    cu_seqlens = None if packing is None else _cu_seqlens(PACKINGS[packing](seq_len))

    out_new, grads_new = _run_fused(inputs, "triton", cu_seqlens)
    out_old, grads_old = _run_fused(inputs, "causal_conv1d", cu_seqlens)

    for new, old in zip(out_new, out_old):
        assert torch.equal(new, old)
    for name, new, old in zip(
        ("d_qkvzba", "d_weight", "d_A_log", "d_dt_bias"), grads_new, grads_old
    ):
        assert new.dtype == old.dtype and new.shape == old.shape
        # Both sides round bf16 results; the weight gradient also differs in fp32
        # summation order.
        _assert_close(new, old, rtol=2e-2, name=name)
    # The gate / beta / decay input gradients do not go through the conv backward at all.
    assert torch.equal(grads_new[0][..., CONV_DIM:], grads_old[0][..., CONV_DIM:])


@pytest.mark.parametrize("packing", [None, *PACKINGS])
@pytest.mark.parametrize("k_w", [2, 4])
@pytest.mark.parametrize("seq_len", [333, 512])
def test_matches_float64_autograd_reference(packing, k_w, seq_len):
    """fp32 run against an independent float64 reference, tight tolerance.

    Only the conv outputs (q, k, v) enter the loss, so the reference needs no gate math.
    """

    batch = 1 if packing is not None else 2
    inputs = _make_inputs(seq_len, batch, k_w, torch.float32, torch.float32)
    lengths = [seq_len] if packing is None else PACKINGS[packing](seq_len)
    cu_seqlens = None if packing is None else _cu_seqlens(lengths)

    outputs, grads = _run_fused(inputs, "triton", cu_seqlens, num_outputs=3)
    weights = _loss_weights(outputs)

    ref_qkvzba = inputs["qkvzba"].detach().double().requires_grad_(True)
    ref_weight = inputs["conv1d_weight"].detach().double().requires_grad_(True)
    ref_outputs = _reference_qkv(ref_qkvzba, ref_weight, lengths)
    ref_loss = sum((out * weight.double()).sum() for out, weight in zip(ref_outputs, weights))
    ref_d_qkvzba, ref_d_weight = torch.autograd.grad(ref_loss, [ref_qkvzba, ref_weight])

    for name, out, ref in zip("qkv", outputs, ref_outputs):
        _assert_close(out, ref, rtol=1e-4, name=name)
    _assert_close(grads[0][..., :CONV_DIM], ref_d_qkvzba[..., :CONV_DIM], rtol=1e-4, name="d_x")
    _assert_close(grads[1], ref_d_weight, rtol=1e-4, name="d_weight")


@pytest.mark.parametrize("k_w", [2, 4])
def test_packed_documents_do_not_leak_gradients(k_w):
    """A document's input gradient only depends on that document's upstream gradient."""

    seq_len = 300
    lengths = [70, 1, 129, 100]
    inputs = _make_inputs(seq_len, 1, k_w, torch.float32, torch.float32)
    cu_seqlens = _cu_seqlens(lengths)
    outputs, _ = _run_fused(inputs, "triton", cu_seqlens)
    weights = _loss_weights(outputs)

    # Zero the loss on document 2 (tokens [71, 200)): its inputs must get no gradient
    # from the conv channels, and the other documents' gradients must not change.
    masked = [weight.clone() for weight in weights]
    for weight in masked:
        weight[:, 71:200] = 0
    _, grads_full = _run_fused(inputs, "triton", cu_seqlens, loss_weights=weights)
    _, grads_masked = _run_fused(inputs, "triton", cu_seqlens, loss_weights=masked)

    assert torch.count_nonzero(grads_masked[0][71:200, :, :CONV_DIM]) == 0
    assert torch.equal(grads_masked[0][:71, :, :CONV_DIM], grads_full[0][:71, :, :CONV_DIM])
    assert torch.equal(grads_masked[0][200:, :, :CONV_DIM], grads_full[0][200:, :, :CONV_DIM])


def _direct_backward(impl, seq_len, k_w, dtype, cu_seqlens, seq_idx, left_boundary, offset, total):
    generator = torch.Generator(device="cuda").manual_seed(7)

    def randn(*shape, out_dtype=dtype):
        return torch.randn(*shape, device="cuda", generator=generator).to(out_dtype)

    args = dict(
        qkvzba=randn(seq_len, 1, TOTAL_CHANNELS),
        conv1d_weight=randn(CONV_DIM, 1, k_w, out_dtype=torch.float32) * 0.5,
        silu_qk_save=randn(1, seq_len, 2 * QK_CHANNELS).permute(0, 2, 1),
        dq=randn(1, seq_len, NUM_VALUE_HEADS, KEY_HEAD_DIM),
        dk=randn(1, seq_len, NUM_VALUE_HEADS, KEY_HEAD_DIM),
        dv=randn(1, seq_len, NUM_VALUE_HEADS, VALUE_HEAD_DIM),
        dgate=randn(1, seq_len, NUM_VALUE_HEADS, VALUE_HEAD_DIM),
        dbeta=randn(1, seq_len, NUM_VALUE_HEADS, out_dtype=torch.float32),
        dg=randn(1, seq_len, NUM_VALUE_HEADS, out_dtype=torch.float32),
        A_log=randn(NUM_VALUE_HEADS, out_dtype=torch.float32) * 0.1,
        dt_bias=randn(NUM_VALUE_HEADS, out_dtype=torch.float32) * 0.1,
    )
    return pre_gdr._triton_pre_gated_delta_rule_backward(
        **args,
        **HEAD_KWARGS,
        seq_idx=seq_idx,
        left_boundary=left_boundary,
        cu_seqlens=cu_seqlens,
        global_token_offset=offset,
        global_seq_len=total,
        conv_backward_impl=impl,
    )


@needs_causal_conv1d
@pytest.mark.parametrize(
    "global_lengths",
    [
        None,  # unpacked: every boundary token feeds the local shard
        [30, 68, 32, 70],  # document [98, 130) spans the shard start at 100
        [100, 100],  # a document starts exactly at the shard start
        [99, 101],  # one boundary token belongs to the first local document
        [200],  # one document across both shards
    ],
)
@pytest.mark.parametrize("k_w", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_left_boundary_matches_causal_conv1d_backward(global_lengths, k_w, dtype):
    """Chunkwise-CP shard of rank 1 of 2: input, weight and left-boundary gradients."""

    seq_len = 100
    offset = 100 if global_lengths is not None else 0
    total = 200 if global_lengths is not None else seq_len
    cu_seqlens = None if global_lengths is None else _cu_seqlens(global_lengths)
    seq_idx = (
        None
        if cu_seqlens is None
        else pre_gdr._resolve_chunkwise_cp_packed_seq_idx(cu_seqlens, seq_len, cp_rank=1)
    )
    generator = torch.Generator(device="cuda").manual_seed(11)
    left_boundary = torch.randn(k_w - 1, 1, CONV_DIM, device="cuda", generator=generator).to(dtype)

    new = _direct_backward(
        "triton", seq_len, k_w, dtype, cu_seqlens, seq_idx, left_boundary, offset, total
    )
    old = _direct_backward(
        "causal_conv1d", seq_len, k_w, dtype, cu_seqlens, seq_idx, left_boundary, offset, total
    )
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    names = ("d_qkvzba", "d_weight", "d_A_log", "d_dt_bias", "d_left_boundary")
    for name, new_grad, old_grad in zip(names, new, old):
        assert new_grad.dtype == old_grad.dtype and new_grad.shape == old_grad.shape
        _assert_close(new_grad, old_grad, rtol=rtol, name=name)
    if global_lengths == [100, 100]:
        assert torch.count_nonzero(new[4]) == 0


@pytest.mark.parametrize("k_w", [4])
def test_cuda_graph_replay_sees_current_packing(k_w):
    """Capture with one packing, replay with another: gradients follow the replay's."""

    seq_len = 320
    num_entries = 9
    packings = {"capture": [seq_len], "replay": [50, 1, 77, 64, 128]}

    def static_inputs(seed):
        return _make_inputs(seq_len, 1, k_w, torch.bfloat16, torch.float32, seed=seed)

    static = static_inputs(seed=3)
    cu_static = _cu_seqlens(packings["capture"], num_entries)
    seq_idx_static = _seq_idx(cu_static, seq_len)
    # Warm up outside the capture: Triton compiles and autotunes on the first calls.
    outputs, _ = _run_fused(static, "triton", cu_static, seq_idx_static)
    weights = _loss_weights(outputs)

    graph = torch.cuda.CUDAGraph()
    with _redirect_stale_autograd_streams(), torch.cuda.graph(graph):
        graph_outputs, graph_grads = _run_fused(
            static, "triton", cu_static, seq_idx_static, loss_weights=weights
        )
        # ``gate`` is a view into ``qkvzba``. The test refills that buffer in place between
        # replays, after which autograd refuses to read the view, so keep captured copies.
        graph_outputs = [out.detach().clone() for out in graph_outputs]

    for name, seed in (("replay", 5), ("capture", 6)):
        fresh = static_inputs(seed=seed)
        cu_fresh = _cu_seqlens(packings[name], num_entries)
        with torch.no_grad():
            for key in static:
                static[key].copy_(fresh[key])
            cu_static.copy_(cu_fresh)
            seq_idx_static.copy_(_seq_idx(cu_fresh, seq_len))
        graph.replay()
        replayed_outputs = [out.clone() for out in graph_outputs]
        replayed_grads = [grad.clone() for grad in graph_grads]

        eager_outputs, eager_grads = _run_fused(
            fresh, "triton", cu_fresh, _seq_idx(cu_fresh, seq_len), loss_weights=weights
        )
        for replayed, eager in zip(replayed_outputs, eager_outputs):
            assert torch.equal(replayed, eager)
        # Input gradients have a fixed evaluation order; the parameter gradients are
        # atomic fp32 sums whose order may change between launches.
        assert torch.equal(replayed_grads[0], eager_grads[0])
        for name_, replayed, eager in zip(
            ("d_weight", "d_A_log", "d_dt_bias"), replayed_grads[1:], eager_grads[1:]
        ):
            _assert_close(replayed, eager, rtol=1e-4, name=name_)


class _PinnedLaunchConfig:
    """Stand-in for the autotuned kernel that launches one fixed configuration."""

    def __init__(self, autotuned_kernel, block_c, block_t, num_warps):
        self.kernel = autotuned_kernel.fn
        self.config = dict(BLOCK_C=block_c, BLOCK_T=block_t)
        self.num_warps = num_warps

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.kernel[grid(self.config)](
                *args, **kwargs, **self.config, num_warps=self.num_warps
            )

        return launch


def _reference_conv_backward(x, weight, d_silu, lengths):
    """Float64 autograd of ``sum(d_silu * silu(conv(x)))`` with per-document causal convs."""

    k_w = weight.shape[-1]
    x = x.double().requires_grad_(True)
    weight = weight.double().requires_grad_(True)
    loss = 0
    start = 0
    for length in lengths:
        doc = x[start : start + length].t().unsqueeze(0)
        conv = F.conv1d(doc, weight, padding=k_w - 1, groups=x.shape[1])[..., :length]
        loss = loss + (F.silu(conv).squeeze(0).t() * d_silu[start : start + length].double()).sum()
        start += length
    d_x, d_weight = torch.autograd.grad(loss, [x, weight])
    # ``[channels, k_w]`` like the kernel's accumulator (the conv weight has a unit dim).
    return d_x, d_weight.view(weight.shape[0], k_w)


@pytest.mark.parametrize(
    "block_c, block_t, num_warps",
    [
        # 512 channels: blocks of 256 / 512 take the unmasked lean scan ...
        (256, 128, 4),
        (512, 64, 4),
        (512, 128, 4),
        # ... a wider block the masked one.
        (1024, 128, 8),
    ],
)
@pytest.mark.parametrize("packing", [None, "one_doc", "short_docs", "block_edges"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_launch_config_matches_float64_reference(
    monkeypatch, block_c, block_t, num_warps, packing, dtype
):
    """The autotuner may pick any config; each must be correct on its own.

    Most token blocks lie inside one document and take the lean scan; the first block and
    the blocks around a document boundary take the checked one.
    """

    monkeypatch.setattr(
        pre_gdr,
        "_conv_silu_backward_kernel",
        _PinnedLaunchConfig(pre_gdr._conv_silu_backward_kernel, block_c, block_t, num_warps),
    )
    seq_len, num_chan, chan_offset, k_w = 700, 512, 64, 4
    generator = torch.Generator(device="cuda").manual_seed(13)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    qkvzba = randn(seq_len, 1, chan_offset + num_chan + 40).to(dtype)
    weight = randn(chan_offset + num_chan, 1, k_w) * 0.5
    d_silu = randn(1, seq_len, num_chan).to(dtype)
    lengths = [seq_len] if packing is None else PACKINGS[packing](seq_len)
    seq_idx = None if packing is None else _seq_idx(_cu_seqlens(lengths), seq_len)

    d_qkvzba = torch.full_like(qkvzba, float("nan"))
    d_weight = torch.zeros(chan_offset + num_chan, k_w, device="cuda")
    pre_gdr._triton_conv_silu_backward(
        qkvzba, weight, d_silu, d_qkvzba, d_weight, chan_offset=chan_offset, seq_idx=seq_idx
    )

    channels = slice(chan_offset, chan_offset + num_chan)
    ref_d_x, ref_d_weight = _reference_conv_backward(
        qkvzba[:, 0, channels], weight[channels], d_silu[0], lengths
    )
    rtol = 1e-4 if dtype == torch.float32 else 2e-2
    _assert_close(d_qkvzba[:, 0, channels], ref_d_x, rtol=rtol, name="d_x")
    _assert_close(d_weight[channels], ref_d_weight, rtol=rtol, name="d_weight")
    # Nothing outside the launch's channel range is written.
    assert torch.isnan(d_qkvzba[:, 0, :chan_offset]).all()
    assert torch.isnan(d_qkvzba[:, 0, chan_offset + num_chan :]).all()
    assert torch.count_nonzero(d_weight[:chan_offset]) == 0


def test_non_unit_channel_stride_uses_checked_scan():
    """A strided upstream gradient is still handled (by the checked scan only)."""

    seq_len, num_chan, k_w = 300, 256, 4
    generator = torch.Generator(device="cuda").manual_seed(17)
    qkvzba = torch.randn(seq_len, 1, num_chan, device="cuda", generator=generator)
    weight = torch.randn(num_chan, 1, k_w, device="cuda", generator=generator) * 0.5
    wide = torch.randn(1, seq_len, 2 * num_chan, device="cuda", generator=generator)
    d_silu = wide[:, :, ::2]
    assert d_silu.stride(2) == 2

    d_qkvzba = torch.empty_like(qkvzba)
    d_weight = torch.zeros(num_chan, k_w, device="cuda")
    pre_gdr._triton_conv_silu_backward(qkvzba, weight, d_silu, d_qkvzba, d_weight, chan_offset=0)
    ref_d_x, ref_d_weight = _reference_conv_backward(qkvzba[:, 0], weight, d_silu[0], [seq_len])
    _assert_close(d_qkvzba[:, 0], ref_d_x, rtol=1e-4, name="d_x")
    _assert_close(d_weight, ref_d_weight, rtol=1e-4, name="d_weight")


def test_impl_selection(monkeypatch):
    monkeypatch.delenv(pre_gdr._CONV_BACKWARD_IMPL_ENV, raising=False)
    assert pre_gdr._resolve_conv_backward_impl(None, 4) == "triton"
    assert pre_gdr._resolve_conv_backward_impl("triton", 1) == "triton"
    with pytest.raises(ValueError, match="kernel width"):
        pre_gdr._resolve_conv_backward_impl("triton", 5)
    with pytest.raises(ValueError, match="Unknown"):
        pre_gdr._resolve_conv_backward_impl("cudnn", 4)
    if HAVE_CAUSAL_CONV1D:
        assert pre_gdr._resolve_conv_backward_impl(None, 5) == "causal_conv1d"
        monkeypatch.setenv(pre_gdr._CONV_BACKWARD_IMPL_ENV, "causal_conv1d")
        assert pre_gdr._resolve_conv_backward_impl(None, 4) == "causal_conv1d"
        # An explicit argument wins over the environment.
        assert pre_gdr._resolve_conv_backward_impl("triton", 4) == "triton"
