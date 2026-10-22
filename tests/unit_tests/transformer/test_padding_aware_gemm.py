# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import padding_aware_gemm as pag

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and pag.HAVE_PADDING_AWARE_GEMM and pag.HAVE_TRITON),
    reason="padding-aware GEMM needs a GPU, Transformer Engine's grouped GEMM and Triton",
)


@pytest.mark.parametrize(
    ("mask_rows", "expected"),
    [
        ([False] * 1000 + [True] * 24, 1024),  # last real row 999 -> 1000 -> rounded to 1024
        ([False] * 1024, 1024),  # no padding
        ([True] * 1024, 128),  # all padding: one alignment block
        ([False] * 5 + [True] * 3 + [False] * 2 + [True] * 1014, 128),  # gaps stay inside
        ([False] * 9000 + [True] * 1240, 9088),  # longer than one kernel block
    ],
)
def test_live_rows_bound(mask_rows, expected):
    mask = torch.tensor(mask_rows, device="cuda").view(1, -1)
    assert pag.live_rows_bound(mask).item() == expected
    # The Triton kernel and the torch fallback agree.
    valid = (~mask.view(-1)).to(torch.int8)
    from_end = torch.argmax(valid.flip(0), dim=0)
    fallback = ((mask.numel() - from_end + 127) // 128 * 128).clamp(max=mask.numel())
    if not valid.any():
        fallback = torch.tensor(128, device=fallback.device)
    assert pag.live_rows_bound(mask).item() == fallback.item()


def test_live_rows_context_checks_the_buffer_length():
    bound = torch.tensor([256], device="cuda", dtype=torch.int64)
    assert pag.current_live_rows() is None
    with pag.padding_aware_live_rows(bound, total_rows=1024):
        assert pag.current_live_rows() is bound
        assert pag.current_live_rows_for(1024) is bound
        assert pag.current_live_rows_for(512) is None  # a differently sized buffer (TP shard)
        with pag.padding_aware_live_rows(None):
            assert pag.current_live_rows() is None
        assert pag.current_live_rows_for(1024) is bound
    assert pag.current_live_rows() is None


# ------------------------------------------------------------------------------------------
# bf16 LM head
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m_valid", [128, 640, 1024])
@pytest.mark.parametrize("fused_wgrad", [False, "grouped", "dense"])
def test_padding_aware_linear_matches_dense_on_live_rows(m_valid, fused_wgrad):
    """fused_wgrad selects the main_grad accumulation: off, the live-row grouped GEMM via the
    fp32 scratch, or mcore's dense fused kernel over the full capacity (both policy outcomes)."""
    torch.manual_seed(0)
    T, K, N = 1024, 256, 512
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    x[m_valid:] = 0  # padded inputs are zero, as data_schedule leaves them
    x_pa = x.clone().requires_grad_()
    x_ref = x.clone().requires_grad_()
    weight = torch.nn.Parameter(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05)
    weight_ref = torch.nn.Parameter(weight.detach().clone())
    if fused_wgrad:
        weight.main_grad = torch.full((N, K), 3.0, device="cuda", dtype=torch.float32)
        weight.grad_added_to_main_grad = False
    m_rows = torch.tensor([m_valid], device="cuda", dtype=torch.int64)

    out = pag.padding_aware_linear(
        x_pa,
        weight,
        m_rows,
        gradient_accumulation_fusion=bool(fused_wgrad),
        allreduce_dgrad=False,
        tp_group=None,
        wgrad_grouped=(fused_wgrad == "grouped"),
    )
    ref = x_ref @ weight_ref.t()
    assert torch.equal(out[:m_valid], ref[:m_valid])
    assert not out[m_valid:].any()

    grad = torch.randn_like(out)
    grad[m_valid:] = 0
    out.backward(grad)
    ref.backward(grad)
    assert torch.equal(x_pa.grad[:m_valid], x_ref.grad[:m_valid])
    assert not x_pa.grad[m_valid:].any()
    if fused_wgrad:
        assert weight.grad_added_to_main_grad
        # The fused path accumulates in fp32; compare against an fp32 reference rather than the
        # bf16-rounded ``weight_ref.grad`` (whose rounding error is ~0.1 at these magnitudes).
        dw_fp32 = grad.view(-1, N).float().t() @ x.view(-1, K).float()
        torch.testing.assert_close(weight.main_grad, 3.0 + dw_fp32, rtol=1e-4, atol=1e-2)
    else:
        torch.testing.assert_close(
            weight.grad.float(), weight_ref.grad.float(), rtol=1e-3, atol=1e-2
        )


def test_padding_aware_linear_replays_in_a_cuda_graph_with_a_new_bound():
    torch.manual_seed(0)
    T, K, N = 1024, 256, 512
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    m_rows = torch.tensor([640], device="cuda", dtype=torch.int64)
    out = torch.empty(T, 1, N, device="cuda", dtype=torch.bfloat16)

    def step():
        out.copy_(
            pag.padding_aware_linear(
                x,
                weight,
                m_rows,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                tp_group=None,
            )
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    x.copy_(torch.randn_like(x))
    m_rows.fill_(256)
    g.replay()
    torch.cuda.synchronize()
    ref = x @ weight.t()
    assert torch.equal(out[:256], ref[:256])
    assert not out[256:].any()


# ------------------------------------------------------------------------------------------
# Static kernel policy (model shape only)
# ------------------------------------------------------------------------------------------


def _fake_module(out_features=4096, in_features=1024, **overrides):
    attrs = dict(
        out_features=out_features,
        in_features=in_features,
        use_bias=False,
        parallel_mode=None,
        tp_size=1,
        symmetric_ar_type=None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _fake_config(**overrides):
    attrs = dict(
        padding_aware_dense_linear=True,
        max_seqlen_per_dp_cp_rank=16384,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        delay_wgrad_compute=False,
        fp8="mxfp8",
        cuda_graph_impl="transformer_engine",
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_dense_linear_policy_is_static_from_the_model_shape(monkeypatch):
    monkeypatch.delenv("MCORE_PADDING_AWARE_WGRAD", raising=False)
    monkeypatch.delenv("MCORE_PADDING_AWARE_FORCE", raising=False)
    big = _fake_module(out_features=4096, in_features=32768)  # out-proj
    policy = pag.dense_linear_policy(big, _fake_config())
    assert policy.enabled and policy.wgrad_scratch  # 512 MiB fp32 gradient -> scratch + add
    # The cost model (assumed 20 % padding) rejects shapes whose GEMM cannot amortise the fixed
    # per-linear costs: small N at any capacity, every DSv4 shape at 4096 rows, and the tail
    # zero-fill + add grow with the shape, so the decision is not a bare capacity threshold.
    small = pag.dense_linear_policy(
        _fake_module(out_features=1024, in_features=4096), _fake_config()
    )
    assert not small.enabled and small.eligible and "predicted saving" in small.reason
    # The assumed padding fraction is a config knob: a recipe with more tail padding enables
    # smaller layers.
    assert pag.dense_linear_policy(
        _fake_module(out_features=1024, in_features=4096),
        _fake_config(padding_aware_assumed_padding=0.5),
    ).enabled
    assert not pag.dense_linear_policy(big, _fake_config(max_seqlen_per_dp_cp_rank=4096)).enabled
    assert pag.dense_linear_policy(
        big, _fake_config(max_seqlen_per_dp_cp_rank=8192, fp8=None)
    ).enabled
    assert pag.dense_linear_policy(
        _fake_module(), _fake_config(fp8=None)
    ).enabled  # bf16 GEMMs are slower
    # Eager training pays the grouped path's CPU cost on the critical path: graph mode only.
    monkeypatch.delenv("MCORE_PADDING_AWARE_EAGER", raising=False)
    eager = pag.dense_linear_policy(big, _fake_config(cuda_graph_impl="none"))
    assert not eager.enabled and eager.eligible and "eager" in eager.reason
    assert pag.dense_linear_policy(
        big, _fake_config(cuda_graph_impl="none", enable_cuda_graph=True)
    ).enabled
    monkeypatch.setenv("MCORE_PADDING_AWARE_EAGER", "1")
    assert pag.dense_linear_policy(big, _fake_config(cuda_graph_impl="none")).enabled
    monkeypatch.delenv("MCORE_PADDING_AWARE_EAGER", raising=False)
    # Hard constraints (the layer is then not even eligible).
    off = pag.dense_linear_policy(big, _fake_config(padding_aware_dense_linear=False))
    assert not off.enabled and not off.eligible
    assert not pag.dense_linear_policy(
        _fake_module(4096, 32768, use_bias=True), _fake_config()
    ).eligible
    assert not pag.dense_linear_policy(big, _fake_config(max_seqlen_per_dp_cp_rank=2048)).enabled
    assert not pag.dense_linear_policy(big, _fake_config(max_seqlen_per_dp_cp_rank=None)).enabled
    assert not pag.dense_linear_policy(
        _fake_module(4096, 32768, use_bias=True), _fake_config()
    ).enabled
    assert not pag.dense_linear_policy(
        _fake_module(4096, 32768, parallel_mode="column", tp_size=2), _fake_config()
    ).enabled
    assert pag.dense_linear_policy(
        _fake_module(4096, 32768, parallel_mode="column", tp_size=1), _fake_config()
    ).enabled
    assert not pag.dense_linear_policy(_fake_module(4096, 1000), _fake_config()).enabled
    assert not pag.dense_linear_policy(big, _fake_config(delay_wgrad_compute=True)).enabled
    # Sequence parallelism shrinks the per-rank capacity.
    assert not pag.dense_linear_policy(
        big,
        _fake_config(
            max_seqlen_per_dp_cp_rank=8192, sequence_parallel=True, tensor_model_parallel_size=4
        ),
    ).enabled
    # Overrides.
    monkeypatch.setenv("MCORE_PADDING_AWARE_FORCE", "1")
    forced = pag.dense_linear_policy(
        _fake_module(out_features=512, in_features=256), _fake_config()
    )
    assert forced.enabled and not forced.wgrad_scratch  # 0.5 MiB: in-kernel accumulation
    monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", "accumulate")
    assert not pag.dense_linear_policy(big, _fake_config()).wgrad_scratch
    monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", "scratch")
    assert pag.dense_linear_policy(
        _fake_module(out_features=64, in_features=64), _fake_config()
    ).wgrad_scratch


def test_wgrad_scratches_share_one_arena():
    pag._wgrad_arenas.clear()
    pag._wgrad_scratches.clear()
    dev = torch.device("cuda", torch.cuda.current_device())
    small = pag._wgrad_scratch((64, 32), torch.float32, dev)
    big = pag._wgrad_scratch((256, 128), torch.float32, dev)
    small2 = pag._wgrad_scratch((64, 32), torch.float32, dev)
    assert small2.data_ptr() == big.data_ptr()  # the small view was rebuilt on the grown arena
    assert small2.shape == (64, 32) and big.shape == (256, 128)
    assert pag._wgrad_scratch((256, 128), torch.float32, dev) is big
    assert len(pag._wgrad_arenas) == 1
    assert pag._wgrad_arenas[(torch.float32, dev)].numel() == 256 * 128
    del small
    pag._wgrad_arenas.clear()
    pag._wgrad_scratches.clear()


def test_lm_head_wgrad_policy(monkeypatch):
    monkeypatch.delenv("MCORE_PADDING_AWARE_WGRAD", raising=False)
    assert pag.lm_head_wgrad_uses_grouped(16384)
    assert pag.lm_head_wgrad_uses_grouped(8192)
    assert not pag.lm_head_wgrad_uses_grouped(4096)
    monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", "grouped")
    assert pag.lm_head_wgrad_uses_grouped(1024)
    monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", "dense")
    assert not pag.lm_head_wgrad_uses_grouped(65536)


def test_layer_decorator_publishes_the_bound_of_the_padding_mask():
    T = 1024
    seen = {}

    class Layer:
        config = _fake_config(max_seqlen_per_dp_cp_rank=T)

        @pag.with_padding_aware_dense_linears
        def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
            seen["bound"] = pag.current_live_rows_for(T)
            return hidden_states

    mask = torch.zeros(1, T, dtype=torch.bool, device="cuda")
    mask[:, 700:] = True
    layer = Layer()
    layer._forward_mlp(torch.zeros(1), padding_mask=mask)
    assert seen["bound"].item() == 768
    layer._forward_mlp(torch.zeros(1), None, mask)  # positional padding_mask
    assert seen["bound"].item() == 768
    layer._forward_mlp(torch.zeros(1))
    assert seen["bound"] is None
    assert pag.current_live_rows() is None
    Layer.config.padding_aware_dense_linear = False
    layer._forward_mlp(torch.zeros(1), padding_mask=mask)
    assert seen["bound"] is None


# ------------------------------------------------------------------------------------------
# Transformer Engine linear layers (bf16 / MXFP8)
# ------------------------------------------------------------------------------------------

te = pytest.importorskip("transformer_engine.pytorch")


def _fp8_context(recipe):
    if recipe == "bf16":
        return nullcontext()
    from transformer_engine.common.recipe import MXFP8BlockScaling

    autocast = getattr(te, "autocast", None)
    if autocast is not None:
        return autocast(enabled=True, recipe=MXFP8BlockScaling())
    return te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling())


def _mxfp8_available():
    try:
        from transformer_engine.pytorch.fp8 import check_mxfp8_support

        ok, _ = check_mxfp8_support()
        return bool(ok)
    except Exception:  # pylint: disable=broad-except
        return torch.cuda.get_device_capability() >= (10, 0)


@pytest.fixture
def te_linears():
    """Two identical mcore ``TELinear`` layers (duplicated weights, no bias) on a THD capacity."""
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    # The test shapes are below the cost model's break-even; exercise the path regardless.
    os.environ["MCORE_PADDING_AWARE_FORCE"] = "1"

    def make(T, K, N, fused_wgrad, padding_aware=True):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=K,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=fused_wgrad,
            padding_aware_dense_linear=padding_aware,
            max_seqlen_per_dp_cp_rank=T,
        )
        torch.manual_seed(0)
        layers = []
        for _ in range(2):
            torch.manual_seed(0)
            layer = TELinear(
                K,
                N,
                parallel_mode="duplicated",
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
            ).cuda()
            layers.append(layer)
        with torch.no_grad():
            layers[1].weight.copy_(layers[0].weight)
        return layers

    yield make
    os.environ.pop("MCORE_PADDING_AWARE_FORCE", None)
    Utils.destroy_model_parallel()


def _spy_padding_aware_calls(monkeypatch, name="te_linear_padding_aware_forward"):
    from megatron.core.extensions import transformer_engine as te_ext

    calls = []
    original = getattr(te_ext, name)

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(te_ext, name, spy)
    return calls


@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
@pytest.mark.parametrize("fused_wgrad", [False, "scratch", "accumulate"])
def test_te_linear_padding_aware_matches_the_dense_path_on_live_rows(
    te_linears, recipe, fused_wgrad, monkeypatch
):
    if recipe == "mxfp8" and not _mxfp8_available():
        pytest.skip("MXFP8 not supported on this GPU")
    if fused_wgrad:
        monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", fused_wgrad)
    T, K, N, m = 4096, 1024, 2048, 2688  # m is a multiple of 128
    layer, ref = te_linears(T, K, N, bool(fused_wgrad))
    if recipe == "mxfp8":
        # Transformer Engine's first-microbatch flag is only trusted for quantized configs
        # (is_first_microbatch_tracked); mark this one quantized so the two-microbatch weight
        # cache is exercised as in training (the autocast recipe still comes from _fp8_context).
        layer.config.fp8 = "e4m3"
    assert layer._padding_aware_policy.enabled, layer._padding_aware_policy
    if fused_wgrad:
        assert layer._padding_aware_policy.wgrad_scratch == (fused_wgrad == "scratch")
    calls = _spy_padding_aware_calls(monkeypatch)
    if fused_wgrad:
        for mod in (layer, ref):
            mod.weight.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.float32)
            mod.weight.grad_added_to_main_grad = False
    torch.manual_seed(1)
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    x[m:] = 0
    dy = torch.randn(T, 1, N, device="cuda", dtype=torch.bfloat16)
    dy[m:] = 0
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)

    # Two microbatches: TE overwrites main_grad on the first (is_first_microbatch=True) and
    # accumulates on the second; the padding-aware path must reproduce both.
    for _ in range(2):
        x_pa = x.clone().requires_grad_()
        x_ref = x[:m].clone().requires_grad_()
        with _fp8_context(recipe), pag.padding_aware_live_rows(bound, T):
            out, _ = layer(x_pa)
        with _fp8_context(recipe):
            out_ref, _ = ref(x_ref)
        assert calls and calls[-1], "padding-aware path was not taken"
        assert out.shape == (T, 1, N)
        assert not out[m:].any()
        torch.testing.assert_close(out[:m], out_ref, rtol=2e-2, atol=2e-2)
        out.backward(dy)
        out_ref.backward(dy[:m])
        assert not x_pa.grad[m:].any()
        torch.testing.assert_close(x_pa.grad[:m], x_ref.grad, rtol=2e-2, atol=2e-2)

    if fused_wgrad:
        assert layer.weight.grad_added_to_main_grad
        torch.testing.assert_close(
            layer.weight.main_grad, ref.weight.main_grad, rtol=5e-3, atol=5e-2
        )
    else:
        torch.testing.assert_close(
            layer.weight.grad.float(), ref.weight.grad.float(), rtol=5e-3, atol=5e-2
        )
    if recipe == "mxfp8":
        # The quantized weight is cached across microbatches like on the dense path.
        assert "weight" in layer._fp8_workspaces


def test_te_linear_falls_back_without_a_bound_or_on_a_different_buffer(te_linears, monkeypatch):
    T, K, N = 4096, 256, 512
    layer, ref = te_linears(T, K, N, False)
    calls = _spy_padding_aware_calls(monkeypatch)
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    out, _ = layer(x)  # no context: dense path
    out_ref, _ = ref(x)
    assert torch.equal(out, out_ref) and not calls
    bound = torch.tensor([1024], device="cuda", dtype=torch.int64)
    with pag.padding_aware_live_rows(bound, total_rows=T // 2):  # bound for another buffer
        out, _ = layer(x)
    assert torch.equal(out, out_ref) and not calls
    disabled, _ = te_linears(2048, K, N, False)  # capacity below the policy minimum
    assert not disabled._padding_aware_policy.enabled


def test_te_linear_padding_aware_replays_in_a_cuda_graph_with_a_new_bound(te_linears):
    if not _mxfp8_available():
        pytest.skip("MXFP8 not supported on this GPU")
    T, K, N = 4096, 512, 1024
    layer, ref = te_linears(T, K, N, False)
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    bound = torch.tensor([2048], device="cuda", dtype=torch.int64)
    out = torch.empty(T, 1, N, device="cuda", dtype=torch.bfloat16)

    def step():
        with torch.no_grad(), _fp8_context("mxfp8"), pag.padding_aware_live_rows(bound, T):
            out.copy_(layer(x)[0])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    x.copy_(torch.randn_like(x))
    bound.fill_(1280)
    g.replay()
    torch.cuda.synchronize()
    with torch.no_grad(), _fp8_context("mxfp8"):
        out_ref, _ = ref(x[:1280])
    torch.testing.assert_close(out[:1280], out_ref, rtol=2e-2, atol=2e-2)
    assert not out[1280:].any()


# ------------------------------------------------------------------------------------------
# Transformer Engine fused norm + linear layers (TELayerNormColumnParallelLinear)
# ------------------------------------------------------------------------------------------


class _UnfusedNormLinear(torch.nn.Module):
    """Reference for the padding-aware fused-norm linear: TE norm (bf16 output) followed by a TE
    linear, i.e. the numerics the padding-aware path implements (the fused ``LayerNormLinear``
    quantizes its norm output straight from fp32, which differs by MXFP8 rounding)."""

    def __init__(self, norm, linear):
        super().__init__()
        self.norm = norm
        self.linear = linear

    @property
    def layer_norm_weight(self):
        return self.norm.weight

    @property
    def layer_norm_bias(self):
        return getattr(self.norm, "bias", None)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(self.norm(x))


@pytest.fixture
def te_layernorm_linears():
    """A mcore ``TELayerNormColumnParallelLinear`` (TP=1, no bias) and an unfused reference with
    the same parameters (TE norm + ``TELinear``)."""
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TELinear,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    os.environ["MCORE_PADDING_AWARE_FORCE"] = "1"

    def make(T, K, N, fused_wgrad, normalization="RMSNorm"):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=K,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=fused_wgrad,
            normalization=normalization,
            padding_aware_dense_linear=True,
            max_seqlen_per_dp_cp_rank=T,
        )
        torch.manual_seed(0)
        layer = TELayerNormColumnParallelLinear(
            K,
            N,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        norm_cls = te.RMSNorm if normalization == "RMSNorm" else te.LayerNorm
        norm = norm_cls(K, eps=layer.eps, params_dtype=torch.bfloat16, device="cuda")
        torch.manual_seed(0)
        linear = TELinear(
            K,
            N,
            parallel_mode="duplicated",
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        ).cuda()
        ref = _UnfusedNormLinear(norm, linear)
        with torch.no_grad():
            linear.weight.copy_(layer.weight)
            # Non-trivial norm parameters so that their gradients are checked for real.
            gamma = torch.linspace(0.5, 1.5, K, device="cuda")
            layer.layer_norm_weight.copy_(gamma)
            norm.weight.copy_(gamma)
            if getattr(layer, "layer_norm_bias", None) is not None:
                beta = torch.linspace(-0.1, 0.1, K, device="cuda")
                layer.layer_norm_bias.copy_(beta)
                norm.bias.copy_(beta)
        return layer, ref

    yield make
    os.environ.pop("MCORE_PADDING_AWARE_FORCE", None)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ("recipe", "normalization"), [("bf16", "RMSNorm"), ("bf16", "LayerNorm"), ("mxfp8", "RMSNorm")]
)
@pytest.mark.parametrize("fused_wgrad", [False, "scratch", "accumulate"])
def test_te_layernorm_linear_padding_aware_matches_the_dense_path_on_live_rows(
    te_layernorm_linears, recipe, normalization, fused_wgrad, monkeypatch
):
    if recipe == "mxfp8" and not _mxfp8_available():
        pytest.skip("MXFP8 not supported on this GPU")
    if fused_wgrad:
        monkeypatch.setenv("MCORE_PADDING_AWARE_WGRAD", fused_wgrad)
    T, K, N, m = 4096, 1024, 2048, 2688
    layer, ref = te_layernorm_linears(T, K, N, bool(fused_wgrad), normalization)
    if recipe == "mxfp8":
        # Transformer Engine's first-microbatch flag is only trusted for quantized configs
        # (is_first_microbatch_tracked); mark this one quantized so the two-microbatch weight
        # cache is exercised as in training (the autocast recipe still comes from _fp8_context).
        layer.config.fp8 = "e4m3"
    assert layer._padding_aware_policy.enabled, layer._padding_aware_policy
    calls = _spy_padding_aware_calls(monkeypatch, "te_layernorm_linear_padding_aware_forward")
    if fused_wgrad:
        for mod in (layer, ref):
            mod.weight.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.float32)
            mod.weight.grad_added_to_main_grad = False
    torch.manual_seed(1)
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    x[m:] = 0
    dy = torch.randn(T, 1, N, device="cuda", dtype=torch.bfloat16)
    dy[m:] = 0
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)

    for _ in range(2):
        x_pa = x.clone().requires_grad_()
        x_ref = x[:m].clone().requires_grad_()
        with _fp8_context(recipe), pag.padding_aware_live_rows(bound, T):
            out, _ = layer(x_pa)
        with _fp8_context(recipe):
            out_ref, _ = ref(x_ref)
        assert calls and calls[-1], "padding-aware path was not taken"
        assert out.shape == (T, 1, N)
        assert not out[m:].any()
        torch.testing.assert_close(out[:m], out_ref, rtol=2e-2, atol=2e-2)
        out.backward(dy)
        out_ref.backward(dy[:m])
        assert not x_pa.grad[m:].any()
        torch.testing.assert_close(x_pa.grad[:m], x_ref.grad, rtol=2e-2, atol=2e-2)

    # Norm parameter gradients reduce over the rows: the zero-filled tail must not change them.
    torch.testing.assert_close(
        layer.layer_norm_weight.grad.float(),
        ref.layer_norm_weight.grad.float(),
        rtol=2e-2,
        atol=5e-2,
    )
    if normalization == "LayerNorm":
        torch.testing.assert_close(
            layer.layer_norm_bias.grad.float(),
            ref.layer_norm_bias.grad.float(),
            rtol=2e-2,
            atol=5e-2,
        )
    if fused_wgrad:
        assert layer.weight.grad_added_to_main_grad
        torch.testing.assert_close(
            layer.weight.main_grad, ref.weight.main_grad, rtol=5e-3, atol=5e-2
        )
    else:
        torch.testing.assert_close(
            layer.weight.grad.float(), ref.weight.grad.float(), rtol=5e-3, atol=5e-2
        )
    if recipe == "mxfp8":
        assert "weight" in layer._fp8_workspaces


def test_te_layernorm_linear_padding_aware_replays_in_a_cuda_graph_with_a_new_bound(
    te_layernorm_linears,
):
    if not _mxfp8_available():
        pytest.skip("MXFP8 not supported on this GPU")
    T, K, N = 4096, 512, 1024
    layer, ref = te_layernorm_linears(T, K, N, False)
    x = torch.randn(T, 1, K, device="cuda", dtype=torch.bfloat16)
    bound = torch.tensor([2048], device="cuda", dtype=torch.int64)
    out = torch.empty(T, 1, N, device="cuda", dtype=torch.bfloat16)

    def step():
        with torch.no_grad(), _fp8_context("mxfp8"), pag.padding_aware_live_rows(bound, T):
            out.copy_(layer(x)[0])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    x.copy_(torch.randn_like(x))
    bound.fill_(1280)
    g.replay()
    torch.cuda.synchronize()
    with torch.no_grad(), _fp8_context("mxfp8"):
        out_ref, _ = ref(x[:1280])
    torch.testing.assert_close(out[:1280], out_ref, rtol=2e-2, atol=2e-2)
    assert not out[1280:].any()


# ------------------------------------------------------------------------------------------
# Fused hyper-connection (mHC) Triton kernels: rows beyond the bound are skipped and zero
# ------------------------------------------------------------------------------------------


def _mhc_inputs(T, n, C, m):
    torch.manual_seed(3)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    h_res = torch.randn(T, 1, n, n, **kw)
    orig = torch.randn(T, 1, n, C, **kw)
    h_post = torch.randn(T, 1, n, **kw)
    x = torch.randn(T, 1, C, **kw)
    go = torch.randn(T, 1, n, C, **kw)
    go[m:] = 0  # the padding rows carry no loss gradient
    return h_res, orig, h_post, x, go


def test_fused_mhc_kernels_skip_the_padding_tail():
    from megatron.core.fusions import fused_mhc_kernels as mhc
    from megatron.core.transformer.hyper_connection import native_h_aggregate, native_h_post_bda

    if mhc._get_triton_h_post_bda_fwd() is None or mhc._get_triton_h_aggregate_fwd() is None:
        pytest.skip("Triton mHC kernels unavailable")
    T, n, C, m = 4096, 4, 256, 2688
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    h_res, orig, h_post, x, go = _mhc_inputs(T, n, C, m)
    leaves = [t.clone().requires_grad_() for t in (h_res, orig, h_post, x)]
    ref_leaves = [t.clone().requires_grad_() for t in (h_res, orig, h_post, x)]

    with pag.padding_aware_live_rows(bound, T):
        out = mhc.fused_h_post_bda(*leaves, None)
    out_ref = native_h_post_bda(*ref_leaves, None)
    # The forward mixes the live rows and passes the original residual streams through for the
    # padding tail (finite and non-zero for the next layer's projected RMS, no mixing work);
    # the backward skips the padding rows. m is a multiple of every autotuned row block.
    torch.testing.assert_close(out[:m], out_ref[:m], rtol=2e-2, atol=2e-2)
    assert torch.equal(out[m:], orig[m:].to(out.dtype))
    out.backward(go)
    out_ref.backward(go)
    for leaf, ref in zip(leaves, ref_leaves):
        assert not leaf.grad[m:].any()
        torch.testing.assert_close(
            leaf.grad[:m].float(), ref.grad[:m].float(), rtol=2e-2, atol=2e-2
        )

    # Aggregation (n streams -> 1): forward skips the tail; the backward is the shared backend.
    x_n = orig.clone().requires_grad_()
    h_pre = h_post.clone().requires_grad_()
    with pag.padding_aware_live_rows(bound, T):
        agg = mhc.fused_h_aggregate(x_n, h_pre)
    agg_ref = native_h_aggregate(orig, h_post)
    assert not agg[m:].any()
    torch.testing.assert_close(agg[:m], agg_ref[:m], rtol=2e-2, atol=2e-2)
    g = torch.randn_like(agg)
    g[m:] = 0
    agg.backward(g)
    assert not x_n.grad[m:].any() and not h_pre.grad[m:].any()

    # Without a published bound the kernels run over every row.
    out_full = mhc.fused_h_post_bda(h_res, orig, h_post, x, None)
    torch.testing.assert_close(
        out_full, native_h_post_bda(h_res, orig, h_post, x, None), rtol=2e-2, atol=2e-2
    )


# ------------------------------------------------------------------------------------------
# Fused MLA RoPE (in place): rows beyond the bound are left untouched
# ------------------------------------------------------------------------------------------


def test_fused_mla_rope_inplace_skips_the_padding_tail():
    pytest.importorskip("triton")
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace

    T, H, nope, emb, m = 4096, 8, 64, 64, 2688
    kw = dict(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(5)
    t = torch.randn(T, H, nope + emb, **kw)
    t[m:] = 0
    cu_seqlens = torch.tensor([0, 1500, m], device="cuda", dtype=torch.int32)
    pos = torch.arange(T, device="cuda", dtype=torch.float32)[:, None]
    freqs = pos * (10000.0 ** (-torch.arange(0, emb, 2, device="cuda").float() / emb))[None, :]
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = freqs.cos().view(T, 1, 1, emb).to(torch.bfloat16)
    sin = freqs.sin().view(T, 1, 1, emb).to(torch.bfloat16)
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)

    x_pa = t.clone().requires_grad_()
    x_ref = t.clone().requires_grad_()
    with pag.padding_aware_live_rows(bound, T):
        out = fused_mla_rope_inplace(x_pa, cos, sin, nope, emb, cu_seqlens_q=cu_seqlens)
    out_ref = fused_mla_rope_inplace(x_ref, cos, sin, nope, emb, cu_seqlens_q=cu_seqlens)
    torch.testing.assert_close(out[:m], out_ref[:m])
    assert not out[m:].any()  # untouched (zeros in)
    go = torch.randn(T, H, nope + emb, **kw)
    go[m:] = 0
    # The backward un-rotates its incoming gradient in place: give each call its own copy.
    out.backward(go.clone())
    out_ref.backward(go.clone())
    torch.testing.assert_close(x_pa.grad[:m], x_ref.grad[:m])
    assert not x_pa.grad[m:].any()


# ------------------------------------------------------------------------------------------
# Grouped output projection (DSv4 attention) over strided groups
# ------------------------------------------------------------------------------------------


def test_padding_aware_grouped_projection_matches_einsum_on_live_rows():
    if not pag.HAVE_TE_GROUPED_LEADING_DIM:
        pytest.skip("Transformer Engine without grouped-tensor leading_dim support")
    T, G, d, r, m = 4096, 8, 512, 128, 2688
    kw = dict(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    x = torch.randn(T, 1, G, d, **kw)
    x[m:] = 0
    w = torch.randn(G, r, d, **kw) * 0.02
    bound = torch.tensor([m], device="cuda", dtype=torch.int64)
    x_pa = x.clone().requires_grad_()
    w_pa = w.clone().requires_grad_()
    x_ref = x[:m].clone().requires_grad_()
    w_ref = w.clone().requires_grad_()
    out = pag.padding_aware_grouped_projection(x_pa, w_pa, bound)
    out_ref = torch.einsum("...gd,grd->...gr", x_ref, w_ref)
    assert out.shape == (T, 1, G, r)
    assert not out[m:].any()
    torch.testing.assert_close(out[:m], out_ref, rtol=2e-2, atol=2e-2)
    go = torch.randn(T, 1, G, r, **kw)
    go[m:] = 0
    out.backward(go)
    out_ref.backward(go[:m])
    assert not x_pa.grad[m:].any()
    torch.testing.assert_close(x_pa.grad[:m], x_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(w_pa.grad.float(), w_ref.grad.float(), rtol=2e-2, atol=2e-1)
