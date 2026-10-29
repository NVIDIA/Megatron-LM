# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.flashinfer_nvfp4 import HAVE_FLASHINFER_ROUTED_NVFP4
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import deterministic_algorithms
from tests.unit_tests.test_utilities import Utils

_ACTIVATIONS = {
    "squared_relu": dict(activation_func=squared_relu, gated_linear_unit=False),
    "swiglu": dict(activation_func=F.silu, gated_linear_unit=True),
}


def _make_nvfp4_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_nvfp4_recipe="per_token",
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_flashinfer_nvfp4_config_accepts_moe(activation):
    config = _make_nvfp4_config(**_ACTIVATIONS[activation])
    assert config.inference_moe_nvfp4_recipe == "per_token"


_BATCH_INVARIANT_OVERRIDES = dict(
    batch_invariant_mode=True,
    params_dtype=torch.bfloat16,
    attention_backend=AttnBackend.flash,
    flash_attention_version=4,
    attention_dropout=0.0,
    expert_model_parallel_size=2,
    expert_tensor_parallel_size=1,
)


def test_flashinfer_nvfp4_config_accepts_batch_invariant_mode():
    config = _make_nvfp4_config(
        **_BATCH_INVARIANT_OVERRIDES, inference_moe_token_dispatcher_type="nvls"
    )
    assert config.batch_invariant_mode


def test_flashinfer_nvfp4_batch_invariant_mode_requires_nvls_with_ep():
    with pytest.raises(ValueError, match="requires inference_moe_token_dispatcher_type='nvls'"):
        _make_nvfp4_config(**_BATCH_INVARIANT_OVERRIDES, inference_moe_token_dispatcher_type="nccl")


_BOUNDED_OVERRIDES = dict(
    inference_flashinfer_mxfp8_token_capacity=64,
    inference_moe_token_dispatcher_type="nvls",
    expert_model_parallel_size=2,
    expert_tensor_parallel_size=1,
)


def test_flashinfer_nvfp4_config_accepts_token_capacity():
    config = _make_nvfp4_config(**_BOUNDED_OVERRIDES)
    assert config.inference_flashinfer_mxfp8_token_capacity == 64


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inference_moe_token_dispatcher_type": "nccl"}, "requires.*nvls"),
        ({"expert_model_parallel_size": 1}, "requires.*expert_model_parallel_size > 1"),
    ],
)
def test_flashinfer_nvfp4_token_capacity_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_nvfp4_config(**{**_BOUNDED_OVERRIDES, **overrides})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inference_grouped_gemm_backend": "torch"}, "requires inference_grouped_gemm_backend"),
        ({"activation_func": F.silu}, "supports only non-gated squared-ReLU"),
        ({"gated_linear_unit": True}, "supports only non-gated squared-ReLU"),
        (
            {**_ACTIVATIONS["swiglu"], "activation_func_clamp_value": 7.0},
            "supports only plain SwiGLU",
        ),
        ({**_ACTIVATIONS["swiglu"], "glu_linear_offset": 1.0}, "supports only plain SwiGLU"),
        (
            {**_ACTIVATIONS["swiglu"], "moe_mlp_glu_interleave_size": 64},
            "supports only plain SwiGLU",
        ),
        ({"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True}, "cannot be combined"),
        ({"transformer_impl": "transformer_engine"}, "requires transformer_impl"),
        ({"activation_func_tanh_clamp_scale": 10.0}, "does not support activation_func_tanh"),
    ],
)
def test_flashinfer_nvfp4_config_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_nvfp4_config(**overrides)


def test_missing_routed_nvfp4_capability_has_precise_error(monkeypatch):
    from megatron.core.inference.moe import flashinfer_nvfp4

    monkeypatch.setattr(flashinfer_nvfp4, "HAVE_FLASHINFER_ROUTED_NVFP4", False)
    monkeypatch.setattr(
        flashinfer_nvfp4,
        "_FLASHINFER_NVFP4_IMPORT_ERROR",
        ImportError("missing per-token NVFP4 API"),
    )

    with pytest.raises(RuntimeError, match="requires FlashInfer >= 0.7.0"):
        flashinfer_nvfp4.require_flashinfer_routed_nvfp4()


def test_flashinfer_nvfp4_refresh_reports_noop_before_weight_build():
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    grouped_mlp = SimpleNamespace(_nvfp4_weights=None)

    assert InferenceGroupedMLP.refresh_flashinfer_nvfp4_weights(grouped_mlp) is False


def test_flashinfer_nvfp4_refresh_requantizes_in_place(monkeypatch):
    from megatron.core.transformer.moe import experts

    captured = {}

    def prepare(fc1_weight, fc2_weight, out=None):
        captured.update(fc1=fc1_weight, fc2=fc2_weight, out=out)
        return out

    monkeypatch.setattr(experts, "prepare_routed_nvfp4_weights", prepare)
    nvfp4_weights = object()
    grouped_mlp = SimpleNamespace(
        _nvfp4_weights=nvfp4_weights,
        _fc1_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
        _fc2_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
    )

    assert experts.InferenceGroupedMLP.refresh_flashinfer_nvfp4_weights(grouped_mlp) is True
    assert captured["out"] is nvfp4_weights
    assert captured["fc1"] is grouped_mlp._fc1_weight
    assert captured["fc2"] is grouped_mlp._fc2_weight


@pytest.mark.parametrize("built", [False, True], ids=["before_first_forward", "after"])
def test_refit_cache_refresh_requantizes_nvfp4(monkeypatch, built):
    """Every refit path ends in refresh_module_caches; it must reach the NVFP4 stacks."""
    from megatron.core.resharding.execution import refresh_module_caches
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    calls, syncs = [], []

    def _refresh(self):
        calls.append(self)
        return built

    monkeypatch.setattr(InferenceGroupedMLP, "refresh_flashinfer_nvfp4_weights", _refresh)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(True))
    grouped_mlp = InferenceGroupedMLP.__new__(InferenceGroupedMLP)
    torch.nn.Module.__init__(grouped_mlp)

    refresh_module_caches([torch.nn.Sequential(grouped_mlp)])

    assert calls == [grouped_mlp]
    assert syncs == ([True] if built else [])


def test_flashinfer_forward_dispatches_to_nvfp4(monkeypatch):
    from megatron.core.transformer.moe import experts

    expected = torch.empty(4, 8, dtype=torch.bfloat16)
    captured = {}

    def routed_nvfp4_moe(hidden, routing_map, probs, weights, **kwargs):
        captured.update(weights=weights, **kwargs)
        return expected

    monkeypatch.setattr(experts, "HAVE_FLASHINFER", True)
    monkeypatch.setattr(experts, "flashinfer_routed_nvfp4_moe", routed_nvfp4_moe)
    monkeypatch.setattr(
        experts,
        "fused_moe",
        SimpleNamespace(cutlass_fused_moe=lambda *a, **k: pytest.fail("BF16 path selected")),
        raising=False,
    )
    nvfp4_weights = object()
    grouped_mlp = SimpleNamespace(
        _uses_mxfp8_weights=False,
        _nvfp4_weights=nvfp4_weights,
        _activation_clamp_scale=None,
        _flashinfer_activation_type=SimpleNamespace(value=7),
        _nvls_dispatcher=None,
        _flashinfer_mxfp8_token_capacity=64,
        num_local_experts=2,
        ep_group=SimpleNamespace(size=lambda: 4, rank=lambda: 3),
    )
    grouped_mlp._flashinfer_routed_forward = functools.partial(
        experts.InferenceGroupedMLP._flashinfer_routed_forward, grouped_mlp
    )
    monkeypatch.setattr(experts.InferenceMode, "_use_bounded_mxfp8_rows", True)

    output, bias = experts.InferenceGroupedMLP._flashinfer_forward(
        grouped_mlp,
        torch.empty(4, 8, dtype=torch.bfloat16),
        torch.zeros(4, 1, dtype=torch.int64),
        torch.zeros(4, 1, dtype=torch.float32),
    )

    assert output is expected
    assert bias is None
    assert captured["weights"] is nvfp4_weights
    assert captured["num_experts"] == 8
    assert captured["local_expert_offset"] == 6
    assert captured["activation_type"] == 7
    assert captured["out"] is None
    assert captured["token_capacity"] == 64
    assert captured["use_bounded_rows"] is True


@pytest.mark.parametrize("built", [False, True], ids=["before_first_forward", "after"])
def test_refit_accepts_built_nvfp4_expert_weights(monkeypatch, built):
    """Refit runs whether or not the NVFP4 copy exists yet.

    Before the first forward there is no copy; it is built from the refit weights.
    After it, execute_reshard_plan's refresh_module_caches requantizes it in place.
    """
    from megatron.core.resharding import refit

    experts_module = torch.nn.Module()
    experts_module._nvfp4_weights = object() if built else None
    target = torch.nn.Sequential(experts_module)
    executed = []
    monkeypatch.setattr(refit, "_unwrap_model_cores", lambda src, tgt: (None, tgt, 2))
    monkeypatch.setattr(refit, "_build_or_get_plan", lambda *args, **kwargs: "plan")
    monkeypatch.setattr(refit, "_harmonize_buffer_dtypes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        refit, "execute_reshard_plan", lambda plan, *args, **kwargs: executed.append(plan)
    )
    service = SimpleNamespace(set_model_roles=lambda **kwargs: None)

    refit.reshard_model_weights(None, target, service)
    assert executed == ["plan"]


def test_gated_fc1_is_padded_per_half_and_interleaved_up_gate():
    """MCore's [gate; up] FC1 becomes FlashInfer's alternating 64-row [up, gate] blocks."""
    from megatron.core.inference.moe.flashinfer_nvfp4 import _pad_fc1

    intermediate, hidden, padded_i, padded_h = 96, 40, 256, 256
    gate = torch.arange(intermediate * hidden, dtype=torch.float32).view(intermediate, hidden) + 1
    up = -gate
    padded = _pad_fc1(torch.cat([gate, up]).bfloat16(), True, padded_i, padded_h)

    def _half(x):
        out = torch.zeros(padded_i, padded_h, dtype=torch.bfloat16)
        out[:intermediate, :hidden] = x.bfloat16()
        return out

    expected = torch.cat(
        [
            torch.cat([_half(up)[rows], _half(gate)[rows]])
            for rows in torch.arange(padded_i).split(64)
        ]
    )
    assert padded.shape == (2 * padded_i, padded_h)
    assert torch.equal(padded, expected)


def test_validate_stacks_infers_gating_from_shapes():
    from megatron.core.inference.moe.flashinfer_nvfp4 import _validate_stacks

    fc2 = torch.empty(2, 64, 48)
    assert _validate_stacks(torch.empty(2, 48, 64), fc2) == (2, 48, 64, False)
    assert _validate_stacks(torch.empty(2, 96, 64), fc2) == (2, 48, 64, True)
    with pytest.raises(ValueError, match="does not match FC2 shape"):
        _validate_stacks(torch.empty(2, 72, 64), fc2)


def _blackwell_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


requires_blackwell = pytest.mark.skipif(
    not _blackwell_or_newer(), reason="per-token NVFP4 MoE requires a Blackwell GPU"
)

requires_flashinfer_nvfp4 = pytest.mark.skipif(
    not HAVE_FLASHINFER_ROUTED_NVFP4,
    reason="FlashInfer >= 0.7.0 per-token NVFP4 MoE is unavailable",
)


def _build_flashinfer_nvfp4_jit_modules():
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        flashinfer_routed_nvfp4_moe,
        prepare_routed_nvfp4_weights,
    )

    experts, hidden_size, tokens = 2, 256, 8
    device = torch.device("cuda")
    fc1 = torch.randn(experts, hidden_size, hidden_size, device=device).bfloat16()
    fc2 = torch.randn(experts, hidden_size, hidden_size, device=device).bfloat16()
    hidden = torch.randn(tokens, hidden_size, device=device).bfloat16()
    topk_ids = torch.arange(experts, dtype=torch.int32, device=device).repeat(tokens, 1)
    topk_weights = torch.full((tokens, experts), 1.0 / experts, device=device)
    flashinfer_routed_nvfp4_moe(
        hidden,
        topk_ids,
        topk_weights,
        prepare_routed_nvfp4_weights(fc1, fc2),
        num_experts=experts,
        local_expert_offset=0,
        activation_type=_activation_type(False),
    )
    torch.cuda.synchronize()


@pytest.fixture(scope="module", autouse=True)
def build_flashinfer_jit_in_rank_order():
    """Builds FlashInfer's JIT modules one rank at a time before any test runs.

    A cold build takes minutes and FlashInfer serializes it across processes, so ranks that
    build inside the tests reach the multi-rank setups minutes apart, while the test
    rendezvous waits one minute. Taking turns bounds each barrier wait to a single build,
    and later ranks load the modules from the shared cache.
    """
    if Utils.world_size < 2 or not _blackwell_or_newer() or not HAVE_FLASHINFER_ROUTED_NVFP4:
        yield
        return
    Utils.initialize_distributed()
    for rank in range(Utils.world_size):
        if rank == Utils.rank:
            _build_flashinfer_nvfp4_jit_modules()
        torch.distributed.barrier()
    yield


def _reference_moe(hidden, fc1, fc2, topk_ids, topk_weights):
    """BF16 squared-ReLU MoE, or SwiGLU when FC1 holds MCore's [gate; up] halves."""
    gated = fc1.shape[1] == 2 * fc2.shape[2]
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for slot in range(topk_ids.shape[1]):
        for expert in range(fc1.shape[0]):
            rows = topk_ids[:, slot] == expert
            if not rows.any():
                continue
            fc1_out = hidden[rows].float() @ fc1[expert].float().t()
            if gated:
                gate, up = fc1_out.chunk(2, dim=-1)
                intermediate = F.silu(gate) * up
            else:
                intermediate = F.relu(fc1_out).square()
            expert_out = intermediate @ fc2[expert].float().t()
            output[rows] += topk_weights[rows, slot : slot + 1] * expert_out
    return output


def _activation_type(gated):
    from flashinfer import ActivationType

    return (ActivationType.Swiglu if gated else ActivationType.Relu2).value


def _assert_close_to_reference(output, reference):
    # Exact NVFP4 emulation of these shapes gives per-token cosine ~0.98 mean and ~0.965 at
    # worst; layout or scale bugs collapse it far below these bounds.
    cosine = F.cosine_similarity(output.float(), reference, dim=-1)
    assert cosine.mean().item() > 0.97, cosine.mean().item()
    assert cosine.min().item() > 0.95, cosine.min().item()


# E2M1 code -> value; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_VALUES = _E2M1_MAGNITUDES + tuple(-value for value in _E2M1_MAGNITUDES)
# (upper bound, bound inclusive, value): round-half-to-even onto the E2M1 grid.
_E2M1_ROUNDING = (
    (0.25, True, 0.0),
    (0.75, False, 0.5),
    (1.25, True, 1.0),
    (1.75, False, 1.5),
    (2.5, True, 2.0),
    (3.5, False, 3.0),
    (5.0, True, 4.0),
)


def _ref_round_to_e2m1(x):
    magnitude = x.abs()
    out = torch.full_like(magnitude, 6.0)
    for bound, inclusive, value in reversed(_E2M1_ROUNDING):
        below = magnitude <= bound if inclusive else magnitude < bound
        out = torch.where(below, torch.full_like(out, value), out)
    return out * torch.sign(x)


def _ref_per_token_nvfp4_quantize(x):
    """Pure-PyTorch per-token NVFP4 quantizer mirroring Transformer Engine's reference.

    Returns E2M1 values ``[rows, cols]`` (FP32), E4M3 block decode scales
    ``[rows, cols // 16]`` and the FP32 per-token decode scales ``[rows]``.
    """
    rows, cols = x.shape
    blocks = x.float().view(rows, cols // 16, 16)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    row_amax = block_amax.amax(dim=1, keepdim=True)

    # Tensor-by-tensor division: a Python scalar numerator becomes reciprocal-then-multiply.
    def const(value):
        return torch.tensor(value, dtype=torch.float32, device=x.device)

    fp32_max = const(torch.finfo(torch.float32).max)
    global_encode = torch.minimum(torch.div(const(448.0 * 6.0), row_amax), fp32_max)
    global_encode = torch.where(global_encode == 0, torch.ones_like(global_encode), global_encode)
    global_decode = torch.div(const(1.0), global_encode)
    decode = torch.minimum(block_amax * (global_encode * torch.reciprocal(const(6.0))), fp32_max)
    decode = decode.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    encode = torch.minimum(torch.div(const(1.0), decode.float() * global_decode), fp32_max)
    values = _ref_round_to_e2m1((blocks * encode).clamp(-6.0, 6.0)).reshape(rows, cols)
    # An all-zero row reports a zero per-token scale.
    per_token_decode = torch.where(row_amax == 0, torch.zeros_like(global_decode), global_decode)
    return values, decode.squeeze(-1), per_token_decode.reshape(-1)


def _unpack_e2m1(packed):
    """Decode packed E2M1 bytes ``[rows, cols // 2]``; the low nibble is the even element."""
    lookup = torch.tensor(_E2M1_VALUES, device=packed.device)
    packed = packed.view(torch.uint8).long()
    return torch.stack((lookup[packed & 0xF], lookup[packed >> 4]), dim=-1).flatten(-2)


@pytest.mark.launch_on_gb200
@requires_blackwell
@requires_flashinfer_nvfp4
@pytest.mark.parametrize("tokens", [1, 37, 256])
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_per_token_nvfp4_quantize_matches_reference(monkeypatch, tokens, hidden_size):
    """FlashInfer's per-token activation quantizer matches the PyTorch reference bit-exactly.

    The default fast-math reciprocal is approximate; IEEE rounding is what a training-side
    quantizer reproduces, so it is the mode compared bit-exactly here.
    """
    from megatron.core.inference.moe.flashinfer_nvfp4 import quantize_routed_nvfp4_input

    monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")
    torch.manual_seed(tokens + hidden_size)
    scale = torch.logspace(-3, 2, tokens, device="cuda").unsqueeze(1)
    hidden = (torch.randn(tokens, hidden_size, device="cuda") * scale).bfloat16()
    if tokens > 1:
        # Padded dispatcher rows are all zero.
        hidden[-1].zero_()

    data, block_scale, per_token_scale = quantize_routed_nvfp4_input(hidden, hidden_size)
    values, decode, expected_per_token = _ref_per_token_nvfp4_quantize(hidden)

    torch.testing.assert_close(per_token_scale.reshape(-1), expected_per_token, atol=0, rtol=0)
    assert torch.equal(block_scale.view(torch.uint8), decode.view(torch.uint8))
    assert torch.equal(_unpack_e2m1(data).reshape(tokens, hidden_size), values)


@pytest.mark.launch_on_gb200
@requires_blackwell
@requires_flashinfer_nvfp4
@pytest.mark.parametrize("gated", [False, True], ids=["squared_relu", "swiglu"])
@pytest.mark.parametrize(("hidden_size", "intermediate_size"), [(512, 512), (320, 384)])
def test_flashinfer_routed_nvfp4_moe_matches_bf16_reference(hidden_size, intermediate_size, gated):
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        flashinfer_routed_nvfp4_moe,
        prepare_routed_nvfp4_weights,
    )

    torch.manual_seed(0)
    experts, top_k, tokens = 4, 2, 64
    device = torch.device("cuda")
    fc1_rows = 2 * intermediate_size if gated else intermediate_size
    fc1 = torch.randn(experts, fc1_rows, hidden_size, device=device) * 0.05
    fc2 = torch.randn(experts, hidden_size, intermediate_size, device=device) * 0.05
    fc1, fc2 = fc1.bfloat16(), fc2.bfloat16()
    # Spread per-token magnitudes so a single tensor-wide activation scale would underflow.
    token_scale = torch.logspace(-2, 1, tokens, device=device).unsqueeze(1)
    hidden = (torch.randn(tokens, hidden_size, device=device) * token_scale).bfloat16()
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1)

    weights = prepare_routed_nvfp4_weights(fc1, fc2)
    assert weights.gated == gated
    output = flashinfer_routed_nvfp4_moe(
        hidden,
        topk_ids,
        topk_weights,
        weights,
        num_experts=experts,
        local_expert_offset=0,
        activation_type=_activation_type(gated),
    )
    reference = _reference_moe(hidden, fc1, fc2, topk_ids, topk_weights)

    assert output.shape == (tokens, hidden_size)
    _assert_close_to_reference(output, reference)

    # In-place refresh keeps addresses stable and tracks new canonical weights.
    data_ptr = weights.fc1_data.data_ptr()
    refreshed = prepare_routed_nvfp4_weights(-fc1, fc2, out=weights)
    assert refreshed is weights
    assert weights.fc1_data.data_ptr() == data_ptr
    output_negated = flashinfer_routed_nvfp4_moe(
        hidden,
        topk_ids,
        topk_weights,
        weights,
        num_experts=experts,
        local_expert_offset=0,
        activation_type=_activation_type(gated),
    )
    reference_negated = _reference_moe(hidden, -fc1, fc2, topk_ids, topk_weights)
    _assert_close_to_reference(output_negated, reference_negated)

    with pytest.raises(ValueError, match="need activation_type"):
        flashinfer_routed_nvfp4_moe(
            hidden,
            topk_ids,
            topk_weights,
            weights,
            num_experts=experts,
            local_expert_offset=0,
            activation_type=_activation_type(not gated),
        )


@pytest.mark.launch_on_gb200
@requires_blackwell
@requires_flashinfer_nvfp4
@pytest.mark.parametrize("gated", [False, True], ids=["squared_relu", "swiglu"])
@pytest.mark.parametrize("mode", ["deterministic", "batch_invariant"])
@pytest.mark.parametrize("local_expert_offset", [0, 2, 4, 6])
def test_flashinfer_routed_nvfp4_is_batch_invariant(local_expert_offset, mode, gated):
    """A token's local-EP output must not depend on its batch or row position."""
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        flashinfer_routed_nvfp4_moe,
        prepare_routed_nvfp4_weights,
    )
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    torch.manual_seed(1234)
    num_experts, local_experts, hidden_size, intermediate_size = 8, 2, 512, 512
    num_tokens, topk = 512, 2
    local = slice(local_expert_offset, local_expert_offset + local_experts)
    fc1_rows = 2 * intermediate_size if gated else intermediate_size
    weights = prepare_routed_nvfp4_weights(
        (torch.randn(num_experts, fc1_rows, hidden_size, device="cuda") * 0.05)[local]
        .bfloat16()
        .contiguous(),
        (torch.randn(num_experts, hidden_size, intermediate_size, device="cuda") * 0.05)[local]
        .bfloat16()
        .contiguous(),
    )
    target = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    target_experts = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
    target_probs = torch.tensor([0.625, 0.375], device="cuda", dtype=torch.float32)

    def _run(active_tokens: int, target_row: int) -> torch.Tensor:
        # Neighbouring rows are 100x larger, which would dominate a tensor-wide scale,
        # and share expert 2 with the target so its per-expert row count changes.
        hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 100
        routing_map = torch.full((num_tokens, topk), -1, device="cuda", dtype=torch.int32)
        routing_map[:active_tokens, 0] = 2
        routing_map[:active_tokens, 1] = 3
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        probs[:active_tokens].fill_(0.5)
        hidden[target_row].copy_(target)
        routing_map[target_row].copy_(target_experts)
        probs[target_row].copy_(target_probs)
        return flashinfer_routed_nvfp4_moe(
            hidden,
            routing_map,
            probs,
            weights,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            activation_type=_activation_type(gated),
        )[target_row]

    context = (
        deterministic_algorithms(True)
        if mode == "deterministic"
        else set_batch_invariant_mode(True)
    )
    with torch.no_grad(), context:
        output_alone = _run(1, 0)
        output_batched = _run(128, 73)

    assert torch.equal(output_alone, output_batched), (
        "per-token NVFP4 output changed with the batch; max abs diff: "
        f"{(output_alone.float() - output_batched.float()).abs().max().item()}"
    )
    if not any(local.start <= expert < local.stop for expert in target_experts.tolist()):
        assert torch.count_nonzero(output_alone).item() == 0


@pytest.mark.launch_on_gb200
@requires_blackwell
@requires_flashinfer_nvfp4
@pytest.mark.parametrize("gated", [False, True], ids=["squared_relu", "swiglu"])
def test_flashinfer_routed_nvfp4_bounded_rows_match_full_buffer(gated):
    """A bounded decode prefix gives the full buffer's rows and leaves the rest of out alone."""
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        flashinfer_routed_nvfp4_moe,
        prepare_routed_nvfp4_weights,
    )
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    torch.manual_seed(1235)
    num_experts, hidden_size, intermediate_size = 4, 512, 512
    full_rows, capacity, active_tokens, topk = 512, 64, 40, 2
    fc1_rows = 2 * intermediate_size if gated else intermediate_size
    weights = prepare_routed_nvfp4_weights(
        (torch.randn(num_experts, fc1_rows, hidden_size, device="cuda") * 0.05).bfloat16(),
        (torch.randn(num_experts, hidden_size, intermediate_size, device="cuda") * 0.05).bfloat16(),
    )
    hidden = torch.randn(full_rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    # The dispatcher marks rows without a token -1, inside and past the prefix.
    routing_map = torch.full((full_rows, topk), -1, device="cuda", dtype=torch.int32)
    routing_map[:active_tokens] = torch.randint(
        num_experts, (active_tokens, topk), device="cuda", dtype=torch.int32
    )
    probs = torch.rand(full_rows, topk, device="cuda", dtype=torch.float32)

    def _run(**bounds):
        out = torch.full((full_rows, hidden_size), 7.0, device="cuda", dtype=torch.bfloat16)
        flashinfer_routed_nvfp4_moe(
            hidden,
            routing_map,
            probs,
            weights,
            num_experts=num_experts,
            local_expert_offset=0,
            activation_type=_activation_type(gated),
            out=out,
            **bounds,
        )
        return out

    with torch.no_grad(), set_batch_invariant_mode(True):
        full = _run()
        bounded = _run(token_capacity=capacity, use_bounded_rows=True)
        unbounded = _run(token_capacity=capacity, use_bounded_rows=False)

    assert torch.equal(bounded[:capacity], full[:capacity])
    assert torch.all(bounded[capacity:] == 7.0), "rows past the prefix were written"
    assert torch.equal(unbounded, full), "use_bounded_rows=False must process the full buffer"


# ──────────────────────────────────────────────────────────────────────
# MoE layer level: InferenceGroupedMLP + inference dispatchers + refresh
# ──────────────────────────────────────────────────────────────────────

_NVLS_MAX_TOKENS = 512


def _make_nvfp4_layer_config(**overrides):
    """Scaled-down Nemotron-H MoE: sigmoid router with expert bias and shared experts."""
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        # Not a multiple of 256, so the intermediate dimension is padded.
        moe_ffn_hidden_size=384,
        moe_router_topk=6,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=2.5,
        moe_shared_expert_intermediate_size=256,
        moe_shared_expert_overlap=False,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        activation_func=squared_relu,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=Utils.world_size,
        use_cpu_initialization=True,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_nvfp4_recipe="per_token",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@requires_blackwell
@requires_flashinfer_nvfp4
class TestNVFP4MoELayer:

    @classmethod
    def setup_class(cls):
        from megatron.training.initialize import _set_random_seed

        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )

        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _make_layer(self, dispatcher, activation):
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
        from megatron.core.parallel_state import get_expert_model_parallel_group
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NCCLAllGatherDispatcher,
            NVLSAllGatherVDispatcher,
        )

        if dispatcher == "nvls":
            if Utils.world_size < 2 or Utils.world_size & (Utils.world_size - 1):
                pytest.skip("NVLS dispatcher requires a power-of-two EP size > 1.")
        config = _make_nvfp4_layer_config(
            inference_moe_token_dispatcher_type=dispatcher, **_ACTIVATIONS[activation]
        )
        if dispatcher == "nvls":
            NVLSAllGatherVDispatcher.allocate_buffers(
                per_rank_worst_case_token_count=_NVLS_MAX_TOKENS,
                topk=config.moe_router_topk,
                hidden_size=config.hidden_size,
                tp_ep_group=get_expert_model_parallel_group(),
            )
        else:
            NCCLAllGatherDispatcher.allocate_buffers()
        return get_inference_optimized_moe_spec()(config=config).cuda().eval()

    @staticmethod
    def _hidden_states(local_tokens, hidden_size):
        # Spread per-token magnitudes over three decades so the per-token scale matters.
        scale = torch.logspace(-1, 2, local_tokens, device="cuda").view(local_tokens, 1, 1)
        hidden = torch.randn(local_tokens, 1, hidden_size, device="cuda") * scale
        return hidden.to(torch.bfloat16)

    @staticmethod
    def _assert_close_to_bf16(inference_output, training_output, activation):
        cosine = F.cosine_similarity(
            inference_output.float().flatten(0, 1), training_output.float().flatten(0, 1), dim=-1
        )
        # SwiGLU multiplies two NVFP4 projections, so it lands near 0.978 where squared ReLU
        # exceeds 0.98; a wrong gate/up layout drops far below either bound.
        min_mean = 0.97 if activation == "swiglu" else 0.98
        assert cosine.mean().item() > min_mean, cosine.mean().item()
        assert cosine.min().item() > 0.9, cosine.min().item()

    @pytest.mark.parametrize("activation", _ACTIVATIONS)
    @pytest.mark.parametrize("dispatcher", ["nccl", "nvls"])
    def test_nvfp4_moe_layer_matches_bf16_training_path(self, dispatcher, activation):
        layer = self._make_layer(dispatcher, activation)
        # NCCL AllGather (non-v) requires equal token counts on every rank.
        local_tokens = 32 if dispatcher == "nccl" else 17 + torch.distributed.get_rank()
        hidden_states = self._hidden_states(local_tokens, layer.config.hidden_size)

        with torch.no_grad():
            training_output, _ = layer(hidden_states.clone())
            with InferenceMode.active():
                inference_output, _ = layer(hidden_states.clone())

        nvfp4_weights = layer.experts._nvfp4_weights
        assert nvfp4_weights.padded_hidden_size == 256
        assert nvfp4_weights.padded_intermediate_size == 512
        assert nvfp4_weights.gated == (activation == "swiglu")
        assert inference_output.shape == training_output.shape
        assert inference_output.dtype == torch.bfloat16
        assert not torch.equal(inference_output, training_output), "NVFP4 path did not run"
        self._assert_close_to_bf16(inference_output, training_output, activation)

    @pytest.mark.parametrize("activation", _ACTIVATIONS)
    @pytest.mark.parametrize("dispatcher", ["nccl", "nvls"])
    def test_nvfp4_moe_layer_refresh_tracks_bf16_weight_updates(self, dispatcher, activation):
        """Simulates an optimizer step on a colocated model followed by a refresh."""
        layer = self._make_layer(dispatcher, activation)
        local_tokens = 32
        hidden_states = self._hidden_states(local_tokens, layer.config.hidden_size)

        # Deterministic so the stale replay can be compared bit-exactly.
        with torch.no_grad(), deterministic_algorithms(True):
            with InferenceMode.active():
                before, _ = layer(hidden_states.clone())
            nvfp4_weights = layer.experts._nvfp4_weights
            fc2_data_ptr = nvfp4_weights.fc2_data.data_ptr()

            # Expert parameters alias the BF16 stacks, so this is what the optimizer sees.
            layer.experts._fc1_weight.mul_(-1)
            training_output, _ = layer(hidden_states.clone())
            with InferenceMode.active():
                stale, _ = layer(hidden_states.clone())
            torch.testing.assert_close(stale, before, atol=0, rtol=0)

            assert layer.experts.refresh_flashinfer_nvfp4_weights() is True
            with InferenceMode.active():
                refreshed, _ = layer(hidden_states.clone())

        assert layer.experts._nvfp4_weights is nvfp4_weights
        assert nvfp4_weights.fc2_data.data_ptr() == fc2_data_ptr
        assert not torch.equal(refreshed, before)
        self._assert_close_to_bf16(refreshed, training_output, activation)

    @pytest.mark.parametrize("activation", _ACTIVATIONS)
    @pytest.mark.parametrize("dispatcher", ["nccl", "nvls"])
    def test_nvfp4_moe_layer_cuda_graph_replay_tracks_inputs_and_refresh(
        self, dispatcher, activation
    ):
        """A captured layer replays like eager, and an in-place refresh reaches the graph."""
        layer = self._make_layer(dispatcher, activation)
        local_tokens = 32 if dispatcher == "nccl" else 17 + torch.distributed.get_rank()
        hidden_size = layer.config.hidden_size
        static_hidden = self._hidden_states(local_tokens, hidden_size)

        def _eager(hidden):
            output, _ = layer(hidden.clone())
            return output

        with torch.no_grad(), deterministic_algorithms(True), InferenceMode.active():
            # The first forward builds the NVFP4 weights and kernel wrappers eagerly.
            _eager(static_hidden)
            nvfp4_weights = layer.experts._nvfp4_weights
            fc1_data_ptr = nvfp4_weights.fc1_data.data_ptr()

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    layer(static_hidden)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output, _ = layer(static_hidden)

            static_hidden.copy_(self._hidden_states(local_tokens, hidden_size).flip(0))
            graph.replay()
            torch.testing.assert_close(graph_output, _eager(static_hidden), atol=0, rtol=0)
            before_refresh = graph_output.clone()

            layer.experts._fc1_weight.mul_(-1)
            assert layer.experts.refresh_flashinfer_nvfp4_weights() is True
            graph.replay()
            torch.testing.assert_close(graph_output, _eager(static_hidden), atol=0, rtol=0)

        assert layer.experts._nvfp4_weights is nvfp4_weights
        assert nvfp4_weights.fc1_data.data_ptr() == fc1_data_ptr
        assert not torch.equal(graph_output, before_refresh), "refresh did not reach the graph"
