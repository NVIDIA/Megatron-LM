# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.inference.moe import flashinfer_mxfp8
from megatron.core.inference.moe.flashinfer_mxfp8 import (
    HAVE_FLASHINFER_ROUTED_MXFP8,
    HAVE_TRITON,
    FlashInferRoutedMXFP8Weight,
    flashinfer_routed_mxfp8_moe,
    select_routed_mxfp8_active_rows,
)
from megatron.core.inference.quantization.mxfp8_quantize import quantize_routed_mxfp8_input
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

requires_cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="CUDA and triton required"
)


@pytest.fixture(autouse=True)
def reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def _mxfp8_weight():
    return MXFP8Tensor(
        data=torch.empty(1, dtype=torch.float8_e4m3fn),
        scale=torch.empty(1, dtype=torch.uint8),
        backend="triton",
    )


def _valid(n: int) -> torch.Tensor:
    return torch.tensor([n], dtype=torch.int32, device="cuda")


def test_inference_mode_tracks_bounded_mxfp8_rows():
    InferenceMode.set_active()
    assert not InferenceMode.use_bounded_mxfp8_rows()

    InferenceMode.set_bounded_mxfp8_rows(True)
    assert InferenceMode.use_bounded_mxfp8_rows()

    InferenceMode.unset_active()
    assert not InferenceMode.use_bounded_mxfp8_rows()


@pytest.mark.parametrize(
    ("token_capacity", "use_bounded_rows", "expected"),
    [
        (None, False, (65536, "full")),
        (1024, False, (65536, "full")),
        (1024, True, (1024, "bounded-decode")),
        (131072, True, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_mxfp8_active_row_policy(token_capacity, use_bounded_rows, expected):
    assert (
        select_routed_mxfp8_active_rows(
            65536, token_capacity=token_capacity, use_bounded_rows=use_bounded_rows
        )
        == expected
    )


def _make_bounded_mxfp8_config(**overrides):
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
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_mxfp8_token_capacity=1024,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        activation_func=squared_relu,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_bounded_flashinfer_mxfp8_config_accepts_nvls_ep():
    config = _make_bounded_mxfp8_config()

    assert config.inference_moe_token_dispatcher_type == "nvls"
    assert config.expert_model_parallel_size == 2


def test_flashinfer_mxfp8_token_capacity_warns_that_mxfp8_ignores_it():
    with pytest.warns(UserWarning, match="ignored by FlashInfer routed MXFP8"):
        _make_bounded_mxfp8_config()


def test_flashinfer_mxfp8_config_accepts_batch_invariant_mode():
    config = _make_bounded_mxfp8_config(
        batch_invariant_mode=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
    )

    assert config.batch_invariant_mode


def test_flashinfer_bf16_config_rejects_batch_invariant_mode():
    with pytest.raises(ValueError, match="only for an MXFP8 model configuration"):
        _make_bounded_mxfp8_config(
            fp8=None,
            fp8_param=False,
            inference_flashinfer_mxfp8_token_capacity=None,
            batch_invariant_mode=True,
            params_dtype=torch.bfloat16,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )


def test_bf16_config_ignores_inactive_mxfp8_recipe_gates():
    config = _make_bounded_mxfp8_config(
        fp8=None,
        fp8_param=False,
        activation_func=F.gelu,
        inference_flashinfer_mxfp8_token_capacity=None,
    )

    assert config.fp8 is None


def test_vllm_backend_accepts_mxfp8_config_for_per_layer_dispatch():
    config = _make_bounded_mxfp8_config(
        inference_grouped_gemm_backend="vllm",
        inference_flashinfer_mxfp8_token_capacity=None,
        expert_model_parallel_size=1,
    )

    assert config.inference_grouped_gemm_backend.value == "vllm"


def test_vllm_mxfp8_layer_dispatches_to_mcore_path():
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    expected = (object(), None)
    grouped_mlp = SimpleNamespace(
        _concatenated_weights_built=True,
        _uses_mxfp8_weights=True,
        _fc1_weight=_mxfp8_weight(),
        _fc2_weight=_mxfp8_weight(),
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.VLLM,
        _mcore_fused_moe_forward=lambda hidden, probs, routing_map: expected,
        _vllm_forward=lambda *args, **kwargs: pytest.fail("BF16 vLLM path was selected"),
    )

    with InferenceMode.active():
        actual = InferenceGroupedMLP.forward(
            grouped_mlp,
            torch.empty(1, 1),
            None,
            torch.empty(1, 1),
            routing_map=torch.zeros(1, 1, dtype=torch.int64),
        )

    assert actual is expected


def test_lazy_weight_build_rejects_partially_selected_expert_projection():
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    grouped_mlp = SimpleNamespace(
        num_local_experts=1,
        linear_fc1=SimpleNamespace(weight0=_mxfp8_weight()),
        linear_fc2=SimpleNamespace(weight0=torch.empty(1, dtype=torch.bfloat16)),
    )

    with pytest.raises(TypeError, match="select both expert projections"):
        InferenceGroupedMLP._expert_weights_use_mxfp8(grouped_mlp)


@pytest.mark.parametrize("activation_func", [F.gelu, F.silu, F.relu])
def test_flashinfer_mxfp8_config_rejects_unsupported_activation(activation_func):
    with pytest.raises(ValueError, match="supports only non-gated squared-ReLU experts"):
        _make_bounded_mxfp8_config(activation_func=activation_func)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fp8_param": False}, "fp8_param must be enabled"),
        ({"fp8": None, "fp8_param": False}, "requires.*FP8 enabled"),
        ({"inference_moe_token_dispatcher_type": "nccl"}, "requires.*nvls"),
        ({"expert_model_parallel_size": 1}, "requires.*expert_model_parallel_size > 1"),
    ],
)
def test_bounded_flashinfer_mxfp8_config_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_bounded_mxfp8_config(**overrides)


def test_missing_routed_mxfp8_capability_has_precise_error(monkeypatch):
    from megatron.core.inference.moe import flashinfer_mxfp8

    monkeypatch.setattr(flashinfer_mxfp8, "HAVE_FLASHINFER_ROUTED_MXFP8", False)
    monkeypatch.setattr(
        flashinfer_mxfp8,
        "_FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR",
        ImportError("missing routed MXFP8 API"),
    )

    with pytest.raises(RuntimeError, match="requires FlashInfer >= 0.6.4"):
        flashinfer_mxfp8.require_flashinfer_routed_mxfp8()


def test_flashinfer_mxfp8_refresh_reports_noop_before_weight_build():
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    grouped_mlp = SimpleNamespace(
        _concatenated_weights_built=False,
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER,
    )

    assert InferenceGroupedMLP.refresh_flashinfer_mxfp8_weights(grouped_mlp) is False


def test_flashinfer_mxfp8_refresh_skips_bf16_expert_weights():
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    grouped_mlp = SimpleNamespace(
        _concatenated_weights_built=True,
        _uses_mxfp8_weights=False,
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER,
        _fc1_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
        _fc2_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
    )

    assert InferenceGroupedMLP.refresh_flashinfer_mxfp8_weights(grouped_mlp) is False


@requires_cuda_triton
@pytest.mark.parametrize(
    ("rows", "valid", "hidden", "padded"),
    [(37, 21, 160, 256), (64, 64, 2688, 2688), (8, 0, 96, 128)],
)
def test_quantize_routed_mxfp8_input_matches_the_mcore_quantizer(rows, valid, hidden, padded):
    torch.manual_seed(0)
    hidden_states = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)

    data, scale = quantize_routed_mxfp8_input(hidden_states, padded, _valid(valid))

    assert data.shape == (rows, padded) and data.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, padded // 32) and scale.dtype == torch.uint8
    if valid:
        expected = MXFP8Tensor.from_bf16(hidden_states[:valid], backend="triton")
        expected_scale = flashinfer_mxfp8._unshuffle_cublas_scale(expected.scale, valid, hidden)
        assert torch.equal(data[:valid, :hidden].view(torch.uint8), expected.data.view(torch.uint8))
        assert torch.equal(scale[:valid, : hidden // 32], expected_scale)
    # Padding columns of live rows are zero data with a zero scale exponent.
    assert not data[:valid, hidden:].view(torch.uint8).any()
    assert not scale[:valid, hidden // 32 :].any()


@requires_cuda_triton
def test_flashinfer_routed_mxfp8_moe_hands_the_kernel_live_rows_and_the_device_count(monkeypatch):
    """Quantize and pack only the live rows; the kernel gets the device count and finalizes in place."""
    rows, valid, topk, hidden, intermediate = 16, 10, 2, 160, 128
    torch.manual_seed(3)
    hidden_states = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    routing_map = torch.randint(0, 8, (rows, topk), device="cuda", dtype=torch.int64)
    probs = torch.rand(rows, topk, device="cuda")
    valid_tokens, seen = _valid(valid), {}

    def fake_kernel(**kwargs):
        seen.update(kwargs)
        kwargs["output"][:valid] = 1.0
        return kwargs["output"]

    def weight(logical_rows, logical_cols):
        padded_rows, padded_cols = -(-logical_rows // 128) * 128, -(-logical_cols // 128) * 128
        return FlashInferRoutedMXFP8Weight(
            data=torch.empty(2, padded_rows, padded_cols, dtype=torch.float8_e4m3fn),
            scale=torch.empty(2, padded_rows * padded_cols // 32, dtype=torch.uint8),
            logical_rows=logical_rows,
            logical_cols=logical_cols,
            padded_rows=padded_rows,
            padded_cols=padded_cols,
        )

    monkeypatch.setattr(flashinfer_mxfp8, "HAVE_FLASHINFER_ROUTED_MXFP8", True)
    monkeypatch.setattr(flashinfer_mxfp8, "trtllm_fp8_block_scale_routed_moe", fake_kernel)
    layout = SimpleNamespace(MajorK=SimpleNamespace(value=0))
    monkeypatch.setattr(flashinfer_mxfp8, "WeightLayout", layout, raising=False)
    monkeypatch.setattr(
        flashinfer_mxfp8, "Fp8QuantizationType", SimpleNamespace(MxFp8=None), raising=False
    )
    out = torch.full((rows, hidden), 7.0, device="cuda")

    returned = flashinfer_routed_mxfp8_moe(
        hidden_states,
        routing_map,
        probs,
        weight(intermediate, hidden),
        weight(hidden, intermediate),
        num_experts=8,
        local_expert_offset=2,
        activation_type=6,
        valid_tokens=valid_tokens,
        out=out,
    )

    assert returned is out and seen["output"] is out and seen["num_active_tokens"] is valid_tokens
    assert (
        seen["do_finalize"] is True and seen["local_expert_offset"] == 2 and seen["top_k"] == topk
    )
    quantized = MXFP8Tensor.from_bf16(hidden_states[:valid], backend="triton")
    assert seen["hidden_states"].shape == (rows, 256)
    assert torch.equal(
        seen["hidden_states"][:valid, :hidden].view(torch.uint8), quantized.data.view(torch.uint8)
    )
    assert torch.equal(
        seen["hidden_states_scale"][:valid, : hidden // 32],
        flashinfer_mxfp8._unshuffle_cublas_scale(quantized.scale, valid, hidden),
    )
    weight_bits = probs.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.equal(seen["topk_ids"][:valid], ((routing_map.int() << 16) | weight_bits)[:valid])
    assert torch.all(seen["topk_ids"][valid:] == -1 << 16)
    assert torch.all(out[:valid] == 1.0) and torch.all(out[valid:] == 7.0)


def _blackwell_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.skipif(
    not HAVE_FLASHINFER_ROUTED_MXFP8 or not _blackwell_or_newer(),
    reason="FlashInfer routed MXFP8 requires Blackwell and FlashInfer >= 0.6.4",
)
@pytest.mark.parametrize("local_expert_offset", [0, 2, 4, 6])
def test_flashinfer_routed_mxfp8_is_batch_invariant(local_expert_offset):
    """A token's local-EP output must not depend on its batch or row position."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        flashinfer_routed_mxfp8_moe,
        prepare_routed_mxfp8_weights,
    )
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    torch.manual_seed(1234)
    num_experts, hidden_size, intermediate_size, topk = 8, 256, 256, 2

    def _prepare_weight(weight: torch.Tensor):
        quantized = [MXFP8Tensor.from_bf16(expert, backend="triton") for expert in weight]
        stacked = MXFP8Tensor(
            data=torch.stack([expert.data for expert in quantized]).contiguous(),
            scale=torch.stack([expert.scale for expert in quantized]).contiguous(),
            backend="triton",
            dtype=torch.bfloat16,
        )
        return prepare_routed_mxfp8_weights(stacked)

    fc1_weight = _prepare_weight(
        torch.randn(
            num_experts, intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )[local_expert_offset : local_expert_offset + 2]
    )
    fc2_weight = _prepare_weight(
        torch.randn(
            num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
        )[local_expert_offset : local_expert_offset + 2]
    )
    target = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    target_experts = torch.tensor([2, 5], device="cuda", dtype=torch.int64)
    target_probs = torch.tensor([0.625, 0.375], device="cuda", dtype=torch.float32)

    def _run(active_tokens: int, target_row: int) -> torch.Tensor:
        num_tokens = 512
        hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        routing_map = torch.full((num_tokens, topk), -1, device="cuda", dtype=torch.int64)
        routing_map[:active_tokens, 0] = 2
        routing_map[:active_tokens, 1] = 3
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        probs[:active_tokens].fill_(0.5)
        hidden_states[target_row].copy_(target)
        routing_map[target_row].copy_(target_experts)
        probs[target_row].copy_(target_probs)
        return flashinfer_routed_mxfp8_moe(
            hidden_states,
            routing_map,
            probs,
            fc1_weight,
            fc2_weight,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            activation_type=6,
            valid_tokens=_valid(num_tokens),
        )[target_row]

    with torch.no_grad():
        output_alone = _run(1, 0)
        output_batched = _run(128, 73)

    assert torch.equal(output_alone, output_batched), (
        "FlashInfer routed MXFP8 output changed with the batch; max abs diff: "
        f"{(output_alone.float() - output_batched.float()).abs().max().item()}"
    )
    if not any(
        local_expert_offset <= expert < local_expert_offset + 2
        for expert in target_experts.tolist()
    ):
        assert torch.count_nonzero(output_alone).item() == 0


@pytest.mark.parametrize(
    ("uses_mxfp8", "has_rsv", "ep_size", "etp_size"),
    [
        (True, True, 4, 1),
        (False, True, 4, 1),
        (False, False, 4, 1),
        (False, True, 1, 2),
        (False, True, 1, 1),
    ],
    ids=[
        "mxfp8-routed",
        "bf16-bounded",
        "bf16-no-nvls-buffer",
        "bf16-expert-tp-only",
        "bf16-single-rank",
    ],
)
def test_flashinfer_forward_dispatch(monkeypatch, uses_mxfp8, has_rsv, ep_size, etp_size):
    """The kernel gets the device count and the fp32 buffer whenever the NVLS dispatcher gathers
    across ranks (expert parallel or expert tensor parallel), and neither otherwise."""
    from megatron.core.transformer.moe import experts

    valid_tokens, calls, rank = torch.tensor([3], dtype=torch.int32), {}, min(1, ep_size - 1)
    rsv = torch.empty(4, 8) if has_rsv else None
    kernel_output = torch.empty(4, 8, dtype=torch.bfloat16)

    def routed(*args, **kwargs):
        calls["routed"] = kwargs
        return rsv

    def cutlass(*args, **kwargs):
        calls["cutlass"] = dict(kwargs, args=args)
        return [kwargs["output"] if kwargs.get("output") is not None else kernel_output]

    monkeypatch.setattr(experts, "HAVE_FLASHINFER", True)
    monkeypatch.setattr(experts, "flashinfer_routed_mxfp8_moe", routed)
    monkeypatch.setattr(experts, "cutlass_fused_moe", cutlass)
    monkeypatch.setattr(
        experts, "fused_moe", SimpleNamespace(cutlass_fused_moe=cutlass), raising=False
    )
    monkeypatch.setattr(
        experts.InferenceAllGatherDispatcherBase,
        "_valid_tokens",
        classmethod(lambda cls: valid_tokens),
    )
    monkeypatch.setattr(
        experts.NVLSAllGatherVDispatcher, "_get_rsv_tensor", classmethod(lambda cls: rsv)
    )
    grouped_mlp = SimpleNamespace(
        config=SimpleNamespace(expert_tensor_parallel_size=etp_size),
        _uses_mxfp8_weights=uses_mxfp8,
        _nvfp4_weights=None,
        _fc1_weight=None,
        _fc2_weight=None,
        _flashinfer_activation_type=SimpleNamespace(value=6),
        _activation_clamp_scale=None,
        _nvls_dispatcher=True,
        num_local_experts=2,
        ep_group=SimpleNamespace(size=lambda: ep_size, rank=lambda: rank),
    )

    output, bias = experts.InferenceGroupedMLP._flashinfer_forward(
        grouped_mlp,
        torch.empty(4, 8, dtype=torch.bfloat16),
        torch.zeros(4, 1, dtype=torch.int64),
        torch.zeros(4, 1),
    )

    assert bias is None
    if uses_mxfp8:
        routed = calls["routed"]
        assert output is rsv and set(calls) == {"routed"}
        assert routed["valid_tokens"] is valid_tokens and routed["out"] is rsv
        assert routed["num_experts"] == 2 * ep_size and routed["local_expert_offset"] == 2 * rank
        return
    cutlass = calls["cutlass"]
    assert (
        set(calls) == {"cutlass"} and cutlass["ep_size"] == ep_size and cutlass["ep_rank"] == rank
    )
    assert cutlass["args"][1].dtype == torch.int32
    if has_rsv and ep_size * etp_size > 1:
        # The kernel finalizes fp32 in place over the live rows of the reduce-scatter buffer.
        assert output is rsv and cutlass["output"] is rsv and cutlass["args"][5] is torch.float32
        assert cutlass["num_active_tokens"] is valid_tokens
    else:
        # No NVLS buffer, or a single rank whose dispatcher never reads it: the kernel's bf16
        # rows go back to token_combine unchanged.
        assert output is kernel_output and cutlass["args"][5] is torch.bfloat16
        assert cutlass["output"] is None and "num_active_tokens" not in cutlass


def test_flashinfer_nvls_clears_routing_before_metadata_fence(monkeypatch):
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe import token_dispatcher_inference

    dispatcher_cls = token_dispatcher_inference.NVLSAllGatherVDispatcher
    base_cls = token_dispatcher_inference.InferenceAllGatherDispatcherBase
    calls = []

    class RoutingTensor:
        def fill_(self, value):
            calls.append(("clear", value))

    monkeypatch.setattr(dispatcher_cls, "_symm_agv_routing", {"tensor": RoutingTensor()})
    monkeypatch.setattr(
        dispatcher_cls, "_symm_metadata", {"tensor": object(), "handle": object()}, raising=False
    )
    monkeypatch.setattr(dispatcher_cls, "_step_metadata", object())
    monkeypatch.setattr(base_cls, "_host_valid_tokens_estimate", 0)
    monkeypatch.setattr(
        token_dispatcher_inference,
        "fused_metadata_update",
        lambda **kwargs: calls.append(("metadata", kwargs["local_tokens"])),
    )
    dispatcher = SimpleNamespace(
        config=SimpleNamespace(
            inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER
        ),
        tp_size=1,
        ep_size=4,
    )

    dispatcher_cls.update_metadata(dispatcher, local_tokens=7)

    assert calls == [("clear", -1), ("metadata", 7)]
    assert base_cls._host_valid_tokens_estimate == 28
