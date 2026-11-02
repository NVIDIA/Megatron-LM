# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Activation-dtype mHC residual rounding through real public module paths.

Independent source: huggingface/transformers commit
93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py:1316-1318,1325-1327.
HF eagerly rounds BF16 post*branch before adding the BF16 mixed residual.
Keep production H4096/n4, S17/B1 and diagnostic seed1234; compare the output and
all four operand gradients exactly and with both similarities strictly>0.999.
This tests the residual operator, not complete mHC mappings or hybrid parity.
The test_h_post_bda_* cases also cover FP32 mappings: mixing without GEMMs, and one
compiled specialization shared by the grad-mode, no-grad and replayed passes.
"""

import json
import os
from argparse import ArgumentParser

import pytest
import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._inductor import list_options

from megatron.core.tensor_parallel.random import (
    MHCCheckpointManager,
    checkpoint,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import hyper_connection as hc
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        num_residual_streams=4,
        enable_hyper_connections=True,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _official_residual(comb, residual, post, branch):
    """HF eager order, using only independent PyTorch ops, without target helpers."""
    streams = residual.unflatten(-1, (4, 4096))
    return (
        post.unsqueeze(-1) * branch.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
    ).flatten(-2)


def _assert_exact_and_similarity(actual, expected, name):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    a, e = actual.detach().flatten().double(), expected.detach().flatten().double()
    aa, ee, dot = a.square().sum(), e.square().sum(), (a * e).sum()
    assert aa > 0 and ee > 0, name
    cosine = (dot / (aa * ee).sqrt()).item()
    similarity = (2 * dot / (aa + ee)).item()
    assert cosine > 0.999, f"{name}: cosine={cosine}"
    assert similarity > 0.999, f"{name}: tensor_similarity={similarity}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"name": name, "cosine": cosine, "tensor_similarity": similarity, "exact": True}


def test_rounding_flag_defaults_and_real_cli():
    from megatron.training.arguments import add_megatron_arguments

    assert _config().mhc_preserve_activation_dtype_rounding is False
    assert _config(
        mhc_preserve_activation_dtype_rounding=True
    ).mhc_preserve_activation_dtype_rounding
    # ArgumentGroupFactory exposes the new dataclass bool without a second,
    # potentially drifting hand-written training-arguments definition.
    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).mhc_preserve_activation_dtype_rounding is False
    assert parser.parse_args(
        ["--mhc-preserve-activation-dtype-rounding"]
    ).mhc_preserve_activation_dtype_rounding


def test_default_module_keeps_legacy_callable_without_strict_factory(monkeypatch):
    def unexpected_factory():
        pytest.fail("the default module must not require strict compiler support")

    monkeypatch.setattr(hc, "_get_precision_preserving_h_post_bda", unexpected_factory)
    module = hc.HyperConnectionModule(_config(), layer_number=1)
    assert module._h_post_bda_op is hc.native_h_post_bda


def test_strict_factory_is_cached_and_preserves_raw_implementation(monkeypatch):
    calls = []
    compiled = []

    def compile_spy(function, **kwargs):
        calls.append((function, kwargs))
        compiled.append(object())
        return compiled[-1]

    hc._get_precision_preserving_h_post_bda.cache_clear()
    try:
        monkeypatch.setattr(torch._inductor, "list_options", lambda: ["emulate_precision_casts"])
        monkeypatch.setattr(torch, "compile", compile_spy)
        module = hc.HyperConnectionModule(
            _config(mhc_preserve_activation_dtype_rounding=True), layer_number=1
        )
        post = module._h_post_bda_op
        assert post is hc._get_precision_preserving_h_post_bda()
        # The raw forward and backward, each compiled with the option, behind one Function.
        assert post.func == hc.NativeHPostBDA.apply and post.args == tuple(compiled)
        assert calls == [
            (hc._native_h_post_bda_impl, {"options": {"emulate_precision_casts": True}}),
            (hc._native_h_post_bda_bwd_impl, {"options": {"emulate_precision_casts": True}}),
        ]
    finally:
        hc._get_precision_preserving_h_post_bda.cache_clear()


def test_strict_factory_rejects_missing_option_without_fallback(monkeypatch):
    hc._get_precision_preserving_h_post_bda.cache_clear()
    try:
        monkeypatch.setattr(torch._inductor, "list_options", lambda: [])
        with pytest.raises(RuntimeError, match="requires torch.compile support.*no eager"):
            hc.HyperConnectionModule(
                _config(mhc_preserve_activation_dtype_rounding=True), layer_number=1
            )
    finally:
        hc._get_precision_preserving_h_post_bda.cache_clear()


def test_strict_factory_propagates_compiler_rejection_without_fallback(monkeypatch):
    rejection = RuntimeError("unsupported compiler configuration")

    def reject_compile(*args, **kwargs):
        raise rejection

    hc._get_precision_preserving_h_post_bda.cache_clear()
    try:
        monkeypatch.setattr(torch._inductor, "list_options", lambda: ["emulate_precision_casts"])
        monkeypatch.setattr(torch, "compile", reject_compile)
        with pytest.raises(RuntimeError, match="requires torch.compile support") as error:
            hc._get_precision_preserving_h_post_bda()
        assert error.value.__cause__ is rejection
    finally:
        hc._get_precision_preserving_h_post_bda.cache_clear()


def test_fused_mhc_rejects_rounding_contract():
    with pytest.raises(ValueError, match="use_fused_mhc does not support"):
        _config(use_fused_mhc=True, mhc_preserve_activation_dtype_rounding=True)


# FP32 mixing (fp32 models, or bf16 activations with mhc_keep_mappings_in_fp32): mapping
# dtype, activation dtype.
_POST_DTYPES = {
    "fp32": (torch.float32, torch.float32),
    "bf16_fp32_mappings": (torch.float32, torch.bfloat16),
}
_GEMMS = {
    torch.ops.aten.bmm,
    torch.ops.aten.mm,
    torch.ops.aten.addmm,
    torch.ops.aten.baddbmm,
    torch.ops.aten.matmul,
}


def _post_inputs(n, dtypes, with_bias, device):
    """h_res, residual, h_post, x, bias and an output gradient, as the module passes them."""
    mapping_dtype, activation_dtype = dtypes
    generator = torch.Generator().manual_seed(1234)
    s, b, hidden = 8, 2, 32

    def draw(*shape, dtype=activation_dtype):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    h_res = torch.randn(s, b, n, n, generator=generator).softmax(-1)
    h_post = 2 * torch.randn(s, b, n, generator=generator).sigmoid()
    inputs = (
        h_res.to(device=device, dtype=mapping_dtype),
        draw(s, b, n, hidden),
        h_post.to(device=device, dtype=mapping_dtype),
        draw(s, b, hidden),
        draw(hidden) if with_bias else None,
    )
    return inputs, draw(s, b, n, hidden)


def _leaves(inputs):
    return [None if t is None else t.clone().requires_grad_() for t in inputs]


class _AtenGraphRecorder:
    """torch.compile backend that keeps the ATen graph of every compilation and runs it."""

    def __init__(self):
        self.graphs = []

    def __call__(self, graph_module, example_inputs):
        def record(aten_module, _):
            self.graphs.append(aten_module)
            return make_boxed_func(aten_module.forward)

        return aot_autograd(fw_compiler=record, bw_compiler=record)(graph_module, example_inputs)

    def ops(self):
        return {
            node.target.overloadpacket
            for graph in self.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and hasattr(node.target, "overloadpacket")
        }


@pytest.mark.parametrize("preserve_rounding", [False, True])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dtypes", list(_POST_DTYPES.values()), ids=list(_POST_DTYPES))
@pytest.mark.parametrize("with_bias", [False, True])
def test_h_post_bda_compiles_once_without_gemm(
    monkeypatch, preserve_rounding, n, dtypes, with_bias
):
    """The reference, a checkpoint's no-grad forward and its replay share one GEMM-free entry.

    Each grad mode and set of requires_grad flags used to be a separate specialization of a
    compiled bmm, which TF32 and Inductor's GEMM padding could split into two precisions.
    Both public entry points run here with their compiled functions on a recording backend.
    """
    recorders = []
    compile_with_inductor = torch.compile

    def compile_recording(function, **_):
        recorders.append(_AtenGraphRecorder())
        return compile_with_inductor(function, backend=recorders[-1])

    hc._get_precision_preserving_h_post_bda.cache_clear()
    torch._dynamo.reset()
    try:
        if preserve_rounding:
            monkeypatch.setattr(torch, "compile", compile_recording)
            post = hc._get_precision_preserving_h_post_bda()
        else:
            for name in ("_native_h_post_bda_forward", "_native_h_post_bda_backward"):
                compiled = getattr(hc, name)
                monkeypatch.setattr(hc, name, compile_recording(compiled.__wrapped__))
            post = hc.native_h_post_bda
        inputs, gradient = _post_inputs(n, dtypes, with_bias, "cpu")
        reference = post(*_leaves(inputs))
        # A checkpoint's no-grad forward sees inputs that require grad; inference sees none.
        with torch.no_grad():
            recomputed = [post(*_leaves(inputs)), post(*inputs)]
        replay = post(*_leaves(inputs))
        replay.backward(gradient)
        reference.backward(gradient)
    finally:
        torch._dynamo.reset()
        hc._get_precision_preserving_h_post_bda.cache_clear()

    assert reference.grad_fn.name() == "NativeHPostBDABackward"
    assert all(torch.equal(output, reference) for output in [*recomputed, replay])
    forward, backward = recorders
    assert len(forward.graphs) == 1 and len(backward.graphs) == 1
    assert not (forward.ops() | backward.ops()) & _GEMMS, forward.ops() | backward.ops()


@pytest.mark.parametrize("preserve_rounding", [False, True])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dtypes", list(_POST_DTYPES.values()), ids=list(_POST_DTYPES))
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_h_post_bda_no_grad_matches_grad_mode_bitwise(
    monkeypatch, preserve_rounding, n, dtypes, with_bias, device
):
    """Bitwise forward/recompute parity, and gradients equal to FP64 within output rounding."""
    if device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        # The compiled bmm split into TF32 and FP32 specializations only with TF32 allowed.
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    post = hc._get_precision_preserving_h_post_bda() if preserve_rounding else hc.native_h_post_bda
    inputs, gradient = _post_inputs(n, dtypes, with_bias, device)
    torch._dynamo.reset()
    try:
        reference_leaves, replay_leaves = _leaves(inputs), _leaves(inputs)
        reference = post(*reference_leaves)
        # A checkpoint's no-grad forward sees inputs that require grad; inference sees none.
        with torch.no_grad():
            recomputed = [post(*_leaves(inputs)), post(*inputs)]
        replay = post(*replay_leaves)
        reference.backward(gradient)
        replay.backward(gradient)
    finally:
        torch._dynamo.reset()

    assert reference.dtype == inputs[1].dtype
    assert reference.grad_fn.name() == "NativeHPostBDABackward"
    assert all(torch.equal(output, reference) for output in [*recomputed, replay])
    fp64_leaves = [None if t is None else t.double().requires_grad_() for t in inputs]
    h_res, residual, h_post, x, bias = fp64_leaves
    branch = x if bias is None else x + bias
    expected = h_res.transpose(-1, -2) @ residual + h_post.unsqueeze(-1) * branch.unsqueeze(-2)
    expected.backward(gradient.double())
    torch.testing.assert_close(reference, expected.to(reference.dtype))
    for actual, again, fp64 in zip(reference_leaves, replay_leaves, fp64_leaves):
        if actual is None:
            continue
        assert actual.grad.dtype == actual.dtype
        assert torch.equal(actual.grad, again.grad)
        torch.testing.assert_close(actual.grad, fp64.grad.to(actual.dtype))


@pytest.fixture
def _model_parallel():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        yield
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("recompute", ["none", "mhc", "full"])
def test_public_mhc_residual_preserves_hf_rounding_and_all_operand_gradients(
    recompute, _model_parallel
):
    assert "emulate_precision_casts" in list_options(), "strict compiler support is required"
    module = hc.HyperConnectionModule(
        _config(mhc_preserve_activation_dtype_rounding=True), layer_number=1
    ).cuda()
    assert module._h_post_bda_op is hc._get_precision_preserving_h_post_bda()
    # Preserve the exact fixture draw order of the preceding microdiagnostic.
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    residual = torch.randn(17, 1, 4, 4096, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn(17, 1, 4096, device="cuda", dtype=torch.bfloat16)
    post = (2 * torch.randn(17, 1, 4, device="cuda").sigmoid()).to(torch.bfloat16)
    comb = torch.randn(17, 1, 4, 4, device="cuda").softmax(-1).to(torch.bfloat16)
    gradient = torch.randn_like(residual).flatten(-2)
    fixture = (comb, residual.flatten(-2), post, branch)
    actual_leaves = [value.clone().requires_grad_() for value in fixture]
    native_leaves = [value.clone().requires_grad_() for value in fixture]
    manager = MHCCheckpointManager() if recompute == "mhc" else None
    grad_modes = []

    def run(comb, residual, post, branch):
        grad_modes.append(torch.is_grad_enabled())
        return module.fused_h_res_h_post_bda(
            comb, residual, post, (branch, None), 0.0, True, False, manager=manager
        )

    actual = checkpoint(run, False, *actual_leaves) if recompute == "full" else run(*actual_leaves)
    expected = _official_residual(*native_leaves)
    comparisons = [_assert_exact_and_similarity(actual, expected, "output")]
    if manager is not None:
        assert len(manager.checkpoints) == 1
        # Form a real consumer before discarding its producer's output storage.
        # The common manager restores the output during this consumer's backward.
        loss = (actual.float() * gradient.float()).sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss)
        assert actual.untyped_storage().size() == 0
        loss.backward()
        assert manager._recomputed
        # As in test_hybrid_block.py: no region of the checkpoint still awaits a rebuild.
        region = manager.checkpoints[0]._current_region()
        assert region is None or not region.awaits_rebuild
    else:
        actual.backward(gradient)
    expected.backward(gradient)
    assert grad_modes == ([False, True] if recompute == "full" else [True])
    for name, observed, reference in zip(
        ("comb", "residual", "post", "branch"), actual_leaves, native_leaves
    ):
        comparisons.append(
            _assert_exact_and_similarity(observed.grad, reference.grad, name + " gradient")
        )
    assert all(parameter.grad is None for parameter in module.parameters())
    print(
        "MHC_RESIDUAL_PRECISION "
        + json.dumps(
            {
                "rank": int(os.environ.get("RANK", "0")),
                "preserve_activation_dtype_rounding": True,
                "compile_options": {"emulate_precision_casts": True},
                "recompute": recompute,
                "global_float32_matmul_precision": torch.get_float32_matmul_precision(),
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )
