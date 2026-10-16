# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Activation-dtype mHC residual rounding through real public module paths.

Independent source: huggingface/transformers commit
93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py:1316-1318,1325-1327.
HF eagerly rounds BF16 post*branch before adding the BF16 mixed residual.
Keep production H4096/n4, S17/B1 and diagnostic seed1234; compare the output and
all four operand gradients exactly and with both similarities strictly>0.999.
This tests the residual operator, not complete mHC mappings or hybrid parity.
"""

import json
import os
from argparse import ArgumentParser

import pytest
import torch
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
    compiled = object()

    def compile_spy(function, **kwargs):
        calls.append((function, kwargs))
        return compiled

    hc._get_precision_preserving_h_post_bda.cache_clear()
    try:
        monkeypatch.setattr(torch._inductor, "list_options", lambda: ["emulate_precision_casts"])
        monkeypatch.setattr(torch, "compile", compile_spy)
        module = hc.HyperConnectionModule(
            _config(mhc_preserve_activation_dtype_rounding=True), layer_number=1
        )
        assert module._h_post_bda_op is hc._get_precision_preserving_h_post_bda() is compiled
        assert calls == [
            (hc._native_h_post_bda_impl, {"options": {"emulate_precision_casts": True}})
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
    with pytest.raises(ValueError, match="use_fused_mhc.*does not implement"):
        _config(use_fused_mhc=True, mhc_preserve_activation_dtype_rounding=True)


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
        assert manager._recomputed and manager.checkpoints[0].ctx is None
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
