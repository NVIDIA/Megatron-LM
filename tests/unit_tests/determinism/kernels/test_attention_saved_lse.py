# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.
"""Actual TE/FA2 vs saved-LSE cuDNN dQ/dK/dV; GPU reference tests, not model proof."""

import importlib
import inspect
from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.ops.attention.te_saved_lse import install_saved_lse_attention
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import set_attention_backend
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA/TE/FA2 required')


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    # Utils deliberately clears these environment variables during group init.
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    for name, value in {
        'NVTE_FLASH_ATTN': '1',
        'NVTE_FLASH_ATTN_V2': '1',
        'NVTE_FLASH_ATTN_V3': '0',
        'NVTE_FLASH_ATTN_V4': '0',
        'NVTE_FUSED_ATTN': '0',
        'NVTE_UNFUSED_ATTN': '0',
        'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '1',
    }.items():
        monkeypatch.setenv(name, value)
    dpa = importlib.import_module(
        'transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention'
    )
    # TE caches backend selection by AttentionParams, not the NVTE environment.
    # Own a fresh cache for this test and restore the original object at teardown.
    expected = {
        'attention_params',
        'use_flash_attention',
        'flash_attention_backend',
        'use_fused_attention',
        'fused_attention_backend',
        'use_unfused_attention',
        'backend_selection_requires_update',
    }
    expected = set(dpa._attention_backends)
    cache = {name: None for name in expected}
    cache['backend_selection_requires_update'] = True
    monkeypatch.setattr(dpa, '_attention_backends', cache)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def build(enabled, dropout=0.0, deterministic=True):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2688,
        num_attention_heads=32,
        num_query_groups=2,
        kv_channels=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=dropout,
        hidden_dropout=0.0,
        attention_saved_lse_cudnn_backward=enabled,
        attention_backend=AttnBackend.flash,
        flash_attention_version=2,
        deterministic_mode=deterministic,
    )
    set_attention_backend(config)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    return TEDotProductAttention(config, 1, AttnMaskType.causal, 'self', pg_collection=pg).cuda()


def operands(sequence, batch, strided, dtype=torch.bfloat16):
    packed = torch.randn(sequence, batch, 36 * 128, device='cuda', dtype=dtype)
    pieces = packed.split([32 * 128, 2 * 128, 2 * 128], dim=-1)
    return [
        (
            (
                x.reshape(sequence, batch, h, 128)
                if strided
                else x.reshape(sequence, batch, h, 128).contiguous()
            )
            .detach()
            .requires_grad_()
        )
        for x, h in zip(pieces, (32, 2, 2), strict=True)
    ]


def relative_l2(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)


@pytest.mark.parametrize('sequence,batch,strided', [(512, 1, False), (512, 1, True)])
def test_exact_forward_saved_lse_native_sbhd_backward(sequence, batch, strided):
    import transformer_engine.pytorch.attention.dot_product_attention.backends as backend

    torch.manual_seed(77211)
    reference, candidate = build(False), build(True)
    assert not getattr(reference.flash_attention, '_saved_lse_adapter_installed', False)
    assert candidate.flash_attention._saved_lse_adapter_installed
    dpa = importlib.import_module(
        'transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention'
    )
    assert reference.config.flash_attention_version == candidate.config.flash_attention_version == 2
    original_descriptor = backend.FlashAttention.forward
    original_flash = backend.flash_attn_func
    original_fused_forward = backend.fused_attn_fwd
    original_backward = backend.fused_attn_bwd
    values = operands(sequence, batch, strided)
    before = [v.detach().clone() for v in values]
    versions = [v._version for v in values]
    dy = torch.randn(sequence, batch, 32 * 128, device='cuda', dtype=torch.bfloat16)
    dy_before = dy.clone()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with (
        patch.object(backend, 'flash_attn_func', wraps=original_flash) as flash_calls,
        patch.object(backend, 'fused_attn_fwd', wraps=original_fused_forward) as fused_forward,
        patch.object(backend, 'fused_attn_bwd', wraps=original_backward) as fused_backward,
    ):
        expected = reference(*values, None, AttnMaskType.causal)
        expected_grad = torch.autograd.grad(expected, values, dy)
        previous_gradients = None
        for _ in range(3):
            actual = candidate(*values, None, AttnMaskType.causal)
            gradients = torch.autograd.grad(actual, values, dy)
            assert torch.equal(actual, expected), 'Native FA2 forward must remain bitwise exact'
            for actual_grad, reference_grad in zip(gradients, expected_grad, strict=True):
                assert relative_l2(actual_grad, reference_grad) <= 0.005
            if previous_gradients is not None:
                assert all(torch.equal(a, b) for a, b in zip(gradients, previous_gradients))
            previous_gradients = tuple(value.clone() for value in gradients)
            assert backend.FlashAttention.forward is original_descriptor
        assert dpa._attention_backends['use_flash_attention']
        assert dpa._attention_backends['flash_attention_backend'].major == 2
        assert flash_calls.call_count == 4
        assert fused_forward.call_count == 0, 'No cuDNN forward permitted'
        assert fused_backward.call_count == 3
    assert backend.flash_attn_func is original_flash
    assert backend.FlashAttention.forward is original_descriptor
    assert candidate.flash_attention._saved_lse_adapter_statistics == {
        'forward': 3,
        'backward': 3,
        'fallback': 0,
    }
    assert all(
        torch.equal(v.detach(), old) and v._version == version
        for v, old, version in zip(values, before, versions, strict=True)
    )
    assert torch.equal(dy, dy_before)
    assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(
        torch.cuda.get_rng_state(), cuda_rng
    )


@pytest.mark.parametrize(
    'dtype,mask',
    [
        (torch.bfloat16, AttnMaskType.causal),
        (torch.float16, AttnMaskType.causal),
        (torch.bfloat16, AttnMaskType.no_mask),
    ],
)
def test_unqualified_geometry_precision_mask_preserves_native_fallback(dtype, mask, monkeypatch):
    # Bitwise fallback equivalence needs the original deterministic FA2 backward,
    # not two independent nondeterministic reference launches.
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '0')
    # B2 remains outside the qualified adapter shapes; no support guard is relaxed.
    reference, candidate = build(False, deterministic=True), build(True, deterministic=True)
    import transformer_engine.pytorch.attention.dot_product_attention.backends as backend

    closure = inspect.getclosurevars(candidate.flash_attention.forward.__func__).nonlocals
    original_delegate = closure['original']
    assert original_delegate.__self__ is candidate.flash_attention
    assert original_delegate.__func__ is backend.FlashAttention.forward
    assert reference.flash_attention.deterministic and candidate.flash_attention.deterministic
    values = operands(512, 2, False, dtype)
    dy = torch.randn(512, 2, 4096, device='cuda', dtype=dtype)
    expected = reference(*values, None, mask)
    reference_grad = torch.autograd.grad(expected, values, dy)
    actual = candidate(*values, None, mask)
    actual_grad = torch.autograd.grad(actual, values, dy)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, reference_grad, strict=True))
    dpa = importlib.import_module(
        'transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention'
    )
    assert dpa._attention_backends['use_flash_attention']
    assert dpa._attention_backends['flash_attention_backend'].major == 2
    assert candidate.flash_attention._saved_lse_adapter_statistics == {
        'forward': 0,
        'backward': 0,
        'fallback': 1,
    }


def test_flag_off_does_not_install_or_import_backend_wrapper():
    original = build(False)
    child = original.flash_attention
    forward = child.forward
    assert install_saved_lse_attention(original) is False
    assert child.forward.__func__ is forward.__func__
    assert not hasattr(child, '_saved_lse_adapter_installed')
