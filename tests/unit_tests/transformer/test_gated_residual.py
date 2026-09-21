# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for the gated-residual hyper-connection variant.

Covers:
1. GroupedRMSNorm parity against the HF Qwen4-Exp reference norm
2. GatedResidualModule forward/backward parity against the HF reference
   (per-sublayer 4-tuple form and the use_combine=False exit-contract form)
3. The h_res=None write-back against the hand-derived formula and against the
   reference decoder-layer injection
4. Forward/backward parity of the CheckpointWithoutOutputManager recompute path
5. TransformerConfig validation of the gated_residual knobs
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.gated_residual import GatedResidualModule, GroupedRMSNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# =============================================================================
# Reference oracle
#
# Ported from huggingface/transformers `src/transformers/models/qwen4_exp/
# modeling_qwen4_exp.py` (commit 99e19a9a), classes Qwen4ExpTextRMSNorm and
# Qwen4ExpTextGatedResidual plus the decoder-layer injection arithmetic.
#
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# =============================================================================


class RefRMSNorm(nn.Module):
    """Qwen4ExpTextRMSNorm: fp32 grouped RMSNorm with zero-centered gamma."""

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class RefGatedResidual(nn.Module):
    """Qwen4ExpTextGatedResidual with the config object replaced by scalars."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float = 1e-6,
        use_combine: bool = True,
    ):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        hc_hidden_size = hc_count * hidden_size
        self.hc_norm = RefRMSNorm(hc_hidden_size, group_size=hidden_size, eps=eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, hc_count, bias=False) if use_combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(hyper_input_normed) / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


def ref_write_back(hyper_input, sublayer_out, injection_weights):
    """The reference decoder layer's injection arithmetic."""
    injection = sublayer_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


# =============================================================================
# Helpers
# =============================================================================

HIDDEN = 64
STREAMS = 4
LOWRANK = 16
SEQ, BATCH = 8, 2


def make_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_connection_variant="gated_residual",
        hc_lowrank=LOWRANK,
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def make_modules(dtype, use_combine=True):
    """Build a GatedResidualModule and a weight-synchronized reference oracle."""
    torch.manual_seed(42)
    module = GatedResidualModule(make_config(), layer_number=1, use_combine=use_combine)
    ref = RefGatedResidual(HIDDEN, STREAMS, LOWRANK, use_combine=use_combine)
    with torch.no_grad():
        ref.hc_norm.weight.copy_(module.hc_norm.weight)
        # Exercise a non-trivial gamma.
        module.hc_norm.weight.normal_(0.0, 0.1)
        ref.hc_norm.weight.copy_(module.hc_norm.weight)
        ref.input_mix_weight_down.weight.copy_(module.input_mix_weight_down.weight)
        ref.input_mix_weight_up.weight.copy_(module.input_mix_weight_up.weight)
        if use_combine:
            ref.block_inject_weight.weight.copy_(module.block_inject_weight.weight)
    module = module.cuda()
    ref = ref.cuda()
    if dtype != torch.float32:
        # Mirror Megatron's dtype policy: norm gamma and write gate stay fp32
        # (mark_keep_in_fp32); the reference runs entirely in the low dtype.
        from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

        convert_module_to_dtype_except_fp32_marked(module, dtype)
        ref = ref.to(dtype)
    return module, ref


def make_input(dtype, requires_grad=True):
    torch.manual_seed(7)
    x = torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda", dtype=dtype)
    return x.requires_grad_(requires_grad)


TOLS = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.bfloat16: dict(atol=1e-2, rtol=1e-2)}


# =============================================================================
# Tests
# =============================================================================


class TestGroupedRMSNorm:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_parity_with_reference(self, dtype):
        torch.manual_seed(0)
        dim, group = STREAMS * HIDDEN, HIDDEN
        norm = GroupedRMSNorm(dim, group_size=group, eps=1e-6, zero_centered_gamma=True).cuda()
        ref = RefRMSNorm(dim, group_size=group, eps=1e-6).cuda()
        with torch.no_grad():
            norm.weight.normal_(0.0, 0.1)
            ref.weight.copy_(norm.weight)
        x = torch.randn(SEQ, BATCH, dim, device="cuda", dtype=dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        out = norm(x)
        out_ref = ref(x_ref)
        torch.testing.assert_close(out, out_ref, **TOLS[dtype])

        grad = torch.randn_like(out)
        out.backward(grad)
        out_ref.backward(grad)
        torch.testing.assert_close(x.grad, x_ref.grad, **TOLS[dtype])
        torch.testing.assert_close(norm.weight.grad, ref.weight.grad, **TOLS[dtype])

    def test_zero_centered_gamma_zero_init_is_identity_scale(self):
        dim, group = 32, 8
        norm = GroupedRMSNorm(dim, group_size=group, zero_centered_gamma=True).cuda()
        x = torch.randn(4, dim, device="cuda")
        out = norm(x)
        grouped = x.view(4, -1, group)
        expected = (grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + norm.eps)).view(
            4, dim
        )
        torch.testing.assert_close(out, expected)

    def test_dim_not_divisible_raises(self):
        with pytest.raises(ValueError):
            GroupedRMSNorm(10, group_size=3)


class TestGatedResidualParity:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_backward_parity(self, dtype):
        module, ref = make_modules(dtype, use_combine=True)
        x = make_input(dtype)
        x_ref = x.detach().clone().requires_grad_(True)

        mixed, h_res, g_write, residual = module(x)
        assert h_res is None
        assert residual is x
        mixed_ref, residual_ref, g_write_ref = ref(x_ref)

        tols = TOLS[dtype]
        torch.testing.assert_close(mixed, mixed_ref, **tols)
        torch.testing.assert_close(g_write, g_write_ref, **tols)

        # Backward through both gate outputs.
        loss = mixed.float().square().sum() + g_write.float().square().sum()
        loss_ref = mixed_ref.float().square().sum() + g_write_ref.float().square().sum()
        loss.backward()
        loss_ref.backward()

        torch.testing.assert_close(x.grad, x_ref.grad, **tols)
        torch.testing.assert_close(
            module.hc_norm.weight.grad.float(), ref.hc_norm.weight.grad.float(), **tols
        )
        torch.testing.assert_close(
            module.input_mix_weight_down.weight.grad.float(),
            ref.input_mix_weight_down.weight.grad.float(),
            **tols,
        )
        torch.testing.assert_close(
            module.input_mix_weight_up.weight.grad.float(),
            ref.input_mix_weight_up.weight.grad.float(),
            **tols,
        )
        torch.testing.assert_close(
            module.block_inject_weight.weight.grad.float(),
            ref.block_inject_weight.weight.grad.float(),
            **tols,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_exit_contract_parity(self, dtype):
        """use_combine=False returns only the n->1 contraction."""
        module, ref = make_modules(dtype, use_combine=False)
        x = make_input(dtype)
        x_ref = x.detach().clone().requires_grad_(True)

        mixed = module(x)
        assert isinstance(mixed, torch.Tensor)
        mixed_ref = ref(x_ref)
        torch.testing.assert_close(mixed, mixed_ref, **TOLS[dtype])

        mixed.float().square().sum().backward()
        mixed_ref.float().square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, **TOLS[dtype])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_write_back_matches_reference(self, dtype, with_bias):
        """fused_h_res_h_post_bda(None, ...) == residual + g_write ⊙ (y + bias)."""
        module, ref = make_modules(dtype, use_combine=True)
        x = make_input(dtype, requires_grad=False)
        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=dtype)
        bias = torch.randn(HIDDEN, device="cuda", dtype=dtype) if with_bias else None

        mixed, h_res, g_write, residual = module(x)
        out = module.fused_h_res_h_post_bda(
            h_res, residual, g_write, (y, bias), dropout_prob=0.0, training=True, fused=False
        )

        y_eff = y + bias if with_bias else y
        expected = ref_write_back(residual, y_eff, g_write)
        torch.testing.assert_close(out, expected, **TOLS[dtype])

    def test_write_back_rejects_h_res(self):
        module, _ = make_modules(torch.float32, use_combine=True)
        x = make_input(torch.float32, requires_grad=False)
        _, _, g_write, residual = module(x)
        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda")
        with pytest.raises(TypeError, match="no h_res"):
            module.fused_h_res_h_post_bda(
                torch.zeros(1, device="cuda"), residual, g_write, (y, None), 0.0, True, False
            )

    def test_fp32_residual_input_with_bf16_params(self):
        """fp32_residual_connection: fp32 streams through a bf16-converted module.

        The gate GEMMs must run in the weight dtype (not the activation dtype),
        `mixed`/`g_write` come out in the params dtype for the bf16 sublayers,
        and the write-back preserves the fp32 residual stream.
        """
        module, _ = make_modules(torch.bfloat16, use_combine=True)
        x = make_input(torch.float32)  # fp32 n-stream residual

        mixed, h_res, g_write, residual = module(x)
        assert h_res is None
        assert mixed.dtype == torch.bfloat16
        assert g_write.dtype == torch.bfloat16
        assert residual.dtype == torch.float32

        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16)
        out = module.fused_h_res_h_post_bda(
            h_res, residual, g_write, (y, None), dropout_prob=0.0, training=True, fused=False
        )
        assert out.dtype == torch.float32  # fp32 residual stream preserved
        # mixed carries the read-gate (down/up) gradient path; out carries the
        # write-gate path — include both so every parameter receives a gradient.
        (out.float().square().sum() + mixed.float().square().sum()).backward()
        for name, param in module.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name

    def test_write_back_dropout_path_runs(self):
        module, _ = make_modules(torch.float32, use_combine=True)
        x = make_input(torch.float32, requires_grad=False)
        _, h_res, g_write, residual = module(x)
        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda")
        out = module.fused_h_res_h_post_bda(
            h_res, residual, g_write, (y, None), dropout_prob=0.5, training=True, fused=False
        )
        assert out.shape == residual.shape


class TestGatedResidualRecompute:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_manager_path_matches_eager(self):
        """Forward values and backward grads match with recompute on."""
        dtype = torch.float32
        module, _ = make_modules(dtype, use_combine=True)
        x = make_input(dtype)
        x_ckpt = x.detach().clone().requires_grad_(True)
        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=dtype)

        # Eager reference pass through gates + write-back.
        mixed, h_res, g_write, residual = module(x)
        out = module.fused_h_res_h_post_bda(h_res, residual, g_write, (y, None), 0.0, True, False)
        loss_ref = out.float().square().sum() + mixed.float().square().sum()
        loss_ref.backward()
        grads_ref = {n: p.grad.detach().clone() for n, p in module.named_parameters()}
        x_grad_ref = x.grad.detach().clone()
        module.zero_grad(set_to_none=True)

        # Recompute pass: gates + write-back checkpointed via one manager.
        manager = CheckpointWithoutOutputManager()
        mixed_c, h_res_c, g_write_c, residual_c = module(x_ckpt, mhc_recompute_manager=manager)
        out_c = module.fused_h_res_h_post_bda(
            h_res_c, residual_c, g_write_c, (y, None), 0.0, True, False, manager=manager
        )
        torch.testing.assert_close(mixed_c, mixed.detach())
        torch.testing.assert_close(out_c, out.detach())

        loss_c = out_c.float().square().sum() + mixed_c.float().square().sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_c)
        # Discarded checkpoint outputs are resized to zero until recompute runs.
        assert out_c.untyped_storage().size() == 0
        loss_c.backward()

        torch.testing.assert_close(x_ckpt.grad, x_grad_ref)
        for name, param in module.named_parameters():
            torch.testing.assert_close(param.grad, grads_ref[name], msg=name)

    def test_manager_path_matches_eager_with_dropout(self):
        """Dropout write-back + recompute manager: the RNG rewind must replay the
        same dropout mask, so gradients match the eager run bit for bit."""
        dtype = torch.float32
        module, _ = make_modules(dtype, use_combine=True)
        y = torch.randn(SEQ, BATCH, HIDDEN, device="cuda", dtype=dtype)

        def run(with_manager):
            module.zero_grad(set_to_none=True)
            torch.manual_seed(99)
            torch.cuda.manual_seed_all(99)
            x = make_input(dtype)
            manager = CheckpointWithoutOutputManager() if with_manager else None
            mixed, h_res, g_write, residual = module(x, mhc_recompute_manager=manager)
            out = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                g_write,
                (y, None),
                dropout_prob=0.5,
                training=True,
                fused=False,
                manager=manager,
            )
            loss = out.float().square().sum() + mixed.float().square().sum()
            if manager is not None:
                manager.discard_all_outputs_and_register_unified_recompute(loss)
            loss.backward()
            return (
                loss.detach().clone(),
                x.grad.detach().clone(),
                {n: p.grad.detach().clone() for n, p in module.named_parameters()},
            )

        loss_ref, x_grad_ref, grads_ref = run(with_manager=False)
        loss_c, x_grad_c, grads_c = run(with_manager=True)

        torch.testing.assert_close(loss_c, loss_ref)
        torch.testing.assert_close(x_grad_c, x_grad_ref)
        for name in grads_ref:
            torch.testing.assert_close(grads_c[name], grads_ref[name], msg=name)


class TestGatedResidualConfigValidation:

    def test_variant_must_be_known(self):
        with pytest.raises(ValueError, match="mhc_connection_variant"):
            make_config(mhc_connection_variant="nope")

    def test_lowrank_must_be_positive(self):
        with pytest.raises(ValueError, match="hc_lowrank"):
            make_config(hc_lowrank=0)

    def test_cuda_graphs_rejected(self):
        with pytest.raises(ValueError, match="CUDA graphs"):
            make_config(cuda_graph_impl="transformer_engine", cuda_graph_modules=["attn"])

    def test_fused_mhc_rejected(self):
        with pytest.raises(ValueError, match="use_fused_mhc"):
            make_config(use_fused_mhc=True)

    def test_sp_off_warns(self):
        with pytest.warns(UserWarning, match="sequence_parallel"):
            make_config(tensor_model_parallel_size=2, sequence_parallel=False)

    def test_mhc_variant_unaffected(self):
        cfg = make_config(mhc_connection_variant="mhc")
        assert cfg.mhc_connection_variant == "mhc"

    def test_variant_without_enable_warns(self):
        with pytest.warns(UserWarning, match="no effect without enable_mhc_connections"):
            make_config(enable_mhc_connections=False)


class TestParameterMarking:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sequence_parallel_attr(self):
        config = make_config(sequence_parallel=True, tensor_model_parallel_size=2)
        module = GatedResidualModule(config, layer_number=1)
        for name, param in module.named_parameters():
            assert getattr(param, 'sequence_parallel', False), name

    def test_fp32_marking_and_conversion(self):
        from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

        module = GatedResidualModule(make_config(), layer_number=1)
        convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
        assert module.hc_norm.weight.dtype == torch.float32
        assert module.block_inject_weight.weight.dtype == torch.float32
        assert module.input_mix_weight_down.weight.dtype == torch.bfloat16
        assert module.input_mix_weight_up.weight.dtype == torch.bfloat16

    def test_param_count(self):
        """6.60M at the real sizing (n=4, C=2560, r=320): sanity-check formula."""
        module = GatedResidualModule(make_config(), layer_number=1)
        nC = STREAMS * HIDDEN
        expected = nC + nC * LOWRANK * 2 + nC * STREAMS
        assert sum(p.numel() for p in module.parameters()) == expected
