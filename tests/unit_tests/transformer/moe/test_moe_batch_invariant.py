# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant MoE grouped GEMMs: training's TE grouped GEMM and inference's grouped_mm."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

BACKENDS = ["te_native", "triton"]


# ---------------------------------------------------------------------------
# Inference: torch grouped_mm (the torch backend's BF16 expert GEMM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm")
@pytest.mark.parametrize("k,n", [(2688, 1856), (1856, 2688)])
def test_grouped_mm_rows_do_not_depend_on_other_rows(k, n):
    """A row's grouped_mm output does not depend on the other rows of its group nor on
    the other groups: the batch-invariance the inference fused MoE relies on."""
    torch.manual_seed(0)
    num_experts = 8
    w = (torch.randn(num_experts, n, k, device="cuda") * 0.02).bfloat16().transpose(-2, -1)
    probe = torch.randn(8, k, device="cuda").bfloat16()

    def probe_rows(counts, group):
        """grouped_mm of the probe rows placed first in ``group``, which uses w[0]."""
        x = torch.randn(sum(counts), k, device="cuda").bfloat16()
        start = sum(counts[:group])
        x[start : start + 8] = probe
        weights = w.clone()
        weights[group] = w[0]
        offs = torch.tensor(counts, device="cuda").cumsum(0).int()
        return torch._grouped_mm(x, weights, offs=offs)[start : start + 8]

    ref = probe_rows([8] + [0] * (num_experts - 1), 0)
    for counts, group in (
        ([8] + [300] * 7, 0),
        ([4096] + [16] * 7, 0),
        ([8, 9, 33, 777, 1, 0, 64, 5000], 0),
        ([128, 64, 500, 500, 500, 500, 500, 500], 1),
    ):
        assert torch.equal(probe_rows(counts, group), ref), (counts, group)


# ---------------------------------------------------------------------------
# End-to-end: TEGroupedMLP batch-invariance
# ---------------------------------------------------------------------------


@pytest.fixture
def _moe_env():
    """Spin up a tiny model-parallel env for MoELayer construction."""
    from megatron.core.utils import is_te_min_version
    from tests.unit_tests.test_utilities import Utils

    if not is_te_min_version("1.9.0.dev0"):
        pytest.skip("TE GroupedLinear requires TE >= 1.9.0.dev0")
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


def _build_moe_layer(hidden_size=64, ffn=128, num_experts=4, topk=1):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=ffn,
        moe_grouped_gemm=True,
        moe_router_topk=topk,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="sinkhorn",
        gated_linear_unit=False,
        activation_func=F.gelu,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
        batch_invariant_mode=True,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            cfg.num_moe_experts, moe_grouped_gemm=True
        ).mlp
    )
    return MoELayer(cfg, submodules).cuda().eval(), cfg


@pytest.mark.parametrize("backend", BACKENDS)
def test_tegroupedmlp_batch_invariant_split(_moe_env, backend):
    """Splitting the batch and concatenating outputs must give bitwise-identical
    results to the full batch — the basic batch-invariance contract."""
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    layer, cfg = _build_moe_layer()
    assert isinstance(layer.experts, TEGroupedMLP)

    torch.manual_seed(0)
    M = 48
    x = torch.randn(M, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad(), set_batch_invariant_mode(True, backend=backend):
        y_full, _ = layer(x)
        y0, _ = layer(x[: M // 2])
        y1, _ = layer(x[M // 2 :])
    y_cat = torch.cat([y0, y1], dim=0)
    assert torch.equal(y_full, y_cat), (
        f"TEGroupedMLP not batch-invariant under halving; max abs diff: "
        f"{(y_full.float() - y_cat.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_tegroupedmlp_per_token_invariance_across_batch_sizes(_moe_env, backend):
    """The strongest batch-invariance check: a fixed set of "target" tokens must
    produce the *exact same output* regardless of what other tokens surround
    them in the batch. This is what RL log-prob parity needs."""
    layer, cfg = _build_moe_layer()
    torch.manual_seed(1)

    # 8 target tokens whose outputs we lock in by running them alone.
    target = torch.randn(8, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), set_batch_invariant_mode(True, backend=backend):
        y_target_alone, _ = layer(target)

    # Now embed those same 8 tokens at different positions inside larger batches
    # of varying sizes, with random surrounding tokens.
    for pad_left, pad_right in [(0, 16), (40, 0), (24, 24), (5, 13), (1, 1)]:
        left = torch.randn(pad_left, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(pad_right, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        big = torch.cat([left, target, right], dim=0)
        with torch.no_grad(), set_batch_invariant_mode(True, backend=backend):
            y_big, _ = layer(big)
        y_target_in_big = y_big[pad_left : pad_left + 8]
        assert torch.equal(y_target_alone, y_target_in_big), (
            f"Per-token output drifted when batch shape changed "
            f"(pad_left={pad_left}, pad_right={pad_right}); "
            f"max abs diff: "
            f"{(y_target_alone.float() - y_target_in_big.float()).abs().max().item()}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_tegroupedmlp_invariance_under_permutation(_moe_env, backend):
    """Permuting the input batch and undoing the permutation in the output
    yields bitwise-identical results. Different routing distribution per
    micro-batch position, same kernel output."""
    layer, cfg = _build_moe_layer()
    torch.manual_seed(2)
    M = 32
    x = torch.randn(M, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    perm = torch.randperm(M, device="cuda")
    with torch.no_grad(), set_batch_invariant_mode(True, backend=backend):
        y_ref, _ = layer(x)
        y_perm, _ = layer(x[perm])
    y_unperm = y_perm[perm.argsort()]
    assert torch.equal(y_ref, y_unperm), (
        f"MoE output not invariant to batch permutation; max abs diff: "
        f"{(y_ref.float() - y_unperm.float()).abs().max().item()}"
    )
