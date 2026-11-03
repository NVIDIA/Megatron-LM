# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for batch-invariant MoE grouped GEMM."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    HAVE_DEEPGEMM_BF16,
    _bf16_grouped_gemm_contiguous,
    _m_splits_to_m_indices,
    _offs_to_m_indices,
    grouped_gemm_batch_invariant,
    grouped_gemm_batch_invariant_alignment,
    set_batch_invariant_mode,
)


def _hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


pytestmark = [
    pytest.mark.skipif(
        not HAVE_DEEPGEMM_BF16,
        reason="DeepGEMM with bf16 grouped bindings is required for MoE batch-invariant tests.",
    ),
    pytest.mark.skipif(
        not _hopper_or_newer(), reason="DeepGEMM bf16 grouped kernels require Hopper (sm_90+)."
    ),
]


# ---------------------------------------------------------------------------
# Index-conversion helpers
# ---------------------------------------------------------------------------


def test_m_splits_to_m_indices_basic():
    m_splits = [3, 0, 5, 2]
    m_total = sum(m_splits)
    out = _m_splits_to_m_indices(m_splits, torch.device("cuda"), m_total)
    expected = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 3, 3], dtype=torch.int32, device="cuda")
    assert torch.equal(out, expected)


def test_offs_to_m_indices_basic():
    # Three experts with 4/2/3 tokens, plus 1 row of post-padding (-1).
    offs = torch.tensor([4, 6, 9], dtype=torch.int32, device="cuda")
    m_total = 10  # one trailing pad row past offs[-1]=9
    out = _offs_to_m_indices(offs, m_total)
    expected = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, -1], dtype=torch.int32, device="cuda")
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# Kernel-level invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E", [2, 4, 8])
def test_grouped_gemm_split_invariance(E):
    """Splitting the M dimension at expert boundaries must give bitwise-identical
    output to the full call."""
    torch.manual_seed(0)
    K, N = 128, 96
    # Build expert-grouped tokens: 32 tokens per expert
    per_expert = 32
    M = per_expert * E
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    m_indices = torch.repeat_interleave(
        torch.arange(E, device="cuda", dtype=torch.int32),
        torch.tensor([per_expert] * E, device="cuda", dtype=torch.int32),
    ).contiguous()

    counts = [per_expert] * E
    y_full = _bf16_grouped_gemm_contiguous(x, w, m_indices, counts)

    # Split into halves at expert boundaries (mid-expert split is not legal for
    # contiguous-layout DeepGEMM — must split on a boundary).
    half = (E // 2) * per_expert
    y0 = _bf16_grouped_gemm_contiguous(
        x[:half].contiguous(), w, m_indices[:half].contiguous(), counts[: E // 2] + [0] * (E // 2)
    )
    y1 = _bf16_grouped_gemm_contiguous(
        x[half:].contiguous(), w, m_indices[half:].contiguous(), [0] * (E // 2) + counts[E // 2 :]
    )
    y_cat = torch.cat([y0, y1], dim=0)
    assert torch.equal(
        y_full, y_cat
    ), f"max abs diff: {(y_full.float() - y_cat.float()).abs().max().item()}"


def test_grouped_gemm_per_expert_token_count_invariance():
    """For a fixed expert id, the per-row output must be identical regardless of
    how many *other* expert rows surround it in the batch."""
    torch.manual_seed(1)
    E, K, N = 4, 64, 48
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    x_target = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)

    # Layout A: just expert 1's tokens.
    m_indices_A = torch.full((8,), 1, dtype=torch.int32, device="cuda")
    y_A = _bf16_grouped_gemm_contiguous(x_target, w, m_indices_A, [0, 8, 0, 0])

    # Layout B: expert 0 (16 rows), then expert 1 (8 rows, same x_target),
    # then expert 3 (12 rows).
    x_pad0 = torch.randn(16, K, device="cuda", dtype=torch.bfloat16)
    x_pad3 = torch.randn(12, K, device="cuda", dtype=torch.bfloat16)
    x_B = torch.cat([x_pad0, x_target, x_pad3], dim=0).contiguous()
    m_indices_B = torch.cat(
        [
            torch.full((16,), 0, dtype=torch.int32, device="cuda"),
            torch.full((8,), 1, dtype=torch.int32, device="cuda"),
            torch.full((12,), 3, dtype=torch.int32, device="cuda"),
        ],
        dim=0,
    ).contiguous()
    y_B = _bf16_grouped_gemm_contiguous(x_B, w, m_indices_B, [16, 8, 0, 12])

    # The 8 rows assigned to expert 1 inside y_B must match y_A bitwise.
    y_B_target = y_B[16 : 16 + 8]
    assert torch.equal(y_A, y_B_target)


# ---------------------------------------------------------------------------
# Inference entry point: the offsets contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_offs", [1, 3], ids=["fewer_than_experts", "more_than_experts"])
def test_grouped_gemm_rejects_offsets_not_one_per_expert(num_offs):
    """offs must hold one end row per expert of the weights, checked before DeepGEMM runs."""
    align = grouped_gemm_batch_invariant_alignment()
    a = torch.randn(align, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
    # Every block past the first is empty, so no row is assigned past the weights' experts.
    offs = torch.full((num_offs,), align, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError, match="one end row per expert"):
        grouped_gemm_batch_invariant(a, w, offs=offs, offs_alignment=align, m_total=align)


@pytest.mark.parametrize("built_with", ["raw_counts", "torch_grouped_mm", "half", "zero"])
def test_grouped_gemm_rejects_offsets_built_with_another_alignment(built_with):
    """Offsets declared with an alignment DeepGEMM cannot use fail before DeepGEMM runs.

    Built from raw cumulative counts (alignment 1) or with torch grouped_mm's 16, expert 1's
    block would start inside a tile and silently take expert 0's weights.
    """
    align = grouped_gemm_batch_invariant_alignment()
    offs_alignment = {"raw_counts": 1, "torch_grouped_mm": 16, "half": align // 2, "zero": 0}[
        built_with
    ]
    counts = [align + 5, align]
    block = max(offs_alignment, 1)
    blocks = [-(-count // block) * block for count in counts]
    m_total = sum(blocks)
    a = torch.randn(m_total, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([blocks[0], m_total], device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError, match="must start at a multiple of"):
        grouped_gemm_batch_invariant(
            a, w, offs=offs, offs_alignment=offs_alignment, m_total=m_total
        )


def test_inference_moe_aligns_every_expert_block(monkeypatch):
    """The inference MoE hands the grouped GEMM the aligned expert blocks it declares.

    grouped_gemm_batch_invariant checks the declared alignment but cannot check the offset
    values without a host sync, so this pins the caller: expert counts around the alignment
    must reach both grouped GEMMs as aligned blocks, declared with DeepGEMM's alignment.
    """
    from megatron.core.inference.moe import batch_invariant
    from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

    torch.manual_seed(3)
    align = grouped_gemm_batch_invariant_alignment()
    counts = [align + 5, 1, 0, 2 * align + 1]
    num_experts, hidden_size, ffn = len(counts), 128, 128
    routes = torch.repeat_interleave(
        torch.arange(num_experts, device="cuda"), torch.tensor(counts, device="cuda")
    )
    num_tokens = routes.numel()
    routing_map = routes[torch.randperm(num_tokens, device="cuda")].unsqueeze(1)
    probs = torch.rand(num_tokens, 1, device="cuda")
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    fc1 = torch.randn(num_experts, ffn, hidden_size, device="cuda", dtype=torch.bfloat16)
    fc2 = torch.randn(num_experts, hidden_size, ffn, device="cuda", dtype=torch.bfloat16)
    valid_tokens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)

    seen = []
    grouped_gemm = batch_invariant.grouped_gemm_batch_invariant

    def recording_grouped_gemm(a, b, *, offs, offs_alignment, m_total):
        seen.append((offs.clone(), offs_alignment))
        return grouped_gemm(a, b, offs=offs, offs_alignment=offs_alignment, m_total=m_total)

    monkeypatch.setattr(batch_invariant, "grouped_gemm_batch_invariant", recording_grouped_gemm)
    with torch.no_grad(), set_batch_invariant_mode(True):
        mcore_fused_moe(
            x,
            probs,
            fc1,
            fc2,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            valid_tokens,
            routing_map,
        )

    blocks = torch.tensor([-(-count // align) * align for count in counts])
    expected = blocks.cumsum(0).to(device="cuda", dtype=torch.int32)
    assert len(seen) == 2, f"expected FC1 and FC2 grouped GEMMs, got {len(seen)} calls"
    for offs, offs_alignment in seen:
        assert offs_alignment == align, f"offs declared with alignment {offs_alignment}"
        assert torch.equal(offs, expected), f"offs {offs.tolist()} != {expected.tolist()}"


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


def test_tegroupedmlp_batch_invariant_split(_moe_env):
    """Splitting the batch and concatenating outputs must give bitwise-identical
    results to the full batch — the basic batch-invariance contract."""
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    layer, cfg = _build_moe_layer()
    assert isinstance(layer.experts, TEGroupedMLP)

    torch.manual_seed(0)
    M = 48
    x = torch.randn(M, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad(), set_batch_invariant_mode(True):
        y_full, _ = layer(x)
        y0, _ = layer(x[: M // 2])
        y1, _ = layer(x[M // 2 :])
    y_cat = torch.cat([y0, y1], dim=0)
    assert torch.equal(y_full, y_cat), (
        f"TEGroupedMLP not batch-invariant under halving; max abs diff: "
        f"{(y_full.float() - y_cat.float()).abs().max().item()}"
    )


def test_tegroupedmlp_per_token_invariance_across_batch_sizes(_moe_env):
    """The strongest batch-invariance check: a fixed set of "target" tokens must
    produce the *exact same output* regardless of what other tokens surround
    them in the batch. This is what RL log-prob parity needs."""
    layer, cfg = _build_moe_layer()
    torch.manual_seed(1)

    # 8 target tokens whose outputs we lock in by running them alone.
    target = torch.randn(8, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), set_batch_invariant_mode(True):
        y_target_alone, _ = layer(target)

    # Now embed those same 8 tokens at different positions inside larger batches
    # of varying sizes, with random surrounding tokens.
    for pad_left, pad_right in [(0, 16), (40, 0), (24, 24), (5, 13), (1, 1)]:
        left = torch.randn(pad_left, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(pad_right, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        big = torch.cat([left, target, right], dim=0)
        with torch.no_grad(), set_batch_invariant_mode(True):
            y_big, _ = layer(big)
        y_target_in_big = y_big[pad_left : pad_left + 8]
        assert torch.equal(y_target_alone, y_target_in_big), (
            f"Per-token output drifted when batch shape changed "
            f"(pad_left={pad_left}, pad_right={pad_right}); "
            f"max abs diff: "
            f"{(y_target_alone.float() - y_target_in_big.float()).abs().max().item()}"
        )


def test_tegroupedmlp_invariance_under_permutation(_moe_env):
    """Permuting the input batch and undoing the permutation in the output
    yields bitwise-identical results. Different routing distribution per
    micro-batch position, same kernel output."""
    layer, cfg = _build_moe_layer()
    torch.manual_seed(2)
    M = 32
    x = torch.randn(M, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    perm = torch.randperm(M, device="cuda")
    with torch.no_grad(), set_batch_invariant_mode(True):
        y_ref, _ = layer(x)
        y_perm, _ = layer(x[perm])
    y_unperm = y_perm[perm.argsort()]
    assert torch.equal(y_ref, y_unperm), (
        f"MoE output not invariant to batch permutation; max abs diff: "
        f"{(y_ref.float() - y_unperm.float()).abs().max().item()}"
    )
