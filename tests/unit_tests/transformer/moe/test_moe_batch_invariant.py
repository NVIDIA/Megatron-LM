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
