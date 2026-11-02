# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for batch-invariant MoE grouped GEMM."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    _bf16_grouped_gemm_contiguous,
    _m_splits_to_m_indices,
    _offs_to_m_indices,
    set_batch_invariant_mode,
)
from tests.unit_tests.test_utilities import is_lts_ci_image


def _hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


pytestmark = [
    # DeepGEMM ships only in the dev CI image; there these tests fail without it.
    pytest.mark.skipif(
        is_lts_ci_image(), reason="DeepGEMM, which these tests need, is not in the LTS image."
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


def _build_moe_layer(hidden_size=64, ffn=128, num_experts=4, topk=1, **overrides):
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
        **overrides,
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


@pytest.mark.parametrize("grouped_tensor", [False, True])
def test_tegroupedmlp_per_token_invariance_across_batch_sizes(_moe_env, grouped_tensor):
    """The strongest batch-invariance check: a fixed set of "target" tokens must
    produce the *exact same output* regardless of what other tokens surround
    them in the batch. This is what RL log-prob parity needs. With grouped
    tensors the experts take TE's GroupedTensor path."""
    layer, cfg = _build_moe_layer(moe_use_grouped_tensor=grouped_tensor)
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


def _forbid_te_grouped_gemms(monkeypatch):
    """Make every TE cuBLASLt grouped GEMM raise, so a bypass fails the test."""
    import transformer_engine_torch as tex

    def forbidden(name):
        def raise_(*args, **kwargs):
            raise AssertionError(f"{name} ran TE's grouped GEMM under batch-invariant mode")

        return raise_

    for name in (
        "te_general_grouped_gemm",
        "te_general_grouped_gemm_for_grouped_tensor",
        "te_general_grouped_gemm_for_discrete_in",
        "te_general_grouped_gemm_for_discrete_out",
    ):
        if hasattr(tex, name):
            monkeypatch.setattr(tex, name, forbidden(name))


def _count_deepgemm_grouped_gemms(monkeypatch):
    calls = {"m_grouped": 0, "k_grouped": 0}
    for key, name in (
        ("m_grouped", "m_grouped_bf16_gemm_nt_contiguous"),
        ("k_grouped", "k_grouped_bf16_gemm_tn_contiguous"),
    ):

        def counted(*args, _key=key, _fn=getattr(bik.deep_gemm, name), **kwargs):
            calls[_key] += 1
            return _fn(*args, **kwargs)

        monkeypatch.setattr(bik.deep_gemm, name, counted)
    return calls


def _param_data(param) -> torch.Tensor:
    """A parameter's (or gradient's) values, unwrapping a TE GroupedTensor."""
    data = getattr(param, "rowwise_data", None)
    return param.data if data is None else data


def _legacy_expert_outputs(x, dy, weights, splits):
    """Forward, dgrad and fp32 wgrad of the legacy (host split sizes) grouped kernels."""
    m_indices = _m_splits_to_m_indices(splits, x.device, x.shape[0])
    out = _bf16_grouped_gemm_contiguous(x, weights, m_indices, splits)
    dx = _bf16_grouped_gemm_contiguous(dy, weights.transpose(1, 2).contiguous(), m_indices, splits)
    dw = bik._bf16_grouped_gemm_wgrad_contiguous(dy, x, splits)
    return out, dx, dw


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "counts, trailing",
    [([37, 0, 300, 5, 128], 0), ([0, 0, 1, 0, 0], 0), ([256] * 5, 0), ([37, 0, 300, 5, 128], 50)],
)
def test_grouped_gemm_device_counts_matches_host_counts(counts, trailing):
    """Reading the group sizes on the device gives the host-count kernels' bits, and rows past
    the groups' total (a caller's capacity buffer) are neither computed nor written."""
    torch.manual_seed(0)
    E, K, N, total = len(counts), 256, 192, sum(counts)
    x = torch.randn(total + trailing, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(total + trailing, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    m_indices = _m_splits_to_m_indices(counts, x.device, total)
    expected = _bf16_grouped_gemm_contiguous(x[:total], w, m_indices, counts)
    out = torch.full((total + trailing, N), 7.0, device="cuda", dtype=torch.bfloat16)
    device_counts = torch.tensor(counts, device="cuda", dtype=torch.int64)
    groups, valid = bik._bf16_grouped_gemm_device_counts(x, w, device_counts, out=out)
    assert torch.equal(out[:total], expected)
    assert torch.all(out[total:] == 7.0)
    assert torch.equal(groups[:total].to(torch.int32), m_indices)
    assert valid.sum().item() == total and bool(valid[:total].all())
    # On Blackwell this takes DeepGEMM's device psum layout; on Hopper the host path.
    dw = bik._bf16_grouped_gemm_wgrad_device_counts(dy, x, device_counts)
    assert dw.dtype == torch.float32
    assert torch.equal(dw, bik._bf16_grouped_gemm_wgrad_contiguous(dy[:total], x[:total], counts))


_GROUPED_LINEAR_VARIANTS = {
    "aligned": dict(splits=[256, 512, 0, 256]),
    "unaligned": dict(splits=[37, 0, 300, 5]),
    "single-weight": dict(splits=[37, 0, 300, 5], moe_single_grouped_weight=True),
    "bias": dict(splits=[37, 0, 300, 5], bias=True),
    "single-weight-and-bias": dict(
        splits=[37, 0, 300, 5],
        bias=True,
        moe_single_grouped_weight=True,
        moe_single_grouped_bias=True,
    ),
    "fp32-main-grad": dict(splits=[37, 0, 300, 5], gradient_accumulation_fusion=True),
    "single-weight-fp32-main-grad": dict(
        splits=[37, 0, 300, 5], moe_single_grouped_weight=True, gradient_accumulation_fusion=True
    ),
}


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("variant", list(_GROUPED_LINEAR_VARIANTS))
@pytest.mark.parametrize("backend", ["deepgemm", "triton"])
def test_te_grouped_linear_grouped_tensor_path_matches_legacy_kernels(
    _moe_env, monkeypatch, backend, variant
):
    """TE GroupedLinear's GroupedTensor path (moe_use_grouped_tensor=True) runs DeepGEMM's
    grouped GEMMs, never TE's, with the legacy kernels' bits in forward, dgrad and wgrad:
    discrete or single grouped weights, discrete or single grouped biases, and wgrad
    accumulated in fp32 into main_grad."""
    from transformer_engine.pytorch.module.grouped_linear import (
        is_module_grouped_tensor_path_supported,
    )

    from megatron.core.extensions.transformer_engine import (
        _TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR,
        TEGroupedLinear,
    )
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal

    # The dev image's TE and cuBLASLt take the GroupedTensor path on this GPU; without it
    # this test would check TE's legacy path.
    assert _TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR
    assert is_module_grouped_tensor_path_supported(None, torch.bfloat16)

    options = dict(_GROUPED_LINEAR_VARIANTS[variant])
    splits = options.pop("splits")
    bias = options.pop("bias", False)
    fused_wgrad = options.get("gradient_accumulation_fusion", False)
    E, k, n, total = len(splits), 512, 256, sum(splits)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=k,
        ffn_hidden_size=n,
        moe_ffn_hidden_size=n,
        num_attention_heads=4,
        num_moe_experts=E,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_use_grouped_tensor=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=bias,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        **options,
    )
    module = TEGroupedLinear(
        E,
        k,
        n,
        parallel_mode=None,
        config=config,
        init_method=init_method_normal(0.02),
        bias=bias,
        skip_bias_add=False,
        is_expert=True,
    ).cuda()
    single = options.get("moe_single_grouped_weight", False)
    weight_params = [module.weight] if single else [getattr(module, f"weight{i}") for i in range(E)]
    torch.manual_seed(3)
    if bias:
        bias_values = (torch.randn(E, n, device="cuda") * 0.1).to(torch.bfloat16)
        with torch.no_grad():
            if options.get("moe_single_grouped_bias", False):
                _param_data(module.bias).view(E, n).copy_(bias_values)
            else:
                for i in range(E):
                    getattr(module, f"bias{i}").copy_(bias_values[i])
    if fused_wgrad:
        # Accumulate (not overwrite) into a pre-filled fp32 main_grad.
        module.is_first_microbatch = False
        for param in weight_params:
            param.main_grad = torch.randn(param.shape, device="cuda", dtype=torch.float32)
        base = torch.cat([param.main_grad.reshape(-1) for param in weight_params])
        base = base.view(E, n, k).clone()
    weights = torch.cat([_param_data(param).reshape(-1) for param in weight_params]).view(E, n, k)
    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(total, n, device="cuda", dtype=torch.bfloat16)
    expected_out, expected_dx, expected_dw = _legacy_expert_outputs(x, dy, weights, splits)
    if bias:
        groups = _m_splits_to_m_indices(splits, x.device, total).long()
        expected_out = expected_out + bias_values[groups]
    _forbid_te_grouped_gemms(monkeypatch)
    calls = _count_deepgemm_grouped_gemms(monkeypatch)
    x_i = x.clone().requires_grad_(True)
    with set_batch_invariant_mode(True, backend=backend):
        out, _ = module(x_i, torch.tensor(splits, device="cuda", dtype=torch.int64))
        out.backward(dy)
    assert calls["m_grouped"] >= 2 and calls["k_grouped"] >= 1, calls
    assert torch.equal(out, expected_out)
    assert torch.equal(x_i.grad, expected_dx)
    if fused_wgrad:
        main_grad = torch.cat([param.main_grad.reshape(-1) for param in weight_params])
        assert torch.equal(main_grad.view(E, n, k), base + expected_dw)
    else:
        grads = torch.cat([_param_data(param.grad).reshape(-1) for param in weight_params])
        assert torch.equal(grads.view(E, n, k), expected_dw.to(torch.bfloat16))
    reference = torch.cat(
        [rows.double() @ weights[i].double().t() for i, rows in enumerate(torch.split(x, splits))]
    )
    if bias:
        reference = reference + bias_values.double()[groups]
    torch.testing.assert_close(out.double(), reference, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("bias", ["none", "bias", "scaled-bias"])
@pytest.mark.parametrize("backend", ["deepgemm", "triton"])
def test_te_op_fuser_grouped_linear_runs_batch_invariant_grouped_gemm(
    _moe_env, monkeypatch, backend, bias
):
    """The op fuser's GroupedLinear (use_transformer_engine_op_fuser experts) runs DeepGEMM's
    grouped GEMMs, never TE's, with the legacy kernels' bits in forward, dgrad and wgrad,
    including its grouped bias and per-token scaled bias."""
    import transformer_engine.pytorch.ops as te_ops
    from transformer_engine.pytorch.ops.basic.grouped_linear import (
        is_op_fuser_grouped_tensor_path_supported,
    )

    # BF16 takes the op fuser's GroupedTensor path in the dev image.
    assert is_op_fuser_grouped_tensor_path_supported(None, torch.bfloat16)
    splits = [256, 512, 0, 256]
    E, k, n, total = len(splits), 512, 256, sum(splits)
    op = te_ops.GroupedLinear(
        E,
        k,
        n,
        bias=bias != "none",
        scale_bias=bias == "scaled-bias",
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.manual_seed(4)
    bias_values = (torch.randn(E, n, device="cuda") * 0.1).to(torch.bfloat16)
    if bias != "none":
        with torch.no_grad():
            for i in range(E):
                getattr(op, f"bias{i}").copy_(bias_values[i])
    scales = torch.rand(total, device="cuda", dtype=torch.float32)
    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(total, n, device="cuda", dtype=torch.bfloat16)
    weights = torch.stack([getattr(op, f"weight{i}").detach() for i in range(E)])
    expected_out, expected_dx, expected_dw = _legacy_expert_outputs(x, dy, weights, splits)
    groups = _m_splits_to_m_indices(splits, x.device, total).long()
    if bias == "bias":
        expected_out = expected_out + bias_values[groups]
    elif bias == "scaled-bias":
        expected_out = expected_out.add_(bias_values[groups] * scales.unsqueeze(-1))
    _forbid_te_grouped_gemms(monkeypatch)
    calls = _count_deepgemm_grouped_gemms(monkeypatch)
    x_i = x.clone().requires_grad_(True)
    extra = (scales,) if bias == "scaled-bias" else ()
    with set_batch_invariant_mode(True, backend=backend):
        out = op(x_i, torch.tensor(splits, device="cuda", dtype=torch.int64), *extra)
        out.backward(dy)
    assert calls["m_grouped"] >= 2 and calls["k_grouped"] >= 1, calls
    assert torch.equal(out, expected_out)
    assert torch.equal(x_i.grad, expected_dx)
    for i in range(E):
        assert torch.equal(
            getattr(op, f"weight{i}").grad, expected_dw[i].to(torch.bfloat16)
        ), f"weight{i} gradient"


def test_batch_invariant_mode_rebinds_every_te_grouped_gemm(_moe_env, monkeypatch):
    """Enabling the mode replaces TE's grouped GEMMs in every loaded TE module that binds them,
    under any name (including the op fuser's fused grouped MLP), and disabling restores them."""
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm
    import transformer_engine.pytorch.ops.fused.grouped_mlp as te_grouped_mlp

    originals = (te_gemm.general_grouped_gemm, te_gemm.general_grouped_gemm_for_grouped_tensor)
    patched = (bik._te_general_grouped_gemm_patched, bik._te_grouped_tensor_gemm_patched)
    # A TE module that a fixed list of sites would not know about.
    probe = types.ModuleType("transformer_engine._batch_invariant_probe")
    probe.grouped, probe.grouped_tensor = originals
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    with set_batch_invariant_mode(True, backend="deepgemm"):
        assert te_grouped_mlp.general_grouped_gemm_for_grouped_tensor is patched[1]
        assert (probe.grouped, probe.grouped_tensor) == patched
        assert not bik._te_bindings_of(originals)
    assert (probe.grouped, probe.grouped_tensor) == originals
    assert te_grouped_mlp.general_grouped_gemm_for_grouped_tensor is originals[1]
    assert not bik._te_bindings_of(patched)


def test_unreplaceable_te_grouped_gemm_binding_fails_closed(_moe_env, monkeypatch):
    """A TE module whose grouped GEMM binding cannot be replaced makes the patch raise."""
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    class _Frozen(types.ModuleType):
        def __setattr__(self, name, value):
            pass  # silently keeps its binding

    probe = _Frozen("transformer_engine._batch_invariant_frozen_probe")
    probe.__dict__["grouped"] = te_gemm.general_grouped_gemm
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    try:
        with pytest.raises(RuntimeError, match="_batch_invariant_frozen_probe.grouped"):
            bik._te_patch_general_grouped_gemm()
    finally:
        bik._te_unpatch_general_grouped_gemm()
    assert te_gemm.general_grouped_gemm is not bik._te_general_grouped_gemm_patched
