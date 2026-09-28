# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 operator gates before block/stack/logit parity.

Run under torchrun, one CUDA rank. Forward and backward are separate cases so a
forward failure never hides input/parameter gradient diagnostics. No checkpoint,
HF kernel dependency, production monkeypatch, or reference alignment patch is
needed. Numerical acceptance uses cosine similarity; magnitude-sensitive errors
remain in the report. Dtype and structural checks are independent of cosine.

Reference: deepseek-ai/DeepSeek-V4.1-Flash, revision
dba1be0a40aa45a94ad051997016db3960a90277, inference/model.py.
The inline _ref_* helpers preserve the published cast locations and use only
PyTorch formulas, without calling Megatron math helpers. Their backward results
are derivatives of those formulas, not results from an HF training backend.
"""

import json
import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa import (
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    csa_sparse_attn,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config

# Accept cosine >= 1 - tolerance for each output/gradient tensor. The numeric
# tolerances are retained from the initial relative-L2 run, but the two metrics
# do NOT have equivalent strictness. Cosine ignores a positive scale error.
# Separate dtype/rounding-contract checks remain independent acceptance gates.
_BF16_U = torch.finfo(torch.bfloat16).eps / 2
_COSINE_TOLS = {
    (torch.float32, "fwd"): 2e-5,
    (torch.float32, "bwd"): 5e-5,
    (torch.bfloat16, "fwd"): 2 * _BF16_U,
    (torch.bfloat16, "bwd"): 4 * _BF16_U,
}
_PATHS = [
    pytest.param(torch.float32, False, id="fp32-native"),
    pytest.param(torch.bfloat16, False, id="bf16-native"),
    pytest.param(torch.bfloat16, True, id="bf16-fused"),
]


def _ref_rms_norm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def _ref_compressor(x, weights, ratio, eps):
    """HF prefill compressor; x is [sequence, batch, hidden].

    For ratio=2 both projections precede the BF16 cast and run in FP32. The
    incomplete final group has no compressed output or compressed-path gradient.
    Reference weights must be cloned as FP32 for ratio=2, BF16 for ratio=1.
    """
    if ratio == 1:
        latent = F.linear(x, weights["linear_wkv.weight"])
    else:
        complete = x.shape[0] // ratio * ratio
        xf = x[:complete].float()
        values = F.linear(xf, weights["linear_wkv.weight"])
        scores = F.linear(xf, weights["linear_wgate.weight"])
        shape = (complete // ratio, ratio, *values.shape[1:])
        latent = (values.reshape(shape) * scores.reshape(shape).softmax(1)).sum(1).to(x.dtype)
    return _ref_rms_norm(latent, weights["norm.weight"], eps)


def _ref_rotary(x, config, *, compressed, stride=1, inverse=False):
    """Adjacent-pair complex rotation in FP32, followed by exactly one cast.

    Generate frequencies independently from HF's formula, including its YaRN
    ramp, instead of reading the implementation's cached frequency table.
    """
    dim = config.qk_pos_emb_head_dim
    base = config.csa_compress_rotary_base if compressed else config.rotary_base
    exponents = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    frequencies = 1.0 / base**exponents
    if compressed:

        def correction(rotations):
            return (
                dim
                * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=x.device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequencies = frequencies / config.rotary_scaling_factor * ramp + frequencies * (1 - ramp)
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.float32) * stride
    angles = torch.outer(positions, frequencies)
    cis = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        cis = cis.conj()
    shape = (x.shape[0], *([1] * (x.ndim - 2)), dim // 2)
    pairs = torch.view_as_complex(x[..., -dim:].float().contiguous().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(pairs * cis.reshape(shape)).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-dim], rotated), -1)


def _ref_indexer_projections(x, qr, latent, weights, config, ratio):
    q = F.linear(qr, weights["linear_wq_b.weight"])
    q = q.reshape(*q.shape[:-1], config.dsa_indexer_n_heads, config.dsa_indexer_head_dim)
    k = _ref_rms_norm(
        F.linear(latent, weights["linear_wk.weight"]),
        weights["k_norm.weight"],
        config.attention_latent_norm_epsilon,
    )
    q = _ref_rotary(q, config, compressed=True)
    k = _ref_rotary(k, config, compressed=True, stride=ratio)
    return q, k, F.linear(x, weights["linear_weights_proj.weight"])


def _ref_indexer_scores(q, k, weights, ratio):
    """Published BF16 score arithmetic, before discrete Top-K.

    Multiplication and reduction remain in the activation dtype. In particular,
    upcasting Q/K/weights before these operations would change the HF baseline.
    """
    scaled = weights * (q.shape[-1] ** -0.5 * q.shape[-2] ** -0.5)
    scores = torch.einsum("sbhd,tbd->bsht", q, k).relu()
    scores = (scores * scaled.permute(1, 0, 2).unsqueeze(-1)).sum(2)
    visible = (torch.arange(q.shape[0], device=q.device) + 1) // ratio
    valid = torch.arange(k.shape[0], device=q.device)[None, :] < visible[:, None]
    return scores.masked_fill(~valid, -torch.inf)


def _ref_sparse_attention(q, kv, sink, indices, scale):
    """Dense FP32 mathematical oracle with sink and a sparse multiset of keys.

    Layout: q [tokens, heads, dim], kv [keys, dim], indices [tokens, slots].
    Counts preserve duplicate key slots, unlike a boolean dense mask. The sink
    has a logit but zero value. This is a mathematical oracle, not a bitwise
    emulation of HF's tiled softmax/probability rounding.
    """
    counts = F.one_hot(indices.clamp_min(0).long(), kv.shape[0]).float()
    counts = (counts * (indices >= 0).unsqueeze(-1)).sum(1)
    scores = torch.einsum("thd,kd->thk", q.float(), kv.float()) * scale
    # log(count) is equivalent to repeating a key that many times in softmax.
    scores = scores + counts.log().unsqueeze(1)
    sink_logits = sink.float().view(1, -1, 1).expand(q.shape[0], -1, -1)
    probabilities = torch.cat((scores, sink_logits), -1).softmax(-1)[..., :-1]
    return torch.einsum("thk,kd->thd", probabilities, kv.float()).to(q.dtype).flatten(1)


def _ref_mhc_projection(x, weight, alpha_pre, alpha_post, alpha_res, bias, streams, eps):
    """HF hc_mixes projection/split before Sinkhorn, with FP32 coefficients.

    V4.1 puts epsilon inside the RMS square root. Residual logits remain FP32;
    their output must meet an FP32 budget even when activations arrive in BF16.
    """
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    projection = F.linear(xf, weight) * rstd
    pre = (projection[:, :streams] * alpha_pre + bias[:streams]).sigmoid() + eps
    post = (
        projection[:, streams : 2 * streams] * alpha_post + bias[streams : 2 * streams]
    ).sigmoid() * 2
    residual = projection[:, 2 * streams :] * alpha_res + bias[2 * streams :]
    return pre, post, residual, rstd.reciprocal()


def _ref_router(x, weight, expert_bias, topk, scale):
    """HF Gate: FP32 scores, selection-only bias, unbiased normalized weights."""
    logits = F.linear(x.reshape(-1, x.shape[-1]).float(), weight.float())
    scores = F.softplus(logits).sqrt()
    indices = (scores + expert_bias.float()).topk(topk, dim=-1).indices
    selected = scores.gather(1, indices)
    if topk > 1:
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    probabilities = torch.zeros_like(scores).scatter(1, indices, selected * scale)
    return logits, probabilities, indices


def _ref_expert(x, fc1, fc2, limit, probabilities=None):
    """HF Expert: two input GEMMs, FP32 clamped SwiGLU/probs, one BF16 cast."""
    w1, w3 = fc1.chunk(2, dim=0)
    gate, up = F.linear(x, w1).float(), F.linear(x, w3).float()
    gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    activation = F.silu(gate) * up
    if probabilities is not None:
        activation = activation * probabilities
    return F.linear(activation.to(x.dtype), fc2)


def _ref_moe(x, weights, expert_bias, config):
    """Differentiable HF MoE formula; accumulate routed and shared outputs in FP32."""
    flat = x.reshape(-1, x.shape[-1])
    _, probabilities, indices = _ref_router(
        flat,
        weights["router"],
        expert_bias,
        config.moe_router_topk,
        config.moe_router_topk_scaling_factor,
    )
    output = torch.zeros_like(flat, dtype=torch.float32)
    for expert in range(config.num_moe_experts):
        rows = (indices == expert).nonzero()[:, 0]
        if not rows.numel():
            continue
        partial = _ref_expert(
            flat[rows],
            weights[f"expert.{expert}.fc1"],
            weights[f"expert.{expert}.fc2"],
            config.activation_func_clamp_value,
            probabilities[rows, expert, None],
        )
        # Each expert contributes at most once to each row, as in HF's y[idx] +=.
        output = output.index_add(0, rows, partial.float())
    shared = _ref_expert(
        flat, weights["shared.fc1"], weights["shared.fc2"], config.activation_func_clamp_value
    )
    return (output + shared.float()).to(x.dtype).reshape_as(x), probabilities, indices


def _moe_weight_tensors(module, *, gradients=False):
    """Expose per-expert checkpoint/gradient views without changing the live parameters."""

    def value(parameter):
        return parameter.grad if gradients else parameter

    result = {"router": value(module.router.weight)}
    for fc in ("fc1", "fc2"):
        result[f"shared.{fc}"] = value(getattr(module.shared_experts, f"linear_{fc}").weight)
        if hasattr(module.experts, "local_experts"):
            tensors = [
                value(getattr(e, f"linear_{fc}").weight) for e in module.experts.local_experts
            ]
        else:
            linear = getattr(module.experts, f"linear_{fc}")
            if linear.single_grouped_weight:
                parameter = value(linear.weight)
                tensors = (
                    [None] * module.config.num_moe_experts
                    if parameter is None
                    else linear._split_grouped_checkpoint_tensor(parameter, f"linear_{fc}.weight")
                )
            else:
                tensors = [
                    value(getattr(linear, f"weight{i}"))
                    for i in range(module.config.num_moe_experts)
                ]
        result.update({f"expert.{i}.{fc}": tensor for i, tensor in enumerate(tensors)})
    return result


@pytest.fixture(scope="module")
def groups():
    if not torch.cuda.is_available() or not HAVE_TE:
        pytest.skip("DSv4.1 native parity requires CUDA and Transformer Engine")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()
    torch.backends.cuda.matmul.allow_tf32 = old_matmul
    torch.backends.cudnn.allow_tf32 = old_cudnn


def _layer(groups, ratio, dtype, fused):
    torch.manual_seed(4101)
    config = _make_config(
        params_dtype=dtype,
        num_layers=1,
        hidden_size=128,
        q_lora_rank=64,
        num_attention_heads=64,
        v_head_dim=512,
        qk_pos_emb_head_dim=64,
        o_groups=8,
        o_lora_rank=32,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=7,
        layernorm_epsilon=1e-6,
        attention_latent_norm_epsilon=1e-6,
        csa_compress_ratios=[ratio],
        csa2_kv_source_layers=[0] if ratio else [],
        csa2_index_source_layers=[0] if ratio else [],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_kernel_backend="cudnn" if fused else "none",
        apply_rope_fusion=fused,
    )
    layer = build_module(
        get_experimental_attention_variant_module_spec(config),
        config=config,
        layer_number=1,
        pg_collection=groups,
    ).cuda()
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "norm.weight" in name:
                parameter.uniform_(0.7, 1.3)
            else:
                parameter.normal_(std=1 / math.sqrt(parameter.shape[-1]))
    return layer


def _clone(tensor, dtype=None):
    return tensor.detach().to(dtype or tensor.dtype).clone().requires_grad_()


def _weights(module, fp32_projections=False):
    return {
        name: _clone(p, torch.float32 if fp32_projections and name.startswith("linear_") else None)
        for name, p in module.named_parameters()
    }


def _metrics(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    error = (a - b).norm()
    denom = b.norm()
    dot = a.dot(b)
    norm_product = a.norm() * denom
    energy = a.square().sum() + b.square().sum()
    return {
        "rel_l2": (error / denom.clamp_min(1e-30)).item(),
        # Both zero is an exact match; only one zero has no matching direction.
        # Clamp away tiny FP64 reduction overshoots beyond [-1, 1].
        "cosine": (dot / norm_product).clamp(-1, 1).item() if norm_product else float(error == 0),
        "tensor_similarity": (2 * dot / energy).item() if energy else 1.0,
        "max_abs": (a - b).abs().max().item(),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
    }


def _check(pairs, dtype, phase, record_property, *, cosine_tols=None, **metadata):
    measurements = {name: _metrics(a, b) for name, (a, b) in pairs.items()}
    cosine_tols = cosine_tols or {name: _COSINE_TOLS[dtype, phase] for name in pairs}
    minimum_cosines = {name: 1 - tolerance for name, tolerance in cosine_tols.items()}
    result = {
        "phase": phase,
        "acceptance_metric": "cosine",
        "cosine_tolerances": cosine_tols,
        "minimum_cosines": minimum_cosines,
        **metadata,
        "tensors": measurements,
    }
    record_property("parity", json.dumps(result, sort_keys=True))
    print("DSV41_PARITY " + json.dumps(result, sort_keys=True))
    failures = {name: m for name, m in measurements.items() if m["cosine"] < minimum_cosines[name]}
    assert not failures, json.dumps(
        {"minimum_cosines": minimum_cosines, "failures": failures}, indent=2
    )


def _gradient_pairs(actual, expected, actual_inputs, ref_inputs, probe):
    assert actual_inputs.keys() == ref_inputs.keys()

    # Fused RoPE can consume grad_output in place. Give both graphs independent
    # copies so the reference receives the same VJP, including with BF16 I/O.
    def copy_probe():
        return tuple(p.clone() for p in probe) if isinstance(probe, tuple) else probe.clone()

    da = torch.autograd.grad(actual, tuple(actual_inputs.values()), copy_probe())
    db = torch.autograd.grad(expected, tuple(ref_inputs.values()), copy_probe())
    return {name: (a, b) for name, a, b in zip(actual_inputs, da, db)}


@pytest.mark.parametrize("dtype,fused", _PATHS)
@pytest.mark.parametrize("topk", [1, 6])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_router_native_parity(groups, dtype, fused, topk, phase, record_property):
    """Compare the actual router, including selection, under identical inputs."""
    config = _make_config(
        params_dtype=dtype,
        hidden_size=128,
        num_moe_experts=16,
        moe_router_topk=topk,
        moe_router_fusion=fused,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
    )
    torch.manual_seed(4110)
    router = TopKRouter(config, pg_collection=groups).cuda().eval()
    with torch.no_grad():
        router.weight.normal_(std=1 / math.sqrt(config.hidden_size))
        router.expert_bias.uniform_(-0.5, 0.5)
    x = torch.randn(7, 1, config.hidden_size, device="cuda", dtype=dtype).requires_grad_()
    xr, wr = _clone(x), _clone(router.weight)
    expected_logits, expected, indices = _ref_router(
        xr, wr, router.expert_bias, topk, config.moe_router_topk_scaling_factor
    )
    actual, routing_map = router(x)
    expected_map = torch.zeros_like(routing_map).scatter(1, indices, True)
    assert torch.equal(routing_map, expected_map), "Same-input router selected different experts"
    if phase == "fwd":
        _check(
            {
                "logits": (router.gating(x).flatten(0, 1), expected_logits),
                "probabilities": (actual, expected),
            },
            torch.float32,
            phase,
            record_property,
            input_dtype=str(dtype),
            fused=fused,
            topk=topk,
        )
    else:
        pairs = _gradient_pairs(
            actual,
            expected,
            {"input": x, "weight": router.weight},
            {"input": xr, "weight": wr},
            torch.randn_like(expected),
        )
        _check(pairs, dtype, phase, record_property, fused=fused, topk=topk)


_MOE_PATHS = [
    pytest.param(torch.float32, "sequential", id="fp32-sequential"),
    pytest.param(torch.bfloat16, "sequential", id="bf16-sequential"),
    pytest.param(torch.bfloat16, "grouped", id="bf16-grouped"),
    pytest.param(torch.bfloat16, "op-fuser", id="bf16-op-fuser"),
]


@pytest.mark.parametrize("dtype,backend", _MOE_PATHS)
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_moe_native_parity(groups, monkeypatch, dtype, backend, phase, record_property):
    """Full router/dispatch/routed+shared MoE parity, with an unused expert as well."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    grouped, fused = backend != "sequential", backend != "sequential"
    config = _make_config(
        params_dtype=dtype,
        hidden_size=128,
        num_moe_experts=16,
        moe_router_topk=6,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_token_dispatcher_type="allgather",
        moe_grouped_gemm=grouped,
        moe_router_fusion=fused,
        moe_permute_fusion=fused,
        bias_activation_fusion=fused,
        gradient_accumulation_fusion=False,
        moe_shared_expert_overlap=False,
        moe_shared_expert_gate=False,
        use_transformer_engine_op_fuser=backend == "op-fuser",
        moe_single_grouped_weight=backend == "op-fuser",
        # Match the E2E construction path. TE packed member views created on
        # CPU can retain CPU storage across a later Module.cuda() conversion.
        use_cpu_initialization=False,
    )
    torch.manual_seed(4111)
    module = (
        get_moe_module_spec(use_te=True, num_experts=16, moe_grouped_gemm=grouped)(
            config=config, pg_collection=groups
        )
        .cuda()
        .eval()
    )
    if dtype == torch.bfloat16:
        Float16Module(config, module)
    actual_weights = _moe_weight_tensors(module)
    assert all(parameter.is_cuda for parameter in actual_weights.values())
    with torch.no_grad():
        for parameter in actual_weights.values():
            parameter.normal_(std=1 / math.sqrt(parameter.shape[-1]))
        module.router.expert_bias.uniform_(-0.05, 0.05)
        module.router.expert_bias[-1] = -100
    weights = {name: _clone(parameter) for name, parameter in actual_weights.items()}
    x = (4 * torch.randn(7, 1, config.hidden_size, device="cuda", dtype=dtype)).requires_grad_()
    xr = _clone(x)
    expected, ref_probs, ref_ids = _ref_moe(xr, weights, module.router.expert_bias, config)
    observed = {}

    def capture_route(_module, _inputs, output):
        observed["probabilities"], observed["routing_map"] = output

    handle = module.router.register_forward_hook(capture_route)
    try:
        actual, bias = module(x)
    finally:
        handle.remove()
    assert bias is None
    expected_map = torch.zeros_like(observed["routing_map"]).scatter(1, ref_ids, True)
    assert torch.equal(observed["routing_map"], expected_map), "Same-input expert choices differ"
    assert not expected_map[:, -1].any(), "Test must include an unused expert"
    if grouped:
        assert module.experts._with_fused_impl == (backend == "op-fuser")
    if phase == "fwd":
        _check({"output": (actual, expected)}, dtype, phase, record_property, backend=backend)
        per_token = F.cosine_similarity(actual.detach().double(), expected.detach().double(), -1)
        assert per_token.min() >= 1 - _COSINE_TOLS[dtype, phase]
        _check(
            {"router_probabilities": (observed["probabilities"], ref_probs)},
            torch.float32,
            phase,
            record_property,
            backend=backend,
        )
    else:
        probe = torch.randn_like(expected)
        actual.backward(probe.clone())
        expected.backward(probe.clone())
        pairs = {"input": (x.grad, xr.grad)}
        gradients = _moe_weight_tensors(module, gradients=True)
        for name, parameter in weights.items():
            ag, rg = gradients[name], parameter.grad
            if rg is None:
                assert ag is None or torch.count_nonzero(ag) == 0, name
            else:
                assert ag is not None, f"Missing parameter gradient: {name}"
                pairs[name] = (ag, rg)
        _check(pairs, dtype, phase, record_property, backend=backend)


@pytest.mark.parametrize("dtype,fused", _PATHS)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_compressor_native_parity(groups, dtype, fused, ratio, phase, record_property):
    module = _layer(groups, ratio, dtype, fused).core_attention.compressor
    torch.manual_seed(4102)
    x = torch.randn(33, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
    xr = _clone(x)
    weights = _weights(module, fp32_projections=ratio == 2)
    actual = module(x)
    expected = _ref_compressor(xr, weights, ratio, module.config.attention_latent_norm_epsilon)
    pairs = {"output": (actual, expected)}
    if phase == "bwd":
        pairs = _gradient_pairs(
            actual,
            expected,
            {"input": x, **dict(module.named_parameters())},
            {"input": xr, **weights},
            torch.randn_like(actual),
        )
        if ratio == 2:
            assert torch.count_nonzero(pairs["input"][0][-1]) == 0
            assert torch.count_nonzero(pairs["input"][1][-1]) == 0
    _check(pairs, dtype, phase, record_property, ratio=ratio, fused=fused)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_training_projection_dtype(groups, ratio):
    """The BF16 training policy intentionally differs from HF's FP32 r2 GEMMs.

    Numerical tests above still compare against the unmodified FP32 HF oracle.
    """
    layer = _layer(groups, ratio, torch.bfloat16, False)
    module = layer.core_attention.compressor
    expected = torch.bfloat16
    snapshots = {name: p.detach().clone() for name, p in module.named_parameters()}
    for name, p in module.named_parameters():
        if name.startswith("linear_"):
            assert p.dtype == expected
    Float16Module(config=layer.config, module=layer)
    assert layer.config.params_dtype == torch.bfloat16
    for name, p in module.named_parameters():
        assert p.dtype == (expected if name.startswith("linear_") else torch.bfloat16)
        torch.testing.assert_close(p, snapshots[name], atol=0, rtol=0)
    projected_dtypes, norm_dtypes = [], []
    handles = [
        linear.register_forward_hook(
            lambda _, args, output: projected_dtypes.append((args[0].dtype, output[0].dtype))
        )
        for linear in (module.linear_wkv, module.linear_wgate)
        if linear is not None
    ]
    handles.append(
        module.norm.register_forward_pre_hook(lambda _, args: norm_dtypes.append(args[0].dtype))
    )
    try:
        output = module(torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16))
    finally:
        for handle in handles:
            handle.remove()
    assert projected_dtypes == [(expected, expected)] * ratio
    assert norm_dtypes == [torch.bfloat16]
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype,fused", _PATHS)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_indexer_projection_native_parity(groups, dtype, fused, ratio, phase, record_property):
    layer = _layer(groups, ratio, dtype, fused)
    module = layer.core_attention.indexer
    torch.manual_seed(4103)
    inputs = {
        "x": torch.randn(33, 2, 128, device="cuda", dtype=dtype, requires_grad=True),
        "qr": torch.randn(33, 2, 64, device="cuda", dtype=dtype, requires_grad=True),
        "latent": torch.randn(33 // ratio, 2, 512, device="cuda", dtype=dtype, requires_grad=True),
    }
    copies = {name: _clone(value) for name, value in inputs.items()}
    weights = _weights(module)
    actual = module._project_inputs(*inputs.values(), layer.core_attention.rotary_pos_emb)
    expected = _ref_indexer_projections(*copies.values(), weights, module.config, ratio)
    pairs = {name: (a, b) for name, a, b in zip(("q", "k", "weights"), actual, expected)}
    if phase == "bwd":
        probes = tuple(torch.randn_like(value) for value in actual)
        pairs = _gradient_pairs(
            actual,
            expected,
            {**inputs, **dict(module.named_parameters())},
            {**copies, **weights},
            probes,
        )
    _check(pairs, dtype, phase, record_property, ratio=ratio, rope_fused=fused)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_indexer_score_native_parity(groups, dtype, ratio, phase, record_property):
    """Continuous scores/gradients; Top-K itself has no derivative.

    This is a diagnostic VJP, not a claim that HF defines MCore's indexer loss.
    Main-attention loss and the auxiliary training objective are separate gates.
    """
    module = _layer(groups, ratio, dtype, False).core_attention.indexer
    torch.manual_seed(4104)
    inputs = {
        "q": torch.randn(65, 2, 32, 128, device="cuda", dtype=dtype, requires_grad=True),
        "k": torch.randn(65 // ratio, 2, 128, device="cuda", dtype=dtype, requires_grad=True),
        "weights": torch.randn(65, 2, 32, device="cuda", dtype=dtype, requires_grad=True),
    }
    copies = {name: _clone(value) for name, value in inputs.items()}
    actual = module._score_projected(*inputs.values())
    expected = _ref_indexer_scores(*copies.values(), ratio)
    valid = expected.isfinite()
    assert torch.equal(actual.isfinite(), valid), "causal mask mismatch"
    pairs = {"scores": (actual[valid], expected[valid])}
    if phase == "bwd":
        pairs = _gradient_pairs(
            actual[valid], expected[valid], inputs, copies, torch.randn_like(actual[valid])
        )
    _check(pairs, dtype, phase, record_property, ratio=ratio)


@pytest.mark.parametrize("dtype,fused", _PATHS)
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_sparse_attention_native_parity(groups, dtype, fused, phase, record_property):
    """Fixed sparse IDs isolate attention from discrete indexer decisions.

    Include invalid slots, a sink-only query, and duplicate IDs. Dense one-hot
    counts provide an independent oracle and accumulate repeated-KV grads in
    FP32, matching CSA2's production adapter rather than BF16 gather backward.
    """
    torch.manual_seed(4105)
    inputs = {
        "q": torch.randn(33, 64, 512, device="cuda", dtype=dtype, requires_grad=True),
        "kv": torch.randn(41, 512, device="cuda", dtype=dtype, requires_grad=True),
        "sink": torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True),
    }
    copies = {name: _clone(value) for name, value in inputs.items()}
    indices = torch.randint(0, 41, (33, 128), device="cuda", dtype=torch.int32)
    indices[:, 17:] = -1
    indices[0] = -1
    indices[1, :2] = 3
    scale = 512**-0.5
    if fused:
        actual = csa_sparse_attn(*inputs.values(), indices, scale, is_thd=True)
    else:
        actual = unfused_compressed_sparse_attn(
            inputs["q"], inputs["kv"].float(), inputs["sink"], indices, scale
        )
    expected = _ref_sparse_attention(*copies.values(), indices, scale)
    assert torch.count_nonzero(actual[0]) == 0
    pairs = {"output": (actual, expected)}
    if phase == "bwd":
        pairs = _gradient_pairs(actual, expected, inputs, copies, torch.randn_like(actual))
    _check(pairs, dtype, phase, record_property, fused=fused)


@pytest.mark.parametrize("fused", [False, True], ids=["native", "cutile"])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_mhc_projection_native_parity(groups, fused, phase, record_property):
    """Isolate mixed BF16-activation/FP32-weight mHC before Sinkhorn/mixing.

    FP32 mappings require cosine >= 0.999998 forward / 0.99998 backward.
    The BF16 activation gradient uses the separate BF16 backward tolerance.
    Relative L2 and the TF32/FP64 controls remain diagnostic, not pass criteria.
    """
    from megatron.core.fusions.fused_mhc_kernels import (
        _torch_proj_rms_compute_h,
        fused_proj_rms_compute_h,
        is_cutile_available,
    )

    if fused and not is_cutile_available():
        pytest.skip("cuTile mHC is unavailable; do not silently test a fallback")
    torch.manual_seed(4106)
    streams, hidden = 4, 4096
    width = streams * hidden
    mappings = streams * streams + 2 * streams
    inputs = {
        "x": torch.randn(128, width, device="cuda", dtype=torch.bfloat16).requires_grad_(),
        "weight": (torch.randn(mappings, width, device="cuda") / math.sqrt(width)).requires_grad_(),
        "alpha_pre": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "alpha_post": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "alpha_res": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "bias": (torch.randn(mappings, device="cuda") * 0.1).requires_grad_(),
    }
    copies = {name: _clone(value) for name, value in inputs.items()}
    function = fused_proj_rms_compute_h if fused else _torch_proj_rms_compute_h
    actual = function(*inputs.values(), streams, 1e-6, 1e-6, eps_inside_sqrt=True)
    expected = _ref_mhc_projection(*copies.values(), streams, 1e-6)
    # Diagnostic control only: emulate one round-to-nearest-even TF32 weight
    # conversion. This never replaces the full-FP32 reference used by the gate.
    control_inputs = {name: _clone(value) for name, value in inputs.items()}
    bits = control_inputs["weight"].detach().contiguous().view(torch.int32)
    rounded = ((bits + 0xFFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)
    control_inputs["weight"] = rounded.requires_grad_()
    control = _ref_mhc_projection(*control_inputs.values(), streams, 1e-6)
    controls = {}
    pairs = {
        name: (a, b)
        for name, a, b in zip(("pre", "post", "residual_logits", "rms"), actual, expected)
    }
    if phase == "bwd":
        # RMS is a saved normalization statistic, not a differentiable public
        # output of the custom kernel. Probe only the three actual mappings.
        probes = tuple(torch.randn_like(value) for value in actual[:3])
        pairs = _gradient_pairs(actual[:3], expected[:3], inputs, copies, probes)
        control_grads = torch.autograd.grad(control[:3], tuple(control_inputs.values()), probes)
        controls["vs_tf32_rounded_weight"] = {
            name: _metrics(pairs[name][0], value)
            for name, value in zip(control_inputs, control_grads)
        }
    else:
        controls["vs_tf32_rounded_weight"] = {
            name: _metrics(a, b) for name, a, b in zip(pairs, actual, control)
        }
        # A second, FP64 arithmetic oracle checks that the FP32 reference is
        # accurate; x retains precisely the same original BF16 values.
        xf, wf = inputs["x"].detach().double(), inputs["weight"].detach().double()
        rms = (xf.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        alpha = torch.cat(
            [
                inputs["alpha_pre"].detach().double().expand(streams),
                inputs["alpha_post"].detach().double().expand(streams),
                inputs["alpha_res"].detach().double().expand(streams * streams),
            ]
        )
        mapping = (xf @ wf.t()) / rms * alpha + inputs["bias"].detach().double()
        fp64 = (
            mapping[:, :streams].sigmoid() + 1e-6,
            mapping[:, streams : 2 * streams].sigmoid() * 2,
            mapping[:, 2 * streams :],
            rms,
        )
        controls["reference_vs_fp64"] = {
            name: _metrics(a, b) for name, a, b in zip(pairs, expected, fp64)
        }
    cosine_tols = {
        name: (
            _COSINE_TOLS[torch.bfloat16, "bwd"]
            if name == "x"
            else (2e-6 if phase == "fwd" else 2e-5)
        )
        for name in pairs
    }
    _check(
        pairs,
        torch.float32,
        phase,
        record_property,
        cosine_tols=cosine_tols,
        fused=fused,
        controls=controls,
    )


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_attention_rope_coefficient_dtype(groups, monkeypatch, ratio):
    """Every fused V4.1 rotation keeps FP32 tables with BF16 activations.

    Observing the real wrapper also covers compressed keys and inverse RoPE;
    indexer-only tests do not exercise those coefficient requests.
    """
    layer = _layer(groups, ratio, torch.bfloat16, True)
    rotary = layer.rotary_pos_emb
    original = rotary.get_cached_cos_sin
    requests = []

    def observe(*args, **kwargs):
        cos, sin = original(*args, **kwargs)
        requests.append((kwargs["dtype"], cos.dtype, sin.dtype))
        return cos, sin

    monkeypatch.setattr(rotary, "get_cached_cos_sin", observe)
    x = torch.randn(17, 1, layer.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        output = layer(x, attention_mask=None)[0]
    assert output.dtype == torch.bfloat16
    assert len(requests) >= (5 if ratio else 2)
    assert all(dtypes == (torch.float32,) * 3 for dtypes in requests), requests


@pytest.mark.parametrize("width", [512, 20480], ids=["small-k", "large-k"])
@pytest.mark.parametrize("phase", ["fwd", "bwd"])
def test_mhc_projection_fp32_operands(groups, width, phase, record_property):
    """Catch TF32 operand truncation in both cuTile backward dispatch paths.

    Compare FP32 mappings/parameter gradients at cosine >= 1 - 2e-12. The
    BF16 input gradient keeps its existing tolerance for final BF16 rounding.
    This focused contract is separate from the broad operator parity gates.
    """
    from megatron.core.fusions.fused_mhc_kernels import (
        fused_proj_rms_compute_h,
        is_cutile_available,
    )

    if not is_cutile_available():
        pytest.skip("cuTile mHC is unavailable; do not silently test a fallback")
    torch.manual_seed(4110)
    streams = 4
    mappings = streams * streams + 2 * streams
    inputs = {
        "x": torch.randn(7, width, device="cuda", dtype=torch.bfloat16).requires_grad_(),
        "weight": (torch.randn(mappings, width, device="cuda") / math.sqrt(width)).requires_grad_(),
        "alpha_pre": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "alpha_post": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "alpha_res": torch.full((1,), 0.1, device="cuda", requires_grad=True),
        "bias": (torch.randn(mappings, device="cuda") * 0.1).requires_grad_(),
    }
    copies = {name: _clone(value) for name, value in inputs.items()}
    actual = fused_proj_rms_compute_h(*inputs.values(), streams, 1e-6, eps_inside_sqrt=True)
    expected = _ref_mhc_projection(*copies.values(), streams, 1e-6)
    pairs = {
        name: (a, b)
        for name, a, b in zip(("pre", "post", "residual_logits", "rms"), actual, expected)
    }
    if phase == "bwd":
        probes = tuple(torch.randn_like(value) for value in actual[:3])
        pairs = _gradient_pairs(actual[:3], expected[:3], inputs, copies, probes)
    tolerances = {
        name: _COSINE_TOLS[torch.bfloat16, "bwd"] if name == "x" else 2e-12 for name in pairs
    }
    _check(pairs, torch.float32, phase, record_property, cosine_tols=tolerances, width=width)
