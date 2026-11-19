# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""mHC precision contracts, including GLM against an independent official-model reference.

GLM reference: huggingface/transformers commit
93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py,
Glm5NextTextHyperConnection and Glm5NextTextHyperHead. The GLM path opts into
HF's normalize-before-projection order (mhc_normalize_before_projection), an
intentional deviation from PR 7521's default order.
"""

import json
import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_mhc_glm_projection as glm
from megatron.core.fusions import fused_mhc_kernels as fmk
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    _get_precision_preserving_h_post_bda,
    native_h_aggregate,
    native_h_aggregate_into,
    native_h_post_bda,
    native_sinkhorn,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.glm53_native_reference import (
    MHC_GROUPS,
    MHC_RECONSTRUCTION_RTOL,
    mhc_contributions,
)
from tests.unit_tests.transformer.mhc_fused_op_spy import spy_fused_mhc_ops

# The fixed-launch Triton Sinkhorn needs both. The CPU test environment can have Triton without
# CUDA, where a Triton-only mark would let these cases error instead of skipping.
_NEEDS_CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and fmk.is_triton_available()), reason="needs CUDA and Triton"
)


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    """CI torchrun workers are independent replicas, each on its own local GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _assert_similarity(actual, expected, name, eps=1e-3):
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    actual, expected = actual.flatten().double(), expected.flatten().double()
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return
    cosine = F.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0)).item()
    similarity = (2 * (actual * expected).sum() / denominator).item()
    assert cosine > 1 - eps, f"{name}: cosine={cosine}"
    assert similarity > 1 - eps, f"{name}: tensor_similarity={similarity}"


def _similarity_values(actual, expected):
    """_assert_similarity's (cosine, tensor similarity) without its asserts; (1.0, 1.0) for two
    zero tensors, which _assert_similarity accepts."""
    actual, expected = actual.flatten().double(), expected.flatten().double()
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return 1.0, 1.0
    cosine = F.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0)).item()
    similarity = (2 * (actual * expected).sum() / denominator).item()
    return cosine, similarity


def _official_mhc(hidden, parameters, num_streams, iterations, eps, capture=None):
    """Torch-only translation preserving the pinned HF operation ordering.

    With ``capture``, the mapping logits are stored there as "logits", their gradient retained.
    """
    flat = hidden.float()
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    logits = F.linear(normalized, parameters["mapping_proj.weight"])
    if capture is not None:
        logits.retain_grad()
        capture["logits"] = logits
    pre_logits, post_logits, residual_logits = logits.split(
        (num_streams, num_streams, num_streams**2), dim=-1
    )
    pre_bias, post_bias, residual_bias = parameters["bias"].split(
        (num_streams, num_streams, num_streams**2)
    )
    pre = (pre_logits * parameters["alpha_pre"] + pre_bias).sigmoid() + 1e-6
    post = 2 * (post_logits * parameters["alpha_post"] + post_bias).sigmoid()
    mixing = residual_logits * parameters["alpha_res"] + residual_bias
    mixing = mixing.unflatten(-1, (num_streams, num_streams)).softmax(-1) + 1e-6
    mixing = mixing / (mixing.sum(-2, keepdim=True) + 1e-6)
    for _ in range(iterations - 1):
        mixing = mixing / (mixing.sum(-1, keepdim=True) + 1e-6)
        mixing = mixing / (mixing.sum(-2, keepdim=True) + 1e-6)
    streams = hidden.unflatten(-1, (num_streams, -1))
    aggregated = (streams * pre.unsqueeze(-1)).sum(-2).to(hidden.dtype)
    return aggregated, post, mixing


def _skip_unless_glm_pair_device():
    """Skip unless the current device's capability is one the GLM projection pair accepts; read
    after the rank's own device is bound, never at collection."""
    capability = torch.cuda.get_device_capability()
    if capability not in glm.GLM_PAIR_DEVICE_CAPABILITIES:
        pytest.skip(
            f"the GLM projection pair is validated on {glm.GLM_PAIR_DEVICE_CAPABILITIES} only"
        )


def _alpha_link_readings(actual, reference):
    """Read each alpha link (pre, post, res) as its scalar gradient and through its contributions.

    ``actual`` and ``reference`` are each (projection, its gradient, parameters by name): the
    projection is what the alphas scale (the module's ``_compute_h`` input, the official logits),
    and the parameters' alpha_pre, alpha_post and alpha_res hold their gradients. A contribution
    is dL/dproj / alpha * proj for one token and column of the group, in float64, so each alpha
    gradient is the sum of its group's contributions, a sum that can cancel heavily.
    """
    contributions, gradients = {}, {}
    for side, (projection, gradient, parameters) in (("actual", actual), ("reference", reference)):
        alphas = [parameters[f"alpha_{group}"] for group, _ in MHC_GROUPS]
        contributions[side] = mhc_contributions(
            gradient, projection, [float(alpha.detach()) for alpha in alphas]
        )
        gradients[side] = [alpha.grad.detach().double().reshape(()) for alpha in alphas]
    readings = {}
    for index, (group, _) in enumerate(MHC_GROUPS):
        reading = {}
        for side in ("actual", "reference"):
            parts, scalar = contributions[side][group], gradients[side][index]
            reading[f"{side}_contributions"] = parts
            reading[f"{side}_alpha_grad"] = scalar.item()
            reading[f"{side}_reconstruction_residual"] = (scalar - parts.sum()).abs().item()
        reading["scalar_cosine"], reading["scalar_similarity"] = _similarity_values(
            gradients["actual"][index], gradients["reference"][index]
        )
        reading["contribution_cosine"], reading["contribution_similarity"] = _similarity_values(
            reading["actual_contributions"], reading["reference_contributions"]
        )
        reference_sum = reading["reference_contributions"].sum().item()
        reference_abs_sum = reading["reference_contributions"].abs().sum().item()
        reading["kappa"] = reference_abs_sum / abs(reference_sum) if reference_sum else float("inf")
        readings[group] = reading
    return readings


def _assert_alpha_links(readings, eps=1e-3):
    """Apply the alpha links' rule to _alpha_link_readings' readings; mark how each was read.

    Each side's alpha gradient must be the sum of its contributions, within
    MHC_RECONSTRUCTION_RTOL of their absolute sum. A link passes _assert_similarity's bars on its
    scalar; only a scalar that fails them is read by its decomposition, whose contribution
    tensors must then pass the same bars. No bar is loosened.
    """
    for group, reading in readings.items():
        link = f"alpha_{group}"
        for side in ("actual", "reference"):
            residual = reading[f"{side}_reconstruction_residual"]
            abs_sum = reading[f"{side}_contributions"].abs().sum().item()
            assert residual <= MHC_RECONSTRUCTION_RTOL * abs_sum, (
                f"{link} {side}: the gradient {reading[f'{side}_alpha_grad']} is not the sum of "
                f"its contributions (residual {residual}, absolute sum {abs_sum})"
            )
        cosine, similarity = reading["scalar_cosine"], reading["scalar_similarity"]
        reading["read_by_decomposition"] = not (cosine > 1 - eps and similarity > 1 - eps)
        if reading["read_by_decomposition"]:
            parts_cosine = reading["contribution_cosine"]
            parts_similarity = reading["contribution_similarity"]
            assert parts_cosine > 1 - eps and parts_similarity > 1 - eps, (
                f"{link}: the gradient fails the bars (cosine={cosine}, "
                f"tensor_similarity={similarity}) and so do its contributions "
                f"(cosine={parts_cosine}, tensor_similarity={parts_similarity})"
            )


@pytest.mark.parametrize("fp32_mixing", [False, True])
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
def test_mhc_precision(fp32_mixing, input_scale):
    torch.manual_seed(42)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        layernorm_epsilon=1e-5,
        mhc_norm_eps_inside_sqrt=fp32_mixing,
        mhc_keep_mappings_in_fp32=fp32_mixing,
    )
    layer = HyperConnectionModule(config, layer_number=1).cuda()
    expected_eps = config.layernorm_epsilon if fp32_mixing else 1e-6
    assert layer.norm_eps == expected_eps
    with torch.no_grad():
        layer.mapping_proj.weight.normal_(std=0.2)
        layer.bias.normal_()
    residual = (torch.randn(16, 2, 128, device="cuda") * input_scale).to(torch.bfloat16)
    residual.requires_grad_()
    output = torch.randn(16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pre, post, comb = layer.compute_mappings(residual)
    expected_dtype = torch.float32 if fp32_mixing else torch.bfloat16
    assert pre.dtype == post.dtype == comb.dtype == expected_dtype

    x = residual.float()
    rms = (
        torch.rsqrt(x.square().mean(-1, keepdim=True) + expected_eps)
        if fp32_mixing
        else (x.norm(dim=-1, keepdim=True) / 128**0.5 + expected_eps).reciprocal()
    )
    scales = torch.cat(
        [layer.alpha_pre.expand(4), layer.alpha_post.expand(4), layer.alpha_res.expand(16)]
    )
    logits = (x @ layer.mapping_proj.weight.T) * rms * scales + layer.bias
    expected_pre = (logits[..., :4].sigmoid() + 1e-6).to(expected_dtype)
    expected_post = (2 * logits[..., 4:8].sigmoid()).to(expected_dtype)
    expected_comb = logits[..., 8:].reshape(16, 2, 4, 4).softmax(-1) + 1e-6
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(config.mhc_sinkhorn_iterations - 1):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + 1e-6)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    torch.testing.assert_close(pre, expected_pre)
    torch.testing.assert_close(post, expected_post)
    torch.testing.assert_close(comb, expected_comb.to(expected_dtype))

    aggregated = layer.aggregate(residual, pre)
    streams = residual.view(16, 2, 4, 32)
    expected_aggregate = (streams.float() * pre.float().unsqueeze(-1)).sum(2).to(residual.dtype)
    torch.testing.assert_close(aggregated, expected_aggregate, rtol=0.02, atol=0.01)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp32_mixing):
        actual = layer.fused_h_res_h_post_bda(
            comb, residual, post, (output, None), 0.0, True, False
        )
    expected_mix = torch.einsum("...ij,...ih->...jh", comb, streams.to(expected_dtype))
    expected_output = (
        expected_mix.float() + post.float().unsqueeze(-1) * output.float().unsqueeze(2)
    ).to(residual.dtype)
    torch.testing.assert_close(actual.view_as(streams), expected_output, rtol=0.02, atol=0.01)
    (actual.float().square().mean() + aggregated.float().square().mean()).backward()
    for tensor in (residual, output, layer.mapping_proj.weight, layer.bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("preserve_rounding", [False, True], ids=["nopreserve", "preserve"])
@pytest.mark.parametrize("input_scale", [1.0, 1e-3])
@pytest.mark.parametrize("fp32_mappings", [False, True])
@pytest.mark.parametrize(
    "fused_ops",
    [
        None,
        pytest.param(["sinkhorn"], marks=_NEEDS_CUDA_TRITON),
        pytest.param(["proj_rms"], marks=_NEEDS_CUDA_TRITON),
        pytest.param(["proj_rms", "sinkhorn"], marks=_NEEDS_CUDA_TRITON),
    ],
    ids=["unfused", "sinkhorn", "proj_rms", "proj_rms_sinkhorn"],
)
def test_glm_mhc_official_forward_and_all_gradients(
    monkeypatch, input_scale, fp32_mappings, fused_ops, preserve_rounding
):
    """Keep the production 4096-wide hidden state and four residual streams.

    With ``fused_ops``, per-op mode binds those operations under the explicit ``triton`` policy,
    so construction fails rather than testing the unfused path when Triton is missing. The
    shape-[1] alpha links are read by _assert_alpha_links.
    """
    ops = fused_ops or []
    if "proj_rms" in ops:
        _skip_unless_glm_pair_device()
    torch.manual_seed(42)
    fused = {}
    if fused_ops is not None:
        fused = dict(use_fused_mhc=True, mhc_fused_ops=fused_ops, mhc_fused_backend="triton")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        num_residual_streams=4,
        enable_hyper_connections=True,
        layernorm_epsilon=1e-5,
        mhc_norm_eps_inside_sqrt=True,
        mhc_normalize_before_projection=True,
        mhc_keep_mappings_in_fp32=fp32_mappings,
        mhc_keep_pre_in_fp32=True,
        mhc_learned_output_contract=False,
        mhc_preserve_activation_dtype_rounding=preserve_rounding,
        **fused,
    )
    module = HyperConnectionModule(config, layer_number=1).cuda()
    # Per-op mode rebinds only the listed operations; the post follows the rounding flag.
    assert module._h_post_bda_op is (
        _get_precision_preserving_h_post_bda() if preserve_rounding else native_h_post_bda
    )
    assert module._glm_proj_rms_op is (glm.glm_proj_rms_op("rn") if "proj_rms" in ops else None)
    assert module._sinkhorn_op is (
        glm.fixed_launch_sinkhorn if "sinkhorn" in ops else native_sinkhorn
    )
    assert module._fused_op_labels == {
        "proj_rms": "glm-triton(tf32=rn)" if "proj_rms" in ops else "native",
        "sinkhorn": "triton(num_warps=1)" if "sinkhorn" in ops else "native",
        "h_aggregate": "native",
        "h_post_bda": "native(emulate-casts)" if preserve_rounding else "native(compiled)",
    }
    with torch.no_grad():
        module.mapping_proj.weight.normal_(std=0.05)
        module.bias.normal_(std=0.1)
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }
    assert set(reference_parameters) == {
        "mapping_proj.weight",
        "bias",
        "alpha_pre",
        "alpha_post",
        "alpha_res",
    }
    hidden = (torch.randn(8, 2, 4 * 4096, device="cuda") * input_scale).bfloat16()
    actual_input = hidden.clone().requires_grad_()
    reference_input = hidden.clone().requires_grad_()
    branch = torch.randn(8, 2, 4096, device="cuda", dtype=torch.bfloat16)
    actual_branch = branch.clone().requires_grad_()
    reference_branch = branch.clone().requires_grad_()

    # The spy counts forward calls only, so the Triton Sinkhorn's backward is counted here.
    sinkhorn_backward_calls = []
    sinkhorn_bwd = glm._sinkhorn_bwd

    def counted_sinkhorn_bwd(*args, **kwargs):
        sinkhorn_backward_calls.append(None)
        return sinkhorn_bwd(*args, **kwargs)

    monkeypatch.setattr(glm, "_sinkhorn_bwd", counted_sinkhorn_bwd)
    # The GLM projection pair's launches, through the module globals its Function calls.
    pair_launches = {"fwd": 0, "bwd": 0}

    def counted_pair(direction, launch):
        def counted(*args, **kwargs):
            pair_launches[direction] += 1
            return launch(*args, **kwargs)

        return counted

    monkeypatch.setattr(glm, "_glm_proj_fwd", counted_pair("fwd", glm._glm_proj_fwd))
    monkeypatch.setattr(glm, "_glm_proj_bwd", counted_pair("bwd", glm._glm_proj_bwd))
    # The projection the alphas scale (the _compute_h input) and its gradient, for the
    # alpha links' contributions.
    captured = {}
    projection_and_get_norm = HyperConnectionModule._projection_and_get_norm

    def capturing_projection_and_get_norm(self, x):
        proj, r = projection_and_get_norm(self, x)
        assert not captured, "one projection per forward"
        captured.update(proj=proj.detach().clone(), grad=None)

        def keep(gradient):
            assert captured["grad"] is None
            captured["grad"] = gradient.detach().clone()

        proj.register_hook(keep)
        return proj, r

    monkeypatch.setattr(
        HyperConnectionModule, "_projection_and_get_norm", capturing_projection_and_get_norm
    )
    with spy_fused_mhc_ops(module) as spy:
        pre, post, mixing = module.compute_mappings(actual_input)
        mapping_dtype = torch.float32 if fp32_mappings else hidden.dtype
        assert pre.dtype == torch.float32
        assert post.dtype == mixing.dtype == mapping_dtype
        aggregated = module.aggregate(actual_input, pre)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = module.fused_h_res_h_post_bda(
                mixing, actual_input, post, (actual_branch, None), 0.0, True, False
            )
    reference_capture = {}
    expected_aggregate, expected_post, expected_mixing = _official_mhc(
        reference_input,
        reference_parameters,
        4,
        config.mhc_sinkhorn_iterations,
        1e-5,
        capture=reference_capture,
    )
    # The official decoder casts post and comb, but not pre, before mixing.
    # The all-FP32 profile is separately supported for actor/serving recipes.
    expected_post = expected_post.to(mapping_dtype)
    expected_mixing = expected_mixing.to(mapping_dtype)
    reference_streams = reference_input.unflatten(-1, (4, 4096)).to(mapping_dtype)
    expected = (
        (
            torch.einsum("...ij,...ih->...jh", expected_mixing, reference_streams)
            + expected_post.unsqueeze(-1) * reference_branch.unsqueeze(-2)
        )
        .flatten(-2)
        .to(hidden.dtype)
    )
    assert aggregated.dtype == actual.dtype == hidden.dtype
    similarities = {}
    for name, observed, reference in (
        ("aggregate", aggregated, expected_aggregate),
        ("post", post, expected_post),
        ("mixing", mixing, expected_mixing),
        ("output", actual, expected),
    ):
        _assert_similarity(observed, reference, name)
        similarities[name] = _similarity_values(observed, reference)

    aggregate_grad = torch.randn_like(aggregated)
    output_grad = torch.randn_like(actual)
    ((aggregated * aggregate_grad).float().sum() + (actual * output_grad).float().sum()).backward()
    (
        (expected_aggregate * aggregate_grad).float().sum() + (expected * output_grad).float().sum()
    ).backward()
    gradients = [
        ("input gradient", actual_input.grad, reference_input.grad),
        ("branch gradient", actual_branch.grad, reference_branch.grad),
    ]
    # The shape-[1] alpha links go to _assert_alpha_links below.
    gradients += [
        (name, parameter.grad, reference_parameters[name].grad)
        for name, parameter in module.named_parameters()
        if not name.startswith("alpha_")
    ]
    for name, observed, reference in gradients:
        _assert_similarity(observed, reference, name)
        similarities[name] = _similarity_values(observed, reference)
    logits = reference_capture["logits"]
    assert captured["grad"] is not None and logits.grad is not None
    readings = _alpha_link_readings(
        (captured["proj"], captured["grad"], dict(module.named_parameters())),
        (logits, logits.grad, reference_parameters),
    )
    _assert_alpha_links(readings)
    if "sinkhorn" in ops:
        # The Triton Sinkhorn ran both ways: one forward call and one backward launch.
        spy.assert_ran("_sinkhorn_op", glm.fixed_launch_sinkhorn)
        assert spy.calls("_sinkhorn_op") == 1
        assert len(sinkhorn_backward_calls) == 1
    else:
        assert spy.targets("_sinkhorn_op") == {native_sinkhorn} and spy.calls("_sinkhorn_op") == 1
        assert sinkhorn_backward_calls == []
    if "proj_rms" in ops:
        spy.assert_ran("_glm_proj_rms_op", glm.glm_proj_rms_op("rn"))
        assert spy.calls("_glm_proj_rms_op") == 1  # one compute_mappings
        spy.assert_no_unsupported()  # redundant under triton, which raises on a refusal
        assert pair_launches == {"fwd": 1, "bwd": 1}
    else:
        assert spy.calls("_glm_proj_rms_op") == 0 and pair_launches == {"fwd": 0, "bwd": 0}
    print(
        "MHC_PREC_HF "
        + json.dumps(
            {
                "rank": int(os.environ.get("RANK", "0")),
                "fused_ops": fused_ops,
                "fp32_mappings": fp32_mappings,
                "input_scale": input_scale,
                "preserve_activation_dtype_rounding": preserve_rounding,
                "matmul_mode": glm._matmul_mode(),
                "labels": module._fused_op_labels,
                "spy_calls": {record.attribute: record.calls for record in spy.records},
                "refusals": spy.unsupported("_glm_proj_rms_op"),
                "launches": dict(pair_launches, sinkhorn_bwd=len(sinkhorn_backward_calls)),
                "similarities": similarities,
                "alpha_links": {
                    group: {
                        key: value for key, value in reading.items() if not torch.is_tensor(value)
                    }
                    for group, reading in readings.items()
                },
                "read_by_decomposition": any(
                    reading["read_by_decomposition"] for reading in readings.values()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


@pytest.mark.parametrize("fp32_mappings", [False, True])
def test_mhc_caller_owned_aggregation_matches_regular_path(fp32_mappings):
    """The graph-input buffer must preserve FP32 mixing before the BF16 cast."""
    torch.manual_seed(43)
    x = torch.randn(8, 2, 4, 4096, device="cuda", dtype=torch.bfloat16)
    mapping_dtype = torch.float32 if fp32_mappings else torch.bfloat16
    weights = torch.randn(8, 2, 4, device="cuda", dtype=mapping_dtype)
    actual_input, reference_input = (x.clone().requires_grad_() for _ in range(2))
    actual_weights, reference_weights = (weights.clone().requires_grad_() for _ in range(2))
    out = torch.empty(8, 2, 4096, device="cuda", dtype=x.dtype)
    actual = native_h_aggregate_into(actual_input, actual_weights, out)
    expected = native_h_aggregate(reference_input, reference_weights)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    # The explicit-buffer custom op accumulates weight gradients in FP32.
    torch.testing.assert_close(
        actual_weights.grad, (grad.float().unsqueeze(2) * x.float()).sum(-1).to(mapping_dtype)
    )


@pytest.mark.parametrize("pre_dtype", [torch.bfloat16, torch.float32])
def test_mhc_caller_owned_aggregation_eager_body_rounds_once(monkeypatch, pre_dtype):
    """The uncompiled caller-owned forward sums the promoted products and rounds once.

    Dynamo runs this body eagerly once its cache is full. With FP32 coefficients and a BF16
    ``out``, ``torch.sum(..., out=out)`` would round each product before the streams are
    summed; the regular path sums in FP32 and casts once.
    """
    import megatron.core.transformer.hyper_connection as hyper_connection_module

    monkeypatch.setattr(
        hyper_connection_module,
        "_native_h_aggregate_into_forward",
        hyper_connection_module._native_h_aggregate_into_forward.__wrapped__,
    )
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(16, 2, 4, 256, generator=generator).to(torch.bfloat16)
    h_pre = torch.rand(16, 2, 4, generator=generator).to(pre_dtype)
    out = torch.empty(16, 2, 256, dtype=torch.bfloat16)
    actual = native_h_aggregate_into(x, h_pre, out)
    expected = (x * h_pre.unsqueeze(-1)).sum(dim=2).to(torch.bfloat16)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("flag", ["mhc_norm_eps_inside_sqrt", "mhc_keep_mappings_in_fp32"])
def test_mhc_fused_rejects_unsupported_precision(flag):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        enable_mhc_connections=True,
        use_fused_mhc=True,
    )
    # Keep the module-level guard for configs modified after constructor validation.
    setattr(config, flag, True)
    with pytest.raises(ValueError, match="Fused mHC does not support"):
        HyperConnectionModule(config, layer_number=1)


def test_glm_alpha_link_rule_reads_cancellation_through_contributions():
    """The alpha links' rule on synthetic float64 readings, on CPU: a cancelled scalar is read
    through its contributions, which must then pass the bars, and every scalar must be the sum of
    its contributions."""
    alphas = (0.01, 0.02, 0.03)
    gradient = torch.cat(
        [
            torch.full((columns.stop - columns.start,), alpha, dtype=torch.float64)
            for alpha, (_, columns) in zip(alphas, MHC_GROUPS)
        ]
    )

    def side(contributions, shifts=(0.0, 0.0, 0.0)):
        """(proj, dL/dproj, parameters) whose contributions are ``contributions``, with alpha
        gradients equal to their sums plus ``shifts``."""
        parameters = {}
        for alpha, shift, (group, columns) in zip(alphas, shifts, MHC_GROUPS):
            parameter = torch.tensor([alpha], dtype=torch.float64, requires_grad=True)
            parameter.grad = contributions[..., columns].sum().reshape(1) + shift
            parameters[f"alpha_{group}"] = parameter
        return contributions, gradient.expand_as(contributions), parameters

    half = torch.rand(4, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) + 0.5
    reference = torch.cat([half, -half])
    # Every column's contributions cancel to 1e-7 of their absolute sum.
    reference[0] += 1e-7 * reference.abs().sum(0)

    # (a) Contributions equal to 1e-7 relative, whose cancelled sums differ by about 2x.
    readings = _alpha_link_readings(
        side(reference * (1 + 1e-7 * reference.sign())), side(reference)
    )
    _assert_alpha_links(readings)
    for group, reading in readings.items():
        assert reading["scalar_similarity"] < 1 - 1e-3, group
        assert reading["read_by_decomposition"], group

    # (b) One token's contributions negated.
    negated = reference.clone()
    negated[5] = -negated[5]
    with pytest.raises(AssertionError, match="contributions") as error:
        _assert_alpha_links(_alpha_link_readings(side(negated), side(reference)))
    assert "is not the sum" not in str(error.value)

    # (c) A scalar that is not the sum of its contributions.
    with pytest.raises(AssertionError, match="is not the sum"):
        _assert_alpha_links(
            _alpha_link_readings(side(reference, shifts=(0.0, 1e-3, 0.0)), side(reference))
        )
