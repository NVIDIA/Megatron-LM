# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Routed-expert probability placement, not complete GLM53/MTP acceptance.

Independent arithmetic source: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py:132-136, SHA256
2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b.
HF weights the BF16 expert output after FC2, rather than its activation input.
BF16/FP32 numerical cases keep H4096/I2048 and two real local TE experts with short
token segments. They do not certify all288 routing, EP communication, or MTP parity.
FP8 rejects the post-FC2 opt-in, so the blockwise-FP8 cases cover the standard
placement (probabilities between the clamped SwiGLU and FC2) at the GLM53 dev-proxy
expert width H1024/I512.
"""

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fp8_utils import HAVE_TE, get_fp8_context, get_fp8_recipe
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

HIDDEN, INTERMEDIATE = 4096, 2048


def _config(*, after=True, fusion=False, dtype=torch.float32, **overrides):
    values = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=12288,
        num_attention_heads=64,
        num_moe_experts=8,
        moe_ffn_hidden_size=INTERMEDIATE,
        moe_grouped_gemm=True,
        moe_router_dtype='fp32',
        moe_token_dispatcher_type='alltoall',
        expert_tensor_parallel_size=1,
        transformer_impl='transformer_engine',
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
        gated_linear_unit=True,
        add_bias_linear=False,
        bias_activation_fusion=fusion,
        use_te_activation_func=False,
        gradient_accumulation_fusion=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        use_cpu_initialization=False,
        moe_apply_probs_after_fc2=after,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_probability_placement_default_is_unchanged():
    assert TransformerConfig.__dataclass_fields__['moe_apply_probs_after_fc2'].default is False
    config = _config(after=False)
    assert config.moe_apply_probs_after_fc2 is False
    # Disabling the opt-in does not impose its new backend/precision scope.
    assert _config(after=False, moe_router_dtype=None).moe_router_dtype is None


@pytest.mark.parametrize('expert_parallel_size', [1, 4])
def test_output_weighting_accepts_declared_expert_parallel_configs(expert_parallel_size):
    config = _config(expert_model_parallel_size=expert_parallel_size)
    assert config.moe_apply_probs_after_fc2 is True
    assert config.expert_model_parallel_size == expert_parallel_size


@pytest.mark.parametrize(
    'overrides',
    [
        {'moe_apply_probs_after_fc2': 1},
        {'transformer_impl': 'local'},
        {'transformer_impl': 'inference_optimized'},
        {'moe_grouped_gemm': False},
        {'num_moe_experts': None},
        {'params_dtype': torch.float16},
        {'fp16': True},
        {'fp8': 'e4m3'},
        {'fp4': 'e2m1'},
        {'fp8_param': True},
        {'fp4_param': True},
        {'activation_func_fp8_input_store': True},
        {'quant_recipe': {}},
        {'tensor_model_parallel_size': 2},
        {'expert_tensor_parallel_size': 2},
        {'sequence_parallel': True, 'tensor_model_parallel_size': 2},
        {'moe_token_dispatcher_type': 'allgather'},
        {'moe_router_dtype': None},
        {'add_bias_linear': True},
        {'gated_linear_unit': False},
        {'activation_func': F.gelu},
        {'use_te_activation_func': True},
        {'moe_mlp_glu_interleave_size': 16},
        {'glu_linear_offset': 1.0},
        {'moe_latent_size': 512},
        {'moe_apply_probs_on_input': True},
        {'use_transformer_engine_op_fuser': True},
        {'moe_use_grouped_tensor': True},
        {'moe_single_grouped_weight': True},
        {'moe_single_grouped_bias': True},
        {'moe_megakernel_backend': 'mok'},
    ],
)
def test_output_weighting_rejects_unsupported_configs(overrides):
    with pytest.raises(ValueError, match='moe_apply_probs_after_fc2'):
        _config(**overrides)


@pytest.mark.parametrize(
    'overrides',
    [
        {'activation_func_clamp_value': None, 'activation_func_tanh_clamp_scale': 8.0},
        {
            'activation_func_clamp_value': None,
            'activation_func_tanh_clamp_scale': 8.0,
            'activation_func_tanh_clamp_scale_linear': 4.0,
        },
    ],
)
def test_output_weighting_rejects_situ_glu_tanh_clamp(monkeypatch, overrides):
    """The fused SwiGLU path does not forward the SiTU-GLU tanh clamp scales."""
    # Reach the activation check on hosts without TE: the TE version gate comes first.
    monkeypatch.setattr(
        'megatron.core.transformer.transformer_config.is_te_min_version', lambda *_: True
    )
    with pytest.raises(ValueError, match='requires bias-free standard SwiGLU'):
        _config(**overrides)


def test_output_weighting_rejects_unavailable_grouped_backend(monkeypatch):
    monkeypatch.setattr(
        'megatron.core.transformer.transformer_config.is_te_min_version', lambda *_: False
    )
    with pytest.raises(ValueError, match='requires TEGroupedMLP'):
        _config()


def test_output_weighting_cannot_fall_back_to_sequential_or_inference_subclasses():
    config = _config()
    with pytest.raises(ValueError, match='not SequentialMLP'):
        SequentialMLP(2, config, None)

    class OtherGroupedMLP(TEGroupedMLP):
        pass

    with pytest.raises(ValueError, match='ordinary TEGroupedMLP only'):
        OtherGroupedMLP(2, config, None)


@pytest.mark.parametrize('larger_group', ['tp', 'expt_tp'])
def test_output_weighting_rejects_actual_parallel_groups(monkeypatch, larger_group):
    config = _config()
    groups = SimpleNamespace(tp=object(), expt_tp=object())
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        'get_world_size',
        lambda group: 2 if group is getattr(groups, larger_group) else 1,
    )
    with pytest.raises(ValueError, match='explicit actual TP/ETP1 groups'):
        TEGroupedMLP(2, config, None, pg_collection=groups)


@pytest.mark.parametrize('unsupported', ['fuser', 'grouped_tensor', 'output_buffer', 'grad_buffer'])
def test_output_weighting_rejects_shortcuts_before_execution(unsupported):
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_apply_probs_after_fc2=True)
    module._with_fused_impl = unsupported == 'fuser'
    module._use_grouped_tensor = unsupported == 'grouped_tensor'
    with pytest.raises(ValueError, match='fused experts or output buffers'):
        module(
            torch.empty(2, HIDDEN),
            torch.tensor([2]),
            torch.ones(2),
            output_buffer=object() if unsupported == 'output_buffer' else None,
            grad_input_buffer=object() if unsupported == 'grad_buffer' else None,
        )


@pytest.mark.parametrize(
    'input_dtype,prob_dtype', [(torch.float16, torch.float32), (torch.float32, torch.bfloat16)]
)
def test_output_weighting_rejects_runtime_dtype_mismatch(input_dtype, prob_dtype):
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_apply_probs_after_fc2=True)
    module._with_fused_impl = module._use_grouped_tensor = False
    with pytest.raises(ValueError, match='FP32 probabilities'):
        module(
            torch.empty(2, HIDDEN, dtype=input_dtype),
            torch.tensor([2]),
            torch.ones(2, dtype=prob_dtype),
        )


@pytest.mark.parametrize('prob_shape,prob_device', [([2, 1], 'cpu'), ([3], 'cpu'), ([2], 'meta')])
def test_output_weighting_rejects_probability_broadcast_or_device_mismatch(prob_shape, prob_device):
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_apply_probs_after_fc2=True)
    module._with_fused_impl = module._use_grouped_tensor = False
    module.input_size = HIDDEN
    with pytest.raises(ValueError, match='co-located'):
        module(
            torch.empty(2, HIDDEN), torch.tensor([2]), torch.ones(prob_shape, device=prob_device)
        )


def test_BF16_probability_placement_is_not_an_algebraic_noop():
    # Exact two-term counterexample, not a production-shape parity substitute.
    activation = torch.ones(1, 2, dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 1.0078125]], dtype=torch.bfloat16)
    probability = torch.tensor([[0.3]], dtype=torch.float32)
    before = F.linear((activation * probability).to(activation.dtype), weight)
    after = (F.linear(activation, weight) * probability).to(activation.dtype)
    assert before.item() == 0.60546875
    assert after.item() == 0.6015625


@pytest.fixture
def _model_parallel():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    previous = torch.get_float32_matmul_precision()
    try:
        # FP32 regression arithmetic is IEEE; this does not change production
        # global precision or the unchanged BF16 composite-test contract.
        torch.set_float32_matmul_precision('highest')
        assert torch.backends.cuda.matmul.allow_tf32 is False
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(531533)
        torch.manual_seed(531533)
        yield
    finally:
        try:
            Utils.destroy_model_parallel()
        finally:
            torch.set_float32_matmul_precision(previous)


def _assert_parity(actual, expected, name):
    assert actual is not None and expected is not None, name
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    a, b = actual.detach().flatten().double(), expected.detach().flatten().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
    aa, bb, ab = torch.dot(a, a), torch.dot(b, b), torch.dot(a, b)
    assert aa > 0 and bb > 0, name
    cosine = (ab / (aa * bb).sqrt()).item()
    similarity = (2 * ab / (aa + bb)).item()
    assert cosine > 0.999, f'{name}: cosine={cosine}'
    assert similarity > 0.999, f'{name}: tensor_similarity={similarity}'


def _clamp_masks(projected):
    gate, up = projected.chunk(2, -1)
    return torch.stack((gate > 10, up > 10, up < -10, (gate <= 10) & (up >= -10) & (up <= 10)))


@pytest.mark.internal
@pytest.mark.parametrize('after', [False, True], ids=['legacy', 'after_fc2'])
@pytest.mark.parametrize('fusion', [False, True], ids=['eager_activation', 'fused_activation'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32], ids=['bf16', 'fp32'])
def test_real_TE_grouped_output_input_weights_and_probability_VJP(
    _model_parallel, record_property, after, fusion, dtype
):
    config = _config(
        after=after,
        fusion=fusion,
        dtype=dtype,
        num_moe_experts=2,
        moe_shared_expert_intermediate_size=INTERMEDIATE,
    )
    # Build through the public MoELayer spec as well: shared-expert config
    # copying changes its FFN width without disabling the routed contract.
    layer = get_moe_module_spec(use_te=True, num_experts=2, moe_grouped_gemm=True)(
        config=config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
    ).cuda()
    real = layer.experts
    assert config.num_moe_experts == 2 and config.ffn_hidden_size == 12288
    assert layer.shared_experts.config.ffn_hidden_size == INTERMEDIATE
    assert type(real) is TEGroupedMLP
    assert real._with_fused_impl is False and real._use_grouped_tensor is False
    parameters = dict(real.named_parameters())
    assert set(parameters) == {
        'linear_fc1.weight0',
        'linear_fc1.weight1',
        'linear_fc2.weight0',
        'linear_fc2.weight1',
    }
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.normal_(std=0.02)
        for expert in range(2):
            weight = parameters[f'linear_fc1.weight{expert}']
            weight[:3].normal_(std=0.5)
            weight[INTERMEDIATE : INTERMEDIATE + 3].normal_(std=0.5)
    native_parameters = {
        name: parameter.detach().clone().requires_grad_() for name, parameter in parameters.items()
    }
    counts = torch.tensor([8, 9], dtype=torch.int64, device='cpu')
    hidden = torch.randn(17, HIDDEN, device='cuda', dtype=dtype)
    actual_input, native_input = (hidden.clone().requires_grad_() for _ in range(2))
    probabilities = torch.linspace(0.21, 0.53, 17, device='cuda', dtype=torch.float32)
    actual_prob, native_prob = (probabilities.clone().requires_grad_() for _ in range(2))
    actual_masks = []

    def observe_fc1(_module, _inputs, output):
        actual_masks.append(_clamp_masks(output[0].detach()))

    hook = real.linear_fc1.register_forward_hook(observe_fc1)
    try:
        actual, bias = real(actual_input, counts, actual_prob)
    finally:
        hook.remove()
    assert bias is None and actual.dtype == dtype
    outputs, native_masks = [], []
    for expert, (x, probability) in enumerate(
        zip(native_input.split([8, 9]), native_prob.split([8, 9]))
    ):
        projected = F.linear(x, native_parameters[f'linear_fc1.weight{expert}'])
        native_masks.append(_clamp_masks(projected.detach()))
        gate, up = projected.chunk(2, -1)
        activation = F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)
        if not after:
            activation = (activation * probability[:, None]).to(dtype)
        output = F.linear(activation, native_parameters[f'linear_fc2.weight{expert}'])
        if after:
            output = (output * probability[:, None]).to(dtype)
        outputs.append(output)
    assert len(actual_masks) == 1
    actual_masks, native_masks = actual_masks[0], torch.cat(native_masks, dim=1)
    actual_counts = actual_masks.sum((1, 2)).cpu().tolist()
    native_counts = native_masks.sum((1, 2)).cpu().tolist()
    record_property(
        'moe_probability_clamp_coverage',
        json.dumps(
            {
                'order': ['gate>10', 'up>10', 'up<-10', 'unclamped'],
                'actual_counts': actual_counts,
                'native_counts': native_counts,
                'active_set_mismatches': (actual_masks != native_masks).sum((1, 2)).cpu().tolist(),
            },
            sort_keys=True,
        ),
    )
    assert all(value > 0 for value in actual_counts), actual_counts
    assert all(value > 0 for value in native_counts), native_counts
    expected = torch.cat(outputs, dim=0)
    _assert_parity(actual, expected, 'grouped output')
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    _assert_parity(actual_input.grad, native_input.grad, 'grouped input VJP')
    _assert_parity(actual_prob.grad, native_prob.grad, 'FP32 routing probability VJP')
    for name, parameter in parameters.items():
        _assert_parity(parameter.grad, native_parameters[name].grad, name)


# Blockwise-FP8 standard placement (the GLM53 FP8 target recipe).
GLM53_PROXY_HIDDEN, GLM53_PROXY_INTERMEDIATE = 1024, 512
CLAMP = 10.0
# TEGroupedMLP's Fp8Padding pads blockwise expert segments to 16 rows (TE's
# recipe alignment, matching MCore's get_fp8_align_size for non-MXFP8 recipes).
BLOCKWISE_FP8_ALIGN = 16
# E4M3 keeps three mantissa bits, so round-to-nearest has unit roundoff u = 2**-4.
# Block scales put each block's amax near the E4M3 maximum, so values within about
# 2**14 of it are normal numbers whose rounding error is uniform within half a
# spacing: the RMS relative error of one quantized operand is at most
# u / sqrt(3) ~= 0.036 and at least u / (2 sqrt(3)) ~= 0.018.
#
# The FP32 reference evaluates FC1 on the operands the FC1 GEMM itself sees: the
# input in 1x128 E4M3 blocks and the FC1 weight in 128x128 E4M3 blocks (TE quantizes
# BF16 primary weights on the fly; FP8 primary weights are used as stored). An
# independent FC1 cannot be compared at this bound. Its E4M3 noise on the std-0.5
# gate/up rows is about sqrt(2) * 0.036 * ||x * w||_2 ~= 0.8, the clamp keeps only
# |v| <= 10 there, and silu'' reaches 0.5 near 0. The dominant channels' SwiGLU and
# its derivative would then differ by 10-20% relative. Everything else stays FP32 on
# the high-precision operands (BF16 x and W, or the dequantized FP8 primary weights).
# Each compared tensor keeps the E4M3 error of the operand uses the reference does
# not share:
#   output: the FC2 input, and W2 for BF16 params (at most 2)
#   probability VJP: dY and W2 through the FC2 dgrad (at most 2)
#   FC2 weight VJP: dY and the FC2 input, both in 128x1 blocks (2)
#   input VJP: dY and W2, then dproj and W1 in the FC1 dgrad (at most 4)
#   FC1 weight VJP: dY and W2 through dproj, then dproj and x in 128x1 blocks (at most 4)
# Independent errors add in quadrature: at most sqrt(4) * 0.036 ~= 0.072 relative
# RMS error, cosine >= 0.997. A cosine and tensor-similarity bound of 0.99 admits up
# to sqrt(2 * 0.01) ~= 0.14: 1.9 times that worst case (3.8 times in 1 - cosine), or
# 2.6 times at the log-uniform mean roundoff 0.027 per operand. No MCore blockwise-FP8
# module test compares against a high-precision reference (test_early_wgrad.py
# compares two FP8 schedules with identical quantization, rtol=0.02); TE's own
# quantization_tols('fp8_blockwise') is the elementwise rtol=0.125 (2u) for one
# quantized op. The 0.999 bound of the BF16/FP32 cases above (relative RMS <= 0.045)
# is not reliable for four unshared E4M3 operand uses (up to 0.072).
FP8_PARITY_BOUND = 0.99
# 2**-7 separates E4M3 from BF16 arithmetic in relative RMS error. One BF16
# round-to-nearest (unit roundoff 2**-8) has RMS relative error at most
# 2**-8 / sqrt(3) ~= 0.0023, while every compared FP8 tensor keeps at least one
# unshared E4M3 operand use (>= 0.018). Each FP8 tensor must exceed it, so a silent
# BF16 forward or backward GEMM fails. The module's FC1 output must stay below it
# against the shared-operand reference: it differs only by one BF16 output rounding
# and accumulation order, so this checks that the reference reproduces TE's operands.
FP8_ERROR_SPLIT = 2.0**-7
# Clamp-set disagreements. The module's FC1 differs from the shared-operand reference
# by its BF16 output rounding (at most one ULP near |10|, 2**-4) and by accumulation
# order. Hopper FP8 tensor cores keep about 14 bits while accumulating each 128-deep
# K block before FP32 promotion (DeepSeek-V3 technical report, "Increasing
# Accumulation Precision"; block scales force that promotion per K block). With
# unit roundoff 2**-13 per addition, the random-walk error of an n-term block sum of
# x_k * w_k has RMS 2**-13 * sqrt((n + 1) / 6) * ||x * w||_2. A disagreement must lie
# within eight such sigma (Gaussian tail ~1e-15 per element) plus one ULP of the
# threshold: about 0.14 on the std-0.5 rows (||x * w||_2 ~= 16).
CLAMP_NOISE_SIGMAS = 8.0
BF16_ULP_AT_CLAMP = 2.0**-4
FP8_ACCUMULATION_ROUNDOFF = 2.0**-13
FP8_GEMM_K_BLOCK = 128
# FC2-input placement checks. The fused SwiGLU rounds its FP32 result to BF16, then
# rounds again after the FP32 probability product: (1 + 2**-8)**2 - 1 = 2**-7 + 2**-16,
# plus a few FP32 roundings (2**-24); 2**-12 covers both. 2**-100 absorbs FP32 underflow
# of silu for gates below about -87. The eager path rounds three times (silu, the
# product, the probability product): its per-row scale is within
# (1 + 2**-8)**3 - 1 ~= 0.0118 < 2**-6 of the probability.
FUSED_SWIGLU_RELATIVE_ROUNDING = 2.0**-7 + 2.0**-12
FP32_SILU_UNDERFLOW = 2.0**-100
PLACEMENT_ROW_SCALE_TOLERANCE = 2.0**-6


def _skip_unless_blockwise_fp8():
    """Mirror the repository's blockwise-FP8 gates (TE recipe, block-scaled GEMM, Hopper)."""
    if not HAVE_TE or not is_te_min_version('2.3.0.dev0'):
        pytest.skip('blockwise FP8 requires Transformer Engine >= 2.3.0.dev0')
    from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support

    available, reason = check_fp8_block_scaling_support()
    if not available:
        pytest.skip(reason)
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('blockwise FP8 is only supported on Hopper')


def _blockwise_fp8_config(*, fusion, fp8_param, num_experts):
    """GLM53 dev-proxy routed experts under the target blockwise-FP8 precision."""
    return _config(
        after=False,
        fusion=fusion,
        dtype=torch.bfloat16,
        hidden_size=GLM53_PROXY_HIDDEN,
        ffn_hidden_size=3 * GLM53_PROXY_HIDDEN,
        num_attention_heads=8,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=GLM53_PROXY_INTERMEDIATE,
        moe_shared_expert_intermediate_size=GLM53_PROXY_INTERMEDIATE,
        fp8='e4m3',
        fp8_recipe='blockwise',
        fp8_param=fp8_param,
    )


def _blockwise_e4m3_operand(tensor, qparams, block_scaling_dim):
    """Dequantized GEMM operand, quantized like TE's Float8BlockScaling quantizers."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    # Float8BlockScalingRecipeState.make_quantizers builds the forward quantizers from
    # these recipe fields; rowwise data is what the forward GEMM reads.
    quantizer = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=qparams.amax_epsilon,
        force_pow_2_scales=qparams.power_2_scale,
        block_scaling_dim=block_scaling_dim,
    )
    with torch.no_grad():
        return quantizer(tensor.detach()).dequantize(dtype=torch.float32)


def _assert_fp8_parity(actual, expected, name):
    """Compare in FP64 against the FP32 native reference; return the relative RMS error."""
    assert actual is not None and expected is not None, name
    assert actual.shape == expected.shape, name
    a, b = actual.detach().flatten().double(), expected.detach().flatten().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
    aa, bb, ab = torch.dot(a, a), torch.dot(b, b), torch.dot(a, b)
    assert aa > 0 and bb > 0, name
    cosine = (ab / (aa * bb).sqrt()).item()
    similarity = (2 * ab / (aa + bb)).item()
    relative = ((a - b).norm() / bb.sqrt()).item()
    assert cosine > FP8_PARITY_BOUND, f'{name}: cosine={cosine}, relative={relative}'
    assert similarity > FP8_PARITY_BOUND, f'{name}: tensor_similarity={similarity}'
    assert relative >= FP8_ERROR_SPLIT, f'{name}: no E4M3 error, relative={relative}'
    return relative


def _assert_clamp_sets_within_rounding(actual_masks, native_masks, projected, product_norm):
    """Clamp-set disagreements must lie within FC1 rounding of the clamp threshold."""
    gate, up = projected.chunk(2, -1)
    sigma = FP8_ACCUMULATION_ROUNDOFF * math.sqrt((FP8_GEMM_K_BLOCK + 1) / 6.0) * product_norm
    band = BF16_ULP_AT_CLAMP + CLAMP_NOISE_SIGMAS * sigma
    gate_band, up_band = band.chunk(2, -1)
    distances = ((gate - CLAMP).abs(), (up - CLAMP).abs(), (up + CLAMP).abs())
    mismatches = []
    for category, (distance, limit) in enumerate(zip(distances, (gate_band, up_band, up_band))):
        mismatch = actual_masks[category] != native_masks[category]
        far = distance[mismatch] > limit[mismatch]
        assert not far.any(), f'clamp set {category}: {distance[mismatch][far].tolist()}'
        mismatches.append(int(mismatch.sum()))
    return mismatches


def _blockwise_fp8_module_run(counts, *, fusion, fp8_param):
    """Run the GLM53-proxy TEGroupedMLP forward under blockwise FP8.

    Fixed seeds give identical weights and inputs on every call. Hooks capture the
    active recipe, the padded FC1 output, and the padded FC2 input.
    """
    from transformer_engine.common.recipe import Float8BlockScaling, Format
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

    counts = list(counts)
    num_experts = len(counts)
    config = _blockwise_fp8_config(fusion=fusion, fp8_param=fp8_param, num_experts=num_experts)
    assert isinstance(get_fp8_recipe(config), Float8BlockScaling)
    # Build like HybridBlock: fp8_param (fp8_param_gather) creates FP8 primary weights
    # inside the parameter-initialization context; it is a no-op otherwise.
    with get_fp8_context(config, is_init=True):
        layer = get_moe_module_spec(use_te=True, num_experts=num_experts, moe_grouped_gemm=True)(
            config=config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        )
    real = layer.cuda().experts
    assert type(real) is TEGroupedMLP
    assert real._with_fused_impl is False and real._use_grouped_tensor is False
    parameters = dict(real.named_parameters())
    assert set(parameters) == {
        f'linear_fc{fc}.weight{expert}' for fc in (1, 2) for expert in range(num_experts)
    }
    for name, parameter in parameters.items():
        assert isinstance(parameter, Float8BlockwiseQTensor) is fp8_param, name

    generator = torch.Generator(device='cuda').manual_seed(20261015)
    with torch.no_grad():
        for name in sorted(parameters):
            template = 0.02 * torch.randn(
                parameters[name].shape, device='cuda', generator=generator
            )
            if name.startswith('linear_fc1.'):
                # std 0.5 gate/up rows reach both clamp sides, as in the BF16/FP32 case.
                for start in (0, GLM53_PROXY_INTERMEDIATE):
                    template[start : start + 3] *= 25.0
            # FP8 primary weights quantize the template in place (QuantizedTensor.copy_).
            parameters[name].copy_(template.to(torch.bfloat16))

    total = sum(counts)
    hidden = torch.randn(total, GLM53_PROXY_HIDDEN, device='cuda', generator=generator)
    hidden = hidden.to(torch.bfloat16)
    actual_input = hidden.clone().requires_grad_()
    probabilities = torch.linspace(0.21, 0.53, total, device='cuda', dtype=torch.float32)
    actual_prob = probabilities.clone().requires_grad_()

    entries, fc1_outputs, fc2_inputs = [], [], []

    def observe_fc1_entry(_module, inputs):
        entries.append(
            (
                FP8GlobalStateManager.is_fp8_enabled(),
                FP8GlobalStateManager.get_fp8_recipe(),
                list(inputs[1]),
            )
        )

    def observe_fc1_output(_module, _inputs, output):
        fc1_outputs.append(output[0].detach())

    def observe_fc2_entry(_module, inputs):
        fc2_inputs.append((inputs[0].detach(), list(inputs[1])))

    hooks = [
        real.linear_fc1.register_forward_pre_hook(observe_fc1_entry),
        real.linear_fc1.register_forward_hook(observe_fc1_output),
        real.linear_fc2.register_forward_pre_hook(observe_fc2_entry),
    ]
    try:
        with get_fp8_context(config):
            actual, bias = real(actual_input, torch.tensor(counts, dtype=torch.int64), actual_prob)
    finally:
        for hook in hooks:
            hook.remove()
    assert bias is None and actual.dtype == torch.bfloat16 and actual.shape == hidden.shape

    # The expert GEMMs ran inside the blockwise autocast on 16-row padded segments.
    padded = [-(-count // BLOCKWISE_FP8_ALIGN) * BLOCKWISE_FP8_ALIGN for count in counts]
    assert len(entries) == 1 and len(fc1_outputs) == 1 and len(fc2_inputs) == 1
    enabled, recipe, splits = entries[0]
    assert enabled and isinstance(recipe, Float8BlockScaling)
    assert recipe.fp8_format == Format.E4M3
    assert splits == padded, (splits, padded)
    fc1 = fc1_outputs[0]
    fc2_input, fc2_splits = fc2_inputs[0]
    assert fc1.shape == (sum(padded), 2 * GLM53_PROXY_INTERMEDIATE)
    assert fc2_input.shape == (sum(padded), GLM53_PROXY_INTERMEDIATE)
    assert fc2_input.dtype == torch.bfloat16 and fc2_splits == padded
    assert torch.isfinite(fc1).all()
    starts = [sum(padded[:expert]) for expert in range(num_experts)]
    real_rows = torch.cat(
        [torch.arange(start, start + count) for start, count in zip(starts, counts)]
    ).cuda()
    is_padding = torch.ones(sum(padded), dtype=torch.bool, device='cuda')
    is_padding[real_rows] = False
    # Zero padding rows are all-zero forward E4M3 blocks; they must stay exactly zero.
    assert torch.count_nonzero(fc1[is_padding]).item() == 0
    return SimpleNamespace(
        config=config,
        recipe=recipe,
        counts=counts,
        padded=padded,
        real_rows=real_rows,
        is_padding=is_padding,
        parameters=parameters,
        hidden=hidden,
        probabilities=probabilities,
        actual_input=actual_input,
        actual_prob=actual_prob,
        actual=actual,
        fc1=fc1,
        fc2_input=fc2_input,
    )


def _run_blockwise_fp8_experts(counts, *, fusion, fp8_param):
    """Run the blockwise-FP8 experts and an FP32 reference on the FC1 GEMM's operands.

    The reference applies the module's own clamp sets (checked against the reference's
    own sets), so a BF16 rounding tie at the clamp threshold cannot become an O(1)
    gradient difference.
    """
    run = _blockwise_fp8_module_run(counts, fusion=fusion, fp8_param=fp8_param)
    counts, recipe = run.counts, run.recipe
    with torch.no_grad():
        native_weights = {
            name: (parameter.dequantize() if fp8_param else parameter).to(torch.float32, copy=True)
            for name, parameter in run.parameters.items()
        }
        # Rowwise 1x128 blocks do not cross rows, so quantizing the padded input at once
        # matches TE's per-expert split quantization.
        padded_hidden = run.hidden.new_zeros(sum(run.padded), GLM53_PROXY_HIDDEN)
        padded_hidden[run.real_rows] = run.hidden
        fc1_inputs = _blockwise_e4m3_operand(
            padded_hidden, recipe.fp8_quant_fwd_inp, recipe.x_block_scaling_dim
        )[run.real_rows]
        fc1_weights = [
            (
                native_weights[f'linear_fc1.weight{expert}'].detach()
                if fp8_param
                else _blockwise_e4m3_operand(
                    run.parameters[f'linear_fc1.weight{expert}'],
                    recipe.fp8_quant_fwd_weight,
                    recipe.w_block_scaling_dim,
                )
            )
            for expert in range(len(counts))
        ]
    for weight in native_weights.values():
        weight.requires_grad_()
    native_input = run.hidden.to(torch.float32).requires_grad_()
    native_prob = run.probabilities.clone().requires_grad_()

    actual_masks = _clamp_masks(run.fc1[run.real_rows])
    outputs, shared_projected, product_norm = [], [], []
    for expert, (x, x_fc1, probability, masks) in enumerate(
        zip(
            native_input.split(counts),
            fc1_inputs.split(counts),
            native_prob.split(counts),
            actual_masks.split(counts, 1),
        )
    ):
        exact = F.linear(x, native_weights[f'linear_fc1.weight{expert}'])
        shared = F.linear(x_fc1, fc1_weights[expert])
        shared_projected.append(shared)
        product_norm.append((x_fc1.square() @ fc1_weights[expert].square().T).sqrt())
        # Value of the shared-operand projection; gradient of the exact projection, so
        # the dgrad and wgrad terms stay on the high-precision x and W1.
        projected = shared + (exact - exact.detach())
        gate, up = projected.chunk(2, -1)
        gate = gate.masked_fill(masks[0], CLAMP)
        up = up.masked_fill(masks[1], CLAMP).masked_fill(masks[2], -CLAMP)
        # Standard placement: weight the clamped SwiGLU activation, then FC2.
        activation = F.silu(gate) * up * probability[:, None]
        outputs.append(F.linear(activation, native_weights[f'linear_fc2.weight{expert}']))
    shared_projected = torch.cat(shared_projected)
    fc1_error = (
        (run.fc1[run.real_rows].float() - shared_projected).norm() / shared_projected.norm()
    ).item()
    assert fc1_error < FP8_ERROR_SPLIT, f'FC1 does not reproduce its E4M3 operands: {fc1_error}'
    native_masks = _clamp_masks(shared_projected)
    mismatches = _assert_clamp_sets_within_rounding(
        actual_masks, native_masks, shared_projected, torch.cat(product_norm)
    )
    actual_counts = actual_masks.sum((1, 2)).cpu().tolist()
    native_counts = native_masks.sum((1, 2)).cpu().tolist()
    assert all(value > 0 for value in actual_counts), actual_counts
    assert all(value > 0 for value in native_counts), native_counts
    run.native_weights = native_weights
    run.native_input = native_input
    run.native_prob = native_prob
    run.expected = torch.cat(outputs)
    run.coverage = {
        'order': ['gate>10', 'up>10', 'up<-10', 'unclamped'],
        'actual_counts': actual_counts,
        'native_counts': native_counts,
        'active_set_mismatches': mismatches,
        'fc1_shared_operand_error': fc1_error,
    }
    return run


def _assert_standard_probability_placement(run, *, fusion):
    """FC2 receives the clamped SwiGLU activation weighted once by its probability."""
    padded_prob = torch.zeros(sum(run.padded), device='cuda', dtype=torch.float32)
    padded_prob[run.real_rows] = run.probabilities
    gate, up = run.fc1.chunk(2, -1)
    fc2_input = run.fc2_input
    if not fusion:
        # TEGroupedMLP's eager glu ops on the captured BF16 FC1 output, op for op.
        replay = F.silu(gate.clamp(max=CLAMP)) * (
            up.clamp(min=-CLAMP, max=CLAMP) + run.config.glu_linear_offset
        )
        replay = (replay * padded_prob[:, None]).to(torch.bfloat16)
        assert torch.equal(fc2_input, replay)
    unweighted = F.silu(gate.float().clamp(max=CLAMP)) * up.float().clamp(min=-CLAMP, max=CLAMP)
    weighted = unweighted * padded_prob[:, None]
    if fusion:
        bound = FUSED_SWIGLU_RELATIVE_ROUNDING * weighted.abs() + FP32_SILU_UNDERFLOW
        excess = ((fc2_input.float() - weighted).abs() - bound).max().item()
        assert excess <= 0, f'fused FC2 input exceeds two BF16 roundings by {excess}'
    assert torch.count_nonzero(fc2_input[run.is_padding]).item() == 0
    # Negative control: after-FC2 placement feeds FC2 the unweighted activation (row
    # scale 1) and a doubled weighting gives p**2. Each real row's scale must be p.
    rows = run.real_rows
    scale = fc2_input[rows].float().norm(dim=-1) / unweighted[rows].norm(dim=-1)
    deviation = ((scale - run.probabilities).abs() / run.probabilities).max().item()
    assert deviation <= PLACEMENT_ROW_SCALE_TOLERANCE, f'FC2 input row scale deviation {deviation}'


def _bf16_cotangent(shape):
    """Fixed-seed BF16 output cotangent."""
    generator = torch.Generator(device='cuda').manual_seed(531533)
    return torch.randn(shape, device='cuda', generator=generator).to(torch.bfloat16)


def _module_gradients(run):
    """The module's input, probability, and expert-weight gradients by name."""
    gradients = {'input': run.actual_input.grad, 'probability': run.actual_prob.grad}
    gradients.update({name: parameter.grad for name, parameter in run.parameters.items()})
    return gradients


def test_blockwise_FP8_target_requires_standard_probability_placement():
    target = dict(fp8='e4m3', fp8_recipe='blockwise', fp8_param=True, dtype=torch.bfloat16)
    config = _config(after=False, **target)
    assert config.moe_apply_probs_after_fc2 is False
    assert (config.fp8, config.fp8_recipe, config.fp8_param) == ('e4m3', 'blockwise', True)
    with pytest.raises(ValueError, match='supports non-quantized BF16/FP32 only'):
        _config(after=True, **target)


@pytest.mark.internal
@pytest.mark.parametrize('fp8_param', [False, True], ids=['bf16_params', 'fp8_params'])
@pytest.mark.parametrize('fusion', [False, True], ids=['eager_activation', 'fused_activation'])
def test_blockwise_FP8_standard_placement_matches_native_reference(
    _model_parallel, record_property, fusion, fp8_param
):
    _skip_unless_blockwise_fp8()
    # 57 and 150 tokens pad to 64 and 160 rows; 160 spans two 128-row wgrad blocks.
    run = _run_blockwise_fp8_experts((57, 150), fusion=fusion, fp8_param=fp8_param)
    # Placement is exact at the FC2 input; the FP8 parity below cannot tell placements
    # apart, since moving p across FC2 only changes rounding.
    _assert_standard_probability_placement(run, fusion=fusion)
    errors = {'output': _assert_fp8_parity(run.actual, run.expected, 'output')}
    gradient = _bf16_cotangent(run.actual.shape)
    run.actual.backward(gradient)
    run.expected.backward(gradient.to(torch.float32))
    assert run.actual_input.grad.dtype == torch.bfloat16
    assert run.actual_prob.grad.dtype == torch.float32
    errors['input VJP'] = _assert_fp8_parity(
        run.actual_input.grad, run.native_input.grad, 'input VJP'
    )
    errors['FP32 probability VJP'] = _assert_fp8_parity(
        run.actual_prob.grad, run.native_prob.grad, 'FP32 probability VJP'
    )
    for name, parameter in run.parameters.items():
        errors[name] = _assert_fp8_parity(parameter.grad, run.native_weights[name].grad, name)
    record_property(
        'moe_blockwise_fp8_standard_placement',
        json.dumps({**run.coverage, 'relative_errors': errors}, sort_keys=True),
    )


@pytest.mark.internal
@pytest.mark.parametrize('fp8_param', [False, True], ids=['bf16_params', 'fp8_params'])
def test_blockwise_FP8_empty_and_zero_cotangent_experts_get_exact_zero_gradients(
    _model_parallel, record_property, fp8_param
):
    _skip_unless_blockwise_fp8()
    # Expert 1 is routed no tokens (empty GEMMs). Expert 2 receives an all-zero
    # cotangent, so every E4M3 block of its FC2 and FC1 backward gradients has amax 0.
    # TE's compute_scale_from_amax (common/recipe/recipe_common.cuh) maps amax 0 to
    # scale 1, so those blocks quantize to exact zeros without any epsilon. MCore's
    # blockwise recipe keeps TE's default backward amax_epsilon of 0.0, so this case
    # runs exactly that amax-0 path.
    counts = (40, 0, 24, 70)
    silent = slice(40, 64)
    run = _run_blockwise_fp8_experts(counts, fusion=True, fp8_param=fp8_param)
    assert run.recipe.fp8_quant_bwd_grad.amax_epsilon == 0.0
    errors = {'output': _assert_fp8_parity(run.actual, run.expected, 'output')}
    gradient = _bf16_cotangent(run.actual.shape)
    gradient[silent] = 0
    run.actual.backward(gradient)
    run.expected.backward(gradient.to(torch.float32))

    gradients = _module_gradients(run)
    for name, grad in gradients.items():
        assert grad is not None and torch.isfinite(grad).all(), name
    # Zero blocks quantize to exact zeros, and empty GEMMs write zero gradients.
    for expert in (1, 2):
        for fc in (1, 2):
            name = f'linear_fc{fc}.weight{expert}'
            assert torch.count_nonzero(gradients[name]).item() == 0, name
    assert torch.count_nonzero(gradients['input'][silent]).item() == 0
    assert torch.count_nonzero(gradients['probability'][silent]).item() == 0

    live = torch.cat((torch.arange(0, 40), torch.arange(64, 134))).cuda()
    errors['input VJP'] = _assert_fp8_parity(
        gradients['input'][live], run.native_input.grad[live], 'input VJP'
    )
    errors['FP32 probability VJP'] = _assert_fp8_parity(
        gradients['probability'][live], run.native_prob.grad[live], 'FP32 probability VJP'
    )
    for expert in (0, 3):
        for fc in (1, 2):
            name = f'linear_fc{fc}.weight{expert}'
            errors[name] = _assert_fp8_parity(gradients[name], run.native_weights[name].grad, name)
    record_property(
        'moe_blockwise_fp8_zero_gradient_blocks',
        json.dumps({**run.coverage, 'relative_errors': errors}, sort_keys=True),
    )
