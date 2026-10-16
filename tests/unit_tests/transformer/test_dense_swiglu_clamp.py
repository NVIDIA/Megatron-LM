# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dense SwiGLU configuration and real-MLP clamp regressions.

HF arithmetic source: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py:86-104, SHA256
2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b.
The dense FFN, not only routed/shared experts, clamps gate/up before SwiGLU.
Use official4096/12288 widths with a short FP32 analytical saturation fixture;
FP32 coordinate checks require IEEE matmul rather than ambient TF32. The
separate composite tests own the BF16 hybrid numerical qualification.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 4096
INTERMEDIATE = 12288


def _config(**overrides):
    values = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=INTERMEDIATE,
        num_attention_heads=64,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_te_activation_func=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_dense_swiglu_default_remains_unclamped():
    config = _config()
    assert config.num_moe_experts is None
    assert config.activation_func_clamp_value is None
    assert config.ffn_hidden_size == INTERMEDIATE


@pytest.mark.parametrize('num_experts', [None, 2])
def test_swiglu_clamp_still_rejects_te_activation(num_experts):
    with pytest.raises(ValueError, match='use_te_activation_func must be False'):
        _config(
            num_moe_experts=num_experts,
            activation_func_clamp_value=10.0,
            use_te_activation_func=True,
        )


@pytest.fixture
def _model_parallel():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        assert torch.get_float32_matmul_precision() == 'highest'
        assert torch.backends.cuda.matmul.allow_tf32 is False
        fp32_precision = getattr(torch.backends.cuda.matmul, 'fp32_precision', None)
        assert fp32_precision in (None, 'ieee')
        print(
            f'DENSE_SWIGLU_PRECISION rank={os.environ.get("RANK", "0")} '
            f'matmul={torch.get_float32_matmul_precision()} '
            f'allow_tf32={torch.backends.cuda.matmul.allow_tf32} fp32_precision={fp32_precision}'
        )
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(531532)
        torch.manual_seed(531532)
        yield
    finally:
        try:
            Utils.destroy_model_parallel()
        finally:
            torch.set_float32_matmul_precision(original_precision)


def test_te_op_fuser_dense_mlp_rejects_clamp(_model_parallel):
    """TEFusedMLP builds an unclamped TE SwiGLU op, so it must refuse the clamp."""
    from megatron.core.extensions import transformer_engine as te_ext

    te_fused_mlp = getattr(te_ext, 'TEFusedMLP', None)
    if not te_ext.HAVE_TE or te_fused_mlp is None:
        pytest.skip('TEFusedMLP needs Transformer Engine operation-based API support')
    submodules = MLPSubmodules(
        linear_fc1=te_ext.TELayerNormColumnParallelLinear, linear_fc2=te_ext.TERowParallelLinear
    )
    small = dict(hidden_size=64, ffn_hidden_size=128, num_attention_heads=4)
    te_fused_mlp(_config(**small), submodules, ffn_hidden_size=128)
    with pytest.raises(ValueError, match='^TEFusedMLP does not support activation_func_clamp'):
        te_fused_mlp(
            _config(activation_func_clamp_value=10.0, **small), submodules, ffn_hidden_size=128
        )
    # The subclass inherits the check, which runs before its own TE-version checks.
    grouped = getattr(te_ext, 'TEFusedMLPWithGroupedLinear', None)
    if grouped is not None:
        with pytest.raises(ValueError, match='^TEFusedMLPWithGroupedLinear does not support'):
            grouped(
                _config(activation_func_clamp_value=10.0, **small), submodules, ffn_hidden_size=128
            )


def _assert_parity(actual, expected, name):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    a, b = actual.detach().flatten().double(), expected.detach().flatten().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
    dot = torch.dot(a, b)
    norm_a, norm_b = torch.dot(a, a), torch.dot(b, b)
    assert norm_a > 0 and norm_b > 0, name
    cosine = (dot / torch.sqrt(norm_a * norm_b)).item()
    similarity = (2 * dot / (norm_a + norm_b)).item()
    assert cosine > 0.999, f'{name}: cosine={cosine}'
    assert similarity > 0.999, f'{name}: tensor_similarity={similarity}'
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.internal
@pytest.mark.parametrize('fusion', [False, True])
@pytest.mark.parametrize('clamp', [None, 10.0])
def test_dense_swiglu_native_output_input_and_all_gradients(_model_parallel, fusion, clamp):
    """Exercise existing dense fused/unfused math, including exact saturated zeros."""
    config = _config(activation_func_clamp_value=clamp, bias_activation_fusion=fusion)
    assert config.num_moe_experts is None
    submodules = get_submodules(get_gpt_layer_local_submodules().mlp)
    real = MLP(config, submodules, ffn_hidden_size=INTERMEDIATE).cuda()
    parameters = dict(real.named_parameters())
    assert set(parameters) == {'linear_fc1.weight', 'linear_fc2.weight'}
    fc1, fc2 = parameters['linear_fc1.weight'], parameters['linear_fc2.weight']
    assert fc1.shape == (2 * INTERMEDIATE, HIDDEN)
    assert fc2.shape == (HIDDEN, INTERMEDIATE)
    with torch.no_grad():
        # All tokens have x[0]=1: the three saturated categories are exact and
        # do not sit on a clamp boundary. Ordinary non-saturated rows coexist.
        gate = torch.tensor([12.0, 3.0, -3.0, -12.0], device='cuda').repeat(INTERMEDIATE // 4)
        up = torch.tensor([2.0, 12.0, -12.0, 2.0], device='cuda').repeat(INTERMEDIATE // 4)
        fc1.zero_()
        fc1[:, 0].copy_(torch.cat((gate, up)))
        fc2.uniform_(0.001, 0.002)
    native_parameters = {
        name: parameter.detach().clone().requires_grad_() for name, parameter in parameters.items()
    }
    hidden = torch.zeros(2, 1, HIDDEN, device='cuda')
    hidden[..., 0] = 1.0
    actual_input = hidden.clone().requires_grad_()
    native_input = hidden.clone().requires_grad_()
    actual, bias = real(actual_input)
    assert bias is None
    projected = F.linear(native_input, native_parameters['linear_fc1.weight'])
    native_gate, native_up = projected.chunk(2, -1)
    if clamp is not None:
        native_gate = native_gate.clamp(max=clamp)
        native_up = native_up.clamp(min=-clamp, max=clamp)
    expected = F.linear(F.silu(native_gate) * native_up, native_parameters['linear_fc2.weight'])
    _assert_parity(actual, expected, 'dense output')
    gradient = torch.ones_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    _assert_parity(actual_input.grad, native_input.grad, 'dense input gradient')
    for name, parameter in parameters.items():
        native = native_parameters[name]
        assert parameter.requires_grad and native.requires_grad, name
        assert parameter.grad is not None and native.grad is not None, name
        _assert_parity(parameter.grad, native.grad, name)
    gate_gradient, up_gradient = fc1.grad.chunk(2, 0)
    if clamp is not None:
        assert (gate > clamp).any() and (up > clamp).any() and (up < -clamp).any()
        assert torch.count_nonzero(gate_gradient[gate > clamp]) == 0
        assert torch.count_nonzero(up_gradient[up.abs() > clamp]) == 0
        assert torch.count_nonzero(gate_gradient[gate <= clamp]) > 0
        assert torch.count_nonzero(up_gradient[up.abs() < clamp]) > 0
    else:
        assert torch.count_nonzero(gate_gradient[gate > 10]) > 0
        assert torch.count_nonzero(up_gradient[up.abs() > 10]) > 0
