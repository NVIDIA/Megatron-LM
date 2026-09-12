# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("num_tokens", [0, 4])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_weighted_mlp_fusion_preserves_swiglu_clamp(monkeypatch, num_tokens, device):
    """Exercise MLP's activation dispatch, including saturated input and router gradients."""
    if device == "cpu":
        monkeypatch.setattr("megatron.core.utils._nvtx_enabled", False)

    class IdentityLinear(torch.nn.Module):
        def forward(self, inputs):
            return inputs, None

    # Isolate activation dispatch from GEMMs and process-group initialization.
    mlp = MLP.__new__(MLP)
    torch.nn.Module.__init__(mlp)
    mlp.config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=1,
        num_moe_experts=2,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=1.0,
        bias_activation_fusion=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    mlp.activation_func = F.silu
    mlp.linear_fc1 = IdentityLinear()
    mlp.linear_fc2 = IdentityLinear()
    x = (
        torch.tensor(
            [[-2, -1, 0, 0.5, 1, 2, -2, -1, -0.5, 0.5, 1, 2]], dtype=torch.bfloat16, device=device
        )
        .repeat(num_tokens, 1)
        .requires_grad_(True)
    )
    probs = torch.linspace(0.25, 1.0, num_tokens, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_probs = probs.detach().clone().requires_grad_(True)
    gate, linear = reference_x.float().chunk(2, dim=-1)
    # Preserve DSv4's existing rounding before multiplying by routing probabilities.
    activation = (F.silu(gate.clamp(max=1.0)) * linear.clamp(-1.0, 1.0)).to(x.dtype)
    expected = (activation * reference_probs.unsqueeze(-1)).to(x.dtype)
    output, bias = mlp(x, per_token_scale=probs)
    assert bias is None
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    grad = torch.ones_like(output)
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(probs.grad, reference_probs.grad, rtol=2e-2, atol=1e-3)
    if num_tokens:
        assert torch.count_nonzero(x.grad[:, [5, 6, 11]]) == 0


class TestParallelMLP:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        mlp_submodules = get_submodules(get_gpt_layer_local_submodules().mlp)
        assert isinstance(mlp_submodules, MLPSubmodules)
        self.mlp = MLP(transformer_config, mlp_submodules)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == 1212

    """
    def test_cpu_forward(self, mlp):
        # [sequence length, micro batch size, hidden size]
        hidden_states = torch.ones((32, 2, mlp.config.hidden_size))
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == torch.float32
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        mlp = self.mlp
        mlp.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        assert output_bias.device.type == 'cuda'
