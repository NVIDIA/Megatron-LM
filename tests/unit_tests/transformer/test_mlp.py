# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import mlp as mlp_module
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


class _CallRecorder:
    """Wraps a callable, records whether it ran, and delegates to the original."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSquaredReluClampDispatch:
    """Pins which activation path MLP.forward takes for squared ReLU with tanh soft clamping.

    The fused kernel is only exercised when all three of ``activation_func_tanh_clamp_scale``,
    ``activation_func == squared_relu`` and ``use_fused_weighted_squared_relu`` hold. None of the
    fusion unit tests go through ``MLP.forward``, so without this test a broken predicate would
    silently fall back to (or wrongly take) the fused path while every other test keeps passing.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def _restore_rng_state(self):
        """Restore the global RNG after each test so the fixed seeds below do not leak."""
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield

    @staticmethod
    def _build_mlp(activation_func, use_fused_weighted_squared_relu, tanh_clamp_scale):
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=activation_func,
            gated_linear_unit=False,
            bias_activation_fusion=False,
            activation_func_tanh_clamp_scale=tanh_clamp_scale,
            use_fused_weighted_squared_relu=use_fused_weighted_squared_relu,
        )
        submodules = get_submodules(get_gpt_layer_local_submodules().mlp)
        return MLP(config, submodules).cuda()

    @pytest.mark.parametrize(
        ("activation_func", "use_fused_weighted_squared_relu", "tanh_clamp_scale", "expect_fused"),
        [
            (squared_relu, True, 5.0, True),
            (squared_relu, False, 5.0, False),
            (squared_relu, True, None, False),
            (torch.nn.functional.relu, True, 5.0, False),
        ],
    )
    def test_dispatch(
        self,
        monkeypatch,
        activation_func,
        use_fused_weighted_squared_relu,
        tanh_clamp_scale,
        expect_fused,
    ):
        fused = _CallRecorder(mlp_module.weighted_squared_relu_impl)
        clamp = _CallRecorder(mlp_module.tanh_soft_clamp)
        monkeypatch.setattr(mlp_module, "weighted_squared_relu_impl", fused)
        monkeypatch.setattr(mlp_module, "tanh_soft_clamp", clamp)

        mlp = self._build_mlp(activation_func, use_fused_weighted_squared_relu, tanh_clamp_scale)
        hidden_states = torch.randn(8, 2, mlp.config.hidden_size, device="cuda")
        mlp(hidden_states)

        if expect_fused:
            assert fused.calls == 1
            assert clamp.calls == 0
        else:
            assert fused.calls == 0
            # The unfused path applies the clamp separately iff a scale is configured.
            assert clamp.calls == (1 if tanh_clamp_scale is not None else 0)

    def test_fused_matches_unfused(self):
        """The fused path must reproduce squared_relu(tanh_soft_clamp(x)) from the unfused path."""
        torch.manual_seed(0)
        hidden_states = torch.randn(8, 2, 12, device="cuda")

        fused_mlp = self._build_mlp(squared_relu, True, 5.0)
        unfused_mlp = self._build_mlp(squared_relu, False, 5.0)
        # Same seed in _build_mlp, but make weight equality explicit rather than assumed.
        unfused_mlp.load_state_dict(fused_mlp.state_dict())

        fused_out, _ = fused_mlp(hidden_states)
        unfused_out, _ = unfused_mlp(hidden_states)
        torch.testing.assert_close(fused_out, unfused_out, rtol=1e-5, atol=1e-5)


class _SigmoidScaleLinear(torch.nn.Module):
    """Small deterministic linear for testing the real MLP activation dispatch on CPU."""

    def __init__(self, input_size, output_size, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(-1, 1, input_size * output_size).reshape(output_size, input_size)
        )

    def forward(self, x):
        return F.linear(x, self.weight), None


class _UnitSigmoidActivation(torch.nn.Module):
    """Stand-in for TE's fixed sigmoid activation; non-unit scales must bypass it."""

    def forward(self, x):
        gate, linear = x.chunk(2, dim=-1)
        return F.silu(gate) * linear


@pytest.mark.parametrize("is_expert", [False, True], ids=["dense_or_shared", "routed"])
@pytest.mark.parametrize("backend", ["eager", "fused", "te"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tanh_clamp_scale", [None, 2.0])
def test_latent_sigmoid_scale_applies_only_to_routed_experts(
    monkeypatch, is_expert, backend, weighted, tanh_clamp_scale
):
    if backend == "te" and tanh_clamp_scale is not None:
        pytest.skip("TE native activations do not support SiTU")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        ffn_hidden_size=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=4,
        moe_latent_size=4,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=backend == "fused",
        use_te_activation_func=backend == "te",
        moe_latent_sigmoid_input_scale=1.7,
        activation_func_tanh_clamp_scale=tanh_clamp_scale,
        activation_func_tanh_clamp_scale_linear=3.0 if tanh_clamp_scale is not None else None,
    )
    monkeypatch.setattr(
        mlp_module, "get_tensor_model_parallel_group_if_none", lambda group, **kwargs: group
    )
    submodules = MLPSubmodules(
        linear_fc1=_SigmoidScaleLinear,
        linear_fc2=_SigmoidScaleLinear,
        activation_func=lambda config: _UnitSigmoidActivation(),
    )
    mlp = MLP(config, submodules, is_expert=is_expert, ffn_hidden_size=4)
    effective_scale = 1.7 if is_expert else 1.0
    assert mlp.sigmoid_input_scale == effective_scale
    x = torch.linspace(-2, 2, 3 * (4 if is_expert else 8)).reshape(3, -1).requires_grad_()
    probs = torch.tensor([0.2, 0.7, 0.3], requires_grad=True) if weighted else None
    gate, linear = F.linear(x, mlp.linear_fc1.weight).chunk(2, dim=-1)
    gate_factor = gate
    if tanh_clamp_scale is not None:
        gate_factor = tanh_clamp_scale * torch.tanh(gate / tanh_clamp_scale)
        linear = 3.0 * torch.tanh(linear / 3.0)
    activation = gate_factor * torch.sigmoid(effective_scale * gate) * linear
    if weighted:
        activation = activation * probs.unsqueeze(-1)
    reference = F.linear(activation, mlp.linear_fc2.weight)
    output, bias = mlp(x, per_token_scale=probs)
    assert bias is None
    torch.testing.assert_close(output, reference)
    inputs = [x, mlp.linear_fc1.weight, mlp.linear_fc2.weight] + ([probs] if weighted else [])
    grad = torch.randn_like(output)
    reference_grads = torch.autograd.grad(reference, inputs, grad)
    actual_grads = torch.autograd.grad(output, inputs, grad)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
