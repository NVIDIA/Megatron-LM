# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

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
