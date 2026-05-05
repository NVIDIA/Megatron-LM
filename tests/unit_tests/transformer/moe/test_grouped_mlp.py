# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import argparse
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.moe.experts as experts_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import _add_network_size_args, parse_args
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def test_op_fuser_transformer_config_args_are_exposed():
    parser = argparse.ArgumentParser()
    _add_network_size_args(parser)

    args = parser.parse_args(
        ["--use-transformer-engine-op-fuser", "--moe-mlp-glu-interleave-size", "16"]
    )

    assert args.use_transformer_engine_op_fuser is True
    assert args.moe_mlp_glu_interleave_size == 16


def test_remove_glu_interleaving_restores_contiguous_gate_and_linear_halves():
    interleaved = torch.tensor(
        [
            [1, 2, 5, 6, 3, 4, 7, 8],
            [11, 12, 15, 16, 13, 14, 17, 18],
        ]
    )
    expected = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
        ]
    )

    output = TEGroupedMLP._remove_glu_interleaving(interleaved, interleave_size=2)

    torch.testing.assert_close(output, expected)


def test_make_fused_ops_reuses_grouped_linear_weights_on_meta_device(monkeypatch):
    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            in_features,
            out_features,
            *,
            bias,
            device,
            dtype,
            accumulate_into_main_grad,
            single_grouped_weight,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.device = device
            self.dtype = dtype
            self.fuse_wgrad_accumulation = accumulate_into_main_grad
            self.single_grouped_weight = single_grouped_weight

        def need_backward_dw(self):
            return False

    class FakeScaledSwiGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size

    class FakeSequential(list):
        def register_forward_pre_hook(self, hook):
            self.forward_pre_hook = hook

    fake_te = SimpleNamespace(
        pytorch=SimpleNamespace(
            GroupedLinear=FakeGroupedLinear,
            ops=SimpleNamespace(
                GroupedLinear=FakeGroupedLinear,
                ScaledSwiGLU=FakeScaledSwiGLU,
                Sequential=FakeSequential,
            ),
        )
    )
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_mlp_glu_interleave_size=16)
    module.linear_fc1 = FakeGroupedLinear(
        2,
        4,
        8,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=True,
        single_grouped_weight=False,
    )
    module.linear_fc2 = FakeGroupedLinear(
        2,
        8,
        4,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=True,
    )
    module.linear_fc1.weight0 = torch.nn.Parameter(torch.ones(8, 4))
    module.linear_fc1.weight1 = torch.nn.Parameter(torch.ones(8, 4) * 2)
    module.linear_fc1.bias0 = torch.nn.Parameter(torch.zeros(8))
    module.linear_fc1.bias1 = torch.nn.Parameter(torch.ones(8))
    module.linear_fc2.weight = torch.nn.Parameter(torch.ones(4, 8))

    ops = module._make_fused_ops()

    assert len(ops) == 3
    assert ops[0].device == "meta"
    assert ops[0].weight0 is module.linear_fc1.weight0
    assert ops[0].weight1 is module.linear_fc1.weight1
    assert ops[0].bias0 is module.linear_fc1.bias0
    assert ops[0].bias1 is module.linear_fc1.bias1
    assert ops[1].glu_interleave_size == 16
    assert ops[2].device == "meta"
    assert ops[2].weight is module.linear_fc2.weight
    assert hasattr(ops, "forward_pre_hook")


def test_fused_forward_caches_ops_and_forwards_expected_arguments():
    class FakeFusedOps:
        def __call__(self, hidden_states, fc1_tokens, probs, fc2_tokens):
            self.args = (hidden_states, fc1_tokens, probs, fc2_tokens)
            return hidden_states + 1

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    module.config = SimpleNamespace(
        fp8=False, fp4=False, moe_router_padding_for_quantization=False
    )
    module._fused_ops = None
    fused_ops = FakeFusedOps()
    module._make_fused_ops = lambda: fused_ops
    hidden_states = torch.zeros(2, 4)
    tokens_per_expert = torch.tensor([1, 1])
    probs = torch.ones(2)

    output = module._fused_forward(hidden_states, tokens_per_expert, probs)

    torch.testing.assert_close(output, torch.ones_like(hidden_states))
    assert module._fused_ops[0] is fused_ops
    assert fused_ops.args[0] is hidden_states
    assert fused_ops.args[1] is tokens_per_expert
    assert fused_ops.args[2] is probs
    assert fused_ops.args[3] is tokens_per_expert


def test_apply_bias_returns_input_unchanged_when_bias_is_none():
    intermediate = torch.arange(6, dtype=torch.float32).view(3, 2)

    output = TEGroupedMLP._apply_bias(
        intermediate, bias_parallel=None, tokens_per_expert=[2, 1], permuted_probs=torch.ones(3)
    )

    assert output is intermediate


def test_apply_bias_combines_per_expert_bias_and_probs():
    intermediate = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )
    bias_parallel = [
        torch.tensor([10.0, 20.0]),
        torch.tensor([100.0, 200.0]),
    ]
    tokens_per_expert = [2, 1]
    permuted_probs = torch.tensor([0.5, 0.5, 1.0])
    expected = torch.tensor(
        [[6.0, 12.0], [8.0, 14.0], [105.0, 206.0]], dtype=torch.float32
    )

    output = TEGroupedMLP._apply_bias(
        intermediate, bias_parallel, tokens_per_expert, permuted_probs
    )

    torch.testing.assert_close(output, expected)
    assert output.dtype == intermediate.dtype


def test_make_fused_impl_pre_forward_hook_dispatches_submodule_hooks():
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    fc1_child = torch.nn.Linear(2, 2)
    fc2_child = torch.nn.Linear(2, 2)
    module.linear_fc1 = torch.nn.Sequential(fc1_child)
    module.linear_fc2 = torch.nn.Sequential(fc2_child)

    calls = []

    def fc1_hook(submodule, _inp):
        calls.append(("fc1", submodule))
        return None

    def fc2_hook(submodule, _inp):
        calls.append(("fc2", submodule))
        return None

    fc1_child.register_forward_pre_hook(fc1_hook)
    fc2_child.register_forward_pre_hook(fc2_hook)

    hook = module._make_fused_impl_pre_forward_hook()
    hook(object())

    visited = {label for label, _ in calls}
    assert visited == {"fc1", "fc2"}


def test_make_fused_impl_pre_forward_hook_rejects_input_modifying_hook():
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    fc1_child = torch.nn.Linear(2, 2)
    module.linear_fc1 = torch.nn.Sequential(fc1_child)
    module.linear_fc2 = torch.nn.Sequential(torch.nn.Linear(2, 2))

    fc1_child.register_forward_pre_hook(lambda submodule, _inp: torch.zeros(1))

    hook = module._make_fused_impl_pre_forward_hook()

    with pytest.raises(RuntimeError, match="modifies the input tensor"):
        hook(object())


def test_make_fused_ops_handles_single_grouped_weight_for_fc1(monkeypatch):
    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            in_features,
            out_features,
            *,
            bias,
            device,
            dtype,
            accumulate_into_main_grad,
            single_grouped_weight,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.device = device
            self.dtype = dtype
            self.fuse_wgrad_accumulation = accumulate_into_main_grad
            self.single_grouped_weight = single_grouped_weight

        def need_backward_dw(self):
            return False

    class FakeScaledSwiGLU(torch.nn.Module):
        def __init__(self, glu_interleave_size):
            super().__init__()
            self.glu_interleave_size = glu_interleave_size

    class FakeSequential(list):
        def register_forward_pre_hook(self, hook):
            self.forward_pre_hook = hook

    fake_te = SimpleNamespace(
        pytorch=SimpleNamespace(
            GroupedLinear=FakeGroupedLinear,
            ops=SimpleNamespace(
                GroupedLinear=FakeGroupedLinear,
                ScaledSwiGLU=FakeScaledSwiGLU,
                Sequential=FakeSequential,
            ),
        )
    )
    monkeypatch.setattr(experts_module, "te", fake_te)

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(moe_mlp_glu_interleave_size=8)
    module.linear_fc1 = FakeGroupedLinear(
        2,
        4,
        8,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=False,
        single_grouped_weight=True,
    )
    module.linear_fc2 = FakeGroupedLinear(
        2,
        8,
        4,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
        accumulate_into_main_grad=True,
        single_grouped_weight=False,
    )
    module.linear_fc1.weight = torch.nn.Parameter(torch.ones(2, 8, 4))
    module.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(4, 8))
    module.linear_fc2.weight1 = torch.nn.Parameter(torch.ones(4, 8) * 2)
    module.linear_fc2.bias0 = torch.nn.Parameter(torch.zeros(4))
    module.linear_fc2.bias1 = torch.nn.Parameter(torch.ones(4))

    ops = module._make_fused_ops()

    assert ops[0].weight is module.linear_fc1.weight
    assert ops[1].glu_interleave_size == 8
    assert ops[2].weight0 is module.linear_fc2.weight0
    assert ops[2].weight1 is module.linear_fc2.weight1
    assert ops[2].bias0 is module.linear_fc2.bias0
    assert ops[2].bias1 is module.linear_fc2.bias1


@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TE Grouped MLP is only supported in TE 1.9.0.dev0 and later.",
)
class TestTEGroupedMLP:

    def setup_method(self, method, use_cpu_initialization=False, swiglu=True):
        Utils.initialize_model_parallel(1, 1)
        num_layers = 1
        self.hidden_size = 16
        self.num_experts = 2
        self.gated_linear_unit = swiglu
        self.activation_func = F.silu if swiglu else F.gelu
        self.use_cpu_initialization = use_cpu_initialization

        tf_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=self.use_cpu_initialization,
            add_bias_linear=False,
            gated_linear_unit=self.gated_linear_unit,
            activation_func=self.activation_func,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
        )

        self.fc1_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc2_ffn_hidden_size = tf_config.ffn_hidden_size
        # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.gated_linear_unit:
            self.fc1_ffn_hidden_size *= 2

        ## Vanilla sequential GEMM
        # Set random seed for reproducability
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        sequential_submodules = get_submodules(
            get_gpt_layer_local_submodules(self.num_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(sequential_submodules, MoESubmodules)
        self.sequential_mlp = MoELayer(tf_config, sequential_submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        self.sequential_mlp = Float16Module(self.sequential_mlp.config, self.sequential_mlp).module

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        tf_config.moe_grouped_gemm = True
        grouped_submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp
        )
        assert isinstance(grouped_submodules, MoESubmodules)
        self.grouped_mlp = MoELayer(tf_config, grouped_submodules)
        assert isinstance(self.grouped_mlp.experts, TEGroupedMLP)
        self.grouped_mlp = Float16Module(self.grouped_mlp.config, self.grouped_mlp).module

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.sequential_mlp, MoELayer)
        assert isinstance(self.grouped_mlp, MoELayer)

        num_weights_smm = sum([p.numel() for p in self.sequential_mlp.parameters()])
        num_weights_gmm = sum([p.numel() for p in self.grouped_mlp.parameters()])

        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # GroupedGEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm
        # expected num weights: router linear weights+bias + MLP weights(no bias) of all experts
        expected_num_weights = (
            self.hidden_size * self.num_experts
            + self.hidden_size
            * (self.fc1_ffn_hidden_size + self.fc2_ffn_hidden_size)
            * self.num_experts
        )
        assert num_weights_smm == expected_num_weights

        assert torch.equal(self.sequential_mlp.router.weight, self.grouped_mlp.router.weight)

        # weights of linear_fc1: [fc1_ffn_hidden_size, hidden_size]
        # weights of linear_fc2: [hidden_size, fc2_ffn_hidden_size]
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").shape == (
                self.fc1_ffn_hidden_size,
                self.hidden_size,
            )
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").shape == (
                self.hidden_size,
                self.fc2_ffn_hidden_size,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward_backward(self):
        self.sequential_mlp.cuda()
        self.grouped_mlp.cuda()
        # Copy the weights to ensure the same init value
        with torch.no_grad():
            for i in range(self.num_experts):
                self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.copy_(
                    getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}")
                )
                self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.copy_(
                    getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}")
                )
        # [sequence length, batch size, hidden size]
        seq_len = 32
        batch_size = 2
        hidden_states = torch.rand(
            (seq_len, batch_size, self.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        hidden_states.retain_grad()

        output_smm, _ = self.sequential_mlp(hidden_states)
        output_smm.mean().backward()
        smm_results = [output_smm, hidden_states.grad]
        for i in range(self.num_experts):
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.grad)
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.grad)

        hidden_states.grad = None
        output_gmm, _ = self.grouped_mlp(hidden_states)
        output_gmm.mean().backward()
        gmm_results = [output_gmm, hidden_states.grad]
        for i in range(self.num_experts):
            gmm_results.append(getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad)
            gmm_results.append(getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad)

        for smm_result, gmm_result in zip(smm_results, gmm_results):
            torch.testing.assert_close(smm_result, gmm_result)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward_backward_with_no_tokens_allocated(self):
        """Test the case when no token is allocated for groupedGEMM kernels."""
        self.grouped_mlp.cuda()
        num_allocated_tokens = 0
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.int32)
        hidden_states = torch.rand((num_allocated_tokens, self.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        probs = torch.rand((num_allocated_tokens,), dtype=torch.float32)
        probs = probs.cuda()
        output, _ = self.grouped_mlp.experts(
            hidden_states, tokens_per_expert=tokens_per_expert, permuted_probs=probs
        )
        assert torch.equal(output, torch.zeros_like(output))
        assert output.shape == (num_allocated_tokens, self.hidden_size)

        output.mean().backward()
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad is not None
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_make_fused_ops_constructs_with_real_te(self):
        """Verify `_make_fused_ops` builds a working TE op-fuser pipeline.

        Regression guard for the `device="meta"` shell + weight-reattachment
        sequence — a TE-side change that allocates state in the constructor
        would break this without any of the mock-based unit tests catching it.
        """
        try:
            from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU
        except ImportError:
            pytest.skip("TE op fuser API not available")
        import inspect

        if "single_grouped_weight" not in inspect.signature(GroupedLinear.__init__).parameters:
            pytest.skip(
                "Installed TE op fuser GroupedLinear lacks `single_grouped_weight` kwarg; "
                "_make_fused_ops requires a TE build that exposes it."
            )

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)

        tf_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=False,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        layer = MoELayer(
            tf_config,
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp.submodules,
        )
        layer = Float16Module(layer.config, layer).module
        layer.cuda()
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)
        assert experts._with_fused_impl

        ops = experts._make_fused_ops()

        assert len(ops) == 3
        assert isinstance(ops[0], GroupedLinear)
        assert isinstance(ops[1], ScaledSwiGLU)
        assert isinstance(ops[2], GroupedLinear)
        # Weights of the wrapper ops must alias the underlying GroupedLinear
        # parameters so optimizer updates are visible to the fused path.
        for idx in range(experts.linear_fc1.num_gemms):
            if not getattr(experts.linear_fc1, "single_grouped_weight", False):
                assert getattr(ops[0], f"weight{idx}") is getattr(
                    experts.linear_fc1, f"weight{idx}"
                )
        for idx in range(experts.linear_fc2.num_gemms):
            if not getattr(experts.linear_fc2, "single_grouped_weight", False):
                assert getattr(ops[2], f"weight{idx}") is getattr(
                    experts.linear_fc2, f"weight{idx}"
                )
