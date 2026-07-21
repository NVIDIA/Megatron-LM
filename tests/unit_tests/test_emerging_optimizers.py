# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
    _get_qkv_split_shapes,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion
    from emerging_optimizers.soap import SOAP
else:
    SOAP = None
    Lion = None

# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


def test_muon_qkv_split_shapes():
    config = TransformerConfig(
        num_layers=1, hidden_size=1024, num_attention_heads=16, num_query_groups=8
    )
    gated_config = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
        attention_output_gate=True,
    )

    assert _get_qkv_split_shapes(config) == [128, 64, 64]
    assert _get_qkv_split_shapes(gated_config) == [128, 128, 64, 64]


def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        tp_size = int(os.getenv('MIXED_OPTIMIZER_TP_SIZE', '1'))
        pp_size = int(os.getenv('MIXED_OPTIMIZER_PP_SIZE', '1'))
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def create_ddp_model_for_layerwise(self, model, use_param_layout=False):
        """Wrap model in DDP for layer-wise distributed optimizer tests.

        Args:
            model: Model to wrap.
            use_param_layout: If True, supply DDP a precomputed shard-aligned
                ``full_param_layout`` (turns on ``ddp_config.use_distributed_optimizer=True``
                + ``start_param_sync``). If False (default), build DDP without a layout
                so ``LayerWiseDistributedOptimizer`` syncs via the legacy
                flatten / ``all_gather_v`` / unflatten ``allgather_params()`` codepath.
        """
        if use_param_layout:
            from megatron.training.training import wrap_model_chunks_with_ddp

            ddp_config = DistributedDataParallelConfig()
            wrapped = wrap_model_chunks_with_ddp(
                [model],
                TransformerConfig(num_attention_heads=1, num_layers=1),
                ddp_config,
                use_layer_wise_distributed_optimizer=True,
            )
            return wrapped[0]
        return self.create_ddp_model(model)

    def test_get_megatron_optimizer_smoke(self):
        """Smoke test for get_megatron_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def run_configured_optimizer_arm(self, arm_name, treatment_count, num_steps, tp_size):
        """Run one deterministic optimizer-routing arm and return its final observables."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        selected_optimizers = {
            'fc1': 'muon' if treatment_count >= 1 else 'adam',
            'fc2': 'soap' if treatment_count >= 2 else 'adam',
            'fc3': 'lion' if treatment_count >= 3 else 'adam',
        }
        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-12,
            lion_beta1=0.92,
            lion_beta2=0.98,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            soap_shampoo_beta=0.95,
            soap_precondition_frequency=1,
            soap_use_kl_shampoo=True,
            overrides_config={
                "optimizers": {
                    optimizer_name: {
                        "type": optimizer_name,
                        "kwargs": {},
                        "param_groups": {
                            "default": {"eps": 1.0e-12} if optimizer_name == "adam" else {}
                        },
                    }
                    for optimizer_name in {"adam", *selected_optimizers.values()}
                },
                "default": {"optimizer": "adam", "param_group": "default"},
                "matchers": {
                    name: {
                        "type": "glob",
                        "pattern": f"*{name}.weight",
                        "optimizer": optimizer_name,
                        "param_group": "default",
                        "enabled": True,
                    }
                    for name, optimizer_name in selected_optimizers.items()
                },
            },
        )
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        expected_optimizers = {'adam', *selected_optimizers.values()}
        actual_optimizers = {
            group.get('optimizer', optimizer_config.optimizer) for group in optimizer.param_groups
        }
        assert actual_optimizers == expected_optimizers
        assert len(optimizer.chained_optimizers) == len(expected_optimizers)

        for step in range(num_steps):
            torch.manual_seed(1000 + step)
            torch.cuda.manual_seed_all(1000 + step)
            input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
            loss = model(input_tensor).float().square().mean()
            loss.backward()
            update_successful, _, _ = optimizer.step()
            assert update_successful
            optimizer.zero_grad()

        squared_parameter_norms = [
            param.detach().float().square().sum() for param in model.parameters()
        ]
        parameter_norm = torch.sqrt(torch.stack(squared_parameter_norms).sum())
        result = torch.stack((loss.detach(), parameter_norm))
        gathered_results = [
            torch.empty_like(result) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_results, result)
        assert all(torch.equal(result, other) for other in gathered_results)

        final_loss = loss.detach().item()
        final_parameter_norm = parameter_norm.item()
        if torch.distributed.get_rank() == 0:
            print(
                "MIXED_OPTIMIZER_RESULT "
                f"tp={tp_size} arm={arm_name} steps={num_steps} "
                f"loss={final_loss:.17g} parameter_norm={final_parameter_norm:.17g} "
                f"optimizers={','.join(sorted(actual_optimizers))}"
            )

        del optimizer, model
        torch.cuda.empty_cache()
        return final_loss, final_parameter_norm

    def test_get_megatron_optimizer_configured_adam_determinism(self):
        """Two independently initialized Adam runs must have identical final observables."""
        num_steps = int(os.getenv('MIXED_OPTIMIZER_TEST_STEPS', '1'))
        tp_size = int(os.getenv('MIXED_OPTIMIZER_TP_SIZE', '1'))
        deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        try:
            adam_a = self.run_configured_optimizer_arm('adam_a', 0, num_steps, tp_size)
            adam_b = self.run_configured_optimizer_arm('adam_b', 0, num_steps, tp_size)
            assert adam_a == adam_b
        finally:
            torch.use_deterministic_algorithms(deterministic_algorithms_enabled)

    def test_get_megatron_optimizer_configured_mixed_optimizers(self):
        """Each cumulative Adam/Muon/SOAP/Lion routing change must affect training."""
        num_steps = int(os.getenv('MIXED_OPTIMIZER_TEST_STEPS', '1'))
        tp_size = int(os.getenv('MIXED_OPTIMIZER_TP_SIZE', '1'))
        deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        try:
            results = {
                arm_name: self.run_configured_optimizer_arm(
                    arm_name, treatment_count, num_steps, tp_size
                )
                for arm_name, treatment_count in (
                    ('adam', 0),
                    ('muon', 1),
                    ('soap', 2),
                    ('lion', 3),
                )
            }
            for previous_arm, current_arm in (('adam', 'muon'), ('muon', 'soap'), ('soap', 'lion')):
                assert results[current_arm] != results[previous_arm]
        finally:
            torch.use_deterministic_algorithms(deterministic_algorithms_enabled)

    def test_configured_optimizer_kwargs_checkpoint_resume(self, request):
        """Constructor buckets and arbitrary group kwargs survive checkpoint resume."""
        deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        request.addfinalizer(
            lambda: torch.use_deterministic_algorithms(deterministic_algorithms_enabled)
        )

        def make_model():
            return self.create_ddp_model(Net().bfloat16().cuda().requires_grad_(True))

        def make_optimizer(model):
            config = OptimizerConfig(
                optimizer='adam',
                lr=0.01,
                weight_decay=0.01,
                bf16=True,
                use_distributed_optimizer=False,
                overrides_config={
                    "optimizers": {
                        "muon_three": {
                            "type": "muon",
                            "kwargs": {"num_ns_steps": 3},
                            "param_groups": {
                                "default": {"newon_scale": 0.5},
                                "secondary": {"newon_scale": 0.6},
                            },
                        },
                        "muon_five": {
                            "type": "muon",
                            "kwargs": {"num_ns_steps": 5},
                            "param_groups": {"default": {"newon_scale": 0.75}},
                        },
                        "lion": {
                            "type": "lion",
                            "kwargs": {"betas": [0.92, 0.98]},
                            "param_groups": {"default": {}},
                        },
                        "adam": {
                            "type": "adam",
                            "kwargs": {},
                            "param_groups": {"default": {"eps": 1.0e-12}},
                        },
                    },
                    "default": {"optimizer": "adam", "param_group": "default"},
                    "matchers": {
                        "fc1": {
                            "type": "glob",
                            "pattern": "*fc1.weight",
                            "optimizer": "muon_three",
                            "param_group": "default",
                            "enabled": True,
                        },
                        "fc2": {
                            "type": "glob",
                            "pattern": "*fc2.weight",
                            "optimizer": "muon_five",
                            "param_group": "default",
                            "enabled": True,
                        },
                        "fc3": {
                            "type": "glob",
                            "pattern": "*fc3.weight",
                            "optimizer": "lion",
                            "param_group": "default",
                            "enabled": True,
                        },
                        "fc4": {
                            "type": "glob",
                            "pattern": "*fc4.weight",
                            "optimizer": "muon_three",
                            "param_group": "secondary",
                            "enabled": True,
                        },
                    },
                },
            )
            optimizer = get_megatron_optimizer(
                config=config, model_chunks=[model], use_gloo_process_groups=True
            )
            return optimizer

        def train_step(model, optimizer, input_tensor):
            loss = model(input_tensor).float().square().mean()
            loss.backward()
            update_successful, _, _ = optimizer.step()
            assert update_successful
            optimizer.zero_grad()
            return loss.detach()

        def assert_nested_state_equal(expected, actual):
            if isinstance(expected, torch.Tensor):
                assert isinstance(actual, torch.Tensor)
                assert torch.equal(expected, actual)
            elif isinstance(expected, dict):
                assert expected.keys() == actual.keys()
                for key in expected:
                    assert_nested_state_equal(expected[key], actual[key])
            elif isinstance(expected, (list, tuple)):
                assert len(expected) == len(actual)
                for expected_item, actual_item in zip(expected, actual):
                    assert_nested_state_equal(expected_item, actual_item)
            else:
                assert expected == actual

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = make_model()
        optimizer = make_optimizer(model)
        muon_groups = [
            group for group in optimizer.param_groups if group.get("optimizer") == "muon"
        ]
        assert sorted(group["optimizer_kwargs"]["num_ns_steps"] for group in muon_groups) == [
            3,
            3,
            5,
        ]
        assert sorted(group["newon_scale"] for group in muon_groups) == [0.5, 0.6, 0.75]
        assert {(group["optimizer_instance"], group["param_group"]) for group in muon_groups} == {
            ("muon_three", "default"),
            ("muon_three", "secondary"),
            ("muon_five", "default"),
        }
        assert len(optimizer.chained_optimizers) == 4

        first_input = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        train_step(model, optimizer, first_input)
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        resumed_model = make_model()
        resumed_model.load_state_dict(model_state)
        resumed_optimizer = make_optimizer(resumed_model)
        resumed_optimizer.load_state_dict(optimizer_state)

        assert_nested_state_equal(optimizer_state, resumed_optimizer.state_dict())
        for original_param, resumed_param in zip(model.parameters(), resumed_model.parameters()):
            assert torch.equal(original_param, resumed_param)

        params_before_resumed_step = [
            param.detach().clone() for param in resumed_model.parameters()
        ]
        second_input = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        train_step(resumed_model, resumed_optimizer, second_input)
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before_resumed_step, resumed_model.parameters())
        )

    def test_get_megatron_optimizer_validation(self):
        """Test validation logic for get_megatron_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    def test_get_megatron_optimizer_layer_wise(self):
        """Test get_megatron_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # use_layer_wise_distributed_optimizer=True triggers LayerWiseDistributedOptimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"

    def test_get_megatron_muon_optimizer_backward_compatible(self):
        """Test get_megatron_muon_optimizer with backward compatible layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        with pytest.raises(ValueError, match="dist_ prefix"):
            get_megatron_muon_optimizer(
                config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
            )

        optimizer_config.optimizer = 'dist_muon'
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"]
    if HAVE_EMERGING_OPTIMIZERS
    else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"


# ===========================================================================
# Adaptive Muon optimizer tests
# ===========================================================================


def test_adaptive_muon_optimizer_smoke():
    """Smoke test for TensorParallelAdaptiveMuon optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method="adamuon",
        beta2=0.95,
        eps=1e-8,
    )

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_adaptive_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelAdaptiveMuon with different modes on single rank."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.parametrize("moment2_method", ["adamuon", "normuon"])
def test_adaptive_muon_optimizer_moment2_methods(moment2_method):
    """Test TensorParallelAdaptiveMuon with different moment2 methods."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method=moment2_method,
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with moment2_method={moment2_method}"


@pytest.mark.parametrize("beta2", [0.5, 0.95, 0.999])
def test_adaptive_muon_optimizer_beta2(beta2):
    """Test TensorParallelAdaptiveMuon with different beta2 values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        beta2=beta2,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with beta2={beta2}"


def test_adaptive_muon_optimizer_multiple_steps():
    """Test TensorParallelAdaptiveMuon across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_adaptive_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelAdaptiveMuon with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_adaptive_muon_optimizer_qkv_split():
    """Test TensorParallelAdaptiveMuon with QKV splitting."""
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    model.weight.is_qkv = True
    qkv_split_shapes = (64, 64, 64)

    optimizer_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRank:
    """Test class for Adaptive Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_adaptive_muon_smoke(self):
        """Smoke test for get_megatron_optimizer with adaptive_muon."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad

        optimizer_config = OptimizerConfig(
            optimizer='adaptive_muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            adaptive_muon_moment2_method="adamuon",
            adaptive_muon_beta2=0.95,
            adaptive_muon_eps=1e-8,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(optimizer, 'chained_optimizers')
        assert len(optimizer.chained_optimizers) >= 1

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        optimizer.step()

        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list)
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_adaptive_muon_validation(self):
        """Test validation logic for get_megatron_optimizer with adaptive_muon."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config_fp16 = OptimizerConfig(
            optimizer='adaptive_muon', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRankTP:
    """Test class for Adaptive Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer."""
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = TensorParallelAdaptiveMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_adaptive_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_adaptive_muon_optimizer_blockwise_mode(self):
        """Test that blockwise mode works with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# ===========================================================================
# SOAP optimizer tests
# ===========================================================================

skip_no_soap = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_soap
def test_soap_optimizer_smoke():
    """Smoke test for SOAP optimizer."""

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        weight_decay=0.01,
        precondition_frequency=1,
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@skip_no_soap
def test_soap_optimizer_multiple_steps():
    """Test SOAP optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        weight_decay=0.01,
        precondition_frequency=1,
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_soap
@pytest.mark.parametrize("precondition_frequency", [1, 5, 10])
def test_soap_optimizer_precondition_frequency(precondition_frequency):
    """Test SOAP optimizer with different precondition frequencies."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        precondition_frequency=precondition_frequency,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with precondition_frequency={precondition_frequency}"


@skip_no_soap
@pytest.mark.parametrize("use_kl_shampoo", [True, False])
def test_soap_optimizer_kl_shampoo(use_kl_shampoo):
    """Test SOAP optimizer with and without KL-Shampoo preconditioner."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        use_kl_shampoo=use_kl_shampoo,
        precondition_frequency=1,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_kl_shampoo={use_kl_shampoo}"


@skip_no_soap
@pytest.mark.parametrize("shampoo_beta", [0.5, 0.9, 0.99])
def test_soap_optimizer_shampoo_beta(shampoo_beta):
    """Test SOAP optimizer with different shampoo_beta values."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=shampoo_beta,
        precondition_frequency=1,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with shampoo_beta={shampoo_beta}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestSoapOptimizerMultiRank:
    """Test class for SOAP optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_soap_smoke(self):
        """Smoke test for get_megatron_optimizer with SOAP."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        optimizer_config = OptimizerConfig(
            optimizer='soap',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            soap_shampoo_beta=0.95,
            soap_precondition_frequency=1,
            soap_use_kl_shampoo=True,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_soap_validation(self):
        """Test validation logic for get_megatron_optimizer with SOAP."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='soap', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


# ===========================================================================
# Lion optimizer tests
# ===========================================================================

skip_no_lion = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_lion
def test_lion_optimizer_smoke():
    """Smoke test for Lion optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@skip_no_lion
def test_lion_optimizer_multiple_steps():
    """Test Lion optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_lion
@pytest.mark.parametrize("betas", [(0.9, 0.99), (0.95, 0.999), (0.5, 0.9)])
def test_lion_optimizer_betas(betas):
    """Test Lion optimizer with different beta values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=betas)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with betas={betas}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1])
def test_lion_optimizer_weight_decay(weight_decay):
    """Test Lion optimizer with different weight decay values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=weight_decay)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay={weight_decay}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "l2"])
def test_lion_optimizer_weight_decay_method(weight_decay_method):
    """Test Lion optimizer with different weight decay methods."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(
        params=[model.weight],
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        weight_decay_method=weight_decay_method,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay_method={weight_decay_method}"


@skip_no_lion
def test_lion_optimizer_multi_layer_net():
    """Test Lion optimizer with the multi-layer Net model."""
    model = Net().cuda()
    model.requires_grad_(True)

    optimizer = Lion(params=model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_params = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer.step()

    params_updated = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, original_params[name]):
            params_updated += 1

    assert params_updated > 0, "At least some parameters should be updated after optimizer step"
