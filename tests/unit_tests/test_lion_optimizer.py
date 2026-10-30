# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for Lion optimizer plumbing between Megatron and emerging_optimizers.

Tests that require emerging_optimizers are skipped when the package is not installed.
Tests that verify Megatron-only behavior (config, error handling) always run.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer import (
    HAVE_EMERGING_OPTIMIZERS,
    OptimizerConfig,
    _get_megatron_optimizer_based_on_param_groups,
    _get_param_groups,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.emerging_optimizers import _eopt_init_state_fn
from megatron.core.optimizer.optimizer import FP32Optimizer

requires_emerging_optimizers = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def _make_param_groups(model):
    """Create simple param groups without distributed calls."""
    return [{"params": list(model.parameters()), "is_expert_parallel": False}]


def _make_pg_collection():
    """Create a mock pg_collection with a tp attribute to avoid parallel_state calls."""
    return SimpleNamespace(tp=None)


def _create_lion_optimizer(model, config):
    """Helper to create a Lion optimizer via Megatron's factory."""
    return _get_megatron_optimizer_based_on_param_groups(
        config=config,
        model_chunks=[model],
        param_groups=_make_param_groups(model),
        pg_collection=_make_pg_collection(),
    )


class TestLionOptimizerConfig:
    """Tests for Lion config plumbing that do not require emerging_optimizers."""

    def test_lion_config_fields_exist(self):
        """Config should accept lion_beta1, lion_beta2, and muon_scalar_optimizer."""
        config = OptimizerConfig(
            optimizer="lion",
            lr=1e-4,
            lion_beta1=0.92,
            lion_beta2=0.97,
            muon_scalar_optimizer="lion",
        )
        assert config.lion_beta1 == 0.92
        assert config.lion_beta2 == 0.97
        assert config.muon_scalar_optimizer == "lion"

    def test_lion_config_defaults(self):
        """Config defaults should match expected Lion defaults."""
        config = OptimizerConfig()
        assert config.lion_beta1 == 0.95
        assert config.lion_beta2 == 0.98
        assert config.muon_scalar_optimizer == "adam"

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch(
        "torch.distributed.all_gather_object",
        lambda output_list, obj, **_: output_list.__setitem__(0, obj),
    )
    def test_lion_param_groups_via_get_param_groups(self, mock_world_size):
        """_get_param_groups should work with lion config (same as adam)."""
        model = SimpleModel()
        config = OptimizerConfig(optimizer="lion", lr=1e-4)
        param_groups = _get_param_groups([model], config, {})

        assert len(param_groups) == 1
        assert param_groups[0]["params"] == list(model.parameters())

    def test_lion_import_error_without_package(self):
        """Should raise ImportError with helpful message if emerging_optimizers not installed."""
        import megatron.core.optimizer as opt_module

        original_have_lion = opt_module.HAVE_EMERGING_OPTIMIZERS
        try:
            opt_module.HAVE_EMERGING_OPTIMIZERS = False

            model = SimpleModel()
            config = OptimizerConfig(optimizer="lion", lr=1e-4)

            with pytest.raises(ImportError, match="emerging_optimizers"):
                _create_lion_optimizer(model, config)
        finally:
            opt_module.HAVE_EMERGING_OPTIMIZERS = original_have_lion


@requires_emerging_optimizers
class TestLionOptimizerExactness:
    """Tests that verify bit-for-bit exactness between Megatron-plumbed Lion and standalone Lion.

    These tests require emerging_optimizers to be installed.
    """

    def test_lion_factory_creates_lion_optimizer(self):
        """_get_megatron_optimizer_based_on_param_groups should create a Lion-backed optimizer."""
        from emerging_optimizers.scalar_optimizers import Lion

        model = SimpleModel()
        config = OptimizerConfig(
            optimizer="lion", lr=3e-4, lion_beta1=0.93, lion_beta2=0.99, weight_decay=0.01
        )

        optimizer = _create_lion_optimizer(model, config)

        # Should be wrapped in FP32Optimizer since no fp16/bf16 set.
        assert isinstance(optimizer, FP32Optimizer)

        # The underlying optimizer should be Lion from emerging_optimizers.
        inner_opt = optimizer.optimizer
        assert isinstance(inner_opt, Lion)

        # Verify betas were plumbed through exactly.
        for group in inner_opt.param_groups:
            assert group["betas"] == (0.93, 0.99)
            assert group["lr"] == 3e-4
            assert group["weight_decay"] == 0.01

    def test_lion_init_state_fn_creates_complete_state(self):
        """Use the installed Lion's complete state, including step in EO >= 0.5."""
        from emerging_optimizers.scalar_optimizers import Lion

        model = SimpleModel()
        config = OptimizerConfig(optimizer="lion", lr=1e-4)

        optimizer = _create_lion_optimizer(model, config)
        inner_opt = optimizer.optimizer

        # State should be empty before init.
        for p in model.parameters():
            assert len(inner_opt.state[p]) == 0

        # Call init_state_fn (stored on the FP32Optimizer wrapper).
        optimizer.init_state_fn(inner_opt)
        reference = Lion(model.parameters(), lr=config.lr)
        for group in reference.param_groups:
            reference._init_group(group, skip_non_grad_params=False)

        # State should now have exp_avg for each param, exactly zero.
        for p in model.parameters():
            assert inner_opt.state[p].keys() == reference.state[p].keys()
            assert "exp_avg" in inner_opt.state[p]
            assert inner_opt.state[p]["exp_avg"].shape == p.shape
            assert inner_opt.state[p]["exp_avg"].dtype == p.dtype
            torch.testing.assert_close(
                inner_opt.state[p]["exp_avg"], torch.zeros_like(p.data), atol=0, rtol=0
            )
            if "step" in reference.state[p]:
                assert inner_opt.state[p]["step"] == reference.state[p]["step"] == 0

    @pytest.mark.parametrize(
        "lr,beta1,beta2,weight_decay",
        [(1e-3, 0.95, 0.98, 0.0), (3e-4, 0.9, 0.99, 0.01), (1e-4, 0.85, 0.95, 0.1)],
    )
    @pytest.mark.parametrize("initialize_state", [False, True])
    def test_megatron_lion_exact_match_with_standalone(
        self, lr, beta1, beta2, weight_decay, initialize_state
    ):
        """Megatron-plumbed Lion must produce bit-for-bit identical results to standalone Lion.

        This is the core correctness test: we run the same forward-backward-step on two
        identical models — one using Lion directly, one through Megatron's factory — and
        verify that all parameters and optimizer states are exactly equal.
        """
        from emerging_optimizers.scalar_optimizers import Lion

        # Create two identical models from the same seed.
        torch.manual_seed(0)
        model_standalone = SimpleModel()
        torch.manual_seed(0)
        model_megatron = SimpleModel()

        # Sanity: models start identical.
        for (n1, p1), (n2, p2) in zip(
            model_standalone.named_parameters(), model_megatron.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1.data, p2.data, atol=0, rtol=0)

        # Create standalone Lion directly from emerging_optimizers.
        opt_standalone = Lion(
            model_standalone.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )

        # Create Lion through Megatron's factory.
        config = OptimizerConfig(
            optimizer="lion", lr=lr, lion_beta1=beta1, lion_beta2=beta2, weight_decay=weight_decay
        )
        megatron_optimizer = _create_lion_optimizer(model_megatron, config)
        opt_megatron = megatron_optimizer.optimizer
        if initialize_state:
            megatron_optimizer.init_state_fn(opt_megatron)

        # Run multiple steps to accumulate momentum differences if any.
        for step in range(3):
            torch.manual_seed(step + 100)
            x = torch.randn(4, 16)

            # Standalone forward-backward-step.
            loss_standalone = model_standalone(x).sum()
            loss_standalone.backward()
            opt_standalone.step()
            opt_standalone.zero_grad()

            # Megatron forward-backward-step (same input).
            loss_megatron = model_megatron(x).sum()
            loss_megatron.backward()
            opt_megatron.step()
            opt_megatron.zero_grad()

            # Verify losses are exactly equal.
            torch.testing.assert_close(
                loss_standalone, loss_megatron, atol=0, rtol=0, msg=f"Step {step}: losses differ"
            )

            # Verify all parameters are exactly equal after each step.
            for (n1, p1), (n2, p2) in zip(
                model_standalone.named_parameters(), model_megatron.named_parameters()
            ):
                torch.testing.assert_close(
                    p1.data,
                    p2.data,
                    atol=0,
                    rtol=0,
                    msg=f"Step {step}, param {n1}: parameters differ",
                )

            # Verify optimizer states (exp_avg) are exactly equal.
            for p_s, p_m in zip(model_standalone.parameters(), model_megatron.parameters()):
                state_s = opt_standalone.state[p_s]
                state_m = opt_megatron.state[p_m]
                assert state_s.keys() == state_m.keys(), (
                    f"Step {step}: optimizer state keys differ: "
                    f"{state_s.keys()} vs {state_m.keys()}"
                )
                for key in state_s:
                    torch.testing.assert_close(
                        state_s[key],
                        state_m[key],
                        atol=0,
                        rtol=0,
                        msg=f"Step {step}, state '{key}': optimizer states differ",
                    )


@requires_emerging_optimizers
@pytest.mark.parametrize("optimizer_name", ["lion", "adam"])
@pytest.mark.parametrize("have_te", [False, True])
@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("preallocate", [False, True])
def test_distributed_optimizer_preserves_per_param_step(
    monkeypatch, optimizer_name, have_te, steps, preallocate
):
    """The common checkpoint state preserves EO's integer and torch's tensor steps."""
    from emerging_optimizers.scalar_optimizers import Lion

    from megatron.core.optimizer import distrib_optimizer as distopt_module

    monkeypatch.setattr(distopt_module, "HAVE_APEX_OR_TE", have_te)
    monkeypatch.setattr(distopt_module, "USING_TE_OPTIMIZER", have_te)
    monkeypatch.setattr(distopt_module, "USING_APEX_OPTIMIZER", False)
    # Exercise the cold-load allocation branch using CPU shards.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    optimizer_cls = Lion if optimizer_name == "lion" else torch.optim.AdamW
    params = [nn.Parameter(torch.randn(8, device="cpu")) for _ in range(2)]
    optimizer = optimizer_cls(params, lr=0.01)
    if optimizer_name == "lion":
        _eopt_init_state_fn(optimizer)
        if "step" not in optimizer.state[params[0]]:
            pytest.skip("This Lion release has no per-parameter step")
    else:
        # Initialize torch AdamW's tensor state without retaining an update.
        for param in params:
            param.grad = torch.zeros_like(param)
        optimizer.step()
        for state in optimizer.state.values():
            state["step"].zero_()
    for _ in range(steps):
        for param in params:
            param.grad = torch.ones_like(param)
        optimizer.step()

    def wrap(inner, inner_params):
        wrapper = object.__new__(DistributedOptimizer)
        wrapper.optimizer = inner
        wrapper.config = OptimizerConfig(optimizer=optimizer_name, lr=0.01)
        wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        wrapper.grad_scaler = None
        wrapper.gbuf_ranges = [
            {"buffer": [{"param_map": {p: {"gbuf_world": range(p.numel())} for p in inner_params}}]}
        ]
        wrapper.model_param_group_index_map = {p: (0, i) for i, p in enumerate(inner_params)}
        return wrapper

    checkpoint = copy.deepcopy(wrap(optimizer, params).state_dict())
    assert checkpoint["optimizer"]["common_step"] == steps
    restored_params = [nn.Parameter(p.detach().clone()) for p in params]
    restored = optimizer_cls(restored_params, lr=0.01)
    if preallocate:
        for source, target in zip(params, restored_params):
            restored.state[target] = copy.deepcopy(optimizer.state[source])
            restored.state[target]["step"] = 0 if optimizer_name == "lion" else torch.tensor(0.0)
    wrap(restored, restored_params).load_state_dict(checkpoint)
    for source, target in zip(params, restored_params):
        source_state, target_state = optimizer.state[source], restored.state[target]
        assert type(target_state["step"]) is type(source_state["step"])
        assert target_state["step"] == steps
        # Parameter-dependent state is restored separately from the common state.
        for key in source_state.keys() - {"step"}:
            target_state[key].copy_(source_state[key])
        source.grad = torch.ones_like(source)
        target.grad = source.grad.clone()
    optimizer.step()
    restored.step()
    for source, target in zip(params, restored_params):
        torch.testing.assert_close(target, source, atol=0, rtol=0)
        assert restored.state[target]["step"] == steps + 1


class TestDistributedOptimizerStateKeys:
    """Tests for DistributedOptimizer.optimizer_state_keys and _get_state_key_dtype.

    These tests use a mock to avoid needing a full distributed setup.
    """

    def _make_mock_distopt(self, optimizer_name):
        """Create a minimal mock with just the config needed for optimizer_state_keys."""
        mock = object.__new__(DistributedOptimizer)
        mock.config = OptimizerConfig(optimizer=optimizer_name, lr=1e-4)
        return mock

    def test_adam_state_keys(self):
        distopt = self._make_mock_distopt("adam")
        assert distopt.optimizer_state_keys == ("exp_avg", "exp_avg_sq")

    def test_lion_state_keys(self):
        distopt = self._make_mock_distopt("lion")
        assert distopt.optimizer_state_keys == ("exp_avg",)

    def test_sgd_state_keys_defaults_to_adam(self):
        distopt = self._make_mock_distopt("sgd")
        assert distopt.optimizer_state_keys == ("exp_avg", "exp_avg_sq")

    def test_get_state_key_dtype_known_keys(self):
        distopt = self._make_mock_distopt("adam")
        assert distopt._get_state_key_dtype("exp_avg") == torch.float32
        assert distopt._get_state_key_dtype("exp_avg_sq") == torch.float32

    def test_get_state_key_dtype_unknown_key(self):
        distopt = self._make_mock_distopt("adam")
        assert distopt._get_state_key_dtype("unknown_key") == torch.float32

    def test_get_state_key_dtype_respects_config(self):
        mock = object.__new__(DistributedOptimizer)
        mock.config = SimpleNamespace(exp_avg_dtype=torch.bfloat16, exp_avg_sq_dtype=torch.float16)
        assert mock._get_state_key_dtype("exp_avg") == torch.bfloat16
        assert mock._get_state_key_dtype("exp_avg_sq") == torch.float16

    def test_muon_with_lion_scalar_optimizer(self):
        mock = object.__new__(DistributedOptimizer)
        mock.config = OptimizerConfig(optimizer="muon", lr=1e-4, muon_scalar_optimizer="lion")
        assert mock.optimizer_state_keys == ("exp_avg",)

    def test_muon_with_adam_scalar_optimizer(self):
        mock = object.__new__(DistributedOptimizer)
        mock.config = OptimizerConfig(optimizer="muon", lr=1e-4, muon_scalar_optimizer="adam")
        assert mock.optimizer_state_keys == ("exp_avg", "exp_avg_sq")
