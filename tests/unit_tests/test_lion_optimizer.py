# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for Lion optimizer plumbing between Megatron and emerging_optimizers.

Tests that require emerging_optimizers are skipped when the package is not installed.
Tests that verify Megatron-only behavior (config, error handling) always run.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import megatron.core.optimizer as opt_module
from megatron.core.optimizer import (
    HAVE_EMERGING_OPTIMIZERS,
    OptimizerConfig,
    _get_megatron_optimizer_based_on_param_groups,
    _get_param_groups,
)
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    _default_adam_based_eopt_config_to_kwargs,
    _muon_default_param_overrides,
)
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

    def test_muon_scalar_optimizer_controls_nonlinear_param_override(self):
        """Muon scalar optimizer selection should flow into default nonlinear param overrides."""
        entry = _EMERGING_OPTIMIZERS["muon"]
        config = OptimizerConfig(muon_scalar_optimizer="lion")

        overrides = entry.config_to_param_overrides(config)
        assert len(overrides) == 1
        (_, override), = overrides.items()
        assert override["optimizer"] == "lion"

    def test_muon_scalar_optimizer_routes_lion_groups_to_lion_entry(self):
        """Muon scalar-optimizer overrides must create a real Lion bucket via param overrides."""
        model = SimpleModel()
        config = OptimizerConfig(
            optimizer="muon",
            lr=1e-4,
            muon_scalar_optimizer="lion",
            adam_beta1=0.81,
            adam_beta2=0.88,
            lion_beta1=0.91,
            lion_beta2=0.97,
        )
        recorded = []

        def fake_create(_config, _groups, eopt_name, _model_chunks, _pg_collection):
            if eopt_name == "lion":
                kwargs = _default_adam_based_eopt_config_to_kwargs(
                    eopt_name, _config, _model_chunks, _pg_collection
                )
                recorded.append((eopt_name, kwargs["betas"]))
            else:
                recorded.append((eopt_name, None))
            return SimpleNamespace(param_groups=[]), (lambda *_args, **_kwargs: None)

        fake_pg_collection = SimpleNamespace(mp=None, tp=None, tp_ep_pp=None)
        fake_muon_entry = SimpleNamespace(
            config_to_param_overrides=_muon_default_param_overrides,
            default_param_overrides={},
            optimizer_cls=object,
            init_state_fn=lambda *_args, **_kwargs: None,
            config_to_kwargs=None,
        )
        fake_lion_entry = SimpleNamespace(
            config_to_param_overrides=None,
            default_param_overrides={},
            optimizer_cls=object,
            init_state_fn=lambda *_args, **_kwargs: None,
            config_to_kwargs=None,
        )

        with patch("torch.distributed.get_world_size", return_value=1), patch(
            "torch.distributed.all_gather_object",
            lambda output_list, obj: output_list.__setitem__(0, obj),
        ), patch.object(
            opt_module, "HAVE_EMERGING_OPTIMIZERS", True
        ), patch.dict(
            opt_module._EMERGING_OPTIMIZERS,
            {"muon": fake_muon_entry, "lion": fake_lion_entry},
            clear=False,
        ), patch.object(
            opt_module, "_create_emerging_optimizer", side_effect=fake_create
        ), patch.object(
            opt_module, "FP32Optimizer", side_effect=lambda optimizer, *_args, **_kwargs: optimizer
        ), patch.object(
            opt_module, "ChainedOptimizer", side_effect=lambda optimizers: optimizers
        ):
            results = opt_module._get_megatron_emerging_optimizer(
                config=config,
                model_chunks=[model],
                config_overrides={},
                pg_collection=fake_pg_collection,
            )

        assert set(recorded) == {("muon", None), ("lion", (0.91, 0.97))}
        assert len(results) == 2

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch(
        "torch.distributed.all_gather_object",
        lambda output_list, obj: output_list.__setitem__(0, obj),
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

    def test_default_emerging_lion_kwargs_use_lion_betas(self):
        """Shared emerging-optimizer kwargs must keep Lion on lion_beta{1,2}."""
        config = OptimizerConfig(
            optimizer="muon",
            lr=1e-4,
            adam_beta1=0.81,
            adam_beta2=0.88,
            lion_beta1=0.91,
            lion_beta2=0.97,
        )

        kwargs = _default_adam_based_eopt_config_to_kwargs("lion", config, [], None)

        assert kwargs["betas"] == (0.91, 0.97)

    def test_lion_init_state_fn_creates_exp_avg(self):
        """init_state_fn should pre-initialize exp_avg state for all params."""
        model = SimpleModel()
        config = OptimizerConfig(optimizer="lion", lr=1e-4)

        optimizer = _create_lion_optimizer(model, config)
        inner_opt = optimizer.optimizer

        # State should be empty before init.
        for p in model.parameters():
            assert len(inner_opt.state[p]) == 0

        # Call init_state_fn (stored on the FP32Optimizer wrapper).
        optimizer.init_state_fn(inner_opt)

        # State should now have exp_avg for each param, exactly zero.
        for p in model.parameters():
            assert "exp_avg" in inner_opt.state[p]
            assert inner_opt.state[p]["exp_avg"].shape == p.shape
            assert inner_opt.state[p]["exp_avg"].dtype == p.dtype
            torch.testing.assert_close(
                inner_opt.state[p]["exp_avg"], torch.zeros_like(p.data), atol=0, rtol=0
            )

    @pytest.mark.parametrize(
        "lr,beta1,beta2,weight_decay",
        [(1e-3, 0.95, 0.98, 0.0), (3e-4, 0.9, 0.99, 0.01), (1e-4, 0.85, 0.95, 0.1)],
    )
    def test_megatron_lion_exact_match_with_standalone(self, lr, beta1, beta2, weight_decay):
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
