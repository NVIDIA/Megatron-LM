# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.elastification.flextron_config."""

import dataclasses
from argparse import Namespace
from types import SimpleNamespace

from megatron.elastification.flextron_config import FlextronConfig, inject_flextron_config


class TestFlextronConfigDefaults:
    def test_default_values(self):
        cfg = FlextronConfig()
        assert cfg.flextron is False
        assert cfg.enable_router is False
        assert cfg.router_inter_dim == 128
        assert cfg.hard_sample_th == 0.996
        assert cfg.tau_init == 1.0
        assert cfg.tau_decay == 0.9999
        assert cfg.router_std == 0.1
        assert cfg.budget_type == 'param'
        assert cfg.original_model_sample_prob == 0.33

    def test_all_fields_accessible_after_construction(self):
        cfg = FlextronConfig()
        for f in dataclasses.fields(FlextronConfig):
            # Every declared field should be readable.
            getattr(cfg, f.name)


class TestInjectFlextronConfig:
    def test_copies_all_fields_from_args(self):
        args = Namespace(
            flextron=True,
            enable_router=True,
            router_inter_dim=256,
            hard_sample_th=0.5,
            tau_init=2.0,
            tau_decay=0.9,
            router_std=0.01,
            budget_type='mem',
            budget_list=[1.0, 0.5],
            original_model_sample_prob=0.0,
        )
        target = SimpleNamespace()
        inject_flextron_config(args, target)
        assert target.flextron is True
        assert target.enable_router is True
        assert target.router_inter_dim == 256
        assert target.hard_sample_th == 0.5
        assert target.tau_init == 2.0
        assert target.tau_decay == 0.9
        assert target.router_std == 0.01
        assert target.budget_type == 'mem'
        assert target.budget_list == [1.0, 0.5]
        assert target.original_model_sample_prob == 0.0

    def test_missing_arg_falls_back_to_default(self):
        # args has only a subset of FlextronConfig fields.
        args = Namespace(flextron=True)
        target = SimpleNamespace()
        inject_flextron_config(args, target)
        # Present-on-args field is copied.
        assert target.flextron is True
        # Absent-on-args field gets FlextronConfig default.
        assert target.router_inter_dim == 128
        assert target.hard_sample_th == 0.996
        assert target.tau_init == 1.0

    def test_preserves_unrelated_config_attributes(self):
        args = Namespace(flextron=True)
        target = SimpleNamespace(hidden_size=1920, num_layers=52)
        inject_flextron_config(args, target)
        # Fields that are not FlextronConfig fields stay untouched.
        assert target.hidden_size == 1920
        assert target.num_layers == 52

    def test_every_flextron_field_is_set_on_target(self):
        args = Namespace()  # totally empty
        target = SimpleNamespace()
        inject_flextron_config(args, target)
        for f in dataclasses.fields(FlextronConfig):
            assert hasattr(target, f.name), f"field {f.name!r} not injected onto target"

    def test_returns_none(self):
        # inject_flextron_config mutates in place and should not return a value.
        args = Namespace(flextron=True)
        target = SimpleNamespace()
        result = inject_flextron_config(args, target)
        assert result is None
