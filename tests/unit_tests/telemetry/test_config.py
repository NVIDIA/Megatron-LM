# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for NemoLensConfig and MegatronSpanGroup."""

import os

import pytest

from nemo.lens.config import NemoLensConfig
from nemo.lens.groups import SpanGroup
from nemo.lens.groups_megatron import MegatronSpanGroup


def _megatron_config(**kwargs):
    """Create a NemoLensConfig with Megatron defaults."""
    return NemoLensConfig.from_env(
        prefix='MEGATRON_OTEL', fallback_prefix='NEMO_LENS',
        span_group_cls=MegatronSpanGroup, **kwargs,
    )


class TestConfigDefaults:
    def test_default_enabled_is_false(self):
        cfg = NemoLensConfig()
        assert cfg.enabled is False

    def test_default_export_rank(self):
        cfg = NemoLensConfig()
        assert cfg.export_rank == -1

    def test_default_traces_enabled(self):
        cfg = NemoLensConfig()
        assert cfg.traces_enabled is True

    def test_default_metrics_enabled(self):
        cfg = NemoLensConfig()
        assert cfg.metrics_enabled is True

    def test_default_span_groups(self):
        cfg = NemoLensConfig()
        assert cfg.span_groups == 'default'

    def test_default_exporter(self):
        cfg = NemoLensConfig()
        assert cfg.exporter == 'otlp'

    def test_default_resolved_span_groups_with_megatron(self):
        cfg = NemoLensConfig(_span_group_cls=MegatronSpanGroup)
        groups = cfg.resolved_span_groups
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups
        assert SpanGroup.EVALUATE in groups
        assert MegatronSpanGroup.INFERENCE in groups
        assert SpanGroup.STEP not in groups
        assert MegatronSpanGroup.MICROBATCH not in groups


class TestConfigFromEnv:
    def test_from_env_returns_defaults_with_no_vars(self, monkeypatch):
        for key in (
            'MEGATRON_OTEL_ENABLED',
            'MEGATRON_OTEL_EXPORT_RANK',
            'MEGATRON_OTEL_TRACES_ENABLED',
            'MEGATRON_OTEL_METRICS_ENABLED',
            'MEGATRON_OTEL_SPAN_GROUPS',
            'MEGATRON_OTEL_EXPORTER',
            'OTEL_SERVICE_NAME',
        ):
            monkeypatch.delenv(key, raising=False)

        cfg = _megatron_config()
        assert cfg.enabled is False
        assert cfg.export_rank == -1

    def test_enabled_set_by_env_var(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_ENABLED', '1')
        cfg = _megatron_config()
        assert cfg.enabled is True

    def test_enabled_false_values(self, monkeypatch):
        for val in ('0', 'false', 'no', 'off', 'FALSE'):
            monkeypatch.setenv('MEGATRON_OTEL_ENABLED', val)
            cfg = _megatron_config()
            assert cfg.enabled is False, f"Expected False for {val!r}"

    def test_enabled_true_values(self, monkeypatch):
        for val in ('1', 'true', 'yes', 'on', 'True', 'YES'):
            monkeypatch.setenv('MEGATRON_OTEL_ENABLED', val)
            cfg = _megatron_config()
            assert cfg.enabled is True, f"Expected True for {val!r}"

    def test_export_rank_set_by_env_var(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_EXPORT_RANK', '0')
        cfg = _megatron_config()
        assert cfg.export_rank == 0

    def test_export_rank_last_rank_default(self, monkeypatch):
        monkeypatch.delenv('MEGATRON_OTEL_EXPORT_RANK', raising=False)
        cfg = _megatron_config()
        assert cfg.export_rank == -1

    def test_service_name_from_otel_standard_var(self, monkeypatch):
        monkeypatch.setenv('OTEL_SERVICE_NAME', 'my-training-run')
        cfg = _megatron_config()
        assert cfg.service_name == 'my-training-run'

    def test_exporter_console(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_EXPORTER', 'console')
        cfg = _megatron_config()
        assert cfg.exporter == 'console'

    def test_span_groups_per_step(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_SPAN_GROUPS', 'per_step')
        cfg = _megatron_config()
        assert cfg.span_groups == 'per_step'
        groups = cfg.resolved_span_groups
        assert SpanGroup.STEP in groups
        assert SpanGroup.FORWARD_BACKWARD in groups
        assert SpanGroup.OPTIMIZER in groups
        assert MegatronSpanGroup.INFERENCE in groups
        assert MegatronSpanGroup.MICROBATCH not in groups

    def test_span_groups_all(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_SPAN_GROUPS', 'all')
        cfg = _megatron_config()
        groups = cfg.resolved_span_groups
        assert MegatronSpanGroup.MICROBATCH in groups

    def test_traces_disabled(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_TRACES_ENABLED', '0')
        cfg = _megatron_config()
        assert cfg.traces_enabled is False

    def test_metrics_disabled(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_METRICS_ENABLED', '0')
        cfg = _megatron_config()
        assert cfg.metrics_enabled is False

    def test_invalid_bool_raises(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_ENABLED', 'maybe')
        with pytest.raises(ValueError, match='MEGATRON_OTEL_ENABLED'):
            _megatron_config()

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv('MEGATRON_OTEL_EXPORT_RANK', 'last')
        with pytest.raises(ValueError, match='MEGATRON_OTEL_EXPORT_RANK'):
            _megatron_config()


class TestMegatronSpanGroupResolve:
    def test_default_preset(self):
        groups = MegatronSpanGroup.resolve('default')
        assert groups == frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE,
            MegatronSpanGroup.INFERENCE,
        ])

    def test_per_step_preset(self):
        groups = MegatronSpanGroup.resolve('per_step')
        assert SpanGroup.STEP in groups
        assert SpanGroup.FORWARD_BACKWARD in groups
        assert SpanGroup.OPTIMIZER in groups
        assert MegatronSpanGroup.INFERENCE in groups
        assert MegatronSpanGroup.MICROBATCH not in groups

    def test_all_preset(self):
        groups = MegatronSpanGroup.resolve('all')
        assert groups == MegatronSpanGroup.ALL_GROUPS
        assert MegatronSpanGroup.MICROBATCH in groups
        assert MegatronSpanGroup.INFERENCE in groups

    def test_individual_group_name(self):
        groups = MegatronSpanGroup.resolve('microbatch')
        assert groups == frozenset([MegatronSpanGroup.MICROBATCH])

    def test_comma_separated_groups(self):
        groups = MegatronSpanGroup.resolve('job,checkpoint')
        assert groups == frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT])

    def test_mix_preset_and_individual(self):
        groups = MegatronSpanGroup.resolve('default,microbatch')
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups
        assert SpanGroup.EVALUATE in groups
        assert MegatronSpanGroup.MICROBATCH in groups

    def test_case_insensitive(self):
        assert MegatronSpanGroup.resolve('DEFAULT') == MegatronSpanGroup.resolve('default')
        assert MegatronSpanGroup.resolve('PER_STEP') == MegatronSpanGroup.resolve('per_step')

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match='Unknown span group'):
            MegatronSpanGroup.resolve('unknown_group')

    def test_whitespace_tolerant(self):
        groups = MegatronSpanGroup.resolve('job , checkpoint')
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups

    def test_all_individual_groups_valid(self):
        for group in MegatronSpanGroup.ALL_GROUPS:
            result = MegatronSpanGroup.resolve(group)
            assert group in result
