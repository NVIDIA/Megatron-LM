# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for TelemetryHandle and setup_telemetry."""

import pytest

from nemo.lens.config import NemoLensConfig
from nemo.lens.groups_megatron import MegatronSpanGroup
from nemo.lens.handle import TelemetryHandle, setup_telemetry
from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups
from nemo.lens.groups import SpanGroup


class TestSetupTelemetryDisabled:
    def test_returns_telemetry_handle(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert isinstance(handle, TelemetryHandle)

    def test_tracer_is_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        tracer = handle.tracer
        assert tracer is not None

    def test_meter_is_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        meter = handle.meter
        assert meter is not None

    def test_no_op_tracer_creates_span_without_error(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        with handle.tracer.start_as_current_span("test.span") as span:
            assert span is not None

    def test_shutdown_completes_without_error(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        handle.shutdown(timeout_ms=100)


class TestSetupTelemetryNonExportRank:
    def test_non_export_rank_gets_noop_tracer(self):
        cfg = NemoLensConfig(enabled=True, export_rank=-1)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert isinstance(handle, TelemetryHandle)
        with handle.tracer.start_as_current_span("test.span") as span:
            assert span is not None

    def test_export_rank_zero_config(self):
        cfg = NemoLensConfig(enabled=True, export_rank=0, exporter='console')
        handle = setup_telemetry(cfg, rank=1, world_size=4)
        assert isinstance(handle, TelemetryHandle)

    def test_last_rank_default_resolution(self):
        cfg = NemoLensConfig(enabled=True, export_rank=-1, exporter='console')
        handle = setup_telemetry(cfg, rank=3, world_size=4)
        assert isinstance(handle, TelemetryHandle)


class TestTelemetryHandleShutdown:
    def test_shutdown_idempotent(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        handle.shutdown(timeout_ms=100)
        handle.shutdown(timeout_ms=100)


class TestSetupTelemetrySpanGroups:
    def setup_method(self):
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        set_enabled_span_groups(frozenset())

    def test_disabled_clears_all_groups(self):
        cfg = NemoLensConfig(enabled=False, span_groups='all', _span_group_cls=MegatronSpanGroup)
        setup_telemetry(cfg, rank=0, world_size=1)
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group)

    def test_enabled_registers_default_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups='default', exporter='console', _span_group_cls=MegatronSpanGroup)
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(MegatronSpanGroup.INFERENCE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False

    def test_enabled_registers_per_step_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups='per_step', exporter='console', _span_group_cls=MegatronSpanGroup)
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.STEP) is True
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is True
        assert is_span_group_enabled(MegatronSpanGroup.MICROBATCH) is False

    def test_non_export_rank_clears_span_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups='all', exporter='console', _span_group_cls=MegatronSpanGroup)
        setup_telemetry(cfg, rank=0, world_size=4)
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group), f"group {group!r} should be disabled"


class TestTelemetryHandleIsExporting:
    def setup_method(self):
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        set_enabled_span_groups(frozenset())

    def test_disabled_is_not_exporting(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert handle.is_exporting is False

    def test_export_rank_is_exporting(self):
        cfg = NemoLensConfig(enabled=True, export_rank=0, exporter='console')
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is True

    def test_non_export_rank_is_not_exporting(self):
        cfg = NemoLensConfig(enabled=True, export_rank=-1, exporter='console')
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is False
