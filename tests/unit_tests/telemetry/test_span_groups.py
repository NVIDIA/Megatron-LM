# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for span group state and control."""

import pytest

from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups
from nemo.lens.groups import SpanGroup
from nemo.lens.groups_megatron import MegatronSpanGroup


class TestSpanGroupState:
    def setup_method(self):
        set_enabled_span_groups(frozenset())

    def teardown_method(self):
        set_enabled_span_groups(frozenset())

    def test_all_disabled_by_default(self):
        set_enabled_span_groups(frozenset())
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group), f"Expected {group} to be disabled"

    def test_set_and_check_single_group(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False

    def test_set_multiple_groups(self):
        groups = frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE])
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.EVALUATE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False
        assert is_span_group_enabled(MegatronSpanGroup.MICROBATCH) is False

    def test_set_overrides_previous(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled(SpanGroup.JOB) is True

        set_enabled_span_groups(frozenset([SpanGroup.STEP]))
        assert is_span_group_enabled(SpanGroup.JOB) is False
        assert is_span_group_enabled(SpanGroup.STEP) is True

    def test_unknown_group_returns_false(self):
        set_enabled_span_groups(frozenset([SpanGroup.JOB]))
        assert is_span_group_enabled("nonexistent_group") is False

    def test_default_preset_groups(self):
        groups = MegatronSpanGroup.resolve('default')
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.EVALUATE) is True
        assert is_span_group_enabled(MegatronSpanGroup.INFERENCE) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is False
        assert is_span_group_enabled(MegatronSpanGroup.MICROBATCH) is False

    def test_per_step_preset_groups(self):
        groups = MegatronSpanGroup.resolve('per_step')
        set_enabled_span_groups(groups)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.STEP) is True
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is True
        assert is_span_group_enabled(SpanGroup.OPTIMIZER) is True
        assert is_span_group_enabled(SpanGroup.MODEL_INIT) is True
        assert is_span_group_enabled(SpanGroup.LOAD_CHECKPOINT) is True
        assert is_span_group_enabled(MegatronSpanGroup.INFERENCE) is True
        assert is_span_group_enabled(MegatronSpanGroup.MICROBATCH) is False

    def test_all_preset_groups(self):
        groups = MegatronSpanGroup.resolve('all')
        set_enabled_span_groups(groups)
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is True, f"Expected {group} to be enabled"

    def test_empty_frozenset_disables_all(self):
        set_enabled_span_groups(MegatronSpanGroup.resolve('all'))
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is True
        set_enabled_span_groups(frozenset())
        for group in MegatronSpanGroup.ALL_GROUPS:
            assert is_span_group_enabled(group) is False


class TestSpanGroupPublicAPI:
    def test_importable_from_nemo_lens(self):
        from nemo.lens import SpanGroup as _SG, is_span_group_enabled as _ise
        assert _SG.JOB == 'job'
        assert callable(_ise)

    def test_set_enabled_span_groups_importable(self):
        from nemo.lens import set_enabled_span_groups as _seg
        assert callable(_seg)
