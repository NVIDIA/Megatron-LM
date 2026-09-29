# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.telemetry.span_groups``.

These run whether or not ``nemo-lens`` is installed. Assertions that depend on
which ``SpanGroup`` base class was picked up are gated on ``HAVE_NEMO_LENS``.
"""

import pytest

from megatron.core.telemetry.span_groups import MegatronSpanGroup, SpanGroup

try:
    import nemo.lens  # noqa: F401

    HAVE_NEMO_LENS = True
except ImportError:
    HAVE_NEMO_LENS = False

# The groups MegatronSpanGroup adds on top of the shared lens groups.
MEGATRON_ONLY_GROUPS = frozenset(
    [
        MegatronSpanGroup.MICROBATCH,
        MegatronSpanGroup.LAYER,
        MegatronSpanGroup.COMMUNICATION,
        MegatronSpanGroup.ACTIVATION_OFFLOAD,
        MegatronSpanGroup.DATA_LOADING,
        MegatronSpanGroup.FIRST_ITERATION,
        MegatronSpanGroup.TRACE_REGION,
        MegatronSpanGroup.INFERENCE,
    ]
)


class TestGroupConstants:
    """The group names are part of the user-facing --otel-span-groups spec."""

    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("MICROBATCH", "microbatch"),
            ("LAYER", "layer"),
            ("COMMUNICATION", "communication"),
            ("ACTIVATION_OFFLOAD", "activation_offload"),
            ("DATA_LOADING", "data_loading"),
            ("FIRST_ITERATION", "first_iteration"),
            ("TRACE_REGION", "trace_region"),
            ("INFERENCE", "inference"),
        ],
    )
    def test_group_name(self, attribute, expected):
        assert getattr(MegatronSpanGroup, attribute) == expected

    def test_group_names_are_unique(self):
        assert len(MEGATRON_ONLY_GROUPS) == 8

    def test_megatron_groups_do_not_collide_with_base_groups(self):
        assert not (MEGATRON_ONLY_GROUPS & SpanGroup.ALL_GROUPS)

    def test_inherits_base_groups(self):
        assert MegatronSpanGroup.JOB == SpanGroup.JOB
        assert MegatronSpanGroup.CHECKPOINT == SpanGroup.CHECKPOINT
        assert MegatronSpanGroup.STEP == SpanGroup.STEP


class TestAllGroups:
    def test_is_a_frozenset(self):
        assert isinstance(MegatronSpanGroup.ALL_GROUPS, frozenset)

    def test_extends_base_all_groups(self):
        assert SpanGroup.ALL_GROUPS <= MegatronSpanGroup.ALL_GROUPS

    def test_contains_every_megatron_group(self):
        assert MEGATRON_ONLY_GROUPS <= MegatronSpanGroup.ALL_GROUPS

    def test_is_exactly_base_plus_megatron(self):
        assert MegatronSpanGroup.ALL_GROUPS == SpanGroup.ALL_GROUPS | MEGATRON_ONLY_GROUPS


class TestPresets:
    """Presets must nest: default subset of per_step subset of all."""

    def test_expected_presets_exist(self):
        assert set(MegatronSpanGroup._PRESETS) == {"default", "per_step", "profiling", "all"}

    def test_every_preset_is_a_subset_of_all_groups(self):
        for name, groups in MegatronSpanGroup._PRESETS.items():
            assert groups <= MegatronSpanGroup.ALL_GROUPS, f"preset {name!r} has unknown groups"

    def test_default_is_a_subset_of_per_step(self):
        assert MegatronSpanGroup._PRESETS["default"] <= MegatronSpanGroup._PRESETS["per_step"]

    def test_all_and_profiling_are_every_group(self):
        assert MegatronSpanGroup._PRESETS["all"] == MegatronSpanGroup.ALL_GROUPS
        assert MegatronSpanGroup._PRESETS["profiling"] == MegatronSpanGroup.ALL_GROUPS

    def test_default_is_coarse_grained(self):
        """`default` is the production preset; it must not carry per-step cost."""
        default = MegatronSpanGroup._PRESETS["default"]
        assert MegatronSpanGroup.JOB in default
        assert MegatronSpanGroup.CHECKPOINT in default
        assert MegatronSpanGroup.EVALUATE in default
        assert MegatronSpanGroup.FIRST_ITERATION in default
        assert MegatronSpanGroup.INFERENCE in default
        assert MegatronSpanGroup.STEP not in default
        assert MegatronSpanGroup.FORWARD_BACKWARD not in default
        assert MegatronSpanGroup.MICROBATCH not in default

    def test_per_step_adds_step_level_groups(self):
        per_step = MegatronSpanGroup._PRESETS["per_step"]
        assert MegatronSpanGroup.STEP in per_step
        assert MegatronSpanGroup.FORWARD_BACKWARD in per_step
        assert MegatronSpanGroup.OPTIMIZER in per_step
        assert MegatronSpanGroup.MODEL_INIT in per_step
        assert MegatronSpanGroup.LOAD_CHECKPOINT in per_step
        assert MegatronSpanGroup.COMMUNICATION in per_step
        assert MegatronSpanGroup.DATA_LOADING in per_step

    @pytest.mark.parametrize(
        "group",
        [
            MegatronSpanGroup.MICROBATCH,
            MegatronSpanGroup.LAYER,
            MegatronSpanGroup.ACTIVATION_OFFLOAD,
            MegatronSpanGroup.TRACE_REGION,
        ],
    )
    def test_verbose_groups_are_opt_in_only(self, group):
        """These emit per-layer or per-marker spans; `all` only."""
        assert group not in MegatronSpanGroup._PRESETS["per_step"]
        assert group in MegatronSpanGroup._PRESETS["all"]

    def test_presets_do_not_shadow_the_base_class(self):
        """Subclassing must not leave MegatronSpanGroup resolving base presets."""
        assert MegatronSpanGroup._PRESETS is not SpanGroup._PRESETS


class TestResolve:
    @pytest.mark.skipif(HAVE_NEMO_LENS, reason="stub SpanGroup is only used without nemo-lens")
    def test_resolve_without_nemo_lens_raises(self):
        with pytest.raises(RuntimeError, match="nemo-lens"):
            MegatronSpanGroup.resolve("default")

    @pytest.mark.skipif(not HAVE_NEMO_LENS, reason="requires nemo-lens")
    def test_resolve_with_nemo_lens_returns_the_preset(self):
        assert MegatronSpanGroup.resolve("default") == MegatronSpanGroup._PRESETS["default"]
        assert MegatronSpanGroup.resolve("all") == MegatronSpanGroup.ALL_GROUPS


@pytest.mark.skipif(HAVE_NEMO_LENS, reason="stub SpanGroup is only used without nemo-lens")
class TestStubFallback:
    """Without nemo-lens the local stub must still expose the shared groups."""

    def test_stub_defines_the_shared_groups(self):
        assert SpanGroup.ALL_GROUPS == frozenset(
            [
                "job",
                "checkpoint",
                "evaluate",
                "model_init",
                "load_checkpoint",
                "step",
                "forward_backward",
                "optimizer",
            ]
        )
