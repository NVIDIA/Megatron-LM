# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""``MCORE_PADDING_AWARE_SKIP``: unset = the measured-net-negative default set, empty = everything on."""

import pytest

from megatron.core.fusions import padding_aware_switches as sw


def test_unset_disables_only_the_default_set(monkeypatch):
    monkeypatch.delenv("MCORE_PADDING_AWARE_SKIP", raising=False)
    assert sw.DEFAULT_SKIP == frozenset({"norm", "add", "argmax"})
    for name in sw.DEFAULT_SKIP:
        assert sw.padding_aware_feature_disabled(name)
    for name in ("ce", "mhc", "rope", "arena"):
        assert not sw.padding_aware_feature_disabled(name)


def test_empty_value_engages_everything(monkeypatch):
    monkeypatch.setenv("MCORE_PADDING_AWARE_SKIP", "")
    assert not any(sw.padding_aware_feature_disabled(n) for n in ("norm", "add", "argmax", "ce"))


@pytest.mark.parametrize("value", ["ce", "ce,mhc", "mhc,norm"])
def test_explicit_list_is_taken_literally(monkeypatch, value):
    monkeypatch.setenv("MCORE_PADDING_AWARE_SKIP", value)
    listed = set(value.split(","))
    for name in ("ce", "mhc", "norm", "add", "argmax", "rope"):
        assert sw.padding_aware_feature_disabled(name) == (name in listed)
