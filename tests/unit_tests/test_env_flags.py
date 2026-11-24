# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.env_flags import env_flag

NAME = "MCORE_UNIT_TEST_ENV_FLAG"


@pytest.mark.parametrize("default", [False, True])
def test_unset_selects_default(monkeypatch, default):
    monkeypatch.delenv(NAME, raising=False)
    assert env_flag(NAME, default) is default


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on", " on\n"])
@pytest.mark.parametrize("default", [False, True])
def test_true_values_enable(monkeypatch, value, default):
    monkeypatch.setenv(NAME, value)
    assert env_flag(NAME, default) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "Off", "\t0 "])
@pytest.mark.parametrize("default", [False, True])
def test_false_values_disable(monkeypatch, value, default):
    monkeypatch.setenv(NAME, value)
    assert env_flag(NAME, default) is False


@pytest.mark.parametrize("value", ["", " ", "2", "-1", "1.0", "y", "enable", "truthy"])
def test_other_values_raise_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv(NAME, value)
    with pytest.raises(ValueError, match=NAME):
        env_flag(NAME, True)
