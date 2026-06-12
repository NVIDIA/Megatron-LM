import builtins
import os
from pathlib import Path

import pytest

from megatron.core.msc_utils import maybe_msc


def test_open_is_builtin_open():
    assert maybe_msc.open is builtins.open


def test_os_is_os_module():
    assert maybe_msc.os is os


def test_Path_is_pathlib_Path():
    assert maybe_msc.Path is Path


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(maybe_msc, "this_attribute_does_not_exist_12345")


def test_path_isdir_delegates_to_os_path_isdir(monkeypatch):
    called = {}

    def fake_isdir(p):
        called['p'] = p
        return True

    # monkeypatch the os.path.isdir used by the fallback path
    monkeypatch.setattr(os.path, 'isdir', fake_isdir)

    result = maybe_msc.path_isdir('/tmp/some-path')
    assert result is True
    assert called['p'] == '/tmp/some-path'
