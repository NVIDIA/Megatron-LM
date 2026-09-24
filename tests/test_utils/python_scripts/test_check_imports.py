# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check that only explicitly optional import failures are accepted by CI."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def checker():
    """Load the CI script without invoking its command-line entry point."""
    path = Path(__file__).resolve().parents[3] / '.gitlab/scripts/check_imports.py'
    spec = importlib.util.spec_from_file_location('check_imports', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImportChecker()


@pytest.mark.parametrize(
    'exception_module',
    ['megatron.core.exceptions', 'megatron.core.distributed.fsdp.src.megatron_fsdp.exceptions'],
)
def test_optional_dependency_failure_is_graceful(checker, tmp_path, monkeypatch, exception_module):
    """Both packages' explicit exceptions work even after checking their defining modules."""
    assert checker.import_module(exception_module) == ('success', '')
    (tmp_path / 'optional_backend.py').write_text(
        f'from {exception_module} import OptionalDependencyUnavailable\n'
        'raise OptionalDependencyUnavailable("optional backend missing")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert checker.import_module('optional_backend') == ('graceful', 'optional backend missing')


@pytest.mark.parametrize(
    'source',
    [
        'raise ImportError("UnavailableError: this is still a real failure")',
        'raise ModuleNotFoundError("required dependency missing")',
        'raise RuntimeError("UnavailableError in an unrelated error")',
        'from megatron.core.exceptions import OptionalDependencyUnavailable\n'
        'try:\n'
        '    raise OptionalDependencyUnavailable("optional backend missing")\n'
        'except ImportError as exc:\n'
        '    raise RuntimeError("broken fallback") from exc',
    ],
)
def test_unexpected_import_failure_is_not_suppressed(checker, tmp_path, monkeypatch, source):
    """Messages and chained optional errors must not hide an unexpected outer exception."""
    (tmp_path / 'broken_backend.py').write_text(source + '\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    status, details = checker.import_module('broken_backend')
    assert status == 'failed'
    assert 'Traceback' in details


def test_successful_import(checker, tmp_path, monkeypatch):
    """A module without missing dependencies remains successful."""
    (tmp_path / 'working_backend.py').write_text('VALUE = 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert checker.import_module('working_backend') == ('success', '')
