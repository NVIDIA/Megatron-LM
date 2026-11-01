# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional-backend contracts without requiring any external kernel libraries."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import Version

from megatron.core.ops import _backends


@pytest.fixture
def backend(monkeypatch):
    """Provide an isolated package with a version and ordinary and nested exports."""
    module = ModuleType('_ops_backend_fixture')
    module.__spec__ = ModuleSpec(module.__name__, loader=None, is_package=True)
    module.__path__ = []
    module.__version__ = '1.2.0'
    module.kernel = object()
    module.nested = SimpleNamespace(kernel=object())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_is_available_does_not_import_package(tmp_path, monkeypatch):
    """Probing a top-level package must not load its optional dependencies."""
    name = '_ops_available_fixture'
    (tmp_path / f'{name}.py').write_text('raise RuntimeError("must not import")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _backends.is_available(name)
    assert name not in sys.modules


def test_is_available_returns_false_for_missing_module(backend):
    assert not _backends.is_available(f'{backend.__name__}.missing')


def test_is_available_returns_false_for_missing_parent(monkeypatch):
    monkeypatch.setitem(sys.modules, '_ops_missing_parent', None)
    assert not _backends.is_available('_ops_missing_parent.child')


def test_is_available_returns_false_without_spec(backend):
    backend.__spec__ = None
    assert not _backends.is_available(backend.__name__)


def test_installed_version_prefers_package_version(backend, monkeypatch):
    def unexpected_metadata_lookup(dist):
        pytest.fail(f'Unexpected metadata lookup for {dist}')

    monkeypatch.setattr(_backends.metadata, 'version', unexpected_metadata_lookup)
    assert _backends.installed_version(f'{backend.__name__}.ops') == Version('1.2.0')


@pytest.mark.parametrize('dist', [None, 'kernel-distribution'])
def test_installed_version_falls_back_to_distribution_metadata(backend, monkeypatch, dist):
    del backend.__version__

    def distribution_version(name):
        assert name == (dist or backend.__name__)
        return '2.0.0rc1'

    monkeypatch.setattr(_backends.metadata, 'version', distribution_version)
    assert _backends.installed_version(f'{backend.__name__}.ops', dist) == Version('2.0.0rc1')


def test_installed_version_logs_import_failure(monkeypatch, caplog):
    name = '_ops_missing_backend'
    monkeypatch.setitem(sys.modules, name, None)
    assert _backends.installed_version(name) is None
    assert name in caplog.text
    assert 'failed to import' in caplog.text


def test_installed_version_logs_missing_metadata(backend, monkeypatch, caplog):
    del backend.__version__

    def missing_metadata(name):
        raise _backends.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_backends.metadata, 'version', missing_metadata)
    assert _backends.installed_version(backend.__name__, 'kernel-distribution') is None
    assert backend.__name__ in caplog.text
    assert 'kernel-distribution' in caplog.text
    assert 'no distribution metadata' in caplog.text


def test_installed_version_logs_invalid_version(backend, caplog):
    backend.__version__ = 'not-a-version'
    assert _backends.installed_version(backend.__name__) is None
    assert backend.__name__ in caplog.text
    assert 'not-a-version' in caplog.text
    assert 'not PEP 440' in caplog.text


@pytest.mark.parametrize(
    'installed,minimum,expected',
    [
        ('1.2.0', '1.2', True),
        ('1.10', '1.2', True),
        ('1.2rc1', '1.2', False),
        ('1.1', '1.2', False),
        ('unknown', '1.2', False),
    ],
)
def test_has_min_version(backend, installed, minimum, expected):
    backend.__version__ = installed
    assert _backends.has_min_version(backend.__name__, minimum) is expected


def test_has_min_version_uses_distribution_override(backend, monkeypatch):
    del backend.__version__

    def distribution_version(name):
        assert name == 'kernel-distribution'
        return '1.2.0'

    monkeypatch.setattr(_backends.metadata, 'version', distribution_version)
    assert _backends.has_min_version(backend.__name__, '1.2', dist='kernel-distribution')


@pytest.mark.parametrize('symbols', [(), ('kernel',), ('kernel', 'nested.kernel')])
def test_require_returns_canonical_module(backend, symbols):
    assert _backends.require(backend.__name__, *symbols, needed_by='Test operation') is backend


@pytest.mark.parametrize('error_type', [ModuleNotFoundError, ImportError, OSError, RuntimeError])
def test_require_chains_import_failure(monkeypatch, error_type):
    cause = error_type('kernel load failed')
    if isinstance(cause, ModuleNotFoundError):
        cause.name = 'transitive_dependency'

    def failed_import(name):
        assert name == '_ops_broken_backend'
        raise cause

    monkeypatch.setattr(_backends, 'import_module', failed_import)
    with pytest.raises(ImportError, match='Test operation requires _ops_broken_backend') as caught:
        _backends.require('_ops_broken_backend', needed_by='Test operation')
    assert caught.value.__cause__ is cause
    assert 'kernel load failed' in str(caught.value)
    if isinstance(cause, ModuleNotFoundError):
        assert isinstance(caught.value, ModuleNotFoundError)
        assert caught.value.name == 'transitive_dependency'
    if isinstance(cause, (OSError, RuntimeError)):
        assert 'installed but failed to load' in str(caught.value)


@pytest.mark.parametrize('symbol', ['missing', 'nested.missing'])
def test_require_rejects_missing_symbol(backend, symbol):
    with pytest.raises(ImportError, match='which is missing') as caught:
        _backends.require(backend.__name__, symbol, needed_by='Test operation')
    assert isinstance(caught.value.__cause__, AttributeError)
    assert f'{backend.__name__}.{symbol}' in str(caught.value)
    assert 'Test operation' in str(caught.value)


def test_require_rejects_unavailable_symbol(backend):
    backend.kernel = None
    with pytest.raises(ImportError, match='unavailable in this installation') as caught:
        _backends.require(backend.__name__, 'kernel', needed_by='Test operation')
    assert f'{backend.__name__}.kernel' in str(caught.value)
    assert 'Test operation' in str(caught.value)


@pytest.mark.parametrize('error_type', [ImportError, OSError, RuntimeError])
def test_require_chains_lazy_symbol_failure(backend, error_type):
    cause = error_type('lazy extension load failed')

    def failed_getattr(name):
        assert name == 'lazy_kernel'
        raise cause

    backend.__getattr__ = failed_getattr
    with pytest.raises(ImportError, match='which failed to load') as caught:
        _backends.require(backend.__name__, 'lazy_kernel', needed_by='Test operation')
    assert caught.value.__cause__ is cause
    assert f'{backend.__name__}.lazy_kernel' in str(caught.value)
    assert 'Test operation' in str(caught.value)


@pytest.mark.parametrize('installed', ['1.2.0', '1.10'])
def test_require_accepts_minimum_version(backend, installed):
    backend.__version__ = installed
    assert (
        _backends.require(backend.__name__, 'kernel', min_version='1.2', needed_by='Test operation')
        is backend
    )


@pytest.mark.parametrize(
    'installed,reason', [('1.2rc1', 'found 1.2rc1'), ('unknown', 'version cannot be determined')]
)
def test_require_rejects_unmet_version(backend, installed, reason):
    backend.__version__ = installed
    with pytest.raises(ImportError, match=reason) as caught:
        _backends.require(backend.__name__, min_version='1.2', needed_by='Test operation')
    assert f'Test operation requires {backend.__name__}>=1.2' in str(caught.value)


def test_require_uses_distribution_override(backend, monkeypatch):
    del backend.__version__

    def distribution_version(name):
        assert name == 'kernel-distribution'
        return '1.2.0'

    monkeypatch.setattr(_backends.metadata, 'version', distribution_version)
    assert (
        _backends.require(
            backend.__name__,
            min_version='1.2',
            dist='kernel-distribution',
            needed_by='Test operation',
        )
        is backend
    )
