# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for moved modules and their deprecated import paths."""

import ast
import importlib
import importlib.util
import pickle
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

from megatron.core.ops import _compat
from tests.unit_tests.ops.deprecated_paths import FORWARDED, PACKAGE_MARKERS

REPO_ROOT = Path(__file__).resolve().parents[3]


def _module_file(name):
    path = REPO_ROOT.joinpath(*name.split('.'))
    return path.with_suffix('.py') if path.with_suffix('.py').is_file() else path / '__init__.py'


@pytest.mark.parametrize('old,targets', FORWARDED.items())
def test_forwarder_targets_exist(old, targets):
    """Every compatibility path must resolve to an implementation in this PR."""
    assert _module_file(old).is_file()
    for target in targets:
        assert _module_file(target).is_file(), target


@pytest.mark.parametrize('old,targets', FORWARDED.items())
def test_forwarder_import_is_lazy(old, targets, monkeypatch):
    """Importing a deprecated path or probing dunders must not load its targets."""

    def unexpected_import(name):
        pytest.fail(f'{old} eagerly imported {name}')

    monkeypatch.setattr(_compat, 'import_module', unexpected_import)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        module = importlib.import_module(old)
        # Other tests may have already imported this path. Reload exercises its body again.
        if not any(old in str(item.message) for item in caught):
            module = importlib.reload(module)
    assert any(old in str(item.message) and targets[0] in str(item.message) for item in caught)
    assert not hasattr(module, '__wrapped__')


@pytest.mark.parametrize('old,targets', FORWARDED.items())
def test_forwarder_preserves_exports(old, targets, monkeypatch):
    """Wildcard imports and private names resolve to the canonical objects."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        module = importlib.import_module(old)
    implementation = ModuleType(targets[0])
    implementation.public = object()
    implementation._private = object()
    implementation.__all__ = ['public']
    for target in targets:
        monkeypatch.setitem(sys.modules, target, implementation)
    assert module.public is implementation.public
    assert module._private is implementation._private
    assert 'public' in dir(module)
    namespace = {}
    exec(f'from {old} import *', namespace)
    assert namespace['public'] is implementation.public
    assert '_private' not in namespace
    with pytest.raises(AttributeError):
        getattr(module, 'missing_export')


@pytest.mark.skipif("megatron.core.ssm.ops.mamba2" not in FORWARDED, reason="moved in a later PR of this stack")
def test_package_child_import_stays_lazy(monkeypatch):
    """Importing a deprecated child must not probe optional kernels in its parent."""

    def unexpected_import(name):
        assert not name.startswith('megatron.core.ops.ssm.mamba2'), name
        return importlib.import_module(name)

    monkeypatch.setattr(_compat, 'import_module', unexpected_import)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        package = importlib.import_module('megatron.core.ssm.ops.mamba2')
        monkeypatch.delattr(package, 'ssd_combined', raising=False)
        child = package.ssd_combined
    assert child.__name__ == 'megatron.core.ssm.ops.mamba2.ssd_combined'


@pytest.mark.skipif("megatron.core.ssm.ops" not in FORWARDED, reason="moved in a later PR of this stack")
def test_old_ssm_entry_points_preserve_missing_dependency_behavior(monkeypatch):
    """The historical optional entry points still return None when kernels are absent."""
    from megatron.core.ops import ssm

    def unavailable(*args):
        raise ImportError('optional kernel unavailable')

    monkeypatch.setattr(ssm, 'import_module', unavailable)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        old = importlib.import_module('megatron.core.ssm.ops')
    assert old.__all__ == ['mamba_chunk_scan_combined_varlen', 'causal_conv1d_varlen_fn']
    assert old.mamba_chunk_scan_combined_varlen is None
    assert old.causal_conv1d_varlen_fn is None


@pytest.mark.skipif("megatron.core.ssm.gated_delta_net" not in FORWARDED, reason="moved in a later PR of this stack")
def test_gdn_package_exports_canonical_classes():
    """The historical GDN package exports both variants and the common submodule spec."""
    from megatron.core.ops.ssm.gated_delta.common import GatedDeltaNetSubmodules
    from megatron.core.ops.ssm.gated_delta.gdn import GatedDeltaNet
    from megatron.core.ops.ssm.gated_delta.gdn2 import GatedDeltaNet2

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        old = importlib.import_module('megatron.core.ssm.gated_delta_net')
    assert old.GatedDeltaNet is GatedDeltaNet
    assert old.GatedDeltaNet2 is GatedDeltaNet2
    assert old.GatedDeltaNetSubmodules is GatedDeltaNetSubmodules
    namespace = {}
    exec('from megatron.core.ssm.gated_delta_net import *', namespace)
    assert namespace['GatedDeltaNet'] is GatedDeltaNet


@pytest.mark.parametrize(
    'old,symbol',
    [
        pair
        for pair in [
            ('megatron.core.ssm.ssm_inference', 'SSMChunking'),
            ('megatron.core.ssm.mamba_mixer', 'MambaMixer'),
            ('megatron.core.ssm.gated_delta_product', 'GatedDeltaProductMixer'),
            ('megatron.core.transformer.experimental_attention_variant.dsa', 'DSAttention'),
        ]
        if pair[0] in FORWARDED  # each family's move adds its entry
    ],
)
def test_canonical_identity_and_legacy_pickle(old, symbol):
    """Old pickle GLOBAL records load the class defined at the new canonical path."""
    canonical = importlib.import_module(FORWARDED[old][0])
    expected = getattr(canonical, symbol)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        alias = importlib.import_module(old)
        assert getattr(alias, symbol) is expected
        restored = pickle.loads(f'c{old}\n{symbol}\n.'.encode())
    assert restored is expected
    assert expected.__module__ == canonical.__name__
    assert pickle.loads(pickle.dumps(expected)) is expected


def test_internal_imports_resolve_after_move():
    """Catch stale relative imports, including the Mamba context-parallel regression."""
    sources = set((REPO_ROOT / 'megatron/core/ops').rglob('*.py'))
    sources.update(_module_file(target) for targets in FORWARDED.values() for target in targets)
    missing = []
    for path in sorted(sources):
        package = '.'.join(path.relative_to(REPO_ROOT).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = '.' * node.level + (node.module or '')
            module = importlib.util.resolve_name(module, package) if node.level else module
            if module.startswith('megatron.') and not _module_file(module).is_file():
                missing.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}: {module}')
    assert not missing, '\n'.join(missing)


def test_in_tree_code_uses_canonical_imports():
    """Production code and tests must exercise implementations instead of the aliases."""
    obsolete = tuple(name + '.' for name in PACKAGE_MARKERS)
    violations = []
    for folder in ('megatron', 'tests', 'examples', 'tools'):
        for path in (REPO_ROOT / folder).rglob('*.py'):
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[:3] == ('tests', 'unit_tests', 'ops'):
                continue
            module = '.'.join(relative.with_suffix('').parts)
            if module.startswith(obsolete):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f'{node.module}.{a.name}' for a in node.names]
                if any(name in PACKAGE_MARKERS or name.startswith(obsolete) for name in names):
                    violations.append(f'{relative}:{node.lineno}')
    assert not violations, '\n'.join(violations)


@pytest.mark.skipif("megatron.core.ssm.ops.mamba2" not in FORWARDED, reason="moved in a later PR of this stack")
def test_batch_invariant_pins_canonical_mamba_kernels(monkeypatch):
    """Moving the SSM kernels must not disconnect batch-invariant autotuner pinning."""
    from megatron.core.ops.ssm.mamba2 import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as kernels

    autotuners = {
        "_bmm_chunk_fwd_kernel": ssd_bmm._bmm_chunk_fwd_kernel,
        "_chunk_scan_fwd_kernel": ssd_chunk_scan._chunk_scan_fwd_kernel,
        "_chunk_cumsum_fwd_kernel": ssd_chunk_state._chunk_cumsum_fwd_kernel,
        "_chunk_state_fwd_kernel": ssd_chunk_state._chunk_state_fwd_kernel,
        "_state_passing_fwd_kernel": ssd_state_passing._state_passing_fwd_kernel,
    }
    original = {name: kernel.configs for name, kernel in autotuners.items()}
    monkeypatch.setattr(kernels, '_PINNED_AUTOTUNERS', [])
    try:
        kernels._pin_mamba_autotuners()
        for name, kernel in autotuners.items():
            assert len(kernel.configs) == 1
            for key, value in kernels._PINNED_MAMBA_CONFIGS[name].items():
                assert kernel.configs[0].kwargs[key] == value
    finally:
        kernels._unpin_mamba_autotuners()
    for name, kernel in autotuners.items():
        assert kernel.configs is original[name]
