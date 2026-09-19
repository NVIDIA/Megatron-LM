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
from tests.unit_tests.ops.deprecated_paths import FORWARDED

REPO_ROOT = Path(__file__).resolve().parents[3]


def _module_file(name):
    path = REPO_ROOT.joinpath(*name.split('.'))
    return path.with_suffix('.py') if path.with_suffix('.py').is_file() else path / '__init__.py'


@pytest.fixture(params=[('first',), ('first', 'second')], ids=['one-target', 'two-targets'])
def synthetic_forwarder(request, tmp_path, monkeypatch):
    """Create importable forwarders without depending on any migrated operation family."""
    package = '_ops_compat_fixture'
    root = tmp_path / package
    root.mkdir()
    targets = tuple(f'{package}.{name}' for name in request.param)
    forwarder = (
        'from megatron.core.ops._compat import deprecated_module\n'
        f'__getattr__, __dir__ = deprecated_module(__name__, *{targets!r}, '
        'removal_version="99.0")\n'
    )
    sources = {
        '__init__.py': '',
        'first.py': '__all__ = ["public"]\npublic = object()\n_private = object()\nshared = object()\n',
        'second.py': 'other = object()\n_other_private = object()\nshared = object()\n',
        'legacy.py': forwarder,
        'legacy_package/__init__.py': forwarder,
        'legacy_package/child.py': 'value = object()\n',
    }
    for filename, source in sources.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield package, targets
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + '.'):
                del sys.modules[name]


def test_synthetic_forwarder_import_is_lazy(synthetic_forwarder):
    """Import and dunder probes warn once without loading optional implementations."""
    package, targets = synthetic_forwarder
    old = f'{package}.legacy'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        module = importlib.import_module(old)
        assert importlib.import_module(old) is module
        assert not hasattr(module, '__wrapped__')
        assert not hasattr(module, '__path__')
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    message = str(caught[0].message)
    assert old in message and targets[0] in message and '99.0' in message
    assert all(target not in sys.modules for target in targets)


def test_synthetic_forwarder_attribute_resolution(synthetic_forwarder):
    """Attributes retain canonical identity, including private names and target precedence."""
    package, targets = synthetic_forwarder
    with pytest.warns(DeprecationWarning):
        module = importlib.import_module(f'{package}.legacy')
    public = module.public
    first = sys.modules[targets[0]]
    assert public is first.public
    assert module._private is first._private
    assert module.shared is first.shared
    if len(targets) == 2:
        assert targets[1] not in sys.modules
        other = module.other
        second = sys.modules[targets[1]]
        assert other is second.other
        assert module._other_private is second._other_private
    with pytest.raises(AttributeError, match='missing_export'):
        getattr(module, 'missing_export')


def test_synthetic_forwarder_wildcard_exports(synthetic_forwarder):
    """Wildcard imports honor explicit exports and discover public names on other targets."""
    package, targets = synthetic_forwarder
    old = f'{package}.legacy'
    with pytest.warns(DeprecationWarning):
        module = importlib.import_module(old)
    expected = {'public'}
    if len(targets) == 2:
        expected.update(('other', 'shared'))
    assert set(module.__all__) == expected
    namespace = {}
    exec(f'from {old} import *', namespace)
    assert set(namespace) - {'__builtins__'} == expected
    for name in expected:
        assert namespace[name] is getattr(module, name)
    expected_dir = {name for target in targets for name in dir(sys.modules[target])}
    assert dir(module) == sorted(expected_dir)


def test_synthetic_forwarder_package_child_is_lazy(synthetic_forwarder):
    """A real child import must not load the parent package's implementation targets."""
    package, targets = synthetic_forwarder
    old = f'{package}.legacy_package'
    with pytest.warns(DeprecationWarning):
        parent = importlib.import_module(old)
    namespace = {}
    exec(f'from {old} import child', namespace)
    assert namespace['child'] is importlib.import_module(f'{old}.child')
    assert parent.child is namespace['child']
    assert all(target not in sys.modules for target in targets)


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


@pytest.mark.skipif(
    "megatron.core.ssm.ops.mamba2" not in FORWARDED, reason="moved in a later PR of this stack"
)
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


@pytest.mark.skipif(
    "megatron.core.ssm.ops" not in FORWARDED, reason="moved in a later PR of this stack"
)
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


@pytest.mark.skipif(
    "megatron.core.ssm.gated_delta_net" not in FORWARDED, reason="moved in a later PR of this stack"
)
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
    # The stack moves one family at a time; unmoved modules retain their original paths.
    obsolete = tuple(name + '.' for name in FORWARDED)
    violations = []
    for folder in ('megatron', 'tests', 'examples', 'tools'):
        for path in (REPO_ROOT / folder).rglob('*.py'):
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[:3] == ('tests', 'unit_tests', 'ops'):
                continue
            package = '.'.join(relative.parent.parts)
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = '.' * node.level + (node.module or '')
                    if node.level:
                        module = importlib.util.resolve_name(module, package)
                    names = [module] + [f'{module}.{a.name}' for a in node.names]
                if any(name in FORWARDED or name.startswith(obsolete) for name in names):
                    violations.append(f'{relative}:{node.lineno}')
    assert not violations, '\n'.join(violations)


@pytest.mark.parametrize(
    'source,moved,rejected',
    [
        ('from megatron.core.ssm.mamba_layer import MambaLayer', True, False),
        ('from .mlp_layer import MLPLayer', True, False),
        (
            'from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention',
            True,
            False,
        ),
        ('from megatron.core.ops.ssm.mamba2.mixer import MambaMixer', True, False),
        ('import megatron.core.ssm.mamba_mixer', False, False),
        ('import megatron.core.ssm.mamba_mixer', True, True),
        ('from megatron.core.ssm import mamba_mixer', True, True),
        ('from .mamba_mixer import MambaMixer', True, True),
        ('from . import mamba_mixer', True, True),
        ('from megatron.core.ssm.ops.mamba2 import ssd_combined', True, True),
    ],
)
def test_canonical_import_check_tracks_partial_moves(
    source, moved, rejected, tmp_path, monkeypatch
):
    """Allow later slices while detecting stale imports inside an unmoved SSM module."""
    path = tmp_path / 'megatron/core/ssm/mamba_layer.py'
    path.parent.mkdir(parents=True)
    path.write_text(source + '\n')
    forwarded = {
        'megatron.core.ssm.mamba_mixer': ('megatron.core.ops.ssm.mamba2.mixer',),
        'megatron.core.ssm.ops': ('megatron.core.ops.ssm',),
    }
    monkeypatch.setattr(sys.modules[__name__], 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(sys.modules[__name__], 'FORWARDED', forwarded if moved else {})
    if rejected:
        with pytest.raises(AssertionError, match='megatron/core/ssm/mamba_layer.py:1'):
            test_in_tree_code_uses_canonical_imports()
    else:
        test_in_tree_code_uses_canonical_imports()


@pytest.mark.skipif(
    "megatron.core.ssm.ops.mamba2" not in FORWARDED, reason="moved in a later PR of this stack"
)
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
