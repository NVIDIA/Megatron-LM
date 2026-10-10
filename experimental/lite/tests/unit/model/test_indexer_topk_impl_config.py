# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The ``indexer_topk`` field of the GLM-5 ``ImplConfig`` (CPU).

``ImplConfig.indexer_topk`` takes an ``IndexerTopKConfig``, its mapping form or None (the
default). A value is validated when the ``ImplConfig`` is built and kept as given;
``build_model`` hands it to ``configure_indexer_topk`` with the model's indexer operand format
(GLM-5 DSA: ``fp8``) and puts the result in ``ModelBundle.extras["indexer_topk"]``. Unset,
nothing of the indexer top-k package is imported (checked in a child interpreter).
``build_model`` runs here on the CPU with a stand-in model chunk.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = LITE_ROOT.parents[1]

# Per model: protocol package, model class, indexer operand format, and the build_model step
# that the binding follows.
_MODELS = {
    "glm5": ("megatron.lite.model.glm5.lite", "Glm5Model", "fp8", "set_cross_entropy_fusion")
}
# The indexer top-k package and the bindings module.
_PACKAGES = (
    "megatron.lite.primitive.kernels.indexer_topk",
    "megatron.lite.primitive.modules.attention.indexer_topk",
)
_SPEC = {
    "backend": "litetopk",
    "precision": "exact",
    "litetopk": {"source": "/deps/plugins/glm-litetopk-raw32-abi1", "expected_source_id": "0" * 12},
    "exact_topk": {"source": "/deps/native-exact-tie"},
}


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _protocol(model: str):
    return importlib.import_module(f"{_MODELS[model][0]}.protocol")


def _package_modules() -> list[str]:
    """The modules of the indexer top-k package and the bindings module that are imported."""
    return sorted(
        name
        for name in sys.modules
        if any(name == package or name.startswith(f"{package}.") for package in _PACKAGES)
    )


class _Consumer(nn.Module):
    """An indexer top-k consumer (``IndexerTopKConsumer``) that records its bindings."""

    def __init__(self) -> None:
        super().__init__()
        self.bindings: list = []

    def indexer_geometry(self):
        return None

    def set_indexer_topk(self, binding) -> None:
        self.bindings.append(binding)


class _Chunk(nn.Module):
    """Stands in for ``Glm5Model``: one parameter, one consumer."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.attention = _Consumer()


def _build(model: str, patch, **impl_kwargs):
    """Run the protocol's ``build_model`` on the CPU with a stand-in model chunk.

    ``patch(owner, name, value)`` installs the stubs: ``monkeypatch.setattr`` in the test
    process, ``setattr`` in a child interpreter.
    """
    from megatron.lite.primitive.parallel import ParallelState

    package, model_class, _, _ = _MODELS[model]
    protocol = importlib.import_module(f"{package}.protocol")
    patch(importlib.import_module(f"{package}.model"), model_class, _Chunk)
    patch(protocol, "init_parallel", lambda _parallel: ParallelState())
    patch(nn.Module, "cuda", lambda module, *_args, **_kwargs: module)
    model_cfg = SimpleNamespace(num_nextn_predict_layers=1, mtp_loss_scaling_factor=0.1)
    impl_cfg = protocol.ImplConfig(optimizer=None, **impl_kwargs)
    return protocol.build_model(model_cfg, impl_cfg=impl_cfg)


def _spy_configure(monkeypatch, events: list[str]):
    """Replace ``configure_indexer_topk``; return its recorded calls and its return value."""
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    calls = []
    installation = object()

    def configure(chunks, config, **kwargs):
        calls.append((chunks, config, kwargs))
        events.append("configure")
        return installation

    monkeypatch.setattr(bindings, "configure_indexer_topk", configure)
    return calls, installation


def _record_steps(model: str, monkeypatch, events: list[str]) -> None:
    """Record the build_model step before the binding and the QAT step after it."""
    protocol = _protocol(model)
    for name, label in ((_MODELS[model][3], "setup"), ("apply_qat_to_chunks", "qat")):
        original = getattr(protocol, name)

        def step(*args, _original=original, _label=label, **kwargs):
            events.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(protocol, name, step)


@pytest.mark.parametrize("model", _MODELS)
def test_defaults_none(model):
    impl_config = _protocol(model).ImplConfig
    field = {field.name: field for field in dataclasses.fields(impl_config)}["indexer_topk"]
    assert field.init and field.default is None
    assert impl_config().indexer_topk is None
    assert impl_config(indexer_topk=None).indexer_topk is None


@pytest.mark.parametrize("model", _MODELS)
def test_dict_normalization_and_unknown_keys(model):
    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerTopKConfig,
        IndexerTopKConfigError,
        LiteTopKPluginConfig,
        normalize_indexer_topk_config,
    )

    impl_config = _protocol(model).ImplConfig
    # Kept as given (build_model normalizes it again when it binds the layers).
    assert impl_config(indexer_topk=_SPEC).indexer_topk is _SPEC
    assert normalize_indexer_topk_config(_SPEC) == IndexerTopKConfig(
        backend="litetopk",
        precision="exact",
        litetopk=LiteTopKPluginConfig(
            source="/deps/plugins/glm-litetopk-raw32-abi1", expected_source_id="0" * 12
        ),
        exact_topk=ExactTopKConfig(source="/deps/native-exact-tie"),
    )
    typed = IndexerTopKConfig(backend="reference", precision="fast")
    assert impl_config(indexer_topk=typed).indexer_topk is typed
    # Any mapping, for example a read-only one.
    proxy = MappingProxyType({"backend": "default"})
    assert impl_config(indexer_topk=proxy).indexer_topk is proxy

    # Unknown keys fail with the valid keys listed; expert knobs (IndexerTopKTuning fields) are
    # not model configuration.
    unknown = (
        (
            {"backend": "reference", "required": True, "tile_rows": 4},
            "indexer_topk has unknown keys ['required', 'tile_rows']; valid keys are backend, "
            "precision",
        ),
        (
            {**_SPEC, "litetopk": {"path": "/deps/plugin"}},
            "indexer_topk.litetopk has unknown keys ['path']; valid keys are source",
        ),
        (
            {**_SPEC, "exact_topk": {"source": "/x", "sha256": {}}},
            "indexer_topk.exact_topk has unknown keys ['sha256']; valid keys are source",
        ),
    )
    for value, message in unknown:
        with pytest.raises(IndexerTopKConfigError, match=re.escape(message)):
            impl_config(indexer_topk=value)


_INVALID = {
    "backend": ({"backend": "fastest"}, "indexer_topk.backend must be one of"),
    "precision": ({"precision": "approximate"}, "indexer_topk.precision must be one of"),
    "litetopk-without-plugin": (
        {"backend": "litetopk", "precision": "fast"},
        "indexer_topk.litetopk.source is required: LiteTopK kernels are not bundled",
    ),
    "exact-without-exact-topk": (
        {"backend": "reference"},
        "indexer_topk.exact_topk.source is required for precision='exact'",
    ),
    "plugin-source-type": (
        {"backend": "litetopk", "precision": "fast", "litetopk": {"source": 1}},
        "indexer_topk.litetopk.source must be a non-empty path string",
    ),
    "plugin-pin": (
        {**_SPEC, "litetopk": {"source": "/p", "expected_source_id": "ABC"}},
        "indexer_topk.litetopk.expected_source_id must be 12 characters in lowercase hex",
    ),
    "not-a-mapping": ("reference", "indexer_topk must be IndexerTopKConfig, a mapping or None"),
}


@pytest.mark.parametrize("case", _INVALID)
@pytest.mark.parametrize("model", _MODELS)
def test_invalid_values_fail_at_construction(model, case):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKConfigError

    value, message = _INVALID[case]
    with pytest.raises(IndexerTopKConfigError, match=re.escape(message)):
        _protocol(model).ImplConfig(indexer_topk=value)
    # A ValueError, as the other ImplConfig validation errors.
    assert issubclass(IndexerTopKConfigError, ValueError)


@pytest.mark.parametrize("form", ["mapping", "typed"])
@pytest.mark.parametrize("model", _MODELS)
def test_build_model_configures_with_native_format(model, form, monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKConfig

    spec = {"backend": "reference", "precision": "fast"}
    if form == "typed":
        spec = IndexerTopKConfig(**spec)
    events: list[str] = []
    calls, installation = _spy_configure(monkeypatch, events)
    _record_steps(model, monkeypatch, events)
    bundle = _build(model, monkeypatch.setattr, indexer_topk=spec)
    ((chunks, config, kwargs),) = calls
    # The bundle's chunks, the configuration as given, the model's operand format and no
    # tuning (models never carry tuning values).
    assert chunks is bundle.chunks and config is spec
    assert kwargs == {"native_format": _MODELS[model][2]}
    assert bundle.extras["indexer_topk"] is installation
    assert events == ["setup", "configure", "qat"]


@pytest.mark.parametrize("model", _MODELS)
def test_build_model_unset_imports_and_configures_nothing(model, monkeypatch):
    events: list[str] = []
    calls, _ = _spy_configure(monkeypatch, events)
    # Importing either module now raises ImportError.
    for package in _PACKAGES:
        monkeypatch.setitem(sys.modules, package, None)
    bundle = _build(model, monkeypatch.setattr)
    assert calls == [] and "indexer_topk" not in bundle.extras
    assert bundle.chunks[0].attention.bindings == []


@pytest.mark.parametrize("model", _MODELS)
def test_build_model_default_backend_unbinds(model, monkeypatch):
    """The real configure_indexer_topk: backend ``default`` unbinds every consumer."""
    bundle = _build(model, monkeypatch.setattr, indexer_topk={"backend": "default"})
    assert bundle.extras["indexer_topk"] is None
    assert bundle.chunks[0].attention.bindings == [None]


def test_not_in_architecture_config():
    from megatron.lite.model.glm5.config import Glm5Config

    assert "indexer_topk" not in {field.name for field in dataclasses.fields(Glm5Config)}
    for model in _MODELS:
        fields = {field.name for field in dataclasses.fields(_protocol(model).ImplConfig)}
        assert "indexer_topk" in fields


@pytest.mark.parametrize("model", _MODELS)
def test_runtime_forwards_indexer_topk_unchanged(model):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKConfigError
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg

    protocol = _protocol(model)
    spec = {"backend": "reference", "precision": "fast"}
    runtime_config = MegatronLiteConfig(model_name=model, impl_cfg={"indexer_topk": spec})
    assert _build_impl_cfg(protocol, runtime_config).indexer_topk is spec
    assert _build_impl_cfg(protocol, MegatronLiteConfig(model_name=model)).indexer_topk is None
    runtime_config = MegatronLiteConfig(
        model_name=model, impl_cfg={"indexer_topk": {**spec, "tile_rows": 4}}
    )
    with pytest.raises(IndexerTopKConfigError, match=re.escape("unknown keys ['tile_rows']")):
        _build_impl_cfg(protocol, runtime_config)


def test_protocol_import_is_package_free_when_unset(tmp_path):
    """Importing the GLM-5 protocol, building its ImplConfig and running build_model with
    ``indexer_topk`` unset import nothing of the indexer top-k package; a configured field
    does (the control)."""
    script = textwrap.dedent("""
        import importlib.util, json, sys, types

        try:
            import transformer_engine.pytorch  # noqa: F401
        except ImportError:
            for name in ("transformer_engine", "transformer_engine.pytorch"):
                sys.modules[name] = types.ModuleType(name)
            sys.modules["transformer_engine"].pytorch = sys.modules["transformer_engine.pytorch"]

        spec = importlib.util.spec_from_file_location("impl_config_tests", sys.argv[1])
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        report = {"start": tests._package_modules()}
        for model in tests._MODELS:
            protocol = tests._protocol(model)
            protocol.ImplConfig()
            protocol.ImplConfig(indexer_topk=None)
            bundle = tests._build(model, setattr)
            report[model] = {"extras": sorted(bundle.extras)}
        report["unset"] = tests._package_modules()
        for model in tests._MODELS:
            bundle = tests._build(model, setattr, indexer_topk={"backend": "default"})
            report[model]["default"] = [
                bundle.extras["indexer_topk"], bundle.chunks[0].attention.bindings
            ]
        report["set"] = tests._package_modules()
        print(json.dumps(report))
        """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(LITE_ROOT), str(REPO_ROOT), environment.get("PYTHONPATH")))
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script, __file__],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["start"] == [] and report["unset"] == []
    for model in _MODELS:
        assert "indexer_topk" not in report[model]["extras"]
        assert report[model]["default"] == [None, [None]]
    assert set(_PACKAGES) <= set(report["set"])
