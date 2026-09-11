# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Metadata conformance and validation policy, separate from numerical audits."""

import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.ops import kernel_metadata as metadata
from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismPolicy,
    DeterminismResult,
    KernelMetadata,
    validate_determinism,
    validate_kernel,
    validate_kernels,
)
from megatron.core.ops.ssm.common.kernel_metadata import CAUSAL_CONV, CAUSAL_CONV_CP
from megatron.core.ops.ssm.gdp.kernel_metadata import GDP_CONV

FAMILIES = (
    "attention",
    "ssm.common",
    "ssm.mamba2",
    "ssm.gated_delta",
    "ssm.gdp",
    "attention.dsa",
    "attention.csa",
)
KERNELS = tuple(
    kernel
    for family in FAMILIES
    for kernel in importlib.import_module("megatron.core.ops." + family).KERNELS
)


def _kernel(status=Determinism.UNKNOWN, requires=()):
    return KernelMetadata(
        name="test.kernel",
        requires=requires,
        determinism=DeterminismResult(status, "Only the test's execution context is covered."),
        contract="test.contract",
    )


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda kernel: kernel.name)
def test_family_metadata_conformance(kernel):
    assert isinstance(kernel, KernelMetadata)
    assert isinstance(kernel.determinism.status, Determinism)
    assert kernel.determinism.reason.strip()
    assert importlib.import_module(kernel.contract).__doc__.strip()
    assert isinstance(kernel.requires, tuple)
    assert all(isinstance(dependency, Dependency) for dependency in kernel.requires)
    with pytest.raises(FrozenInstanceError):
        kernel.name = "changed"


def test_unique_entry_point_names():
    assert len({kernel.name for kernel in KERNELS}) == len(KERNELS)


def test_metadata_inspection_does_not_import_optional_libraries():
    code = f"""
import importlib
import sys
import megatron.core
before = set(sys.modules)
for family in {FAMILIES!r}:
    module = importlib.import_module('megatron.core.ops.' + family)
    assert module.KERNELS
    for kernel in module.KERNELS:
        assert kernel.determinism.reason
        assert isinstance(kernel.requires, tuple)
optional = {{'triton', 'tilelang', 'fla', 'mamba_ssm', 'causal_conv1d', 'gdp_attn',
             'cudnn', 'flash_mla', 'fast_hadamard_transform'}}
assert not {{name for name in set(sys.modules) - before if name.split('.')[0] in optional}}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("error", [ImportError("not installed"), OSError("missing native library")])
def test_dependency_error_identifies_kernel_and_keeps_cause(monkeypatch, error):
    def fail(_name):
        raise error

    monkeypatch.setattr(metadata, "import_module", fail)
    with pytest.raises(
        ImportError, match=r"test.kernel requires example-dist.*example_module"
    ) as raised:
        validate_kernel(_kernel(requires=(Dependency("example-dist", "example_module"),)))
    assert raised.value.__cause__ is error


def test_dependency_checks_real_export_not_only_package_name(monkeypatch):
    monkeypatch.setattr(metadata, "import_module", lambda _name: SimpleNamespace(kernel=None))
    with pytest.raises(ImportError, match="missing export example_module.kernel"):
        Dependency("example-dist", "example_module", ("kernel",)).validate("selected")


@pytest.mark.parametrize(
    ("installed", "accepted"), [("1.2.0", True), ("1.2.0+dev", True), ("1.1", False)]
)
def test_version_constraints_use_distribution_name(monkeypatch, installed, accepted):
    seen = []
    monkeypatch.setattr(metadata, "import_module", lambda _name: SimpleNamespace(kernel=object()))

    def version(name):
        seen.append(name)
        return installed

    monkeypatch.setattr(metadata.metadata, "version", version)
    dependency = Dependency("example-dist>=1.2", "example_module", ("kernel",))
    if accepted:
        dependency.validate("selected")
    else:
        with pytest.raises(ImportError, match="found version 1.1"):
            dependency.validate("selected")
    assert seen == ["example-dist"]


def test_source_import_requires_metadata_only_when_versioned(monkeypatch):
    monkeypatch.setattr(metadata, "import_module", lambda _name: SimpleNamespace())

    def missing(_name):
        raise metadata.metadata.PackageNotFoundError

    monkeypatch.setattr(metadata.metadata, "version", missing)
    Dependency("example-dist", "example_module").validate("selected")
    with pytest.raises(ImportError, match="cannot verify the installed version"):
        Dependency("example-dist>=1", "example_module").validate("selected")


def test_dependency_checks_exports_inside_a_lazy_namespace(monkeypatch):
    target = object()
    monkeypatch.setattr(
        metadata, "import_module", lambda name: SimpleNamespace(DSA=SimpleNamespace(kernel=target))
    )
    Dependency("example-dist", "example", ("DSA.kernel",)).validate("selected")
    with pytest.raises(ImportError, match="missing export example.DSA.missing"):
        Dependency("example-dist", "example", ("DSA.missing",)).validate("selected")


def test_lazy_namespace_native_loading_error_preserves_cause(monkeypatch):
    cause = OSError("lazy native library cannot load")

    class Namespace:
        def __getattr__(self, name):
            raise cause

    monkeypatch.setattr(metadata, "import_module", lambda name: SimpleNamespace(DSA=Namespace()))
    with pytest.raises(ImportError, match="Kernel selected requires") as caught:
        Dependency("example-dist", "example", ("DSA.kernel",)).validate("selected")
    assert caught.value.__cause__ is cause


def test_conditional_dependency_is_checked_only_for_selected_feature(monkeypatch):
    seen = []
    monkeypatch.setattr(
        metadata, "import_module", lambda name: seen.append(name) or SimpleNamespace()
    )
    kernel = _kernel(requires=(Dependency("example-dist", "example_module", feature="qk_l2norm"),))
    validate_kernel(kernel)
    assert seen == []
    validate_kernel(kernel, features=("qk_l2norm",))
    assert seen == ["example_module"]


@pytest.mark.parametrize("status", list(Determinism))
@pytest.mark.parametrize("policy", list(DeterminismPolicy))
def test_determinism_policy_matrix(status, policy):
    kernel = _kernel(status)
    if policy is DeterminismPolicy.IGNORE or status is Determinism.DETERMINISTIC:
        validate_kernel(kernel, determinism=policy)
    elif status is Determinism.UNKNOWN and policy is DeterminismPolicy.WARN:
        with pytest.warns(UserWarning, match="test.kernel determinism is unknown"):
            validate_kernel(kernel, determinism=policy)
    else:
        with pytest.raises(RuntimeError, match="test.kernel determinism is " + status.value):
            validate_kernel(kernel, determinism=policy)


def test_environment_assessment_is_not_cached(monkeypatch):
    state = [Determinism.DETERMINISTIC]
    kernel = replace(
        _kernel(), determinism_check=lambda: DeterminismResult(state[0], "Current test mode.")
    )
    validate_kernel(kernel, determinism=DeterminismPolicy.ERROR)
    state[0] = Determinism.NONDETERMINISTIC
    with pytest.raises(RuntimeError, match="Current test mode"):
        validate_kernel(kernel, determinism=DeterminismPolicy.ERROR)


@pytest.mark.parametrize("status", [None, True, "deterministic"])
def test_untyped_or_missing_determinism_is_rejected(status):
    with pytest.raises(ValueError, match="explicit status"):
        DeterminismResult(status, "Not an enum.")


def test_invalid_declarations_and_policy_are_rejected():
    with pytest.raises(ValueError, match="explicit determinism"):
        replace(_kernel(), determinism=None)
    with pytest.raises(ValueError, match="tuple of Dependency"):
        replace(_kernel(), requires=[Dependency("example-dist", "example_module")])
    with pytest.raises(ValueError, match="name and version specifier"):
        Dependency("example-dist[extra]", "example_module")
    with pytest.raises(ValueError, match="tuple of nonempty names"):
        Dependency("example-dist", "example_module", symbols=["mutable"])
    with pytest.raises(ValueError, match="DeterminismPolicy"):
        validate_kernel(_kernel(), determinism="ignore")
    with pytest.raises(TypeError, match="invalid determinism assessment"):
        validate_kernel(
            replace(_kernel(), determinism_check=lambda: True), determinism=DeterminismPolicy.ERROR
        )


@pytest.mark.parametrize(
    ("env", "torch_enabled", "enabled"),
    [("1", False, True), ("0", True, False), ("", True, True), ("other", False, False)],
)
@pytest.mark.parametrize(
    "kernel", [CAUSAL_CONV, CAUSAL_CONV_CP, GDP_CONV], ids=lambda kernel: kernel.name
)
def test_causal_conv_assessment_respects_environment(
    monkeypatch, env, torch_enabled, enabled, kernel
):
    from megatron.core.ops.ssm.common import causal_conv1d_cp

    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", env)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: torch_enabled)
    monkeypatch.setattr(causal_conv1d_cp, "is_causal_conv1d_min_version", lambda _version: True)
    expected = Determinism.UNKNOWN if enabled else Determinism.NONDETERMINISTIC
    assert kernel.determinism_check().status is expected


@pytest.mark.parametrize(
    "kernel", [CAUSAL_CONV, CAUSAL_CONV_CP, GDP_CONV], ids=lambda kernel: kernel.name
)
def test_causal_conv_old_version_is_not_marked_deterministic(monkeypatch, kernel):
    from megatron.core.ops.ssm.common import causal_conv1d_cp

    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    monkeypatch.setattr(causal_conv1d_cp, "is_causal_conv1d_min_version", lambda _version: False)
    with pytest.raises(RuntimeError, match="causal_conv1d >= 1.6.0"):
        validate_determinism(kernel, DeterminismPolicy.WARN)


def test_csa_mapping_is_derived_from_kernel_metadata():
    from megatron.core.ops.attention.csa import modules as csa
    from megatron.core.ops.attention.csa.kernel_metadata import CSA_OPERATION_DETERMINISM

    assert csa.CSA_OPERATION_DETERMINISM is CSA_OPERATION_DETERMINISM
    assert CSA_OPERATION_DETERMINISM == {
        "unfused_sparse_attention": "unknown",
        "non_compressed_lse": "unknown",
        "compressor_pooling": "unknown",
    }


def test_gdn_reference_selector_checks_only_its_conditional_dependency(monkeypatch):
    from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule

    calls = []
    monkeypatch.setattr(
        Dependency, "validate", lambda self, name: calls.append((self.module, name))
    )
    with pytest.warns(UserWarning, match="unknown"):
        select_gated_delta_rule("gdn2", deterministic=True)
    assert calls == []
    with pytest.warns(UserWarning, match="unknown"):
        select_gated_delta_rule("gdn2", deterministic=True, use_qk_l2norm_in_kernel=True)
    assert calls == [("fla.modules.l2norm", "gdn2.torch_chunk_gdn2")]


@pytest.mark.parametrize("use_cutedsl", [False, True])
def test_gdp_validates_selected_implementation_and_preserves_identity(monkeypatch, use_cutedsl):
    from megatron.core.ops.ssm.gdp import backends
    from megatron.core.ops.ssm.gdp.kernel_metadata import GDP_CUTEDSL, GDP_FLA

    target = lambda **kwargs: kwargs
    module_name = "gdp_attn" if use_cutedsl else "fla.ops.gated_delta_product"
    module = ModuleType(module_name)
    module.chunk_gated_delta_product = target
    monkeypatch.setitem(sys.modules, module_name, module)
    seen = []
    monkeypatch.setattr(backends, "validate_kernel", lambda kernel, **kwargs: seen.append(kernel))
    assert backends.select_gated_delta_product(use_cutedsl) is target
    assert seen == [GDP_CUTEDSL if use_cutedsl else GDP_FLA]


def test_disabled_dsa_never_validates_fused_dependencies(monkeypatch):
    from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

    def unexpected(self, name):
        pytest.fail(f"Disabled fused DSA checked {self.module} for {name}")

    monkeypatch.setattr(Dependency, "validate", unexpected)
    select_dsa_kernels(SimpleNamespace(attention_backend="unfused", dsa_kernel_backend="cudnn"))


def test_selected_dsa_native_dependency_failure_is_early(monkeypatch):
    from megatron.core.ops.attention.dsa import backends

    monkeypatch.setattr(backends, "import_module", lambda name: SimpleNamespace())

    def fail(self, name):
        raise ImportError(f"{name}: missing {self.module}")

    monkeypatch.setattr(Dependency, "validate", fail)
    with pytest.raises(RuntimeError, match="run_fused_qk_topk: missing cudnn"):
        backends.select_dsa_kernels(
            SimpleNamespace(attention_backend="auto", dsa_kernel_backend="cudnn")
        )


def test_batch_validation_checks_every_selected_kernel(monkeypatch):
    seen = []
    monkeypatch.setattr(Dependency, "validate", lambda self, name: seen.append((name, self.module)))
    first = _kernel(requires=(Dependency("first-dist", "first_module"),))
    second = replace(
        _kernel(requires=(Dependency("second-dist", "second_module"),)), name="second.kernel"
    )
    validate_kernels((first, second))
    assert seen == [("test.kernel", "first_module"), ("second.kernel", "second_module")]


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("pipeline_local", [False, True])
def test_inference_initialization_validates_before_state_allocation(
    monkeypatch, missing, pipeline_local
):
    from megatron.core.inference.config import MambaInferenceStateConfig
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    seen = []

    class Mixer:
        chunk_size = 64
        ssm_inference_chunk_size = 64

        def get_inference_kernel_metadata(self):
            return (_kernel(requires=(Dependency("inference-dist", "inference_module"),)),)

    def check(self, name):
        seen.append("validate")
        if missing:
            raise ImportError("missing inference_module")

    def shapes():
        seen.append("states")
        return (16, 4), (2, 8, 16)

    monkeypatch.setattr(Dependency, "validate", check)
    model = SimpleNamespace(
        config=SimpleNamespace(
            params_dtype=torch.float32, batch_invariant_mode=False, deterministic_mode=False
        ),
        decoder=SimpleNamespace(
            layer_type_list=(
                [Symbols.ATTENTION, Symbols.MAMBA] if pipeline_local else [Symbols.MAMBA]
            ),
            layers=[SimpleNamespace(mixer=Mixer())],
            mamba_state_shapes_per_request=shapes,
        ),
    )
    if missing:
        with pytest.raises(ImportError, match="missing inference_module"):
            MambaInferenceStateConfig.from_model(model)
        assert seen == ["validate"]
    else:
        assert MambaInferenceStateConfig.from_model(model) is not None
        assert seen == ["validate", "states"]


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_mamba_inference_declarations_follow_selected_mode(batch_invariant):
    from megatron.core.ops.ssm.common.kernel_metadata import CAUSAL_CONV_CUDA_UPDATE
    from megatron.core.ops.ssm.mamba2.kernel_metadata import MAMBA_BATCH_INVARIANT, MAMBA_DECODE
    from megatron.core.ops.ssm.mamba2.mixer import MambaMixer

    mixer = SimpleNamespace(config=SimpleNamespace(batch_invariant_mode=batch_invariant))
    kernels = MambaMixer.get_inference_kernel_metadata(mixer)
    assert (MAMBA_BATCH_INVARIANT in kernels) is batch_invariant
    assert (CAUSAL_CONV_CUDA_UPDATE in kernels) is batch_invariant
    assert (MAMBA_DECODE in kernels) is not batch_invariant


def test_gdp_dynamic_inference_does_not_require_training_backends():
    from megatron.core.ops.ssm.gdp.mixer import GatedDeltaProductMixer

    kernels = GatedDeltaProductMixer.get_inference_kernel_metadata(None)
    assert kernels
    assert {dependency.module for kernel in kernels for dependency in kernel.requires} == {
        "triton",
        "triton.language.extra.libdevice",
    }


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_reference_missing_runtime_normalization_has_useful_error(monkeypatch, variant):
    suffix = "reference" if variant == "gdn" else "reference_gdn2"
    module = importlib.import_module("megatron.core.ops.ssm.gated_delta." + suffix)

    def missing(_name):
        raise ImportError("missing l2norm")

    monkeypatch.setattr(metadata, "import_module", missing)
    value = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ImportError, match="gdn.fla.l2norm requires flash-linear-attention"):
        if variant == "gdn":
            module.torch_chunk_gated_delta_rule(
                value, value, value, value, value, use_qk_l2norm_in_kernel=True
            )
        else:
            module.torch_chunk_gdn2(
                value, value, value, value, value, value, use_qk_l2norm_in_kernel=True
            )


@pytest.mark.parametrize("owner", ["indexer", "compressor"])
def test_selected_hadamard_dependency_is_checked_before_parameters(monkeypatch, owner):
    from megatron.core.ops.attention.csa.modules import Compressor
    from megatron.core.ops.attention.dsa.modules import DSAIndexer

    def missing(self, name):
        assert self.module == "fast_hadamard_transform"
        raise ImportError(f"{name}: missing fast_hadamard_transform")

    monkeypatch.setattr(Dependency, "validate", missing)
    config = SimpleNamespace(
        hidden_size=16,
        deterministic_mode=False,
        dsa_indexer_rotate_activation=True,
        apply_rope_fusion=False,
    )
    with pytest.raises(ImportError, match="dsa.hadamard.rotate_activation: missing"):
        if owner == "indexer":
            DSAIndexer(config, submodules=None)
        else:
            Compressor(
                config,
                submodules=None,
                compress_ratio=4,
                head_dim=16,
                rotate=True,
                pg_collection=SimpleNamespace(),
            )
