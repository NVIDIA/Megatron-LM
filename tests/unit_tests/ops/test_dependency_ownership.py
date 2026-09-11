# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction owns selection and validation; model code must not second-guess it."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from megatron.core.ops import kernel_metadata as metadata
from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule
from megatron.core.ops.ssm.gdp.backends import select_gated_delta_product, select_gdp_cp_backend
from megatron.core.ops.ssm.mamba2.backends import select_mamba_kernels
from megatron.core.transformer import TransformerConfig


def _module(monkeypatch, name, **exports):
    module = ModuleType(name)
    module.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_selectors_do_not_import_unselected_optional_libraries():
    code = """
import importlib
import importlib.abc
import sys
import megatron.core
blocked = {'fla', 'mamba_ssm', 'gdp_attn', 'tilelang', 'cudnn', 'flash_mla'}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unselected dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
before = set(sys.modules)
for family in ('ssm.gdp', 'ssm.gated_delta', 'ssm.mamba2', 'attention.dsa'):
    importlib.import_module('megatron.core.ops.' + family + '.backends')
assert not {name for name in set(sys.modules) - before if name.split('.')[0] in blocked}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize(
    "select",
    [
        lambda: select_gated_delta_rule("gdn"),
        lambda: select_gated_delta_rule("gdn2"),
        lambda: select_gated_delta_product(False),
        lambda: select_gated_delta_product(True),
        lambda: select_gdp_cp_backend(False),
        lambda: select_gdp_cp_backend(True),
        lambda: select_mamba_kernels(False),
        lambda: select_mamba_kernels(True),
    ],
)
@pytest.mark.parametrize("error", [ImportError, OSError])
def test_selected_dependency_failure_preserves_cause_without_fallback(monkeypatch, select, error):
    cause = error("selected native library cannot load")
    calls = []

    def fail(name):
        calls.append(name)
        raise cause

    monkeypatch.setattr(metadata, "import_module", fail)
    with pytest.raises(ImportError, match="Kernel .* requires") as caught:
        select()
    assert caught.value.__cause__ is cause
    assert len(calls) == 1


def test_mamba_does_not_require_unselected_split_scan(monkeypatch):
    target = lambda state_dtype=None: state_dtype
    _module(monkeypatch, "mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=target)
    kernels = select_mamba_kernels(False)
    assert kernels.scan is target
    assert kernels.split_scan is None
    assert kernels.has_state_dtype
    with pytest.raises(ImportError, match="missing export .*mamba_split_conv1d_scan_combined"):
        select_mamba_kernels(True)


@pytest.mark.parametrize("use_mem_eff_path", [False, True])
@pytest.mark.parametrize("missing_package", [False, True])
def test_mamba_convolution_fallback_only_handles_an_absent_package(
    monkeypatch, use_mem_eff_path, missing_package
):
    target = lambda: None
    _module(
        monkeypatch,
        "mamba_ssm.ops.triton.ssd_combined",
        mamba_chunk_scan_combined=target,
        mamba_split_conv1d_scan_combined=target,
    )
    original_import = metadata.import_module
    cause = (
        ModuleNotFoundError("no causal_conv1d", name="causal_conv1d")
        if missing_package
        else OSError("causal_conv1d native library cannot load")
    )

    def load(name):
        if name == "causal_conv1d":
            raise cause
        return original_import(name)

    monkeypatch.setattr(metadata, "import_module", load)
    if missing_package and not use_mem_eff_path:
        assert select_mamba_kernels(False).causal_conv1d is None
    else:
        with pytest.raises(ImportError, match="requires causal-conv1d") as caught:
            select_mamba_kernels(use_mem_eff_path)
        assert caught.value.__cause__ is cause


def test_mamba_training_does_not_require_cuda_convolution_update(monkeypatch):
    target = lambda: None
    _module(monkeypatch, "mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=target)
    _module(monkeypatch, "causal_conv1d", causal_conv1d_fn=target)
    assert select_mamba_kernels(False).causal_conv1d is target


def test_packed_cp_convolution_checks_its_version_before_communication(monkeypatch):
    from megatron.core.ops.ssm.common.causal_conv1d_cp import causal_conv1d_cp
    from megatron.core.ops.ssm.common.kernel_metadata import CAUSAL_CONV_CP

    _module(monkeypatch, "causal_conv1d", causal_conv1d_fn=lambda: None)
    monkeypatch.setattr(metadata.metadata, "version", lambda name: "1.6.1")
    metadata.validate_kernel(CAUSAL_CONV_CP)
    with pytest.raises(ImportError, match="causal-conv1d>=1.7.0.*found version 1.6.1"):
        # No tensors or group are needed: selected requirements must fail first.
        causal_conv1d_cp(None, None, None, None, None, global_seq_idx=object())


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_mamba_decode_requirements_follow_configuration_not_import_sentinels(
    monkeypatch, batch_invariant
):
    from megatron.core.ops.ssm.mamba2 import mixer

    monkeypatch.setattr(mixer, "causal_conv1d_update", None)
    owner = SimpleNamespace(config=SimpleNamespace(batch_invariant_mode=batch_invariant))
    selected = mixer.MambaMixer.get_inference_kernel_metadata(owner)
    assert (mixer.CAUSAL_CONV_TRITON_UPDATE in selected) is (not batch_invariant)
    assert (mixer.CAUSAL_CONV_CUDA_UPDATE in selected) is batch_invariant
    assert (mixer.MAMBA_DECODE in selected) is (not batch_invariant)
    assert (mixer.MAMBA_BATCH_INVARIANT in selected) is batch_invariant


def test_gdp_decode_prepare_checks_libdevice_exports(monkeypatch):
    from megatron.core.ops.ssm.gdp.kernel_metadata import GDP_PREPARE

    _module(monkeypatch, "triton.language.extra.libdevice", exp=lambda: None, log1p=lambda: None)
    with pytest.raises(ImportError, match="gdp_decode_prepare.*missing export .*div_rn"):
        metadata.validate_kernel(GDP_PREPARE)


@pytest.mark.parametrize("missing", ["lighting_indexer", "SparseMLA", "SparseIndexerKLLoss"])
def test_dsa_selection_rejects_missing_concrete_tilelang_entry_points(monkeypatch, missing):
    from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

    exports = {
        name: (None if name == missing else lambda: None)
        for name in (
            "lighting_indexer",
            "lighting_indexer_indices",
            "SparseMLA",
            "SparseIndexerKLLoss",
            "sparse_indexer_target_interface",
        )
    }
    monkeypatch.setattr(metadata, "import_module", lambda name: SimpleNamespace(**exports))
    with pytest.raises(RuntimeError, match="missing export .*" + missing):
        select_dsa_kernels(SimpleNamespace(attention_backend="auto", dsa_kernel_backend="tilelang"))


@pytest.mark.parametrize("use_cutedsl", [False, True])
def test_gdp_determinism_is_validated_by_selected_provider_target(monkeypatch, use_cutedsl):
    from megatron.core.ops.ssm.gdp import backends

    name = "gdp_attn" if use_cutedsl else "fla.ops.gated_delta_product"
    target = lambda: None
    _module(monkeypatch, name, chunk_gated_delta_product=target)
    calls = []
    monkeypatch.setattr(
        backends, "validate_kernel", lambda kernel, **kw: calls.append((kernel, kw))
    )
    assert select_gated_delta_product(use_cutedsl, deterministic=True) is target
    assert calls[0][1] == {"determinism": metadata.DeterminismPolicy.WARN}
    assert ("cutedsl" in calls[0][0].name) is use_cutedsl


def test_dsa_selection_checks_cudnn_namespace_members(monkeypatch):
    from megatron.core.ops.attention.dsa.backends import select_dsa_kernels

    monkeypatch.setattr(
        metadata, "import_module", lambda name: SimpleNamespace(DSA=SimpleNamespace())
    )
    with pytest.raises(RuntimeError, match="missing export cudnn.DSA.indexer_top_k_wrapper"):
        select_dsa_kernels(SimpleNamespace(attention_backend="auto", dsa_kernel_backend="cudnn"))


class _BeforeParameters(Exception):
    pass


def _stop_before_parameters(*args, **kwargs):
    raise _BeforeParameters


def _config(**kwargs):
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        mamba_num_heads=4,
        mamba_num_groups=1,
        mamba_head_dim=32,
        mamba_state_dim=32,
        transformer_impl="local",
        **kwargs,
    )


def _groups():
    return SimpleNamespace(tp=SimpleNamespace(size=lambda: 1), cp=SimpleNamespace(size=lambda: 1))


@pytest.mark.parametrize("use_cutedsl", [False, True])
@pytest.mark.parametrize("rmsnorm", [False, True])
def test_gdp_custom_provider_is_not_gated_by_default_recurrence(monkeypatch, use_cutedsl, rmsnorm):
    from megatron.core.ops.ssm.gdp import mixer

    target = lambda *args, **kwargs: None
    calls = []

    class Provider:
        def gated_delta_product(self, use_cutedsl=False, deterministic=False):
            calls.append((use_cutedsl, deterministic))
            return target

    _module(monkeypatch, "fla.modules.l2norm", l2_norm=target)
    _module(monkeypatch, "causal_conv1d", causal_conv1d_fn=target)
    seen = []

    def validate(dependency, kernel_name):
        seen.append((dependency.module, kernel_name))
        if dependency.module.startswith("mamba_ssm"):
            raise ImportError("selected RMSNorm is missing")
        assert dependency.module in ("causal_conv1d", "fla.modules.l2norm")

    monkeypatch.setattr(metadata.Dependency, "validate", validate)
    monkeypatch.setattr(mixer, "build_module", _stop_before_parameters)
    model = mixer.GatedDeltaProductMixer.__new__(mixer.GatedDeltaProductMixer)
    expected = ImportError if rmsnorm else _BeforeParameters
    with pytest.raises(expected):
        model.__init__(
            _config(gdp_cutedsl_kernel=use_cutedsl),
            mixer.GatedDeltaProductMixerSubmodules(),
            128,
            pg_collection=_groups(),
            kernel_backend=Provider(),
            rmsnorm=rmsnorm,
        )
    assert model.gdp_kernel is target
    assert calls == [(use_cutedsl, False)]
    assert not model._parameters and not model._modules
    assert any(name == "fla.modules.l2norm" for name, _ in seen) is (not use_cutedsl)
    assert any(name.startswith("mamba_ssm") for name, _ in seen) is rmsnorm


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_custom_provider_does_not_require_default_recurrence(monkeypatch, variant, normalize):
    from megatron.core.ops.ssm.gated_delta import common, gdn, gdn2

    target = lambda *args, **kwargs: None
    _module(monkeypatch, "fla.modules.convolution", causal_conv1d=target)
    _module(monkeypatch, "fla.modules.l2norm", l2norm=target)
    calls = []
    seen = []

    class Provider:
        def gated_delta_rule(self, variant, deterministic=False):
            calls.append((variant, deterministic))
            return target

    def validate(dependency, kernel_name):
        seen.append(dependency.module)
        assert dependency.module in ("fla.modules.convolution", "fla.modules.l2norm")

    monkeypatch.setattr(metadata.Dependency, "validate", validate)
    monkeypatch.setattr(common, "build_module", _stop_before_parameters)
    cls = gdn.GatedDeltaNet if variant == "gdn" else gdn2.GatedDeltaNet2
    model = cls.__new__(cls)
    with pytest.raises(_BeforeParameters):
        model.__init__(
            _config(),
            common.GatedDeltaNetSubmodules(),
            pg_collection=_groups(),
            kernel_backend=Provider(),
            use_qk_l2norm=normalize,
        )
    assert model.gated_delta_rule is target
    assert calls == [(variant, False)]
    assert ("fla.modules.l2norm" in seen) is normalize
    assert not model._parameters and not model._modules


@pytest.mark.parametrize("owner", ["mla", "dsv4", "compressor", "csa_indexer"])
def test_fused_rope_dependencies_are_checked_before_attention_allocation(monkeypatch, owner):
    from megatron.core.ops.attention.csa.modules import Compressor, CSAIndexer
    from megatron.core.ops.attention.dsv4 import DSv4HybridSelfAttention
    from megatron.core.ops.attention.mla import AbsorbedMLASelfAttention

    def fail(dependency, name):
        assert name in ("mla.triton.fused_rope", "dsv4.triton.fused_rope")
        raise ImportError("selected fused RoPE dependency missing")

    monkeypatch.setattr(metadata.Dependency, "validate", fail)
    config = SimpleNamespace(apply_rope_fusion=True, deterministic_mode=False)
    with pytest.raises(ImportError, match="selected fused RoPE"):
        if owner == "mla":
            AbsorbedMLASelfAttention(config, submodules=None, layer_number=1)
        elif owner == "dsv4":
            DSv4HybridSelfAttention(
                config, submodules=None, layer_number=1, pg_collection=_groups()
            )
        elif owner == "compressor":
            Compressor(config, None, compress_ratio=4, head_dim=16, pg_collection=_groups())
        else:
            CSAIndexer(config, None, compress_ratio=4, pg_collection=_groups())


@pytest.mark.parametrize("owner", ["mla", "dsv4"])
def test_disabled_rope_fusion_does_not_import_or_validate_its_kernel(monkeypatch, owner):
    from megatron.core.ops.attention.dsv4 import DSv4HybridSelfAttention
    from megatron.core.ops.attention.mla import AbsorbedMLASelfAttention
    from megatron.core.transformer.attention import Attention

    monkeypatch.setitem(sys.modules, "megatron.core.fusions.fused_mla_yarn_rope_apply", None)
    monkeypatch.setattr(Attention, "__init__", _stop_before_parameters)

    def unexpected(dependency, name):
        pytest.fail(f"Disabled fusion checked {dependency.module} for {name}")

    monkeypatch.setattr(metadata.Dependency, "validate", unexpected)
    config = SimpleNamespace(apply_rope_fusion=False)
    with pytest.raises(_BeforeParameters):
        if owner == "mla":
            AbsorbedMLASelfAttention(config, None, 1, pg_collection=_groups())
        else:
            DSv4HybridSelfAttention(
                config, submodules=None, layer_number=1, pg_collection=_groups()
            )


def test_operation_constructors_do_not_gate_on_kernel_availability_flags():
    import megatron.core.ops

    root = Path(megatron.core.ops.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name != "__init__"
            ):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id.startswith("HAVE_"):
                    # These guard existing TE class checks/GTP integration, not kernel selection.
                    assert child.id in {"HAVE_TE", "HAVE_GTP"}, (path, child.lineno, child.id)
