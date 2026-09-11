# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical operation ownership and removal of the retired implementation packages."""

import ast
import importlib
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

_RETIRED_PACKAGES = (
    "megatron.core.ssm",
    "megatron.core.transformer.experimental_attention_variant",
)

_OPERATION_MODULES = (
    "ops.ssm.mamba2.mixer",
    "ops.ssm.mamba2.context_parallel",
    "ops.ssm.gdp.mixer",
    "ops.ssm.gdp.context_parallel",
    "ops.ssm.gated_delta.common",
    "ops.ssm.gated_delta.gdn",
    "ops.ssm.gated_delta.gdn2",
    "ops.ssm.context_parallel.chunkwise",
    "ops.ssm.context_parallel.gdp",
    "ops.ssm.context_parallel.gdp_common",
    "ops.ssm.context_parallel.gdp_cutedsl",
    "ops.ssm.common.causal_conv1d_cp",
    "ops.ssm.common.packed_seq",
    "ops.ssm.common.checkpointing",
    "ops.attention.dsa.modules",
    "ops.attention.csa.modules",
    "ops.attention.mla",
    "ops.attention.dsv4",
)


@pytest.mark.parametrize("module", _OPERATION_MODULES)
def test_operation_module_has_canonical_source(module):
    path = "megatron.core." + module
    if (
        module == "ops.ssm.context_parallel.gdp_cutedsl"
        and importlib.util.find_spec("gdp_attn") is None
    ):
        with pytest.raises(ImportError, match="CuTeDSL GDP chunkwise CP backend is unavailable"):
            importlib.import_module(path)
        return
    target = importlib.import_module(path)
    assert target.__name__ == path
    assert Path(target.__file__).as_posix().endswith(path.replace(".", "/") + ".py")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("ops.ssm.mamba2.mixer", "MambaMixer"),
        ("ops.ssm.mamba2.mixer", "MambaMixerSubmodules"),
        ("ops.ssm.gdp.mixer", "GatedDeltaProductMixer"),
        ("ops.ssm.gated_delta.gdn", "GatedDeltaNet"),
        ("ops.ssm.gated_delta.gdn2", "GatedDeltaNet2"),
        ("ops.ssm.gated_delta.common", "GatedDeltaNetSubmodules"),
        ("ops.ssm.common.inference", "SSMDynamicInferenceMixin"),
        ("ops.ssm.context_parallel.chunkwise", "PackedSequenceCPMetadata"),
        ("ops.attention.dsa.modules", "DSAttention"),
        ("ops.attention.dsa.modules", "DSAIndexer"),
        ("ops.attention.csa.modules", "Compressor"),
        ("ops.attention.csa.modules", "CompressedSparseAttention"),
        ("ops.attention.mla", "AbsorbedMLASelfAttention"),
        ("ops.attention.dsv4", "DSv4HybridSelfAttention"),
        ("transformer.dsa_loss", "DSAIndexerLossLoggingHelper"),
        ("transformer.dsa_loss", "DSAIndexerLossAutoScaler"),
        ("transformer.mamba_layer", "MambaLayer"),
        ("transformer.mamba_layer", "MambaLayerSubmodules"),
        ("transformer.mlp_layer", "MLPLayer"),
        ("transformer.mamba_layer_config", "MambaLayerConfig"),
        ("transformer.gdn_layer_config", "GDNLayerConfig"),
        ("transformer.mlp_layer_config", "MLPLayerConfig"),
        ("transformer.dsa_layer_config", "DSALayerConfig"),
        ("inference.ssm_config", "SSMChunking"),
    ],
)
def test_class_pickle_records_its_canonical_owner(module, name):
    path = "megatron.core." + module
    target = getattr(importlib.import_module(path), name)
    assert target.__module__ == path
    assert pickle.loads(pickle.dumps(target)) is target


@pytest.mark.parametrize("reverse", [False, True])
def test_construction_import_order_does_not_load_retired_packages(reverse):
    modules = (
        "megatron.core.inference.config",
        "megatron.core.transformer.mamba_layer",
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        "megatron.core.models.hybrid.hybrid_layer_specs",
    )
    code = f"""
import importlib
import sys
for module in {modules[::-1] if reverse else modules!r}:
    importlib.import_module(module)
for retired in {_RETIRED_PACKAGES!r}:
    assert not any(name == retired or name.startswith(retired + '.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_training_loss_state_does_not_import_attention_implementations():
    code = """
import sys
from megatron.core.transformer.dsa_loss import DSAIndexerLossLoggingHelper
assert DSAIndexerLossLoggingHelper.tracker == {}
assert 'megatron.core.ops.attention.dsa.modules' not in sys.modules
assert 'megatron.core.ops.attention.csa.modules' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_gdn_construction_targets_are_canonical():
    from megatron.core.ops.ssm.gated_delta import common, gdn, gdn2, modules

    assert modules.GatedDeltaNet is gdn.GatedDeltaNet
    assert modules.GatedDeltaNet2 is gdn2.GatedDeltaNet2
    assert modules.GatedDeltaNetSubmodules is common.GatedDeltaNetSubmodules


@pytest.mark.parametrize("retired", _RETIRED_PACKAGES)
def test_retired_packages_have_no_source_or_runtime_references(retired):
    root = Path(__file__).resolve().parents[3]
    assert not list((root / retired.replace(".", "/")).rglob("*.py"))
    for directory in ("megatron", "examples", "tools", "tests/functional_tests"):
        for path in (root / directory).rglob("*"):
            if path.suffix == ".py":
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    elif isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    else:
                        continue
                    assert not any(
                        name == retired or name.startswith(retired + ".") for name in names
                    ), path
            elif path.suffix in (".sh", ".yaml", ".yml"):
                assert retired not in path.read_text(), path
