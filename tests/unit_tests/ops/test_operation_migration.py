# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Operation ownership and compatibility across the implementation relocation."""

import importlib
import importlib.util
import pickle
import subprocess
import sys

import pytest

_MODULE_MOVES = (
    ("ssm.mamba_mixer", "ops.ssm.mamba2.mixer"),
    ("ssm.mamba_context_parallel", "ops.ssm.mamba2.context_parallel"),
    ("ssm.gated_delta_product", "ops.ssm.gdp.mixer"),
    ("ssm.gdp_context_parallel", "ops.ssm.gdp.context_parallel"),
    ("ssm.gated_delta_net.common", "ops.ssm.gated_delta.common"),
    ("ssm.gated_delta_net.gdn", "ops.ssm.gated_delta.gdn"),
    ("ssm.gated_delta_net.gdn2", "ops.ssm.gated_delta.gdn2"),
    ("ssm.context_parallel.chunkwise", "ops.ssm.context_parallel.chunkwise"),
    ("ssm.context_parallel.gdp", "ops.ssm.context_parallel.gdp"),
    ("ssm.context_parallel.gdp_common", "ops.ssm.context_parallel.gdp_common"),
    ("ssm.context_parallel.gdp_cutedsl", "ops.ssm.context_parallel.gdp_cutedsl"),
    ("ssm.causal_conv1d", "ops.ssm.common.causal_conv1d_cp"),
    ("ssm.packed_seq_helpers", "ops.ssm.common.packed_seq"),
    ("ssm.utils", "ops.ssm.common.checkpointing"),
    ("transformer.experimental_attention_variant.dsa", "ops.attention.dsa.modules"),
    ("transformer.experimental_attention_variant.csa", "ops.attention.csa.modules"),
    ("transformer.experimental_attention_variant.absorbed_mla", "ops.attention.mla"),
    (
        "transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
        "ops.attention.dsv4",
    ),
)


@pytest.mark.parametrize(("old", "new"), _MODULE_MOVES)
def test_operation_legacy_module_preserves_import_and_global_behavior(monkeypatch, old, new):
    if (
        new == "ops.ssm.context_parallel.gdp_cutedsl"
        and importlib.util.find_spec("gdp_attn") is None
    ):
        for path in (old, new):
            with pytest.raises(
                ImportError, match="CuTeDSL GDP chunkwise CP backend is unavailable"
            ):
                importlib.import_module("megatron.core." + path)
        return
    legacy = importlib.import_module("megatron.core." + old)
    canonical = importlib.import_module("megatron.core." + new)
    assert legacy is canonical
    probe = object()
    monkeypatch.setattr(legacy, "_migration_identity_probe", probe, raising=False)
    assert canonical._migration_identity_probe is probe


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("ssm.mamba_mixer", "ops.ssm.mamba2.mixer", "MambaMixer"),
        ("ssm.mamba_mixer", "ops.ssm.mamba2.mixer", "MambaMixerSubmodules"),
        ("ssm.gated_delta_product", "ops.ssm.gdp.mixer", "GatedDeltaProductMixer"),
        ("ssm.gated_delta_net", "ops.ssm.gated_delta.gdn", "GatedDeltaNet"),
        ("ssm.gated_delta_net", "ops.ssm.gated_delta.gdn2", "GatedDeltaNet2"),
        ("ssm.gated_delta_net", "ops.ssm.gated_delta.common", "GatedDeltaNetSubmodules"),
        ("ssm.ssm_inference", "ops.ssm.common.inference", "SSMDynamicInferenceMixin"),
        (
            "ssm.context_parallel.chunkwise",
            "ops.ssm.context_parallel.chunkwise",
            "PackedSequenceCPMetadata",
        ),
        (
            "transformer.experimental_attention_variant.dsa",
            "ops.attention.dsa.modules",
            "DSAttention",
        ),
        (
            "transformer.experimental_attention_variant.dsa",
            "ops.attention.dsa.modules",
            "DSAIndexer",
        ),
        (
            "transformer.experimental_attention_variant.csa",
            "ops.attention.csa.modules",
            "Compressor",
        ),
        (
            "transformer.experimental_attention_variant.csa",
            "ops.attention.csa.modules",
            "CompressedSparseAttention",
        ),
        (
            "transformer.experimental_attention_variant.absorbed_mla",
            "ops.attention.mla",
            "AbsorbedMLASelfAttention",
        ),
        (
            "transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
            "ops.attention.dsv4",
            "DSv4HybridSelfAttention",
        ),
        (
            "transformer.experimental_attention_variant.dsa",
            "transformer.dsa_loss",
            "DSAIndexerLossLoggingHelper",
        ),
        (
            "transformer.experimental_attention_variant.dsa",
            "transformer.dsa_loss",
            "DSAIndexerLossAutoScaler",
        ),
    ],
)
def test_operation_class_and_historical_pickle_globals_resolve(old, new, name):
    old_path = "megatron.core." + old
    new_path = "megatron.core." + new
    target = getattr(importlib.import_module(new_path), name)
    assert getattr(importlib.import_module(old_path), name) is target
    assert target.__module__ == new_path
    # Protocol 0 GLOBAL records the original import path without constructing a model.
    historical_global = f"c{old_path}\n{name}\n.".encode("ascii")
    assert pickle.loads(historical_global) is target
    assert pickle.loads(pickle.dumps(target)) is target


@pytest.mark.parametrize("legacy_first", [False, True])
def test_operation_alias_import_order_does_not_duplicate_classes(legacy_first):
    pairs = [
        ("ssm.gated_delta_net.gdn", "ops.ssm.gated_delta.gdn"),
        ("ssm.context_parallel.chunkwise", "ops.ssm.context_parallel.chunkwise"),
        ("transformer.experimental_attention_variant.dsa", "ops.attention.dsa.modules"),
    ]
    code = f"""
import importlib
for old, new in {pairs!r}:
    paths = (old, new) if {legacy_first!r} else (new, old)
    first, second = [importlib.import_module('megatron.core.' + path) for path in paths]
    assert first is second, paths
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_training_loss_state_is_shared_without_importing_attention_implementations():
    code = """
import sys
from megatron.core.transformer.dsa_loss import DSAIndexerLossLoggingHelper
assert DSAIndexerLossLoggingHelper.tracker == {}
assert 'megatron.core.ops.attention.dsa.modules' not in sys.modules
assert 'megatron.core.ops.attention.csa.modules' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_gdn_compatibility_package_exports_canonical_implementations():
    legacy = importlib.import_module("megatron.core.ssm.gated_delta_net")
    canonical = importlib.import_module("megatron.core.ops.ssm.gated_delta.modules")
    assert set(legacy.__all__) == set(canonical.__all__)
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(canonical, name)


def test_legacy_ssm_kernel_namespace_keeps_public_exports():
    legacy = importlib.import_module("megatron.core.ssm.ops")
    canonical = importlib.import_module("megatron.core.ops.ssm")
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(canonical, name)
