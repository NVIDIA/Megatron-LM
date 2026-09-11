# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ownership, compatibility and construction-time kernel selection contracts."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider, backend_slot
from megatron.core.ops.attention.dsa import backends as dsa_backends
from megatron.core.ops.attention.dsa.backends import DSAKernels, select_dsa_kernels
from megatron.core.ops.ssm.gated_delta.backends import select_gated_delta_rule


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("ssm.ops.common.causal_conv1d_varlen", "ops.ssm.common.causal_conv1d_varlen"),
        ("ssm.ops.mamba2.ssd_combined", "ops.ssm.mamba2.ssd_combined"),
        ("ssm.ops.gdp.decode_prepare", "ops.ssm.gdp.decode_prepare"),
        ("ssm.triton_cache_manager", "ops.ssm.triton_cache_manager"),
        *[
            (f"transformer.experimental_attention_variant.{name}", f"ops.attention.dsa.{name}")
            for name in ("dsa_kernels", "dsa_layout", "dsa_masking", "dsa_indexer_loss")
        ],
    ],
)
def test_legacy_leaf_is_canonical_module(old, new):
    canonical = importlib.import_module("megatron.core." + new)
    legacy = importlib.import_module("megatron.core." + old)
    assert legacy is canonical


@pytest.mark.parametrize(
    ("old", "new", "symbol"),
    [
        (
            "transformer.experimental_attention_variant.csa",
            "ops.attention.csa.reference",
            "_pool_compressor_values",
        ),
        (
            "ssm.gated_delta_net.gdn",
            "ops.ssm.gated_delta.reference",
            "torch_chunk_gated_delta_rule",
        ),
        ("ssm.gated_delta_net.gdn2", "ops.ssm.gated_delta.reference_gdn2", "torch_chunk_gdn2"),
        (
            "transformer.experimental_attention_variant.dsa",
            "ops.attention.dsa.reference",
            "unfused_dsa_fn",
        ),
        (
            "transformer.experimental_attention_variant.dsa",
            "ops.attention.dsa.reference",
            "FusedDSAIndexerLoss",
        ),
        (
            "transformer.experimental_attention_variant.csa",
            "ops.attention.csa.reference",
            "get_window_topk_idxs",
        ),
        (
            "transformer.experimental_attention_variant.csa",
            "ops.attention.csa.reference",
            "unfused_compressed_sparse_attn",
        ),
    ],
)
def test_reference_reexports_keep_identity(old, new, symbol):
    assert getattr(importlib.import_module("megatron.core." + old), symbol) is getattr(
        importlib.import_module("megatron.core." + new), symbol
    )


def test_family_namespaces_do_not_load_optional_dependencies():
    code = """
import importlib
import sys
import megatron.core
before = set(sys.modules)
for family in ('', '.ssm', '.ssm.common', '.ssm.mamba2', '.ssm.gated_delta', '.ssm.gdp',
               '.attention', '.attention.dsa', '.attention.csa'):
    importlib.import_module('megatron.core.ops' + family)
optional = {'triton', 'tilelang', 'fla', 'mamba_ssm', 'causal_conv1d',
            'fast_hadamard_transform', 'cudnn', 'gdp_attn'}
assert not {name for name in set(sys.modules) - before if name.split('.')[0] in optional}
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_ops_do_not_import_model_owners():
    import megatron.core.ops

    root = Path(megatron.core.ops.__file__).parent
    forbidden = (
        "megatron.core.ssm",
        "megatron.core.models",
        "megatron.core.transformer.experimental_attention_variant",
    )
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(name.startswith(forbidden) for name in names), path


def test_dsa_binds_direct_hooks_once_and_keeps_instances_independent(monkeypatch):
    first_hook = lambda **kwargs: kwargs
    second_hook = lambda **kwargs: None
    modules = {
        "tilelang": SimpleNamespace(run_fused_qk_topk=first_hook),
        "cudnn": SimpleNamespace(run_fused_qk_topk=second_hook),
    }
    calls = []

    def load(module_name):
        name = "tilelang" if "tilelang" in module_name else "cudnn"
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(dsa_backends, "import_module", load)
    config = SimpleNamespace(attention_backend="auto", dsa_kernel_backend="tilelang")
    first = select_dsa_kernels(config)
    config.dsa_kernel_backend = "cudnn"
    second = select_dsa_kernels(config)
    assert first.run_fused_qk_topk is first_hook
    assert second.run_fused_qk_topk is second_hook
    assert first.run_fused_qk_topk(q=1) == {"q": 1}
    assert second.run_fused_qk_topk(q=1) is None
    assert first.backend == "tilelang"
    assert first.run_fused_dsa_attention is None
    assert calls == ["tilelang", "cudnn"]


@pytest.mark.parametrize(("attention", "kernel"), [("unfused", "cudnn"), ("auto", "none")])
def test_disabled_dsa_does_not_import_backend(monkeypatch, attention, kernel):
    def unexpected(_config):
        pytest.fail("disabled fused DSA must not import a backend")

    monkeypatch.setattr(dsa_backends, "import_module", unexpected)
    config = SimpleNamespace(attention_backend=attention, dsa_kernel_backend=kernel)
    assert select_dsa_kernels(config) == DSAKernels()


def test_invalid_dsa_backend_is_rejected():
    with pytest.raises(ValueError, match="dsa_kernel_backend"):
        select_dsa_kernels(SimpleNamespace(attention_backend="auto", dsa_kernel_backend="invalid"))


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_missing_selected_dsa_backend_fails_at_construction(monkeypatch, error):
    def fail_import(_name):
        raise error("missing extension")

    monkeypatch.setattr(dsa_backends, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="Failed to import DSA kernel backend"):
        select_dsa_kernels(SimpleNamespace(attention_backend="auto", dsa_kernel_backend="cudnn"))


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_gated_delta_reference_and_missing_selected_kernel(monkeypatch, variant):
    from megatron.core.ops.ssm.gated_delta import fla
    from megatron.core.ops.ssm.gated_delta.reference import torch_chunk_gated_delta_rule
    from megatron.core.ops.ssm.gated_delta.reference_gdn2 import torch_chunk_gdn2

    reference = torch_chunk_gated_delta_rule if variant == "gdn" else torch_chunk_gdn2
    assert select_gated_delta_rule(variant, deterministic=True) is reference
    name = "chunk_gated_delta_rule" if variant == "gdn" else "chunk_gdn2"
    monkeypatch.setattr(fla, name, None)
    with pytest.raises(ImportError, match="flash-linear-attention"):
        select_gated_delta_rule(variant)


@pytest.mark.parametrize("slot", ["dsa_kernels", "gated_delta_rule", "gated_delta_product"])
def test_older_providers_use_family_defaults(slot):
    # Protocol methods may be inherited as stubs or absent on structural providers.
    inherited = SimpleNamespace()
    setattr(inherited, slot, MethodType(getattr(BackendSpecProvider, slot), inherited))
    for provider in (SimpleNamespace(), inherited):
        sentinel = object()
        assert backend_slot(provider, slot, default=lambda: sentinel) is sentinel


def test_local_and_te_preserve_gated_delta_defaults():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    for variant in ("gdn", "gdn2"):
        assert LocalSpecProvider().gated_delta_rule(variant, deterministic=True) is (
            TESpecProvider().gated_delta_rule(variant, deterministic=True)
        )


def test_specs_preserve_the_explicit_kernel_provider():
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsa_module_spec_for_backend,
        get_gated_delta_net_module_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, multi_latent_attention=True
    )
    provider = TESpecProvider()
    dsa = get_dsa_module_spec_for_backend(config, provider)
    assert dsa.submodules.core_attention.params["kernel_backend"] is provider
    gdn = get_gated_delta_net_module_spec(config, provider)
    assert gdn.params["kernel_backend"] is provider


def test_dsa_construction_uses_explicit_provider_without_rebuilding_it():
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAttention,
        DSAttentionSubmodules,
    )

    kernels = DSAKernels(backend="custom")
    seen = []

    class CustomProvider:
        def dsa_kernels(self, config):
            seen.append(config)
            return kernels

    config = SimpleNamespace(
        dsa_indexer_topk=8, dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=1, kv_channels=16
    )
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
        pg_collection=SimpleNamespace(),
        kernel_backend=CustomProvider(),
    )
    assert attention.dsa_kernels is kernels
    assert seen == [config]
