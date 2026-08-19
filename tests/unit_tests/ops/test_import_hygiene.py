# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Importing the ops package must not pull in a backend nobody selected."""

import importlib
import subprocess
import sys
import textwrap

import pytest

_MIGRATED_MODULES = (
    "megatron.core.ops",
    "megatron.core.models.gpt.gpt_layer_specs",
    "megatron.core.models.gpt.moe_module_specs",
    "megatron.core.models.gpt.experimental_attention_variant_module_specs",
    "megatron.core.transformer.multi_token_prediction",
    "megatron.core.models.common.language_module.language_module",
)


def test_importing_ops_does_not_import_an_optional_backend():
    """Measured against a bare ``import megatron.core``, so this only blames core.ops.

    Runs in a fresh interpreter because sys.modules here is already populated by other tests.
    """
    script = textwrap.dedent("""
        import sys

        optional = {"transformer_engine", "apex", "nvidia_kitchen"}
        import megatron.core

        before = {name for name in optional if name in sys.modules}
        import megatron.core.ops

        after = {name for name in optional if name in sys.modules}
        print(",".join(sorted(after - before)))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    added = result.stdout.strip()
    assert added == "", f"importing megatron.core.ops additionally imported: {added}"


@pytest.mark.parametrize("module_name", _MIGRATED_MODULES)
def test_migrated_modules_define_no_availability_flags(module_name):
    """Backends state their requirements in core.ops instead of exporting a HAVE_* global."""
    module = importlib.import_module(module_name)
    flags = sorted(name for name in vars(module) if name.startswith(("HAVE_", "_HAVE_")))
    assert not flags, f"{module_name} still defines {flags}"
