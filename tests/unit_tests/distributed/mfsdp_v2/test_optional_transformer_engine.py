# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression coverage for MFSDP imports without Transformer Engine."""

import subprocess
import sys
import textwrap


def test_parameter_group_without_transformer_engine():
    """Parameter groups import and preserve native dtypes without TE."""
    # Isolate the simulated dependency failure from the rest of the test suite.
    script = textwrap.dedent("""
        import builtins
        import sys

        import torch
        from megatron.core.distributed.fsdp.src.megatron_fsdp import experimental, utils

        utils.HAVE_TE = False

        # Initialize the parent package before simulating an unavailable dependency
        # so unrelated Megatron/TE imports do not obscure this module's behavior.
        for module in ("parameter_group", "quantized_dbuffer"):
            sys.modules.pop(experimental.__name__ + "." + module, None)
            if hasattr(experimental, module):
                delattr(experimental, module)

        original_import = builtins.__import__

        def import_without_dependency(name, *args, **kwargs):
            if name.startswith("transformer_engine"):
                raise ImportError("Simulated missing dependency: " + name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_dependency

        from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import parameter_group
        assert parameter_group.QuantizedDBuffer is parameter_group.DBuffer
        assert experimental.__name__ + ".quantized_dbuffer" not in sys.modules
        for dtype in (torch.float32, torch.bfloat16, torch.uint8):
            assert parameter_group.effective_dtype(torch.empty(1, dtype=dtype)) == dtype

        """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
