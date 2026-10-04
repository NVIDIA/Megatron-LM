# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import subprocess
import sys
import textwrap


def test_import_with_transformer_engine_without_fused_mla_support():
    """Verify older Transformer Engine releases can import the MLA module."""
    script = textwrap.dedent("""
        import builtins

        original_import = builtins.__import__

        def import_from_older_te(name, globals=None, locals=None, fromlist=(), level=0):
            importing_module = (globals or {}).get("__name__")
            if importing_module == "megatron.core.transformer.multi_latent_attention":
                if name == "transformer_engine.pytorch.attention" and (
                    "FusedMLAQUpProjFunction" in fromlist
                    or "FusedMLAQUpProjRopeQuant" in fromlist
                ):
                    raise ImportError("fused MLA support is unavailable")
                if name in {
                    "transformer_engine.pytorch.quantized_tensor",
                    "transformer_engine.pytorch.tensor.mxfp8_tensor",
                }:
                    raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_from_older_te

        from megatron.core.transformer import multi_latent_attention

        assert multi_latent_attention.FusedMLAQUpProjFunction is None
        assert multi_latent_attention.FusedMLAQUpProjRopeQuant is None
        assert multi_latent_attention.mxfp8_quantize_only is None
        assert multi_latent_attention.mxfp8_transpose_swizzle is None
        assert multi_latent_attention.QuantizedTensor is None
        assert multi_latent_attention.MXFP8Quantizer is None
        """)

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
