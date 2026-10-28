# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Test that the deprecated megatron.core.ssm.ops package keeps its historical exports."""

import unittest
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from megatron.core.ssm import ops as ssm_ops

    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
class TestOpsPackagePublicAPI(unittest.TestCase):
    """Ensure the deprecated ops package exposes its historical public API."""

    def test_all_exported(self):
        self.assertIn("mamba_chunk_scan_combined_varlen", ssm_ops.__all__)
        self.assertIn("causal_conv1d_varlen_fn", ssm_ops.__all__)

    def test_mamba_chunk_scan_combined_varlen_importable(self):
        self.assertTrue(hasattr(ssm_ops, "mamba_chunk_scan_combined_varlen"))
        self.assertTrue(callable(ssm_ops.mamba_chunk_scan_combined_varlen))

    def test_causal_conv1d_varlen_fn_importable(self):
        self.assertTrue(hasattr(ssm_ops, "causal_conv1d_varlen_fn"))
        self.assertTrue(callable(ssm_ops.causal_conv1d_varlen_fn))


if __name__ == "__main__":
    unittest.main()
