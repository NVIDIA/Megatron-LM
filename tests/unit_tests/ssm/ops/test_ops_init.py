# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Test that the megatron.core.ssm.ops package exports the public API."""

import unittest

try:
    from megatron.core.ssm import ops as ssm_ops
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
class TestOpsPackagePublicAPI(unittest.TestCase):
    """Ensure the ops package exposes the documented public API."""

    def test_all_exported(self):
        self.assertIn("mamba_chunk_scan_combined_varlen", ssm_ops.__all__)

    def test_mamba_chunk_scan_combined_varlen_importable(self):
        self.assertTrue(hasattr(ssm_ops, "mamba_chunk_scan_combined_varlen"))
        self.assertTrue(callable(ssm_ops.mamba_chunk_scan_combined_varlen))


if __name__ == "__main__":
    unittest.main()
