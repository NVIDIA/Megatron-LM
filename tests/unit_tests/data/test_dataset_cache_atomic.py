# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy

try:
    from megatron.core.datasets.utils import atomic_save, atomic_write_text
except Exception:
    # Fallback to direct import when running outside the full GPU/CUDA/TE container environment
    import sys
    import types

    if "megatron" not in sys.modules:
        sys.modules["megatron"] = types.ModuleType("megatron")
    if "megatron.core" not in sys.modules:
        sys.modules["megatron.core"] = types.ModuleType("megatron.core")
    if "megatron.core.utils" not in sys.modules:
        mutils = types.ModuleType("megatron.core.utils")
        mutils.log_single_rank = lambda *args, **kwargs: None
        sys.modules["megatron.core.utils"] = mutils

    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../megatron/core/datasets/utils.py")
    )
    spec = importlib.util.spec_from_file_location("megatron.core.datasets.utils", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    atomic_save = mod.atomic_save
    atomic_write_text = mod.atomic_write_text


class TestDatasetCacheAtomic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_atomic_save_success(self):
        target_path = os.path.join(self.dir_path, "sample_index.npy")
        arr = numpy.arange(100, dtype=numpy.int64)

        atomic_save(target_path, arr)

        self.assertTrue(os.path.isfile(target_path))
        loaded = numpy.load(target_path, mmap_mode="r")
        numpy.testing.assert_array_equal(loaded, arr)

        # Ensure no temporary files remain
        remaining_files = os.listdir(self.dir_path)
        self.assertEqual(remaining_files, ["sample_index.npy"])

    def test_atomic_save_failure_leaves_no_corrupted_target(self):
        target_path = os.path.join(self.dir_path, "sample_index.npy")
        arr = numpy.arange(100, dtype=numpy.int64)

        def failing_save(*args, **kwargs):
            raise IOError("Simulated disk error or crash")

        with patch("numpy.save", side_effect=failing_save):
            with self.assertRaises(IOError):
                atomic_save(target_path, arr)

        # Target file must NOT exist, and no orphaned temp files should remain
        self.assertFalse(os.path.isfile(target_path))
        self.assertEqual(os.listdir(self.dir_path), [])

    def test_atomic_write_text(self):
        target_path = os.path.join(self.dir_path, "description.txt")
        content = "unique_description_string_123"

        atomic_write_text(target_path, content)

        self.assertTrue(os.path.isfile(target_path))
        with open(target_path, "r", encoding="utf-8") as f:
            read_content = f.read()
        self.assertEqual(read_content, content)

        remaining_files = os.listdir(self.dir_path)
        self.assertEqual(remaining_files, ["description.txt"])


if __name__ == "__main__":
    unittest.main()
