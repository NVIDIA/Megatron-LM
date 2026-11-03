# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run isolated LayerWise ownership tests on a workstation without CUDA.

This does not execute Megatron training or simulate CUDA kernels. Only the
unused, eagerly imported paged-stash kernels are replaced by failing sentinels;
optimizer methods, tensor operations and Gloo collectives remain real.
"""

import os
import sys
import types
from pathlib import Path


def _unavailable(*args, **kwargs):
    raise RuntimeError("CUDA paged-stash kernels are outside the CPU test scope")


def _bootstrap():
    sys.dont_write_bytecode = True
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    # Spawned workers also execute this bootstrap before importing the test module.
    import torch

    if not torch.cuda.is_available():
        name = "megatron.core.transformer.moe.ops.paged_stash"
        stub = types.ModuleType(name)
        stub.GLOBAL_BLOCK_SIZE = 256
        stub.GLOBAL_MAX_BLOCKS = 1
        stub.paged_stash_copy_kernel = _unavailable
        stub.paged_stash_pop_kernel = _unavailable
        sys.modules[name] = stub
    return root


ROOT = _bootstrap()

if __name__ == "__main__":
    import pytest

    # The parent conftest targets CUDA CI and downloads unrelated model datasets.
    # These tests intentionally use no fixtures from it.
    test_dir = ROOT / "tests/unit_tests/distributed"
    targets = [
        str(test_dir / "test_layer_wise_optimizer_instances.py"),
        str(test_dir / "test_layer_wise_param_layout.py"),
    ]
    raise SystemExit(
        pytest.main(
            ["--confcutdir", str(test_dir), "-p", "no:cacheprovider", "-q", *targets, *sys.argv[1:]]
        )
    )
