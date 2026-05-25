# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GTP unit tests — launched torchrun-native (same as the rest of Megatron's unit tests).

    export TE_PATH=/path/to/TransformerEngine
    export PYTHONPATH="${TE_PATH}:${PYTHONPATH}"
    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/generalized_tensor_parallel/ -v

Tests use the torchrun-managed dist group (initialized once per module via
``Utils.initialize_model_parallel``) and build their own GTP subgroups with
``dist.new_group(...)``. Multi-GPU tests skip when the world_size requested by a
test doesn't match what torchrun launched with (all GTP multi-GPU tests need 4).
"""
