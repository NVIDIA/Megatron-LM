# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared cleanup helpers for MFSDP v2 MCore tests."""

from torch.distributed.tensor.debug import _clear_sharding_prop_cache

from tests.unit_tests.test_utilities import Utils


def destroy_model_parallel():
    """Reset model-parallel state; distributed_setup owns process-group teardown."""
    # TODO(#7322): Remove this workaround when runtime parameters become local tensors.
    # Clear both sharding caches so a mid-test reinitialization binds parameters
    # to the new groups instead of reusing cached meshes through detach().
    _clear_sharding_prop_cache()
    Utils.destroy_model_parallel()
