# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for torch-version gating in the torch_dcp checkpoint strategy.

Covers the `mcore_to_pyt_state_dict` uneven-sharding branch, which is gated on
`is_torch_min_version("2.6a0")` and reports the resolved torch version via
`get_torch_version()` in the error message. Both helpers are imported
function-locally from `megatron.core.utils`, so both sides of the version
condition are exercised by patching `megatron.core.utils._torch_version`
(the module global both helpers read) without requiring multiple PyTorch
installations.
"""

import tempfile
from contextlib import contextmanager

import pytest
import torch
from packaging.version import Version as PkgVersion

import megatron.core.utils as core_utils
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.strategies.checkpointable import CheckpointableShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import mcore_to_pyt_state_dict


@contextmanager
def single_rank_process_group():
    """Provide a process group for `torch.distributed.get_rank()`.

    Reuses the process group if one is already initialized (e.g. when the suite
    runs under torchrun); otherwise initializes a hermetic 1-rank gloo group.
    """
    if torch.distributed.is_initialized():
        yield
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        torch.distributed.init_process_group(
            backend='gloo', rank=0, world_size=1, init_method=f'file://{tmp_dir}/rendezvous'
        )
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()


def _uneven_sharded_tensor() -> ShardedTensor:
    """A ShardedTensor without a regular sharding grid (axis_fragmentations=None).

    This is the sharding layout produced e.g. by the distributed optimizer state
    and the only layout that reaches the uneven-sharding version gate.
    """
    return ShardedTensor(
        'mdp.uneven_key',
        torch.zeros(4),
        torch.float32,
        local_shape=(4,),
        global_shape=(8,),
        global_offset=(0,),
        axis_fragmentations=None,
    )


@pytest.mark.parametrize('old_torch_version', ['2.2.0', '2.4.0'])
def test_uneven_sharding_error_message_on_pre_2_6_torch(monkeypatch, old_torch_version):
    """Loading uneven sharding on torch < 2.6a0 must raise CheckpointingException
    whose message reports the torch version.

    Regression test: the error path referenced `get_torch_version()` that was no
    longer imported into the module, so instead of this error a `NameError` was
    raised, masking the real checkpointing problem.
    """
    monkeypatch.setattr(core_utils, '_torch_version', PkgVersion(old_torch_version))

    with single_rank_process_group():
        with pytest.raises(
            CheckpointingException, match='Uneven sharding not supported'
        ) as exc_info:
            mcore_to_pyt_state_dict({'mdp': [_uneven_sharded_tensor()]})

        # The raised error must carry the resolved torch version in its message.
        assert old_torch_version in str(exc_info.value)


def test_uneven_sharding_uses_checkpointable_tensor_on_min_2_6_torch(monkeypatch):
    """On torch >= 2.6a0 uneven sharding is supported via CheckpointableShardedTensor
    and must not raise (nor report the version gate error)."""
    monkeypatch.setattr(core_utils, '_torch_version', PkgVersion('2.6.0'))

    with single_rank_process_group():
        pyt_state_dict = mcore_to_pyt_state_dict({'mdp': [_uneven_sharded_tensor()]})
        assert isinstance(pyt_state_dict['mdp'], CheckpointableShardedTensor)
