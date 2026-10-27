# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from unittest import mock

import pytest
import torch
from packaging import version
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import FileSystemReader, TensorStorageMetadata
from torch.distributed.checkpoint.planner_helpers import _create_write_items

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.checkpointable import (
    CheckpointableShardedTensor,
    LocalShardsContainer,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    mcore_to_pyt_state_dict,
    sharded_tensor_to_torch_sharded_tensor,
)
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not is_torch_min_version("2.6a0"),
    reason="CheckpointableShardedTensor requires PyTorch 2.6 or later",
)
class TestCheckpointableProtocol:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sharded_tensor_checkpointing(self, tmp_path_dist_ckpt):
        """Test sharded tensor checkpointing with pure DCP."""

        def get_sd(val=3):
            sh_ten = ShardedTensor.from_rank_offsets(
                'b_ten', torch.ones(3) * Utils.rank + val, (0, Utils.rank, Utils.world_size)
            )
            return {'b_ten_sd': CheckpointableShardedTensor.from_sh_ten(sh_ten)}

        state_dict = get_sd(3)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_sharded_objects') as ckpt_dir:
            torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)
            torch.distributed.barrier()

            loaded_state_dict = get_sd(4)
            assert torch.all(loaded_state_dict['b_ten_sd']._sh_ten.data == Utils.rank + 4)
            torch.distributed.checkpoint.load(loaded_state_dict, checkpoint_id=ckpt_dir)
            assert torch.all(loaded_state_dict['b_ten_sd']._sh_ten.data == Utils.rank + 3)

    def test_multiple_local_shards(self, tmp_path_dist_ckpt):
        def get_sd(val=3):
            sh_ten_part_one = ShardedTensor.from_rank_offsets(
                'b_ten', torch.ones(3) * Utils.rank + val, (0, Utils.rank, Utils.world_size * 2)
            )
            sh_ten_part_two = ShardedTensor.from_rank_offsets(
                'b_ten',
                torch.ones(3) * Utils.rank + val,
                (0, Utils.world_size + Utils.rank, Utils.world_size * 2),
            )

            return {
                'b_ten_sd': LocalShardsContainer(
                    [
                        CheckpointableShardedTensor.from_sh_ten(sh_ten_part_one),
                        CheckpointableShardedTensor.from_sh_ten(sh_ten_part_two),
                    ]
                )
            }

        state_dict = get_sd(3)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_sharded_objects') as ckpt_dir:
            torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)
            torch.distributed.barrier()

            metadata = FileSystemReader(ckpt_dir).read_metadata()
            assert isinstance(metadata.state_dict_metadata['b_ten_sd'], TensorStorageMetadata)

            loaded_state_dict = get_sd(4)
            for shard in loaded_state_dict['b_ten_sd']._local_shards:
                assert torch.all(shard._sh_ten.data == Utils.rank + 4)
            torch.distributed.checkpoint.load(loaded_state_dict, checkpoint_id=ckpt_dir)
            for shard in loaded_state_dict['b_ten_sd']._local_shards:
                assert torch.all(shard._sh_ten.data == Utils.rank + 3)


@pytest.mark.skipif(
    not is_torch_min_version("2.6a0"),
    reason="CheckpointableShardedTensor requires PyTorch 2.6 or later",
)
class TestPrependedAxisCheckpointable:
    """Prepended-axis shards (e.g. per-expert MoE weights) must use the checkpointable path.

    Such a shard has a local extent of one along every prepended axis, so DCP metadata can
    describe it directly instead of enumerating the whole global shard grid.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def get_prepended_sh_ten(key='prep_ten', val=3.0):
        """One expert index per rank, exposed through a prepended axis."""
        return ShardedTensor.from_rank_offsets(
            key,
            torch.full((4, 8), val + Utils.rank),
            (0, Utils.rank, Utils.world_size),
            prepend_axis_num=1,
        )

    def test_routing_uses_checkpointable_path(self):
        """Prepended-axis shards are converted into a DCP-checkpointable object."""
        pyt_state_dict = mcore_to_pyt_state_dict({'prep_ten': [self.get_prepended_sh_ten()]})

        assert not isinstance(pyt_state_dict['prep_ten'], TorchShardedTensor)
        assert isinstance(pyt_state_dict['prep_ten'], CheckpointableShardedTensor)

    def test_write_items_match_legacy_conversion(self):
        """The checkpointable description is identical to the legacy PyT ShardedTensor one."""
        # NOTE: the legacy conversion reshapes the shard data in place, so the two
        # conversions must not share ShardedTensor instances.
        legacy_items = _create_write_items(
            'prep_ten',
            sharded_tensor_to_torch_sharded_tensor([self.get_prepended_sh_ten()], Utils.rank),
        )
        new_items = _create_write_items(
            'prep_ten', CheckpointableShardedTensor.from_sh_ten(self.get_prepended_sh_ten())
        )

        assert len(legacy_items) == len(new_items) == 1
        for legacy_item, new_item in zip(legacy_items, new_items):
            # prepended axes are exposed as singleton dimensions of the local chunk
            assert tuple(legacy_item.tensor_data.chunk.sizes) == (1, 4, 8)
            assert tuple(new_item.tensor_data.chunk.sizes) == (1, 4, 8)
            assert legacy_item.tensor_data.chunk.offsets == new_item.tensor_data.chunk.offsets
            assert legacy_item.tensor_data.size == new_item.tensor_data.size
            assert legacy_item.tensor_data.properties.dtype == new_item.tensor_data.properties.dtype

    def test_prepended_axis_roundtrip(self, tmp_path_dist_ckpt):
        """Save and load a prepended-axis shard through the MCore entrypoints."""

        with TempNamedDir(tmp_path_dist_ckpt / 'test_prepended_axis') as ckpt_dir:
            save(
                {'prep_ten_sd': self.get_prepended_sh_ten(val=3)},
                ckpt_dir,
                async_sharded_save=False,
            )

            metadata = FileSystemReader(ckpt_dir).read_metadata()
            stored_metadata = metadata.state_dict_metadata['prep_ten']
            assert tuple(stored_metadata.size) == (Utils.world_size, 4, 8)
            assert (
                sorted(tuple(chunk.sizes) for chunk in stored_metadata.chunks)
                == [(1, 4, 8)] * Utils.world_size
            )

            loaded_sd = load({'prep_ten_sd': self.get_prepended_sh_ten(val=0)}, ckpt_dir)
            assert loaded_sd['prep_ten_sd'].data.shape == (4, 8)
            assert torch.equal(loaded_sd['prep_ten_sd'].data, self.get_prepended_sh_ten(val=3).data)

    def test_prepended_axis_loads_legacy_checkpoint(self, tmp_path_dist_ckpt):
        """Checkpoints written before this change (legacy conversion) must still load."""
        with TempNamedDir(tmp_path_dist_ckpt / 'test_prepended_axis_legacy') as ckpt_dir:
            # Forcing `is_torch_min_version` to False routes every shard through the legacy
            # conversion, which is how prepended-axis shards used to be saved.
            with mock.patch('megatron.core.utils.is_torch_min_version', lambda *args: False):
                save({'prep_ten_sd': self.get_prepended_sh_ten(val=3)}, ckpt_dir)

            loaded_sd = load({'prep_ten_sd': self.get_prepended_sh_ten(val=0)}, ckpt_dir)
            assert torch.equal(loaded_sd['prep_ten_sd'].data, self.get_prepended_sh_ten(val=3).data)
