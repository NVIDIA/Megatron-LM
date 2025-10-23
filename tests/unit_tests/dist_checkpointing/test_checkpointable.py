# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch
from packaging import version
from torch.distributed.checkpoint import FileSystemReader, TensorStorageMetadata

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies.checkpointable import (
    CheckpointableShardedTensor,
    LocalShardsContainer,
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
