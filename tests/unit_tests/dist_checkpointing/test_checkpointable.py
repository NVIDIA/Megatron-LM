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

    @staticmethod
    def _expert_shards(val, n_experts=4, egtp=None):
        """Grouped-expert layout: expert axis prepended, weight axis 0 split across ranks (EGTP)."""
        egtp = Utils.world_size if egtp is None else egtp
        sh_tens = []
        for e in range(n_experts):
            data = (
                torch.arange(6, dtype=torch.float32).view(2, 3) + 100 * e + 1000 * Utils.rank + val
            )
            sh_tens.append(
                ShardedTensor.from_rank_offsets(
                    'experts.weight',
                    data,
                    (0, e, n_experts),
                    (1, Utils.rank % egtp, egtp),
                    replica_id=Utils.rank // egtp,
                    prepend_axis_num=1,
                )
            )
        return sh_tens

    def test_prepended_axis_write_items_match_legacy_translation(self):
        """Chunk metadata of the Checkpointable path equals the torch ShardedTensor translation."""
        from torch.distributed.checkpoint.planner_helpers import (
            _create_chunk_list,
            _create_write_items,
        )

        from megatron.core.dist_checkpointing.strategies.torch import (
            mcore_to_pyt_state_dict,
            sharded_tensor_to_torch_sharded_tensor,
        )

        legacy = sharded_tensor_to_torch_sharded_tensor(self._expert_shards(0), Utils.rank)
        fixed = mcore_to_pyt_state_dict({'experts.weight': self._expert_shards(0)}, False)[
            'experts.weight'
        ]
        assert isinstance(fixed, LocalShardsContainer)

        def view(w):
            td = w.tensor_data
            return (
                w.index.fqn,
                tuple(w.index.offset),
                tuple(td.chunk.offsets),
                tuple(td.chunk.sizes),
                td.properties,
                tuple(td.size),
            )

        assert [view(w) for w in _create_write_items('experts.weight', legacy)] == [
            view(w) for w in _create_write_items('experts.weight', fixed)
        ]
        assert [(tuple(c.offsets), tuple(c.sizes)) for c in _create_chunk_list(legacy)] == [
            (tuple(c.offsets), tuple(c.sizes)) for c in _create_chunk_list(fixed)
        ]
        # the tensor DCP writes has the chunk shape and aliases the MCore data
        for shard in fixed._local_shards:
            t = shard.__get_tensor_shard__(None)
            assert tuple(t.shape) == (1, 2, 3)
            assert t.untyped_storage().data_ptr() == shard._sh_ten.data.untyped_storage().data_ptr()

    def test_prepended_axis_checkpointing(self, tmp_path_dist_ckpt):
        """Save and load grouped-expert shards (prepended axis) through the Checkpointable path."""
        from megatron.core.dist_checkpointing.strategies.torch import mcore_to_pyt_state_dict

        def get_sd(val):
            return mcore_to_pyt_state_dict({'experts.weight': self._expert_shards(val)}, False)

        with TempNamedDir(tmp_path_dist_ckpt / 'test_prepended_axis') as ckpt_dir:
            state_dict = get_sd(3)
            torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)
            torch.distributed.barrier()

            metadata = FileSystemReader(ckpt_dir).read_metadata()
            md = metadata.state_dict_metadata['experts.weight']
            assert isinstance(md, TensorStorageMetadata)
            assert tuple(md.size) == (4, 2 * Utils.world_size, 3)
            assert all(tuple(c.sizes) == (1, 2, 3) for c in md.chunks)
            assert len(md.chunks) == 4 * Utils.world_size

            loaded_state_dict = get_sd(4)
            torch.distributed.checkpoint.load(loaded_state_dict, checkpoint_id=ckpt_dir)
            expected = self._expert_shards(3)
            for shard, exp in zip(loaded_state_dict['experts.weight']._local_shards, expected):
                assert shard._sh_ten.data.shape == (2, 3)
                assert torch.equal(shard._sh_ten.data, exp.data)

    def test_prepended_axis_interoperable_with_legacy_translation(self, tmp_path_dist_ckpt):
        """A checkpoint written through the torch ShardedTensor translation loads through the
        Checkpointable path and vice versa (same chunk layout)."""
        from megatron.core.dist_checkpointing.strategies.torch import (
            mcore_to_pyt_state_dict,
            sharded_tensor_to_torch_sharded_tensor,
        )

        def legacy_sd(val):
            return {
                'experts.weight': sharded_tensor_to_torch_sharded_tensor(
                    self._expert_shards(val), Utils.rank
                )
            }

        def fixed_sd(val):
            return mcore_to_pyt_state_dict({'experts.weight': self._expert_shards(val)}, False)

        for writer, reader, name in (
            (legacy_sd, fixed_sd, 'legacy_to_fixed'),
            (fixed_sd, legacy_sd, 'fixed_to_legacy'),
        ):
            with TempNamedDir(tmp_path_dist_ckpt / f'test_interop_{name}') as ckpt_dir:
                torch.distributed.checkpoint.save(writer(3), checkpoint_id=ckpt_dir)
                torch.distributed.barrier()
                loaded = reader(4)
                torch.distributed.checkpoint.load(loaded, checkpoint_id=ckpt_dir)
                expected = self._expert_shards(3)
                loaded_ten = loaded['experts.weight']
                if isinstance(loaded_ten, LocalShardsContainer):
                    got = [s._sh_ten.data for s in loaded_ten._local_shards]
                else:
                    got = [s.tensor.view(2, 3) for s in loaded_ten.local_shards()]
                for g, exp in zip(got, expected):
                    assert torch.equal(g, exp.data), name
