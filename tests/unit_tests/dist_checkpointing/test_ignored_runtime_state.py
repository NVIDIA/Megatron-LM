# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.training.checkpointing import _ignored_runtime_state_template
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestIgnoredRuntimeState:
    def setup_method(self, method):
        Utils.initialize_distributed()

    @pytest.mark.parametrize("key", ["rerun_state_machine_state", "rng_state"])
    def test_strict_load_consumes_runtime_object(self, tmp_path_dist_ckpt, key):
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        model = ShardedTensor.from_rank_offsets(
            "model.weight", torch.arange(4, dtype=torch.float32), replica_id=rank
        )
        runtime = ShardedObject(key, {"owner": rank}, (world,), (rank,))
        with TempNamedDir(tmp_path_dist_ckpt / key) as checkpoint:
            dist_checkpointing.save({"model": model, "runtime": runtime}, checkpoint)
            # This reproduces strict resume rejecting an omitted rerun sentinel.
            with pytest.raises(CheckpointingException, match=key):
                dist_checkpointing.load({"model": model}, checkpoint, strict="raise_all")
            template = _ignored_runtime_state_template(checkpoint, {key})
            loaded = dist_checkpointing.load(
                {"model": model, "_ignored_runtime_state": template}, checkpoint, strict="raise_all"
            )
            torch.testing.assert_close(loaded["model"], torch.arange(4, dtype=torch.float32))
            assert len(loaded["_ignored_runtime_state"]) == 1
            assert set(loaded["_ignored_runtime_state"]) == set(template)
            # The runtime allowance must not hide a missing model parameter.
            with pytest.raises(CheckpointingException, match="model.weight"):
                dist_checkpointing.load(
                    {"_ignored_runtime_state": template}, checkpoint, strict="raise_all"
                )

    @pytest.mark.parametrize("rank", [0, 1, 2, 3])
    def test_saved_layout_is_preserved_when_world_size_changes(self, rank):
        metadata = {
            str(i): ShardedObject("rng_state", None, (1, 1, 8), (0, 0, i)) for i in range(8)
        }
        metadata["model"] = ShardedObject("model.extra_state", None, (1,), (0,))
        with (
            mock.patch(
                "megatron.training.checkpointing.load_sharded_metadata", return_value=metadata
            ),
            mock.patch.object(torch.distributed, "get_rank", return_value=rank),
            mock.patch.object(torch.distributed, "get_world_size", return_value=4),
        ):
            template = _ignored_runtime_state_template("checkpoint", {"rng_state"})
        assert set(template) == {str(rank), str(rank + 4)}
        assert all(value.global_shape == (1, 1, 8) for value in template.values())
        assert all(value.key == "rng_state" for value in template.values())

    def test_no_ignored_keys_does_not_read_metadata(self):
        with mock.patch("megatron.training.checkpointing.load_sharded_metadata") as read:
            assert _ignored_runtime_state_template("checkpoint", set()) == {}
        read.assert_not_called()
