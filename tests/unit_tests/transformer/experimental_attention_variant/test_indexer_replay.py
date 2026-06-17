# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.indexer_replay import (
    IndexerReplay,
    IndexerReplayAction,
)


def setup_function():
    IndexerReplay.global_indexer_replay_instances.clear()


def teardown_function():
    IndexerReplay.global_indexer_replay_instances.clear()


class TestIndexerReplayUnit:
    """Unit tests for IndexerReplay class — no CUDA required."""

    def test_constructor_and_registration(self):
        ir = IndexerReplay()
        assert ir.target_topk_idx is None
        assert ir.recorded_topk_idx is None
        assert ir.indexer_replay_action is None
        assert ir.replay_backward_list == []
        assert ir.static_buffer is None
        assert len(IndexerReplay.global_indexer_replay_instances) == 1
        assert IndexerReplay.global_indexer_replay_instances[0] is ir

    def test_set_and_clear_action(self):
        ir = IndexerReplay()
        ir.set_indexer_replay_action(IndexerReplayAction.RECORD)
        assert ir.indexer_replay_action == IndexerReplayAction.RECORD
        ir.set_indexer_replay_action(IndexerReplayAction.REPLAY_FORWARD)
        assert ir.indexer_replay_action == IndexerReplayAction.REPLAY_FORWARD
        ir.set_indexer_replay_action(IndexerReplayAction.REPLAY_BACKWARD)
        assert ir.indexer_replay_action == IndexerReplayAction.REPLAY_BACKWARD
        ir.clear_indexer_replay_action()
        assert ir.indexer_replay_action is None

    def test_record_indices_eager(self):
        ir = IndexerReplay()
        indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ir.record_indices(indices)
        assert ir.recorded_topk_idx is indices

    def test_record_indices_static_buffer(self):
        ir = IndexerReplay()
        buffer = torch.zeros(10, 2, dtype=torch.int32)
        ir.set_static_buffer(buffer)
        indices = torch.tensor([[5, 6], [7, 8], [9, 0]], dtype=torch.int32)
        ir.record_indices(indices)
        assert torch.equal(buffer[:3], indices)
        assert ir.recorded_topk_idx is buffer[:3]

    def test_set_target_indices(self):
        ir = IndexerReplay()
        indices = torch.tensor([[0, 1]], dtype=torch.long)
        ir.set_target_indices(indices)
        assert ir.target_topk_idx is indices
        assert len(ir.replay_backward_list) == 1
        assert ir.replay_backward_list[0] is indices

    def test_clear_indices(self):
        ir = IndexerReplay()
        ir.record_indices(torch.tensor([[0, 1]], dtype=torch.long))
        ir.set_target_indices(torch.tensor([[1, 0]], dtype=torch.long))
        ir.clear_indices()
        assert ir.recorded_topk_idx is None
        assert ir.target_topk_idx is None
        assert ir.replay_backward_list == []

    def test_get_recorded_indices(self):
        ir = IndexerReplay()
        assert ir.get_recorded_indices() is None
        indices = torch.tensor([[0, 1]], dtype=torch.long)
        ir.record_indices(indices)
        assert ir.get_recorded_indices() is indices

    def test_get_replay_topk_indices_record(self):
        ir = IndexerReplay()
        ir.set_indexer_replay_action(IndexerReplayAction.RECORD)
        computed = torch.tensor([[5, 6]], dtype=torch.long)

        def compute():
            return computed

        result, replay = ir.get_replay_topk_indices(compute)
        assert result is computed
        assert replay is None
        assert ir.recorded_topk_idx is computed

    def test_get_replay_topk_indices_replay_forward(self):
        ir = IndexerReplay()
        target = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ir.set_target_indices(target)
        ir.set_indexer_replay_action(IndexerReplayAction.REPLAY_FORWARD)

        result, replay = ir.get_replay_topk_indices(lambda: None)
        assert result is target
        assert replay is target

    def test_get_replay_topk_indices_replay_backward(self):
        ir = IndexerReplay()
        target = torch.tensor([[4, 5]], dtype=torch.long)
        ir.set_target_indices(target)
        ir.set_indexer_replay_action(IndexerReplayAction.REPLAY_BACKWARD)

        result, replay = ir.get_replay_topk_indices(lambda: None)
        assert result is target
        assert replay is target
        # backward list should now be empty
        assert ir.replay_backward_list == []

    def test_get_replay_topk_indices_none_action(self):
        ir = IndexerReplay()
        result, replay = ir.get_replay_topk_indices(lambda: None)
        assert result is None
        assert replay is None


class TestIndexerReplayGlobalOps:
    """Tests for IndexerReplay static/global methods."""

    def setup_method(self):
        IndexerReplay.global_indexer_replay_instances.clear()

    def teardown_method(self):
        IndexerReplay.global_indexer_replay_instances.clear()

    def test_global_action_set_and_clear(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        IndexerReplay.set_global_indexer_replay_action(
            IndexerReplayAction.REPLAY_FORWARD
        )
        assert i1.indexer_replay_action == IndexerReplayAction.REPLAY_FORWARD
        assert i2.indexer_replay_action == IndexerReplayAction.REPLAY_FORWARD
        IndexerReplay.clear_global_indexer_replay_action()
        assert i1.indexer_replay_action is None
        assert i2.indexer_replay_action is None

    def test_global_set_replay_data(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        t1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        t2 = torch.tensor([[3, 2], [1, 0]], dtype=torch.long)
        IndexerReplay.set_replay_data([t1, t2])
        assert torch.equal(i1.target_topk_idx, t1)
        assert torch.equal(i2.target_topk_idx, t2)
        assert len(i1.replay_backward_list) == 1
        assert len(i2.replay_backward_list) == 1

    def test_global_set_replay_data_length_mismatch(self):
        _ = IndexerReplay()
        with pytest.raises(ValueError, match="does not match"):
            IndexerReplay.set_replay_data(
                [
                    torch.tensor([[0, 1]], dtype=torch.long),
                    torch.tensor([[1, 0]], dtype=torch.long),
                ]
            )

    def test_global_get_recorded_data(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        t1 = torch.tensor([[0, 1]], dtype=torch.long)
        t2 = torch.tensor([[1, 0]], dtype=torch.long)
        i1.record_indices(t1)
        i2.record_indices(t2)
        data = IndexerReplay.get_recorded_data()
        assert len(data) == 2
        assert torch.equal(data[0], t1)
        assert torch.equal(data[1], t2)

    def test_global_clear_indices(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        i1.set_target_indices(torch.tensor([[0, 1]], dtype=torch.long))
        i2.set_target_indices(torch.tensor([[1, 0]], dtype=torch.long))
        i1.record_indices(torch.tensor([[2, 3]], dtype=torch.long))
        i2.record_indices(torch.tensor([[3, 2]], dtype=torch.long))
        IndexerReplay.clear_global_indices()
        assert i1.target_topk_idx is None
        assert i2.target_topk_idx is None
        assert i1.recorded_topk_idx is None
        assert i2.recorded_topk_idx is None
        assert i1.replay_backward_list == []
        assert i2.replay_backward_list == []

    def test_global_clear_instances(self):
        _ = IndexerReplay()
        _ = IndexerReplay()
        assert len(IndexerReplay.global_indexer_replay_instances) == 2
        IndexerReplay.clear_global_indexer_replay_instances()
        assert len(IndexerReplay.global_indexer_replay_instances) == 0

    def test_global_set_static_buffers(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        buffer = torch.zeros(10, 2, 4, dtype=torch.int32)
        IndexerReplay.set_global_static_buffers(buffer)
        assert i1.static_buffer is buffer[:, 0, :]
        assert i2.static_buffer is buffer[:, 1, :]

    def test_global_clear_static_buffers(self):
        i1 = IndexerReplay()
        i2 = IndexerReplay()
        buffer = torch.zeros(10, 2, 4, dtype=torch.int32)
        IndexerReplay.set_global_static_buffers(buffer)
        IndexerReplay.clear_global_static_buffers()
        assert i1.static_buffer is None
        assert i2.static_buffer is None
