# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from enum import Enum
from typing import List, Optional, Tuple

import torch


class IndexerReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class IndexerReplay:
    global_indexer_replay_instances: List["IndexerReplay"] = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor]):
        if len(all_layers_topk_indices) != len(
            IndexerReplay.global_indexer_replay_instances
        ):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match instances ({len(IndexerReplay.global_indexer_replay_instances)})."
            )
        for i, instance in enumerate(IndexerReplay.global_indexer_replay_instances):
            instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> List[torch.Tensor]:
        return [
            instance.get_recorded_indices()
            for instance in IndexerReplay.global_indexer_replay_instances
        ]

    @staticmethod
    def clear_global_indices():
        for instance in IndexerReplay.global_indexer_replay_instances:
            instance.clear_indices()

    @staticmethod
    def set_global_indexer_replay_action(action: IndexerReplayAction):
        for instance in IndexerReplay.global_indexer_replay_instances:
            instance.set_indexer_replay_action(action)

    @staticmethod
    def clear_global_indexer_replay_action():
        for instance in IndexerReplay.global_indexer_replay_instances:
            instance.clear_indexer_replay_action()

    @staticmethod
    def clear_global_indexer_replay_instances():
        IndexerReplay.global_indexer_replay_instances.clear()

    @staticmethod
    def set_global_static_buffers(static_buffer: torch.Tensor):
        num_layers = len(IndexerReplay.global_indexer_replay_instances)
        assert static_buffer.shape[1] == num_layers, (
            f"Buffer has {static_buffer.shape[1]} layers but there are "
            f"{num_layers} IndexerReplay instances."
        )
        for layer_idx, instance in enumerate(
            IndexerReplay.global_indexer_replay_instances
        ):
            instance.set_static_buffer(static_buffer[:, layer_idx, :])

    @staticmethod
    def clear_global_static_buffers():
        for instance in IndexerReplay.global_indexer_replay_instances:
            instance.clear_static_buffer()

    def __init__(self):
        self.target_topk_idx: Optional[torch.Tensor] = None
        self.recorded_topk_idx: Optional[torch.Tensor] = None
        self.indexer_replay_action: Optional[IndexerReplayAction] = None
        self.replay_backward_list: List[torch.Tensor] = []
        self.static_buffer: Optional[torch.Tensor] = None
        IndexerReplay.global_indexer_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        return self.recorded_topk_idx

    def clear_indices(self):
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_indexer_replay_action(self, action: IndexerReplayAction):
        self.indexer_replay_action = action

    def clear_indexer_replay_action(self):
        self.indexer_replay_action = None

    def get_replay_topk_indices(
        self, default_compute_topk_indices
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.indexer_replay_action == IndexerReplayAction.RECORD:
            topk_indices = default_compute_topk_indices()
            self.record_indices(topk_indices)
            return topk_indices, None
        elif self.indexer_replay_action == IndexerReplayAction.REPLAY_FORWARD:
            topk_indices = self.target_topk_idx
            return topk_indices, topk_indices
        elif self.indexer_replay_action == IndexerReplayAction.REPLAY_BACKWARD:
            topk_indices = self.replay_backward_list.pop(0)
            return topk_indices, topk_indices
        else:
            return None, None

    def set_static_buffer(self, buffer: torch.Tensor):
        self.static_buffer = buffer

    def clear_static_buffer(self):
        self.static_buffer = None

    def record_indices(self, topk_indices: torch.Tensor):
        if self.static_buffer is not None:
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices
