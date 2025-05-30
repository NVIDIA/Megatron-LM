# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class ChunkedPipelineParallelParams:
    '''Parameters for chunked pipeline model parallel.'''

    micro_batch_idx: int
    span_idx_in_micro: int
    spans: List[int]
    kv_cache: Optional[Dict[str, torch.Tensor]] = None

    def __hash__(self):
        return hash((self.micro_batch_idx, self.span_idx_in_micro, len(self.spans)))


class ChunkedPipelineParallelDataIterator:
    '''An adapter of data iterator for chunked pipeline model parallel.'''

    def __init__(self, data_iterator: RerunDataIterator, config: TransformerConfig):
        self.original_data = None
        self.token_slices = self.position_ids_slices = self.loss_mask_slices = (
            self.labels_slices
        ) = None
        self.count = 0
        self.config = config
        self.chunked_pp_splits = config.chunked_pipeline_model_parallel_splits
        self.data_iterator = data_iterator
        self.current_span_idx = -1
        self.current_seq_length = None
        self.current_split_span = None

    def __next__(self):
        assert (
            self.data_iterator is not None
        ), "data_iterator is None. Please use mock_next instead."

        if self.current_span_idx == -1 or self.current_span_idx + 1 == self.chunked_pp_splits:
            # Sample original data and split it into chunks
            self.original_data = next(self.data_iterator)
            assert (
                "attention_mask" not in self.original_data
                or self.original_data["attention_mask"] is None
            ), "Attention mask is not supported with chunked pipeline model parallel for now."
            tokens, labels, loss_mask, position_ids = (
                self.original_data["tokens"],
                self.original_data["labels"],
                self.original_data["loss_mask"],
                self.original_data["position_ids"],
            )
            self.count += 1
            self.current_span_idx = 0

            # Split the input tensors into chunks
            new_seq_length = tokens.size(1)
            if new_seq_length != self.current_seq_length:
                self.current_seq_length = new_seq_length
                self.current_split_span = self._get_span(new_seq_length)
            self.token_slices = tokens.split(self.current_split_span, dim=1)
            self.labels_slices = labels.split(self.current_split_span, dim=1)
            self.loss_mask_slices = loss_mask.split(self.current_split_span, dim=1)
            self.position_ids_slices = position_ids.split(self.current_split_span, dim=1)
        else:
            self.current_span_idx += 1

        # Get the current chunked tensors
        slice_data = {
            'tokens': self.token_slices[self.current_span_idx],
            'labels': self.labels_slices[self.current_span_idx],
            'loss_mask': self.loss_mask_slices[self.current_span_idx],
            'position_ids': self.position_ids_slices[self.current_span_idx],
        }

        return slice_data

    def _get_span(self, seq_length: int) -> List[int]:
        spans = [seq_length // self.chunked_pp_splits] * self.chunked_pp_splits
        return spans

    def mock_next(self, seq_length: int):
        assert self.data_iterator is None, "mock_next is only available for data_iterator is None."
        if self.current_span_idx == -1 or self.current_span_idx + 1 == self.chunked_pp_splits:
            self.count += 1
            self.current_span_idx = 0
        else:
            self.current_span_idx += 1

        if seq_length != self.current_seq_length:
            self.current_seq_length = seq_length
            self.current_split_span = self._get_span(seq_length)

    def get_current_chunked_pp_params(self) -> ChunkedPipelineParallelParams:
        return ChunkedPipelineParallelParams(
            micro_batch_idx=self.count - 1,
            span_idx_in_micro=self.current_span_idx,
            spans=self.current_split_span,
        )


class ChunkedPipelineParallelQueue:
    """A two-stage queue to store the chunked pipeline model parallel input or output tensors."""

    def __init__(self, chunked_pp_splits):
        self.queues = [[]]

        self._outer_idx = 0  # Tracking how many inner queues exist.
        self._inner_cnt = 0  # Tracking how many chunks have been added to the last inner queue.
        self._count = 0  # Total number of items in the queue.
        self.chunked_pp_splits = chunked_pp_splits
        self.tail_obj = None

    def __len__(self):
        return self._count

    def append(self, obj):
        """Append tensor"""

        self.tail_obj = obj
        self.queues[self._outer_idx].append(obj)
        self._inner_cnt += 1
        if self._inner_cnt == self.chunked_pp_splits:
            # The current queue is full, create a new queue
            self.queues.append([])
            self._inner_cnt = 0
            self._outer_idx += 1
        self._count += 1

    def pop(self, idx=0):
        """Pop the tail item from the first queue, i.e., queues[0][-1]"""

        assert idx == 0, "ChunkedPipelineParallelQueue only supports popping the head item."
        ret = self.queues[0].pop(-1)
        self._count -= 1

        if len(self.queues[0]) == 0 and self._outer_idx > 0:
            self.queues.pop(0)
            self._outer_idx -= 1

        return ret

    def __getitem__(self, idx):
        assert idx == -1, "ChunkedPipelineParallelQueue only supports getting the tail item."
        return self.tail_obj
