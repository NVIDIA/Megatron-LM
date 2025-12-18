# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions

from .metadata_base import MetadataBase


class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.
    """

    def __init__(
        self,
        block_count_total,
        max_kv_block_count,
        max_requests,
        block_size_tokens,
        max_seqlen,
        **tensor_dict,
    ):
        super().__init__()
        self.max_blocks = block_count_total
        self.max_kv_blocks = max_kv_block_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen

        self._query_lengths_buf = tensor_dict["query_lengths_buf"]
        self._cu_query_seq_lengths_buf = tensor_dict["cu_query_seq_lengths_buf"]
        self._kv_seq_lengths_buf = tensor_dict["kv_seq_lengths_buf"]
        self._cu_kv_seq_lengths_buf = tensor_dict["cu_kv_seq_lengths_buf"]
        self._block_table_buf = tensor_dict["block_table_buf"]

        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}

    def update(
        self,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks)
            batch_dimensions: Configuration object containing real batch settings
            padded_batch_dimensions: Configuration object containing padded batch settings
        """
        # Extract values from configs
        real_batch_size = batch_dimensions.req_count
        padded_active_token_count = padded_batch_dimensions.token_count
        padded_active_request_count = padded_batch_dimensions.req_count

        assert real_batch_size <= padded_active_request_count <= self.max_bs

        self.tensor_pad(self._query_lengths_buf, real_batch_size, padded_active_request_count)
        self.tensor_pad(
            self._cu_query_seq_lengths_buf,
            real_batch_size,
            padded_active_request_count,
            is_cumulative_tensor=True,
        )
        self.tensor_pad(self._kv_seq_lengths_buf, real_batch_size, padded_active_request_count)
        self.tensor_pad(
            self._block_table_buf, real_batch_size, padded_active_request_count, pad_value=-1
        )
        self.tensor_pad(
            self._cu_kv_seq_lengths_buf,
            real_batch_size,
            padded_active_request_count,
            is_cumulative_tensor=True,
        )

        if padded_batch_dimensions.prefill_req_count == 0:
            self._max_seqlen_q = 1
        else:
            # Make sure we will launch the prefill kernel for prefill graphs
            self._max_seqlen_q = max(2, padded_batch_dimensions.token_count)

        self._max_seqlen_k = self.max_seqlen

        self.state_data = {
            "query_lengths": self._query_lengths_buf[:padded_active_request_count],
            "cu_query_seq_lengths": self._cu_query_seq_lengths_buf[
                : padded_active_request_count + 1
            ],
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: padded_active_request_count + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:padded_active_request_count],
            "block_table": self._block_table_buf[0:padded_active_request_count, :],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
        }

    def reset(self):
        """
        Reset the metadata for the next batch.
        """
        self._query_lengths_buf.fill_(0)
        self._cu_query_seq_lengths_buf.fill_(0)
        self._cu_kv_seq_lengths_buf.fill_(0)
        self._kv_seq_lengths_buf.fill_(0)
        self._block_table_buf.fill_(0)
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0


class GraphedMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention with CUDA graphs.
    """

    ...


class NonGraphedMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention without CUDA graphs.
    """

    def update(
        self,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
    ):
        """
        Args:
            batch_dimensions: Configuration object containing real batch settings
            padded_batch_dimensions: Configuration object containing padded batch settings
        """
        super().update(batch_dimensions, padded_batch_dimensions)
        if len(self.state_data["query_lengths"]) > 0:
            self.state_data["max_seqlen_q"] = torch.max(self.state_data["query_lengths"]).item()
            self.state_data["max_seqlen_k"] = torch.max(self.state_data["kv_seq_lengths"]).item()
        else:
            self.state_data["max_seqlen_q"] = 1
            self.state_data["max_seqlen_k"] = 1
