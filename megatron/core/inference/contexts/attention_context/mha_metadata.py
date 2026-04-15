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
        max_requests: int,
        max_seqlen: int,
        *,
        query_lengths_buf: torch.Tensor,
        cu_query_seq_lengths_buf: torch.Tensor,
        kv_seq_lengths_buf: torch.Tensor,
        cu_kv_seq_lengths_buf: torch.Tensor,
        block_table_buf: torch.Tensor,
    ):
        super().__init__()
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen

        self._query_lengths_buf = query_lengths_buf
        self._cu_query_seq_lengths_buf = cu_query_seq_lengths_buf
        self._kv_seq_lengths_buf = kv_seq_lengths_buf
        self._cu_kv_seq_lengths_buf = cu_kv_seq_lengths_buf
        self._block_table_buf = block_table_buf

        self._max_seqlen_q = 0
        self._max_seqlen_k = 0

    def update(
        self,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """Assemble state_data.

        Args:
            batch_dimensions: Configuration object with real batch settings.
            padded_batch_dimensions: Configuration object with padded batch settings.
            num_speculative_tokens: Number of speculative tokens.
        """
        padded_active_request_count = padded_batch_dimensions.req_count

        if padded_batch_dimensions.prefill_req_count == 0:
            self._max_seqlen_q = num_speculative_tokens + 1
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


class NonGraphedMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention without CUDA graphs.
    """

    def update(
        self,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        super().update(
            batch_dimensions,
            padded_batch_dimensions,
            num_speculative_tokens,
        )
        if len(self.state_data["query_lengths"]) > 0:
            self.state_data["max_seqlen_q"] = torch.max(self.state_data["query_lengths"])
            self.state_data["max_seqlen_k"] = torch.max(self.state_data["kv_seq_lengths"])
        else:
            self.state_data["max_seqlen_q"] = num_speculative_tokens + 1
            self.state_data["max_seqlen_k"] = 1
