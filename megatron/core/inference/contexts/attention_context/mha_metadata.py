from typing import Optional

import torch

from .metadata_base import MetadataBase


class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.
    """
    def __init__(
        self,
        chunk_count_total,
        max_kv_chunk_count,
        max_requests,
        chunk_size_tokens,
        max_seqlen,
        debug=False,
    ):
        super().__init__(debug)
        device = torch.cuda.current_device()
        self.device = device
        self.max_blocks = chunk_count_total
        self.max_kv_chunks = max_kv_chunk_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen
        self._query_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._cu_query_seq_lengths_buf = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=device
        )
        self._cu_kv_seq_lengths_buf = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=device)
        self._kv_seq_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._block_table_buf = torch.zeros(
            (self.max_bs, self.max_kv_chunks), dtype=torch.int32, device=device
        )
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.effective_batch_size = 0
        self.state_data = {}

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_chunk_ids: torch.Tensor,
        padded_active_token_count: int,
        real_batch_size: int,
        graph_batch_size: Optional[int] = None,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_chunk_ids: (>real_batch_size, max_kv_chunks)
            padded_active_token_count: int
            real_batch_size: int
            graph_batch_size: Optional[int]
        """
        if graph_batch_size is None:
            graph_batch_size = real_batch_size

        assert real_batch_size <= graph_batch_size <= self.max_bs
        assert request_query_lengths.shape[0] == real_batch_size
        assert request_kv_length_offsets.shape[0] == real_batch_size
        assert request_to_kv_chunk_ids.shape[0] == real_batch_size

        self.tensor_copy_and_pad(
            self._query_lengths_buf, request_query_lengths, real_batch_size, graph_batch_size
        )
        self._cu_query_seq_lengths_buf[0] = 0
        self.tensor_copy_and_pad(
            self._cu_query_seq_lengths_buf[1:],
            torch.cumsum(request_query_lengths, dim=0),
            real_batch_size,
            graph_batch_size,
            is_cumulative_tensor=True,
        )
        self.tensor_copy_and_pad(
            self._kv_seq_lengths_buf,
            request_kv_length_offsets + request_query_lengths,
            real_batch_size,
            graph_batch_size,
        )
        self.tensor_copy_and_pad(
            self._block_table_buf,
            request_to_kv_chunk_ids,
            real_batch_size,
            graph_batch_size,
            pad_value=torch.tensor(self.max_kv_chunks, dtype=torch.int32, device=self.device).fill_(
                -1
            ),
        )
        self._cu_kv_seq_lengths_buf[0] = 0
        self.tensor_copy_and_pad(
            self._cu_kv_seq_lengths_buf[1:],
            torch.cumsum(self._kv_seq_lengths_buf, dim=0),
            real_batch_size,
            graph_batch_size,
            is_cumulative_tensor=True,
        )
        self._max_seqlen_q = padded_active_token_count
        if torch.all(self._query_lengths_buf[:graph_batch_size] == 1):
            self._max_seqlen_q = 1
        self._max_seqlen_k = self.max_seqlen

        self.state_data = {
            "query_lengths": self._query_lengths_buf[:graph_batch_size],
            "cu_query_seq_lengths": self._cu_query_seq_lengths_buf[: graph_batch_size + 1],
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: graph_batch_size + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:graph_batch_size],
            "block_table": self._block_table_buf[0:graph_batch_size, :],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
            "effective_batch_size": self.effective_batch_size,
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
        self.effective_batch_size = 0


class GraphMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention with CUDA graphs.
    """
    def __init__(
        self,
        chunk_count_total,
        max_kv_chunk_count,
        max_requests,
        chunk_size_tokens,
        max_seqlen,
        debug=False,
    ):
        super().__init__(
            chunk_count_total,
            max_kv_chunk_count,
            max_requests,
            chunk_size_tokens,
            max_seqlen,
            debug,
        )

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_chunk_ids: torch.Tensor,
        padded_active_token_count: int,
        real_batch_size: int,
        graph_batch_size: Optional[int] = None,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_chunk_ids: (>real_batch_size, max_kv_chunks)
            padded_active_token_count: int
            real_batch_size: int
            graph_batch_size: Optional[int]
        """
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_chunk_ids,
            padded_active_token_count,
            real_batch_size,
            graph_batch_size,
        )

    def reset(self):
        super().reset()


class NonGraphMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention without CUDA graphs.
    """
    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_chunk_ids: torch.Tensor,
        padded_active_token_count: int,
        real_batch_size: int,
        graph_batch_size: Optional[int] = None,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_chunk_ids: (>real_batch_size, max_kv_chunks)
            padded_active_token_count: int
            real_batch_size: int
            graph_batch_size: Optional[int]
        """
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_chunk_ids,
            padded_active_token_count,
            real_batch_size,
            graph_batch_size,
        )
        self.state_data["max_seqlen_q"] = torch.max(self.state_data["query_lengths"]).item()
        self.state_data["max_seqlen_k"] = torch.max(self.state_data["kv_seq_lengths"]).item()
