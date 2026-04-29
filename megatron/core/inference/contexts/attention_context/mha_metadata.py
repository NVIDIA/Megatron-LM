# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions

from .metadata_base import MetadataBase


def update_mha_metadata(
    query_lengths: torch.Tensor,
    kv_length_offsets: torch.Tensor,
    query_lengths_buf: torch.Tensor,
    cu_query_seq_lengths_buf: torch.Tensor,
    kv_seq_lengths_buf: torch.Tensor,
    cu_kv_seq_lengths_buf: torch.Tensor,
    real_batch_size: int,
    padded_batch_size: int,
) -> None:
    """Compute all 1D MHA metadata buffers using pure PyTorch ops.

    Args:
        query_lengths: ``[>=real_batch_size]`` int32 - per-request query lengths.
        kv_length_offsets: ``[>=real_batch_size]`` int32 - per-request KV offsets.
        query_lengths_buf: ``[>=padded_batch_size]`` int32 - output buffer.
        cu_query_seq_lengths_buf: ``[>=padded_batch_size+1]`` int32 - output buffer.
        kv_seq_lengths_buf: ``[>=padded_batch_size]`` int32 - output buffer.
        cu_kv_seq_lengths_buf: ``[>=padded_batch_size+1]`` int32 - output buffer.
        real_batch_size: Number of real requests.
        padded_batch_size: Padded request count (≥ real_batch_size).
    """
    rbs = real_batch_size
    pbs = padded_batch_size

    if pbs == 0:
        cu_query_seq_lengths_buf[0] = 0
        cu_kv_seq_lengths_buf[0] = 0
        return

    # query_lengths_buf: copy real, zero-pad rest
    query_lengths_buf[:rbs] = query_lengths[:rbs]
    if pbs > rbs:
        query_lengths_buf[rbs:pbs] = 0

    # kv_seq_lengths = kv_offsets + query_lengths, zero-padded
    kv_seq_lengths_buf[:rbs] = kv_length_offsets[:rbs] + query_lengths[:rbs]
    if pbs > rbs:
        kv_seq_lengths_buf[rbs:pbs] = 0

    # cumsum on the padded buffer: zeros propagate last real value
    cu_query_seq_lengths_buf[0] = 0
    torch.cumsum(query_lengths_buf[:pbs], dim=0, out=cu_query_seq_lengths_buf[1 : pbs + 1])

    cu_kv_seq_lengths_buf[0] = 0
    torch.cumsum(kv_seq_lengths_buf[:pbs], dim=0, out=cu_kv_seq_lengths_buf[1 : pbs + 1])


class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.
    """

    def __init__(
        self, block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
    ):
        super().__init__()
        device = torch.cuda.current_device()
        self.device = device
        self.max_blocks = block_count_total
        self.max_kv_blocks = max_kv_block_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen
        self._query_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._cu_query_seq_lengths_buf = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=device
        )
        self._cu_kv_seq_lengths_buf = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=device)
        self._kv_seq_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._block_table_buf = torch.zeros(
            (self.max_bs, self.max_kv_blocks), dtype=torch.int32, device=device
        )
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks)
            batch_dimensions: Configuration object containing real batch settings
            padded_batch_dimensions: Configuration object containing padded batch settings
            num_speculative_tokens: Number of speculative tokens
        """
        # Extract values from configs
        real_batch_size = batch_dimensions.req_count
        padded_active_token_count = padded_batch_dimensions.token_count
        padded_active_request_count = padded_batch_dimensions.req_count

        assert real_batch_size <= padded_active_request_count <= self.max_bs
        assert request_query_lengths.shape[0] == real_batch_size
        assert request_kv_length_offsets.shape[0] == real_batch_size
        assert request_to_kv_block_ids.shape[0] == real_batch_size

        update_mha_metadata(
            query_lengths=request_query_lengths,
            kv_length_offsets=request_kv_length_offsets,
            query_lengths_buf=self._query_lengths_buf,
            cu_query_seq_lengths_buf=self._cu_query_seq_lengths_buf,
            kv_seq_lengths_buf=self._kv_seq_lengths_buf,
            cu_kv_seq_lengths_buf=self._cu_kv_seq_lengths_buf,
            real_batch_size=real_batch_size,
            padded_batch_size=padded_active_request_count,
        )

        # Block table is 2D — handled separately.
        self.tensor_copy_and_pad(
            self._block_table_buf,
            request_to_kv_block_ids,
            real_batch_size,
            padded_active_request_count,
            pad_value=-1,
        )

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

    def __init__(
        self, block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
    ):
        super().__init__(
            block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
        )

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks)
            batch_dimensions: Configuration object containing real batch settings
            padded_batch_dimensions: Configuration object containing padded batch settings
            num_speculative_tokens: Number of speculative tokens
        """
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_block_ids,
            batch_dimensions,
            padded_batch_dimensions,
            num_speculative_tokens,
        )

    def reset(self):
        super().reset()


class NonGraphedMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention without CUDA graphs.
    """

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        batch_dimensions: InferenceBatchDimensions,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """
        Args:
            request_query_lengths: (>real_batch_size,)
            request_kv_length_offsets: (>real_batch_size,)
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks)
            batch_dimensions: Configuration object containing real batch settings
            padded_batch_dimensions: Configuration object containing padded batch settings
            num_speculative_tokens: Number of speculative tokens
        """
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_block_ids,
            batch_dimensions,
            padded_batch_dimensions,
            num_speculative_tokens,
        )
        if len(self.state_data["query_lengths"]) > 0:
            self.state_data["max_seqlen_q"] = torch.max(self.state_data["query_lengths"]).item()
            self.state_data["max_seqlen_k"] = torch.max(self.state_data["kv_seq_lengths"]).item()
        else:
            self.state_data["max_seqlen_q"] = num_speculative_tokens + 1
            self.state_data["max_seqlen_k"] = 1
