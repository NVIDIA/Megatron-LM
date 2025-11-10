# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from .metadata_base import MetadataBase
from .triton import compute_layout_triton
from .triton import attn_partial_copy_triton
from .triton import attn_merge_triton

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


class MHASplitPDMetadata(MetadataBase):
    """
    Metadata for MHA layer with split prefill/decode support using FlashInfer.
    Provides comprehensive buffers for both prefill and decode phases.
    """

    def __init__(
        self, block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
    ):
        super().__init__()
        device = torch.cuda.current_device()
        self.device = device
        self.max_blocks = block_count_total
        self.max_kv_chunks = max_kv_block_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen

        # Per-step fields
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0

        # Prefill buffers (sized to max batch size)
        self._prefill_qo_indptr_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self._prefill_paged_kv_indptr_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self._prefill_paged_kv_indices_buf = torch.zeros(
            block_count_total, dtype=torch.int32, device=device
        )
        self._prefill_paged_kv_last_page_len_buf = torch.zeros(
            max_requests, dtype=torch.int32, device=device
        )
        self._prefill_block_table_buf = torch.zeros(
            (max_requests, max_kv_block_count), dtype=torch.int32, device=device
        )
        self._prefill_cum_kv_seq_len_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=device
        )

        # Decode buffers (sized to max batch size)
        self._decode_qo_indptr_buf = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
        self._decode_paged_kv_indptr_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self._decode_paged_kv_indices_buf = torch.zeros(
            block_count_total, dtype=torch.int32, device=device
        )
        self._decode_paged_kv_last_page_len_buf = torch.zeros(
            max_requests, dtype=torch.int32, device=device
        )
        self._decode_block_table_buf = torch.zeros(
            (max_requests, max_kv_block_count), dtype=torch.int32, device=device
        )

        # Full outputs used by Triton layout kernel (graph mode)
        self._full_qo_indptr_buf = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
        self._full_indptr_buf = torch.zeros(max_requests + 1, dtype=torch.int32, device=device)
        self._full_last_page_len_buf = torch.zeros(max_requests, dtype=torch.int32, device=device)
        self._full_cum_kv_seq_len_buf = torch.zeros(
            max_requests + 1, dtype=torch.int32, device=device
        )
        self._full_max_metadata_buf = torch.zeros(2, dtype=torch.int32, device=device)
        self._full_block_table_buf = torch.zeros(
            (max_requests, max_kv_block_count), dtype=torch.int32, device=device
        )
        self._kv_seq_lengths_buf = torch.zeros(max_requests, dtype=torch.int32, device=device)
        self._device_decode_prefill_buf = torch.zeros(2, dtype=torch.int32, device=device)

        self.state_data = {}
        self.block_size_tokens = block_size_tokens

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        real_config,
        padded_config,
    ):
        """
        Update metadata using Triton kernel for layout computation.

        Args:
            request_query_lengths: (>real_batch_size,) query lengths for each request
            request_kv_length_offsets: (>real_batch_size,) KV cache offsets for each request
            request_to_kv_block_ids: (>real_batch_size, max_kv_blocks) block table mapping
            real_config: Configuration object containing real batch settings
            padded_config: Configuration object containing padded batch settings
        """
        # Extract values from configs
        real_batch_size = real_config.req_count
        padded_active_request_count = padded_config.req_count
        self.real_config = real_config
        self.padded_config = padded_config

        # Get prefill and decode counts from configs
        pf_count = real_config.prefill_req_count
        dc_count = real_config.decode_req_count

        # Get padded target sizes from padded config
        pf_target_size = padded_config.prefill_req_count
        dc_target_size = padded_config.decode_req_count

        assert real_batch_size <= padded_active_request_count <= self.max_bs
        assert pf_count + dc_count == real_batch_size
        assert pf_target_size + dc_target_size == padded_active_request_count
        assert request_query_lengths.shape[0] >= real_batch_size
        assert request_kv_length_offsets.shape[0] >= real_batch_size
        assert request_to_kv_block_ids.shape[0] >= real_batch_size

        # Create view of input tensors for the real batch
        query_lengths_view = request_query_lengths[:real_batch_size]

        # Call Triton kernel with preallocated buffers
        compute_layout_triton(
            request_query_lengths_view=query_lengths_view,
            request_kv_length_offsets_view=request_kv_length_offsets[:real_batch_size],
            block_table=request_to_kv_block_ids[:real_batch_size, :],
            chunk_size_tokens=self.block_size_tokens,
            dc_count=dc_count,
            pf_count=pf_count,
            dc_target_size=dc_target_size,
            pf_target_size=pf_target_size,
            # Full outputs
            qo_indptr=self._full_qo_indptr_buf,
            last_page_len=self._full_last_page_len_buf,
            indptr=self._full_indptr_buf,
            kv_indices=self._prefill_paged_kv_indices_buf,  # Reuse prefill indices buffer
            # Prefill outputs
            pf_qo_indptr=self._prefill_qo_indptr_buf,
            pf_last_page_len=self._prefill_paged_kv_last_page_len_buf,
            pf_indptr=self._prefill_paged_kv_indptr_buf,
            pf_cum_kv_seq_len=self._prefill_cum_kv_seq_len_buf,
            # Decode outputs
            dc_qo_indptr=self._decode_qo_indptr_buf,
            dc_last_page_len=self._decode_paged_kv_last_page_len_buf,
            dc_indptr=self._decode_paged_kv_indptr_buf,
            # Block tables
            prefill_block_table=self._prefill_block_table_buf,
            decode_block_table=self._decode_block_table_buf,
            full_block_table=self._full_block_table_buf,
            # Additional outputs
            cum_kv_seq_len=self._full_cum_kv_seq_len_buf,
            max_metadata=self._full_max_metadata_buf,
            kv_seq_lengths=self._kv_seq_lengths_buf,
            device_decode_prefill=self._device_decode_prefill_buf,
            MAX_BATCH_SIZE_CONST=self.max_bs,
            check_layout=False,
        )

        # Extract max_seqlen_q and max_seqlen_k from metadata buffer
        self.cpu_max_metadata = self._full_max_metadata_buf.cpu()
        self._max_seqlen_q = int(self.cpu_max_metadata[0].item())
        self._max_seqlen_k = int(self.cpu_max_metadata[1].item())

        # Build state_data with properly sliced tensors
        self.state_data = {
            # Full batch metadata
            "cu_query_seq_lengths": self._full_qo_indptr_buf[: padded_active_request_count + 1],
            "cu_kv_seq_lengths": self._full_cum_kv_seq_len_buf[: padded_active_request_count + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:padded_active_request_count],
            "block_table": self._full_block_table_buf[:padded_active_request_count, :],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
            # Prefill-specific metadata
            "prefill_qo_indptr": self._prefill_qo_indptr_buf[: pf_target_size + 1],
            "prefill_paged_kv_indptr": self._prefill_paged_kv_indptr_buf[: pf_target_size + 1],
            "prefill_paged_kv_last_page_len": self._prefill_paged_kv_last_page_len_buf[
                :pf_target_size
            ],
            "prefill_block_table": self._prefill_block_table_buf[:pf_target_size, :],
            "prefill_cum_kv_seq_len": self._prefill_cum_kv_seq_len_buf[: pf_target_size + 1],
            # Decode-specific metadata
            "decode_qo_indptr": self._decode_qo_indptr_buf[: dc_target_size + 1],
            "decode_paged_kv_indptr": self._decode_paged_kv_indptr_buf[: dc_target_size + 1],
            "decode_paged_kv_last_page_len": self._decode_paged_kv_last_page_len_buf[
                :dc_target_size
            ],
            "decode_block_table": self._decode_block_table_buf[:dc_target_size, :],
            # Counts
            "decode_count": dc_count,
            "prefill_count": pf_count,
            "device_decode_prefill": self._device_decode_prefill_buf,
        }

    def reset(self):
        """
        Reset the metadata for the next batch.
        """
        # Reset max sequence lengths
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0

        # Reset prefill buffers
        self._prefill_qo_indptr_buf.fill_(0)
        self._prefill_paged_kv_indptr_buf.fill_(0)
        self._prefill_paged_kv_indices_buf.fill_(0)
        self._prefill_paged_kv_last_page_len_buf.fill_(0)
        self._prefill_block_table_buf.fill_(0)
        self._prefill_cum_kv_seq_len_buf.fill_(0)

        # Reset decode buffers
        self._decode_qo_indptr_buf.fill_(0)
        self._decode_paged_kv_indptr_buf.fill_(0)
        self._decode_paged_kv_indices_buf.fill_(0)
        self._decode_paged_kv_last_page_len_buf.fill_(0)
        self._decode_block_table_buf.fill_(0)

        # Reset full buffers (for graph mode)
        self._full_qo_indptr_buf.fill_(0)
        self._full_indptr_buf.fill_(0)
        self._full_last_page_len_buf.fill_(0)
        self._full_cum_kv_seq_len_buf.fill_(0)
        self._full_max_metadata_buf.fill_(0)
        self._full_block_table_buf.fill_(0)
        self._kv_seq_lengths_buf.fill_(0)
        self._device_decode_prefill_buf.fill_(0)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        q = q.squeeze(1)

        # prefill only:
        if self.padded_config.prefill_req_count > 0 and self.padded_config.decode_req_count == 0:
            o_pf = flash_attn_varlen_func(
                q,
                k,
                v,
                self.state_data["prefill_qo_indptr"],
                self.state_data["prefill_cum_kv_seq_len"],
                self.state_data["max_seqlen_q"],
                self.state_data["max_seqlen_k"],
                softmax_scale=softmax_scale,
                causal=True,
                block_table=self.state_data["prefill_block_table"],
            )
            return o_pf.unsqueeze(1)

        # decode only:
        if self.padded_config.decode_req_count > 0 and self.padded_config.prefill_req_count == 0:
            flash_attn_args = {
                "q": q[: self.padded_config.decode_req_count].unsqueeze(1),
                "k_cache": k,
                "v_cache": v,
                "cache_seqlens": self.state_data["kv_seq_lengths"][
                    : self.padded_config.decode_req_count
                ],
                "causal": True,
                "block_table": self.state_data["decode_block_table"],
            }
            o_dc = flash_attn_with_kvcache(**flash_attn_args).squeeze(1)
            return o_dc.unsqueeze(1)

        # prefill and decode:
        q_pf = torch.empty_like(q)
        attn_partial_copy_triton(
            q, q_pf, self.state_data["device_decode_prefill"], check_bounds=False
        )
        if self.padded_config.prefill_req_count > 0:
            o_pf = flash_attn_varlen_func(
                q_pf[: self.padded_config.token_count - self.padded_config.decode_req_count],
                k,
                v,
                self.state_data["prefill_qo_indptr"],
                self.state_data["prefill_cum_kv_seq_len"],
                self.state_data["max_seqlen_q"],
                self.state_data["max_seqlen_k"],
                softmax_scale=softmax_scale,
                causal=True,
                block_table=self.state_data["prefill_block_table"],
            )
        else:
            o_pf = torch.empty_like(q)

        if self.padded_config.decode_req_count > 0:
            flash_attn_args = {
                "q": q[: self.padded_config.decode_req_count].unsqueeze(1),
                "k_cache": k,
                "v_cache": v,
                "cache_seqlens": self.state_data["kv_seq_lengths"][
                    : self.padded_config.decode_req_count
                ],
                "causal": True,
                "block_table": self.state_data["decode_block_table"],
            }
            o_dc = flash_attn_with_kvcache(**flash_attn_args).squeeze(1)
        else:
            o_dc = torch.empty_like(q)
        o_final = torch.empty_like(q)

        attn_merge_triton(
            decode_tensor=o_dc,
            prefill_tensor=o_pf,
            output_tensor=o_final,
            device_dc=self.state_data["device_decode_prefill"],
            pf_useful_from_beginning=True,
        )
        return o_final.unsqueeze(1)
