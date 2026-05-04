# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.attention_context.mamba_metadata import MambaMetadata


class TestMambaMetadata:

    @pytest.fixture
    def metadata_context(self):
        """Fixture to initialize MambaMetadata with standard constraints."""
        max_requests = 16
        max_tokens = 2048
        metadata = MambaMetadata(max_requests=max_requests, max_tokens=max_tokens)

        # Manually allocate some slots to simulate a running state.
        # We assume request_id i maps to mamba_slot i for simplicity in assertions.
        for i in range(max_requests):
            metadata.request_to_mamba_state_idx[i] = i

        yield metadata
        metadata.reset()

    def _run_update_test(
        self,
        metadata: MambaMetadata,
        req_seq_lengths: list[int],
        num_decode_requests: int,
        padded_dims: InferenceBatchDimensions,
    ):
        """
        Helper to construct inputs and run update().

        Args:
            metadata: The MambaMetadata instance.
            req_seq_lengths: List of sequence lengths for all active requests.
                             Order must be [decode_requests..., prefill_requests...].
            num_decode_requests: Number of requests in req_seq_lengths that are in the decode phase.
            padded_dims: The padded batch dimensions to test against.
        """
        num_active_requests = len(req_seq_lengths)
        total_tokens = sum(req_seq_lengths)
        num_prefill_requests = num_active_requests - num_decode_requests
        max_requests = metadata.max_requests
        device = metadata.device

        real_dims = InferenceBatchDimensions(
            token_count=total_tokens,
            prefill_req_count=num_prefill_requests,
            decode_req_count=num_decode_requests,
        )

        # Full-sized active_mamba_indices (of size `max_requests`).
        # The real entries use a 1:1 mapping; slots past the real count are -1.
        active_mamba_indices = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        active_mamba_indices[:num_active_requests] = torch.arange(
            num_active_requests, dtype=torch.int32, device=device
        )

        # Full-sized cu_seqlens (of size `max_requests + 1`).
        # Positions past `num_active` are padded with the last value.
        cu_values = [0]
        for length in req_seq_lengths:
            cu_values.append(cu_values[-1] + length)
        last_real_cu = cu_values[-1]
        while len(cu_values) < max_requests + 1:
            cu_values.append(last_real_cu)
        cu_seqlens_tensor = torch.tensor(cu_values, dtype=torch.int32, device=device)

        real_decode_count_gpu = torch.tensor(
            [num_decode_requests], dtype=torch.int32, device=device
        )
        real_prefill_count_gpu = torch.tensor(
            [num_prefill_requests], dtype=torch.int32, device=device
        )
        arange_buf = torch.arange(max_requests, dtype=torch.int32, device=device)

        metadata.update(
            active_mamba_indices=active_mamba_indices,
            cu_seqlens=cu_seqlens_tensor,
            real_decode_count_gpu=real_decode_count_gpu,
            real_prefill_count_gpu=real_prefill_count_gpu,
            arange_buf=arange_buf,
            padded_batch_dimensions=padded_dims,
        )

        return real_dims, active_mamba_indices

    # -------------------------------------------------------------------------
    # Scenario 1: Decode Only
    # -------------------------------------------------------------------------

    @pytest.mark.internal
    def test_update_decode_only_exact_match(self, metadata_context):
        """Test simple decode only case where real dims match padded dims."""
        seq_lengths = [1, 1, 1, 1]  # 4 requests
        num_decode = 4
        padded_dims = InferenceBatchDimensions(
            token_count=4, prefill_req_count=0, decode_req_count=4
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.arange(4, dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        assert metadata_context.batch_indices_prefill is None
        assert metadata_context.device_decode_prefill is None
        assert metadata_context.cu_seqlens is None
        assert metadata_context.seq_idx is None

    @pytest.mark.internal
    def test_update_decode_only_padded(self, metadata_context):
        """Test decode only with padding (e.g. using CUDA graphs bucket)."""
        seq_lengths = [1, 1]  # 2 requests
        num_decode = 2
        # Padding to 4 requests
        padded_dims = InferenceBatchDimensions(
            token_count=4, prefill_req_count=0, decode_req_count=4
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.tensor(
            [0, 1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        assert metadata_context.batch_indices_prefill is None
        assert metadata_context.device_decode_prefill is None

    @pytest.mark.internal
    def test_update_chunked_enabled_no_prefill_reqs(self, metadata_context):
        """Test edge case: Chunked prefill enabled, but only decode requests exist."""
        seq_lengths = [1, 1]
        num_decode = 2
        padded_dims = InferenceBatchDimensions(
            token_count=2, prefill_req_count=0, decode_req_count=2
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        # Should behave exactly like decode-only (chunked logic skipped if real_prefill == 0)
        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        assert metadata_context.batch_indices_prefill is None
        assert metadata_context.cu_seqlens is None
        assert metadata_context.seq_idx is None

    # -------------------------------------------------------------------------
    # Scenario 2: Prefill Only
    # -------------------------------------------------------------------------

    @pytest.mark.internal
    def test_update_prefill_only_exact(self, metadata_context):
        """Test prefill only scenario (exact match)."""
        seq_lengths = [10, 20]  # 2 requests
        num_decode = 0
        padded_dims = InferenceBatchDimensions(
            token_count=30, prefill_req_count=2, decode_req_count=0
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_prefill = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_cu_seqlens = torch.tensor(
            [0, 10, 30], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx_0 = torch.zeros((1, 10), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx_1 = torch.ones((1, 20), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx = torch.cat([expected_seq_idx_0, expected_seq_idx_1], dim=1)
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

        assert metadata_context.batch_indices_decode is None
        assert metadata_context.device_decode_prefill is None

    @pytest.mark.internal
    def test_update_prefill_only_padded(self, metadata_context):
        """Test prefill only with padding."""
        seq_lengths = [10]  # 1 request
        num_decode = 0
        # Pad to 3 prefill requests
        padded_dims = InferenceBatchDimensions(
            token_count=30, prefill_req_count=3, decode_req_count=0
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_prefill = torch.tensor(
            [0, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_cu_seqlens = torch.tensor(
            [0, 10, 10, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx = torch.full(
            (1, 30), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx[:, :10] = 0
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

        assert metadata_context.batch_indices_decode is None
        assert metadata_context.device_decode_prefill is None

    # -------------------------------------------------------------------------
    # Scenario 3: Mixed Batch (Decode + Prefill)
    # -------------------------------------------------------------------------

    @pytest.mark.internal
    def test_update_mixed_batch_exact(self, metadata_context):
        """Test mix of decode and prefill requests (exact match)."""
        # 2 decode (len 1), 2 prefill (len 10, 20)
        seq_lengths = [1, 1, 10, 20]
        num_decode = 2
        padded_dims = InferenceBatchDimensions(
            token_count=32, prefill_req_count=2, decode_req_count=2
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_prefill = torch.tensor([2, 3], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        # device_decode_prefill stores [decode_token_count, prefill_token_count].
        expected_device_counts = torch.tensor(
            [2, 30], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu_seqlens = torch.tensor(
            [0, 10, 30], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx_0 = torch.zeros((1, 10), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx_1 = torch.ones((1, 20), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx_padding = torch.full(
            (1, 2), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx = torch.cat(
            [expected_seq_idx_0, expected_seq_idx_1, expected_seq_idx_padding], dim=1
        )
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

    @pytest.mark.internal
    def test_update_mixed_batch_mtp(self, metadata_context):
        """Test mixed batch with MTP (decode seq_len > 1) to verify token-count split."""
        # 2 decode requests each with 3 tokens (1 accepted + 2 speculative),
        # 1 prefill request with 20 tokens.
        seq_lengths = [3, 3, 20]
        num_decode = 2
        padded_dims = InferenceBatchDimensions(
            token_count=26, prefill_req_count=1, decode_req_count=2
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_prefill = torch.tensor([2], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        # device_decode_prefill stores [decode_token_count, prefill_token_count].
        # 2 decode requests * 3 tokens each = 6 decode tokens, 20 prefill tokens.
        expected_device_counts = torch.tensor(
            [6, 20], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

    @pytest.mark.internal
    def test_update_padded_prefill_and_decode(self, metadata_context):
        """Test scenario where padded dimensions differ from real dimensions (Mixed)."""
        # Real: 1 decode, 1 prefill.
        seq_lengths = [1, 10]
        num_decode = 1

        # Padded: 4 decode, 4 prefill. Total tokens 32.
        padded_dims = InferenceBatchDimensions(
            token_count=32, prefill_req_count=4, decode_req_count=4
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.tensor(
            [0, -1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_prefill = torch.tensor(
            [1, -1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        # device_decode_prefill stores [decode_token_count, prefill_token_count].
        expected_device_counts = torch.tensor(
            [1, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu = torch.tensor(
            [0, 10, 10, 10, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu)

        expected_seq_idx = torch.full(
            (1, 32), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx[:, :10] = 0
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

    # -------------------------------------------------------------------------
    # Scenario 4: Chunked Prefill
    #
    # In our unified implementation, all prefill requests (including chunked)
    # go through the same varlen path and are stored in batch_indices_prefill.
    # There is no separate batch_indices_chunked_prefill.
    # -------------------------------------------------------------------------

    @pytest.mark.internal
    def test_update_chunked_prefill_mixed_exact(self, metadata_context):
        """Test chunked prefill mixed with decode (Exact match)."""
        # 1 decode, 1 chunked prefill (len 50), 1 regular prefill (len 10)
        seq_lengths = [1, 50, 10]
        num_decode = 1

        # Exact dimensions
        padded_dims = InferenceBatchDimensions(
            token_count=61, prefill_req_count=2, decode_req_count=1
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        # All prefill requests (chunked + regular) are unified in batch_indices_prefill.
        expected_prefill = torch.tensor([1, 2], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        # device_decode_prefill stores [decode_token_count, prefill_token_count].
        expected_device_counts = torch.tensor(
            [1, 60], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu_seqlens = torch.tensor(
            [0, 50, 60], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx_0 = torch.zeros((1, 50), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx_1 = torch.ones((1, 10), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx_padding = torch.full(
            (1, 1), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx = torch.cat(
            [expected_seq_idx_0, expected_seq_idx_1, expected_seq_idx_padding], dim=1
        )
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

    @pytest.mark.internal
    def test_update_chunked_prefill_mixed_padded(self, metadata_context):
        """Test chunked prefill mixed with decode (Padded)."""
        # 2 decode, 1 chunked prefill (len 50), 1 regular prefill (len 10)
        seq_lengths = [1, 1, 50, 10]
        num_decode = 2
        padded_dims = InferenceBatchDimensions(
            token_count=62, prefill_req_count=2, decode_req_count=2
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        # All prefill requests unified in batch_indices_prefill.
        expected_prefill = torch.tensor([2, 3], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        # device_decode_prefill stores [decode_token_count, prefill_token_count].
        expected_device_counts = torch.tensor(
            [2, 60], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu = torch.tensor([0, 50, 60], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.cu_seqlens, expected_cu)

        expected_seq_idx = torch.full(
            (1, 62), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx[:, :50] = 0
        expected_seq_idx[:, 50:60] = 1
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

    @pytest.mark.internal
    def test_update_chunked_only_padded(self, metadata_context):
        """Test a case with only chunked prefill (no decode, no regular prefill) but with padding."""
        # 1 chunked prefill request.
        seq_lengths = [100]
        num_decode = 0

        padded_dims = InferenceBatchDimensions(
            token_count=128, prefill_req_count=2, decode_req_count=0
        )

        self._run_update_test(metadata_context, seq_lengths, num_decode, padded_dims)

        assert metadata_context.batch_indices_decode is None

        # Single prefill request unified in batch_indices_prefill, with padding.
        expected_prefill = torch.tensor([0, -1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_cu_seqlens = torch.tensor(
            [0, 100, 100], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx = torch.full(
            (1, 128), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx[:, :100] = 0
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

        assert metadata_context.device_decode_prefill is None

    # -------------------------------------------------------------------------
    # Scenario 5: Paused requests (non-zero active slot IDs)
    # -------------------------------------------------------------------------

    @pytest.mark.internal
    def test_update_with_paused_requests_local_seq_idx(self, metadata_context):
        """seq_idx is 0-based local even when active mamba slots start above 0.

        Regression: paused requests can hold low-numbered mamba slots, so the
        active prefill requests sit at non-zero absolute slot IDs. The kernel
        consumes its own input slice, so seq_idx must be local indices
        (0, 1, ...) even though batch_indices_prefill carries the absolute IDs.
        """
        metadata = metadata_context
        max_requests = metadata.max_requests
        device = metadata.device

        # Two active prefill requests of lengths 10 and 20 occupy mamba slots 2
        # and 3; slots 0 and 1 simulate paused requests.
        active_mamba_indices = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        active_mamba_indices[:2] = torch.tensor([2, 3], dtype=torch.int32, device=device)

        cu_values = [0, 10, 30] + [30] * (max_requests + 1 - 3)
        cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)

        real_decode_count_gpu = torch.tensor([0], dtype=torch.int32, device=device)
        real_prefill_count_gpu = torch.tensor([2], dtype=torch.int32, device=device)
        arange_buf = torch.arange(max_requests, dtype=torch.int32, device=device)

        padded_dims = InferenceBatchDimensions(
            token_count=30, prefill_req_count=2, decode_req_count=0
        )

        metadata.update(
            active_mamba_indices=active_mamba_indices,
            cu_seqlens=cu_seqlens,
            real_decode_count_gpu=real_decode_count_gpu,
            real_prefill_count_gpu=real_prefill_count_gpu,
            arange_buf=arange_buf,
            padded_batch_dimensions=padded_dims,
        )

        # Absolute slot IDs preserved in batch_indices_prefill.
        expected_prefill = torch.tensor([2, 3], dtype=torch.int32, device=device)
        assert torch.equal(metadata.batch_indices_prefill, expected_prefill)

        # seq_idx is 0-based local (0s for the first request, 1s for the second).
        expected_seq_idx = torch.cat(
            [
                torch.zeros((1, 10), dtype=torch.int32, device=device),
                torch.ones((1, 20), dtype=torch.int32, device=device),
            ],
            dim=1,
        )
        assert torch.equal(metadata.seq_idx, expected_seq_idx)
