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
        enable_chunked_prefill: bool,
    ):
        """
        Helper to construct inputs and run update().

        Args:
            metadata: The MambaMetadata instance.
            req_seq_lengths: List of sequence lengths for all active requests.
                             Order must be [decode_requests..., prefill_requests...].
            num_decode_requests: Number of requests in req_seq_lengths that are in the decode phase.
            padded_dims: The padded batch dimensions to test against.
            enable_chunked_prefill: Whether chunked prefill is enabled.
        """
        num_active_requests = len(req_seq_lengths)
        total_tokens = sum(req_seq_lengths)
        num_prefill_requests = num_active_requests - num_decode_requests

        real_dims = InferenceBatchDimensions(
            token_count=total_tokens,
            prefill_req_count=num_prefill_requests,
            decode_req_count=num_decode_requests,
        )

        # Assuming 1:1 mapping (req_id i -> slot i)
        active_mamba_indices = torch.arange(
            num_active_requests, dtype=torch.int32, device=metadata.device
        )

        cu_seqlens = [0]
        current_len = 0
        for l in req_seq_lengths:
            current_len += l
            cu_seqlens.append(current_len)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=metadata.device)

        token_to_req = []
        for req_idx, length in enumerate(req_seq_lengths):
            token_to_req.extend([req_idx] * length)
        token_to_req_tensor = torch.tensor(token_to_req, dtype=torch.int32, device=metadata.device)

        metadata.update(
            active_mamba_indices=active_mamba_indices,
            token_to_request_idx=token_to_req_tensor,
            cu_seqlens=cu_seqlens_tensor,
            batch_dimensions=real_dims,
            padded_batch_dimensions=padded_dims,
            enable_chunked_prefill=enable_chunked_prefill,
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

        expected_decode = torch.arange(4, dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)
        assert metadata_context.batch_indices_prefill is None
        assert metadata_context.batch_kernel_batch_indices is None
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

        expected_decode = torch.tensor(
            [0, 1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)
        assert metadata_context.batch_indices_prefill is None
        assert metadata_context.batch_kernel_batch_indices is None
        assert metadata_context.device_decode_prefill is None

    @pytest.mark.internal
    def test_update_chunked_enabled_no_prefill_reqs(self, metadata_context):
        """Test edge case: Chunked prefill enabled, but only decode requests exist."""
        seq_lengths = [1, 1]
        num_decode = 2
        padded_dims = InferenceBatchDimensions(
            token_count=2, prefill_req_count=0, decode_req_count=2
        )

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=True
        )

        # Should behave exactly like decode-only (chunked logic skipped if real_prefill == 0)
        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)
        assert metadata_context.batch_kernel_batch_indices is None
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

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
        assert metadata_context.batch_kernel_batch_indices is None
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

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
        assert metadata_context.batch_kernel_batch_indices is None
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_prefill = torch.tensor([2, 3], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_device_counts = torch.tensor(
            [2, 2], dtype=torch.int32, device=metadata_context.device
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
    def test_update_padded_prefill_and_decode(self, metadata_context):
        """Test scenario where padded dimensions differ from real dimensions (Mixed)."""
        # Real: 1 decode, 1 prefill.
        seq_lengths = [1, 10]
        num_decode = 1

        # Padded: 4 decode, 4 prefill. Total tokens 32.
        padded_dims = InferenceBatchDimensions(
            token_count=32, prefill_req_count=4, decode_req_count=4
        )

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=False
        )

        expected_decode = torch.tensor(
            [0, -1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_prefill = torch.tensor(
            [1, -1, -1, -1], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_device_counts = torch.tensor(
            [1, 1], dtype=torch.int32, device=metadata_context.device
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=True
        )

        expected_device_chunked_prefill = torch.tensor(
            [50, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_chunked_prefill, expected_device_chunked_prefill)

        assert metadata_context.batch_kernel_batch_indices[0] == 1

        expected_prefill = torch.tensor([2, -1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_device_counts = torch.tensor(
            [1, 2], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu_seqlens = torch.tensor(
            [0, 10, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx = torch.zeros((1, 61), dtype=torch.int32, device=metadata_context.device)
        expected_seq_idx[:, 10:] = -1
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=True
        )

        expected_decode = torch.tensor([0, 1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_decode, expected_decode)

        expected_device_chunked_prefill = torch.tensor(
            [50, 10], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_chunked_prefill, expected_device_chunked_prefill)

        assert metadata_context.batch_kernel_batch_indices[0] == 2

        expected_prefill = torch.tensor([3, -1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_device_counts = torch.tensor(
            [2, 2], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_decode_prefill, expected_device_counts)

        expected_cu = torch.tensor([0, 10, 10], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.cu_seqlens, expected_cu)

        expected_seq_idx = torch.full(
            (1, 62), -1, dtype=torch.int32, device=metadata_context.device
        )
        expected_seq_idx[:, :10] = 0
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

        self._run_update_test(
            metadata_context, seq_lengths, num_decode, padded_dims, enable_chunked_prefill=True
        )

        assert metadata_context.batch_indices_decode is None

        assert metadata_context.batch_kernel_batch_indices[0] == 0

        expected_prefill = torch.tensor([-1, -1], dtype=torch.int32, device=metadata_context.device)
        assert torch.equal(metadata_context.batch_indices_prefill, expected_prefill)

        expected_cu_seqlens = torch.tensor(
            [0, 0, 0], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.cu_seqlens, expected_cu_seqlens)

        expected_seq_idx = torch.full(
            (1, 128), -1, dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.seq_idx, expected_seq_idx)

        expected_device_chunked_prefill = torch.tensor(
            [100, 0], dtype=torch.int32, device=metadata_context.device
        )
        assert torch.equal(metadata_context.device_chunked_prefill, expected_device_chunked_prefill)

        assert metadata_context.device_decode_prefill is None
