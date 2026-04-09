# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.gated_delta_net import GatedDeltaNet

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGDNCuSeqlensResolve:

    @pytest.fixture
    def mock_gdn(self):
        class MockGDN:
            cp_size = 2
            _resolve_cu_seqlens = GatedDeltaNet._resolve_cu_seqlens

        return MockGDN()

    def test_padded_preferred_when_available(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q")
        assert torch.equal(result, padded)

    def test_actual_used_when_no_padding(self, mock_gdn):
        actual = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")
        assert torch.equal(result, actual)

    def test_single_seq_auto_extended(self, mock_gdn):
        actual = torch.tensor([0, 500], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 504, "cu_seqlens_q")
        assert result[-1].item() == 504
        assert actual[-1].item() == 500  # original not modified

    def test_multi_seq_raises_on_mismatch(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="4194"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")

    def test_cp1_skips_validation(self, mock_gdn):
        mock_gdn.cp_size = 1
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")
        assert torch.equal(result, actual)
