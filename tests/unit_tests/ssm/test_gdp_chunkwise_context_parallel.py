# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.context_parallel.chunkwise import build_packed_sequence_cp_metadata
from megatron.core.ssm.gated_delta_product import HAVE_FLA_GDP_CP, GatedDeltaProductMixer

if HAVE_FLA_GDP_CP:
    import megatron.core.ssm.context_parallel.gdp as gdp_cp_module
else:
    gdp_cp_module = None


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.mark.skipif(not HAVE_FLA_GDP_CP, reason="FLA GDP CP kernels are not installed")
def test_interleave_last_update():
    """Verify backward values align with each token's final Householder update."""
    assert gdp_cp_module is not None
    tensor = torch.arange(12).reshape(1, 3, 2, 2)

    interleaved = gdp_cp_module._interleave_last_update(tensor, num_householder=3)

    assert interleaved.shape == (1, 9, 2, 2)
    torch.testing.assert_close(interleaved[:, 2::3], tensor)
    torch.testing.assert_close(interleaved[:, 0::3], torch.zeros_like(tensor))
    torch.testing.assert_close(interleaved[:, 1::3], torch.zeros_like(tensor))


def test_reuse_rank_local_metadata():
    """Verify GDP reuses cached local sequence boundaries and CP summary bounds."""
    global_seq_idx = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32)
    metadata = build_packed_sequence_cp_metadata(global_seq_idx, cp_rank=1, cp_size=2)
    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    mixer.pg_collection = SimpleNamespace(cp=_FakeGroup(rank=1, size=2))
    packed_seq_params = PackedSeqParams(qkv_format="thd", seq_idx=global_seq_idx)

    returned_seq_idx, local_cu_seqlens, preceding_start, following_stop = (
        mixer._chunkwise_packed_metadata(
            packed_seq_params, local_sequence_length=4, metadata=metadata
        )
    )

    assert returned_seq_idx is global_seq_idx
    assert local_cu_seqlens is metadata.local_cu_seqlens
    torch.testing.assert_close(local_cu_seqlens, torch.tensor([0, 1, 4], dtype=torch.int32))
    assert preceding_start == 0
    assert following_stop == 2
