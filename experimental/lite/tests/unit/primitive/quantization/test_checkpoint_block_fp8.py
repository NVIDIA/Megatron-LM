from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.quantization.checkpoint_block_fp8 import (
    BlockFP8CheckpointDequantAdapter,
)


def test_dequantizes_partial_128_blocks_to_bf16() -> None:
    qweight = torch.ones(130, 129, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)

    actual = BlockFP8CheckpointDequantAdapter()(qweight, scales)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == qweight.shape
    assert torch.all(actual[:128, :128] == 1)
    assert torch.all(actual[:128, 128:] == 2)
    assert torch.all(actual[128:, :128] == 4)
    assert torch.all(actual[128:, 128:] == 8)


def test_rejects_missing_or_misaligned_scale_contract() -> None:
    adapter = BlockFP8CheckpointDequantAdapter()
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        adapter(torch.ones(128, 128, dtype=torch.bfloat16), torch.ones(1, 1))
    with pytest.raises(ValueError, match="shape"):
        adapter(
            torch.ones(129, 128, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fused_dequant_matches_cpu_reference_bitwise() -> None:
    torch.manual_seed(7)
    qweight = (
        torch.randn(257, 259)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.rand(3, 3, dtype=torch.float32)
    adapter = BlockFP8CheckpointDequantAdapter()

    expected = adapter(qweight, scales)
    actual = adapter(qweight.cuda(), scales.cuda()).cpu()

    assert torch.equal(actual, expected)
