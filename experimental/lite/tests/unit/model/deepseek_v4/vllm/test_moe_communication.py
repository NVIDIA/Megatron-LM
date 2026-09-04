from __future__ import annotations

import pytest
import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    _route_hashes,
)


@pytest.mark.gpus(1)
def test_route_hashes_match_cpu_bitwise() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")
    generator = torch.Generator(device="cuda").manual_seed(42)
    fingerprints = torch.randn(
        (257, 16), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    indices = torch.randint(
        0, 256, (257,), dtype=torch.int64, device="cuda", generator=generator
    )
    weights = torch.randn(
        (257,), dtype=torch.float32, device="cuda", generator=generator
    )

    expected = _route_hashes(
        fingerprints.cpu(), indices.cpu(), weights.cpu()
    )
    actual = _route_hashes(fingerprints, indices, weights)

    assert torch.equal(actual.cpu(), expected)
    assert _route_hashes(
        fingerprints[:0], indices[:0], weights[:0]
    ).numel() == 0
