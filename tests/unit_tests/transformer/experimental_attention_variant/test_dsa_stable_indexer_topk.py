# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels


@pytest.mark.parametrize("return_values", (False, True))
def test_stable_indexer_topk_uses_exact_scores_and_prefixes(monkeypatch, return_values):
    # One CP16 rank in an eight-document 256K pack. Include a large boundary tie,
    # two different adjacent FP32 values, signed zero, padding and a short prefix.
    k, width = 2048, 17408
    generator = torch.Generator().manual_seed(42)
    scores = torch.randint(-2, 4, (8, width), generator=generator).float()
    scores[1, ::3] = 0.0
    scores[1, 1::3] = -0.0
    scores[2].fill_(1)
    scores[2, 2052:3999] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    scores[4, 15:30] = float("-inf")
    lengths = [15360, 15233, 17408, 0, 89, 2047, 2048, 2049]
    indices = []
    expected_values = []
    for row, length in zip(scores.tolist(), lengths):
        ordered = sorted(range(length), key=lambda i: (-row[i], i))[:k]
        indices.append(ordered + [-1] * (k - len(ordered)))
        expected_values.append([row[i] for i in ordered] + [float("-inf")] * (k - len(ordered)))
    scores = scores.cuda()
    original = scores.clone()
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    monkeypatch.setattr(dsa_cudnn_kernels, "_STABLE_INDEXER_TOPK", True)
    for _ in range(3):
        result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(scores, lengths, k, return_values)
        assert result["indices"].dtype == torch.int32
        assert torch.equal(result["indices"].cpu(), torch.tensor(indices))
        if return_values:
            assert torch.equal(result["values"].cpu(), torch.tensor(expected_values))
        else:
            assert result["values"] is None
    assert torch.equal(original, scores)
