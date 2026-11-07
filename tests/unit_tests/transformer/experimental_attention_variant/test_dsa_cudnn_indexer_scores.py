# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.utils import _missing_cudnn_dsa_kernel_dependencies

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or bool(_missing_cudnn_dsa_kernel_dependencies()),
    reason="requires CUDA and the cuDNN DSA kernels",
)

ROWS = 1024
SK = 8192
NUM_HEADS = 32
HEAD_DIM = 128


def _seq_lens(kind: str, rows: int, sk: int) -> torch.Tensor | None:
    rows_idx = torch.arange(rows, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1)
    lens = {
        "causal": None,
        "pooled": (1234 + rows_idx + 1) // 4,
        "pooled_with_empty_rows": (rows_idx + 1) // 4,
        "non_monotonic": torch.randint(0, sk + 1, (rows,), device="cuda", generator=generator),
        "beyond_sk": torch.full((rows,), sk + 500, device="cuda"),
        "packed_resets": rows_idx % 300 + 1,
    }[kind]
    return None if lens is None else lens.to(torch.int32)


def _reference_scores(q, k, w, row_start, ratio, sm_scale, seq_lens):
    sq, sk = q.size(1), k.size(1)
    logits = torch.einsum("bshd,bkd->bshk", q.float(), k[:, :, 0].float()).relu_()
    scores = (logits * w.float().unsqueeze(-1)).sum(dim=2) * sm_scale
    if seq_lens is None:
        seq_lens = (torch.arange(row_start, row_start + sq, device=q.device) + 1) // ratio
    keys = torch.arange(sk, device=q.device)
    return scores.masked_fill(keys >= seq_lens.clamp(max=sk).view(1, sq, 1), float("-inf"))


@pytest.mark.parametrize(
    "kind,sk,row_start,ratio,rows",
    [
        ("causal", SK, 0, 1, ROWS),
        ("causal", SK, 3000, 1, ROWS),
        ("causal", SK // 4, 4096, 4, ROWS),
        ("causal", SK // 4, 4099, 4, ROWS),
        ("pooled", SK // 4, 0, 1, ROWS),
        ("pooled_with_empty_rows", SK // 4, 0, 1, ROWS),
        ("non_monotonic", SK, 0, 1, ROWS),
        ("beyond_sk", SK, 0, 1, ROWS),
        ("packed_resets", SK, 0, 1, ROWS),
        # More query rows than pooled keys * ratio: cuDNN FE rejects such a ratio as given.
        ("pooled", 578, 0, 1, 2312),
        ("non_monotonic", 578, 0, 1, 2312),
        ("beyond_sk", 578, 0, 1, 2312),
        ("packed_resets", 578, 0, 1, 2312),
        ("non_monotonic", 300, 0, 2, 1000),
        ("pooled", 300, 0, 2, 1000),
    ],
)
def test_indexer_scores_match_reference(kind, sk, row_start, ratio, rows):
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    dsa_cudnn_kernels._ensure_dsa_namespace()
    torch.manual_seed(0)
    q = torch.randn(1, rows, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, rows, NUM_HEADS, device="cuda", dtype=torch.bfloat16)
    seq_lens = _seq_lens(kind, rows, sk)
    sm_scale = dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE

    scores = dsa_cudnn_kernels._compute_indexer_scores_chunk_with_global_rows(
        q, k, w, row_start=row_start, indexer_ratio=ratio, sm_scale=sm_scale, seq_lens=seq_lens
    )
    reference = _reference_scores(q, k, w, row_start, ratio, sm_scale, seq_lens)

    finite = torch.isfinite(reference)
    assert torch.equal(torch.isfinite(scores), finite)
    tolerance = 1e-5 * reference[finite].abs().max().item()
    torch.testing.assert_close(scores[finite], reference[finite], rtol=0, atol=tolerance)
