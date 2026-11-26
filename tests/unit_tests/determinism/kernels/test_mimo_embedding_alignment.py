# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of MiMo's token-row embedding merge and its input gradients."""

import pytest
import torch

from megatron.core.models.mimo.model.base import MimoModel
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precomputed", [False, True])
def test_mimo_embedding_alignment_replays(dtype, precomputed):
    """Unique token rows have no write conflicts, including with strided modality inputs."""
    seeded()
    batch_size, seq_length, hidden_dim = 2, 2048, 256
    input_ids = (
        torch.arange(batch_size * seq_length, device="cuda").reshape(batch_size, seq_length) % 3
    )
    masks = {"vision": input_ids == 1, "text": input_ids == 0, "audio": input_ids == 2}
    token_indices = {name: mask.flatten().nonzero(as_tuple=True)[0] for name, mask in masks.items()}
    embeddings = {
        name: torch.randn(hidden_dim, indices.numel(), device="cuda", dtype=dtype)
        .t()
        .requires_grad_()
        for name, indices in token_indices.items()
    }
    # Alignment is stateless; avoid constructing the unrelated language model and process groups.
    model = MimoModel.__new__(MimoModel)

    def merge(modality_embeddings):
        return model.align_embeddings_by_token_positions(
            modality_embeddings,
            input_ids,
            {"vision": 1, "audio": 2},
            modality_token_indices=token_indices if precomputed else None,
        )

    assert_replays_bit_exact(
        merge,
        (embeddings,),
        grad_outputs={
            "out": torch.randn(seq_length, batch_size, hidden_dim, device="cuda", dtype=dtype)
        },
        replays=3,
        contention=True,
        what=f"MiMo embedding alignment [{dtype}, precomputed={precomputed}]",
    )
