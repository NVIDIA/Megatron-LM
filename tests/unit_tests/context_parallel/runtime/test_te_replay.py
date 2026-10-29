# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay Transformer Engine's runtime GDN head geometry without persistent mutation."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_TE), reason="needs a GPU and Transformer Engine"
    ),
]


@pytest.mark.parametrize("packed", [False, True])
def test_te_gdn_runtime_head_selection_replays(monkeypatch, packed):
    """Replay TE's real frontend and reference recurrence across CP1/2/4/1 head counts."""
    from megatron.core.extensions.transformer_engine import HAVE_TE_GDN, TEGatedDeltaNetAttention

    if not HAVE_TE_GDN:
        pytest.skip("requires Transformer Engine GDN")
    from tests.unit_tests.ssm.te_gdn_test_utils import patch_te_gdn_op

    Utils.initialize_distributed()
    seeded()
    # Keep TE's layout, gate and head validation; the reference op runs on pre-Hopper GPUs too.
    patch_te_gdn_op(monkeypatch)
    module = TEGatedDeltaNetAttention(8, qk_head_dim=4, value_head_dim=6, layer_number=1)
    default_backend = module.te_attention
    batch, sequence = (1, 8) if packed else (2, 4)
    bounds = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32) if packed else None
    q, k, v = [
        (torch.randn(batch, sequence, 8, dim, device="cuda") * 0.05).bfloat16() for dim in (4, 4, 6)
    ]
    g = torch.full((batch, sequence, 8), -0.1, device="cuda")
    beta = torch.full_like(g, 0.2)
    reference = None
    for heads in (8, 4, 2, 8):
        inputs = tuple(
            tensor[:, :, :heads].detach().requires_grad_() for tensor in (q, k, v, g, beta)
        )

        def run(query, key, value, decay, strength):
            return module(query, key, value, decay, strength, cu_seqlens=bounds)[0]

        result = assert_replays_bit_exact(
            run, inputs, replays=3, contention=True, what=f"TE GDN runtime heads={heads}"
        )
        assert module.te_attention is default_backend
        assert default_backend.num_attention_heads == 8
        assert len(result[1]) == 5
        if reference is None:
            reference = result
        for expected, actual in zip(reference, result):
            assert expected.keys() == actual.keys()
            assert all(
                bytes_equal(tensor[:, :, :heads], actual[name]) for name, tensor in expected.items()
            )
