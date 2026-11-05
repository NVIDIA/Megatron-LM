# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact tiled loss and gradient replay under side-stream contention."""

import pytest
import torch

from megatron.core.models import backends
from megatron.core.ops.cross_entropy import triton_cross_entropy
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded


class _TP1:
    def size(self):
        return 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [4097, 262272])
def test_tiled_cross_entropy_replay(dtype, vocab):
    seeded(4731)
    logits = torch.randn(33, vocab, device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(vocab, (33,), device="cuda")
    labels[::5] = -100
    scale = torch.randn(33, device="cuda")
    scale[::3] = 0
    assert_replays_bit_exact(
        lambda x: triton_cross_entropy(x, labels, _TP1()) * scale,
        (logits,),
        replays=3,
        contention=True,
        what="tiled TP1 cross entropy",
    )


def test_auto_te_cross_entropy_replay():
    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    if not backends._te_has_memory_efficient_cross_entropy():
        pytest.skip("TE >= 2.19 with memory-efficient cross entropy required")
    seeded(4732)
    logits = torch.randn(33, 1, 4097, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(4097, (33, 1), device="cuda")
    labels[::5] = -100
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        assert_replays_bit_exact(
            lambda x: backends.select_cross_entropy(True, "auto")(x, labels, group),
            (logits,),
            replays=3,
            contention=True,
            what="TE memory-efficient cross entropy",
        )
    finally:
        Utils.destroy_model_parallel()
