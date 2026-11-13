# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact Liger CE loss and input gradients under contention."""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module", params=[1, 2, 4], ids=["tp1", "tp2", "tp4"])
def tp_group(request):
    size = request.param
    if Utils.world_size < size or Utils.world_size % size:
        pytest.skip(f"needs a world size divisible by TP={size}")
    with Utils.interleaved_group(size) as group:
        yield group


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_liger_cross_entropy_replays(tp_group, dtype):
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.ops.loss.liger_cross_entropy import liger_cross_entropy

    generator = torch.Generator().manual_seed(
        1912 + Utils.rank % (Utils.world_size // tp_group.size())
    )
    logits = torch.randn(1024, 4, 1024, generator=generator).to("cuda", dtype)
    logits.requires_grad_(True)
    labels = torch.randint(1024 * tp_group.size(), (1024, 4), generator=generator).cuda()
    upstream = torch.randn(4, 1024, generator=generator).cuda().t()

    def run(logits, labels):
        return liger_cross_entropy(logits, labels, tp_group)

    outputs, gradients = assert_replays_bit_exact(
        run,
        (logits, labels),
        grad_outputs={"out": upstream},
        replays=4,
        contention=True,
        what=f"Liger CE[tp={tp_group.size()},dtype={dtype}]",
    )
    loss = outputs["out"]
    assert loss.shape == labels.shape and loss.dtype == torch.float32
    assert gradients["in[0]"].shape == logits.shape and gradients["in[0]"].dtype == dtype
