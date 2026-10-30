# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact Liger CE loss, input gradients and original-logit argmax under contention."""

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
    Utils.initialize_distributed()
    num_groups = Utils.world_size // size
    group = None
    for offset in range(num_groups):
        ranks = list(range(offset, Utils.world_size, num_groups))
        candidate = torch.distributed.new_group(ranks)
        if Utils.rank in ranks:
            group = candidate
    yield group
    torch.distributed.barrier()
    torch.distributed.destroy_process_group(group)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("return_argmax", [False, True])
def test_liger_cross_entropy_replays(tp_group, dtype, return_argmax):
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.ops.loss.liger_cross_entropy import liger_cross_entropy

    generator = torch.Generator().manual_seed(
        1912 + Utils.rank % (Utils.world_size // tp_group.size())
    )
    logits = torch.randn(1024, 4, 1024, generator=generator).to("cuda", dtype)
    # Tie across every shard, exercising the smallest-global-index prediction rule.
    logits[0, :, 3] = 16
    logits.requires_grad_(True)
    labels = torch.randint(1024 * tp_group.size(), (1024, 4), generator=generator).cuda()
    upstream = torch.randn(4, 1024, generator=generator).cuda().t()

    def run(logits, labels):
        return liger_cross_entropy(logits, labels, tp_group, return_argmax=return_argmax)

    outputs, gradients = assert_replays_bit_exact(
        run,
        (logits, labels),
        grad_outputs={"out[0]" if return_argmax else "out": upstream},
        replays=4,
        contention=True,
        what=f"Liger CE[tp={tp_group.size()},dtype={dtype},argmax={return_argmax}]",
    )
    loss = outputs["out[0]" if return_argmax else "out"]
    assert loss.shape == labels.shape and loss.dtype == torch.float32
    assert gradients["in[0]"].shape == logits.shape and gradients["in[0]"].dtype == dtype
    if return_argmax:
        predictions = outputs["out[1]"]
        assert predictions.dtype == torch.int64 and not predictions.requires_grad
        assert torch.equal(predictions[0], torch.full_like(predictions[0], 3))
