# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the BF16-param + int16-remainder update kernel.

``apply_update_with_remainder`` rebuilds each FP32 main value from the BF16 model param and
its remainder, applies the decoupled weight decay and the update, and writes both halves
back. It runs once per step on every Muon-owned matrix, so a drift here changes the whole
trajectory. The kernel is elementwise with unique stores; the replay covers the stored BF16
params, the remainders, and the FP32 main params they encode.
"""

import pytest
import torch

from megatron.core.optimizer import param_remainders
from megatron.core.optimizer.param_remainders import (
    apply_update_with_remainder,
    merge_main_param,
    split_main_param,
)
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and param_remainders.HAVE_TRITON),
    reason="needs a GPU and Triton",
)


@pytest.mark.parametrize("weight_decay_alpha", [None, -1e-4], ids=["no_decay", "weight_decay"])
def test_apply_update_with_remainder_replays(weight_decay_alpha):
    """Two consecutive updates of a matrix whose size is not a multiple of the kernel block."""
    seeded()
    shape = (CONTENTION_TOKENS, 4099)
    main = torch.randn(shape, device="cuda")
    main[0, :64] = 0.0  # exact zeros of both signs keep their bit patterns
    main[0, 32:64] = -0.0
    param = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    remainder = torch.empty(shape, device="cuda", dtype=torch.int16)
    split_main_param(main, param, remainder)
    assert bytes_equal(merge_main_param(param, remainder), main)
    update = torch.randn(shape, device="cuda")

    def fn(param, remainder, update):
        for _ in range(2):
            apply_update_with_remainder(param, remainder, update, -1e-3, weight_decay_alpha)
        return param, remainder, merge_main_param(param, remainder)

    outputs, _ = assert_replays_bit_exact(
        fn,
        (param, remainder, update),
        replays=6,
        backward=False,
        contention=True,
        what="apply_update_with_remainder",
    )
    assert not bytes_equal(outputs["out[2]"], main), "the update left the main params unchanged"
