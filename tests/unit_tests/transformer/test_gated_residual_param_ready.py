# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The gated residual reads its child modules' weights without calling them.

With ``--overlap-param-gather`` DDP publishes a module's parameters in that module's forward
pre-hook, so a reader that bypasses the module must publish them itself through
``ensure_params_ready`` -- otherwise it reads last step's shard or races the all-gather.
"""

import pytest
import torch

from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
from tests.unit_tests.transformer.test_gated_residual import (
    BATCH,
    HIDDEN,
    SEQ,
    STREAMS,
    make_config,
)


class _Recorder:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __call__(self):
        self.calls.append(self.name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("use_combine", [True, False])
def test_gates_publish_child_weights_before_reading_them(use_combine):
    module = GatedResidualModule(make_config(), layer_number=1, use_combine=use_combine).cuda()
    calls = []
    for name, param in module.named_parameters():
        setattr(param, PARAM_READY_CALLBACK_ATTR, _Recorder(name, calls))
    hidden = torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda")

    module(hidden)

    assert sorted(calls) == sorted(name for name, _ in module.named_parameters())
