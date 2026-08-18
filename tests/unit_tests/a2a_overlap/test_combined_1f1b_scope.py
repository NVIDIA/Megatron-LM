# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.pipeline_parallel import combined_1f1b


@pytest.mark.parametrize("fp8_recipe", [Fp8Recipe.delayed, Fp8Recipe.custom])
def test_fp8_backward_scope_covers_schedule_and_wgrad(monkeypatch, fp8_recipe):
    events = []

    @contextmanager
    def backward_update_context():
        events.append("scope_enter")
        try:
            yield
        finally:
            events.append("scope_exit")

    class LossNode:
        inputs = ()

        def get_grad(self):
            return object()

        def _release_state(self):
            pass

    class SchedulePlan:
        @staticmethod
        def run(_forward_plan, _backward_plan, **_kwargs):
            events.append("schedule_backward")
            events.append("backward_dw")
            events.append("final_wait")
            return object()

    class Output:
        schedule_plan = SchedulePlan()
        loss_func = LossNode()

    monkeypatch.setattr(
        combined_1f1b,
        "get_fp8_backward_quantization_update_context",
        backward_update_context,
    )
    monkeypatch.setattr(combined_1f1b, "get_fp8_context", lambda _config: nullcontext())
    monkeypatch.setattr(torch.autograd, "backward", lambda *_args, **_kwargs: events.append("loss_backward"))

    config = SimpleNamespace(
        enable_autocast=False,
        fp8="hybrid",
        fp8_recipe=fp8_recipe,
        grad_scale_func=None,
        timers=None,
    )
    combined_1f1b.combined_forward_backward_step(
        forward_step_func=None,
        data_iterator=None,
        f_model=None,
        num_microbatches=1,
        input_tensor=None,
        forward_data_store=[],
        b_model=object(),
        b_input_tensor=None,
        b_output_tensor=Output(),
        b_output_tensor_grad=None,
        config=config,
    )

    assert events == [
        "scope_enter",
        "loss_backward",
        "schedule_backward",
        "backward_dw",
        "final_wait",
        "scope_exit",
    ]
