# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step
from megatron.core.transformer.transformer_config import TransformerConfig


class _AutocastProbeSchedulePlan:
    """Stands in for TransformerModelChunkSchedulePlan and records the autocast state.

    combined_forward_backward_step calls type(plan).run(...) inside the context it opens, so a
    plan that reports torch.is_autocast_enabled() from inside run() observes exactly whether the
    combined forward/backward chunk executes under autocast.
    """

    def __init__(self):
        self.autocast_enabled = None

    @staticmethod
    def run(f_schedule_plan, b_schedule_plan, b_grad=None, **kwargs):
        """Record the autocast state seen by the combined chunk."""
        plan = b_schedule_plan if b_schedule_plan is not None else f_schedule_plan
        plan.autocast_enabled = torch.is_autocast_enabled("cuda")
        return None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.autocast('cuda') disables itself without CUDA, so the check cannot fail",
)
def test_combined_1f1b_runs_the_combined_chunk_under_autocast():
    """enable_autocast must reach the combined forward/backward chunk, not just the plan build."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        enable_autocast=True,
    )
    # Precondition, not the assertion under test: CUDA autocast supports only float16/bfloat16 and
    # silently disables itself for anything else, which would make the check below unfalsifiable.
    assert config.autocast_dtype == torch.bfloat16

    plan = _AutocastProbeSchedulePlan()
    b_input_tensor = torch.zeros(2, 2, device="cuda", requires_grad=True)
    b_output_tensor = torch.zeros(2, 2, device="cuda", requires_grad=True)
    b_output_tensor.schedule_plan = plan
    b_output_tensor.loss_func = None

    combined_forward_backward_step(
        forward_step_func=None,
        data_iterator=None,
        f_model=None,
        num_microbatches=1,
        input_tensor=None,
        forward_data_store=[],
        b_model=torch.nn.Identity(),
        b_input_tensor=b_input_tensor,
        b_output_tensor=b_output_tensor,
        b_output_tensor_grad=torch.ones(2, 2, device="cuda"),
        config=config,
    )

    assert plan.autocast_enabled is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.autocast('cuda') disables itself without CUDA, so the check cannot fail",
)
def test_combined_1f1b_leaves_autocast_off_when_not_enabled():
    """The control: without enable_autocast the same chunk must run outside autocast."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        enable_autocast=False,
    )

    plan = _AutocastProbeSchedulePlan()
    b_input_tensor = torch.zeros(2, 2, device="cuda", requires_grad=True)
    b_output_tensor = torch.zeros(2, 2, device="cuda", requires_grad=True)
    b_output_tensor.schedule_plan = plan
    b_output_tensor.loss_func = None

    combined_forward_backward_step(
        forward_step_func=None,
        data_iterator=None,
        f_model=None,
        num_microbatches=1,
        input_tensor=None,
        forward_data_store=[],
        b_model=torch.nn.Identity(),
        b_input_tensor=b_input_tensor,
        b_output_tensor=b_output_tensor,
        b_output_tensor_grad=torch.ones(2, 2, device="cuda"),
        config=config,
    )

    assert plan.autocast_enabled is False
