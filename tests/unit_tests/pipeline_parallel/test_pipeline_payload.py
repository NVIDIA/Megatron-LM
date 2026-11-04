"""Joint backward, gradient activity and ownership checks for typed boundaries."""

from dataclasses import dataclass

import pytest
import torch

from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineGradientMessage,
    PipelinePayload,
    PipelinePayloadSpec,
    PipelineTensorSpec,
    backward_pipeline_payload,
)
from megatron.core.transformer.state_boundary import TensorField


@dataclass
class Payload(PipelinePayload):
    tensors: tuple[torch.Tensor, ...]
    spec: PipelinePayloadSpec

    @property
    def tensor_specs(self):
        return self.spec.tensor_specs


def payload(tensors):
    specs = tuple(
        PipelineTensorSpec.from_field(
            TensorField(
                str(i), tuple(t.shape), t.dtype, "dense", t.dtype.is_floating_point
            )
        )
        for i, t in enumerate(tensors)
    )
    return Payload(tensors, PipelinePayloadSpec(specs, (0, 0)))


def test_joint_backward_shared_roots_aliases_and_frozen_field():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    incoming = payload((x,))
    shared = x.square()
    output = payload((shared, shared, shared.view(2) * 3, torch.ones(2)))
    output.release_output()
    gradients = backward_pipeline_payload(
        incoming, output, (torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2))
    )
    torch.testing.assert_close(gradients[0], 10 * x)
    assert output._backward_state.outputs == ()
    with pytest.raises(RuntimeError, match="already been consumed"):
        backward_pipeline_payload(incoming, output, (torch.ones(2),) * 4)


def test_relay_adds_local_and_downstream_contributions_once():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    incoming = payload((x,))
    outgoing = payload((x.square(), x))
    gradients = backward_pipeline_payload(
        incoming, outgoing, (torch.ones(2), torch.full((2,), 7.0))
    )
    torch.testing.assert_close(gradients[0], 2 * x + 7)


def test_unused_and_numeric_zero_remain_distinct():
    x, unused, zero = (torch.ones(2, requires_grad=True) for _ in range(3))
    incoming = payload((x, unused, zero))
    gradients = backward_pipeline_payload(incoming, (x.square() + zero * 0).sum(), None)
    assert gradients[1] is None
    torch.testing.assert_close(gradients[2], torch.zeros(2))
    message = PipelineGradientMessage(
        tuple(torch.zeros(2) for _ in range(3)),
        torch.tensor([2, 0, 99, 1, 0, 1]),
        (2, 0, 99),
        (0, 1, 2),
    )
    resolved = message.resolve()
    assert resolved[1] is None
    assert resolved[2] is not None


def test_overlapping_microbatches_keep_independent_graphs():
    inputs = [torch.tensor([float(i + 1)], requires_grad=True) for i in range(4)]
    outputs = [payload((x.square(), x * 3)) for x in inputs]
    for output in outputs:
        output.release_output()
    for x, output in zip(inputs, outputs):
        backward_pipeline_payload(None, output, (torch.ones(1), torch.ones(1)))
        torch.testing.assert_close(x.grad, 2 * x + 3)


def test_absent_schema_field_does_not_shift_gradient_slots():
    fields = (
        TensorField("x", (2,), torch.float32, "dense", True),
        TensorField("absent", (2,), torch.float32, "dense", True, False),
        TensorField("index", (2,), torch.int64, "dense", False),
    )
    spec = PipelinePayloadSpec(
        tuple(PipelineTensorSpec.from_field(f) for f in fields), ()
    )
    assert spec.schema.present_spec_indices == (0, 2)
    assert spec.schema.grad_tensor_indices == (0,)
    with pytest.raises(ValueError, match="incompatible"):
        spec.schema.validate((torch.zeros(3), torch.zeros(2, dtype=torch.int64)))


def test_backward_rejects_wrong_microbatch_identity():
    message = PipelineGradientMessage(
        (torch.zeros(2),), torch.tensor([3, 0, 99, 1]), (2, 0, 99), (0,)
    )
    with pytest.raises(ValueError, match="microbatch"):
        message.resolve()
