"""VPP chunk bookkeeping for typed cross-layer pipeline boundaries."""

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayload,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import TypedP2PCommunicator
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Payload(PipelinePayload):
    tensors: tuple[torch.Tensor, ...]
    spec: PipelinePayloadSpec

    @property
    def tensor_specs(self):
        return self.spec.tensor_specs


def _spec(key: str, size: int) -> PipelinePayloadSpec:
    field = TensorField(key, (size,), torch.float32, "S", True)
    return PipelinePayloadSpec((PipelineTensorSpec.from_field(field),), (0, 0))


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_vpp_chunks_keep_independent_payload_queues(tmp_path):
    if dist.is_initialized():
        pytest.skip("the distributed test runner owns the process group")

    rendezvous = tmp_path / "gloo"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        descriptors = (_spec("chunk-0", 2), _spec("chunk-1", 3))
        plans = tuple(
            PipelinePayloadPlan((descriptor,), (descriptor,)) for descriptor in descriptors
        )
        config = TransformerConfig(
            num_layers=4,
            hidden_size=8,
            num_attention_heads=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=2,
            use_cpu_initialization=True,
        )
        communicator = TypedP2PCommunicator(
            dist.group.WORLD,
            config,
            (lambda tensors, spec: Payload(tensors, spec),) * 2,
            payload_plan=plans,
            device=torch.device("cpu"),
        )
        communicator._transfer = lambda **kwargs: {}

        for chunk_id, descriptor in enumerate(descriptors):
            payload = Payload((torch.ones(descriptor.tensor_specs[0].shape),), descriptor)
            communicator.send_forward(payload, False, send_chunk_id=chunk_id)

        assert [len(queue) for queue in communicator._sent] == [1, 1]
        message = communicator.recv_backward((), False, recv_chunk_id=1)
        assert message.identity[1] == 1
        assert len(communicator._sent[0]) == 1
        assert not communicator._sent[1]
    finally:
        dist.destroy_process_group()
