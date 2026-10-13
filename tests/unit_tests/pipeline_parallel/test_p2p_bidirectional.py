# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify activation/gradient direction matching when PP neighbors coincide."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator, _batched_p2p_ops
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('pipeline_size', [2, 4])
def test_batched_bidirectional_messages_keep_their_direction(pipeline_size: int) -> None:
    """Different payloads to the same peer must retain their logical direction."""
    if Utils.world_size % pipeline_size:
        pytest.skip(f'requires a world size divisible by PP={pipeline_size}')
    Utils.initialize_model_parallel(pipeline_model_parallel_size=pipeline_size)
    try:
        group = parallel_state.get_pipeline_model_parallel_group()
        rank = group.rank()
        previous = torch.distributed.get_global_rank(group, (rank - 1) % pipeline_size)
        following = torch.distributed.get_global_rank(group, (rank + 1) % pipeline_size)
        global_rank = torch.distributed.get_rank()
        activation = torch.full((4, 2, 8), 1000 + global_rank, device='cuda', dtype=torch.float32)
        gradient = torch.full_like(activation, 2000 + global_rank)
        received_activation = torch.empty_like(activation)
        received_gradient = torch.empty_like(gradient)
        requests = _batched_p2p_ops(
            tensor_send_prev=gradient,
            tensor_recv_prev=received_activation,
            tensor_send_next=activation,
            tensor_recv_next=received_gradient,
            group=group,
            prev_pipeline_rank=previous,
            next_pipeline_rank=following,
        )
        for request in requests:
            request.wait()
        torch.testing.assert_close(
            received_activation, torch.full_like(activation, 1000 + previous), rtol=0, atol=0
        )
        torch.testing.assert_close(
            received_gradient, torch.full_like(gradient, 2000 + following), rtol=0, atol=0
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('pipeline_size', [2, 4])
def test_batched_bidirectional_variable_shapes_match_payloads(pipeline_size: int) -> None:
    """Shape exchange and payload exchange must use the same directional ordering."""
    if Utils.world_size % pipeline_size:
        pytest.skip(f'requires a world size divisible by PP={pipeline_size}')
    Utils.initialize_model_parallel(pipeline_model_parallel_size=pipeline_size)
    try:
        group = parallel_state.get_pipeline_model_parallel_group()
        communicator = P2PCommunicator(
            group,
            ModelParallelConfig(
                pipeline_model_parallel_size=pipeline_size,
                pipeline_dtype=torch.float32,
                variable_seq_lengths=True,
                batch_p2p_comm=True,
            ),
        )
        rank = torch.distributed.get_rank()
        activation = torch.full((rank + 2, 1, 8), 1000 + rank, device='cuda', dtype=torch.float32)
        gradient = torch.full((rank + 3, 2, 8), 2000 + rank, device='cuda', dtype=torch.float32)
        received_activation, received_gradient = (
            communicator.send_forward_backward_recv_forward_backward(
                output_tensor=activation,
                input_tensor_grad=gradient,
                recv_prev=True,
                recv_next=True,
                tensor_shape=(4, 1, 8),
            )
        )
        assert communicator.prev_rank is not None
        assert communicator.next_rank is not None
        expected_activation = torch.full(
            (communicator.prev_rank + 2, 1, 8),
            1000 + communicator.prev_rank,
            device='cuda',
            dtype=torch.float32,
        )
        expected_gradient = torch.full(
            (communicator.next_rank + 3, 2, 8),
            2000 + communicator.next_rank,
            device='cuda',
            dtype=torch.float32,
        )
        torch.testing.assert_close(received_activation, expected_activation, rtol=0, atol=0)
        torch.testing.assert_close(received_gradient, expected_gradient, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


class _AffinePipelineChunk(torch.nn.Module):
    """Scalar affine stage for comparing native VPP with an unpartitioned reference."""

    def __init__(self, config: TransformerConfig, vp_stage: int, position: int) -> None:
        super().__init__()
        self.config = config
        self.vp_stage = vp_stage
        self.position = position
        self.model_type = ModelType.encoder_or_decoder
        self.weight = torch.nn.Parameter(torch.tensor(1.0 + position / 10, device="cuda"))
        self.input_tensor = None

    def set_input_tensor(self, input_tensor: list[torch.Tensor | None]) -> None:
        """Receive the actual tensor delivered by the native pipeline schedule."""
        self.input_tensor = input_tensor[0]

    def forward(self, microbatch: int) -> torch.Tensor:
        """Run one stage without replacing pipeline communication or autograd."""
        inputs = (
            torch.full((4, 2, 8), 1.0 + microbatch / 10, device="cuda")
            if self.position == 0
            else self.input_tensor
        )
        return inputs * self.weight + self.position / 20


@pytest.mark.parametrize("pipeline_size", [2, 4])
@pytest.mark.parametrize("forward_only", [False, True])
def test_native_vpp_matches_unpartitioned_forward_backward(
    pipeline_size: int, forward_only: bool, monkeypatch
) -> None:
    """Exercise native VPP bidirectional calls and its forward-only control path."""
    if Utils.world_size % pipeline_size:
        pytest.skip(f"requires a world size divisible by PP={pipeline_size}")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pipeline_size, virtual_pipeline_model_parallel_size=2
    )
    try:
        rank = parallel_state.get_pipeline_model_parallel_rank()
        config = TransformerConfig(
            num_layers=2 * pipeline_size,
            hidden_size=8,
            num_attention_heads=2,
            pipeline_model_parallel_size=pipeline_size,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
        )
        chunks = [_AffinePipelineChunk(config, v, v * pipeline_size + rank) for v in range(2)]
        num_microbatches = 4 * pipeline_size
        simultaneous_sends = []
        original = P2PCommunicator.send_forward_backward_recv_forward_backward

        def trace(self, output_tensor, input_tensor_grad, **kwargs):
            if output_tensor is not None and input_tensor_grad is not None:
                simultaneous_sends.append((output_tensor.shape, input_tensor_grad.shape))
            return original(self, output_tensor, input_tensor_grad, **kwargs)

        monkeypatch.setattr(P2PCommunicator, "send_forward_backward_recv_forward_backward", trace)

        def forward_step(iterator, model):
            output = model(next(iterator))

            def loss_func(tensor):
                loss = tensor.square().mean()
                return loss, {"loss": loss.detach().clone()}

            return output, loss_func

        results = forward_backward_pipelining_with_interleaving(
            forward_step_func=forward_step,
            data_iterator=[iter(range(num_microbatches)) for _ in chunks],
            model=chunks,
            num_microbatches=num_microbatches,
            seq_length=4,
            micro_batch_size=2,
            forward_only=forward_only,
        )
        weights = [
            torch.tensor(1.0 + position / 10, device="cuda", requires_grad=True)
            for position in range(2 * pipeline_size)
        ]
        reference_losses = []
        for microbatch in range(num_microbatches):
            value = torch.full((4, 2, 8), 1.0 + microbatch / 10, device="cuda")
            for position, weight in enumerate(weights):
                value = value * weight + position / 20
            reference_losses.append(value.square().mean())
        reference_loss = torch.stack(reference_losses).mean()
        reference_loss.backward()
        if rank == pipeline_size - 1:
            assert len(results) == num_microbatches
            torch.testing.assert_close(
                torch.stack([item["loss"] for item in results]), torch.stack(reference_losses)
            )
        if forward_only:
            assert not simultaneous_sends
            assert all(chunk.weight.grad is None for chunk in chunks)
        else:
            assert simultaneous_sends
            for chunk in chunks:
                torch.testing.assert_close(chunk.weight.grad, weights[chunk.position].grad)
    finally:
        Utils.destroy_model_parallel()
