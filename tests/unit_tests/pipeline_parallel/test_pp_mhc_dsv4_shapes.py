# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real PP/VPP transport at DeepSeek-V4 Flash/Pro mHC activation widths.

These are transport-shape tests, not DeepSeek-V4 attention/MoE model parity.
Four small learnable affine layers send BF16 [sequence, 1, 4 * hidden] tensors
through the production schedules and NCCL P2P. An independent serial PyTorch
reference checks boundary values, input gradients, and every parameter gradient.
TP2 uses distinct sequence shards; TP/DDP gradient finalization is out of scope.

Run with four GPUs (eight also works). The largest individual wire tensor is
128 * 1 * (4 * 7168) * 2 = 7 MiB; no hidden-by-hidden weights are allocated.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]

_SEQUENCE_LENGTH = 128
_NUM_STREAMS = 4
_NUM_LAYERS = 4
_NUM_MICROBATCHES = 4


def _parameter_values(hidden_size, layer):
    # Powers of two keep the transport oracle exact, including BF16 gradients.
    channels = torch.arange(_NUM_STREAMS * hidden_size, device="cuda")
    gain = 2.0 ** ((channels + layer) % 3 - 1).float()
    bias = ((channels // hidden_size + 1) * (layer + 1)).float() / 128
    return gain, bias


def _input_values(hidden_size, microbatch, tp_size, tp_rank, dp_rank):
    length = _SEQUENCE_LENGTH // tp_size
    positions = torch.arange(length, device="cuda") + tp_rank * length
    channels = torch.arange(_NUM_STREAMS * hidden_size, device="cuda")
    # Distinguish microbatches, DP replicas, TP sequence shards, residual streams,
    # and adjacent features. Flattening is stream-major, as for main's mHC.
    values = (
        (microbatch + 1) / 8
        + dp_rank / 32
        + positions[:, None, None] / 1024
        + (channels // hidden_size)[None, None, :] / 16
        + (channels % 8)[None, None, :] / 128
    )
    return values.to(torch.bfloat16).requires_grad_()


def _loss(output, microbatch, hidden_size, tp_rank):
    positions = torch.arange(output.size(0), device=output.device) + tp_rank * output.size(0)
    channels = torch.arange(output.size(-1), device=output.device)
    probe = (
        1
        + microbatch
        + positions[:, None, None] % 4
        + (channels // hidden_size)[None, None, :]
        + (channels % 3)[None, None, :]
    )
    # FP64 reduction makes this dyadic fixture independent of reduction order;
    # the gradient entering the pipeline is still BF16. This is a test objective.
    return (output.double() * probe).sum() / (2**20)


class _AffineStage(MegatronModule):
    """Minimal learnable stage with the native pipeline model interface."""

    def __init__(self, config, layer_ids, vp_stage, tp_rank, dp_rank):
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.layer_ids = layer_ids
        self.vp_stage = vp_stage
        self.pre_process = layer_ids[0] == 0
        self.post_process = layer_ids[-1] == _NUM_LAYERS - 1
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.gains = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for layer in layer_ids:
            gain, bias = _parameter_values(config.hidden_size, layer)
            self.gains.append(torch.nn.Parameter(gain))
            self.biases.append(torch.nn.Parameter(bias))
        self.input_tensor = None
        self.outputs = {}
        self.input_grads = {}
        self.sent_tensors = []

    def set_input_tensor(self, input_tensor):
        """Receive the one-element list passed by the production schedule."""
        assert len(input_tensor) == 1
        self.input_tensor = input_tensor[0]

    def forward(self, microbatch):
        """Apply this stage's layers and observe values before pseudo-deallocation."""
        if self.pre_process:
            hidden_states = _input_values(
                self.config.hidden_size,
                microbatch,
                self.config.tensor_model_parallel_size,
                self.tp_rank,
                self.dp_rank,
            )
        else:
            hidden_states = self.input_tensor

        def save_input_grad(grad):
            self.input_grads[microbatch] = grad.detach().cpu().clone()

        hidden_states.register_hook(save_input_grad)
        for gain, bias in zip(self.gains, self.biases):
            hidden_states = (hidden_states.float() * gain + bias).to(torch.bfloat16)
        self.outputs[microbatch] = hidden_states.detach().cpu().clone()
        if not self.post_process:
            # Keep the actual tensor object so the test also proves that the
            # schedule pseudo-frees its data and uses its real backward path.
            self.sent_tensors.append(hidden_states)
        return hidden_states


def _serial_reference(hidden_size, tp_size, tp_rank, dp_rank):
    """Compute all four layers without schedules, P2P, or stage-module reuse."""
    parameters = []
    for layer in range(_NUM_LAYERS):
        gain, bias = _parameter_values(hidden_size, layer)
        parameters.append((gain.requires_grad_(), bias.requires_grad_()))
    boundary_values = {}
    boundary_grads = {}
    losses = []
    for microbatch in range(_NUM_MICROBATCHES):
        value = _input_values(hidden_size, microbatch, tp_size, tp_rank, dp_rank)
        inputs = []
        for layer, (gain, bias) in enumerate(parameters):
            value.retain_grad()
            inputs.append(value)
            value = (value.float() * gain + bias).bfloat16()
            boundary_values[layer, microbatch] = value.detach().cpu().clone()
        loss = _loss(value, microbatch, hidden_size, tp_rank)
        losses.append(loss.detach().cpu().clone())
        (loss / _NUM_MICROBATCHES).backward()
        for layer, value in enumerate(inputs):
            boundary_grads[layer, microbatch] = value.grad.detach().cpu().clone()
    parameter_grads = [tuple(param.grad.cpu().clone() for param in pair) for pair in parameters]
    return boundary_values, boundary_grads, parameter_grads, losses


@pytest.mark.parametrize("hidden_size", [4096, 7168], ids=["flash", "pro"])
@pytest.mark.parametrize(
    "vp_size,tp_size", [(None, 1), (2, 1), (2, 2)], ids=["pp2", "pp2_vpp2", "pp2_vpp2_tp2_sp"]
)
def test_dsv4_mhc_pipeline_transport(hidden_size, vp_size, tp_size):
    """Check real forward/backward transport for all microbatches and local layers."""
    if Utils.world_size < 2 * tp_size or Utils.world_size % (2 * tp_size):
        pytest.skip("Requires a world size divisible by PP2 * TP")
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=vp_size,
        )
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        config = TransformerConfig(
            num_layers=_NUM_LAYERS,
            hidden_size=hidden_size,
            num_attention_heads=32,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=vp_size,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
            enable_mhc_connections=True,
            mhc_num_residual_streams=_NUM_STREAMS,
            gradient_accumulation_fusion=False,
            deallocate_pipeline_outputs=True,
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
        )
        models = []
        layers_per_chunk = _NUM_LAYERS // (2 * (vp_size or 1))
        for chunk in range(vp_size or 1):
            first_layer = (chunk * 2 + pp_rank) * layers_per_chunk
            layer_ids = list(range(first_layer, first_layer + layers_per_chunk))
            models.append(
                _AffineStage(config, layer_ids, chunk if vp_size else None, tp_rank, dp_rank)
            )

        def forward_step(iterator, model):
            microbatch = next(iterator)
            output = model(microbatch)

            def loss_func(value):
                loss = _loss(value, microbatch, hidden_size, tp_rank)
                return loss, {"loss": loss.detach().clone()}

            return output, loss_func

        iterators = [iter(range(_NUM_MICROBATCHES)) for _ in models]
        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iterators if vp_size else iterators[0],
            model=models if vp_size else models[0],
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=1,
            forward_only=False,
        )
        expected_values, expected_input_grads, expected_parameter_grads, expected_losses = (
            _serial_reference(hidden_size, tp_size, tp_rank, dp_rank)
        )
        expected_shape = (_SEQUENCE_LENGTH // tp_size, 1, _NUM_STREAMS * hidden_size)
        for model in models:
            assert set(model.outputs) == set(range(_NUM_MICROBATCHES))
            assert set(model.input_grads) == set(range(_NUM_MICROBATCHES))
            for microbatch in range(_NUM_MICROBATCHES):
                output = model.outputs[microbatch]
                assert output.shape == expected_shape
                assert output.dtype == torch.bfloat16
                torch.testing.assert_close(
                    output, expected_values[model.layer_ids[-1], microbatch], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    model.input_grads[microbatch],
                    expected_input_grads[model.layer_ids[0], microbatch],
                    rtol=0,
                    atol=0,
                )
            for layer, gain, bias in zip(model.layer_ids, model.gains, model.biases):
                for param, expected in zip((gain, bias), expected_parameter_grads[layer]):
                    assert param.grad is not None
                    torch.testing.assert_close(param.grad.cpu(), expected, rtol=0, atol=0)
            if not model.post_process:
                assert len(model.sent_tensors) == _NUM_MICROBATCHES
                assert all(tensor.numel() == 1 for tensor in model.sent_tensors)
        if any(model.post_process for model in models):
            assert len(losses) == _NUM_MICROBATCHES
            torch.testing.assert_close(
                torch.stack([item["loss"].cpu() for item in losses]),
                torch.stack(expected_losses),
                rtol=0,
                atol=0,
            )
        else:
            assert not losses
    finally:
        Utils.destroy_model_parallel()
