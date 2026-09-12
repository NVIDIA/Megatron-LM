# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU offloading of saved activations under full block-wise activation recompute.

The recompute loop drives Transformer Engine's manual offload controller: checkpoint inputs
are copied to the CPU asynchronously during the forward pass and reloaded a configurable
number of layers ahead in the backward pass.
"""

from unittest import mock

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS = 6
SEQ_LEN = 32
MICRO_BATCH = 2
HIDDEN = 64


def _build_block(
    cpu_offloading: bool,
    num_offload_layers: int = 0,
    prefetch_num_layers: int = 1,
    recompute_num_layers: int = 4,
) -> TransformerBlock:
    # Same seeds for every build so offload and baseline blocks share weights. Parameters
    # are initialized on the CPU, so the CPU generator has to be seeded as well.
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=recompute_num_layers,
        cpu_offloading=cpu_offloading,
        cpu_offloading_num_layers=num_offload_layers,
        cpu_offloading_prefetch_num_layers=prefetch_num_layers,
    )
    block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
    block.train()
    return block


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn((SEQ_LEN, MICRO_BATCH, HIDDEN), device="cuda", generator=generator)
    attention_mask = torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.bool, device="cuda")
    return hidden_states, attention_mask


def _train_step(block: TransformerBlock):
    hidden_states, attention_mask = _inputs()
    hidden_states.requires_grad_(True)
    output = block(hidden_states=hidden_states, attention_mask=attention_mask)
    output.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    grads = [p.grad.detach().clone() for p in block.parameters() if p.grad is not None]
    return output.detach().clone(), hidden_states.grad.detach().clone(), grads


@pytest.mark.skipif(not HAVE_TE, reason="CPU offloading needs Transformer Engine")
class TestRecomputeCpuOffloading:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_manual_controller_only_with_full_recompute(self):
        offload_block = _build_block(cpu_offloading=True, num_offload_layers=2)
        assert offload_block.offload_manual_controller is not None
        baseline = _build_block(cpu_offloading=False)
        assert baseline.offload_manual_controller is None

    @pytest.mark.parametrize(
        "num_offload_layers, prefetch_num_layers, recompute_num_layers",
        [
            # Offload window inside the checkpointed window: only checkpoint inputs move.
            (3, 1, 4),
            # Reload two layers ahead.
            (3, 2, 4),
            # Offload window past the checkpointed window: full activations of layer 4 move.
            (5, 1, 4),
            # Every layer checkpointed, all but the last offloaded.
            (NUM_LAYERS - 1, 1, NUM_LAYERS),
        ],
    )
    def test_matches_baseline(self, num_offload_layers, prefetch_num_layers, recompute_num_layers):
        baseline = _build_block(cpu_offloading=False, recompute_num_layers=recompute_num_layers)
        ref_out, ref_in_grad, ref_grads = _train_step(baseline)

        offload_block = _build_block(
            cpu_offloading=True,
            num_offload_layers=num_offload_layers,
            prefetch_num_layers=prefetch_num_layers,
            recompute_num_layers=recompute_num_layers,
        )
        out, in_grad, grads = _train_step(offload_block)

        # Offloading only moves bytes, so the result has to be bitwise identical.
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(in_grad, ref_in_grad, rtol=0, atol=0)
        assert len(grads) == len(ref_grads) > 0
        for grad, ref_grad in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)

    @pytest.mark.parametrize("prefetch_num_layers", [1, 2])
    def test_offload_and_reload_schedule(self, prefetch_num_layers):
        num_offload_layers = 3
        block = _build_block(
            cpu_offloading=True,
            num_offload_layers=num_offload_layers,
            prefetch_num_layers=prefetch_num_layers,
        )
        controller = block.offload_manual_controller
        events = []

        def _spy(name):
            original = getattr(controller, name)

            def _record(layer_id):
                events.append((name, layer_id))
                return original(layer_id)

            return _record

        with (
            mock.patch.object(controller, "start_offload_layer", _spy("start_offload_layer")),
            mock.patch.object(
                controller,
                "release_activation_forward_gpu_memory",
                _spy("release_activation_forward_gpu_memory"),
            ),
            mock.patch.object(controller, "start_reload_layer", _spy("start_reload_layer")),
        ):
            _train_step(block)

        offloaded = [layer for name, layer in events if name == "start_offload_layer"]
        released = [
            layer for name, layer in events if name == "release_activation_forward_gpu_memory"
        ]
        reloaded = [layer for name, layer in events if name == "start_reload_layer"]
        expected = list(range(num_offload_layers))
        assert offloaded == expected
        assert released == expected
        # Backward walks the layers top-down, so reloads arrive in reverse layer order.
        assert reloaded == expected[::-1]

        # Each layer is released only after its offload started, and reloaded only after it
        # was released; the reload of layer i is issued before its own backward.
        for layer in expected:
            offload_pos = events.index(("start_offload_layer", layer))
            release_pos = events.index(("release_activation_forward_gpu_memory", layer))
            reload_pos = events.index(("start_reload_layer", layer))
            assert offload_pos < release_pos < reload_pos

    def test_second_iteration_matches_baseline(self):
        # The controller's per-layer state must reset cleanly between iterations. Dropout
        # draws fresh masks each step, so compare against a baseline that took the same
        # two steps rather than against the first iteration.
        baseline = _build_block(cpu_offloading=False)
        _train_step(baseline)
        baseline.zero_grad(set_to_none=True)
        ref_out, ref_in_grad, ref_grads = _train_step(baseline)

        block = _build_block(cpu_offloading=True, num_offload_layers=3)
        _train_step(block)
        block.zero_grad(set_to_none=True)
        out, in_grad, grads = _train_step(block)

        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(in_grad, ref_in_grad, rtol=0, atol=0)
        for grad, ref_grad in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
