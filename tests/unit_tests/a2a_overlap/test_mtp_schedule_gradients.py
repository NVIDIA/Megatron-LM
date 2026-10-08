# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check MTP depth branches independently of the MoE communication backend."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.fine_grained_callables import wrap_mtp_layer_callables
from megatron.core.models.common.utils import TransformerLayerNode


class _Depth(torch.nn.Module):
    def __init__(self, num_groups, detach_heads) -> None:
        super().__init__()
        self.config = SimpleNamespace(sequence_parallel=False, mtp_detach_heads=detach_heads)
        self.eh_proj = torch.nn.Linear(16, 8, bias=False)
        self.groups = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh()) for _ in range(num_groups)]
        )
        self.final_layernorm = torch.nn.LayerNorm(8)

    def _get_embeddings(
        self,
        input_ids,
        position_ids,
        embedding,
        hidden_states,
        packed_seq_params,
        padding_mask,
        mtp_input_mask,
    ):
        input_ids = input_ids.roll(-1, -1)
        position_ids = position_ids.roll(-1, -1)
        decoder_input = embedding(input_ids).transpose(0, 1)
        if self.config.mtp_detach_heads:
            decoder_input = decoder_input.detach()
        return (
            input_ids,
            position_ids,
            padding_mask.roll(-1, -1),
            mtp_input_mask.roll(-1, -1),
            decoder_input,
            hidden_states,
        )

    def _concat_embeddings(self, hidden_states, decoder_input):
        return self.eh_proj(torch.cat((decoder_input, hidden_states), dim=-1))

    def _postprocess(self, hidden_states):
        return self.final_layernorm(hidden_states)


class _Slot(TransformerLayerNode):
    """Exercise the real node's autograd boundaries without CUDA streams."""

    def __init__(self, func, state, first, last) -> None:
        self.submodule = func
        self.chunk_state = state
        self.is_first_layer = first
        self.is_last_layer = last
        self.detached = ()
        self.before_detached = ()

    def run_forward(self, value):
        self.input = value.detach().requires_grad_(value.requires_grad)
        self.output = self.forward_impl(self.input)
        return self.output

    def run_backward(self, grad):
        self.backward_impl((self.output,), (grad,))
        return self.input.grad


@pytest.mark.parametrize("num_depths", [1, 2, 4])
@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("detach_heads", [False, True])
def test_mtp_schedule_preserves_all_depth_gradients(
    monkeypatch, num_depths, num_groups, detach_heads
):
    """Both the prediction head and the next depth must contribute exactly once."""
    monkeypatch.setattr(
        "megatron.core.models.common.fine_grained_callables.get_mtp_layer_offset", lambda *a, **k: 0
    )
    torch.manual_seed(123)
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(16, 8)
    model.depths = torch.nn.ModuleList(
        [_Depth(num_groups, detach_heads) for _ in range(num_depths)]
    )
    model.double()
    model.vp_stage = None
    model.pg_collection = SimpleNamespace(pp=SimpleNamespace(rank=lambda: 0))
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    position_ids = torch.arange(4).expand_as(input_ids)
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    input_mask = ~padding_mask
    original_inputs = (input_ids, position_ids, padding_mask, input_mask)
    source = torch.randn(4, 2, 8, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(num_depths + 1, 4, 2, 16, dtype=torch.float64)

    def loss(outputs):
        result = 0
        for depth_idx, output in enumerate(outputs):
            weight = model.embedding.weight
            if detach_heads and depth_idx > 0:
                weight = weight.detach()
            logits = torch.nn.functional.linear(output, weight)
            result = result + (depth_idx + 1) * (logits - targets[depth_idx]).square().mean()
        return result

    # Ordinary autograd reference: each depth output has two consumers.
    outputs = [source]
    hidden_states = source.detach() if detach_heads else source
    for depth in model.depths:
        input_ids, position_ids, padding_mask, input_mask, embeddings, _ = depth._get_embeddings(
            input_ids, position_ids, model.embedding, hidden_states, None, padding_mask, input_mask
        )
        hidden_states = depth._concat_embeddings(hidden_states, embeddings)
        for group in depth.groups:
            hidden_states = group(hidden_states)
        hidden_states = depth._postprocess(hidden_states)
        outputs.append(hidden_states)
    expected_outputs = torch.cat(outputs).detach().clone()
    loss(outputs).backward()
    expected_input_grad = source.grad.clone()
    expected_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    source.grad = None

    state = SimpleNamespace(
        model=model,
        input_ids=original_inputs[0],
        position_ids=original_inputs[1],
        padding_mask=original_inputs[2],
        mtp_input_mask=original_inputs[3],
        context=None,
        packed_seq_params=None,
    )
    slots = []
    hidden_states = source
    for depth_idx, depth in enumerate(model.depths):
        for group_idx, group in enumerate(depth.groups):
            first = depth_idx == 0 and group_idx == 0
            last = depth_idx == num_depths - 1 and group_idx == num_groups - 1
            callables, _ = wrap_mtp_layer_callables(
                depth,
                [
                    lambda node, value, group=group: group(value),
                    lambda node, value: value,
                    lambda node, value: value,
                    lambda node, value: value,
                    None,
                ],
                {},
                pre_process=group_idx == 0,
                post_process=group_idx == num_groups - 1,
            )
            for func in callables:
                if func is not None:
                    slot = _Slot(func, state, first, last)
                    hidden_states = slot.run_forward(hidden_states)
                    slots.append(slot)

    torch.testing.assert_close(hidden_states, expected_outputs, rtol=1e-12, atol=1e-12)
    # The final model postprocess must see the original conditioning mask and IDs.
    for name, value in zip(
        ("input_ids", "position_ids", "padding_mask", "mtp_input_mask"), original_inputs
    ):
        assert getattr(state, name) is value
    head_input = hidden_states.detach().requires_grad_()
    loss(head_input.chunk(num_depths + 1)).backward()
    grad = head_input.grad
    for slot in reversed(slots):
        grad = slot.run_backward(grad)
    torch.testing.assert_close(grad, expected_input_grad, rtol=1e-10, atol=1e-10)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, expected_grads[name], rtol=1e-10, atol=1e-10)
