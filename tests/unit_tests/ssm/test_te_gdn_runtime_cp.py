# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime-head selection through the real Transformer Engine GDN frontend."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE_GDN, TEGatedDeltaNetAttention
from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (HAVE_TE_GDN and torch.cuda.is_available()), reason="requires CUDA and TE GDN"
)


def _torch_te_gdn_op(q, k, v, g, beta, cu_seqlens, **kwargs):
    """Use MCore's reference recurrence behind TE's actual layout and shape validation."""
    outputs, states = [], []
    initial_state = kwargs.pop("initial_state", None)
    bounds = cu_seqlens.tolist()
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = (
            None if initial_state is None else initial_state[index : index + 1].transpose(-1, -2)
        )
        output, final_state = torch_chunk_gated_delta_rule(
            *(tensor[start:end].unsqueeze(0) for tensor in (q, k, v, g, beta)),
            initial_state=state,
            chunk_size=4,
            **kwargs,
        )
        outputs.append(output.squeeze(0))
        if final_state is not None:
            states.append(final_state.squeeze(0).transpose(-1, -2))
    return torch.cat(outputs), torch.stack(states) if states else None


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("build_heads", [4, 8])
def test_te_gdn_runtime_heads_preserve_backend_and_backward(
    monkeypatch, packed, output_final_state, build_heads
):
    """CP1/2/4/2/1 head changes preserve TE validation, cached instances and pending backward."""
    from transformer_engine.pytorch.attention.linear_attention import base as te_base

    Utils.initialize_distributed()
    monkeypatch.setattr(te_base, "_import_linear_attention_op", lambda *args: _torch_te_gdn_op)
    module = TEGatedDeltaNetAttention(build_heads, qk_head_dim=4, value_head_dim=6, layer_number=1)
    default_backend = module.te_attention
    checkpoint_keys = set(module.state_dict())
    seen = []
    original_forward = type(default_backend).forward

    def capture_forward(backend, *args, **kwargs):
        seen.append((backend.num_attention_heads, backend.training, id(backend)))
        return original_forward(backend, *args, **kwargs)

    monkeypatch.setattr(type(default_backend), "forward", capture_forward)
    bounds = torch.tensor([0, 3, 8] if packed else [0, 4, 8], dtype=torch.int32, device="cuda")
    batch, sequence = (1, 8) if packed else (2, 4)
    pending = []
    for index, heads in enumerate((8, 4, 2, 4, 8)):
        module.train(index % 3 == 0)
        inputs = {
            name: (torch.randn(batch, sequence, heads, dim, device="cuda") * 0.05)
            .bfloat16()
            .requires_grad_()
            for name, dim in (("q", 4), ("k", 4), ("v", 6))
        }
        inputs["g"] = torch.full((batch, sequence, heads), -0.1, device="cuda", requires_grad=True)
        inputs["beta"] = torch.full(
            (batch, sequence, heads), 0.2, device="cuda", requires_grad=True
        )
        reference_inputs = {
            name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()
        }
        output, state = module(
            **inputs, cu_seqlens=bounds if packed else None, output_final_state=output_final_state
        )
        expected, expected_state = _torch_te_gdn_op(
            *(
                reference_inputs[name].reshape(-1, heads, dim)
                for name, dim in (("q", 4), ("k", 4), ("v", 6))
            ),
            reference_inputs["g"].reshape(-1, heads),
            reference_inputs["beta"].reshape(-1, heads),
            bounds,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )
        expected = expected.reshape(batch, sequence, heads, 6)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        loss, expected_loss = output.float().square().sum(), expected.float().square().sum()
        if output_final_state:
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            loss = loss + state.square().sum()
            expected_loss = expected_loss + expected_state.square().sum()
        else:
            assert state is None
        pending.append((loss, expected_loss, inputs, reference_inputs))
        assert module.te_attention is default_backend
        assert default_backend.num_attention_heads == build_heads

    # Delay every backward until all runtime geometries have been used.
    for loss, expected_loss, inputs, reference_inputs in pending:
        loss.backward()
        expected_loss.backward()
        for name, tensor in inputs.items():
            assert tensor.grad is not None, name
            torch.testing.assert_close(tensor.grad, reference_inputs[name].grad, rtol=0, atol=0)
    assert [heads for heads, _, _ in seen] == [8, 4, 2, 4, 8]
    assert [training for _, training, _ in seen] == [True, False, False, True, False]
    assert seen[0][2] == seen[-1][2]
    assert seen[1][2] == seen[3][2]
    assert len({identity for _, _, identity in seen}) == 3
    assert set(module.state_dict()) == checkpoint_keys
    assert list(module.named_children()) == [("te_attention", default_backend)]
