# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical equivalence tests for the MoE expert offloading path.

``OffloadingExpertsMLP`` replaces the ordinary expert MLP with a custom autograd function
that keeps expert weights in pinned host memory, stages them to GPU chunk by chunk, runs
grouped GEMMs against a fused weighted-SwiGLU, and writes weight gradients straight into
``main_grad``. None of that is exercised by a shape/finiteness check, so these tests pin
the whole forward and backward against ``grouped_swiglu_mlp_torch_ref`` -- a plain
``torch.matmul`` implementation whose backward is ordinary autograd.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import OffloadingExpertsMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 64
FFN_HIDDEN_SIZE = 128


def grouped_swiglu_mlp_torch_ref(
    w1,
    w2,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference path for the grouped SwiGLU MoE experts.

    Per expert ``i`` (``x_i: [t_i, in]``, ``w1[i]: [in, 2H]``, ``w2[i]: [H, in]``):

        fc1 = x_i @ w1[i]                 # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = (silu(gate) * lin) * probs_i  # [t_i, H]
        y_i = s @ w2[i]                   # [t_i, in]
    """
    # Normalize weights to a list of per-expert 2D tensors (supports both the
    # per-expert parameter list and a stacked [E, in, 2H] / [E, H, in] tensor).
    w1_list = list(torch.unbind(w1, dim=0)) if isinstance(w1, torch.Tensor) else list(w1)
    w2_list = list(torch.unbind(w2, dim=0)) if isinstance(w2, torch.Tensor) else list(w2)

    # torch.split needs python ints; .tolist() syncs if tokens_per_expert is on GPU.
    tokens = (
        tokens_per_expert.tolist()
        if isinstance(tokens_per_expert, torch.Tensor)
        else list(tokens_per_expert)
    )

    x_chunks = torch.split(permuted_local_hidden_states, tokens, dim=0)
    probs_chunks = torch.split(permuted_probs.reshape(-1), tokens, dim=0)

    outputs = []
    for i in range(num_local_experts):
        x_i = x_chunks[i]
        fc1 = torch.matmul(x_i, w1_list[i])  # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = F.silu(gate) * lin  # [t_i, H]
        s = (s * probs_chunks[i].unsqueeze(-1)).to(x_i.dtype)
        outputs.append(torch.matmul(s, w2_list[i]))  # [t_i, in]

    return torch.cat(outputs, dim=0)


def _make_config(num_experts, num_chunks, num_stages=2):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=FFN_HIDDEN_SIZE,
        moe_router_topk=2,
        add_bias_linear=False,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=True,
        expert_tensor_parallel_size=1,
        moe_use_offloading_experts=True,
        moe_offloading_num_chunks=num_chunks,
        moe_offloading_num_stages=num_stages,
    )


def _build_experts(config, num_local_experts):
    """Build OffloadingExpertsMLP and give every expert weight an fp32 main_grad.

    The default ``init_method`` uses std 0.02, which after two chained matmuls leaves
    outputs around 1e-3 -- small enough that any sane absolute tolerance would exceed the
    whole signal and the comparison would pass no matter what the kernels computed. Weights
    are re-drawn at 1/sqrt(fan_in) so activations, gradients and wgrads are all O(1) and the
    tolerances below actually bite.
    """
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    experts = OffloadingExpertsMLP(num_local_experts, config, pg_collection=pg_collection)
    for weight in list(experts.weight1) + list(experts.weight2):
        with torch.no_grad():
            weight.normal_(0.0, weight.shape[0] ** -0.5)
        # gradient_accumulation_fusion accumulates wgrads directly into main_grad, which
        # DDP would allocate in real training.
        weight.main_grad = torch.zeros(
            weight.shape, dtype=torch.float32, device=torch.cuda.current_device()
        )
    return experts


def _assert_close_scaled(actual, expected, name, rtol=2e-2, atol_frac=5e-3):
    """Compare two tensors with an absolute tolerance scaled to ``expected``'s magnitude.

    A fixed ``atol`` is the wrong tool for a bf16 GEMM comparison. Too large and it exceeds
    the whole signal, so the assertion passes no matter what the kernels computed; too small
    and it trips on the handful of near-zero output elements where the accumulated sum
    cancels and the relative error is meaningless. Scaling to ``max|expected|`` keeps the
    bound at a few bf16 ULPs of the tensor's own range, independent of how the weights
    happen to be initialized.
    """
    scale = expected.abs().max().item()
    assert scale > 0.0, f"{name} is all zeros, so the comparison would be vacuous"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol_frac * scale)


def _reference_weights(experts):
    """Detached GPU copies of the offloaded weights, as ordinary autograd leaves."""
    w1 = [
        w.detach().to(torch.cuda.current_device()).clone().requires_grad_(True)
        for w in experts.weight1
    ]
    w2 = [
        w.detach().to(torch.cuda.current_device()).clone().requires_grad_(True)
        for w in experts.weight2
    ]
    return w1, w2


def _make_inputs(tokens_per_expert, dtype=torch.bfloat16):
    total_tokens = int(sum(tokens_per_expert))
    device = torch.cuda.current_device()
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
    probs = torch.rand(total_tokens, device=device, dtype=dtype) + 0.5
    return hidden_states, probs


class TestOffloadingExpertsNumerics:
    """OffloadingExpertsMLP must match a plain-PyTorch grouped SwiGLU MLP."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "num_local_experts,num_chunks",
        [
            (2, 1),  # single chunk: no prefetch pipelining
            (4, 2),  # two chunks per pass: double-buffered staging
            (4, 4),  # one expert per chunk: maximum staging turnover
        ],
    )
    def test_forward_backward_matches_torch_reference(self, num_local_experts, num_chunks):
        """Forward output, input grad, and expert wgrads must all match the reference."""
        torch.manual_seed(1234)
        config = _make_config(num_local_experts, num_chunks)
        experts = _build_experts(config, num_local_experts)

        tokens_per_expert = torch.tensor(
            [7, 3, 11, 5][:num_local_experts], dtype=torch.long, device="cpu"
        )
        hidden_states, probs = _make_inputs(tokens_per_expert)

        # Offloading path.
        x_offload = hidden_states.clone().requires_grad_(True)
        out_offload, bias = experts(x_offload, tokens_per_expert, probs)
        assert bias is None
        out_offload.backward(torch.ones_like(out_offload))

        # Reference path, same weights and same input.
        ref_w1, ref_w2 = _reference_weights(experts)
        x_ref = hidden_states.clone().requires_grad_(True)
        out_ref = grouped_swiglu_mlp_torch_ref(
            ref_w1, ref_w2, x_ref, tokens_per_expert, num_local_experts, probs
        )
        out_ref.backward(torch.ones_like(out_ref))

        # bf16 grouped GEMM vs torch.matmul: both accumulate in fp32, so the gap is
        # dominated by the final bf16 rounding and the fused SwiGLU.
        _assert_close_scaled(out_offload, out_ref, "forward output")
        _assert_close_scaled(x_offload.grad, x_ref.grad, "input gradient")

        # Expert wgrads land in main_grad (fp32), not param.grad. They accumulate over all
        # tokens routed to the expert, so they tolerate a little more drift.
        for i in range(num_local_experts):
            assert experts.weight1[i].grad_added_to_main_grad
            assert experts.weight2[i].grad_added_to_main_grad
            _assert_close_scaled(
                experts.weight1[i].main_grad,
                ref_w1[i].grad.float(),
                f"expert {i} w1 grad",
                rtol=3e-2,
                atol_frac=1e-2,
            )
            _assert_close_scaled(
                experts.weight2[i].main_grad,
                ref_w2[i].grad.float(),
                f"expert {i} w2 grad",
                rtol=3e-2,
                atol_frac=1e-2,
            )

    def test_chunking_does_not_change_results(self):
        """Chunk count is a scheduling knob: it must not change the numbers."""
        torch.manual_seed(4321)
        num_local_experts = 4
        tokens_per_expert = torch.tensor([6, 2, 9, 4], dtype=torch.long, device="cpu")
        hidden_states, probs = _make_inputs(tokens_per_expert)

        results = []
        reference_state = None
        for num_chunks in (1, 2, 4):
            config = _make_config(num_local_experts, num_chunks)
            experts = _build_experts(config, num_local_experts)

            # Every variant must start from identical weights.
            if reference_state is None:
                reference_state = [w.detach().clone() for w in experts.weight1] + [
                    w.detach().clone() for w in experts.weight2
                ]
            else:
                weights = list(experts.weight1) + list(experts.weight2)
                for weight, saved in zip(weights, reference_state):
                    with torch.no_grad():
                        weight.copy_(saved)

            x = hidden_states.clone().requires_grad_(True)
            out, _ = experts(x, tokens_per_expert, probs)
            out.backward(torch.ones_like(out))
            results.append(
                (out.detach().clone(), x.grad.clone(), experts.weight1[0].main_grad.clone())
            )

        base_out, base_dgrad, base_wgrad = results[0]
        for out, dgrad, wgrad in results[1:]:
            _assert_close_scaled(out, base_out, "forward output")
            _assert_close_scaled(dgrad, base_dgrad, "input gradient")
            _assert_close_scaled(wgrad, base_wgrad, "expert 0 w1 grad", atol_frac=1e-2)

    def test_weights_stay_on_pinned_host_memory(self):
        """The whole point of the path: weights never migrate to GPU."""
        config = _make_config(num_experts=2, num_chunks=2)
        experts = _build_experts(config, num_local_experts=2).cuda()

        for weight in list(experts.weight1) + list(experts.weight2):
            assert weight.device.type == "cpu", f"expert weight moved to {weight.device}"
            assert weight.is_pinned(), "expert weight must live in pinned host memory"

    def test_empty_expert_is_handled(self):
        """An expert with zero routed tokens must not corrupt the other experts."""
        torch.manual_seed(99)
        num_local_experts = 2
        config = _make_config(num_local_experts, num_chunks=1)
        experts = _build_experts(config, num_local_experts)

        tokens_per_expert = torch.tensor([0, 8], dtype=torch.long, device="cpu")
        hidden_states, probs = _make_inputs(tokens_per_expert)

        x_offload = hidden_states.clone().requires_grad_(True)
        out_offload, _ = experts(x_offload, tokens_per_expert, probs)
        out_offload.backward(torch.ones_like(out_offload))

        ref_w1, ref_w2 = _reference_weights(experts)
        x_ref = hidden_states.clone().requires_grad_(True)
        out_ref = grouped_swiglu_mlp_torch_ref(
            ref_w1, ref_w2, x_ref, tokens_per_expert, num_local_experts, probs
        )
        out_ref.backward(torch.ones_like(out_ref))

        _assert_close_scaled(out_offload, out_ref, "forward output")
        _assert_close_scaled(x_offload.grad, x_ref.grad, "input gradient")
