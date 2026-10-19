# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay coverage for the GroupedTensor paged-stash boundaries.

The main mirror uses the kernel replay harness, which is not yet on dev. Keep
these checks in dev's existing correctness bucket until that harness is ported.
These cover marker transparency and expert determinism, not stash/reload itself;
the latter is covered by TestPagedStashingGroupedTensor.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.fusions.fused_bias_geglu import weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.utils import RacingStreams, collect_grads, zero_grads
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def deterministic_algorithms():
    """Restore the caller's deterministic-algorithm settings after each test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _snapshot(fn, inputs, module=None):
    local = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
    if module is not None:
        zero_grads(module)
    output = fn(*local)
    output.backward(torch.ones_like(output))
    result = {"output": output.detach().clone()}
    for index, tensor in enumerate(local):
        if tensor.requires_grad:
            assert tensor.grad is not None
            result[f"input{index}"] = tensor.grad.detach().clone()
    if module is not None:
        grads = collect_grads([module])
        assert grads, "No expert parameter gradients were computed"
        result.update(grads)
    torch.cuda.synchronize()
    return result


def _assert_same_bits(actual, expected):
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        reference = expected[name]
        assert tensor.shape == reference.shape and tensor.dtype == reference.dtype, name
        assert torch.equal(
            tensor.contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
        ), name


def _assert_replays(fn, inputs, module=None):
    reference = _snapshot(fn, inputs, module)
    for _ in range(2):
        with RacingStreams() if module is not None else nullcontext():
            actual = _snapshot(fn, inputs, module)
        _assert_same_bits(actual, reference)


@pytest.mark.skipif(
    te_ext._te_mark_grouped_tensor is None,
    reason="Transformer Engine does not provide mark_grouped_tensor",
)
def test_mark_grouped_tensor_is_numerically_transparent():
    """Marking must preserve values and gradients, including their bit patterns."""
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = _snapshot(lambda x: x.square(), (x,))

    def marked_square(x):
        te_ext.mark_grouped_tensor(x)
        assert te_ext.is_grouped_tensor_marked(x)
        return x.square()

    for _ in range(3):
        _assert_same_bits(_snapshot(marked_square, (x,)), reference)


@pytest.mark.skipif(
    te_ext._te_mark_grouped_tensor is None,
    reason="Transformer Engine does not provide mark_grouped_tensor",
)
@pytest.mark.parametrize(
    "activation",
    [weighted_bias_swiglu_impl, weighted_bias_quick_geglu_impl],
    ids=["swiglu", "quick-geglu"],
)
def test_paged_stash_marked_activation_fusions_replay_bit_exactly(activation):
    """Marker propagation through fused-activation views preserves repeatability."""
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.rand(4096, 1, device="cuda", requires_grad=True)

    def marked_activation(x, weights):
        te_ext.mark_grouped_tensor(x)
        return activation(x, None, weights)

    _assert_replays(marked_activation, (x, weights))


@pytest.mark.skipif(not te_ext.HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
@pytest.mark.parametrize(
    "use_grouped_tensor",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not te_ext._TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR,
                reason="Installed TE GroupedLinear does not expose use_grouped_tensor",
            ),
        ),
    ],
    ids=["multi-stream", "device-initiated"],
)
def test_te_grouped_mlp_replays_on_uneven_experts(use_grouped_tensor):
    """Replay expert outputs and input/parameter gradients on uneven token counts."""
    Utils.initialize_model_parallel(expert_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=2048,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=use_grouped_tensor,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            deterministic_mode=True,
            sequence_parallel=False,
        )
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)
        experts = (
            get_submodules(spec.submodules.mlp)
            .experts(
                num_local_experts=8,
                config=config,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
        )
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        _assert_replays(
            lambda *args: experts(*args)[0], (hidden, tokens_per_expert, probs), experts
        )
    finally:
        Utils.destroy_model_parallel()
