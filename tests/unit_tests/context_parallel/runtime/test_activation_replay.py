# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay saved-activation marker propagation as runtime CP token geometry changes."""

import pytest
import torch

import megatron.core.extensions.transformer_engine as te_ext
from megatron.core.fusions.fused_bias_geglu import weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    bytes_equal,
    run_once,
    seeded,
)

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.mark.skipif(
    te_ext._te_mark_grouped_tensor is None,
    reason="Transformer Engine does not provide mark_grouped_tensor",
)
@pytest.mark.parametrize("fp8_input_store", [False, True])
@pytest.mark.parametrize(
    "activation",
    [weighted_bias_swiglu_impl, weighted_bias_quick_geglu_impl],
    ids=["swiglu", "quick_geglu"],
)
def test_paged_stash_marker_survives_runtime_token_geometries(activation, fp8_input_store):
    """Packed microbatch row counts may change; views/casts must keep the stashing marker.

    Exercise the actual fused-activation saved tensor, including FP8 storage and
    QuickGeGLU's clamp/concatenate path, without changing outputs or input/weight
    gradients. CP1/2/4/1 local token counts also check that a preceding microbatch
    cannot mark the next microbatch's unmarked reference.
    """
    seeded()
    width = 1024
    full_x = torch.randn(
        CONTENTION_TOKENS, 1, 2 * width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    full_weights = torch.rand(
        CONTENTION_TOKENS, 1, device="cuda", dtype=torch.float32, requires_grad=True
    )
    saved_dtype = torch.float8_e4m3fn if fp8_input_store else torch.bfloat16
    for cp_size in (1, 2, 4, 1):
        local_tokens = CONTENTION_TOKENS // cp_size
        inputs = (full_x[:local_tokens], full_weights[:local_tokens])

        def fused(x, weights, marked):
            if marked:
                te_ext.mark_grouped_tensor(x)
            saved_inputs = []

            def pack(tensor):
                if tensor.shape == (local_tokens, 2 * width):
                    assert te_ext.is_grouped_tensor_marked(tensor) is marked
                    assert tensor.dtype == saved_dtype
                    saved_inputs.append(tensor)
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = activation(
                    x, None, weights, fp8_input_store=fp8_input_store, clamp_value=3.0
                )
            assert len(saved_inputs) == 1
            return output

        reference = run_once(lambda x, weights: fused(x, weights, False), inputs)
        replay = assert_replays_bit_exact(
            lambda x, weights: fused(x, weights, True),
            inputs,
            replays=3,
            contention=True,
            what=f"paged stash CP{cp_size} token geometry, FP8 store={fp8_input_store}",
        )
        for expected_tensors, actual_tensors in zip(reference, replay):
            assert expected_tensors.keys() == actual_tensors.keys()
            for name in expected_tensors:
                assert bytes_equal(expected_tensors[name], actual_tensors[name]), name
