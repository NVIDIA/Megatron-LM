# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Output-RoPE reconstruction, replay and shared-storage regression coverage."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_out_of_place
from megatron.core.ops.attention.csa.output_rope import (
    OutputRopeParams,
    undo_output_rope_for_backward,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as fused_csa,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("layout", ["sbhd", "thd", "thd_cp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [2, 64])
def test_output_rope_reconstruction(layout, dtype, heads):
    """Preserve aliases and the NoPE prefix, including B>1 and packed CP positions."""
    seeded()
    cu = None if layout == "sbhd" else torch.tensor([0, 6, 6, 16], device="cuda", dtype=torch.int32)
    positions = (
        torch.tensor([2, 3, 4, 5, 0, 1, 2, 3], device="cuda") if layout == "thd_cp" else None
    )
    rows = 8 if layout == "thd_cp" else 16
    x = torch.randn(rows, heads, 512, device="cuda", dtype=dtype)
    grad = torch.randn_like(x)
    angles = torch.randn(16, 32, device="cuda").repeat(1, 2)
    cos, sin = angles.cos(), angles.sin()
    params = OutputRopeParams(
        cos, sin, 448, 64, cu, position_ids=positions, sbhd_batch_size=2 if cu is None else None
    )
    original, original_grad = x.clone(), grad.clone()

    # Compare with the existing autograd path, including its exact backward arithmetic.
    ref = x.detach().clone().requires_grad_(True)
    ref_view = ref.view(8, 2, heads, 512) if cu is None else ref
    expected = fused_mla_rope_out_of_place(
        ref_view,
        cos,
        sin,
        448,
        64,
        cu,
        0,
        1,
        inverse=True,
        remove_interleaving=True,
        position_ids=positions,
    ).view_as(x)
    expected.backward(grad.clone())

    def run(value, upstream):
        # Exercise the real adapter's padded-head slice before RoPE reconstruction.
        backend_out = value.new_zeros((rows, 64, 512))
        backend_out[:, :heads] = value
        lse = torch.zeros(rows, 64, device=value.device)
        indices = torch.zeros(rows, 1, device=value.device, dtype=torch.int32)
        with (
            patch.object(fused_csa, "_flash_mla_sparse_fwd", return_value=(backend_out, None, lse)),
            patch.object(fused_csa, "_get_head_padding", return_value=64),
            patch.object(fused_csa, "_get_topk_alignment", return_value=1),
        ):
            rotated, _, _ = fused_csa._csa_fwd_flash_mla(
                value, value.new_empty((rows, 512)), indices, 1.0
            )
        params.apply_(rotated)
        saved = rotated.clone()
        restored, restored_grad = undo_output_rope_for_backward(params, rotated, upstream)
        # Forward output may still be saved by the projection, and upstream can alias another edge.
        torch.testing.assert_close(rotated, saved, rtol=0, atol=0)
        torch.testing.assert_close(upstream, original_grad, rtol=0, atol=0)
        return rotated, restored, restored_grad

    actual, restored, restored_grad = run(x, grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored_grad, ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(restored[..., :448], original[..., :448], rtol=0, atol=0)
    # Saved BF16 outputs undergo one extra rounding; reconstruction is not bit-exact to the input.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(restored, original, rtol=tolerance, atol=tolerance)
    assert_replays_bit_exact(
        run, (x, grad), backward=False, contention=True, what="CSA output RoPE"
    )


def test_output_rope_shared_expanded_gradient():
    """A broadcast upstream gradient must remain unchanged after attention backward preparation."""
    seeded()
    angles = torch.randn(8, 32, device="cuda").repeat(1, 2)
    params = OutputRopeParams(angles.cos(), angles.sin(), 448, 64, sbhd_batch_size=1)
    x = params.apply_(torch.randn(8, 3, 512, device="cuda"))
    upstream = torch.ones(1, 1, 512, device="cuda").expand_as(x)
    saved = x.clone()
    _, grad = undo_output_rope_for_backward(params, x, upstream)
    torch.testing.assert_close(x, saved, rtol=0, atol=0)
    assert torch.equal(upstream, torch.ones_like(upstream))
    assert grad.is_contiguous()
