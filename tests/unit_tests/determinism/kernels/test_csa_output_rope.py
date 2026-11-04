# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Output-RoPE reconstruction, replay and shared-storage regression coverage."""

import os
import subprocess
import sys
from pathlib import Path
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

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@_requires_cuda
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


@_requires_cuda
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


@pytest.mark.parametrize("missing_triton", [False, True])
def test_csa_output_rope_optional_import(missing_triton):
    # Other existing package imports (e.g. paged-stash) require Triton. Initialize
    # those normally, then reload the actual MLA/helper/CSA modules with Triton
    # blocked. No module under test is replaced by a stub.
    script = r"""
import importlib
import importlib.abc
import sys
from types import SimpleNamespace

import torch

mla_name = "megatron.core.fusions.fused_mla_yarn_rope_apply"
helper_name = "megatron.core.ops.attention.csa.output_rope"
csa_name = "megatron.core.transformer.experimental_attention_variant.csa"
importlib.import_module(csa_name)
if not importlib.import_module(mla_name).HAVE_TRITON:
    sys.exit(77)  # The normal-dependency control requires a supported Triton version.

class BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable", name=fullname)

missing = sys.argv[1] == "True"
if missing:
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            del sys.modules[name]
    sys.meta_path.insert(0, BlockTriton())
    try:
        importlib.import_module("triton")
    except ModuleNotFoundError as exc:
        assert exc.name == "triton"
    else:
        raise AssertionError("Triton was not blocked")

for name in (csa_name, helper_name, mla_name):
    sys.modules.pop(name, None)
mla = importlib.import_module(mla_name)
assert bool(mla.HAVE_TRITON) != missing
csa = importlib.import_module(csa_name)
helper = importlib.import_module(helper_name)
assert csa.OutputRopeParams is helper.OutputRopeParams
assert bool(helper.HAVE_TRITON) != missing

# Exercise the real native, fusion-disabled RoPE fallback on CPU.
from megatron.core.transformer.transformer_config import TransformerConfig
config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                           apply_rope_fusion=False, use_cpu_initialization=True)
# Supply CPU frequencies: the ordinary RotaryEmbedding cache migrates to CUDA.
angles = torch.arange(4).float()[:, None] * torch.tensor([1.0, 0.01])[None, :]
freqs = torch.cat((angles, angles), dim=-1)[:, None, None, :]
rotary = lambda length, packed_seq: freqs[:length]
x = torch.randn(4, 2, 2, 8, requires_grad=True)
before = x.detach().clone()
cp = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
y = csa._apply_rope(x, 4, 4, rotary, config, rotary_seq_len=4, cp_group=cp)
assert y.shape == x.shape and torch.isfinite(y).all()
torch.testing.assert_close(y[..., :4], x[..., :4], rtol=0, atol=0)
y.sum().backward()
assert torch.isfinite(x.grad).all()
torch.testing.assert_close(x, before, rtol=0, atol=0)
flat, grad = torch.randn(8, 2, 8), torch.randn(8, 2, 8)
o, do = helper.undo_output_rope_for_backward(None, flat, grad)
assert o is flat and do is grad

if missing:
    params = helper.OutputRopeParams(torch.ones(4, 4), torch.zeros(4, 4), 4, 4,
                                   sbhd_batch_size=2)
    for call in (lambda: params.apply_(flat), lambda: params.undo(flat),
                 lambda: params.undo(flat, out=torch.empty_like(flat))):
        try:
            call()
        except RuntimeError as exc:
            assert str(exc) == "CSA output RoPE fusion requires Triton."
        else:
            raise AssertionError("Fused RoPE accepted missing Triton")
else:
    kernel = helper._autotuned_mla_rope_reconstruct_output_kernel
    assert type(kernel).__module__.startswith("triton.")
print("CSA/helper import and native fallback passed; missing_triton=", missing)
"""
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[4]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", script, str(missing_triton)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("requires supported Triton for the normal-import control")
    assert result.returncode == 0, result.stdout + result.stderr
