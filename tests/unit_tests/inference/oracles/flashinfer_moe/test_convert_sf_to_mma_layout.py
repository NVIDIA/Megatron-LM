# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``convert_sf_to_mma_layout`` (``flashinfer.cute_dsl.utils``) and its
three call sites in ``prepare_routed_nvfp4_weights``
(``megatron/core/inference/moe/flashinfer_nvfp4.py``): the FC1 and FC2 scale views built with a
new weight set, and the refresh loop that keeps them on a refit.

Contract (from FlashInfer's documented API and the call site; written before reading the body):

* ``convert_sf_to_mma_layout(sf, m, k, num_groups)`` takes the scale bytes of ``num_groups``
  matrices ``[m, k]`` (16-element blocks, ``m % 128 == 0``, ``k % 64 == 0``), each in the cuBLAS
  128x4 layout and stacked as ``[num_groups * m, k / 16]``, and returns a 6-D view
  ``(32, 4, m / 128, 4, k / 64, num_groups)``: entry ``(m % 32, m % 128 // 32, m // 128, c % 4,
  c // 4, g)`` is the scale of group ``g``, row ``m``, scale column ``c``. It is a view (no copy,
  same storage), so writing the stacked bytes changes it; a buffer with the wrong number of
  elements raises ValueError.
* Megatron stores ``fc1_scale = view(fc1_scale_swizzled, m = rows of one FC1 expert (2 I padded
  when gated), k = padded H, groups = E)`` and ``fc2_scale = view(fc2_scale_swizzled, m = padded
  H, k = padded I, groups = E)``. A refresh (``out=``) requantizes into the stacked buffers in
  place, and afterwards each view has its old address and shows the new scales (a CUDA graph
  captured on the view must see the refit).

Oracle: the documented index mapping applied to the logical scale matrices
(``spec.mma_layout_expected``), which never goes through the 128x4 swizzle the stacked bytes
use (``spec.swizzle_128x4``); tolerance class exact. Megatron's call sites are tested with the
quantizer replaced by ``spec.nvfp4_quantize_standin`` (spec codes and 128x4 scales), so they run
on any device with FlashInfer installed; on Blackwell the real quantizer's views are checked by
``test_nvfp4_quantize_expert_weights.py``.

Structure: a reshape to ``(groups, m / 128, k / 64, 32, 4, 4)`` and a permute (FlashInfer
0.7.2.post1, read afterwards); no kernel. Sizes cover one and several 128-row and 4-column tiles
(m 128, 256, 384; k 64, 128, 192, 256, 320), 1, 2, 5 groups, Megatron's gated and plain FC1 with
padded H and I, and production (H 2688, I 1856, 8 local experts). Hardware: any (H100 entry).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

ALIGNMENT = 256  # flashinfer_nvfp4._ALIGNMENT


def _convert():
    return spec.require_flashinfer("flashinfer.cute_dsl.utils").convert_sf_to_mma_layout


def _same_storage(view: torch.Tensor, base: torch.Tensor) -> bool:
    return view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


DIRECT = ou.pairwise_cases(
    {"m": [128, 256, 384], "k": [64, 128, 192, 256, 320], "groups": [1, 2, 5]}
)


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=f"m{c['m']}-k{c['k']}-g{c['groups']}") for c in DIRECT]
)
def test_view_maps_each_scale_to_its_position(case):
    convert = _convert()
    m, k, groups = case["m"], case["k"], case["groups"]
    g = ou.make_generator()
    logical = torch.randint(0, 256, (groups, m, k // 16), generator=g, dtype=torch.uint8)
    logical = logical.to(spec.device())
    sf = torch.stack([spec.swizzle_128x4(s) for s in logical]).reshape(groups * m, k // 16)
    before = sf.clone()
    view = convert(sf, m=m, k=k, num_groups=groups)
    assert tuple(view.shape) == (32, 4, m // 128, 4, k // 64, groups), tuple(view.shape)
    assert _same_storage(view, sf), "the MMA layout must be a view of the scale buffer"
    ou.assert_exact(sf, before, what="the scale buffer is read only")
    ou.assert_exact(view, spec.mma_layout_expected(logical), what="MMA view")
    sf.view(-1)[0] ^= 0xFF  # aliasing: a write to the stacked bytes shows in the view
    logical[0, 0, 0] ^= 0xFF
    ou.assert_exact(view, spec.mma_layout_expected(logical), what="MMA view after a write")


def test_wrong_element_count_raises():
    convert = _convert()
    sf = torch.zeros(256, 16, dtype=torch.uint8, device=spec.device())
    with pytest.raises(ValueError):
        convert(sf, m=256, k=256, num_groups=2)
    with pytest.raises(ValueError):
        convert(sf[:128], m=256, k=256, num_groups=1)


def _standin(monkeypatch):
    """Replace the quantizer Megatron calls by the spec stand-in. Returns the list each call
    appends ``(matrix shape, logical scale bytes [M, K / 16])`` to."""
    from megatron.core.inference.moe import flashinfer_nvfp4

    if not flashinfer_nvfp4.HAVE_FLASHINFER_ROUTED_NVFP4:
        reason = f"FlashInfer per-token NVFP4: {flashinfer_nvfp4._FLASHINFER_NVFP4_IMPORT_ERROR}"
        if ou.session_target_sm() is not None:
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    calls = []

    def standin(a, a_global_sf, *args, **kwargs):
        data, flat = spec.nvfp4_quantize_standin(a, a_global_sf)
        rows, cols = a.shape
        calls.append(((rows, cols), spec.unswizzle_128x4(flat, rows, cols // 16)))
        return data, flat

    monkeypatch.setattr(flashinfer_nvfp4, "nvfp4_quantize", standin)
    return calls


def _stacks(hidden, inter, experts, gated, generator):
    rows = 2 * inter if gated else inter
    w1 = torch.randn(experts, rows, hidden, generator=generator) * 0.05
    w2 = torch.randn(experts, hidden, inter, generator=generator) * 0.05
    return w1.bfloat16().to(spec.device()), w2.bfloat16().to(spec.device())


def _check_views(weights, calls, gated, what):
    """Each view against the mapping of the scale bytes the quantizer returned for each expert
    (the last ``2 E`` calls: FC1 then FC2 per expert)."""
    ph, pi = weights.padded_hidden_size, weights.padded_intermediate_size
    experts = weights.num_local_experts
    fc1_rows = 2 * pi if gated else pi
    recent = calls[-2 * experts :]
    assert [shape for shape, _ in recent] == [(fc1_rows, ph), (ph, pi)] * experts, recent
    for i, (which, m, k) in enumerate((("fc1", fc1_rows, ph), ("fc2", ph, pi))):
        view, base = getattr(weights, f"{which}_scale"), getattr(weights, f"{which}_scale_swizzled")
        assert tuple(view.shape) == (32, 4, m // 128, 4, k // 64, experts), (which, view.shape)
        assert tuple(base.shape) == (experts * m, k // 16), (which, base.shape)
        assert _same_storage(view, base), f"{which}_scale is not a view of {which}_scale_swizzled"
        logical = torch.stack([scales for _, scales in recent[i::2]])
        ou.assert_exact(view, spec.mma_layout_expected(logical), what=f"{what}: {which}_scale")


CALL_SITES = ou.pairwise_cases(
    {
        "shape": [(256, 256), (320, 384), (512, 256), (257, 1000)],
        "gated": [False, True],
        "experts": [1, 3],
    }
)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=f"H{c['shape'][0]}I{c['shape'][1]}-{'swiglu' if c['gated'] else 'relu2'}"
            f"-E{c['experts']}",
        )
        for c in CALL_SITES
    ],
)
def test_prepare_routed_nvfp4_weights_scale_views(case, monkeypatch):
    """Megatron's FC1/FC2 scale views (m, k and groups of each call) and their refresh: the views
    keep their addresses and show the requantized scales."""
    calls = _standin(monkeypatch)
    from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

    g = ou.make_generator()
    (hidden, inter), gated, experts = case["shape"], case["gated"], case["experts"]
    w1, w2 = _stacks(hidden, inter, experts, gated, g)
    weights = prepare_routed_nvfp4_weights(w1, w2)
    ph, pi = spec._ceil_to(hidden, ALIGNMENT), spec._ceil_to(inter, ALIGNMENT)
    assert (weights.padded_hidden_size, weights.padded_intermediate_size) == (ph, pi)
    assert len(calls) == 2 * experts
    _check_views(weights, calls, gated, what="build")
    pointers = (weights.fc1_scale.data_ptr(), weights.fc2_scale.data_ptr())
    w1b, w2b = _stacks(hidden, inter, experts, gated, g)
    assert prepare_routed_nvfp4_weights(w1b, w2b, out=weights) is weights
    assert len(calls) == 4 * experts
    assert (weights.fc1_scale.data_ptr(), weights.fc2_scale.data_ptr()) == pointers
    _check_views(weights, calls, gated, what="refresh")


@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_production_shape_scale_views(gated, monkeypatch):
    """H 2688 (padded 2816), I 1856 (padded 2048), 8 local experts: FC1 m 2048 or 4096, k 2816;
    FC2 m 2816, k 2048."""
    calls = _standin(monkeypatch)
    from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

    w1, w2 = _stacks(2688, 1856, 8, gated, ou.make_generator())
    _check_views(prepare_routed_nvfp4_weights(w1, w2), calls, gated, what="production")


@pytest.mark.sanitizer
def test_sanitizer_small_case(monkeypatch):
    """Small gated case with partial blocks and padding, two experts, then a refresh (the
    compute-sanitizer case: the stacked-buffer slices Megatron writes per expert)."""
    calls = _standin(monkeypatch)
    from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

    g = ou.make_generator()
    w1, w2 = _stacks(257, 96, 2, True, g)
    weights = prepare_routed_nvfp4_weights(w1, w2)
    _check_views(weights, calls, True, what="build")
    w1b, w2b = _stacks(257, 96, 2, True, g)
    prepare_routed_nvfp4_weights(w1b, w2b, out=weights)
    _check_views(weights, calls, True, what="refresh")
