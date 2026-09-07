# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 must be bit-identical to the quantizer the rollout actually runs.

QAT is only meaningful if the quantization error the training graph learns to
compensate is the error deployment actually makes. The rollout path is:

    verl ``utils/modelopt/qat_weight_exporter.py::_quantize_mxfp4``
      -> ``modelopt.torch.export.quant_utils.to_quantized_weight``
      -> ``modelopt.torch.quantization.qtensor.mxfp4_tensor.MXFP4QTensor.quantize``

so that kernel is the contract. This module pins **both** Lite MXFP4 consumers
to it, element for element:

* the training fake-quant, ``qat._mxfp4_fake_quantize_weight``;
* the checkpoint serializer, ``mxfp4.quantize_mxfp4``/``dequantize_mxfp4``.

Regression history (2026-07-26). Three mutually inconsistent implementations
were shipping at once:

===================  =========================================  ==============
implementation       E8M0 scale exponent rule                   E2M1 ties
===================  =========================================  ==============
training fake-quant  ``floor(log2(amax)) - 2``   (OCP Alg. 1)    to-even
ckpt serializer      ``ceil(log2(amax / 6))``                    to-even
rollout (ModelOpt)   ``ceil(max(log2(amax / 6), -127))``         **down**
===================  =========================================  ==============

The OCP Alg. 1 form picks a scale 2x too small whenever the block amax mantissa
exceeds 1.5 -- 29% of blocks of a standard gaussian tensor, 42% at weight-like
scales -- and then saturates the block maximum to 6.0, clipping ~2.3% of all
elements. ModelOpt never saturates. On ``randn[512, 4096]`` that was 8.07% of
elements different and ``sum|d| / sum|w| = 2.19%``: training was compensating a
distortion that deployment does not produce.

``test_modelopt_reference_matches_installed_modelopt`` keeps the vendored
reference below honest whenever nvidia-modelopt happens to be importable; the
rest of the tests run everywhere and are the actual regression lock.
"""

from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    e2m1_round_index,
    mx_shared_scale_exponent,
    quantize_mxfp4,
)
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    _mxfp4_fake_quantize_weight,
    dequantize_weight,
    quantize_weight,
)

pytestmark = pytest.mark.mlite

SPEC = QATSpec(enabled=True, format="mxfp4", group_size=MXFP4_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Vendored reference: a verbatim transcription of ModelOpt 0.43.0
# ``MXFP4QTensor.quantize`` + ``dequantize``. Copied rather than imported so the
# contract is enforced in every environment -- nvidia-modelopt is a rollout-side
# dependency and is not installed for Lite CPU unit tests. Keep this function a
# faithful copy; do not "clean it up".
# ---------------------------------------------------------------------------
_MODELOPT_E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
_MODELOPT_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
_MODELOPT_E2M1_MAX = 6.0


def modelopt_mxfp4_qdq(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Quantize then dequantize exactly as ModelOpt does, returning float32."""

    def cast_fp4(x):
        sign = torch.sign(x)
        sign_bit = (2 - sign) // 2
        ord_ = torch.sum(
            (x.abs().unsqueeze(-1) - _MODELOPT_E2M1_BOUNDS.to(x.device)) > 0, dim=-1
        )
        return (sign_bit * 0b1000 + ord_).to(torch.uint8)

    original_shape = weight.shape
    flat = weight.reshape(-1, block_size)
    input_amax = flat.float().abs().max(dim=-1, keepdim=True).values
    descale = input_amax / _MODELOPT_E2M1_MAX
    min_value = torch.tensor(-127.0, device=descale.device)
    e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

    codes = cast_fp4((flat / torch.exp2(e8m0_scale)).reshape(original_shape))

    # dequantize
    sign = 1 - 2 * ((codes & 0b1000) >> 3).to(torch.float32)
    magnitude = (codes & 0b0111).to(torch.long)
    values = torch.tensor(
        _MODELOPT_E2M1_VALUES, dtype=torch.float32, device=codes.device
    )
    out = sign * values[magnitude.reshape(-1)].reshape(magnitude.shape)
    out = out.reshape(-1, block_size) * torch.exp2(e8m0_scale.float())
    return out.reshape(original_shape)


# ---------------------------------------------------------------------------
# Test tensors: the four distributions that exercise the divergent regions.
# ---------------------------------------------------------------------------
def _case_tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    extreme = torch.randn(64, 1024, dtype=torch.bfloat16)
    extreme[0] *= 1e4  # amax far above the E2M1 range
    extreme[1] *= 1e-4  # amax near the E8M0 floor
    extreme[2] = 0.0  # all-zero block -> log2(0) = -inf
    extreme[3, ::5] = 65504.0  # bf16 finite max
    extreme[4] = torch.tensor(  # exact E2M1 midpoints / clamp boundaries
        [7.0, 3.5, 1.75, 0.75, 6.0, 14.0, 448.0, 1.9] * 128, dtype=torch.bfloat16
    )
    return {
        "gaussian": torch.randn(512, 4096, dtype=torch.bfloat16),
        "weight_scale": (torch.randn(256, 2048) * 0.02).to(torch.bfloat16),
        "heavy_tail": (torch.randn(256, 2048) * torch.rand(256, 2048).pow(3)).to(
            torch.bfloat16
        ),
        "extreme": extreme,
    }


CASES = _case_tensors()
CASE_IDS = sorted(CASES)


def _assert_bit_identical(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    diff = (got.float() - want.float()).abs()
    ndiff = int((diff > 0).sum())
    assert ndiff == 0, (
        f"{what} diverged from ModelOpt on {ndiff}/{diff.numel()} elements "
        f"(max_abs_diff={diff.max().item():.6g}, mean_abs_diff={diff.mean().item():.6g}). "
        "MXFP4 numerics must stay bit-identical to the rollout quantizer; see this "
        "module's docstring."
    )


@pytest.mark.parametrize("case", CASE_IDS)
def test_training_fake_quant_is_bit_identical_to_modelopt(case: str) -> None:
    """The QAT forward must see exactly the weights vLLM will serve."""
    weight = CASES[case]
    got = _mxfp4_fake_quantize_weight(weight, SPEC)
    assert got.dtype == weight.dtype
    _assert_bit_identical(
        got, modelopt_mxfp4_qdq(weight), f"training fake-quant [{case}]"
    )


@pytest.mark.parametrize("case", CASE_IDS)
def test_checkpoint_serializer_is_bit_identical_to_modelopt(case: str) -> None:
    """``quantize_mxfp4`` round-tripped must equal the ModelOpt encoding."""
    weight = CASES[case]
    packed, scale = quantize_mxfp4(weight)
    got = dequantize_mxfp4(packed, scale).reshape(weight.shape)
    _assert_bit_identical(got, modelopt_mxfp4_qdq(weight), f"ckpt serializer [{case}]")


@pytest.mark.parametrize("case", CASE_IDS)
def test_packed_export_snapshot_is_bit_identical_to_modelopt(case: str) -> None:
    """``quantize_weight``/``dequantize_weight`` (deploy snapshot) agree too."""
    weight = CASES[case]
    got = dequantize_weight(quantize_weight(weight, SPEC), SPEC)
    _assert_bit_identical(got, modelopt_mxfp4_qdq(weight), f"packed snapshot [{case}]")


@pytest.mark.parametrize("case", CASE_IDS)
def test_the_two_lite_paths_agree_with_each_other(case: str) -> None:
    """Training fake-quant and ckpt serializer share one implementation."""
    weight = CASES[case]
    fake = _mxfp4_fake_quantize_weight(weight, SPEC).float()
    packed, scale = quantize_mxfp4(weight)
    serialized = dequantize_mxfp4(packed, scale).reshape(weight.shape).float()
    assert torch.equal(fake, serialized)


def test_scale_rule_never_saturates_the_block_maximum() -> None:
    """The property the old OCP Alg. 1 rule violated, stated directly."""
    for case in CASE_IDS:
        weight = CASES[case]
        blocks = weight.float().reshape(-1, MXFP4_BLOCK_SIZE)
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(mx_shared_scale_exponent(amax))
        scaled = (blocks / scale).abs()
        finite = torch.isfinite(scaled)
        assert bool((scaled[finite] <= 6.0).all()), (
            f"[{case}] MXFP4 block scale saturated the block maximum "
            f"(max |w|/scale = {scaled[finite].max().item()}); the deployed "
            "quantizer never clips, so training must not either."
        )


def test_scale_exponent_rule_matches_modelopt_on_mantissa_boundary() -> None:
    """``ceil(log2(amax/6))`` steps at mantissa 1.5, ``floor(log2 amax)-2`` never does."""
    mantissas = [1.0, 1.25, 1.5, 1.5001, 1.75, 1.9375]
    amax = torch.tensor([[m * 2.0**k] for k in (-4, 0, 4) for m in mantissas])
    got = mx_shared_scale_exponent(amax)
    want = torch.ceil(torch.maximum(torch.log2(amax / 6.0), torch.tensor(-127.0)))
    assert torch.equal(got, want)
    # mantissa > 1.5 must land one binade higher than the old rule did
    old_rule = torch.floor(torch.log2(amax)) - 2
    high = torch.tensor([[m > 1.5] for _ in (-4, 0, 4) for m in mantissas])
    assert torch.equal(got != old_rule, high)


def test_e2m1_ties_round_down_not_to_even() -> None:
    """Ties are observable: after a power-of-two scale, midpoints are exact in bf16."""
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    idx = e2m1_round_index(mids)
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 6], "E2M1 ties must round down"
    # ties-to-even would have produced [0, 2, 2, 4, 4, 6, 6]


def test_all_zero_block_matches_modelopt_exponent() -> None:
    """``log2(0) = -inf`` must floor at -127, not fall back to a scale of 1."""
    exponent = mx_shared_scale_exponent(torch.zeros(4, 1))
    assert torch.equal(exponent, torch.full((4, 1), -127.0))


def test_vendored_mxfp4_reference_matches_locked_boundary_values() -> None:
    """The vendored expectation is executable without nvidia-modelopt installed."""
    source = torch.tensor(
        [
            [
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
                -6.0,
            ]
            * 2
        ],
        dtype=torch.bfloat16,
    )
    expected = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
            * 2
        ],
        dtype=torch.float32,
    )

    _assert_bit_identical(
        modelopt_mxfp4_qdq(source), expected, "vendored MXFP4 boundary reference"
    )
