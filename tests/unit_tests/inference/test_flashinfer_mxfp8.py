# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.utils import InferenceMode


@pytest.fixture(autouse=True)
def reset_inference_mode():
    InferenceMode.unset_active()
    yield
    InferenceMode.unset_active()


def test_inference_mode_tracks_decode_state():
    InferenceMode.set_active()
    assert not InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() is None

    InferenceMode.set_decode_state(True, 512)
    assert InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() == 512

    InferenceMode.unset_active()
    assert not InferenceMode.is_decode_only()
    assert InferenceMode.decode_token_upper_bound() is None


def test_inference_mode_rejects_invalid_decode_bound():
    with pytest.raises(ValueError, match="must be positive"):
        InferenceMode.set_decode_state(True, 0)


@pytest.mark.parametrize(
    ("token_capacity", "decode_only", "decode_upper_bound", "expected"),
    [
        (None, False, None, (65536, "full")),
        (1024, False, 512, (65536, "full-mixed")),
        (1024, True, 512, (1024, "bounded-decode")),
        (1024, True, 2048, (65536, "full-decode-over-capacity")),
        (131072, True, 512, (65536, "bounded-decode")),
    ],
)
def test_flashinfer_mxfp8_active_row_policy(
    token_capacity, decode_only, decode_upper_bound, expected
):
    pytest.importorskip("flashinfer")
    from megatron.core.inference.moe.flashinfer_mxfp8 import select_routed_mxfp8_active_rows

    assert (
        select_routed_mxfp8_active_rows(
            65536,
            token_capacity=token_capacity,
            decode_only=decode_only,
            decode_token_upper_bound=decode_upper_bound,
        )
        == expected
    )
