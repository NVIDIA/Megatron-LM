# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the internal GDR backend adapter."""

import pytest
import torch

from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk as backend


def _inputs():
    return {
        "q": torch.empty(1, 2, 3, 4),
        "k": torch.empty(1, 2, 3, 4),
        "v": torch.empty(1, 2, 3, 4),
        "g": torch.empty(1, 2, 3),
        "beta": torch.empty(1, 2, 3),
    }


def test_internal_backend_forwards_fla_compatible_arguments(monkeypatch):
    calls = []
    expected = (torch.empty(1), None)

    def implementation(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(backend, "_load_optimized_chunk_gated_delta_rule", lambda: implementation)

    result = backend.chunk_gated_delta_rule(
        **_inputs(),
        scale=0.5,
        output_final_state=True,
        transpose_state_layout=True,
        custom_option="value",
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["scale"] == 0.5
    assert calls[0]["output_final_state"] is True
    assert calls[0]["transpose_state_layout"] is True
    assert calls[0]["custom_option"] == "value"


@pytest.mark.parametrize("option", ["use_beta_sigmoid_in_kernel", "allow_neg_eigval"])
def test_internal_backend_rejects_unsupported_semantics(option):
    with pytest.raises(ValueError, match="internal GDR backend does not support"):
        backend.chunk_gated_delta_rule(**_inputs(), **{option: True})


def test_internal_backend_rejects_conflicting_state_layout_options():
    with pytest.raises(ValueError, match="conflicts with transpose_state_layout"):
        backend.chunk_gated_delta_rule(
            **_inputs(), state_v_first=True, transpose_state_layout=False
        )
