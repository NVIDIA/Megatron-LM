# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the pretrain_multimodal entry configuration."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.pretrain_multimodal import (
    validate_entry_args,
)


def _entry_args(**overrides):
    base = dict(
        pipeline_model_parallel_size=1,
        mtp_num_layers=0,
        use_packed_sequence=True,
        cuda_graph_impl="none",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        dataset_provider="mock_varlen",
        total_seq_length=None,
        seq_length=4096,
    )
    return SimpleNamespace(**{**base, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # model_provider ignores its pre_process/post_process arguments and
        # builds the whole model on every rank, so PP>1 silently violates the
        # pipeline contract rather than failing.
        ({"pipeline_model_parallel_size": 2}, "does not support pipeline_model_parallel_size"),
        # MTP conflicts only with an EFFECTIVE sequence-parallel layout; the
        # narrower case (MTP without SP) must stay supported, pinned below.
        (
            {"mtp_num_layers": 1, "sequence_parallel": True, "tensor_model_parallel_size": 2},
            "MTP is not supported together with sequence parallelism",
        ),
        # forward_step keeps a runtime guard as defense in depth; this one
        # fails in seconds instead of after multi-node model construction.
        ({"cuda_graph_impl": "local"}, "cuda-graph-impl"),
        # The fixed-shape providers size samples from --total-seq-length while
        # the packer caps at --seq-length; the packer refuses to truncate, so
        # this always dies at step 1. Catch it before the run costs anything.
        (
            {"dataset_provider": "mock", "total_seq_length": 8192, "seq_length": 4096},
            "exceeds --seq-length",
        ),
    ],
)
def test_entry_rejects_unsupported_configurations(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_entry_args(_entry_args(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        # Each drops one conjunct of (mtp and sp and tp>1); SP is a no-op at TP=1.
        pytest.param({"mtp_num_layers": 1, "sequence_parallel": True}, id="sp-at-tp1"),
        pytest.param({"mtp_num_layers": 1, "tensor_model_parallel_size": 2}, id="tp2-no-sp"),
    ],
)
def test_mtp_is_accepted_without_an_effective_sequence_parallel_layout(overrides):
    """The guard must stay narrow: MTP itself is wired through (models/base.py
    passes mtp_block_spec down), so only SP-with-TP>1 may be refused."""
    validate_entry_args(_entry_args(**overrides))


def test_total_seq_length_is_ignored_for_mock_varlen():
    """mock_varlen treats --seq-length as the sole capacity authority, so a
    larger --total-seq-length is inert rather than a misconfiguration."""
    validate_entry_args(
        _entry_args(dataset_provider="mock_varlen", total_seq_length=8192, seq_length=4096)
    )
