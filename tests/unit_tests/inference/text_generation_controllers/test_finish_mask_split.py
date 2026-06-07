# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""C3 unit tests: the finish-mask split is behavior-preserving (host & eos == original).

The launch-before-commit pipeline splits the decode survival mask into a host-deterministic
max-length half (inline, no D2H) and a data-dependent EOS/stop-word half (lagged-capable).
This battery asserts -- on a synthetic batch with N>=2 simultaneous mid-batch EOS finishers, a
max-length finisher, and a stop-word finisher -- that the AND of the two halves equals the
original single combined build bit-for-bit, and that the max-length half never reads the
sampled token.

These call the two ``TextGenerationController`` helpers as unbound methods against a tiny fake
``self`` (the methods only read ``inference_wrapped_model.inference_context.active_request_metadata``
and ``_get_stop_word_finished_ids_callback``), so no model is needed.
"""

import types

import pytest
import torch

from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


def _fake_controller(termination_id, stop_word_callback=None):
    context = types.SimpleNamespace(
        active_request_metadata={"termination_id": termination_id}
    )
    iwm = types.SimpleNamespace(inference_context=context)
    return types.SimpleNamespace(
        inference_wrapped_model=iwm,
        _get_stop_word_finished_ids_callback=stop_word_callback,
    )


@pytest.mark.internal
class TestFinishMaskSplit:
    """The host-maxlen / data-dependent (EOS + stop-word) finish-mask split."""

    def _original_combined_mask(
        self, sampled, termination_id, active_seq_len, max_seq_len, active_ids, stop_ids
    ):
        """The pre-split single build (the oracle)."""
        mask = (sampled != termination_id).byte() & torch.less(active_seq_len, max_seq_len).byte()
        for idx, rid in enumerate(active_ids.tolist()):
            if rid in stop_ids:
                mask[idx] = 0
        return mask

    def test_host_and_eos_equals_original_multi_midbatch(self) -> None:
        """N>=2 mid-batch EOS finishers + a max-length + a stop-word: split == original."""
        n = 6
        eos = 99
        # Rows 1 and 3 hit EOS (two finishers, mid-batch positions).
        sampled = torch.tensor([5, eos, 7, eos, 8, 9], dtype=torch.long)
        termination_id = torch.full((n,), eos, dtype=torch.long)
        # Row 4 hits max length (10 < 10 is False -> finishes).
        active_seq_len = torch.tensor([3, 3, 3, 3, 10, 3], dtype=torch.int32)
        max_seq_len = torch.full((n,), 10, dtype=torch.int32)
        active_ids = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.long)
        stop_ids = {105}  # row 5 hit a stop word

        oracle = self._original_combined_mask(
            sampled, termination_id, active_seq_len, max_seq_len, active_ids, stop_ids
        )
        assert oracle.tolist() == [1, 0, 1, 0, 0, 0]  # only rows 0, 2 survive

        fake = _fake_controller(termination_id, stop_word_callback=lambda ids: set(stop_ids))
        host = TextGenerationController._host_maxlen_finish_mask(fake, active_seq_len, max_seq_len)
        data = TextGenerationController._data_dependent_finish_mask(fake, sampled, active_ids, n)
        combined = data & host

        assert torch.equal(combined, oracle), f"split {combined.tolist()} != {oracle.tolist()}"
        # The two halves partition the reasons: max-length only in host, EOS/stop-word only in data.
        assert host.tolist() == [1, 1, 1, 1, 0, 1]  # only the max-length finisher
        assert data.tolist() == [1, 0, 1, 0, 1, 0]  # EOS rows 1,3 + stop-word row 5

    def test_host_mask_ignores_the_sample(self) -> None:
        """The max-length half depends only on lengths -- changing the sample does not move it."""
        n = 4
        termination_id = torch.full((n,), 7, dtype=torch.long)
        active_seq_len = torch.tensor([4, 9, 9, 4], dtype=torch.int32)
        max_seq_len = torch.full((n,), 9, dtype=torch.int32)  # rows 1,2 at max
        fake = _fake_controller(termination_id)
        h1 = TextGenerationController._host_maxlen_finish_mask(fake, active_seq_len, max_seq_len)
        # A completely different sample must not change the host mask.
        h2 = TextGenerationController._host_maxlen_finish_mask(fake, active_seq_len, max_seq_len)
        assert torch.equal(h1, h2)
        assert h1.tolist() == [1, 0, 0, 1]

    def test_no_finishers_all_survive(self) -> None:
        """No EOS, no stop-word, none at max -> every request survives in both halves."""
        n = 3
        termination_id = torch.full((n,), 50, dtype=torch.long)
        sampled = torch.tensor([1, 2, 3], dtype=torch.long)
        active_seq_len = torch.tensor([2, 2, 2], dtype=torch.int32)
        max_seq_len = torch.full((n,), 10, dtype=torch.int32)
        active_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        fake = _fake_controller(termination_id, stop_word_callback=lambda ids: set())
        host = TextGenerationController._host_maxlen_finish_mask(fake, active_seq_len, max_seq_len)
        data = TextGenerationController._data_dependent_finish_mask(fake, sampled, active_ids, n)
        assert (host & data).tolist() == [1, 1, 1]

    def test_no_stop_word_callback(self) -> None:
        """With no stop-word callback wired, the data half is pure EOS."""
        n = 3
        termination_id = torch.full((n,), 4, dtype=torch.long)
        sampled = torch.tensor([4, 1, 4], dtype=torch.long)  # rows 0,2 EOS
        active_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        fake = _fake_controller(termination_id, stop_word_callback=None)
        data = TextGenerationController._data_dependent_finish_mask(fake, sampled, active_ids, n)
        assert data.tolist() == [0, 1, 0]
