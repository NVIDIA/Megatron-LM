# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for THD / padded batch construction in multimodal_dev.

Exercises the production data path :func:`pack_or_pad_batch`, which
consumes a list of per-sample dicts produced by the dataset's
``__getitem__`` and produces either a packed THD batch (``[1, T]``) or a
padded BSHD batch (``[B, S]``).

``pack_or_pad_batch`` ends with a TP-group broadcast, so these tests
require ``torch.distributed`` to be initialised.  Run via::

    torchrun --nproc-per-node 1 -m pytest -q \\
        examples/multimodal_dev/tests/test_thd_e2e.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.forward_step import _build_packed_seq_params, pack_or_pad_batch
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def _init_model_parallel():
    """Single-rank TP init so pack_or_pad_batch's TP broadcast is a no-op."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _make_sample(
    seq_len: int, *, base: int = 0, num_patches: int = 4, pixel_dim: int = 8, device: str = "cuda"
):
    """Per-sample dict in the shape produced by ``CordV2VLMDataset.__getitem__``.

    1-D ``input_ids`` / ``labels`` / ``loss_mask`` over the sequence dim;
    ``pixel_values`` is ``[num_patches, pixel_dim]``; ``image_grid_thw`` is
    ``[1, 3]``.
    """
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long, device=device) + base,
        "labels": (torch.arange(seq_len, dtype=torch.long, device=device) + base + 100),
        "loss_mask": torch.ones(seq_len, dtype=torch.float, device=device),
        "pixel_values": torch.full((num_patches, pixel_dim), float(base), device=device),
        "image_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long, device=device),
    }


# ===================================================================
# _build_packed_seq_params — pure helper, exercised independently
# ===================================================================


class TestBuildPackedSeqParams:
    """Tests for ``_build_packed_seq_params``."""

    def test_basic(self):
        """Mixed-length sample build sanity check."""
        params = _build_packed_seq_params(torch.tensor([5, 3, 7], dtype=torch.int32), device="cpu")
        assert params.qkv_format == "thd"
        assert params.cu_seqlens_q.tolist() == [0, 5, 8, 15]
        assert params.cu_seqlens_kv.tolist() == [0, 5, 8, 15]
        assert params.max_seqlen_q == 7
        assert params.max_seqlen_kv == 7
        assert params.total_tokens == 15
        assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8, 15]
        assert params.cu_seqlens_kv_padded.tolist() == [0, 5, 8, 15]

    def test_equal_lengths(self):
        """Equal-length samples produce uniform cu_seqlens."""
        params = _build_packed_seq_params(torch.tensor([4, 4, 4], dtype=torch.int32), device="cpu")
        assert params.cu_seqlens_q.tolist() == [0, 4, 8, 12]
        assert params.max_seqlen_q == 4
        assert params.total_tokens == 12

    def test_single_sample(self):
        """Single-sample batch round-trips its own length."""
        params = _build_packed_seq_params(torch.tensor([10], dtype=torch.int32), device="cpu")
        assert params.cu_seqlens_q.tolist() == [0, 10]
        assert params.max_seqlen_q == 10
        assert params.total_tokens == 10

    def test_dtype_is_int32(self):
        """``cu_seqlens_q`` is cast to int32 regardless of input dtype."""
        params = _build_packed_seq_params(torch.tensor([3, 5], dtype=torch.int32), device="cpu")
        assert params.cu_seqlens_q.dtype == torch.int32

    def test_seq_idx_computed(self):
        """``__post_init__`` computes ``seq_idx`` for Mamba compatibility."""
        params = _build_packed_seq_params(torch.tensor([3, 2], dtype=torch.int32), device="cpu")
        assert params.seq_idx is not None
        assert params.seq_idx.shape == (1, 5)
        assert params.seq_idx[0].tolist() == [0, 0, 0, 1, 1]


# ===================================================================
# pack_or_pad_batch — packed (THD) mode
# ===================================================================


class TestPackOrPadBatchPacked:
    """``pack_or_pad_batch(..., use_packed_sequence=True)`` produces ``[1, T]``."""

    def test_equal_lengths(self):
        """Two equal-length samples → packed ``[1, 2S]``."""
        S = 8
        batch = [_make_sample(S, base=0), _make_sample(S, base=1000)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        T = 2 * S
        assert packed["input_ids"].shape == (1, T)
        assert packed["labels"].shape == (1, T)
        assert packed["loss_mask"].shape == (1, T)
        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, S, T]
        assert psp.cu_seqlens_q_padded.tolist() == [0, S, T]
        assert psp.max_seqlen_q == S
        assert psp.total_tokens == T

    def test_variable_lengths(self):
        """Variable-length samples concatenated end-to-end."""
        lens = [5, 8, 3]
        batch = [_make_sample(L, base=i * 1000) for i, L in enumerate(lens)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        T = sum(lens)
        assert packed["input_ids"].shape == (1, T)
        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 5, 13, 16]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 5, 13, 16]
        assert psp.max_seqlen_q == 8
        assert psp.total_tokens == T

    def test_token_order_preserved(self):
        """Sample 0's tokens precede sample 1's tokens in the packed sequence."""
        s0 = _make_sample(3, base=10)
        s1 = _make_sample(3, base=40)
        packed = pack_or_pad_batch([s0, s1], use_packed_sequence=True, device="cuda")
        assert packed["input_ids"][0].tolist() == [10, 11, 12, 40, 41, 42]

    def test_labels_loss_mask_content_preserved(self):
        """labels and loss_mask carry through unchanged when divisible_by=1."""
        s = _make_sample(4, base=0)
        packed = pack_or_pad_batch([s], use_packed_sequence=True, device="cuda")
        assert packed["labels"][0].tolist() == [100, 101, 102, 103]
        assert packed["loss_mask"][0].tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_pixel_values_concatenated(self):
        """``pixel_values`` are concatenated along the patch dim."""
        s0 = _make_sample(4, base=0, num_patches=4, pixel_dim=8)
        s1 = _make_sample(4, base=10, num_patches=6, pixel_dim=8)
        packed = pack_or_pad_batch([s0, s1], use_packed_sequence=True, device="cuda")
        assert packed["pixel_values"].shape == (10, 8)
        assert packed["pixel_values"][:4].eq(0.0).all().item()
        assert packed["pixel_values"][4:].eq(10.0).all().item()

    def test_image_grid_thw_concatenated(self):
        """``image_grid_thw`` rows are concatenated along the first dim."""
        s0 = _make_sample(4, base=0)
        s1 = _make_sample(4, base=10)
        packed = pack_or_pad_batch([s0, s1], use_packed_sequence=True, device="cuda")
        assert packed["image_grid_thw"].shape == (2, 3)

    def test_single_sample_round_trip(self):
        """A single sample packs to its own length."""
        s = _make_sample(7, base=0)
        packed = pack_or_pad_batch([s], use_packed_sequence=True, device="cuda")
        assert packed["input_ids"].shape == (1, 7)
        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 7]
        assert psp.max_seqlen_q == 7
        assert psp.total_tokens == 7


# ===================================================================
# pack_or_pad_batch — padded (BSHD) mode
# ===================================================================


class TestPackOrPadBatchPadded:
    """``pack_or_pad_batch(..., use_packed_sequence=False)`` produces ``[B, S]``."""

    def test_equal_lengths(self):
        """Equal-length samples → ``[B, S]`` without further padding."""
        S = 6
        batch = [_make_sample(S, base=0), _make_sample(S, base=10)]
        padded = pack_or_pad_batch(batch, use_packed_sequence=False, seq_length=S, device="cuda")
        assert padded["input_ids"].shape == (2, S)
        assert padded["labels"].shape == (2, S)
        assert padded["loss_mask"].shape == (2, S)
        # Sample-0 content is preserved verbatim.
        assert padded["input_ids"][0].tolist() == list(range(S))

    def test_pads_short_sample_to_batch_max(self):
        """Shorter sample is right-padded to match the batch max length."""
        long_sample = _make_sample(7, base=0)
        short_sample = _make_sample(3, base=10)
        padded = pack_or_pad_batch(
            [long_sample, short_sample], use_packed_sequence=False, seq_length=7, device="cuda"
        )
        assert padded["input_ids"].shape == (2, 7)
        # Short sample: original [10, 11, 12] then pad zeros.
        assert padded["input_ids"][1].tolist() == [10, 11, 12, 0, 0, 0, 0]
        # labels pad with -100 (ignore index).
        assert padded["labels"][1].tolist() == [110, 111, 112, -100, -100, -100, -100]
        # loss_mask pads with 0.
        assert padded["loss_mask"][1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    def test_seq_length_required(self):
        """``seq_length`` must be provided in padded mode."""
        s = _make_sample(4, base=0)
        with pytest.raises(AssertionError, match="seq_length"):
            pack_or_pad_batch([s], use_packed_sequence=False, seq_length=None, device="cuda")

    def test_pixel_values_concatenated(self):
        """``pixel_values`` concat preserves both samples' patches."""
        s0 = _make_sample(4, base=0, num_patches=4, pixel_dim=8)
        s1 = _make_sample(4, base=10, num_patches=6, pixel_dim=8)
        padded = pack_or_pad_batch([s0, s1], use_packed_sequence=False, seq_length=4, device="cuda")
        assert padded["pixel_values"].shape == (10, 8)
        assert padded["pixel_values"][:4].eq(0.0).all().item()
        assert padded["pixel_values"][4:].eq(10.0).all().item()


# ===================================================================
# pack_or_pad_batch — divisible_by = 4 alignment
# ===================================================================


class TestPackOrPadBatchDivisibleBy4:
    """Per-sample sequence alignment when ``divisible_by = 4``.

    The function computes ``divisible_by`` from the parallel state.  With
    ``world_size=1`` we cannot stand up a real CP=2 group, so we patch
    :func:`mpu.get_context_parallel_world_size` to return 2; the function
    then takes the ``cp_size > 1`` branch and yields
    ``divisible_by = cp_size * 2 = 4`` (no SP).
    """

    @pytest.fixture
    def cp2(self, monkeypatch):
        """Patch CP world size to 2 so ``divisible_by = 4``."""
        from examples.multimodal_dev import forward_step

        monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 2)

    def test_packed_aligned_samples_no_padding(self, cp2):
        """Samples already multiples of 4 → cu_seqlens == cu_seqlens_padded."""
        batch = [_make_sample(8, base=0), _make_sample(4, base=100)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        T = 12
        assert packed["input_ids"].shape == (1, T)
        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 8, 12]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 8, 12]
        assert psp.max_seqlen_q == 8
        assert psp.total_tokens == 12

    def test_packed_misaligned_samples_padded_per_sample(self, cp2):
        """Each sample padded up to the nearest multiple of 4."""
        # lens=[5, 8, 3] → padded=[8, 8, 4] → T_padded = 20.
        batch = [_make_sample(5, base=0), _make_sample(8, base=100), _make_sample(3, base=200)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        T_padded = 20
        assert packed["input_ids"].shape == (1, T_padded)
        psp = packed["packed_seq_params"]
        # cu_seqlens reflects real per-sample lengths.
        assert psp.cu_seqlens_q.tolist() == [0, 5, 13, 16]
        # cu_seqlens_padded reflects per-sample alignment to 4.
        assert psp.cu_seqlens_q_padded.tolist() == [0, 8, 16, 20]
        # max_seqlen comes from the padded lengths.
        assert psp.max_seqlen_q == 8
        assert psp.total_tokens == T_padded

    def test_packed_pad_values(self, cp2):
        """Pad slots filled with input_ids=0, labels=-100, loss_mask=0."""
        # Single sample len=3 → target_len=4 → 1 pad slot at position 3.
        batch = [_make_sample(3, base=10)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, 4)
        assert packed["input_ids"][0].tolist() == [10, 11, 12, 0]
        assert packed["labels"][0].tolist() == [110, 111, 112, -100]
        assert packed["loss_mask"][0].tolist() == [1.0, 1.0, 1.0, 0.0]
        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 3]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 4]
        assert psp.max_seqlen_q == 4
        assert psp.total_tokens == 4

    def test_padded_target_rounded_up_to_multiple_of_4(self, cp2):
        """Padded (BSHD) mode: ``target = ceil(min(max, seq_length) / 4) * 4``."""
        # lens=[5, 3], seq_length=10 → min(5, 10) = 5 → ceil(5/4)*4 = 8.
        long_sample = _make_sample(5, base=0)
        short_sample = _make_sample(3, base=10)
        padded = pack_or_pad_batch(
            [long_sample, short_sample], use_packed_sequence=False, seq_length=10, device="cuda"
        )

        assert padded["input_ids"].shape == (2, 8)
        assert padded["input_ids"][0].tolist() == [0, 1, 2, 3, 4, 0, 0, 0]
        assert padded["input_ids"][1].tolist() == [10, 11, 12, 0, 0, 0, 0, 0]
        assert padded["labels"][1].tolist() == [110, 111, 112, -100, -100, -100, -100, -100]
        assert padded["loss_mask"][1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
