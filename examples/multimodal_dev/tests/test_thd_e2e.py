# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
from types import SimpleNamespace

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
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long, device=device),
    }


def _make_multimodal_sample(
    seq_len: int,
    num_images: int = 0,
    *,
    base: int = 0,
    zero_loss: bool = False,
    device: str = "cuda",
):
    """Hand-built multimodal sample (no dataset dependency).

    ``num_images`` leading image blocks, each a vision_start (96) plus one
    merged image token (97) with a ``[1, 2, 2]`` grid (4 raw patches of
    pixel_dim 1536); the rest is filler text. Pixel rows carry the image
    ordinal so payload order survives packing assertions.
    """
    assert seq_len >= 2 * num_images
    input_ids = torch.full((seq_len,), 5 + base, dtype=torch.long, device=device)
    for image_index in range(num_images):
        input_ids[2 * image_index] = 96
        input_ids[2 * image_index + 1] = 97
    if num_images:
        pixel_values = (
            torch.arange(num_images, dtype=torch.float, device=device)
            .repeat_interleave(4)
            .unsqueeze(1)
            .expand(4 * num_images, 1536)
            .contiguous()
        )
        image_grid_thw = torch.tensor([[1, 2, 2]] * num_images, dtype=torch.long, device=device)
    else:
        pixel_values = torch.empty((0, 1536), dtype=torch.float, device=device)
        image_grid_thw = torch.empty((0, 3), dtype=torch.long, device=device)
    fill = torch.zeros if zero_loss else torch.ones
    return {
        "input_ids": input_ids,
        "labels": torch.arange(seq_len, dtype=torch.long, device=device) + base + 100,
        "loss_mask": fill(seq_len, dtype=torch.float, device=device),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
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
# Multimodal model — CP must precede sequence-parallel scatter
# ===================================================================


def test_multimodal_model_defers_embedding_sequence_parallel_scatter(monkeypatch):
    """The GPT embedding remains full until multimodal CP partitioning."""
    from examples.multimodal_dev.models import base

    captured = {}

    class _FakeGPT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(base, "GPTModel", _FakeGPT)

    base.MultimodalModel(
        language_config=SimpleNamespace(),
        language_spec=None,
        vision_encoder=None,
        vocab_size=100,
        max_sequence_length=8,
        image_token_id=97,
    )

    assert captured["scatter_embedding_sequence_parallel"] is False


def test_multimodal_forward_partitions_cp_before_scattering_sequence_parallel(monkeypatch):
    """A full multimodal sequence is CP-sharded before its TP/SP shard."""
    from examples.multimodal_dev.models import base

    events = []

    class _FakeVision(torch.nn.Module):
        def forward(self, pixel_values, image_grid_thw):
            return torch.tensor([[100.0, 101.0]])

    class _FakeLanguage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_kwargs = None

        def embedding(self, input_ids, position_ids):
            events.append(("embedding", input_ids.shape[1]))
            return torch.arange(16, dtype=torch.float32).view(8, 1, 2)

        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            events.append(("language", kwargs["decoder_input"].shape[0]))
            return kwargs["decoder_input"]

    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=True)
    model.image_token_id = 97
    model.vision_model = _FakeVision()
    model.language_model = _FakeLanguage()

    monkeypatch.setattr(base.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base.parallel_state, "get_context_parallel_rank", lambda: 0)

    def _fake_cp_index(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
        events.append(("cp", total_tokens))
        assert total_tokens == 8
        return torch.tensor([0, 1, 6, 7], dtype=torch.long)

    def _fake_sp_scatter(tensor):
        events.append(("sp", tensor.shape[0]))
        return tensor[: tensor.shape[0] // 2]

    monkeypatch.setattr(base, "_thd_cp_partition_index", _fake_cp_index)
    monkeypatch.setattr(
        base.tensor_parallel, "scatter_to_sequence_parallel_region", _fake_sp_scatter
    )

    input_ids = torch.tensor([[10, 97, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
    output = model(
        input_ids=input_ids,
        position_ids=torch.arange(8).unsqueeze(0),
        labels=input_ids.clone(),
        loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
        padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        pixel_values=torch.ones(1, 2),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
        packed_seq_params=SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 8])),
    )

    assert events == [("embedding", 8), ("cp", 8), ("sp", 4), ("language", 2)]
    assert output.shape == (2, 1, 2)
    assert output[1, 0].tolist() == [100.0, 101.0]
    assert model.language_model.forward_kwargs["input_ids"].tolist() == [[10, 97, 16, 17]]


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

    def test_multi_image_loader_microbatch_concatenates_thd_vision_payloads(self):
        """A loader microbatch mixing 1/2/4-image samples packs in sample order."""
        image_counts = [1, 2, 4]
        lengths = [13, 17, 23]
        batch = [
            _make_multimodal_sample(length, count, base=1000 * index)
            for index, (count, length) in enumerate(zip(image_counts, lengths))
        ]

        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, sum(lengths))
        assert packed["image_grid_thw"].tolist() == [[1, 2, 2]] * sum(image_counts)
        assert packed["pixel_values"].shape == (4 * sum(image_counts), 1536)
        assert int((packed["input_ids"] == 96).sum().item()) == sum(image_counts)
        assert int((packed["input_ids"] == 97).sum().item()) == sum(image_counts)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 13, 30, 53]
        assert packed["packed_seq_params"].cu_seqlens_q_padded.tolist() == [0, 13, 30, 53]

    def test_mixed_modality_microbatch_packs_empty_and_real_vision_payloads(self):
        """Text-only, interleaved, and vision-dominant zero-loss samples share one buffer."""
        text_only = _make_multimodal_sample(9, 0)
        interleaved = _make_multimodal_sample(13, 1)
        # Vision-dominant zero-loss tail: vision_start + 1 merged token.
        vision_tail = _make_multimodal_sample(2, 1, zero_loss=True)

        assert text_only["pixel_values"].shape == (0, 1536)
        assert vision_tail["input_ids"].numel() == 2

        packed = pack_or_pad_batch(
            [text_only, interleaved, vision_tail], use_packed_sequence=True, device="cuda"
        )

        assert packed["input_ids"].shape == (1, 24)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 9, 22, 24]
        assert packed["pixel_values"].shape == (8, 1536)
        assert packed["image_grid_thw"].tolist() == [[1, 2, 2]] * 2
        assert int((packed["input_ids"] == 96).sum().item()) == 2
        assert int((packed["input_ids"] == 97).sum().item()) == 2
        # The zero-loss vision tail contributes no loss tokens.
        assert packed["loss_mask"][0, 22:].sum().item() == 0.0

    def test_all_text_microbatch_packs_zero_pixel_rows(self):
        """A text-only microbatch keeps empty vision tensors through packing."""
        batch = [
            _make_multimodal_sample(length, 0, base=1000 * index)
            for index, length in enumerate((9, 13))
        ]

        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, 22)
        assert packed["pixel_values"].shape == (0, 1536)
        assert packed["image_grid_thw"].shape == (0, 3)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 9, 22]


class TestSegmentPacking:
    """packed_window samples: per-sample ``seq_lens`` splice into cu_seqlens.

    Each segment is an independent logical sequence with its own CP/SP
    alignment padding, so the physical layout always matches
    ``cu_seqlens_padded`` (v2 spec §4).
    """

    def test_seq_lens_splice_into_cu_seqlens(self):
        multi = _make_sample(8, base=0)
        multi["seq_lens"] = torch.tensor([3, 5], device="cuda")
        plain = _make_sample(4, base=100)
        packed = pack_or_pad_batch([multi, plain], use_packed_sequence=True, device="cuda")

        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 3, 8, 12]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 3, 8, 12]
        assert packed["input_ids"][0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103]

    def test_cp2_pads_each_segment_independently(self, monkeypatch):
        from examples.multimodal_dev import forward_step

        monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 2)
        sample = _make_sample(8, base=0)
        sample["seq_lens"] = torch.tensor([3, 5], device="cuda")
        packed = pack_or_pad_batch([sample], use_packed_sequence=True, device="cuda")

        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 3, 8]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 4, 12]
        # Real padding tokens sit after each internal segment so the tensor
        # layout matches cu_seqlens_padded (P0-2 correctness core).
        assert packed["input_ids"].shape == (1, 12)
        assert packed["input_ids"][0].tolist() == [0, 1, 2, 0, 3, 4, 5, 6, 7, 0, 0, 0]
        assert packed["labels"][0, 3].item() == -100
        assert packed["labels"][0, 9:].eq(-100).all().item()
        assert packed["padding_mask"][0].tolist() == [
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ]

    def test_cp8_alignment_targets(self, monkeypatch):
        from examples.multimodal_dev import forward_step

        monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 8)
        sample = _make_sample(32, base=0)
        sample["seq_lens"] = torch.tensor([17, 15], device="cuda")
        packed = pack_or_pad_batch([sample], use_packed_sequence=True, device="cuda")

        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 17, 32]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 32, 48]
        assert psp.total_tokens == 48

    def test_single_segment_seq_lens_is_equivalent_to_plain(self):
        tagged = _make_sample(6, base=0)
        tagged["seq_lens"] = torch.tensor([6], device="cuda")
        plain = _make_sample(6, base=0)
        lhs = pack_or_pad_batch([tagged], use_packed_sequence=True, device="cuda")
        rhs = pack_or_pad_batch([plain], use_packed_sequence=True, device="cuda")
        assert torch.equal(lhs["input_ids"], rhs["input_ids"])
        assert lhs["packed_seq_params"].cu_seqlens_q.tolist() == (
            rhs["packed_seq_params"].cu_seqlens_q.tolist()
        )

    def test_seq_lens_sum_mismatch_raises(self):
        sample = _make_sample(8, base=0)
        sample["seq_lens"] = torch.tensor([3, 4], device="cuda")
        with pytest.raises(ValueError, match="does not match the sample length"):
            pack_or_pad_batch([sample], use_packed_sequence=True, device="cuda")

    def test_bshd_rejects_multi_segment_samples(self):
        sample = _make_sample(8, base=0)
        sample["seq_lens"] = torch.tensor([3, 5], device="cuda")
        with pytest.raises(ValueError, match="no segment representation"):
            pack_or_pad_batch([sample], use_packed_sequence=False, seq_length=8, device="cuda")

    def test_packed_window_dataset_roundtrip(self):
        from examples.multimodal_dev.data.mock_varlen import PackedWindowQwen35VLDataset

        dataset = PackedWindowQwen35VLDataset(
            num_samples=4,
            seq_length=64,
            seed=1234,
            vocab_size=100,
            image_token_id=97,
            video_token_id=98,
            vision_start_token_id=96,
            window_config={
                "doc_length": {
                    "components": [
                        {
                            "name": "short",
                            "weight": 3,
                            "min": 8,
                            "max": 63,
                            "mean": 24,
                            "sigma": 0.8,
                        },
                        {
                            "name": "long",
                            "weight": 1,
                            "min": 64,
                            "max": 256,
                            "mean": 128,
                            "sigma": 0.5,
                        },
                    ]
                },
                "text_only_document_probability": 0.4,
                "image_poisson_rate_per_1k_text_tokens": 50,
                "image_density_gamma_shape": 1.0,
                "max_boundary_fill_fraction": None,
            },
            image_size_config={"mode": "buckets", "resolutions": [[8, 8], [8, 16]]},
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
        batch = [{key: value.to("cuda") for key, value in dataset[idx].items()} for idx in range(2)]
        segment_counts = [sample["seq_lens"].numel() for sample in batch]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        psp = packed["packed_seq_params"]
        assert len(psp.cu_seqlens_q) == 1 + sum(segment_counts)
        assert psp.total_tokens == 2 * 64  # divisible_by == 1: no physical padding
        expected_images = sum(sample["image_grid_thw"].shape[0] for sample in batch)
        assert packed["image_grid_thw"].shape[0] == expected_images
        assert int((packed["input_ids"] == 96).sum().item()) == expected_images


class TestPackOrPadBatchPackedAlignment:
    """Core packed-padding flags also apply to the multimodal local packer."""

    @pytest.fixture
    def alignment_128(self, monkeypatch):
        """Enable eager 128-token padding with an appended dummy sequence."""
        from examples.multimodal_dev import forward_step

        args = SimpleNamespace(
            sequence_parallel=False,
            pad_packed_seq_alignment=128,
            pad_packed_seq_by_appending_dummy_seq=True,
            max_seqlen_per_dp_cp_rank=128,
            thd_max_packed_sequences=None,
            cuda_graph_impl="none",
        )
        monkeypatch.setattr(forward_step, "get_args", lambda: args)
        return args

    def test_alignment_appends_dummy_sequence(self, alignment_128):
        """A packed 7/11 batch pads to 128 without changing vision payloads."""
        batch = [_make_sample(7, base=0, num_patches=4), _make_sample(11, base=100, num_patches=12)]
        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, 128)
        assert packed["labels"].shape == (1, 128)
        assert packed["loss_mask"].shape == (1, 128)
        assert packed["padding_mask"].shape == (1, 128)
        assert not packed["padding_mask"][0, :18].any().item()
        assert packed["padding_mask"][0, 18:].all().item()
        assert not packed["loss_mask"][0, 18:].any().item()
        assert packed["labels"][0, 18:].eq(-100).all().item()
        assert packed["pixel_values"].shape == (16, 8)
        assert packed["image_grid_thw"].shape == (2, 3)

        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 7, 18, 128]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 7, 18, 128]
        assert psp.max_seqlen_q == 110
        assert psp.total_tokens == 128
        assert psp.pad_between_seqs is False

    def test_alignment_dummy_tail_does_not_add_multi_image_vision_payloads(self, alignment_128):
        """The appended token-only dummy sequence leaves 1/2/4-image payloads untouched."""
        image_counts = [1, 2, 4]
        lengths = [13, 17, 23]
        batch = [
            _make_multimodal_sample(length, count, base=1000 * index)
            for index, (count, length) in enumerate(zip(image_counts, lengths))
        ]
        expected_pixels = torch.cat([sample["pixel_values"] for sample in batch]).cuda()
        expected_grids = torch.cat([sample["image_grid_thw"] for sample in batch]).cuda()

        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, 128)
        assert packed["padding_mask"][0, 53:].all().item()
        assert torch.equal(packed["pixel_values"], expected_pixels)
        assert torch.equal(packed["image_grid_thw"], expected_grids)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 13, 30, 53, 128]
        assert packed["packed_seq_params"].cu_seqlens_q_padded.tolist() == [0, 13, 30, 53, 128]

    def test_alignment_preserves_cp2_per_sample_padding(self, alignment_128, monkeypatch):
        """CP2 aligns each rank to 128 after preserving real sample boundaries."""
        from examples.multimodal_dev import forward_step

        monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 2)
        packed = pack_or_pad_batch(
            [_make_sample(7, base=0), _make_sample(11, base=100)],
            use_packed_sequence=True,
            device="cuda",
        )

        assert packed["input_ids"].shape == (1, 256)
        assert packed["padding_mask"][0, 7].item()
        assert packed["padding_mask"][0, 19:].all().item()
        assert int(packed["padding_mask"].sum().item()) == 238
        assert packed["labels"][0, 20:].eq(-100).all().item()

        psp = packed["packed_seq_params"]
        assert psp.cu_seqlens_q.tolist() == [0, 7, 18, 254]
        assert psp.cu_seqlens_q_padded.tolist() == [0, 8, 20, 256]
        assert psp.max_seqlen_q == 236
        assert psp.total_tokens == 256

        from examples.multimodal_dev.models.base import _thd_cp_partition_index
        from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index

        rank_indices = [
            _thd_cp_partition_index(psp.cu_seqlens_q_padded, 256, 2, rank) for rank in range(2)
        ]
        assert [indices.numel() for indices in rank_indices] == [128, 128]
        covered = torch.cat(rank_indices).sort().values
        assert torch.equal(covered, torch.arange(256, device=covered.device))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=10_001,
            video_token_id=10_002,
            vision_start_token_id=10_003,
            input_ids=packed["input_ids"],
            packed_seq_params=psp,
        )
        expected_dummy_positions = torch.arange(236, device="cuda").expand(3, -1)
        assert position_ids.shape == (3, 1, 256)
        assert torch.equal(position_ids[:, 0, 20:], expected_dummy_positions)
        assert deltas.tolist() == [[0], [0], [0]]

    def test_alignment_requires_dummy_sequence(self, alignment_128):
        """The pre-CP path rejects an uncovered padding tail."""
        alignment_128.pad_packed_seq_by_appending_dummy_seq = False
        with pytest.raises(ValueError, match="appending-dummy-seq"):
            pack_or_pad_batch(
                [_make_sample(7), _make_sample(11)], use_packed_sequence=True, device="cuda"
            )

    def test_packed_thd_rejects_cuda_graph_without_alignment(self, alignment_128):
        """Local THD is unsupported by CUDA Graph even without padding enabled."""
        alignment_128.pad_packed_seq_alignment = None
        alignment_128.cuda_graph_impl = "local"
        with pytest.raises(ValueError, match="does not yet support CUDA Graph"):
            pack_or_pad_batch(
                [_make_sample(7), _make_sample(11)], use_packed_sequence=True, device="cuda"
            )


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
        assert "padding_mask" not in padded
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
        assert padded["padding_mask"].dtype == torch.bool
        assert padded["padding_mask"].tolist() == [
            [False, False, False, False, False, False, False],
            [False, False, False, True, True, True, True],
        ]

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

    def test_multi_image_loader_microbatch_concatenates_bshd_vision_payloads(self):
        """BSHD padding preserves a mixed 1/2/4-image loader microbatch."""
        image_counts = [1, 2, 4]
        lengths = [13, 17, 23]
        batch = [
            _make_multimodal_sample(length, count, base=1000 * index)
            for index, (count, length) in enumerate(zip(image_counts, lengths))
        ]

        padded = pack_or_pad_batch(
            batch, use_packed_sequence=False, seq_length=max(lengths), device="cuda"
        )

        assert padded["input_ids"].shape == (3, 23)
        assert padded["padding_mask"].sum(dim=1).tolist() == [10, 6, 0]
        assert padded["image_grid_thw"].tolist() == [[1, 2, 2]] * sum(image_counts)
        assert padded["pixel_values"].shape == (4 * sum(image_counts), 1536)
        assert int((padded["input_ids"] == 96).sum().item()) == sum(image_counts)
        assert int((padded["input_ids"] == 97).sum().item()) == sum(image_counts)


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

    def test_multi_image_payloads_survive_static_cp_alignment(self, cp2):
        """Static CP padding changes token layout but not 1/2/4-image payload order."""
        image_counts = [1, 2, 4]
        lengths = [13, 17, 23]
        batch = [
            _make_multimodal_sample(length, count, base=1000 * index)
            for index, (count, length) in enumerate(zip(image_counts, lengths))
        ]
        expected_pixels = torch.cat([sample["pixel_values"] for sample in batch]).cuda()
        expected_grids = torch.cat([sample["image_grid_thw"] for sample in batch]).cuda()

        packed = pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        assert packed["input_ids"].shape == (1, 60)
        assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 13, 30, 53]
        assert packed["packed_seq_params"].cu_seqlens_q_padded.tolist() == [0, 16, 36, 60]
        assert torch.equal(packed["pixel_values"], expected_pixels)
        assert torch.equal(packed["image_grid_thw"], expected_grids)

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
        assert padded["padding_mask"].tolist() == [
            [False, False, False, False, False, True, True, True],
            [False, False, False, True, True, True, True, True],
        ]
