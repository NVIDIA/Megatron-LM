# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""The two contracts that make multimodal dynamic inference correct or silently wrong.

1. **Token-count contract.** The host reserves N placeholder slots; the encoders produce N
   rows. Off by one and every subsequent position shifts, so the language model reads image
   rows at text positions. Nothing raises -- output just degrades. Covered by driving the
   tiler and the prompt expander together, which is how they are used.

2. **Chunk-aware injection.** A prefill chunk boundary can land in the middle of an image's
   token span, and prefix-cache skipping can drop the front of a chunk. The context must pick
   up exactly the rows whose positions fall in the window it is writing. Covered by scattering
   across chunk boundaries and checking which rows land where.

Both run on CPU: they are pure index arithmetic, and the point is to catch the arithmetic.
"""

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig
from megatron.core.inference.multimodal.nemotron_omni.image_processor import (
    DynamicResolutionImageTiler,
)
from megatron.core.inference.multimodal.nemotron_omni.prompt import (
    NemotronOmniPromptExpander,
    video_frame_separators,
)
from megatron.core.models.vision.evs import compute_retained_tokens_count


class _FakeTokenizer:
    """Whitespace tokenizer with reserved ids for the media markers.

    Deliberately merges nothing across boundaries, so any count mismatch the test catches comes
    from the expander's arithmetic rather than from tokenizer behaviour.
    """

    SPECIALS = {
        "<img>": 1,
        "</img>": 2,
        "<image>": 3,
        "<so_start>": 4,
        "<so_embedding>": 5,
        "<so_end>": 6,
    }

    def encode(self, text, add_special_tokens=False):
        """Map reserved markers to their ids and every other word to a stable hash."""
        if text in self.SPECIALS:
            return [self.SPECIALS[text]]
        return [100 + (hash(word) % 1000) for word in text.split()]


@pytest.fixture(name="config")
def _config():
    return NemotronOmniConfig()


@pytest.fixture(name="expander")
def _expander(config):
    return NemotronOmniPromptExpander(_FakeTokenizer(), config)


class TestTokenCountContract:
    """Host-reserved placeholder slots must equal the encoder's row count."""

    @pytest.mark.parametrize(
        "width, height", [(1024, 768), (768, 1024), (4096, 512), (37, 37), (16, 16), (3000, 2000)]
    )
    def test_tiler_grid_is_even_and_within_budget(self, config, width, height):
        """Every grid tiles exactly under pixel shuffle and respects the shared budget."""
        tiler = DynamicResolutionImageTiler(
            patch_size=config.vision.patch_size,
            min_num_patches=config.vision.min_num_patches,
            max_num_patches=config.vision.max_num_patches,
            norm_mean=config.vision.norm_mean,
            norm_std=config.vision.norm_std,
        )
        token_budget = 4096
        (patch_w, patch_h) = tiler.compute_grids([(width, height)], token_budget)[0]

        # Odd axes would make the 2x2 fold drop a row or column of patches.
        assert patch_w % 2 == 0 and patch_h % 2 == 0
        assert patch_w * patch_h <= token_budget * tiler.PATCHES_PER_TOKEN
        # Never upscaled beyond the source's own patch grid, plus the +0.5 rounding slack.
        assert patch_w <= round(width / config.vision.patch_size + 0.5)
        assert patch_h <= round(height / config.vision.patch_size + 0.5)

    def test_shared_budget_shrinks_each_image(self, config):
        """Adding images reduces the per-image grid rather than overflowing the budget."""
        tiler = DynamicResolutionImageTiler(
            patch_size=config.vision.patch_size,
            min_num_patches=64,
            max_num_patches=config.vision.max_num_patches,
            norm_mean=config.vision.norm_mean,
            norm_std=config.vision.norm_std,
        )
        token_budget = 2048
        sizes = [(1024, 1024)] * 4
        grids = tiler.compute_grids(sizes, token_budget)

        total_patches = sum(w * h for w, h in grids)
        assert total_patches <= token_budget * tiler.PATCHES_PER_TOKEN

        solo = tiler.compute_grids([(1024, 1024)], token_budget)[0]
        assert grids[0][0] * grids[0][1] < solo[0] * solo[1]

    def test_image_span_reserves_exactly_the_encoder_row_count(self, expander, config):
        """Placeholder slots equal the predicted token count, and land inside the markers."""
        counts = [12, 5]
        expanded = expander.expand(
            "look at <image> and also <image> please", image_token_counts=counts
        )

        assert len(expanded.embed_positions) == sum(counts)
        assert expanded.embed_positions == sorted(expanded.embed_positions)

        specials = _FakeTokenizer.SPECIALS
        for span, count in zip(expanded.spans, counts):
            assert len(span.embed_positions) == count
            # Every reserved slot holds the context token, bracketed by the span markers.
            for pos in span.embed_positions:
                assert expanded.token_ids[pos] == specials["<image>"]
            assert expanded.token_ids[span.embed_positions[0] - 1] == specials["<img>"]
            assert expanded.token_ids[span.embed_positions[-1] + 1] == specials["</img>"]

    def test_interleaved_modalities_keep_prompt_order(self, expander):
        """Markers are consumed left to right across modalities, not modality by modality."""
        expanded = expander.expand(
            "a <so_embedding> b <image> c", image_token_counts=[4], audio_token_counts=[7]
        )

        assert [span.modality for span in expanded.spans] == ["audio", "image"]
        assert expanded.embed_positions == sorted(expanded.embed_positions)
        assert len(expanded.embed_positions) == 11

    def test_placeholder_count_mismatch_is_rejected(self, expander):
        """A prompt with fewer markers than media items fails loudly, not silently."""
        with pytest.raises(AssertionError, match="placeholder"):
            expander.expand("only one <image> here", image_token_counts=[4, 4])

    def test_video_separators_match_tubelet_count(self, config):
        """One separator per tubelet, including the padded tail tubelet."""
        temporal = config.vision.video_temporal_patch_size
        # 5 frames at temporal patch size 2 -> 3 tubelets, the last padded by frame repetition.
        separators = video_frame_separators(
            list(range(5)), frame_duration_ms=33, temporal_patch_size=temporal
        )

        assert len(separators) == 3
        # Groups after the first are newline-prefixed to match the training format.
        assert not separators[0].startswith("\n")
        assert all(sep.startswith("\n") for sep in separators[1:])
        # Frames within a tubelet are joined, and only the first is capitalized.
        assert " and frame 2 sampled at" in separators[0]

    def test_video_span_reserves_evs_pruned_counts(self, expander):
        """After EVS the span must reserve the *retained* counts, not the pre-pruning ones."""
        tokens_per_frame, num_tubelets, pruning_rate = 256, 8, 0.7
        retained = compute_retained_tokens_count(tokens_per_frame, num_tubelets, pruning_rate)
        # EVS keeps at least one full frame, so the first tubelet survives intact.
        assert retained >= tokens_per_frame

        per_tubelet = [retained - (num_tubelets - 1)] + [1] * (num_tubelets - 1)
        expanded = expander.expand(
            "watch <video>",
            video_plans=[
                {
                    "tokens_per_tubelet": per_tubelet,
                    "separators": [f"Frame {i}: " for i in range(num_tubelets)],
                }
            ],
        )

        assert len(expanded.embed_positions) == retained


class TestChunkAwareInjection:
    """Scatter must select rows by absolute prompt position, not by chunk-relative offset."""

    HIDDEN = 8
    MAX_TOKENS = 64

    def _context(self):
        """A context with only the multimodal buffers wired up.

        Bypasses `__init__` deliberately: a real context needs a KV cache, a block allocator,
        and a distributed group, none of which the scatter arithmetic touches.
        """
        context = DynamicInferenceContext.__new__(DynamicInferenceContext)
        context.enable_multimodal = True
        context.active_token_count = 0
        context.mm_token_count = 0
        context.token_to_mm_embedding = torch.zeros(self.MAX_TOKENS, 1, self.HIDDEN)
        context.token_to_is_mm = torch.zeros(self.MAX_TOKENS, 1, 1, dtype=torch.bool)
        return context

    def _request(self, positions):
        """A stand-in request whose row `i` is filled with the value `i + 1`."""
        rows = torch.arange(1, len(positions) + 1, dtype=torch.float32)
        return type(
            "_Req",
            (),
            {
                "mm_embeddings": rows.unsqueeze(-1).expand(len(positions), self.HIDDEN),
                "mm_embed_positions": torch.tensor(positions, dtype=torch.int64),
            },
        )()

    def test_chunk_boundary_inside_an_image_span(self):
        """An image split across two chunks contributes its rows to both, in order."""
        # Rows at prompt positions 4..11; chunks are [0, 8) then [8, 16).
        positions = list(range(4, 12))
        request = self._request(positions)
        context = self._context()

        context.scatter_multimodal_embeddings(request, chunk_start=0, chunk_length=8)
        assert context.mm_token_count == 4
        # Positions 4..7 land at slots 4..7 of the first chunk.
        assert context.token_to_is_mm[:, 0, 0].tolist()[:8] == [False] * 4 + [True] * 4
        assert context.token_to_mm_embedding[4, 0, 0].item() == 1.0
        assert context.token_to_mm_embedding[7, 0, 0].item() == 4.0

        # Second chunk: token slots restart at 0 while prompt positions continue at 8.
        context.clear_multimodal_mask()
        context.active_token_count = 0
        context.scatter_multimodal_embeddings(request, chunk_start=8, chunk_length=8)
        assert context.mm_token_count == 4
        assert context.token_to_is_mm[:4, 0, 0].tolist() == [True] * 4
        assert context.token_to_mm_embedding[0, 0, 0].item() == 5.0
        assert context.token_to_mm_embedding[3, 0, 0].item() == 8.0

    def test_rows_are_offset_by_active_token_count(self):
        """A prefill chunk appended after decode tokens writes past them."""
        context = self._context()
        context.active_token_count = 10
        request = self._request([0, 1, 2])

        context.scatter_multimodal_embeddings(request, chunk_start=0, chunk_length=4)

        assert context.token_to_is_mm[:10, 0, 0].tolist() == [False] * 10
        assert context.token_to_is_mm[10:13, 0, 0].tolist() == [True] * 3

    def test_prefix_skip_window_selects_no_rows(self):
        """A chunk whose window contains no media positions is a no-op."""
        context = self._context()
        request = self._request([0, 1, 2])

        context.scatter_multimodal_embeddings(request, chunk_start=8, chunk_length=8)

        assert context.mm_token_count == 0
        assert not context.token_to_is_mm.any()

    def test_apply_is_identity_when_no_media_is_staged(self):
        """The masked overwrite runs unconditionally, so an empty mask must not perturb input.

        Guards the CUDA-graph decision: injection is never branched on at the Python level,
        because a branch resolved at capture time would be frozen for every replay.
        """
        context = self._context()
        decoder_input = torch.randn(16, 1, self.HIDDEN)

        result = context.apply_multimodal_embeddings(decoder_input.clone())

        torch.testing.assert_close(result, decoder_input)

    def test_apply_overwrites_only_masked_rows(self):
        """Staged rows replace the embedding output exactly at their own positions."""
        context = self._context()
        request = self._request([2, 5])
        context.scatter_multimodal_embeddings(request, chunk_start=0, chunk_length=8)

        decoder_input = torch.zeros(8, 1, self.HIDDEN)
        result = context.apply_multimodal_embeddings(decoder_input)

        assert result[2, 0, 0].item() == 1.0
        assert result[5, 0, 0].item() == 2.0
        untouched = [i for i in range(8) if i not in (2, 5)]
        assert result[untouched].abs().sum().item() == 0.0
