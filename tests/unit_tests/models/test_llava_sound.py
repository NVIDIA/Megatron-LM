# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Unit tests for the audio (sound) integration in ``LLaVAModel``.

We avoid constructing a full ``LLaVAModel`` (which would build real GPT and
RADIO models) and instead invoke the relevant unbound methods on a
``SimpleNamespace`` stub. This focuses the test surface on the new sound
code paths without dragging in the full multimodal stack.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.multimodal.llava_model import (
    DEFAULT_SOUND_TOKEN_INDEX,
    SOUND_TOKEN,
    LLaVAModel,
)

# ---------------------------------------------------------------------------
# Sentinel constants
# ---------------------------------------------------------------------------


class TestSoundSentinelConstants:
    """The PR exposes ``SOUND_TOKEN`` / ``DEFAULT_SOUND_TOKEN_INDEX`` as the
    public contract. Pin them so accidental changes break tests rather than
    consumers (data pipelines, tokenizer config, etc.)."""

    @pytest.mark.internal
    def test_sound_token_string(self):
        assert SOUND_TOKEN == "<so_embedding>"

    @pytest.mark.internal
    def test_default_sound_token_index_is_negative(self):
        # Negative index puts it in the "not a real token" range alongside
        # DEFAULT_IMAGE_TOKEN_INDEX (-200).
        assert DEFAULT_SOUND_TOKEN_INDEX == -300


# ---------------------------------------------------------------------------
# freeze() with sound modules
# ---------------------------------------------------------------------------


class _ModuleWithParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class TestFreezeSoundModules:
    """``LLaVAModel.freeze`` must respect the new ``freeze_sound_*`` flags
    and gracefully no-op when the corresponding module is ``None``."""

    def _stub(
        self,
        *,
        language_model=None,
        vision_model=None,
        vision_projection=None,
        sound_model=None,
        sound_projection=None,
    ):
        return SimpleNamespace(
            language_model=language_model,
            vision_model=vision_model,
            vision_projection=vision_projection,
            sound_model=sound_model,
            sound_projection=sound_projection,
        )

    @pytest.mark.internal
    def test_freezes_only_requested_modules(self):
        sound_model = _ModuleWithParam()
        sound_projection = _ModuleWithParam()
        language_model = _ModuleWithParam()
        stub = self._stub(
            language_model=language_model,
            sound_model=sound_model,
            sound_projection=sound_projection,
        )

        LLaVAModel.freeze(
            stub,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            freeze_sound_model=True,
            freeze_sound_projection=True,
        )

        assert all(not p.requires_grad for p in sound_model.parameters())
        assert all(not p.requires_grad for p in sound_projection.parameters())
        # language_model was NOT requested to freeze.
        assert all(p.requires_grad for p in language_model.parameters())

    @pytest.mark.internal
    def test_no_op_when_sound_modules_are_none(self):
        """Freezing a non-existent sound model must not raise."""
        stub = self._stub()  # everything None
        LLaVAModel.freeze(
            stub,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            freeze_sound_model=True,
            freeze_sound_projection=True,
        )

    @pytest.mark.internal
    def test_default_kwargs_do_not_touch_sound_modules(self):
        """Existing callers that don't pass freeze_sound_* must be unaffected."""
        sound_model = _ModuleWithParam()
        stub = self._stub(sound_model=sound_model)
        LLaVAModel.freeze(
            stub,
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )
        assert all(p.requires_grad for p in sound_model.parameters())


# ---------------------------------------------------------------------------
# has_sounds sentinel detection (B2 regression)
# ---------------------------------------------------------------------------


class TestHasSoundsSentinel:
    """The forward path treats a ``[1, 1]`` zero tensor as "no sound this batch".

    These tests reproduce the sentinel decision logic to lock the contract in
    place. If this test breaks because the decision changed, also update the
    data pipeline that emits the sentinel.
    """

    @staticmethod
    def _has_sounds(sound_clips):
        """Mirror of the logic at the top of ``LLaVAModel.forward`` post-fix."""
        has = sound_clips is not None and sound_clips.numel() > 0
        if has and sound_clips.shape == torch.Size([1, 1]):
            has = sound_clips[0, 0].item() != 0
        return has

    @pytest.mark.internal
    def test_none_is_not_sounds(self):
        assert self._has_sounds(None) is False

    @pytest.mark.internal
    def test_single_zero_is_not_sounds(self):
        assert self._has_sounds(torch.zeros((1, 1))) is False

    @pytest.mark.internal
    def test_single_nonzero_is_sounds(self):
        # A 1x1 *non-zero* clip is treated as real audio (degenerate but legal).
        assert self._has_sounds(torch.ones((1, 1))) is True

    @pytest.mark.internal
    def test_real_audio_is_sounds(self):
        assert self._has_sounds(torch.randn(2, 16000)) is True

    @pytest.mark.internal
    def test_empty_tensor_is_not_sounds(self):
        assert self._has_sounds(torch.empty(0)) is False


# ---------------------------------------------------------------------------
# Sound replacement in _preprocess_data (light integration)
# ---------------------------------------------------------------------------


def _build_preprocess_stub(
    *,
    language_max_sequence_length: int = 64,
    sound_token_index: int = DEFAULT_SOUND_TOKEN_INDEX,
    sound_pad_to_clip_duration: bool = False,
    sound_model_present: bool = True,
):
    """Build the minimum ``self`` that ``_preprocess_data`` requires to run."""
    if sound_model_present:
        sound_model = SimpleNamespace(
            config=SimpleNamespace(sound_pad_to_clip_duration=sound_pad_to_clip_duration)
        )
    else:
        sound_model = None
    return SimpleNamespace(
        pre_process=True,
        post_process=True,
        add_decoder=True,
        img_seq_len=576,
        sound_token_index=sound_token_index,
        sound_model=sound_model,
        vision_model=None,
        _language_max_sequence_length=language_max_sequence_length,
        _language_is_pipeline_parallel=False,
        context_parallel_lm=1,
        sequence_parallel_lm=False,
    )


def _make_inputs(*, batch=1, text_seq=8, embed_dim=4, sound_position=4, image_token_index=-200):
    """Build a single text-only batch with one sound token at ``sound_position``."""
    input_ids = torch.arange(text_seq, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    # Mark the sound token position.
    input_ids[:, sound_position] = DEFAULT_SOUND_TOKEN_INDEX
    labels = torch.zeros((batch, text_seq), dtype=torch.long)
    loss_mask = torch.ones((batch, text_seq), dtype=torch.float32)
    language_embeddings = torch.arange(batch * text_seq * embed_dim, dtype=torch.float32).reshape(
        batch, text_seq, embed_dim
    )
    # No images.
    image_embeddings = torch.empty((0, 0, embed_dim), dtype=torch.float32)
    num_image_tiles = torch.zeros((0,), dtype=torch.int32)
    return SimpleNamespace(
        image_embeddings=image_embeddings,
        language_embeddings=language_embeddings,
        input_ids=input_ids,
        loss_mask=loss_mask,
        labels=labels,
        num_image_tiles=num_image_tiles,
        image_token_index=image_token_index,
    )


class TestPreprocessDataSoundReplacement:
    """Light integration tests for the sound replacement block of
    ``_preprocess_data`` (lines ~654–679 of ``llava_model.py``)."""

    @pytest.mark.internal
    def test_sound_pad_to_clip_duration_false_packs_per_length(self):
        """Without ``sound_pad_to_clip_duration``, embeddings are concatenated
        per ``sound_embeddings_len`` so padding tokens are dropped."""
        stub = _build_preprocess_stub(sound_pad_to_clip_duration=False)
        inputs = _make_inputs(batch=1, text_seq=8, embed_dim=4, sound_position=4)

        # One sound clip per sample: capacity 5 slots but only 1 real embedding
        # (4 padding slots dropped). One sound token in input_ids maps to the
        # 1 real embedding; the remaining 4 capacity slots are ignored.
        sound_embeddings = torch.full((5, 1, 4), 7.0)  # [s, b, h], capacity 5
        sound_embeddings_len = torch.tensor([1])  # only the first 1 is real

        final_embedding, _final_labels, _final_loss_mask = LLaVAModel._preprocess_data(
            stub,
            image_embeddings=inputs.image_embeddings,
            language_embeddings=inputs.language_embeddings,
            input_ids=inputs.input_ids,
            loss_mask=inputs.loss_mask,
            labels=inputs.labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=inputs.image_token_index,
            num_image_tiles=inputs.num_image_tiles,
            sound_embeddings=sound_embeddings,
            sound_embeddings_len=sound_embeddings_len,
            sound_timestamps=None,
        )

        # final_embedding is returned as [s, b, h] (transposed) when CP=1.
        assert final_embedding.shape[2] == 4
        # The sound token at position 4 should have been replaced by 7.0 across embed_dim.
        # In [seq, batch, hidden] layout, position 4 of batch 0:
        sound_slot = final_embedding[4, 0]
        assert torch.allclose(
            sound_slot, torch.full((4,), 7.0)
        ), f"sound slot was {sound_slot.tolist()}, expected all 7.0"

    @pytest.mark.internal
    def test_sound_pad_to_clip_duration_true_uses_full_buffer(self):
        """With ``sound_pad_to_clip_duration``, the full padded buffer is used
        (no length-based truncation), so the wrapper can pre-shape clips to a
        fixed duration."""
        stub = _build_preprocess_stub(sound_pad_to_clip_duration=True)
        # Use 3 sound tokens so the flat buffer matches: 3 tokens * embed_dim = 12 values.
        # sound_embeddings shape [s=3, b=1, h=4] → flatten to 3 positions of length 4.
        inputs = _make_inputs(batch=1, text_seq=8, embed_dim=4, sound_position=4)
        # Place additional sound tokens at positions 5 and 6.
        inputs.input_ids[:, 5] = DEFAULT_SOUND_TOKEN_INDEX
        inputs.input_ids[:, 6] = DEFAULT_SOUND_TOKEN_INDEX
        sound_embeddings = torch.arange(3 * 1 * 4, dtype=torch.float32).reshape(3, 1, 4)
        sound_embeddings_len = torch.tensor([3])

        final_embedding, _, _ = LLaVAModel._preprocess_data(
            stub,
            image_embeddings=inputs.image_embeddings,
            language_embeddings=inputs.language_embeddings,
            input_ids=inputs.input_ids,
            loss_mask=inputs.loss_mask,
            labels=inputs.labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=inputs.image_token_index,
            num_image_tiles=inputs.num_image_tiles,
            sound_embeddings=sound_embeddings,
            sound_embeddings_len=sound_embeddings_len,
            sound_timestamps=None,
        )

        # All three sound positions should now hold the corresponding rows
        # from sound_embeddings (after the [s, b, h] → flat reshape).
        # sound_embeddings.permute(1, 0, 2) → [1, 3, 4]; reshape(-1, 4) → [3, 4].
        expected = sound_embeddings.permute(1, 0, 2).reshape(-1, 4)
        for i, pos in enumerate([4, 5, 6]):
            assert torch.allclose(final_embedding[pos, 0], expected[i]), (
                f"sound slot at pos {pos} was {final_embedding[pos, 0].tolist()}, "
                f"expected {expected[i].tolist()}"
            )

    @pytest.mark.internal
    def test_no_sound_token_in_input_is_no_op(self):
        """If ``input_ids`` contains no ``sound_token_index`` the sound block
        must not raise, even with non-empty ``sound_embeddings``."""
        stub = _build_preprocess_stub()
        inputs = _make_inputs(batch=1, text_seq=8, embed_dim=4, sound_position=4)
        # Replace the sound token with a regular text token so sound_mask is empty.
        inputs.input_ids[:, 4] = 99
        sound_embeddings = torch.full((1, 1, 4), 7.0)
        sound_embeddings_len = torch.tensor([1])

        # Should run without error and not modify the embedding at the would-be slot.
        final_embedding, _, _ = LLaVAModel._preprocess_data(
            stub,
            image_embeddings=inputs.image_embeddings,
            language_embeddings=inputs.language_embeddings,
            input_ids=inputs.input_ids,
            loss_mask=inputs.loss_mask,
            labels=inputs.labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=inputs.image_token_index,
            num_image_tiles=inputs.num_image_tiles,
            sound_embeddings=sound_embeddings,
            sound_embeddings_len=sound_embeddings_len,
            sound_timestamps=None,
        )
        # Position 4 was a regular text token (id=99) so it should hold the
        # corresponding language embedding row — not 7.0.
        assert not torch.allclose(final_embedding[4, 0], torch.full((4,), 7.0))

    @pytest.mark.internal
    def test_sound_pad_branch_safe_when_sound_model_missing_attr(self):
        """B1 regression: ``getattr`` chain must not crash when sound_model has no config attr."""
        # sound_model present but config has no ``sound_pad_to_clip_duration`` attribute.
        stub = SimpleNamespace(
            pre_process=True,
            post_process=True,
            add_decoder=True,
            img_seq_len=576,
            sound_token_index=DEFAULT_SOUND_TOKEN_INDEX,
            sound_model=SimpleNamespace(config=SimpleNamespace()),  # no attr
            vision_model=None,
            _language_max_sequence_length=64,
            _language_is_pipeline_parallel=False,
            context_parallel_lm=1,
            sequence_parallel_lm=False,
        )
        inputs = _make_inputs(batch=1, text_seq=8, embed_dim=4, sound_position=4)
        sound_embeddings = torch.full((1, 1, 4), 7.0)
        sound_embeddings_len = torch.tensor([1])

        # Should fall through to the False branch (per-length packing) without
        # raising AttributeError.
        LLaVAModel._preprocess_data(
            stub,
            image_embeddings=inputs.image_embeddings,
            language_embeddings=inputs.language_embeddings,
            input_ids=inputs.input_ids,
            loss_mask=inputs.loss_mask,
            labels=inputs.labels,
            use_inference_kv_cache=False,
            inference_context=None,
            image_token_index=inputs.image_token_index,
            num_image_tiles=inputs.num_image_tiles,
            sound_embeddings=sound_embeddings,
            sound_embeddings_len=sound_embeddings_len,
            sound_timestamps=None,
        )
