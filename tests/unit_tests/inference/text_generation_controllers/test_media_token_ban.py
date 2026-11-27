# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import types
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
    _ban_token_ids,
)

_VOCAB = 100
_IMAGE_ID = 18
_VIDEO_ID = 19
_SENTINEL_ID = 21


class _Tokenizer:
    """Knows `<image>` and `<video>`, like a tokenizer whose vocabulary has the media tokens."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<image>": _IMAGE_ID, "<video>": _VIDEO_ID}.get(token, self.unk_token_id)


def _controller(
    *, prompt_config="default", preexpanded_id=_SENTINEL_ID, tokenizer=None, banned_token_ids=None
):
    """A stand-in carrying just what `_media_token_ids` and `_banned_token_ids` read."""
    if prompt_config == "default":
        prompt_config = MultimodalPromptConfig(
            image_spec=MediaPromptSpec(model_token="<image>"),
            video_spec=MediaPromptSpec(model_token="<video>"),
        )
    wrapper = SimpleNamespace(
        multimodal_prompt_config=prompt_config,
        inference_context=SimpleNamespace(
            config=SimpleNamespace(banned_token_ids=banned_token_ids)
        ),
    )
    # The real resolver, bound to the stand-in, so token lookup is not re-implemented here.
    wrapper.resolve_media_token_id = types.MethodType(
        AbstractModelInferenceWrapper.resolve_media_token_id, wrapper
    )

    def get_preexpanded_media_token_id(modality):
        if preexpanded_id is None:
            raise NotImplementedError("no pre-expanded token")
        return preexpanded_id

    wrapper.get_preexpanded_media_token_id = get_preexpanded_media_token_id
    return SimpleNamespace(
        inference_wrapped_model=wrapper, tokenizer=tokenizer or _Tokenizer(), vocab_size=_VOCAB
    )


def _media_token_ids(controller):
    return TextGenerationController._media_token_ids(controller)


def _banned_token_ids(controller):
    # `_banned_token_ids` calls `self._media_token_ids()`, so bind it on the stand-in.
    controller._media_token_ids = types.MethodType(
        TextGenerationController._media_token_ids, controller
    )
    return TextGenerationController._banned_token_ids(controller)


def test_media_placeholders_are_banned():
    # The tokenizer ids of both modalities, and the model's own pre-expanded media token.
    assert _media_token_ids(_controller()) == [_IMAGE_ID, _VIDEO_ID, _SENTINEL_ID]


def test_a_wrapper_without_a_prompt_contract_bans_nothing():
    # A text-only wrapper has no media token to look up.
    assert _media_token_ids(_controller(prompt_config=None, preexpanded_id=None)) == []


def test_a_media_token_the_tokenizer_lacks_is_skipped_not_an_error():
    class OnlyImage(_Tokenizer):
        def convert_tokens_to_ids(self, token):
            return _IMAGE_ID if token == "<image>" else self.unk_token_id

        def encode(self, *args, **kwargs):  # the resolver's fallback finds nothing either
            return []

    controller = _controller(tokenizer=OnlyImage(), preexpanded_id=None)
    assert _media_token_ids(controller) == [_IMAGE_ID]


def test_the_model_without_a_pre_expanded_token_still_bans_the_tokenizer_ids():
    assert _media_token_ids(_controller(preexpanded_id=None)) == [_IMAGE_ID, _VIDEO_ID]


def test_an_id_outside_the_vocabulary_is_dropped():
    assert _media_token_ids(_controller(preexpanded_id=_VOCAB + 5)) == [_IMAGE_ID, _VIDEO_ID]


def test_the_media_tokens_are_banned_without_any_configuration():
    assert _banned_token_ids(_controller(banned_token_ids=None)) == [
        _IMAGE_ID,
        _VIDEO_ID,
        _SENTINEL_ID,
    ]


def test_configured_ids_are_banned_in_addition_to_the_media_tokens():
    controller = _controller(banned_token_ids=[7, _IMAGE_ID])
    assert _banned_token_ids(controller) == [7, _IMAGE_ID, _VIDEO_ID, _SENTINEL_ID]


def test_a_text_only_wrapper_bans_just_the_configured_ids():
    # No prompt contract, so no media token to look up: its media id has to be configured.
    controller = _controller(prompt_config=None, preexpanded_id=None, banned_token_ids=[_IMAGE_ID])
    assert _banned_token_ids(controller) == [_IMAGE_ID]
    assert _banned_token_ids(_controller(prompt_config=None, preexpanded_id=None)) == []


def test_a_configured_id_outside_the_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="outside the model's vocabulary"):
        _banned_token_ids(_controller(banned_token_ids=[_VOCAB]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ban_token_ids_makes_the_ids_unsampleable_and_leaves_the_rest(dtype):
    torch.manual_seed(0)
    logits = torch.randn(1, 6, _VOCAB, dtype=dtype)
    # Make a banned id the most likely token everywhere, so only the ban can stop it.
    logits[..., _IMAGE_ID] = 100.0
    original = logits.clone()
    banned = torch.tensor([_IMAGE_ID, _SENTINEL_ID])

    _ban_token_ids(logits, banned)

    assert torch.isinf(logits[..., banned]).all() and (logits[..., banned] < 0).all()
    kept = torch.ones(_VOCAB, dtype=torch.bool)
    kept[banned] = False
    assert torch.equal(logits[..., kept], original[..., kept])
    assert torch.softmax(logits.float(), dim=-1)[..., banned].eq(0).all()
    assert not torch.isin(logits.argmax(dim=-1), banned).any()


def test_ban_token_ids_on_a_slice_of_a_larger_buffer_leaves_the_rest_of_the_buffer():
    buffer = torch.zeros(1, 8, _VOCAB)

    _ban_token_ids(buffer[:, :3, :], torch.tensor([_IMAGE_ID]))

    assert torch.isinf(buffer[:, :3, _IMAGE_ID]).all()
    assert (buffer[:, 3:, :] == 0).all()


def test_ban_token_ids_without_ids_changes_nothing():
    logits = torch.randn(1, 3, _VOCAB)
    expected = logits.clone()

    _ban_token_ids(logits, None)

    assert torch.equal(logits, expected)
