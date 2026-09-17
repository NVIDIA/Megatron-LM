# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import base64

import pytest

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _extract_media_url_bytes,
    _has_previous_turn_tokens,
    _last_assistant_message,
    _replace_prefix_tokens_metadata,
    _tokenize_with_media_slots_sync,
)


def test_extract_media_data_url_accepts_payload_at_limit():
    payload = b"four"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    assert _extract_media_url_bytes(url, max_bytes=len(payload)) == payload


def test_extract_media_data_url_rejects_decoded_payload_over_limit():
    # Four- and five-byte payloads both occupy eight base64 characters, so
    # this exercises the decoded-size check in addition to the encoded bound.
    payload = b"five!"
    url = f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"

    with pytest.raises(ValueError, match="data:video/mp4;base64 payload exceeds 4 byte limit"):
        _extract_media_url_bytes(url, max_bytes=4)


def test_replace_prefix_tokens_metadata_ships_the_rendered_prefix_and_eos():
    eos = 99
    template_prefix = (1, 99, 2, 99)
    offload_params = {"ng_capture": {"staging_chain": ["k1"]}}

    out = _replace_prefix_tokens_metadata(eos, template_prefix, offload_params)

    assert out[PREFIX_TEMPLATE_TOKEN_IDS_FIELD] == [1, 99, 2, 99]
    assert out[PREFIX_EOS_TOKEN_ID_FIELD] == 99
    assert out["ng_capture"] == {"staging_chain": ["k1"]}
    assert offload_params == {"ng_capture": {"staging_chain": ["k1"]}}  # input not mutated


_USER = {"role": "user", "content": "hi"}
_ASSISTANT_TEXT = {"role": "assistant", "content": "hello"}
_ASSISTANT_WITH_TOKENS = {
    "role": "assistant",
    "content": "hello",
    "prompt_token_ids": [1, 2],
    "compact_prompt_token_ids": [1, 2],
    "generation_token_ids": [3, 99],
}
_ENGINE_METADATA = {"ng_capture": {"staging_chain": ["k1"]}}


def test_has_previous_turn_tokens():
    assert _has_previous_turn_tokens(None) is False
    assert _has_previous_turn_tokens(_ASSISTANT_TEXT) is False  # dataset-provided history
    assert _has_previous_turn_tokens(_ASSISTANT_WITH_TOKENS) is True


def test_last_assistant_message_returns_the_last_assistant_turn():
    assert _last_assistant_message([_USER]) == (None, None)
    assert _last_assistant_message([_USER, _ASSISTANT_TEXT, _USER]) == (1, _ASSISTANT_TEXT)
    messages = [_USER, _ASSISTANT_WITH_TOKENS, _USER, _ASSISTANT_TEXT, _USER]
    assert _last_assistant_message(messages) == (3, _ASSISTANT_TEXT)


def test_media_slot_uses_tokenizer_id_when_model_id_is_unspecified():
    class _Tokenizer:
        unk_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return "__MEDIA__"

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

    spec = MediaPromptSpec(model_token="<image>")
    prompt_config = MultimodalPromptConfig(image_spec=spec, video_spec=spec)

    tokens = _tokenize_with_media_slots_sync(
        _Tokenizer(),
        messages=[],
        media_slots=[("__MEDIA__", "image", 0)],
        prompt_config=prompt_config,
        tools=None,
        chat_template_kwargs={},
    )

    assert tokens == [99]


def test_media_tokenization_is_synchronous_so_it_can_be_offloaded_whole():
    """Lowering media slots must not be a coroutine.

    Rendering is only the first of 3N+1 tokenizer calls for N slots. While this
    was async, awaiting the render put the remaining encodes back on the event
    loop, where they stall every other request the replica owns. Being a plain
    function is what lets the endpoint hand the whole thing to the tokenize
    executor in one hop, on the thread that owns the private tokenizer copy.
    """
    import inspect

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    assert not inspect.iscoroutinefunction(chat_completions._tokenize_with_media_slots_sync)
    # And the endpoint must not have kept a direct call that skips the executor.
    src = inspect.getsource(chat_completions.chat_completions)
    assert "_tokenize_with_media_slots_sync" in src
    for line in src.splitlines():
        if "_tokenize_with_media_slots_sync" in line:
            assert "await" not in line, f"must be dispatched via the executor, got: {line.strip()}"
