# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import base64
from unittest import mock

import pytest

from megatron.core.inference.config import MediaPromptSpec, MultimodalPromptConfig
from megatron.core.inference.inference_request import (
    compute_media_cache_key,
    serialize_multimodal_data,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _extract_media_url_bytes,
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


@pytest.mark.asyncio
async def test_n_choices_prepare_and_serialize_shared_media_once():
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        chat_completions,
    )

    class _Tokenizer:
        chat_template = "test-template"
        unk_token_id = 0
        eod = None

        def apply_chat_template(self, messages, **_kwargs):
            return "".join(message["content"] for message in messages)

        def convert_tokens_to_ids(self, token):
            return 99 if token == "<image>" else self.unk_token_id

        def __call__(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return []

        def detokenize(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class _Client:
        def __init__(self):
            self.serialized_media = []

        def add_request_with_id(self, prompt_tokens, sampling_params, *, multi_modal_data=None):
            wire = serialize_multimodal_data(multi_modal_data)
            self.serialized_media.append(wire)
            request_id = len(self.serialized_media)
            future = asyncio.get_running_loop().create_future()
            future.set_result(
                {
                    "uid": f"choice-{request_id}",
                    "status": "COMPLETED",
                    "generated_tokens": [request_id],
                    "prompt_length": len(prompt_tokens),
                    "num_cached_tokens": 0,
                    "sampling_params": sampling_params.serialize(),
                    "routing_indices": None,
                }
            )
            return request_id, future

        def abort_request(self, _request_id):
            raise AssertionError("Successful choices must not be aborted")

    tokenizer = _Tokenizer()
    client = _Client()
    spec = MediaPromptSpec(model_token="<image>")
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=tokenizer,
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(image_spec=spec, video_spec=spec),
        default_temperature=1.0,
        default_top_p=1.0,
        default_top_k=0,
        eval_mode=False,
    )
    app.register_blueprint(chat_completions.bp)
    image = b"shared-image"
    image_url = f"data:image/png;base64,{base64.b64encode(image).decode()}"

    with mock.patch(
        "megatron.core.inference.inference_request.compute_media_cache_key",
        wraps=compute_media_cache_key,
    ) as compute_key:
        response = await app.test_client().post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    }
                ],
                "n": 3,
                "max_tokens": 1,
            },
        )

    assert response.status_code == 200
    assert len((await response.get_json())["choices"]) == 3
    assert len(client.serialized_media) == 3
    assert all(wire == client.serialized_media[0] for wire in client.serialized_media)
    assert all(wire is not client.serialized_media[0] for wire in client.serialized_media[1:])
    assert compute_key.call_count == 1
