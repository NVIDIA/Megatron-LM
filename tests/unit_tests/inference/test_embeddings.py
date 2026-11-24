# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for /v1/embeddings and for the controller's pooling of prompt hidden states.

The handler runs under Quart's test client against the fake inference client from
test_endpoints_common.py. The pooling runs on a stand-in controller and context that hold only
the fields it reads, against sums computed here in float64.
"""

import base64
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

pytestmark = [pytest.mark.internal]

EMBEDDINGS_PATH = "/v1/embeddings"


# --- endpoint ----------------------------------------------------------------


def _app(client):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        embeddings,
    )
    from tests.unit_tests.inference.test_endpoints_common import Tokenizer

    app = quart.Quart(__name__)
    app.config.update(client=client, tokenizer=Tokenizer(), verbose=False)
    app.register_blueprint(embeddings.bp)
    return app


def _client(*embeddings):
    """A fake client that answers each submission with its own tokens and the next embedding."""
    from tests.unit_tests.inference.test_endpoints_common import ReplyingClient, completed_reply

    class EmbeddingClient(ReplyingClient):
        def add_request_with_id(self, prompt_tokens, sampling_params, **kwargs):
            self.replies = [
                completed_reply(
                    f"req-{len(self.prompt_tokens)}",
                    prompt_tokens,
                    [12],
                    prompt_embedding=embeddings[len(self.prompt_tokens)],
                )
            ]
            return super().add_request_with_id(prompt_tokens, sampling_params, **kwargs)

    return EmbeddingClient()


@pytest.mark.asyncio
async def test_embeddings_are_returned_per_input_with_the_token_ids_that_were_embedded():
    client = _client([0.5, -1.25, 3.0], [2.0, 0.0, -0.75])
    response = (
        await _app(client)
        .test_client()
        .post(EMBEDDINGS_PATH, json={"input": [[1, 5, 7], [1, 9]], "skip_tokens": 1})
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    payload = await response.get_json()
    assert payload["object"] == "list"
    assert payload["id"] == "req-0"
    assert payload["usage"] == {"prompt_tokens": 5, "total_tokens": 5}
    assert payload["data"] == [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.5, -1.25, 3.0],
            "prompt_token_ids": [1, 5, 7],
        },
        {
            "object": "embedding",
            "index": 1,
            "embedding": [2.0, 0.0, -0.75],
            "prompt_token_ids": [1, 9],
        },
    ]
    # Each input is one request: prefill, one discarded token, the mean after `skip_tokens`.
    assert client.prompt_tokens == [[1, 5, 7], [1, 9]]
    for params in client.sampling_params:
        assert params.return_prompt_embedding is True
        assert params.prompt_embedding_skip_tokens == 1
        assert params.num_tokens_to_generate == 1
        assert params.return_prompt_tokens is True
        assert params.return_log_probs is False


@pytest.mark.asyncio
async def test_base64_embeddings_are_little_endian_float32():
    values = [0.1, -2.5, 1e-3]
    response = (
        await _app(_client(values))
        .test_client()
        .post(EMBEDDINGS_PATH, json={"input": [1, 5, 7], "encoding_format": "base64"})
    )

    assert response.status_code == 200, await response.get_data(as_text=True)
    encoded = (await response.get_json())["data"][0]["embedding"]
    assert base64.b64decode(encoded) == struct.pack("<3f", *values)


@pytest.mark.asyncio
async def test_text_inputs_are_tokenized_and_skip_tokens_defaults_to_zero():
    client = _client([1.0], [2.0])
    response = await _app(client).test_client().post(EMBEDDINGS_PATH, json={"input": ["a", "b"]})

    assert response.status_code == 200, await response.get_data(as_text=True)
    assert client.prompt_tokens == [[10, 11], [10, 11]]  # The test tokenizer's fixed tokens.
    assert [params.prompt_embedding_skip_tokens for params in client.sampling_params] == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, message",
    [
        ({}, "Missing 'input' field"),
        ({"input": [[1, 2], "text"]}, "Invalid 'input' format"),
        ({"input": [1, 2], "skip_tokens": -1}, "skip_tokens must be a nonnegative integer"),
        ({"input": [1, 2], "skip_tokens": True}, "skip_tokens must be a nonnegative integer"),
        ({"input": [1, 2], "skip_tokens": 1.5}, "skip_tokens must be a nonnegative integer"),
        ({"input": [1, 2], "encoding_format": "hex"}, "encoding_format must be"),
        ({"input": [[1, 2, 3], [1, 2]], "skip_tokens": 2}, "inputs [1] have no token after"),
    ],
)
async def test_malformed_embeddings_requests_are_rejected_before_submission(body, message):
    client = _client()
    response = await _app(client).test_client().post(EMBEDDINGS_PATH, json=body)

    assert response.status_code == 400
    assert message in await response.get_data(as_text=True)
    assert client.prompt_tokens == []


@pytest.mark.asyncio
async def test_a_reply_without_an_embedding_is_an_error_naming_the_input():
    response = (
        await _app(_client([1.0], None))
        .test_client()
        .post(EMBEDDINGS_PATH, json={"input": [[1, 2], [1, 3]]})
    )

    assert response.status_code == 500
    assert "input 1 has no embedding" in await response.get_data(as_text=True)


@pytest.mark.asyncio
async def test_a_request_the_engine_fails_is_reported_as_the_completions_endpoint_does():
    from tests.unit_tests.inference.test_endpoints_common import ReplyingClient, failed_reply

    client = ReplyingClient(
        [
            failed_reply(
                {
                    "type": "ERROR_NONTRANSIENT",
                    "payload": "ValueError: Prompt embeddings require capture_prompt_hidden_states",
                }
            )
        ]
    )
    response = await _app(client).test_client().post(EMBEDDINGS_PATH, json={"input": [1, 2]})

    assert response.status_code == 400
    assert "capture_prompt_hidden_states" in await response.get_data(as_text=True)


# --- pooling -----------------------------------------------------------------

HIDDEN = 5


def _controller(hidden_states=None, requests=(), paused=0):
    """A stand-in with the fields the pooling reads.

    `requests` lists the active requests in order as `(request_id, rows, prefill, skip, start)`:
    the request has `rows` rows in the forward, the first at position `start` of the request,
    and asks for an embedding without its first `skip` positions (-1: no embedding).
    """
    ids, rows, prefill, skips, starts = ([r[i] for r in requests] for i in range(5))
    slots = torch.arange(paused, paused + len(ids))
    positions = [torch.arange(start, start + n) for start, n in zip(starts, rows)]
    context = SimpleNamespace(
        prompt_hidden_states=hidden_states,
        active_token_count=sum(rows),
        token_to_request_idx=slots.repeat_interleave(torch.tensor(rows, dtype=torch.long)).int(),
        token_to_position_in_request=torch.cat(positions or [torch.empty(0)]).int(),
        request_metadata={
            "return_prompt_embedding": torch.tensor([False] * paused + [s >= 0 for s in skips]),
            "prompt_embedding_skip_tokens": torch.tensor(
                [0] * paused + [max(s, 0) for s in skips], dtype=torch.int32
            ),
        },
        request_in_prefill_status_tensor=torch.tensor([0] * paused + prefill),
        request_ids=torch.tensor([-1] * paused + ids),
    )
    return SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(inference_context=context), _prompt_hidden_sums={}
    )


def _forward(controller, hidden_states, requests, paused=0):
    """Point the stand-in's context at one forward's layout and pool it."""
    fresh = _controller(torch.from_numpy(hidden_states), requests, paused)
    controller.inference_wrapped_model = fresh.inference_wrapped_model
    TextGenerationController._accumulate_prompt_hidden_states(controller)


def _pop(controller, request_id, num_positions):
    return TextGenerationController.pop_prompt_embedding(controller, request_id, num_positions)


def _assert_mean(got, rows):
    """`got` against the float64 mean of float32 `rows`, element by element.

    Both are float64 sums of the same n float32 values, possibly in different orders. Each of a
    sum's n - 1 additions is off by at most 2**-53 of a partial sum, itself at most n * max|row|,
    so a sum is off by (n - 1) * n * max * 2**-53 and its mean by (n - 1) * max * 2**-53; the
    division rounds once more. Twice that, for the two sides: 2 * n * max * 2**-53.
    """
    reference = rows.astype(np.float64).mean(axis=0)
    bound = 2 * len(rows) * float(np.abs(rows).max()) * 2.0**-53
    difference = np.abs(np.asarray(got) - reference)
    worst = int(difference.argmax())
    assert difference[worst] <= bound, (
        f"element {worst}: got {got[worst]!r}, reference {reference[worst]!r}, "
        f"difference {difference[worst]:.3e} > bound {bound:.3e}"
    )


def test_each_prefill_request_gets_the_mean_of_its_own_rows_after_skip():
    rows = np.random.default_rng(0).standard_normal((16, HIDDEN)).astype(np.float32) * 30
    controller = _controller()
    # A decode request with its base and two speculative rows, prefill requests of 4, 2 and 5
    # rows, then two padding rows (sequence parallelism pads the forward).
    _forward(
        controller, rows, [(7, 3, 0, 1, 9), (3, 4, 1, 1, 0), (9, 2, 1, -1, 0), (4, 5, 1, 0, 0)]
    )

    assert set(controller._prompt_hidden_sums) == {3, 4}  # 7 decodes, 9 did not ask.
    _assert_mean(_pop(controller, 3, 3), rows[4:7])
    _assert_mean(_pop(controller, 4, 5), rows[9:14])
    assert controller._prompt_hidden_sums == {}


def test_a_prompt_prefilled_in_chunks_is_summed_across_forwards():
    generator = np.random.default_rng(1)
    first = generator.standard_normal((6, HIDDEN)).astype(np.float32)
    second = generator.standard_normal((5, HIDDEN)).astype(np.float32)
    controller = _controller()
    # Request 5 with skip=4: positions 0..2 in the first forward, 3..6 in the second.
    _forward(controller, first, [(2, 3, 1, 0, 0), (5, 3, 1, 4, 0)])
    _forward(controller, second, [(8, 1, 0, -1, 9), (5, 4, 1, 4, 3)])

    _assert_mean(_pop(controller, 5, 3), second[2:5])
    _assert_mean(_pop(controller, 2, 3), first[0:3])


def test_paused_requests_do_not_shift_the_rows():
    rows = np.arange(6 * HIDDEN, dtype=np.float32).reshape(6, HIDDEN)
    controller = _controller()
    _forward(controller, rows, [(3, 2, 1, 0, 0), (4, 4, 1, 1, 0)], paused=2)

    _assert_mean(_pop(controller, 3, 2), rows[0:2])
    _assert_mean(_pop(controller, 4, 3), rows[3:6])


@pytest.mark.parametrize(
    "skip, num_positions",
    [(0, 5), (0, 3), (4, 0)],
    ids=["rows missing", "rows seen twice", "nothing left after skip"],
)
def test_no_embedding_unless_each_position_was_summed_exactly_once(skip, num_positions):
    controller = _controller()
    _forward(controller, np.ones((4, HIDDEN), dtype=np.float32), [(1, 4, 1, skip, 0)])

    assert _pop(controller, 1, num_positions) is None
    assert controller._prompt_hidden_sums == {}  # The running sum is dropped either way.


def test_nothing_is_pooled_without_the_buffer_or_a_request_that_asks():
    controller = _controller(None, [(1, 4, 1, 0, 0)])
    TextGenerationController._accumulate_prompt_hidden_states(controller)
    _forward(controller, np.ones((4, HIDDEN), dtype=np.float32), [(1, 4, 1, -1, 0)])

    assert controller._prompt_hidden_sums == {}
    assert _pop(controller, 1, 4) is None


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_the_model_copies_the_gathered_hidden_states_into_the_buffer(
    monkeypatch, sequence_parallel
):
    from megatron.core.models.common.language_module import language_module

    full = torch.arange(6.0 * HIDDEN).reshape(6, 1, HIDDEN)
    monkeypatch.setattr(
        language_module, "gather_from_sequence_parallel_region", lambda shard, group: full
    )
    model = SimpleNamespace(
        output_layer=SimpleNamespace(sequence_parallel=sequence_parallel),
        pg_collection=SimpleNamespace(tp=None),
    )
    context = SimpleNamespace(prompt_hidden_states=torch.zeros(8, HIDDEN))
    capture = language_module.LanguageModule._capture_prompt_hidden_states
    # Under sequence parallelism the model holds this rank's shard only.
    capture(model, full[:3] if sequence_parallel else full, context)
    capture(model, full, None)  # No context, no buffer: nothing to do.

    assert torch.equal(context.prompt_hidden_states[:6], full.squeeze(1))
    assert not context.prompt_hidden_states[6:].any()
