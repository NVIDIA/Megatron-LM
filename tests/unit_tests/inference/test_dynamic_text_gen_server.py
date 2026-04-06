# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the extracted endpoint helpers.

Each test function targets one of the nine helpers factored out of the
completions / chat_completions route handlers.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    apply_parsers,
    format_chat_response,
    parse_chat_messages,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    add_moe_routing_to_choice,
    check_failed_requests,
    parse_sampling_params,
    run_inference,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
    format_completions_response,
    parse_prompt,
)

# Patch targets (patch where the name is looked up, not where it is defined)
_COMP = (
    "megatron.core.inference.text_generation_server"
    ".dynamic_text_gen_server.endpoints.completions"
)
_CHAT = (
    "megatron.core.inference.text_generation_server"
    ".dynamic_text_gen_server.endpoints.chat_completions"
)

# Sentinel for parametrised exception cases
_RAISE = object()


# ------------------------------------------------------------------ #
# Lightweight mocks
# ------------------------------------------------------------------ #


class _Tok:
    """Minimal tokenizer mock (no chat template)."""

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(chr(i) for i in ids)


class _ChatTok(_Tok):
    """Tokenizer mock with chat-template support."""

    chat_template = "<mock>"
    eos_id = 2
    bos = None

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kw
    ):
        # Deterministic: 3 header tokens + one per message
        return [10, 20, 30] + [40 + i for i in range(len(messages))]


class _RetokTok(_ChatTok):
    """Tokenizer that returns different tokens on the retokenization call."""

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kw
    ):
        if add_generation_prompt:
            return [10, 20, 2, 50, 60]
        # Retokenization of just the previous turn
        return [10, 20, 2]


def _result(**overrides):
    """Build a minimal inference-result dict."""
    base = {
        "generated_text": "hello",
        "prompt_tokens": [65, 66, 67],
        "generated_tokens": [68, 69],
        "routing_indices": None,
        "prompt": "input ",
        "sampling_params": {"num_tokens_to_generate": 100},
        "policy_epoch": 0,
        "kv_cache_epoch": 0,
        "events": [],
        "generated_log_probs": [-0.1, -0.2],
        "log_probs": [-0.1, -0.2],
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestDynamicTextGenServer:
    """One test per extracted helper, heavily parametrised."""

    # ------ 1. parse_sampling_params -------------------------------- #

    @pytest.mark.parametrize(
        "req, completions_mode, expected_sp, expected_extras",
        [
            pytest.param(
                {},
                False,
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "return_log_probs": False,
                    "top_n_logprobs": 0,
                    "skip_prompt_log_probs": True,
                    "num_tokens_to_generate": None,
                    "add_BOS": False,
                    "stop_words": None,
                    "return_segments": False,
                    "num_tokens_total": None,
                    "termination_id": None,
                    "detokenize_stop_sequence": False,
                },
                {"n": 1},
                id="chat-all-defaults",
            ),
            pytest.param(
                {},
                True,
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "return_log_probs": False,
                    "num_tokens_to_generate": 16,
                    "skip_prompt_log_probs": True,
                },
                {"echo": False},
                id="completions-all-defaults",
            ),
            pytest.param(
                {"temperature": 0.0, "top_p": 0.9, "top_k": 50},
                False,
                {"temperature": 0.0, "top_p": 0.0, "top_k": 1},
                {"n": 1},
                id="greedy-forces-topk1-topp0",
            ),
            pytest.param(
                {"logprobs": 5, "echo": True},
                True,
                {"return_log_probs": True, "top_n_logprobs": 5, "skip_prompt_log_probs": False},
                {"echo": True},
                id="completions-logprobs-int-echo-true",
            ),
            pytest.param(
                {"logprobs": 3},
                True,
                {"return_log_probs": True, "top_n_logprobs": 3, "skip_prompt_log_probs": True},
                {"echo": False},
                id="completions-logprobs-int-echo-false",
            ),
            pytest.param(
                {"logprobs": True, "top_logprobs": 10, "skip_prompt_log_probs": False},
                False,
                {"return_log_probs": True, "top_n_logprobs": 10, "skip_prompt_log_probs": False},
                {"n": 1},
                id="chat-logprobs-bool-top-logprobs",
            ),
            pytest.param(
                {"logprobs": False, "top_logprobs": 5},
                False,
                {"return_log_probs": False, "top_n_logprobs": 0},
                {"n": 1},
                id="chat-logprobs-false-ignores-top",
            ),
            pytest.param(
                {"max_completion_tokens": 256, "max_tokens": 128},
                False,
                {"num_tokens_to_generate": 256},
                {"n": 1},
                id="chat-max-completion-tokens-priority",
            ),
            pytest.param({"n": 4}, False, {}, {"n": 4}, id="chat-n-parsed"),
            pytest.param(
                {"stop": "END", "max_tokens": 32},
                True,
                {"stop_words": ["END"], "num_tokens_to_generate": 32},
                {"echo": False},
                id="completions-stop-string-coerced",
            ),
            pytest.param(
                {"stop": ["A", "B"]},
                True,
                {"stop_words": ["A", "B"]},
                {"echo": False},
                id="completions-stop-list",
            ),
            pytest.param(
                {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "logprobs": True,
                    "top_logprobs": 5,
                    "skip_prompt_log_probs": True,
                    "max_tokens": 200,
                    "stop": ["END"],
                    "add_BOS": True,
                    "return_segments": True,
                    "detokenize_stop_sequence": True,
                    "num_tokens_total": 1024,
                    "termination_id": 42,
                },
                False,
                {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "return_log_probs": True,
                    "top_n_logprobs": 5,
                    "skip_prompt_log_probs": True,
                    "num_tokens_to_generate": 200,
                    "stop_words": ["END"],
                    "add_BOS": True,
                    "return_segments": True,
                    "detokenize_stop_sequence": True,
                    "num_tokens_total": 1024,
                    "termination_id": 42,
                },
                {"n": 1},
                id="every-field-non-default",
            ),
        ],
    )
    def test_parse_sampling_params(self, req, completions_mode, expected_sp, expected_extras):
        sp, extras = parse_sampling_params(req, completions_mode=completions_mode)
        for field, val in expected_sp.items():
            actual = getattr(sp, field)
            assert actual == val, f"{field}: expected {val!r}, got {actual!r}"
        assert extras == expected_extras

        # Structural completeness guard.  The "every-field-non-default" case
        # lists all 13 non-deprecated fields.  If someone adds a new field to
        # SamplingParams without updating parse_sampling_params, this fires.
        non_deprecated = set(SamplingParams.__dataclass_fields__) - {"return_prompt_top_n_logprobs"}
        if len(expected_sp) >= len(non_deprecated) - 1:
            missing = non_deprecated - set(expected_sp)
            assert not missing, (
                f"SamplingParams fields not covered by this test: {missing}. "
                f"Update parse_sampling_params and this test case."
            )
            # Every tested field must differ from the dataclass default,
            # proving it was actually piped through from the request.
            sp_default = SamplingParams()
            for field in non_deprecated:
                assert getattr(sp, field) != getattr(sp_default, field), (
                    f"SamplingParams.{field} still has its default value "
                    f"{getattr(sp_default, field)!r} — "
                    f"parse_sampling_params may not be setting it"
                )

    # ------ 2. run_inference ---------------------------------------- #

    @pytest.mark.parametrize(
        "values, verbose",
        [
            pytest.param([10, 20, 30], False, id="gather-multiple"),
            pytest.param([10, 20, 30], True, id="gather-with-timing"),
            pytest.param([], False, id="empty-tasks"),
            pytest.param(_RAISE, False, id="exception-propagates"),
        ],
    )
    def test_run_inference(self, values, verbose):
        async def _ok(v):
            return v

        async def _fail():
            raise ValueError("boom")

        async def _run():
            if values is _RAISE:
                with pytest.raises(ValueError, match="boom"):
                    await run_inference([_fail()], verbose)
            else:
                results = await run_inference([_ok(v) for v in values], verbose)
                assert results == values

        asyncio.run(_run())

    # ------ 3. check_failed_requests -------------------------------- #

    @pytest.mark.parametrize(
        "batch, expected",
        [
            pytest.param([{"status": "COMPLETED"}], None, id="all-success"),
            pytest.param([], None, id="empty-batch"),
            pytest.param(
                [
                    {
                        "status": "FAILED",
                        "events": [{"type": "ERROR_NONTRANSIENT", "payload": "bad"}],
                    }
                ],
                ("Request 0: bad", 400),
                id="nontransient-400",
            ),
            pytest.param(
                [
                    {
                        "status": "FAILED",
                        "events": [{"type": "ERROR_TRANSIENT", "payload": "timeout"}],
                    }
                ],
                ("Request 0: timeout", 500),
                id="transient-500",
            ),
            pytest.param(
                [
                    {"status": "FAILED", "events": [{"type": "ERROR_TRANSIENT", "payload": "t"}]},
                    {"status": "COMPLETED"},
                    {
                        "status": "FAILED",
                        "events": [{"type": "ERROR_NONTRANSIENT", "payload": "nt"}],
                    },
                ],
                ("Request 0: t; Request 2: nt", 400),
                id="mixed-nontransient-wins",
            ),
            pytest.param(
                [{"status": "FAILED", "events": []}],
                ("Request 0: Unknown error", 500),
                id="no-events-unknown",
            ),
            pytest.param(
                [{"status": "FAILED"}], ("Request 0: Unknown error", 500), id="missing-events-key"
            ),
        ],
    )
    def test_check_failed_requests(self, batch, expected):
        result = check_failed_requests(batch)
        assert result == expected

    # ------ 4. add_moe_routing_to_choice ---------------------------- #

    @pytest.mark.parametrize(
        "routing, prompt_tokens, expected_keys",
        [
            pytest.param(None, [1, 2, 3], set(), id="none-routing-noop"),
            pytest.param(
                [10, 20, 30, 40, 50],
                [1, 2, 3],
                {"moe_topk_indices", "prompt_moe_topk_indices"},
                id="routing-with-prompt",
            ),
            pytest.param([10, 20], None, {"moe_topk_indices"}, id="routing-no-prompt-tokens"),
            pytest.param([10, 20, 30], [], {"moe_topk_indices"}, id="routing-empty-prompt-tokens"),
        ],
    )
    def test_add_moe_routing_to_choice(self, routing, prompt_tokens, expected_keys):
        choice = {"index": 0}
        add_moe_routing_to_choice(
            choice, {"routing_indices": routing, "prompt_tokens": prompt_tokens}
        )
        added = set(choice.keys()) - {"index"}
        assert added == expected_keys
        if "moe_topk_indices" in expected_keys:
            assert choice["moe_topk_indices"] == routing
        if "prompt_moe_topk_indices" in expected_keys:
            assert choice["prompt_moe_topk_indices"] == routing[: len(prompt_tokens)]

    # ------ 5. parse_prompt ----------------------------------------- #

    @pytest.mark.parametrize(
        "prompt_data, expected_tokens, expected_strings",
        [
            pytest.param("AB", [[65, 66]], ["AB"], id="string-prompt"),
            pytest.param(["A", "B"], [[65], [66]], ["A", "B"], id="list-of-strings"),
            pytest.param([65, 66], [[65, 66]], ["AB"], id="list-of-ints"),
            pytest.param([[65, 66], [67]], [[65, 66], [67]], ["AB", "C"], id="list-of-int-lists"),
            pytest.param(None, ValueError, None, id="none-raises"),
            pytest.param("", ValueError, None, id="empty-string-raises"),
            pytest.param(42, ValueError, None, id="int-raises"),
            pytest.param([1.5], ValueError, None, id="bad-list-raises"),
        ],
    )
    def test_parse_prompt(self, prompt_data, expected_tokens, expected_strings):
        tok = _Tok()
        if expected_tokens is ValueError:
            with pytest.raises(ValueError):
                parse_prompt(prompt_data, tok)
        else:
            tokens, strings = parse_prompt(prompt_data, tok)
            assert tokens == expected_tokens
            assert strings == expected_strings

    # ------ 6. format_completions_response -------------------------- #

    @pytest.mark.parametrize(
        "echo, return_log_probs, extra_result_fields, check",
        [
            pytest.param(
                False,
                False,
                {},
                lambda r: (
                    r["choices"][0]["text"] == "hello" and r["choices"][0]["logprobs"] is None
                ),
                id="basic-no-echo-no-logprobs",
            ),
            pytest.param(
                True,
                False,
                {},
                lambda r: r["choices"][0]["text"] == "input hello",
                id="echo-prepends-prompt",
            ),
            pytest.param(
                False,
                True,
                {"generated_log_probs": [-0.5, -0.3]},
                lambda r: (
                    r["choices"][0]["logprobs"]["token_logprobs"] == [None, -0.5, -0.3]
                    and r["choices"][0]["logprobs"]["tokens"] == ["D", "E"]
                ),
                id="logprobs-no-echo",
            ),
            pytest.param(
                True,
                True,
                {"prompt_log_probs": [-1.0, -2.0], "generated_log_probs": [-0.5, -0.3]},
                lambda r: (
                    r["choices"][0]["logprobs"]["tokens"] == ["A", "B", "C", "D", "E"]
                    and r["choices"][0]["logprobs"]["token_logprobs"]
                    == [None, -1.0, -2.0, -0.5, -0.3]
                    and r["choices"][0]["logprobs"]["text_offset"] == [0, 1, 2, 3, 4]
                ),
                id="logprobs-with-echo",
            ),
            pytest.param(
                False,
                False,
                {"routing_indices": [1, 2, 3, 4, 5]},
                lambda r: (
                    r["choices"][0]["moe_topk_indices"] == [1, 2, 3, 4, 5]
                    and r["choices"][0]["prompt_moe_topk_indices"] == [1, 2, 3]
                ),
                id="moe-routing",
            ),
        ],
    )
    def test_format_completions_response(self, echo, return_log_probs, extra_result_fields, check):
        sp = SamplingParams(return_log_probs=return_log_probs)
        batch = [_result(**extra_result_fields)]
        prompts = ["input "]
        with patch(f"{_COMP}.unwrap_serialized_tensors", side_effect=lambda x: x):
            resp = format_completions_response(batch, prompts, sp, echo, _Tok())
        assert "choices" in resp
        assert len(resp["choices"]) == 1
        assert check(resp), f"Check failed on {json.dumps(resp, indent=2, default=str)}"

    # ------ 7. parse_chat_messages ---------------------------------- #

    @pytest.mark.parametrize(
        "req, tokenizer, expected",
        [
            pytest.param(
                {"messages": [{"role": "user", "content": "Hi"}]},
                _ChatTok(),
                [10, 20, 30, 40],
                id="basic-chat-template",
            ),
            pytest.param(
                {"messages": [{"role": "user", "content": "Hi"}], "prevent_retokenization": False},
                _ChatTok(),
                [10, 20, 30, 40],
                id="retokenization-disabled",
            ),
            pytest.param(
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {
                            "role": "assistant",
                            "content": "Hello",
                            "prompt_token_ids": [10, 21],
                            "generation_token_ids": [22, 2],
                        },
                    ]
                },
                _RetokTok(),
                [10, 21, 22, 2, 50, 60],
                id="retokenization-replaces-prefix",
            ),
            pytest.param(
                {"messages": [{"role": "user", "content": "Hi"}]},
                _Tok(),  # no chat_template
                [ord(c) for c in "Hi"],
                id="fallback-to-tokenize",
            ),
            pytest.param({"messages": None}, _ChatTok(), ValueError, id="missing-messages-raises"),
            pytest.param(
                {"messages": "not a list"}, _ChatTok(), ValueError, id="messages-not-list-raises"
            ),
        ],
    )
    def test_parse_chat_messages(self, req, tokenizer, expected):
        if expected is ValueError:
            with pytest.raises(ValueError):
                parse_chat_messages(req, tokenizer)
        else:
            assert parse_chat_messages(req, tokenizer) == expected

    # ------ 8. apply_parsers ---------------------------------------- #

    @pytest.mark.parametrize(
        "parsers_list, tools_requested, parser_output, expected_text, expected_meta_keys",
        [
            pytest.param(
                ["p1"], True, ("parsed", {}), "parsed", set(), id="single-parser-no-tool-calls"
            ),
            pytest.param(
                ["p1"],
                True,
                (
                    "parsed",
                    {"tool_calls": [{"function": {"name": "fn", "arguments": "{}"}, "id": "c1"}]},
                ),
                "parsed",
                {"tool_calls"},
                id="parser-with-tool-calls",
            ),
            pytest.param(
                ["p1"],
                False,
                (
                    "parsed",
                    {"tool_calls": [{"function": {"name": "fn", "arguments": "{}"}, "id": "c1"}]},
                ),
                "original text",
                set(),
                id="tool-calls-suppressed-when-not-requested",
            ),
            pytest.param(["missing"], True, None, None, None, id="unknown-parser-raises"),
        ],
    )
    def test_apply_parsers(
        self, parsers_list, tools_requested, parser_output, expected_text, expected_meta_keys
    ):
        mock_parser = MagicMock()
        if parser_output is not None:
            mock_parser.parse.return_value = parser_output
        mapping = {"p1": mock_parser}

        with patch(f"{_CHAT}.PARSER_MAPPING", mapping):
            if expected_text is None:
                with pytest.raises(ValueError, match="not found"):
                    apply_parsers("original text", None, parsers_list, tools_requested)
            else:
                text, meta = apply_parsers("original text", None, parsers_list, tools_requested)
                assert text == expected_text
                assert set(meta.keys()) == expected_meta_keys

    # ------ 9. format_chat_response --------------------------------- #

    @pytest.mark.parametrize(
        "results, return_log_probs, check",
        [
            pytest.param(
                [_result()],
                False,
                lambda r: (
                    r["object"] == "chat.completion"
                    and r["model"] == "EMPTY"
                    and len(r["choices"]) == 1
                    and r["choices"][0]["message"]["role"] == "assistant"
                    and r["choices"][0]["message"]["content"] == "hello"
                    and r["choices"][0]["finish_reason"] == "stop"
                    and r["choices"][0]["logprobs"] is None
                    and r["usage"]["prompt_tokens"] == 3
                    and r["usage"]["completion_tokens"] == 2
                    and r["usage"]["total_tokens"] == 5
                ),
                id="basic-structure-and-usage",
            ),
            pytest.param(
                [
                    _result(),
                    _result(
                        generated_text="world",
                        generated_tokens=[70, 71, 72],
                        generated_log_probs=[-0.1, -0.2, -0.3],
                        log_probs=[-0.1, -0.2, -0.3],
                        prompt_tokens=[65, 66, 67, 68],
                    ),
                ],
                False,
                lambda r: (
                    len(r["choices"]) == 2
                    and r["choices"][1]["message"]["content"] == "world"
                    and r["usage"]["prompt_tokens"] == 4
                    and r["usage"]["completion_tokens"] == 5
                ),
                id="multi-choice-usage-stats",
            ),
            pytest.param(
                [_result()],
                True,
                lambda r: (
                    r["choices"][0]["logprobs"]["content"] is not None
                    and len(r["choices"][0]["logprobs"]["content"]) == 2
                    and r["choices"][0]["logprobs"]["content"][0]["token"] == "D"
                    and r["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.1
                    and r["choices"][0]["logprobs"]["content"][0]["bytes"]
                    == list("D".encode("utf-8"))
                ),
                id="logprobs-content-format",
            ),
            pytest.param(
                [
                    _result(
                        generated_tokens=list(range(100)),
                        generated_log_probs=[0.0] * 100,
                        log_probs=[0.0] * 100,
                        sampling_params={"num_tokens_to_generate": 100},
                    )
                ],
                False,
                lambda r: r["choices"][0]["finish_reason"] == "length",
                id="finish-reason-length",
            ),
            pytest.param(
                [_result(routing_indices=[1, 2, 3, 4, 5])],
                False,
                lambda r: (
                    r["choices"][0]["moe_topk_indices"] == [1, 2, 3, 4, 5]
                    and r["choices"][0]["prompt_moe_topk_indices"] == [1, 2, 3]
                ),
                id="moe-routing",
            ),
        ],
    )
    def test_format_chat_response(self, results, return_log_probs, check):
        sp = SamplingParams(return_log_probs=return_log_probs)
        with patch(f"{_CHAT}.unwrap_serialized_tensors", side_effect=lambda x: x):
            resp = format_chat_response(
                results, sp, _Tok(), None, None, False, None, True, False
            )
        assert check(resp), f"Check failed on {json.dumps(resp, indent=2, default=str)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
