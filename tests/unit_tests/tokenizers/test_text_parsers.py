# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parity tests for the `<think>`/`</think>` reasoning parsers.

Ground truth for `NemotronV3ReasoningParser` is derived from vLLM's actual
implementation:

- Base extraction: `BaseThinkingReasoningParser.extract_reasoning` in
  `vllm/reasoning/basic_parsers.py` (used unmodified by `DeepSeekR1ReasoningParser`
  for non-streaming extraction). Notably `final_content = content or None`, so an
  empty string after a closing `</think>` collapses to `None`, same as a missing
  closing tag entirely.
- Override: `SuperV3ReasoningParser`/`UltraV3ReasoningParser.extract_reasoning` in
  `super_v3_reasoning_parser.py`/`ultra_v3_reasoning_parser.py` (from
  huggingface.co/nvidia/NVIDIA-Nemotron-3-{Super,Ultra}-*), which swaps all text
  into content when `final_content is None` and either `enable_thinking is False`
  or `force_nonempty_content is True`.

"""

import json
import time

import pytest

from megatron.core.tokenizers.text.parsers import PARSER_MAPPING
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.nemotron_v3_reasoning_parser import (
    NemotronV3ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import (
    Qwen3CoderToolParser,
    _Qwen3CoderToolParser,
)

# (text, kwargs, expected_content, expected_info)
# `kwargs` is expanded into `parse(text, **kwargs)`; the override flags reach the
# parser inside `chat_template_kwargs`, exactly as the chat-completions endpoint
# forwards them from the request.
NEMOTRON_V3_CASES = [
    # No chat_template_kwargs override: behaves exactly like DeepSeekR1ReasoningParser.
    ("<think>hello", {}, "", {"reasoning": "hello"}),
    ("<think>hello</think>world", {}, "world", {"reasoning": "hello"}),
    # Closing tag present but nothing follows it: vLLM's `content or None` treats
    # this the same as a missing closing tag, so it is empty here too.
    ("<think>hello</think>", {}, "", {"reasoning": "hello"}),
    # No `<think>` tag at all: vLLM assumes the whole string is reasoning.
    ("just an answer", {}, "", {"reasoning": "just an answer"}),
    # enable_thinking=False surfaces would-be-empty content as the reasoning text,
    # for both the "unterminated" and "closes with nothing following" cases.
    ("<think>hello", {"chat_template_kwargs": {"enable_thinking": False}}, "hello", {}),
    ("<think>hello</think>", {"chat_template_kwargs": {"enable_thinking": False}}, "hello", {}),
    # force_nonempty_content=True has the same effect as enable_thinking=False.
    (
        "<think>hello</think>",
        {"chat_template_kwargs": {"force_nonempty_content": True}},
        "hello",
        {},
    ),
    ("<think>hello", {"chat_template_kwargs": {"force_nonempty_content": True}}, "hello", {}),
    # The override only fires when there would otherwise be no content.
    (
        "<think>hello</think>world",
        {"chat_template_kwargs": {"enable_thinking": False}},
        "world",
        {"reasoning": "hello"},
    ),
    # Text preceding `<think>` is discarded, override still applies past it.
    (
        "prefix<think>hello</think>",
        {"chat_template_kwargs": {"enable_thinking": False}},
        "hello",
        {},
    ),
    # enable_thinking=True (or omitted) must not trigger the override.
    (
        "<think>hello</think>",
        {"chat_template_kwargs": {"enable_thinking": True}},
        "",
        {"reasoning": "hello"},
    ),
]


@pytest.mark.parametrize("text,kwargs,expected_content,expected_info", NEMOTRON_V3_CASES)
def test_nemotron_v3_reasoning_parser_matches_vllm(text, kwargs, expected_content, expected_info):
    content, info = NemotronV3ReasoningParser.parse(text, **kwargs)
    assert content == expected_content
    assert info == expected_info


@pytest.mark.parametrize(
    "text", ["<think>hello", "<think>hello</think>world", "<think>hello</think>", "just an answer"]
)
def test_nemotron_v3_reasoning_parser_without_override_matches_deepseek_r1(text):
    """With no `enable_thinking`/`force_nonempty_content` kwargs, the Nemotron 3
    parser must be observably identical to the DeepSeek R1 parser it extends."""
    assert NemotronV3ReasoningParser.parse(text) == DeepSeekR1ReasoningParser.parse(text)


def test_parser_mapping_registers_nemotron_v3_reasoning():
    """Super and Ultra share identical reasoning-extraction logic upstream, so
    both models are served by a single consolidated parser and registry key."""
    assert PARSER_MAPPING["nemotron-v3-reasoning"] is NemotronV3ReasoningParser


def test_tool_call_marker_implicitly_ends_reasoning_for_downstream_parser():
    tool_text = (
        "<tool_call><function=bash><parameter=command>echo hi</parameter>" "</function></tool_call>"
    )
    model_output = f"I should inspect this first.\n{tool_text}"
    tool_parser = PARSER_MAPPING["qwen3-coder-tool"]

    content, reasoning_info = DeepSeekR1ReasoningParser.parse(
        model_output, implicit_reasoning_end_markers=tool_parser.implicit_reasoning_end_markers
    )
    parsed_content, tool_info = tool_parser.parse(
        content,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
    )

    assert reasoning_info == {"reasoning": "I should inspect this first.\n"}
    assert parsed_content is None
    assert tool_info["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command": "echo hi"}',
    }


def test_tool_call_marker_does_not_end_reasoning_unless_configured():
    model_output = "reasoning<tool_call>not enabled</tool_call>"

    assert DeepSeekR1ReasoningParser.parse(model_output) == ("", {"reasoning": model_output})


# ---------------------------------------------------------------------------
# Qwen3-Coder tool parser: argument coercion
# ---------------------------------------------------------------------------
# Expected values here are what vLLM 0.25.1 actually returned when the same
# inputs were run through it (`tool_parser: qwen3_coder`, which resolves to
# Qwen3EngineToolParser -> Qwen3ParserToolAdapter -> the ParserEngine in
# vllm/parser/qwen3.py), not values derived from reading the source.
#
# The engine strips each value in `_qwen3_arg_converter` and then applies
# `coerce_to_schema_type` from vllm/tool_parsers/utils.py, which tries the
# property's declared types in the order
# null > integer > number > boolean > object > array > string and falls back to
# a plain JSON parse, then to the raw string.

COERCION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "t",
            "parameters": {
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "o": {"type": "object"},
                    "maybe": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                    "choice": {"enum": ["a", "b", None]},
                }
            },
        },
    }
]


def coerce(param, value):
    """Run one parameter through the tool parser and return its argument value."""
    text = (
        f"<tool_call>\n<function=t>\n<parameter={param}>\n{value}\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    info = _Qwen3CoderToolParser().extract_tool_calls(text, tools=COERCION_TOOLS)
    return json.loads(info["tool_calls"][0]["function"]["arguments"])[param]


@pytest.mark.parametrize(
    "param,value,expected",
    [
        # A "null" literal only becomes JSON null where the property admits
        # null. For a string property the model meant the four characters.
        ("s", "null", "null"),
        ("s", "NULL", "NULL"),
        ("maybe", "null", None),
        ("choice", "null", None),
        # An unconvertible boolean falls through to the raw string rather than
        # silently degenerating to False.
        ("b", "yes", "yes"),
        ("b", "true", True),
        ("b", "1", True),
        ("b", "0", False),
        # int() fails on "42.7", so the JSON fallback supplies the number.
        ("i", "42.7", 42.7),
        ("i", "7", 7),
        ("i", "abc", "abc"),
        # Values are stripped before coercion.
        ("i", "  7  ", 7),
        ("s", "   spaced   ", "spaced"),
        # A whole float collapses to an int.
        ("n", "5.0", 5),
        ("n", "2.5", 2.5),
        ("o", '{"a": 1}', {"a": 1}),
        ("maybe", '{"k": "v"}', {"k": "v"}),
    ],
)
def test_qwen3_coder_coercion_matches_vllm(param, value, expected):
    assert coerce(param, value) == expected


def test_qwen3_coder_arguments_are_always_valid_json():
    """inf/nan cannot be serialized as JSON, so such values stay strings.

    vllm/tool_parsers/utils.py guards this with _is_json_finite for the same
    reason: json.dumps(inf) emits `Infinity`, which no JSON reader accepts.
    """
    for value in ("NaN", "Infinity", "-Infinity", "1e400"):
        for param in ("n", "i", "o"):
            text = (
                f"<tool_call>\n<function=t>\n<parameter={param}>\n{value}\n"
                "</parameter>\n</function>\n</tool_call>"
            )
            info = _Qwen3CoderToolParser().extract_tool_calls(text, tools=COERCION_TOOLS)
            raw = info["tool_calls"][0]["function"]["arguments"]

            def _bare(constant):
                raise AssertionError(f"emitted bare {constant} in {raw}")

            json.loads(raw, parse_constant=_bare)


# ---------------------------------------------------------------------------
# Qwen3-Coder tool parser: grammar parity
# ---------------------------------------------------------------------------
# Coercion parity (above) left the surrounding grammar untouched. These cases
# cover the shapes where the two grammars visibly disagreed; every expectation
# is what vLLM 0.25.1 actually returned for the same input, measured by running
# both stacks over a shared corpus in one process.
#
# The reference implementation is vllm/parser/qwen3.py:
#
#     _PARAM_RE = r"<\s*parameter\s*=\s*([^>]*)>(.*?)"
#                 r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))"
#     params[name] = value.strip()      # the NAME is deliberately not stripped

GRAMMAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        },
    }
]


def call_args(body, tools=GRAMMAR_TOOLS):
    """Parse one tool call and return its arguments dict (None if no call)."""
    info = _Qwen3CoderToolParser().extract_tool_calls(body, tools=tools)
    calls = info.get("tool_calls") or []
    if not calls:
        return None
    return json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize(
    "body,expected",
    [
        # Whitespace is permitted anywhere inside the parameter tag. `\s*` after
        # `=` consumes leading whitespace in the name; `[^>]*` keeps trailing
        # whitespace, which vLLM also keeps.
        ("<function=f>\n< parameter = a >x</parameter>\n</function>", {"a ": "x"}),
        ("<function=f>\n<parameter =a>x</parameter>\n</function>", {"a": "x"}),
        ("<function=f>\n<parameter= a>x</parameter>\n</function>", {"a": "x"}),
        ("<function=f>\n<parameter=a >x</parameter>\n</function>", {"a ": "x"}),
        ("<function=f>\n<parameter= a >x</parameter>\n</function>", {"a ": "x"}),
        # ...and inside the closing tag.
        ("<function=f>\n<parameter=a>x</ parameter >\n</function>", {"a": "x"}),
        ("<function=f>\n<parameter=a>x</parameter >\n</function>", {"a": "x"}),
        # A parameter is only recognised when closed by </parameter> or followed
        # by another <parameter=>. An unclosed one is dropped, not salvaged.
        ("<function=f>\n<parameter=a>x\n</function>", {}),
        ("<function=f>\n<parameter=a>x\n<parameter=b>7\n</function>", {"a": "x"}),
    ],
)
def test_qwen3_coder_parameter_grammar_matches_vllm(body, expected):
    assert call_args(f"<tool_call>\n{body}\n</tool_call>") == expected


def test_qwen3_coder_tool_call_end_inside_value_does_not_terminate_the_call():
    """A </tool_call> written inside a parameter value belongs to the value.

    vLLM's engine tracks that it is inside a parameter; matching that here means
    consuming complete <parameter>...</parameter> blocks atomically.
    """
    body = (
        "<tool_call>\n<function=f>\n"
        "<parameter=a>see </tool_call> here</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert call_args(body) == {"a": "see </tool_call> here"}


@pytest.mark.parametrize(
    "text,expected_name",
    [
        # Generation stopped before the function tag closed. vLLM still emits a
        # call named after the remaining text; dropping it instead left
        # "<tool_call>" in the post-parse content, which NeMo-Gym scores as
        # is_invalid_tool_call and converts into a -5.0 advantage.
        ("<tool_call>\n<function=f", "f"),
        ("text mentioning <function= but never closing", "but never closing"),
    ],
)
def test_qwen3_coder_truncated_function_tag_still_yields_a_call(text, expected_name):
    info = _Qwen3CoderToolParser().extract_tool_calls(text, tools=GRAMMAR_TOOLS)
    calls = info.get("tool_calls") or []
    assert [c["function"]["name"] for c in calls] == [expected_name]
    assert "<tool_call>" not in (info.get("content") or "")


def test_qwen3_coder_truncated_tool_call_without_function_yields_nothing():
    """`<tool_call>` with no `<function=` is not a call in either engine."""
    info = _Qwen3CoderToolParser().extract_tool_calls("<tool_call>\n", tools=GRAMMAR_TOOLS)
    assert not (info.get("tool_calls") or [])


def _unterminated_tool_call(n_parameters: int) -> str:
    """`<tool_call><function=f>` plus N closed parameter blocks, deliberately
    missing the closing `</tool_call>` -- the shape that made the old
    `tool_call_regex` backtrack exponentially: with no `</tool_call>` to match,
    it tried every way to partition the input between its parameter-block and
    plain-character alternatives before giving up.
    """
    body = "<tool_call><function=f>"
    for i in range(n_parameters):
        body += f"<parameter=a{i}>x</parameter>"
    return body


def test_qwen3_coder_unterminated_call_with_many_parameters_is_not_exponential():
    """Regression test for exponential backtracking on an unterminated call.

    The reviewer's repro (`<tool_call><function=f>` + 20 closed parameter
    blocks, no closing `</tool_call>`) exceeded 3s against the old regex.
    Streaming hits this shape on every chunk before the closing tag arrives,
    so this has to stay fast, not just eventually finish. A generous absolute
    bound is used rather than asserting a precise linear ratio, since CI
    machines are shared and timing noise is real -- but 20 parameters
    completing in over a second would still mean the fix regressed back
    toward exponential, and this catches that.
    """
    text = _unterminated_tool_call(20)
    start = time.perf_counter()
    _Qwen3CoderToolParser().extract_tool_calls(text, tools=GRAMMAR_TOOLS)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"took {elapsed:.3f}s on the reviewer's exact repro shape"

    # 16x the reviewer's parameter count should not cost anywhere near 16x the
    # time if the fix is actually linear (let alone the "orders of magnitude
    # worse" a regression to exponential behavior would show).
    larger_text = _unterminated_tool_call(320)
    start = time.perf_counter()
    _Qwen3CoderToolParser().extract_tool_calls(larger_text, tools=GRAMMAR_TOOLS)
    larger_elapsed = time.perf_counter() - start
    assert larger_elapsed < 2.0, f"took {larger_elapsed:.3f}s at 16x the parameter count"


@pytest.mark.parametrize(
    ("finished", "expect_call"),
    [
        # Generation is genuinely done: the truncated name is all there ever
        # will be, so vLLM parity requires emitting it (this PR's original fix).
        (True, True),
        # Mid-stream: `<function=get` is not "the model stopped after `get`",
        # it is "`get_weather` has not fully arrived in this chunk yet".
        # StreamingChatParser emits a tool call's name delta exactly once, on
        # the first non-None name it sees, so firing here would permanently
        # lock in a truncated name once more of the text does arrive.
        (False, False),
    ],
)
def test_qwen3_coder_truncated_function_name_fallback_is_gated_on_finished(
    finished, expect_call
):
    text = "<tool_call>\n<function=get"
    info = Qwen3CoderToolParser.parse(text, tools=GRAMMAR_TOOLS, finished=finished)
    _, metadata = info
    calls = metadata.get("tool_calls") or []
    if expect_call:
        assert [c["function"]["name"] for c in calls] == ["get"]
    else:
        assert calls == []


def test_qwen3_coder_parse_defaults_to_finished():
    """Every non-streaming caller (chat_completions.py's finished-response path,
    completions.py, and any direct one-shot use) calls `.parse()` without a
    `finished` kwarg. The truncated-function-name fallback must still fire for
    them by default -- only the streaming path explicitly opts into
    `finished=False` per chunk.
    """
    text = "<tool_call>\n<function=get"
    _, metadata = Qwen3CoderToolParser.parse(text, tools=GRAMMAR_TOOLS)
    calls = metadata.get("tool_calls") or []
    assert [c["function"]["name"] for c in calls] == ["get"]
