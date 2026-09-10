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

import pytest

from megatron.core.tokenizers.text.parsers import PARSER_MAPPING
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.nemotron_v3_reasoning_parser import (
    NemotronV3ReasoningParser,
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


# --- Qwen3-Coder tool parser: parity with vLLM 0.25.1 ---------------------------------
#
# Expected values below were produced by vLLM 0.25.1's `Qwen3EngineToolParser` (the class
# `qwen3_coder` resolves to in `vllm/tool_parsers/__init__.py`) called through a real
# `ChatCompletionRequest` carrying `_PARITY_TOOLS`, i.e. exactly what vLLM's OpenAI server
# does. They cover the three behaviours this parser takes from vLLM: `str.strip()` on
# parameter values, stripped content around tool calls, and schema-driven type coercion
# (`vllm/tool_parsers/utils.py::coerce_to_schema_type`).

_PARITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "o": {"type": "object"},
                    "a": {"type": "array", "items": {"type": "integer"}},
                    "sn": {"type": ["string", "null"]},
                    "e": {"enum": [1, 2, 3]},
                    "any": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "code": {"type": "string"},
                    "foo": {"type": "foo"},
                    "lst": {"type": "list"},
                    "ntyped": {},
                },
            },
        },
    }
]


def _qwen3_call(params, fn="f"):
    body = "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={fn}>\n{body}</function>\n</tool_call>"


# (case id, model output, use tool schema, expected content, expected [(name, arguments)])
QWEN3_CODER_PARITY_CASES = [
    (
        'strip_indent',
        '<tool_call>\n<function=f>\n<parameter=code>\n    def f():\n        pass\n</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'code': 'def f():\n        pass'}]],
    ),
    (
        'strip_spaces',
        '<tool_call>\n<function=f>\n<parameter=s>   hello world   </parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'s': 'hello world'}]],
    ),
    (
        'int_ok',
        '<tool_call>\n<function=f>\n<parameter=i>\n42\n</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'i': 42}]],
    ),
    (
        'int_bad',
        '<tool_call>\n<function=f>\n<parameter=i>forty</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'i': 'forty'}]],
    ),
    (
        'int_from_float',
        '<tool_call>\n<function=f>\n<parameter=i>3.0</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'i': 3.0}]],
    ),
    (
        'num_integral',
        '<tool_call>\n<function=f>\n<parameter=n>3.0</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'n': 3}]],
    ),
    (
        'num_frac',
        '<tool_call>\n<function=f>\n<parameter=n>3.5</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'n': 3.5}]],
    ),
    (
        'num_inf',
        '<tool_call>\n<function=f>\n<parameter=n>inf</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'n': 'inf'}]],
    ),
    (
        'num_huge',
        '<tool_call>\n<function=f>\n<parameter=n>1e999</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'n': '1e999'}]],
    ),
    (
        'bool_true',
        '<tool_call>\n<function=f>\n<parameter=b>True</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'b': True}]],
    ),
    (
        'bool_one',
        '<tool_call>\n<function=f>\n<parameter=b>1</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'b': True}]],
    ),
    (
        'bool_zero',
        '<tool_call>\n<function=f>\n<parameter=b> 0 </parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'b': False}]],
    ),
    (
        'bool_yes',
        '<tool_call>\n<function=f>\n<parameter=b>yes</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'b': 'yes'}]],
    ),
    (
        'bool_42',
        '<tool_call>\n<function=f>\n<parameter=b>42</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'b': 42}]],
    ),
    (
        'null_string_type',
        '<tool_call>\n<function=f>\n<parameter=s>null</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'s': 'null'}]],
    ),
    (
        'null_nullable',
        '<tool_call>\n<function=f>\n<parameter=sn>null</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'sn': None}]],
    ),
    (
        'sn_value',
        '<tool_call>\n<function=f>\n<parameter=sn>x</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'sn': 'x'}]],
    ),
    (
        'sn_int',
        '<tool_call>\n<function=f>\n<parameter=sn>5</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'sn': '5'}]],
    ),
    (
        'obj_json',
        '<tool_call>\n<function=f>\n<parameter=o>{"k": [1, 2]}</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'o': {'k': [1, 2]}}]],
    ),
    (
        'obj_pyliteral',
        "<tool_call>\n<function=f>\n<parameter=o>{'k': 1}</parameter>\n</function>\n</tool_call>",
        True,
        None,
        [['f', {'o': "{'k': 1}"}]],
    ),
    (
        'obj_bad',
        '<tool_call>\n<function=f>\n<parameter=o>not json</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'o': 'not json'}]],
    ),
    (
        'arr_json',
        '<tool_call>\n<function=f>\n<parameter=a>[1, 2, 3]</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'a': [1, 2, 3]}]],
    ),
    (
        'arr_strings_items_int',
        '<tool_call>\n<function=f>\n<parameter=a>["1","2"]</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'a': ['1', '2']}]],
    ),
    (
        'enum_int',
        '<tool_call>\n<function=f>\n<parameter=e>2</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'e': 2}]],
    ),
    (
        'enum_bad',
        '<tool_call>\n<function=f>\n<parameter=e>x</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'e': 'x'}]],
    ),
    (
        'anyof_int',
        '<tool_call>\n<function=f>\n<parameter=any>7</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'any': 7}]],
    ),
    (
        'anyof_str',
        '<tool_call>\n<function=f>\n<parameter=any>seven</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'any': 'seven'}]],
    ),
    (
        'unknown_type_foo',
        '<tool_call>\n<function=f>\n<parameter=foo>3</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'foo': 3}]],
    ),
    (
        'list_alias',
        '<tool_call>\n<function=f>\n<parameter=lst>[1,2]</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'lst': [1, 2]}]],
    ),
    (
        'untyped_prop',
        '<tool_call>\n<function=f>\n<parameter=ntyped>5</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'ntyped': '5'}]],
    ),
    (
        'unknown_param',
        '<tool_call>\n<function=f>\n<parameter=zzz>  v  </parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'zzz': 'v'}]],
    ),
    (
        'unknown_param_num',
        '<tool_call>\n<function=f>\n<parameter=zzz>5</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'zzz': '5'}]],
    ),
    (
        'unknown_function',
        '<tool_call>\n<function=g>\n<parameter=q>5</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['g', {'q': '5'}]],
    ),
    (
        'content_before',
        'Let me check.\n\n<tool_call>\n<function=f>\n<parameter=s>x</parameter>\n</function>\n</tool_call>',
        True,
        'Let me check.',
        [['f', {'s': 'x'}]],
    ),
    (
        'content_ws_only',
        '\n\n<tool_call>\n<function=f>\n<parameter=s>x</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'s': 'x'}]],
    ),
    (
        'content_after',
        '<tool_call>\n<function=f>\n<parameter=s>x</parameter>\n</function>\n</tool_call>\nDone.\n',
        True,
        None,
        [['f', {'s': 'x'}]],
    ),
    (
        'two_calls',
        '<tool_call>\n<function=f>\n<parameter=s>a</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=f>\n<parameter=i>1</parameter>\n</function>\n</tool_call>',
        True,
        None,
        [['f', {'s': 'a'}], ['f', {'i': 1}]],
    ),
    (
        'no_close_tool_call',
        '<tool_call>\n<function=f>\n<parameter=s>abc</parameter>\n</function>',
        True,
        None,
        [['f', {'s': 'abc'}]],
    ),
    (
        'no_tools_schema',
        '<tool_call>\n<function=f>\n<parameter=i>42</parameter>\n<parameter=b>true</parameter>\n</function>\n</tool_call>',
        False,
        None,
        [['f', {'i': '42', 'b': 'true'}]],
    ),
]


@pytest.mark.parametrize(
    "case_id,model_output,use_tools,expected_content,expected_calls",
    QWEN3_CODER_PARITY_CASES,
    ids=[c[0] for c in QWEN3_CODER_PARITY_CASES],
)
def test_qwen3_coder_tool_parser_matches_vllm(
    case_id, model_output, use_tools, expected_content, expected_calls
):
    parser = PARSER_MAPPING["qwen3-coder-tool"]
    content, info = parser.parse(model_output, tools=_PARITY_TOOLS if use_tools else None)

    assert content == expected_content
    calls = [
        [call["function"]["name"], json.loads(call["function"]["arguments"])]
        for call in info.get("tool_calls", [])
    ]
    assert calls == expected_calls


def test_qwen3_coder_tool_parser_keeps_unclosed_last_parameter():
    """Deliberate difference from vLLM: an unterminated final parameter is kept
    (vLLM's non-streaming converter drops it). The lenient regex is what lets the
    streaming path parse partially received tool calls, and its value is stripped
    like every other value."""
    text = "<tool_call>\n<function=f>\n<parameter=s>\nabc\n</function>\n</tool_call>"
    content, info = PARSER_MAPPING["qwen3-coder-tool"].parse(text, tools=_PARITY_TOOLS)
    assert content is None
    assert json.loads(info["tool_calls"][0]["function"]["arguments"]) == {"s": "abc"}


def test_qwen3_coder_tool_parser_without_tool_calls_is_passthrough():
    text = "plain answer with <function= mention but no call"
    assert PARSER_MAPPING["qwen3-coder-tool"].parse(text, tools=_PARITY_TOOLS) == (text, {})
