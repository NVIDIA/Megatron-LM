# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :mod:`megatron.training.datasets.varlen_dataset`.

These tests cover the schema-detection and message-normalization helpers and
the :class:`VarlenLowLevelDataset` loader. The end-to-end SFTDataset packing
behavior is exercised by the existing SFT test suite; here we focus on the
varlen-specific contracts (auto-detect schema, normalize to messages,
ValueError on unsupported shapes).
"""

import json
from pathlib import Path

import pytest

# Import via the public module path so this test gets discovered through the
# regular pytest entry point. The functions under test are pure Python and do
# not require torch.distributed.
from megatron.training.datasets.varlen_dataset import (
    VarlenLowLevelDataset,
    _alpaca_to_messages,
    _looks_like_hf_id,
    _messages_passthrough,
    _select_converter,
    _sharegpt_to_messages,
)

# ----------------------------------------------------------------------------
# _looks_like_hf_id heuristic
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("Yukang/LongAlpaca-12k", True),
        ("HuggingFaceH4/no_robots", True),
        ("databricks/databricks-dolly-15k", True),
        ("/tmp/foo.jsonl", False),
        ("./local.jsonl", False),
        ("../up.jsonl", False),
        ("singlename", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_hf_id(path, expected):
    assert _looks_like_hf_id(path) is expected


# ----------------------------------------------------------------------------
# Schema converters
# ----------------------------------------------------------------------------


def test_alpaca_canonical_with_input():
    out = _alpaca_to_messages(
        {"instruction": "Summarize.", "input": "Long passage", "output": "It says X."}
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[1]["content"] == "Summarize.\n\nLong passage"
    assert out[2]["content"] == "It says X."


def test_alpaca_without_input():
    out = _alpaca_to_messages({"instruction": "Hi.", "output": "Hello."})
    assert out[0] == {"role": "system", "content": ""}
    assert out[1]["content"] == "Hi."
    assert out[2]["content"] == "Hello."


@pytest.mark.parametrize(
    "instr_key,out_key",
    [
        ("prompt", "response"),
        ("query", "answer"),
        ("question", "completion"),
        ("instruction", "answer"),
    ],
)
def test_alpaca_field_synonyms(instr_key, out_key):
    out = _alpaca_to_messages({instr_key: "Q?", out_key: "A."})
    assert out[1]["content"] == "Q?"
    assert out[2]["content"] == "A."


def test_dolly_instruction_context_response():
    """Dolly-15k: instruction + context + response, all via synonyms."""
    out = _alpaca_to_messages(
        {
            "instruction": "Who wrote 1984?",
            "context": "1984 was written in 1948.",
            "response": "George Orwell.",
        }
    )
    assert out[1]["content"] == "Who wrote 1984?\n\n1984 was written in 1948."
    assert out[2]["content"] == "George Orwell."


def test_sharegpt_human_gpt():
    out = _sharegpt_to_messages(
        {
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ]
        }
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[1]["content"] == "hi"


def test_sharegpt_preserves_existing_system_turn():
    out = _sharegpt_to_messages(
        {
            "conversations": [
                {"from": "system", "value": "be terse"},
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ]
        }
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[0]["content"] == "be terse"


@pytest.mark.parametrize(
    "speaker,expected_role",
    [
        ("human", "user"),
        ("user", "user"),
        ("gpt", "assistant"),
        ("assistant", "assistant"),
        ("model", "assistant"),
        ("chatgpt", "assistant"),
        ("tool", "tool"),
        ("function", "tool"),
        ("alien", "user"),  # unknown speakers fall back to user
    ],
)
def test_sharegpt_role_map(speaker, expected_role):
    out = _sharegpt_to_messages({"conversations": [{"from": speaker, "value": "x"}]})
    # First entry is the prepended system turn; second is the actual content.
    assert out[1]["role"] == expected_role


def test_messages_passthrough_prepends_system_when_missing():
    out = _messages_passthrough(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]


def test_messages_passthrough_keeps_existing_system():
    out = _messages_passthrough(
        {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        }
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[0]["content"] == "be terse"


def test_messages_passthrough_strips_extra_keys():
    """OpenAI-style messages may carry ``name`` / ``tool_calls`` etc.;
    chat-template input only wants ``role`` and ``content``."""
    out = _messages_passthrough(
        {
            "messages": [
                {"role": "user", "content": "hi", "name": "alice"},
                {
                    "role": "assistant",
                    "content": "hi alice",
                    "tool_calls": [{"function": "foo"}],
                },
            ]
        }
    )
    for m in out:
        assert set(m.keys()) == {"role", "content"}


# ----------------------------------------------------------------------------
# Shape validation: reject multi-modal / non-string content
# ----------------------------------------------------------------------------


def test_messages_rejects_list_content():
    with pytest.raises(ValueError, match="must be a string"):
        _messages_passthrough(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "url": "x.png"}]},
                ]
            }
        )


def test_alpaca_rejects_non_string_field():
    with pytest.raises(ValueError, match="must be a string"):
        _alpaca_to_messages({"instruction": ["a", "b"], "output": "x"})


def test_sharegpt_rejects_list_value():
    with pytest.raises(ValueError, match="must be a string"):
        _sharegpt_to_messages(
            {"conversations": [{"from": "human", "value": [1, 2, 3]}]}
        )


# ----------------------------------------------------------------------------
# Schema selector priority
# ----------------------------------------------------------------------------


def test_select_converter_alpaca():
    fn, name = _select_converter(["instruction", "output", "file"])
    assert name == "alpaca"
    assert fn is _alpaca_to_messages


def test_select_converter_alpaca_via_synonyms():
    fn, name = _select_converter(["prompt", "response"])
    assert name == "alpaca"


def test_select_converter_dolly_columns():
    fn, name = _select_converter(["instruction", "context", "response", "category"])
    assert name == "alpaca"


def test_select_converter_sharegpt():
    fn, name = _select_converter(["conversations", "id"])
    assert name == "sharegpt"
    assert fn is _sharegpt_to_messages


def test_select_converter_messages():
    fn, name = _select_converter(["messages"])
    assert name == "openai-messages"
    assert fn is _messages_passthrough


def test_select_converter_priority_messages_over_alpaca():
    # When both ``messages`` and alpaca-style columns are present, the more
    # explicit ``messages`` schema wins.
    fn, name = _select_converter(["messages", "instruction", "output"])
    assert name == "openai-messages"


def test_select_converter_unrecognized_columns():
    with pytest.raises(ValueError, match="cannot infer schema"):
        _select_converter(["foo", "bar"])


def test_select_converter_alpaca_missing_output():
    """Having an instruction column but no output column is not a match."""
    with pytest.raises(ValueError, match="cannot infer schema"):
        _select_converter(["instruction", "category"])


# ----------------------------------------------------------------------------
# VarlenLowLevelDataset on local jsonl (no HF Hub network needed)
# ----------------------------------------------------------------------------


def _write_jsonl(tmp_path: Path, rows):
    p = tmp_path / "data.jsonl"
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(p)


def test_low_level_loads_jsonl_alpaca(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(
        tmp_path,
        [
            {"instruction": "i1", "output": "o1"},
            {"instruction": "i2", "output": "o2", "file": "extra"},
        ],
    )
    ll = VarlenLowLevelDataset(path)
    assert len(ll) == 2
    assert ll.schema_name == "alpaca"
    sample = ll[0]
    assert [m["role"] for m in sample] == ["system", "user", "assistant"]
    assert sample[1]["content"] == "i1"
    assert sample[2]["content"] == "o1"


def test_low_level_loads_jsonl_sharegpt(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ]
            },
        ],
    )
    ll = VarlenLowLevelDataset(path)
    assert len(ll) == 2
    assert ll.schema_name == "sharegpt"
    sample = ll[1]
    # system prepended + 2 turns from the conversation
    assert [m["role"] for m in sample] == ["system", "user", "assistant"]
    assert sample[1]["content"] == "q2"


def test_low_level_loads_jsonl_messages(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(
        tmp_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
        ],
    )
    ll = VarlenLowLevelDataset(path)
    assert ll.schema_name == "openai-messages"
    sample = ll[0]
    assert [m["role"] for m in sample] == ["system", "user", "assistant"]


def test_low_level_jsonl_heterogeneous_columns(tmp_path):
    """Real datasets often mix rows that have / lack an optional field. Our
    pandas-based loader must accept the union schema without ``CastError``."""
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    rows = [{"instruction": "a", "output": "x"}] * 100 + [
        {"instruction": "b", "output": "y", "file": "extra"}
    ] * 100
    path = _write_jsonl(tmp_path, rows)
    ll = VarlenLowLevelDataset(path)
    assert len(ll) == 200
    # Both halves should normalize to the same messages structure.
    assert [m["role"] for m in ll[0]] == ["system", "user", "assistant"]
    assert [m["role"] for m in ll[150]] == ["system", "user", "assistant"]


def test_low_level_rejects_unknown_schema(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(tmp_path, [{"foo": "bar"}])
    with pytest.raises(ValueError, match="cannot infer schema"):
        VarlenLowLevelDataset(path)
