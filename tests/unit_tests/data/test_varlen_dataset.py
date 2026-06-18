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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Import via the public module path so this test gets discovered through the
# regular pytest entry point. The functions under test are pure Python and do
# not require torch.distributed.
from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.datasets.varlen_dataset import (
    MockVarlenDataset,
    VarlenDataset,
    VarlenLowLevelDataset,
    _alpaca_to_messages,
    _looks_like_hf_id,
    _messages_passthrough,
    _raw_text_loader,
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
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
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
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
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
                {"role": "assistant", "content": "hi alice", "tool_calls": [{"function": "foo"}]},
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
            {"messages": [{"role": "user", "content": [{"type": "image", "url": "x.png"}]}]}
        )


def test_alpaca_rejects_non_string_field():
    with pytest.raises(ValueError, match="must be a string"):
        _alpaca_to_messages({"instruction": ["a", "b"], "output": "x"})


def test_sharegpt_rejects_list_value():
    with pytest.raises(ValueError, match="must be a string"):
        _sharegpt_to_messages({"conversations": [{"from": "human", "value": [1, 2, 3]}]})


# ----------------------------------------------------------------------------
# Pretrain-text schema
# ----------------------------------------------------------------------------


def test_raw_text_loader_returns_string():
    """``text``-column samples are returned as plain strings (not messages)."""
    out = _raw_text_loader({"text": "Once upon a time...", "id": "doc-1"})
    assert isinstance(out, str)
    assert out == "Once upon a time..."


def test_raw_text_loader_handles_empty():
    assert _raw_text_loader({"text": None}) == ""
    assert _raw_text_loader({}) == ""


def test_raw_text_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        _raw_text_loader({"text": [1, 2, 3]})


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


def test_select_converter_pretrain_text():
    fn, name = _select_converter(["text", "id"])
    assert name == "pretrain-text"
    assert fn is _raw_text_loader


def test_select_converter_pretrain_text_with_metadata():
    """Real corpora (e.g. Dolma) have ``text`` + ``url`` + ``metadata``."""
    fn, name = _select_converter(["text", "url", "metadata", "id"])
    assert name == "pretrain-text"


def test_select_converter_alpaca_beats_pretrain_text():
    """When both ``instruction``/``output`` and ``text`` are present (rare),
    the alpaca schema is more specific and should win."""
    fn, name = _select_converter(["text", "instruction", "output"])
    assert name == "alpaca"


def test_select_converter_messages_beats_pretrain_text():
    fn, name = _select_converter(["text", "messages"])
    assert name == "openai-messages"


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
            {"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]},
            {"conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]},
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
            }
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


def test_low_level_loads_jsonl_pretrain_text(tmp_path):
    """Pretrain-text corpora (Dolma / OLMo midtraining) typically have
    ``text`` + extra fields like ``id`` / ``url`` / ``metadata``."""
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(
        tmp_path,
        [
            {"text": "Doc one body...", "id": "1", "url": "https://x/1"},
            {"text": "Doc two body...", "id": "2", "url": "https://x/2"},
        ],
    )
    ll = VarlenLowLevelDataset(path)
    assert ll.schema_name == "pretrain-text"
    assert len(ll) == 2
    # Each item is a raw string, NOT a messages list.
    assert ll[0] == "Doc one body..."
    assert ll[1] == "Doc two body..."


# ----------------------------------------------------------------------------
# VarlenDataset / MockVarlenDataset __getitem__ (fake tokenizer, no GPU)
#
# These bypass the heavy SFTDataset.__init__ and inject the minimal attributes
# __getitem__ reads, so the EOD handling / position-based loss masking /
# pad-to-divisor / packing-metadata contracts can be unit tested without a
# real tokenizer or torch.distributed.
# ----------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer for exercising VarlenDataset.__getitem__.

    ``tokenize`` maps each character to a non-zero id (so plain text never
    collides with ``eod``/``pad``); ``tokenize("")`` returns ``[]`` to exercise
    the empty-row guard. ``tokenize_conversation`` masks non-assistant turns
    with ``IGNORE_INDEX`` in the targets.
    """

    def __init__(self, eod: int = 0, pad=None):
        self._eod = eod
        self._pad = pad

    @property
    def eod(self):
        return self._eod

    @property
    def pad(self):
        return self._pad

    def tokenize(self, text):
        return [ord(c) % 100 + 1 for c in text]  # always >= 1, never eod (0)

    def tokenize_conversation(self, messages, return_target=True, add_generation_prompt=False):
        tokens, targets = [], []
        for m in messages:
            ids = self.tokenize(m["content"])
            tokens.extend(ids)
            # Only assistant turns contribute to the loss; prompt is masked.
            targets.extend(ids if m["role"] == "assistant" else [IGNORE_INDEX] * len(ids))
        return (torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64))


def _make_config(tokenizer, seq_length=64, *, cp=1, dp=1, sp=1, sbhd=False):
    return SimpleNamespace(
        tokenizer=tokenizer,
        sequence_length=seq_length,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        varlen_sbhd_validation=sbhd,
        data_parallel_size=dp,
        context_parallel_size=cp,
        sequence_parallel_size=sp,
    )


def _make_varlen(items, config):
    ds = VarlenDataset.__new__(VarlenDataset)
    ds.config = config
    ds.dataset = items
    ds.indices = np.arange(len(items))
    return ds


def _make_mock_varlen(token_arrays, config):
    ds = MockVarlenDataset.__new__(MockVarlenDataset)
    ds.config = config
    ds.dataset = token_arrays  # each item exposes .tolist()
    ds.indices = np.arange(len(token_arrays))
    return ds


def test_getitem_thd_pretrain_text_keys_and_shapes():
    tok = _FakeTokenizer(eod=0, pad=7)
    ds = _make_varlen(["hello world"], _make_config(tok, seq_length=64))
    out = ds[0]
    assert set(out) == {
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "original_seq_len",
        "padded_seq_len",
    }
    n = out["tokens"].numel()
    assert out["labels"].numel() == n
    assert out["loss_mask"].numel() == n
    assert out["position_ids"].numel() == n
    assert int(out["padded_seq_len"].item()) == n


def test_getitem_thd_sft_prompt_is_masked():
    tok = _FakeTokenizer(eod=0, pad=7)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    ds = _make_varlen([messages], _make_config(tok, seq_length=64))
    out = ds[0]
    # Prompt (user) tokens are IGNORE_INDEX in labels and must be masked out;
    # assistant tokens must contribute to the loss.
    labels = out["labels"]
    loss_mask = out["loss_mask"]
    assert torch.all(loss_mask[labels == IGNORE_INDEX] == 0.0)
    assert loss_mask.sum() > 0  # assistant span still contributes


def test_getitem_thd_pad_masked_by_position_keeps_real_eod():
    """Regression: with pad falling back to eod, the real end-of-document EOD
    target must stay in the loss (masked by position, not by value)."""
    tok = _FakeTokenizer(eod=0, pad=None)  # pad falls back to eod
    # cp=2 -> pad divisor = cp*2 = 4, so a 3-token doc gets a padding tail.
    ds = _make_varlen(["abc"], _make_config(tok, seq_length=64, cp=2))
    out = ds[0]
    loss_mask = out["loss_mask"].tolist()
    labels = out["labels"].tolist()
    # tokens=[a,b,c,eod] padded to 4 -> labels=[b,c,eod,eod(pad)]
    assert len(loss_mask) == 4
    # index 2 is the real end-of-document EOD target -> kept (would be wrongly
    # dropped by value-based ``labels == pad`` masking).
    assert labels[2] == tok.eod and loss_mask[2] == 1.0
    # index 3 is the appended pad -> masked.
    assert loss_mask[3] == 0.0


def test_getitem_thd_padded_to_divisor():
    tok = _FakeTokenizer(eod=0, pad=7)
    ds = _make_varlen(["abcde"], _make_config(tok, seq_length=64, cp=2))  # divisor 4
    out = ds[0]
    assert int(out["padded_seq_len"].item()) % 4 == 0


def test_getitem_thd_empty_text_does_not_crash():
    """A blank pretrain-text row tokenizes to [] -> must not crash and must
    yield a valid (non-zero-length) sample."""
    tok = _FakeTokenizer(eod=0, pad=7)
    ds = _make_varlen([""], _make_config(tok, seq_length=64))
    out = ds[0]
    assert out["tokens"].numel() >= 1
    assert out["labels"].numel() == out["tokens"].numel()
    assert out["loss_mask"].numel() == out["tokens"].numel()


def test_getitem_sbhd_pads_to_seq_length_and_masks_tail():
    tok = _FakeTokenizer(eod=0, pad=None)
    ds = _make_varlen(["abc"], _make_config(tok, seq_length=8, sbhd=True))
    out = ds[0]
    # SBHD emits fixed [seq_length] samples with no packing metadata.
    assert set(out) == {"tokens", "labels", "loss_mask", "position_ids"}
    assert out["tokens"].numel() == 8
    loss_mask = out["loss_mask"].tolist()
    # tokens=[a,b,c,eod]: valid_len=3 -> first 3 kept (incl. real eod), rest masked.
    assert loss_mask[0:3] == [1.0, 1.0, 1.0]
    assert all(v == 0.0 for v in loss_mask[3:])


def test_mock_getitem_thd_keys_and_pad_fallback():
    tok = _FakeTokenizer(eod=0, pad=None)  # exercise the eod fallback (no crash)
    ds = _make_mock_varlen([np.array([1, 2, 3, 4], dtype=np.int64)], _make_config(tok, cp=2))
    out = ds[0]
    assert set(out) == {
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "original_seq_len",
        "padded_seq_len",
    }
    n = out["tokens"].numel()
    assert out["labels"].numel() == n and out["loss_mask"].numel() == n
    assert int(out["padded_seq_len"].item()) % 4 == 0


# ----------------------------------------------------------------------------
# THD handoff: _unpack_batch contract for VarlenDataset-style samples
#
# VarlenDataset already emits one unpacked sub-sample carrying ``padded_seq_len``,
# so _unpack_batch must short-circuit (no cu_seqlens slicing) and only normalize
# the collate batch dim. SFTDataset-style pre-packed samples (cu_seqlens, no
# padded_seq_len) still take the slicing path.
# ----------------------------------------------------------------------------


def test_unpack_batch_short_circuits_for_varlen_samples():
    from megatron.core.datasets.data_schedule_utils import _unpack_batch

    # Two VarlenDataset-style samples, each already a single sub-sample with a
    # leading batch dim (as added by the default collate_fn) and padded_seq_len.
    batch = [
        {
            "tokens": torch.arange(4, dtype=torch.int64).view(1, 4),
            "labels": torch.arange(4, dtype=torch.int64).view(1, 4),
            "loss_mask": torch.ones(1, 4),
            "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
            "padded_seq_len": torch.tensor([4], dtype=torch.int32),
        },
        {
            "tokens": torch.arange(8, dtype=torch.int64).view(1, 8),
            "labels": torch.arange(8, dtype=torch.int64).view(1, 8),
            "loss_mask": torch.ones(1, 8),
            "position_ids": torch.arange(8, dtype=torch.int64).view(1, 8),
            "padded_seq_len": torch.tensor([8], dtype=torch.int32),
            "original_seq_len": torch.tensor([8], dtype=torch.int32),
        },
    ]
    out = _unpack_batch(batch)
    # Short-circuit: same number of samples (no slicing into sub-samples).
    assert len(out) == 2
    # Leading collate batch dim dropped.
    assert out[0]["tokens"].shape == (4,)
    assert out[1]["tokens"].shape == (8,)
    # Missing original_seq_len synthesized from padded_seq_len.
    assert "original_seq_len" in out[0]
    assert int(out[0]["original_seq_len"].item()) == 4
    # Existing original_seq_len preserved.
    assert int(out[1]["original_seq_len"].item()) == 8


def test_unpack_batch_slices_prepacked_cu_seqlens_samples():
    from megatron.core.datasets.data_schedule_utils import _unpack_batch

    # SFTDataset-style pre-packed sample: two sub-sequences [0:3) and [3:5),
    # described by cu_seqlens, NO padded_seq_len -> takes the slicing path.
    batch = [
        {
            "tokens": torch.arange(5, dtype=torch.int64),
            "labels": torch.arange(5, dtype=torch.int64),
            "loss_mask": torch.ones(5),
            "position_ids": torch.arange(5, dtype=torch.int64),
            "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        }
    ]
    out = _unpack_batch(batch)
    # One packed sample with two sub-sequences -> two unpacked samples.
    assert len(out) == 2
    assert out[0]["tokens"].numel() == 3
    assert out[1]["tokens"].numel() == 2
    assert int(out[0]["padded_seq_len"].item()) == 3
    assert int(out[1]["padded_seq_len"].item()) == 2


# ----------------------------------------------------------------------------
# DataLoader collate selection (distributed; run under torch.distributed.run).
#
# Validates the build_pretraining_data_loader contract for the varlen paths:
#   * --varlen-sbhd-validation emits fixed-length [seq_length] samples that the
#     DEFAULT collate stacks into a [mbs, seq_length] batch.
#   * The THD path (--use-varlen-dataset without SBHD) uses the identity collate
#     (variable-length dicts are returned as a list, not stacked).
# ----------------------------------------------------------------------------


def _build_varlen_for_loader(items, config, num_samples):
    from megatron.core.datasets.utils import Split

    ds = VarlenDataset.__new__(VarlenDataset)
    ds.config = config
    ds.dataset = items
    ds.indices = np.arange(len(items))
    ds.num_samples = num_samples
    ds.index_split = Split.train
    return ds


def _loader_args(*, use_varlen, sbhd, scheduler, mbs, gbs=None):
    return SimpleNamespace(
        dataloader_type='single',
        micro_batch_size=mbs,
        global_batch_size=mbs if gbs is None else gbs,
        full_validation=False,
        num_workers=0,
        use_varlen_dataset=use_varlen,
        varlen_sbhd_validation=sbhd,
        sequence_packing_scheduler=scheduler,
    )


def test_sbhd_validation_dataloader_uses_default_collate():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tok = _FakeTokenizer(eod=0, pad=7)
        seq_len, mbs = 16, 2
        # One global batch needs micro_batch_size * data_parallel_size samples;
        # size the dataset off the runtime DP world size so this passes under
        # any --nproc-per-node (the CI default is 8 ranks -> dp=8).
        dp = parallel_state.get_data_parallel_world_size()
        n = mbs * dp * 4
        cfg = _make_config(tok, seq_length=seq_len, sbhd=True)
        ds = _build_varlen_for_loader(["hello world"] * n, cfg, num_samples=n)
        set_args(_loader_args(use_varlen=True, sbhd=True, scheduler=None, mbs=mbs))
        loader = build_pretraining_data_loader(ds, consumed_samples=0)
        batch = next(iter(loader))
        # Default collate stacks fixed-length SBHD samples into a tensor batch.
        assert isinstance(batch, dict)
        assert batch["tokens"].shape == (mbs, seq_len)
        assert batch["labels"].shape == (mbs, seq_len)
        assert batch["loss_mask"].shape == (mbs, seq_len)
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()


def test_thd_dataloader_uses_identity_collate():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tok = _FakeTokenizer(eod=0, pad=7)
        mbs = 2
        dp = parallel_state.get_data_parallel_world_size()
        n = mbs * dp * 4
        cfg = _make_config(tok, seq_length=64, sbhd=False)
        # Variable-length samples so identity collate is required.
        variable = ["a", "abcdef", "xy", "qwerty"]
        items = [variable[i % len(variable)] for i in range(n)]
        ds = _build_varlen_for_loader(items, cfg, num_samples=n)
        set_args(_loader_args(use_varlen=True, sbhd=False, scheduler="dp_balanced", mbs=mbs))
        loader = build_pretraining_data_loader(ds, consumed_samples=0)
        batch = next(iter(loader))
        # Identity collate returns the raw list of per-sample dicts (unstacked).
        assert isinstance(batch, list)
        assert len(batch) == mbs
        assert "padded_seq_len" in batch[0]
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()


def test_packing_scheduler_dataloader_yields_microbatches():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tok = _FakeTokenizer(eod=0, pad=7)
        mbs = 2
        num_microbatches = 3
        dp = parallel_state.get_data_parallel_world_size()
        gbs = mbs * dp * num_microbatches
        n = gbs * 2
        cfg = _make_config(tok, seq_length=64, dp=dp, cp=1)
        variable = ["a", "abcdef", "xy", "qwerty"]
        items = [variable[i % len(variable)] for i in range(n)]
        ds = _build_varlen_for_loader(items, cfg, num_samples=n)
        set_args(
            _loader_args(
                use_varlen=True,
                sbhd=False,
                scheduler="dp_balanced",
                mbs=mbs,
                gbs=gbs,
            )
        )
        loader = build_pretraining_data_loader(ds, consumed_samples=0)
        batch = next(iter(loader))
        # The packing scheduler calls next(data_iterator) num_microbatches times;
        # each loader step must therefore be one local microbatch, not all
        # local samples from the global batch.
        assert isinstance(batch, list)
        assert len(batch) == mbs
        assert "padded_seq_len" in batch[0]
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()
