# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests of the v3 codec and replay, without Megatron/CUDA initialization.

Run with ``uv run --no-project --with torch --with numpy --with pyarrow
--with pytest python -m pytest tests/offline_kd_cpu``. Load the deliberately
runtime-independent modules under a private namespace to avoid importing the
Megatron package's GPU initialization code.
"""

import argparse
import ast
import collections
import importlib
import json
import sys
import types
import typing
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

_namespace = "offline_kd_cpu"
_package = types.ModuleType(_namespace)
_package.__path__ = [str(Path(__file__).resolve().parents[2] / "megatron/training/distillation")]
sys.modules[_namespace] = _package
codec = importlib.import_module(f"{_namespace}.v3_format")
arrow = importlib.import_module(f"{_namespace}.format.arrow")
storage_module = importlib.import_module(f"{_namespace}.v3_storage")
replay = importlib.import_module(f"{_namespace}.v3_replay")
Storage = storage_module.Storage


def inputs(ids, length=8, packed=False):
    tokens = torch.tensor(ids).reshape(-1, 1) * 100 + torch.arange(length)
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "position_ids": torch.arange(length).expand(len(ids), -1),
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
    }
    batch["loss_mask"][:, 0] = 0
    if packed:
        batch["cu_seqlens"] = torch.tensor([[0, length // 2, length]]).expand(len(ids), -1)
    return batch


def settings(dp=2, mbs=2, gbs=8, *, contract=None, recorded=None, provenance=None):
    """Cache metadata shared by every file: required settings, recorded settings, provenance.

    Overrides of recorded keys may also be passed through ``contract`` for brevity.
    """
    required = {
        "first_sample": 0,
        "writers": dp,
        "writer_mbs": mbs,
        "attention_mode": "causal",
        "prefix_stable": True,
        "tokenizer": {"sha256": "test", "size": 3, "special_ids": {}, "eod_id": 2},
        "logprob_vocab_size": 1 << 17,
    }
    noted = {
        "teacher": "/teacher@100",
        "seq_length": 8,
        "padding_divisor": 1,
        "data": {"kind": "blend", "blend": [[1.0, "corpus"]]},
        "split": "99,1,0",
        "seed": 7,
        "num_samples": 1024,
        "builder": "GPTDataset",
    }
    for key, value in (contract or {}).items():
        (noted if key in codec.RECORDED_KEYS else required)[key] = value
    noted.update(recorded or {})
    return {
        "format_version": 3,
        "contract": required,
        "recorded": noted,
        "provenance": {"producer": "megatron", "save_interval": 2, **(provenance or {})},
    }


def scored(tokens, loss_mask):
    """Identity targets as replayed: rows exist only where the loss weight is nonzero."""
    return torch.where(loss_mask != 0, tokens.long(), 0)


def make_record(captured, contract, logprobs=False):
    """Finalize captured inputs and targets exactly as the saver does.

    Targets are identity rows (value = index = token ID) by default, with no stored
    tail; with ``logprobs`` they are genuine log probabilities with a stored fp32 tail.
    Returns ``(record, values, indices, tail)``.
    """
    target_ids = captured["tokens"].to(torch.int64).reshape(-1, 1)
    values = target_ids.float()
    if logprobs:
        values = (0.5 + (target_ids % 7).float() / 20).log()
    record, values, indices, tail = codec.finalize_record(
        captured,
        values,
        target_ids,
        codec.tail_logprob(values),
        isolated=contract["attention_mode"] == "isolated_segments",
    )
    return record, values, indices, tail if logprobs else None


# Small blocks so every file has several and block boundaries are exercised.
TEST_BLOCK_BYTES = 256


def write_flush(
    storage,
    metadata,
    rank,
    begin,
    finish,
    *,
    gbs=8,
    packed=False,
    source=None,
    logprobs=False,
    block_bytes=TEST_BLOCK_BYTES,
):
    """Write one writer's flush exactly as the saver does."""
    contract = metadata["contract"]
    dp, mbs = contract["writers"], contract["writer_mbs"]
    batches = []
    for iteration in range(begin, finish):
        start = iteration * gbs
        ids = [
            start + (mb * dp + rank) * mbs + j
            for mb in range(gbs // (dp * mbs))
            for j in range(mbs)
        ]
        batch = (source or inputs)(ids, packed=packed)
        batches.append(
            arrow.samples_batch(
                *make_record(codec.capture_batch(batch), contract, logprobs),
                vocab_size=contract["logprob_vocab_size"],
            )
        )
    data, index = arrow.write_file(batches, metadata, block_bytes=block_bytes)
    name = storage_module.FileName(rank, begin * gbs, finish * gbs)
    storage_module.publish(storage, name, data, index)


def write_cache(
    root,
    dp=2,
    mbs=2,
    gbs=8,
    iterations=4,
    bundle=2,
    packed=False,
    contract=None,
    provenance=None,
    metadata=None,
    start=0,
    source=None,
    logprobs=False,
):
    storage = Storage(str(root))
    if metadata is None:
        mode = {"attention_mode": "isolated_segments"} if packed else {}
        metadata = settings(
            dp, mbs, gbs, contract={**mode, **(contract or {})}, provenance=provenance
        )
    for begin in range(start, iterations, bundle):
        finish = min(begin + bundle, iterations)
        for rank in range(dp):
            name = storage_module.FileName(rank, begin * gbs, finish * gbs)
            if not storage.exists(name.index):
                write_flush(
                    storage,
                    metadata,
                    rank,
                    begin,
                    finish,
                    gbs=gbs,
                    packed=packed,
                    source=source,
                    logprobs=logprobs,
                )
    return storage, metadata


def groups(storage, metadata):
    return replay.discover_groups(storage, metadata)[0]


def reader(storage, metadata, **kwargs):
    return replay.ReplayReader(storage, metadata, groups(storage, metadata), **kwargs)


def ids_of(samples):
    return [s["sample_id"] for s in samples]


def test_capture_owns_a_compact_snapshot():
    source = inputs([0, 1], packed=True)
    source["labels"][0, 0] = -100
    source["cu_seqlens"] = torch.tensor([[0, 4, 8, 8], [0, 2, 8, 8]])
    captured = codec.capture_batch(source)
    assert captured["tokens"].dtype == torch.int32
    assert captured["loss_mask"].dtype == torch.bool
    assert captured["labels"][0] == -100
    assert captured["cu_seqlens"][1].tolist() == [0, 2, 8]
    source["tokens"].fill_(999)
    assert captured["tokens"][0] == 0  # Owned snapshot, not an alias.


def test_fractional_mask_and_integer_overflow():
    source = inputs([0])
    source["loss_mask"][0, 2] = 0.25
    captured = codec.capture_batch(source)
    assert captured["loss_mask"].dtype == torch.float32
    assert captured["loss_mask"][2] == 0.25
    source["tokens"][0, 0] = 1 << 40
    with pytest.raises(ValueError, match="int32"):
        codec.capture_batch(source)


def roundtrip(record, values, indices, tail, *, vocab_size=1 << 17, block_bytes=1 << 20):
    """Write one file of these samples and decode every block back."""
    batch = arrow.samples_batch(record, values, indices, tail, vocab_size=vocab_size)
    data, index = arrow.write_file([batch], settings(), block_bytes=block_bytes)
    decoded = []
    for block in range(len(index["blocks"])):
        offset, size = arrow.block_range(index, block, inputs_only=False)
        decoded.append(
            arrow.decode_block(
                arrow.read_block(index, block, data[offset : offset + size]), vocab_size=vocab_size
            )
        )
    return data, index, decoded


def target_roundtrip(values, indices, tail=None, *, vocab_size):
    """Store target rows (one token each) in a file and decode them back."""
    rows = values.shape[0]
    record = {
        "sample_offsets": torch.tensor([0, rows]),
        **{key: torch.zeros(rows, dtype=torch.int32) for key in ("tokens", "labels")},
        "position_ids": torch.arange(rows),
        "loss_mask": torch.ones(rows, dtype=torch.bool),
        "segment_lengths": torch.tensor([rows] if rows else []),
        "segment_counts": torch.tensor([1 if rows else 0]),
    }
    _, _, [(_, targets)] = roundtrip(record, values, indices, tail, vocab_size=vocab_size)
    return targets


def batch_targets(batch):
    """Raw ``(values, indices, tail)`` columns of a sample batch, without validation."""
    values = arrow._flat(batch.column("values"))[1]
    high = (
        arrow._flat(batch.column("indices_high"))[1]
        if "indices_high" in batch.schema.names
        else None
    )
    indices = codec.join_indices(arrow._flat(batch.column("indices_low"))[1], high)
    tail = arrow._flat(batch.column("tail"))[1] if "tail" in batch.schema.names else None
    return arrow._tensor(values), arrow._tensor(indices), tail


def test_files_are_plain_arrow_ipc_with_settings_in_the_schema(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2, logprobs=True)
    name = storage_module.FileName(1, 0, 16)
    table = pa.ipc.open_file(storage.path(name.data)).read_all()
    assert table.schema.names == [
        "tokens",
        "labels",
        "position_ids",
        "loss_mask",
        "segments",
        "values",
        "indices_low",
        "indices_high",
        "tail",
    ]
    assert json.loads(table.schema.metadata[arrow.META_KEY]) == metadata
    assert table.num_rows == 8 and table.column("tokens")[0].as_py() == list(range(200, 208))
    index = storage_module.read_index(storage, name)
    assert index["size"] == storage.size(name.data) and len(index["blocks"]) > 1
    assert sum(block["rows"] for block in index["blocks"]) == 8
    assert storage_module.read_meta(storage, name) == metadata


def test_input_only_reads_fetch_one_prefix_range(tmp_path):
    storage, _ = write_cache(tmp_path, iterations=2, logprobs=True)
    name = storage_module.FileName(0, 0, 16)
    index = storage_module.read_index(storage, name)
    data = (tmp_path / name.data).read_bytes()
    for block, entry in enumerate(index["blocks"]):
        assert entry["input_length"] < entry["length"]
        offset, size = arrow.block_range(index, block, inputs_only=True)
        batch = arrow.read_block(index, block, data[offset : offset + size], inputs_only=True)
        assert batch.schema.names == list(arrow.INPUT_COLUMNS)
        inputs_, targets = arrow.decode_block(batch)
        assert targets is None and inputs_["sample_offsets"].numel() == entry["rows"] + 1
        # The prefix alone cannot serve the targets: no other bytes are ever fetched.
        with pytest.raises(OSError, match="did not fetch"):
            arrow.read_block(index, block, data[offset : offset + size])


def test_index_is_bound_to_its_data_file(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2)
    name = storage_module.FileName(0, 0, 16)
    index = storage_module.read_index(storage, name)
    arrow.check_index(index, storage.size(name.data))
    with pytest.raises(ValueError, match="another version of the file"):
        arrow.check_index(index, storage.size(name.data) + 1)
    with pytest.raises(ValueError, match="index version"):
        arrow.check_index({**index, "index_version": 99})


def test_binary_and_fractional_loss_masks_share_one_file():
    batches = []
    for weight in (1.0, 0.25):
        source = inputs([0])
        source["loss_mask"][0, 2] = weight
        record = codec.finalize_record(
            codec.capture_batch(source),
            torch.zeros(8, 1),
            torch.zeros(8, 1, dtype=torch.long),
            torch.zeros(8),
            isolated=False,
        )
        batches.append(arrow.samples_batch(*record[:3], None, vocab_size=8))
    assert batches[0].schema.field("loss_mask").type == pa.large_list(pa.bool_())
    data, index = arrow.write_file(batches, settings())
    table = pa.ipc.open_file(pa.py_buffer(data)).read_all()
    assert table.schema.field("loss_mask").type == pa.large_list(pa.float32())
    assert table.column("loss_mask").to_pylist() == [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0.25, 1, 1, 1, 1, 1],
    ]


def test_oversized_sample_gets_its_own_block():
    record = codec.finalize_record(
        codec.capture_batch(inputs([0, 1, 2])),
        torch.zeros(24, 1),
        torch.zeros(24, 1, dtype=torch.long),
        torch.zeros(24),
        isolated=False,
    )
    _, index, decoded = roundtrip(*record[:3], None, block_bytes=1)
    assert [block["rows"] for block in index["blocks"]] == [1, 1, 1]
    assert [block["supervised"] for block in index["blocks"]] == [7, 7, 7]
    assert [inputs_["tokens"].tolist() for inputs_, _ in decoded][1] == list(range(100, 108))


def test_tail_is_the_full_precision_mass_outside_retained_entries():
    full = torch.tensor([[3.0, 1.0, 0.5, -2.0, -4.0]]).log_softmax(-1)
    head = full[:, :3]
    assert torch.allclose(codec.tail_logprob(head), full[:, 3:].logsumexp(-1))
    # Sentinel entries are not retained, so their mass stays in the tail.
    masked = head.clone()
    masked[0, 2] = codec.LOGPROB_SENTINEL
    assert torch.allclose(codec.tail_logprob(masked), full[:, 2:].logsumexp(-1))
    # A fully retained distribution has no tail; a row with nothing retained is all tail.
    assert codec.tail_logprob(full).item() == codec.LOGPROB_SENTINEL
    assert codec.tail_logprob(torch.full((1, 2), codec.LOGPROB_SENTINEL)).item() == 0
    # Rounding the head to fp16 first distorts a small tail; the stored fp32 tail does not.
    confident = torch.tensor([[9.0, 5.0, 4.5, 4.0, 3.0, 2.0, 2.0]]).log_softmax(-1)
    exact = confident[:, 6:].logsumexp(-1)  # About 0.09% of the mass.
    rounded = codec.tail_logprob(confident[:, :6].to(torch.float16))
    assert (rounded - exact).abs() > 0.01  # Off by about 2% of the tail mass.
    assert torch.allclose(codec.tail_logprob(confident[:, :6]), exact, atol=1e-4)


@pytest.mark.parametrize("case", ["mass", "nan", "dtype", "shape"])
def test_malformed_tail_rejected(case):
    values = torch.tensor([[0.6, 0.3]]).log()
    indices = torch.tensor([[0, 1]])
    tail = torch.tensor([0.1]).log()
    codec.check_targets(values, indices, tail, vocab_size=4)
    if case == "mass":
        tail = torch.tensor([0.5]).log()
    elif case == "nan":
        tail = torch.tensor([float("nan")])
    elif case == "dtype":
        tail = tail.double()
    else:
        tail = torch.zeros(2)
    with pytest.raises(ValueError, match="tail"):
        codec.check_targets(values, indices, tail, vocab_size=4)


def test_targets_are_stored_only_where_loss_is_applied():
    # A conversation: prompt (0), answer (1), next prompt (0), answer, trailing padding.
    weights = [0, 0, 1, 1, 0, 0, 1, 0]
    source = inputs([0])
    source["loss_mask"] = torch.tensor([weights], dtype=torch.float32)
    captured = codec.capture_batch(source)
    ids = captured["tokens"].long().reshape(-1, 1)
    values = (0.5 + (ids % 7).float() / 20).log()
    record, kept, kept_ids, kept_tail = codec.finalize_record(
        captured, values, ids, codec.tail_logprob(values), isolated=False
    )
    assert record["tokens"].tolist() == [0, 1, 2, 3, 4, 5, 6]  # Only the padding is stripped.
    assert kept_ids.reshape(-1).tolist() == [2, 3, 6]  # Context tokens carry no targets.
    _, _, [(inputs_, targets)] = roundtrip(record, kept, kept_ids, kept_tail)
    with pytest.raises(ValueError, match="one row per nonzero loss weight"):
        arrow.samples_batch(record, kept[:2], kept_ids[:2], kept_tail[:2], vocab_size=1 << 17)
    assert torch.equal(targets[1].reshape(-1), kept_ids.reshape(-1))
    sample = {field: inputs_[field] for field in codec.TOKEN_FIELDS}
    sample.update(
        segments=inputs_["segment_lengths"],
        sample_id=0,
        teacher_values=targets[0],
        teacher_indices=targets[1],
        teacher_tail=targets[2],
    )
    batch = replay.collate_samples([sample], 8)
    replayed = batch["_kd_indices"][0].reshape(-1)
    assert replayed.tolist() == [0, 0, 2, 3, 0, 0, 6, 0]
    tail = batch["_kd_tail"][0].reshape(-1)
    assert torch.equal(tail[[2, 3, 6]], kept_tail)
    assert (tail[[0, 1, 4, 5, 7]] == 0).all()
    assert (batch["_kd_values"][0].reshape(-1)[[0, 1, 4, 5, 7]] == codec.LOGPROB_SENTINEL).all()


def test_entries_beyond_the_student_vocabulary_join_the_tail(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2, logprobs=True)
    full = reader(storage, metadata)
    small = reader(storage, metadata, student_vocab_size=104)
    try:
        exact, masked = full.samples([1], 2)[0], small.samples([0, 1], 2)[1]
    finally:
        full.close()
        small.close()
    # Sample 1's rows are tokens 101..107; the student cannot represent 104 and up.
    gone = exact["teacher_indices"].reshape(-1) >= 104
    assert (masked["teacher_values"].reshape(-1)[gone] == codec.LOGPROB_SENTINEL).all()
    moved = exact["teacher_values"].reshape(-1)[gone]
    assert torch.allclose(
        masked["teacher_tail"][gone], torch.logaddexp(exact["teacher_tail"][gone], moved)
    )
    assert torch.equal(masked["teacher_tail"][~gone], exact["teacher_tail"][~gone])


@pytest.mark.parametrize(
    "vocab_size,encoding",
    [(1, "u16"), (65536, "u16"), (65537, "u16+u8"), (1 << 17, "u16+u8"), (1 << 24, "u16+u8")],
)
def test_index_encoding_roundtrip(vocab_size, encoding):
    indices = torch.tensor([[0, vocab_size - 1, -1, 0, vocab_size - 1]])
    values = torch.tensor([[-1.0, -2.0, -1000.0, -3.0, -4.0]])
    low, high = codec.split_indices(indices.numpy(), values.numpy(), vocab_size=vocab_size)
    assert low.dtype == np.uint16 and (high is None) == (encoding == "u16")
    if high is not None:
        assert high.dtype == np.uint8 and high.shape == indices.shape
    record = {
        "sample_offsets": torch.tensor([0, 1]),
        "tokens": torch.tensor([0]),
        "labels": torch.tensor([0]),
        "position_ids": torch.tensor([0]),
        "loss_mask": torch.tensor([True]),
        "segment_lengths": torch.tensor([1]),
        "segment_counts": torch.tensor([1]),
    }
    _, _, [(_, (restored, ids, _))] = roundtrip(
        record, values, indices, None, vocab_size=vocab_size
    )
    assert torch.equal(restored, values)
    keep = values != -1000.0
    assert torch.equal(ids[keep], indices[keep])
    # The -1 selection sentinel decodes to all ones; validity comes from the logprob sentinel.
    assert ids[0, 2] == (0xFFFF if encoding == "u16" else 0xFFFFFF)


@pytest.mark.parametrize("vocab_size", [0, -1, (1 << 24) + 1, 1.5, True])
def test_index_encoding_rejects_invalid_vocab_size(vocab_size):
    with pytest.raises(ValueError, match="vocabulary size"):
        codec.index_encoding_for_vocab(vocab_size)


@pytest.mark.parametrize("case", ["bounds", "shape", "high_shape", "integer_values"])
def test_malformed_targets_rejected(case):
    values, indices = torch.zeros(1, 2), torch.tensor([[0, 65538]])
    if case == "bounds":  # ID 65538 exceeds the declared vocabulary.
        with pytest.raises(ValueError, match="out of range"):
            codec.check_targets(values, indices, None, vocab_size=65537)
    elif case == "shape":
        with pytest.raises(ValueError, match="Malformed"):
            codec.check_targets(values[:, :1], indices, None, vocab_size=None)
    elif case == "high_shape":
        with pytest.raises(ValueError, match="Malformed"):
            codec.join_indices(np.zeros((1, 2), np.uint16), np.zeros((1, 1), np.uint8))
    else:
        with pytest.raises(ValueError, match="Malformed"):
            codec.check_targets(indices, indices, None, vocab_size=None)


@pytest.mark.parametrize("index,value", [(-2, -1000.0), (-1, -1.0), (65537, -1.0)])
def test_target_encoder_rejects_invalid_ids(index, value):
    with pytest.raises(ValueError, match="vocabulary index"):
        codec.split_indices(np.array([[index]]), np.array([[value]]), vocab_size=65537)


@pytest.mark.parametrize(
    "contract",
    [
        {"attention_mode": "unknown"},
        {"prefix_stable": "yes"},
        {"logprob_vocab_size": 0},
        {"tokenizer": {"size": 3}},
        {"writers": 0},
        {"first_sample": -1},
    ],
)
def test_unsupported_required_metadata_rejected(contract):
    with pytest.raises(ValueError):
        codec.validate_metadata(settings(contract=contract))


@pytest.mark.parametrize("key", [*codec.REQUIRED_KEYS, *codec.REPLAY_KEYS])
def test_metadata_requires_every_required_key(key):
    metadata = settings()
    del metadata["contract"][key]
    with pytest.raises(ValueError, match=f"missing required settings: {key}"):
        codec.validate_metadata(metadata)


def test_recorded_settings_and_provenance_are_optional():
    metadata = settings()
    del metadata["recorded"], metadata["provenance"]
    codec.validate_metadata(metadata)
    # A corpus file (another producer) needs no replay layout.
    for key in codec.REPLAY_KEYS:
        del metadata["contract"][key]
    codec.validate_metadata(metadata, replay=False)


def test_required_and_recorded_comparisons():
    base = settings()
    assert codec.mismatched_settings(settings(provenance={"save_interval": 4}), base) == ([], [])
    assert codec.mismatched_settings(settings(contract={"seq_length": 16}), base) == (
        [],
        ["seq_length"],
    )
    assert codec.mismatched_settings(
        settings(contract={"writers": 4}, recorded={"teacher": "other", "seed": 1}), base
    ) == (["writers"], ["seed", "teacher"])


@pytest.mark.parametrize(
    "sft,inter_document,reset,create,expected",
    [
        (False, False, False, False, "causal"),
        (False, False, False, True, "causal"),
        (False, False, True, False, "causal"),
        (False, False, True, True, "causal_eod_reset"),
        (True, False, False, False, "isolated_segments"),
        (False, True, False, False, "isolated_segments"),
    ],
)
def test_effective_attention_mode(sft, inter_document, reset, create, expected):
    assert (
        codec.attention_mode_for_layout(
            sft=sft,
            inter_document_masking=inter_document,
            reset_attention_mask=reset,
            create_attention_mask=create,
        )
        == expected
    )
    codec.validate_metadata(settings(contract={"attention_mode": expected}))


def test_tokenizer_fingerprint_is_canonical():
    vocab = {"a": 0, "b": 1, "c": 2}
    special = {"bos": None, "eos": 2, "pad": None, "unk": None}
    first = codec.tokenizer_fingerprint(vocab, special, 3)
    assert first == codec.tokenizer_fingerprint(dict(reversed(vocab.items())), special, 3)
    assert first != codec.tokenizer_fingerprint({**vocab, "d": 3}, special, 4)
    assert first != codec.tokenizer_fingerprint(vocab, {**special, "pad": 1}, 3)

    class HF:
        bos_token_id, eos_token_id, pad_token_id, unk_token_id = None, 2, None, None

        def get_vocab(self):
            return vocab

        def __len__(self):
            return 3

    identity = codec.hf_tokenizer_identity(HF(), eod_id=2)
    assert identity == {"sha256": first, "size": 3, "special_ids": special, "eod_id": 2}
    assert codec.hf_tokenizer_identity(HF(), eod_id=0)["sha256"] == first  # EOD is not hashed.


def packed_capture(weights, *, bounds=(0, 4, 8)):
    """One packed sample of 8 tokens with explicit loss weights."""
    source = inputs([3], packed=True)
    source["cu_seqlens"] = torch.tensor([list(bounds)])
    source["position_ids"] = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])  # SFT restarts per document.
    source["loss_mask"] = torch.tensor([weights], dtype=torch.float32)
    return codec.capture_batch(source)


def test_finalize_strips_only_trailing_zero_weight_positions():
    captured = packed_capture([0, 1, 1, 0, 0, 1, 0, 0])
    targets = torch.arange(8).reshape(-1, 1)
    tail = -torch.arange(8, dtype=torch.float32)
    record, values, indices, kept_tail = codec.finalize_record(
        captured, targets.float(), targets, tail, isolated=True
    )
    # Leading prompts (positions 0 and 4) stay; trailing padding/masked tails go.
    assert record["tokens"].tolist() == [300, 301, 302, 304, 305]
    assert record["segment_lengths"].tolist() == [3, 2]
    assert record["segment_counts"].tolist() == [2]
    assert record["sample_offsets"].tolist() == [0, 5]
    # Prompt tokens keep their inputs but get no target rows.
    assert indices.reshape(-1).tolist() == [1, 2, 5]
    assert torch.equal(values, indices.float())
    assert kept_tail.tolist() == [-1, -2, -5]
    # Every token-aligned field is kept for the kept positions.
    assert record["position_ids"].tolist() == [0, 1, 2, 0, 1]
    assert record["loss_mask"].float().tolist() == [0, 1, 1, 0, 1]
    assert record["labels"].tolist() == captured["labels"][[0, 1, 2, 4, 5]].tolist()


def test_finalize_drops_fully_masked_segments_and_samples():
    captured = packed_capture([0, 0, 0, 0, 0, 1, 1, 1])
    targets = torch.arange(8).reshape(-1, 1)
    tail = torch.zeros(8)
    record, *_ = codec.finalize_record(captured, targets.float(), targets, tail, isolated=True)
    assert record["segment_lengths"].tolist() == [4]
    empty = packed_capture([0] * 8)
    record, values, *_ = codec.finalize_record(empty, targets.float(), targets, tail, isolated=True)
    assert record["sample_offsets"].tolist() == [0, 0]
    assert record["segment_counts"].tolist() == [0]
    assert values.shape == (0, 1)
    _, _, [(restored, (stored, *_))] = roundtrip(record, values, values.long(), None)
    assert restored["sample_offsets"].tolist() == [0, 0] and stored.shape == (0, 1)


def test_unpacked_samples_are_one_segment():
    source = inputs([0, 1])
    source["loss_mask"][1, 6:] = 0
    captured = codec.capture_batch(source)
    targets = torch.arange(16).reshape(-1, 1)
    record, *_ = codec.finalize_record(
        captured, targets.float(), targets, torch.zeros(16), isolated=False
    )
    assert record["sample_offsets"].tolist() == [0, 8, 14]
    assert record["segment_lengths"].tolist() == [8, 6]


@pytest.mark.parametrize("case", ["zero", "sum", "misaligned"])
def test_malformed_blocks_rejected(case):
    record = codec.finalize_record(
        packed_capture([0, 1, 1, 0, 0, 1, 0, 0]),
        torch.zeros(8, 1),
        torch.zeros(8, 1, dtype=torch.long),
        torch.zeros(8),
        isolated=True,
    )
    batch = arrow.samples_batch(*record[:3], None, vocab_size=8)
    if case == "misaligned":
        name, values = "labels", [[1, 2, 3]]
    else:
        name, values = "segments", [[5, 0]] if case == "zero" else [[3, 3]]
    column = pa.array(values, pa.large_list(pa.int32()))
    batch = batch.set_column(batch.schema.get_field_index(name), name, column)
    with pytest.raises(ValueError, match="segments|aligned"):
        arrow.decode_block(batch)


def test_files_are_self_describing_and_immutable(tmp_path):
    storage, metadata = write_cache(tmp_path)
    assert storage.list("*") == sorted(
        name
        for rank in range(2)
        for a, b in ((0, 16), (16, 32))
        for name in (storage_module.data_name(rank, a, b), storage_module.index_name(rank, a, b))
    )
    name = storage_module.FileName(1, 16, 32)
    meta = storage_module.read_meta(storage, name)
    assert codec.mismatched_settings(meta, metadata) == ([], [])
    with pytest.raises(RuntimeError, match="replace published"):
        storage_module.publish(storage, name, b"", {})
    # A data file without an index (an interrupted attempt) is invisible and replaceable.
    orphan = storage_module.FileName(0, 32, 48)
    storage.write(orphan.data, b"partial")
    assert orphan not in storage_module.list_published(storage)
    storage_module.publish(storage, orphan, b"data", {"index_version": 1})
    assert storage.read(orphan.data) == b"data"


@pytest.mark.parametrize(
    "teacher_dp,teacher_mbs,student_dp,student_mbs", [(2, 2, 3, 1), (3, 1, 2, 2), (1, 4, 4, 1)]
)
def test_resharding_and_prefix_shortening(
    tmp_path, teacher_dp, teacher_mbs, student_dp, student_mbs
):
    gbs = 12
    storage, metadata = write_cache(tmp_path, dp=teacher_dp, mbs=teacher_mbs, gbs=gbs)
    all_ids = []
    for rank in range(student_dp):
        iterator = replay.ReplayIterator(
            reader(storage, metadata),
            consumed=0,
            dp_rank=rank,
            dp_size=student_dp,
            micro_batch_size=student_mbs,
            seq_length=4,
            num_microbatches=lambda: gbs // (student_dp * student_mbs),
        )
        try:
            for _ in range(2 * gbs // (student_dp * student_mbs)):
                batch = next(iterator)
                assert torch.equal(
                    scored(batch["tokens"], batch["loss_mask"]).T,
                    batch["_kd_indices"][0].squeeze(-1),
                )
                assert batch["tokens"].shape == (student_mbs, 4)
                all_ids.extend(batch["_kd_sample_ids"])
        finally:
            iterator.close()
    assert sorted(all_ids) == list(range(24))


def test_shortening_that_removes_all_supervision_is_reported(tmp_path, capsys):
    def late_supervision(ids, packed=False):
        batch = inputs(ids, packed=packed)
        batch["loss_mask"][:, :5] = 0
        return batch

    storage, metadata = write_cache(tmp_path, iterations=2, source=late_supervision)
    for seq_length, expected in ((4, 8), (8, 0)):
        iterator = replay.ReplayIterator(
            reader(storage, metadata),
            consumed=0,
            dp_rank=0,
            dp_size=1,
            micro_batch_size=2,
            seq_length=seq_length,
            num_microbatches=lambda: 4,
        )
        try:
            for _ in range(4):
                next(iterator)
        finally:
            iterator.close()
        assert iterator.unsupervised_samples == expected
        assert capsys.readouterr().out.count("no supervised tokens") == min(expected, 1)


def test_rampup_changes_iteration_size_without_skips(tmp_path):
    storage, metadata = write_cache(tmp_path)
    count = [1]

    def make(consumed, counter):
        return replay.ReplayIterator(
            reader(storage, metadata),
            consumed=consumed,
            dp_rank=0,
            dp_size=1,
            micro_batch_size=2,
            seq_length=8,
            num_microbatches=counter,
        )

    iterator = make(0, lambda: count[0])
    assert next(iterator)["_kd_sample_ids"] == [0, 1]
    count[0] = 3
    assert [next(iterator)["_kd_sample_ids"] for _ in range(3)] == [[2, 3], [4, 5], [6, 7]]
    assert iterator.consumed == 8
    iterator.close()
    resumed = make(8, lambda: 3)
    assert next(resumed)["_kd_sample_ids"] == [8, 9]
    resumed.close()


def test_shuffle_is_deterministic_and_group_granular(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=8)
    first = reader(storage, metadata, shuffle=True, seed=7)
    second = reader(storage, metadata, shuffle=True, seed=7)
    assert first.groups == second.groups
    assert first.groups != groups(storage, metadata)
    ids = ids_of(first.samples(list(range(64)), 64))
    assert sorted(ids) == list(range(64))
    for start in range(0, 64, 16):
        assert ids[start : start + 16] == list(range(ids[start], ids[start] + 16))
    first.close()
    second.close()


def test_prefetch_stays_within_iteration_and_preserves_rampup(tmp_path):
    storage, metadata = write_cache(tmp_path)
    count = [2]
    iterator = replay.ReplayIterator(
        reader(storage, metadata),
        consumed=0,
        dp_rank=0,
        dp_size=1,
        micro_batch_size=2,
        seq_length=8,
        num_microbatches=lambda: count[0],
        prefetch=True,
    )
    try:
        assert next(iterator)["_kd_sample_ids"] == [0, 1]
        assert iterator._prefetched is not None
        assert next(iterator)["_kd_sample_ids"] == [2, 3]
        assert iterator._prefetched is None
        count[0] = 1
        assert next(iterator)["_kd_sample_ids"] == [4, 5]
        assert iterator._prefetched is None
    finally:
        iterator.close()


@pytest.mark.parametrize("missing", ["dp1__0-16.index.json", "dp0__0-16.index.json"])
def test_range_is_published_only_when_every_writer_published(tmp_path, missing):
    storage, metadata = write_cache(tmp_path)
    (tmp_path / missing).unlink()
    found, hole = replay.discover_groups(storage, metadata)
    assert found == []
    assert hole == (0, 16, 1)
    with pytest.raises(RuntimeError, match="exhausted"):
        reader(storage, metadata).samples([0], 1)


def test_hole_from_parallel_jobs_is_a_warning_until_filled(tmp_path):
    # The job owning iterations 2-4 finishes before the one owning 0-2.
    storage, metadata = write_cache(tmp_path, start=2)
    found, hole = replay.discover_groups(storage, metadata)
    assert found == [] and hole == (0, 16, 1)
    assert "parallel dump job may still be writing" in replay.describe_hole(hole)
    write_cache(tmp_path, iterations=2)
    found, hole = replay.discover_groups(storage, metadata)
    assert found == [(0, 16), (16, 32)] and hole is None
    sequential_root = tmp_path / "sequential"
    sequential, _ = write_cache(sequential_root)
    parallel_reader = reader(storage, metadata)
    sequential_reader = reader(sequential, metadata)
    try:
        for got, want in zip(
            parallel_reader.samples(list(range(32)), 32),
            sequential_reader.samples(list(range(32)), 32),
        ):
            assert got["sample_id"] == want["sample_id"]
            assert torch.equal(got["tokens"], want["tokens"])
            assert torch.equal(got["teacher_indices"], want["teacher_indices"])
    finally:
        parallel_reader.close()
        sequential_reader.close()


def test_overlapping_ranges_are_rejected(tmp_path):
    storage, metadata = write_cache(tmp_path)
    write_cache(tmp_path / "other", iterations=3, bundle=3)
    for path in list((tmp_path / "other").iterdir()):
        path.rename(tmp_path / path.name)
    with pytest.raises(ValueError, match="Overlapping"):
        replay.discover_groups(storage, metadata)


def test_sequential_reader_extends_with_new_groups(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2)
    live = reader(storage, metadata)
    assert ids_of(live.samples([0], 1)) == [0]
    write_cache(tmp_path, iterations=4)
    with pytest.raises(RuntimeError, match="exhausted at 16 samples; requested 17"):
        live.ensure_available(17)
    live.extend(groups(storage, metadata)[1:])
    assert ids_of(live.samples([16], 17)) == [16]
    assert live.total_samples == 32
    live.close()


def test_shuffled_reader_never_extends(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2)
    shuffled = reader(storage, metadata, shuffle=True)
    with pytest.raises(RuntimeError, match="cannot extend"):
        shuffled.extend([(16, 32)])
    shuffled.close()


class _CountingStorage(Storage):
    """Local storage that records every ranged read."""

    def __init__(self, root):
        super().__init__(str(root))
        self.reads = []

    def read_range(self, name, offset, size):
        self.reads.append((name, offset, size))
        return super().read_range(name, offset, size)


@pytest.mark.parametrize("targets", [True, False])
def test_each_block_is_one_ranged_read(tmp_path, targets):
    write_cache(tmp_path, logprobs=True)
    storage = _CountingStorage(tmp_path)
    metadata = storage_module.read_meta(storage, storage_module.FileName(0, 0, 16))
    live = reader(storage, metadata, targets=targets)
    samples = live.samples(list(range(32)), 32)
    live.close()
    assert ids_of(samples) == list(range(32))
    assert ("teacher_values" in samples[0]) == targets
    expected = []
    for start, end in ((0, 16), (16, 32)):
        for rank in range(2):
            name = storage_module.FileName(rank, start, end)
            index = storage_module.read_index(storage, name)
            expected += [
                (name.data, *arrow.block_range(index, block, inputs_only=not targets))
                for block in range(len(index["blocks"]))
            ]
    assert sorted(storage.reads) == sorted(expected)


def test_files_differing_in_recorded_settings_warn_and_in_required_ones_fail(tmp_path, capsys):
    storage, metadata = write_cache(tmp_path, iterations=2)
    write_cache(tmp_path, iterations=4, contract={"seq_length": 16, "teacher": "other"})
    write_cache(tmp_path, iterations=6, contract={"prefix_stable": False})
    live = reader(storage, metadata)
    try:
        assert ids_of(live.samples(list(range(32)), 32)) == list(range(32))
        assert (
            "recorded settings (seq_length, teacher); it is still used" in capsys.readouterr().out
        )
        with pytest.raises(ValueError, match="prefix_stable"):
            live.samples([32], 33)  # Rejected before any of its records is used.
    finally:
        live.close()


def test_index_of_another_file_version_is_rejected(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2)
    # A retry rewrote the data file, but its index is still the earlier one.
    (tmp_path / "dp0__0-16.arrow").write_bytes((tmp_path / "dp0__0-16.arrow").read_bytes() + b"x")
    live = reader(storage, metadata)
    with pytest.raises(ValueError, match="another version of the file"):
        live.samples([0], 1)
    live.close()


def test_packed_samples_must_fit_the_student_length(tmp_path):
    storage, metadata = write_cache(tmp_path, packed=True)
    live = reader(storage, metadata)
    samples = live.samples([0, 1], 2)
    live.close()
    with pytest.raises(ValueError, match="student sequence length"):
        replay.collate_samples(samples, 4, packed=True)
    result = replay.collate_samples(samples, 8, packed=True)
    assert result["cu_seqlens"].tolist() == [[0, 4, 8], [0, 4, 8]]


def tail_masked(ids, packed=False):
    """Packed samples whose documents end with a zero-weight position (padding-like)."""
    batch = inputs(ids, packed=packed)
    batch["position_ids"] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).expand(len(ids), -1)
    batch["loss_mask"][:, [3, 7]] = 0
    return batch


@pytest.mark.parametrize(
    "divisor,length,bounds,tokens",
    [
        (1, 8, [0, 3, 8], [0, 1, 2, 4, 5, 6, None, None]),
        (2, 8, [0, 4, 8], [0, 1, 2, None, 4, 5, 6, None]),
        (2, 12, [0, 4, 12], [0, 1, 2, None, 4, 5, 6, None, None, None, None, None]),
    ],
)
def test_stripped_segments_are_repadded_for_the_student(tmp_path, divisor, length, bounds, tokens):
    storage, metadata = write_cache(
        tmp_path, packed=True, contract={"padding_divisor": 2}, source=tail_masked
    )
    live = reader(storage, metadata)
    sample = live.samples([0], 1)[0]
    live.close()
    assert sample["segments"].tolist() == [3, 3]  # Stored without padding.
    batch = replay.collate_samples([sample], length, packed=True, divisor=divisor)
    assert batch["cu_seqlens"].tolist() == [bounds]
    expected = [0 if t is None else t for t in tokens]
    assert batch["tokens"][0].tolist() == expected
    padding = torch.tensor([t is None for t in tokens])
    assert (batch["loss_mask"][0][padding] == 0).all()
    values = batch["_kd_values"][0].reshape(-1)
    target = batch["loss_mask"][0] != 0  # Padding and zero-weight positions have no rows.
    assert (values[~target] == -1000).all()
    assert (values[target] == torch.tensor(expected)[target]).all()
    # Positions restart per segment and continue through each segment's padding.
    starts = torch.tensor(bounds[:-1])
    for a, b in zip(bounds[:-1], bounds[1:]):
        assert batch["position_ids"][0][a:b].tolist() == list(range(b - a))
    assert int(starts[-1]) < length


def test_missing_teacher_entries_preserve_scoring_mask():
    sample = {
        "tokens": torch.tensor([1, 2, 3], dtype=torch.int32),
        "labels": torch.tensor([2, 3, 4], dtype=torch.int32),
        "position_ids": torch.arange(3, dtype=torch.int32),
        "loss_mask": torch.ones(3, dtype=torch.bool),
        "segments": torch.tensor([3]),
        "sample_id": 0,
        "teacher_values": torch.tensor([[-1.0], [-1000.0], [-2.0]]),
        "teacher_indices": torch.tensor([[1], [0], [2]]),
        "teacher_tail": torch.zeros(3),
    }
    batch = replay.collate_samples([sample], 4)
    assert batch["loss_mask"][0].tolist() == [1.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("packed", [False, True])
def test_collation_maps_targets_for_every_cp_rank(tmp_path, packed):
    storage, metadata = write_cache(tmp_path, packed=packed)
    live = reader(storage, metadata)
    samples = live.samples([0, 1], 2)
    live.close()
    result = replay.collate_samples(samples, 8, cp_size=2)
    captured = codec.capture_batch(
        {
            key: result[key]
            for key in ("tokens", "labels", "position_ids", "loss_mask", "cu_seqlens")
            if result[key] is not None
        }
    )
    for rank in range(2):
        mapping = replay.token_map(captured, rank, 2)
        assert torch.equal(
            result["_kd_indices"][rank].squeeze(-1),
            scored(captured["tokens"], captured["loss_mask"])[mapping],
        )


@pytest.mark.parametrize(
    "packed,layout",
    [(False, "zigzag"), (False, "contiguous"), (True, "zigzag"), (True, "contiguous")],
)
def test_cp_token_maps_cover_every_token_and_match_targets(packed, layout):
    captured = codec.capture_batch(inputs([0, 1], length=16, packed=packed))
    values = captured["tokens"].float().reshape(-1, 1)
    indices = captured["tokens"].long().reshape(-1, 1)
    maps = [replay.token_map(captured, rank, 4, layout=layout) for rank in range(4)]
    positions = torch.cat([m.reshape(-1) for m in maps])
    assert sorted(positions.tolist()) == list(range(32))
    for mapping in maps:
        tail = -captured["tokens"].float()
        local_values, local_indices, local_tail = replay.map_targets(values, indices, tail, mapping)
        assert torch.equal(local_values.long(), local_indices)
        assert torch.equal(local_tail, -local_values.squeeze(-1))
        assert torch.equal(local_indices.squeeze(-1), captured["tokens"][mapping].long())


def test_packed_cp_padding_validation():
    captured = codec.capture_batch(inputs([0], length=8, packed=True))
    captured["cu_seqlens"] = [torch.tensor([0, 2, 8])]
    with pytest.raises(ValueError, match="padding"):
        replay.token_map(captured, 0, 2)


@pytest.mark.parametrize(
    "packed,layout",
    [(False, "zigzag"), (True, "zigzag"), (False, "contiguous"), (True, "contiguous")],
)
def test_token_maps_match_megatron_cpu_partitioning(packed, layout, monkeypatch):
    """Compare against the actual entrypoint flattening and CP helpers."""
    root = Path(_package.__path__[0]).parents[1]
    namespace = {"torch": torch}
    names = {
        "_merge_cu_seqlens_across_micro_batch",
        "flatten_batch_for_packed_sequences",
        "_get_batch_on_this_cp_rank_per_sequence_balancing",
        "_get_batch_on_this_cp_rank_contiguous",
    }
    for path in (root / "core/utils.py", root / "core/context_parallel/utils.py"):
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        future = ast.parse("from __future__ import annotations").body
        exec(
            compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), namespace
        )
    source = inputs([0, 1, 2], length=16, packed=packed)
    source["tokens"] = torch.arange(48).reshape(3, 16)
    captured = codec.capture_batch(source)
    full = namespace["flatten_batch_for_packed_sequences"](dict(source))
    helper = namespace[
        (
            "_get_batch_on_this_cp_rank_per_sequence_balancing"
            if layout == "zigzag"
            else "_get_batch_on_this_cp_rank_contiguous"
        )
    ]
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    for rank in range(4):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
        actual = helper(dict(full), None)["tokens"].T.contiguous()
        mapping = replay.token_map(captured, rank, 4, layout=layout, per_sequence=True)
        assert torch.equal(mapping, actual)


def test_discard_unpublished_only_touches_its_own_range(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=6)
    (tmp_path / "dp1__16-32.index.json").unlink()
    (tmp_path / "dp0__16-32.arrow.0123456789abcdef0123456789abcdef.tmp").write_bytes(b"x")
    removed = storage_module.discard_unpublished(storage, 16, 32)
    # Indexes go first, so an interrupted cleanup leaves only unpublished data files.
    assert removed == [
        "dp0__16-32.index.json",
        "dp0__16-32.arrow",
        "dp0__16-32.arrow.0123456789abcdef0123456789abcdef.tmp",
        "dp1__16-32.arrow",
    ]
    # Another job's range (32-48) is untouched.
    assert len(storage.list("dp*__32-48.*")) == 4


@pytest.fixture
def runtime_environment(monkeypatch):
    """Provide CPU process-group boundaries, retaining actual codec/replay code."""
    args = types.SimpleNamespace(
        logits_load_inputs=True,
        logits_save_inputs=False,
        logits_save_dir=None,
        logits_load_dir=None,
        logits_load_shuffle_shards=7,
        logits_load_decode_threads=2,
        logits_load_msc_prefetch_depth=2,
        logits_load_replay_end=None,
        seq_length=8,
        context_parallel_size=1,
        consumed_train_samples=0,
        sft=False,
        dataloader_inter_document_masking=False,
        reset_attention_mask=False,
        create_attention_mask_in_dataloader=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        micro_batch_size=2,
        global_batch_size=8,
        logits_load_kd_loss_alpha=0.9,
        padded_vocab_size=1 << 17,
        save="/progress",
        save_interval=2,
        exit_interval=None,
        train_iters=128,
        train_samples=None,
        freeze_all_layers=True,
        load="original-weights",
        override_ckpt_iteration=None,
        iterations_to_skip=[],
        seed=1234,
        split="99,1,0",
        params_dtype=torch.bfloat16,
    )
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: args
    training.get_tensorboard_writer = lambda: None
    training.get_tokenizer = lambda: types.SimpleNamespace(
        vocab={"a": 0, "b": 1, "c": 2}, vocab_size=3, eod=2, pad=-1, bos=None, eos=None
    )
    core = types.ModuleType("megatron.core")
    core.parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_src_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_world_size=lambda: 2,
        is_pipeline_first_stage=lambda: True,
        is_pipeline_last_stage=lambda: True,
    )
    calculator = types.ModuleType("megatron.core.num_microbatches_calculator")
    calculator.get_num_microbatches = lambda: 2
    calculator.get_current_global_batch_size = lambda: args.global_batch_size
    for name, module in (
        ("megatron", types.ModuleType("megatron")),
        ("megatron.training", training),
        ("megatron.core", core),
        ("megatron.core.num_microbatches_calculator", calculator),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    legacy_saver = types.ModuleType(f"{_namespace}.logits_saver")
    legacy_saver.LogitsSaverHooks = object
    monkeypatch.setitem(sys.modules, legacy_saver.__name__, legacy_saver)
    utils = types.ModuleType(f"{_namespace}.utils")
    utils._broadcast_without_pp = lambda factory: factory()
    utils._blend_identifiers = lambda args: {"kind": "blend", "blend": [[1.0, "corpus"]]}
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    runtime = importlib.import_module(f"{_namespace}.v3_runtime")
    runtime._PLAN = None
    yield args, runtime, legacy_saver
    if runtime._PLAN is not None:
        for live in runtime._PLAN["readers"]:
            live.close()
    runtime._PLAN = None


@pytest.mark.parametrize("seed", [None, 0, 7, -5])
def test_single_shuffle_argument_parses_enablement_and_seed(seed):
    path = Path(_package.__path__[0]).parent / "arguments.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_add_logits_distillation_args"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    parser = namespace["_add_logits_distillation_args"](argparse.ArgumentParser())
    argv = [] if seed is None else ["--logits-load-shuffle-shards", str(seed)]
    args = parser.parse_args(argv)
    assert args.logits_load_shuffle_shards == seed
    with pytest.raises(SystemExit):
        parser.parse_args(["--logits-load-shuffle-shards"])


@pytest.mark.parametrize("seed", [0, 7])
def test_shuffle_seed_requires_replay(runtime_environment, seed):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = seed
    args.logits_load_inputs = False
    with pytest.raises(ValueError, match="requires --logits-load-inputs"):
        runtime.validate_options(args)


def dump_args(args):
    args.logits_load_inputs = False
    args.logits_load_shuffle_shards = None
    args.logits_save_inputs = True
    args.logits_save_dir = "/dump"
    args.sequence_packing_scheduler = args.hybrid_context_parallel = args.use_varlen_dataset = False
    args.mtp_num_layers = None
    return args


@pytest.mark.parametrize(
    "change,message",
    [
        ({"iterations_to_skip": [5]}, "iterations-to-skip"),
        ({"allow_ambiguous_pad_tokens": True}, "ambiguous-pad"),
        ({"save_interval": None}, "save-interval"),
        ({"exit_interval": 3}, "exit-interval to be a multiple"),
        ({"override_ckpt_iteration": 3}, "override-ckpt-iteration to be a multiple"),
        ({"hidden_dropout": 0.1}, "deterministically; disable --hidden-dropout"),
        ({"attention_dropout": 0.1}, "--attention-dropout"),
        ({"moe_input_jitter_eps": 0.01}, "--moe-input-jitter-eps"),
        ({"moe_expert_capacity_factor": 1.0}, "--moe-expert-capacity-factor"),
        ({"moe_token_dropping": True}, "--moe-token-dropping"),
    ],
)
def test_dump_validation_rejects_unsafe_options(runtime_environment, change, message):
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.exit_interval = 4
    args.override_ckpt_iteration = 4
    runtime.validate_options(args)
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        runtime.validate_options(args)


corpus = importlib.import_module(f"{_namespace}.corpus_runtime")


def corpus_args(args, tmp_path, name="web"):
    dump_args(args)
    args.logits_save_dir = str(tmp_path)
    args.kd_corpus_dump, args.kd_corpus_split = name, "train"
    args.data_path, args.mock_data, args.eval_iters, args.train_samples = (
        ["web_text"],
        False,
        0,
        None,
    )
    return args


def test_corpus_dump_validation_sets_its_folder_and_one_epoch_budget(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    corpus_args(args, tmp_path)
    runtime.validate_options(args)
    assert args.logits_save_inputs and args.kd_corpus_root == str(tmp_path)
    assert args.logits_save_dir == str(tmp_path / "web" / "train")
    assert args.train_iters * args.global_batch_size >= 1 << 39  # Until the epoch is known.
    runtime.validate_options(args)  # Idempotent.
    assert args.logits_save_dir == str(tmp_path / "web" / "train")


@pytest.mark.parametrize(
    "change,message",
    [
        ({"freeze_all_layers": False}, "frozen teacher"),
        ({"data_path": ["0.5", "a", "0.5", "b"]}, "exactly one unweighted"),
        ({"eval_iters": 10}, "--eval-iters 0"),
        ({"train_samples": 100}, "omit --train-samples"),
        ({"kd_corpus_dump": "a/b"}, "plain directory name"),
        ({"valid_data_path": ["v"]}, "valid-data-path"),
    ],
)
def test_corpus_dump_validation_rejects_other_setups(
    tmp_path, runtime_environment, change, message
):
    args, runtime, _ = runtime_environment
    corpus_args(args, tmp_path)
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        runtime.validate_options(args)


def test_corpus_epoch_is_one_seeded_permutation_padded_to_whole_batches():
    class Rows:
        """An SFT-like dataset: built with num_samples=None, its length is its rows."""

        num_samples = None
        indices = np.arange(10)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, index):
            return {"row": index}

    epoch = corpus.CorpusEpoch(Rows(), seed=3, global_batch_size=4)
    assert (epoch.num_samples, len(epoch)) == (10, 12)
    rows = [epoch[i]["row"] for i in range(12)]
    assert sorted(rows[:10]) == list(range(10)) and rows[10:] == rows[:2]  # Filler repeats.
    again = corpus.CorpusEpoch(Rows(), seed=3, global_batch_size=4)
    assert [again[i]["row"] for i in range(10)] == rows[:10]
    other = corpus.CorpusEpoch(Rows(), seed=4, global_batch_size=4)
    assert [other[i]["row"] for i in range(10)] != rows[:10]


def test_corpus_epoch_length_sizes_the_run(runtime_environment):
    args, _, _ = runtime_environment
    args.kd_corpus_dump, args.kd_corpus_split, args.kd_corpus_samples = "web", "train", 17
    corpus.agree_corpus_epoch(args)
    assert args.train_iters == 3  # 17 samples in global batches of 8.


def test_corpus_metadata_requires_the_epoch_size(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    corpus_args(args, tmp_path)
    runtime.validate_options(args)
    args.kd_corpus_samples = 20
    metadata = runtime._dump_settings(teacher_saver(runtime, tmp_path))
    assert metadata["contract"]["num_samples"] == 20 and "num_samples" not in metadata["recorded"]
    assert metadata["recorded"]["split"] == "99,1,0:train"
    metadata["contract"]["first_sample"] = 0
    codec.validate_metadata(metadata, corpus=True)
    del metadata["contract"]["num_samples"]
    with pytest.raises(ValueError, match="num_samples"):
        codec.validate_metadata(metadata, corpus=True)


def test_corpus_sources_must_share_a_tokenizer(tmp_path, runtime_environment, capsys):
    args, runtime, _ = runtime_environment
    corpus_args(args, tmp_path, name="code")
    runtime.validate_options(args)
    args.kd_corpus_samples = 32
    metadata = runtime._dump_settings(teacher_saver(runtime, tmp_path))
    metadata["contract"]["first_sample"] = 0
    other = json.loads(json.dumps(metadata))
    other["contract"]["num_samples"] = 64  # Sources differ in size and data, but share a teacher.
    other["recorded"]["teacher"] = "/other@1"
    write_cache(tmp_path / "web" / "train", metadata=other, iterations=2)
    saver = teacher_saver(runtime, tmp_path / "code" / "train")
    runtime.initialize_dump_metadata(saver)
    assert "recorded settings (teacher)" in capsys.readouterr().out
    other["contract"]["tokenizer"] = {**other["contract"]["tokenizer"], "sha256": "different"}
    write_cache(tmp_path / "math" / "train", metadata=other, iterations=2)
    with pytest.raises(RuntimeError, match="math/train uses a different tokenizer"):
        runtime.initialize_dump_metadata(saver)


def test_storage_lists_nested_names_relative_to_its_root(tmp_path):
    write_cache(tmp_path / "web" / "train", iterations=2)
    assert Storage(str(tmp_path)).list("*/*/dp0__*.index.json") == [
        "web/train/dp0__0-16.index.json"
    ]
    assert Storage(str(tmp_path / "web" / "train")).list("dp0__*.index.json") == [
        "dp0__0-16.index.json"
    ]


blend = importlib.import_module(f"{_namespace}.format.blend")


def corpus_source(root, name, *, samples, iterations, base=0, contract=None):
    """Write a corpus source whose sample ``i`` has tokens (base + i) * 100 + position."""
    metadata = settings(contract=contract)
    metadata["recorded"].pop("num_samples")
    metadata["contract"]["num_samples"] = samples  # Required for corpus sources.
    write_cache(
        root / name / "train",
        iterations=iterations,
        metadata=metadata,
        source=lambda ids, packed=False: inputs([base + i for i in ids], packed=packed),
    )
    return metadata


def blend_config(root, **overrides):
    return {"corpus": str(root), "split": "train", "seed": 1, "lanes": 4, "allow_partial": []} | {
        "sources": {"web": 3, "math": 1},
        **overrides,
    }


def two_sources(root, **contract):
    corpus_source(root, "web", samples=32, iterations=4, contract=contract)
    corpus_source(root, "math", samples=16, iterations=2, base=500, contract=contract)


def test_blend_plan_is_deterministic_and_follows_the_weights(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=38)
    assert (manifest["lanes"], manifest["lane_samples"]) == (4, 10)  # Whole lanes >= budget.
    assert {s["name"]: s["samples"] for s in manifest["sources"]} == {"web": 30, "math": 10}
    assert all(s["passes"] == 1 for s in manifest["sources"])
    segments = manifest["segments"]
    lengths = np.bincount(segments[:, 0], weights=segments[:, 4] - segments[:, 3])
    assert lengths.tolist() == [10, 10, 10, 10]
    assert blend.plan(blend_config(tmp_path), budget=38)["hash"] == manifest["hash"]
    assert blend.plan(blend_config(tmp_path, seed=2), budget=38)["hash"] != manifest["hash"]
    assert "web: weight 3, 30 samples (75.0%" in blend.describe(manifest)


def test_blend_upsamples_with_whole_passes(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path, sources={"web": 1, "math": 1}), budget=64)
    math = next(s for s in manifest["sources"] if s["name"] == "math")
    assert (math["samples"], math["available"], math["passes"]) == (32, 16, 2)


def test_blend_excludes_filler_and_refuses_incomplete_sources(tmp_path):
    corpus_source(tmp_path, "web", samples=13, iterations=2)  # 16 positions, 3 filler.
    corpus_source(tmp_path, "math", samples=32, iterations=2, base=500)  # Half dumped.
    with pytest.raises(ValueError, match="math covers samples 0-16 of 32"):
        blend.plan(blend_config(tmp_path), budget=8)
    manifest = blend.plan(blend_config(tmp_path, allow_partial=["math"]), budget=8)
    available = {s["name"]: s["available"] for s in manifest["sources"]}
    assert available == {"web": 13, "math": 16}


def test_blend_sources_must_agree_on_required_settings(tmp_path):
    corpus_source(tmp_path, "web", samples=32, iterations=4)
    corpus_source(tmp_path, "math", samples=16, iterations=2, contract={"prefix_stable": False})
    with pytest.raises(ValueError, match="differs on prefix_stable"):
        blend.plan(blend_config(tmp_path), budget=8)


def test_corpus_reader_serves_the_planned_samples(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=40)
    live = replay.CorpusReader(str(tmp_path), manifest)
    try:
        samples = live.samples(list(range(40)), 40)
        with pytest.raises(RuntimeError, match="holds 40 samples"):
            live.samples([40], 41)
    finally:
        live.close()
    origin = [int(sample["tokens"][0]) // 100 for sample in samples]
    web = [o for o in origin if o < 500]
    math = [o - 500 for o in origin if o >= 500]
    assert len(web) == 30 and len(set(web)) == 30 and max(web) < 32
    assert len(math) == 10 and len(set(math)) == 10 and max(math) < 16
    for sample in samples:  # Identity targets: each scored token's target is the token.
        assert torch.equal(
            sample["teacher_indices"].reshape(-1), sample["tokens"][sample["loss_mask"] != 0].long()
        )
    # Sources are spread along the stream, so every lane mixes them.
    lane_origins = [{origin[p] >= 500 for p in range(lane, 40, 4)} for lane in range(4)]
    assert sum(len(found) == 2 for found in lane_origins) >= 2


def test_corpus_reader_only_opens_the_lanes_of_its_rank(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=40)
    live = replay.CorpusReader(str(tmp_path), manifest, targets=False)
    try:
        # DP rank 1 of 2 with MBS 2 reads positions 2, 3, 6, 7, ...: lanes 2 and 3 only.
        positions = [p for p in range(40) if (p % 4) // 2 == 1]
        assert len(live.samples(positions, 40)) == 20
        assert sorted(live.lanes) == [2, 3]
    finally:
        live.close()


def test_corpus_reader_rejects_files_changed_since_planning(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=40)
    manifest["files"][0]["size"] += 1
    live = replay.CorpusReader(str(tmp_path), manifest)
    file_lanes = {int(lane) for lane, file_id, *_ in manifest["segments"] if file_id == 0}
    lane = min(file_lanes)
    try:
        with pytest.raises(ValueError, match="changed since the blend was planned"):
            live.samples(list(range(lane, 40, 4)), 40)
    finally:
        live.close()


def test_student_trains_on_a_corpus_blend_and_resumes_only_on_the_same_plan(
    tmp_path, runtime_environment
):
    args, runtime, _ = runtime_environment
    two_sources(tmp_path, tokenizer=runtime.tokenizer_identity())
    config = tmp_path / "blend.json"
    config.write_text(json.dumps(blend_config(tmp_path)))
    args.logits_load_inputs, args.logits_load_dir, args.logits_load_shuffle_shards = (
        False,
        None,
        None,
    )
    args.kd_corpus_blend = str(config)
    args.sequence_packing_scheduler = args.hybrid_context_parallel = args.use_varlen_dataset = False
    args.mtp_num_layers, args.dataloader_type, args.train_iters = None, "single", 5
    runtime.validate_options(args)
    assert args.logits_load_inputs and args.logits_load_dir == str(tmp_path)
    runtime.initialize_replay(args)
    assert runtime._PLAN["corpus"] and runtime._PLAN["available"] == 40
    planned = args.kd_corpus_manifest_hash
    iterator = build_loader_iterator(runtime, args, 0)
    try:
        batch = next(iterator)
        assert batch["tokens"].shape == (2, 8) and batch["_kd_sample_ids"] == [0, 1]
    finally:
        iterator.close()
    runtime.ensure_replay_frontier(args)  # 0 + 8 <= 40, no listing.
    args.consumed_train_samples = 40
    with pytest.raises(RuntimeError, match="exhausted at 40"):
        runtime.ensure_replay_frontier(args)
    # A resume plans the same schedule; a changed config is refused.
    runtime._PLAN, args.consumed_train_samples = None, 8
    runtime.initialize_replay(args)
    assert args.kd_corpus_manifest_hash == planned
    runtime._PLAN = None
    config.write_text(json.dumps(blend_config(tmp_path, seed=9)))
    with pytest.raises(RuntimeError, match="no longer plans the schedule"):
        runtime.initialize_replay(args)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"logits_load_shuffle_shards": 3}, "ordered by its plan"),
        ({"logits_save_inputs": True}, "both dump and load"),
    ],
)
def test_corpus_blend_validation(tmp_path, runtime_environment, change, message):
    args, runtime, _ = runtime_environment
    config = tmp_path / "blend.json"
    config.write_text(json.dumps(blend_config(tmp_path)))
    args.kd_corpus_blend = str(config)
    args.logits_load_shuffle_shards = None
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        runtime.validate_options(args)


hf = importlib.import_module(f"{_namespace}.format.hf")


def test_hf_stream_yields_a_ranks_samples_in_megatron_order(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=40)
    stream = hf.CorpusStream(str(tmp_path), manifest, rank=1, world_size=2, micro_batch_size=2)
    ids = [sample["sample_id"] for sample in stream]
    assert ids == [p for p in range(40) if (p % 4) // 2 == 1]
    resumed = hf.CorpusStream(
        str(tmp_path), manifest, rank=1, world_size=2, micro_batch_size=2, start=20
    )
    assert [sample["sample_id"] for sample in resumed] == [p for p in ids if p >= 20]


def test_hf_collate_aligns_labels_and_targets(tmp_path):
    two_sources(tmp_path)
    manifest = blend.plan(blend_config(tmp_path), budget=40)
    samples = list(hf.CorpusStream(str(tmp_path), manifest, micro_batch_size=2))[:3]
    short = dict(samples[1])  # A shorter sample: its first 5 tokens and their target rows.
    rows = int((short["loss_mask"][:5] != 0).sum())
    for key in codec.TOKEN_FIELDS:
        short[key] = short[key][:5]
    for key in ("teacher_values", "teacher_indices", "teacher_tail"):
        short[key] = short[key][:rows]
    short["segments"] = torch.tensor([5])
    samples[1] = short
    batch = hf.collate(samples, pad_token_id=7)
    assert batch["input_ids"].shape == (3, 8) and batch["input_ids"][1, 5:].tolist() == [7] * 3
    assert batch["attention_mask"][1].tolist() == [1] * 5 + [0] * 3
    scored = batch["loss_weights"] != 0
    assert (batch["labels"][~scored] == -100).all()
    # Identity targets: the scored rows' teacher index is the token at that position.
    assert torch.equal(batch["teacher_indices"][..., 0][scored], batch["input_ids"][scored])
    assert (batch["teacher_values"][~scored] == codec.LOGPROB_SENTINEL).all()


def test_hf_topk_kl_matches_megatron(monkeypatch):
    cached = _loss_namespace(monkeypatch)
    torch.manual_seed(0)
    batch, length, vocab, k = 2, 5, 11, 3
    logits = torch.randn(batch, length, vocab)
    teacher = torch.randn(batch, length, vocab).log_softmax(-1)
    values, indices = teacher.topk(k, dim=-1)
    values[0, 1, 2] = codec.LOGPROB_SENTINEL  # A top-P masked entry.
    tail = codec.tail_logprob(values.reshape(-1, k)).reshape(batch, length)
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.5], [1.0, 0.0, 1.0, 1.0, 1.0]])
    reference = cached.topk_kl_div(
        logits.transpose(0, 1).clone(),
        values.transpose(0, 1),
        indices.transpose(0, 1),
        1,
        0,
        None,
        teacher_tail_logprobs=tail.transpose(0, 1),
    )
    expected = (reference * weights).sum() / weights.sum()
    assert torch.allclose(hf.topk_kl(logits, values, indices, tail, weights), expected, atol=1e-6)


def test_unfrozen_teacher_may_use_dropout(runtime_environment):
    # An unfrozen teacher's dump is a record of its training run, dropout included.
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.freeze_all_layers = False
    args.hidden_dropout = args.attention_dropout = 0.1
    runtime.validate_options(args)


@pytest.mark.parametrize(
    "fp8,fp4,expected",
    [
        (None, None, "bf16"),
        ("e4m3", None, "bf16+fp8:e4m3/delayed"),
        (None, "e2m1", "bf16+fp4:e2m1/nvfp4"),
    ],
)
def test_teacher_identity_and_precision_are_recorded(
    tmp_path, runtime_environment, fp8, fp4, expected
):
    args, runtime, _ = runtime_environment
    args.fp8, args.fp8_recipe, args.fp4, args.fp4_recipe = fp8, "delayed", fp4, "nvfp4"
    args.load = str(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2000\n")
    metadata = runtime._dump_settings(teacher_saver(runtime, tmp_path))
    assert metadata["provenance"]["teacher_precision"] == expected
    assert metadata["recorded"]["teacher"] == f"{tmp_path}@2000"
    assert metadata["recorded"]["num_samples"] == args.train_iters * args.global_batch_size


def test_format_contract_imports_without_megatron():
    import subprocess

    path = Path(_package.__path__[0]) / "format" / "contract.py"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('contract', {str(path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert not any(name == 'megatron' or name.startswith('megatron.') for name in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def replay_cache(tmp_path, args, runtime, *, iterations=4):
    """Create a runtime-compatible cache."""
    contract = {"tokenizer": runtime.tokenizer_identity()}
    storage, metadata = write_cache(tmp_path, iterations=iterations, contract=contract)
    args.logits_load_dir = str(tmp_path)
    return storage, metadata


def build_loader_iterator(runtime, args, consumed):
    loader = runtime.build_replay_loader(args, consumed=consumed)
    return iter(loader.dataset)


def test_loader_enables_shuffling_with_seed_zero(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = 0
    args.train_iters = 8  # Exactly the 64 cached samples: one pass.
    storage, metadata = replay_cache(tmp_path, args, runtime, iterations=8)
    runtime.initialize_replay(args)
    iterator = build_loader_iterator(runtime, args, 0)
    expected = reader(storage, metadata, shuffle=True, seed=0)
    try:
        assert iterator.reader.groups == expected.groups
        assert iterator.reader.groups != groups(storage, metadata)
        batch = next(iterator)
        assert batch["_kd_sample_ids"] == ids_of(expected.samples([0, 1], 4))
        assert torch.is_tensor(batch["_kd_values"])  # This CP rank's shard only.
    finally:
        iterator.close()
        expected.close()


def test_shuffled_resume_reuses_the_checkpointed_extent(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    storage, metadata = replay_cache(tmp_path, args, runtime)
    before = set(storage.list("*"))
    runtime.initialize_replay(args)
    assert args.logits_load_replay_end == 32
    expected_reader = reader(storage, metadata, shuffle=True, seed=7)
    expected = ids_of(expected_reader.samples([8, 9], 10))
    expected_reader.close()
    assert set(storage.list("*")) == before  # Students need only read access.
    # The teacher keeps publishing; a resume must still shuffle the original extent.
    replay_cache(tmp_path, args, runtime, iterations=8)
    args.consumed_train_samples = 8
    runtime._PLAN = None
    runtime.initialize_replay(args)
    assert runtime._PLAN["groups"] == [(0, 16), (16, 32)]
    iterator = build_loader_iterator(runtime, args, args.consumed_train_samples)
    try:
        assert next(iterator)["_kd_sample_ids"] == expected
    finally:
        iterator.close()


def test_shuffled_resume_requires_the_planned_extent(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    replay_cache(tmp_path, args, runtime)
    args.logits_load_replay_end = 48
    with pytest.raises(RuntimeError, match="planned over samples up to 48"):
        runtime.initialize_replay(args)


def test_shuffled_replay_repeats_whole_passes(tmp_path, runtime_environment, monkeypatch):
    args, runtime, _ = runtime_environment
    args.train_iters = 10  # 80 samples from a 32-sample cache: three passes.
    storage, metadata = replay_cache(tmp_path, args, runtime)
    runtime.initialize_replay(args)
    assert runtime._PLAN["passes"] == 3 and runtime._PLAN["available"] == 96
    single = reader(storage, metadata, shuffle=True, seed=7)
    full = reader(storage, metadata, shuffle=True, seed=7, passes=3)
    args.consumed_train_samples = 32  # The second pass starts here.
    iterator = build_loader_iterator(runtime, args, args.consumed_train_samples)
    try:
        assert iterator.reader.groups == full.groups
        passes = [full.groups[i : i + 2] for i in (0, 2, 4)]
        assert passes[0] == single.groups  # The first pass is the single-pass order.
        assert all(sorted(order) == groups(storage, metadata) for order in passes)
        assert next(iterator)["_kd_sample_ids"] == ids_of(full.samples([32, 33], 34))
    finally:
        iterator.close()
        single.close()
        full.close()
    # A fixed extent never lists again, and running past the last pass fails on every rank.
    monkeypatch.setattr(runtime, "discover_groups", lambda *a: pytest.fail("listed"))
    args.consumed_train_samples = 88
    runtime.ensure_replay_frontier(args)
    args.consumed_train_samples = 96
    with pytest.raises(RuntimeError, match="exhausted at 96 samples; requested 104"):
        runtime.ensure_replay_frontier(args)


def test_repeat_budget_growth_only_appends_passes(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    storage, metadata = replay_cache(tmp_path, args, runtime, iterations=8)
    short = reader(storage, metadata, shuffle=True, seed=7, passes=2)
    long = reader(storage, metadata, shuffle=True, seed=7, passes=5)
    try:
        assert long.groups[: len(short.groups)] == short.groups
        assert long.total_samples == 5 * 64
    finally:
        short.close()
        long.close()
    # A resumed cursor beyond the budget still gets enough passes.
    args.train_iters = 1
    args.consumed_train_samples = 100
    runtime.initialize_replay(args)
    assert runtime._PLAN["passes"] == 2


def test_sequential_replay_never_repeats(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    replay_cache(tmp_path, args, runtime)
    runtime.initialize_replay(args)  # Budget 1024 > 32 published samples.
    assert runtime._PLAN["passes"] == 1 and runtime._PLAN["available"] == 32


def test_sequential_resume_sees_new_groups(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    storage, metadata = replay_cache(tmp_path, args, runtime)
    runtime.initialize_replay(args)
    args.consumed_train_samples = 16
    replay_cache(tmp_path, args, runtime, iterations=6)
    runtime._PLAN = None
    runtime.initialize_replay(args)
    assert runtime._PLAN["available"] == 48
    assert args.logits_load_replay_end is None


def test_resume_validates_sample_cursor(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None  # A shuffled plan would repeat to cover it.
    replay_cache(tmp_path, args, runtime)
    args.consumed_train_samples = 40
    with pytest.raises(RuntimeError, match="cursor 40 exceeds"):
        runtime.initialize_replay(args)


def test_student_rejects_incompatible_cache(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    replay_cache(tmp_path, args, runtime)
    args.padded_vocab_size = 2  # Smaller than the 3-token tokenizer: real tokens would be lost.
    with pytest.raises(RuntimeError, match="smaller than the tokenizer"):
        runtime.initialize_replay(args)


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_can_switch_to_identically_dumped_replacement(
    tmp_path, runtime_environment, shuffle
):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = 7 if shuffle else None
    original, metadata = replay_cache(tmp_path / "original", args, runtime)
    runtime.initialize_replay(args)
    original_reader = reader(original, metadata, shuffle=shuffle, seed=7)
    expected = ids_of(original_reader.samples([8, 9], 10))
    original_reader.close()
    args.consumed_train_samples = 8
    replay_cache(tmp_path / "replacement", args, runtime)
    runtime._PLAN = None
    runtime.initialize_replay(args)
    iterator = build_loader_iterator(runtime, args, args.consumed_train_samples)
    try:
        batch = next(iterator)
        assert batch["_kd_sample_ids"] == expected
        assert torch.equal(
            scored(batch["tokens"], batch["loss_mask"]).T, batch["_kd_indices"].squeeze(-1)
        )
    finally:
        iterator.close()


def test_frontier_is_collective_and_errors_together(tmp_path, runtime_environment, monkeypatch):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    storage, metadata = replay_cache(tmp_path, args, runtime, iterations=2)
    runtime.initialize_replay(args)
    live = reader(storage, metadata)
    runtime._PLAN["readers"].append(live)
    listings = []
    original = runtime.discover_groups
    monkeypatch.setattr(runtime, "discover_groups", lambda *a: listings.append(a) or original(*a))
    args.consumed_train_samples = 8
    runtime.ensure_replay_frontier(args)  # 8 + 8 <= 16: no listing.
    assert not listings
    replay_cache(tmp_path, args, runtime, iterations=4)
    args.consumed_train_samples = 16
    runtime.ensure_replay_frontier(args)
    assert len(listings) == 1
    assert runtime._PLAN["available"] == 32 and live.total_samples == 32
    # A hole: the next range's owner has not published yet; every rank raises the same error.
    write_cache(tmp_path, start=6, iterations=8, metadata=metadata)
    args.consumed_train_samples = 32
    payload = []

    def broadcast(result, src):
        if payload:
            result[:] = payload
        else:
            payload[:] = result

    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(runtime.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="exhausted at 32 samples; requested 40.*parallel dump"):
        runtime.ensure_replay_frontier(args)
    monkeypatch.setattr(runtime.dist, "get_rank", lambda: 1)  # A peer receives the same plan.
    with pytest.raises(RuntimeError, match="exhausted at 32 samples; requested 40"):
        runtime.ensure_replay_frontier(args)
    assert len(listings) == 2


def test_first_train_step_establishes_the_replay_plan(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    replay_cache(tmp_path, args, runtime)
    loader = runtime.build_replay_loader(args, consumed=0)  # Before any plan exists.
    assert runtime._PLAN is None
    assert not runtime.before_train_step(args)
    assert runtime._PLAN["available"] == 32
    iterator = iter(loader.dataset)
    try:
        assert next(iterator)["_kd_sample_ids"] == [0, 1]
    finally:
        iterator.close()


def test_targets_are_handed_from_get_batch_to_forward_step(runtime_environment, monkeypatch):
    args, runtime, _ = runtime_environment
    runtime._PLAN = {"metadata": {"tokenizer": {"eod": 2}}, "readers": []}
    mpu = sys.modules["megatron.core"].parallel_state
    mpu.is_pipeline_last_stage = lambda **kw: True
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cached = types.ModuleType(f"{_namespace}.cached_logits_loss")
    logits = iter(["logits-0", "logits-1"])
    cached.get_student_logits_capture = lambda: types.SimpleNamespace(pop=lambda: next(logits))
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    cached_batch = {
        "tokens": torch.zeros(1, 8, dtype=torch.long),
        "_kd_values": torch.ones(8, 1, 1),
        "_kd_indices": torch.zeros(8, 1, 1, dtype=torch.long),
        "_kd_tail": torch.zeros(8, 1),
        "_kd_sample_ids": [3],
    }
    for attempt in range(2):  # A rerun re-reads the same cached batch.
        model_batch = runtime.on_tp0_batch(cached_batch)
        assert set(model_batch) == {"tokens"}
        runtime.stage_targets()
        bound = runtime.take_targets()
        assert bound["sample_ids"] == [3] and bound["logits"] == f"logits-{attempt}"
        assert torch.equal(bound["values"], cached_batch["_kd_values"])
        assert runtime.take_targets() is None  # Consumed by exactly one forward.
    runtime.on_tp0_batch({"tokens": torch.zeros(1, 8, dtype=torch.long)})  # Validation data.
    runtime.stage_targets()
    assert runtime.take_targets() is None


def test_frontier_rejects_changed_published_prefix(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    replay_cache(tmp_path, args, runtime, iterations=2)
    runtime.initialize_replay(args)
    (tmp_path / "dp1__0-16.index.json").unlink()
    args.consumed_train_samples = 16
    with pytest.raises(RuntimeError, match="Previously published"):
        runtime.ensure_replay_frontier(args)


def test_context_parallel_replay_scatters_shards_and_errors(runtime_environment, monkeypatch):
    _, runtime, _ = runtime_environment
    mpu = sys.modules["megatron.core"].parallel_state
    mpu.get_context_parallel_group = lambda: "cp"
    monkeypatch.setattr(runtime.dist, "get_global_rank", lambda group, rank: 11)
    sent = []

    def scatter(output, payload, src, group):
        assert (src, group) == (11, "cp")
        if payload is not None:
            sent[:] = payload
        output[0] = sent[rank[0]]

    monkeypatch.setattr(runtime.dist, "scatter_object_list", scatter)
    batch = {"tokens": torch.zeros(1), "_kd_values": ["v0", "v1"], "_kd_indices": ["i0", "i1"]}
    rank = [0]
    sender = runtime._ContextParallelReplay(lambda: iter([batch]), 0, 2)
    assert next(sender)["_kd_values"] == "v0"
    rank[0] = 1
    receiver = runtime._ContextParallelReplay(None, 1, 2)
    assert next(receiver)["_kd_indices"] == "i1"

    def broken():
        raise ValueError("corrupt file")
        yield

    rank[0] = 0
    with pytest.raises(RuntimeError, match="corrupt file"):
        next(runtime._ContextParallelReplay(broken, 0, 2))
    rank[0] = 1
    with pytest.raises(RuntimeError, match="corrupt file"):
        next(receiver)


@pytest.mark.parametrize(
    "option", ["logits_load_inputs", "logits_load_shuffle_shards", "seq_length"]
)
def test_standard_checkpoint_arg_checks_reject_replay_option_changes(runtime_environment, option):
    args, _, _ = runtime_environment
    args.num_layers = 2
    args.hidden_size = 8
    args.num_attention_heads = 2
    args.add_position_embedding = True
    args.vocab_file = None
    args.data_parallel_random_init = False
    args.phase_transition_iterations = None
    args.use_dist_ckpt = True
    path = Path(_package.__path__[0]).parent / "checkpointing.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "check_checkpoint_args"
    )
    namespace = {"get_args": lambda: args, "get_checkpoint_version": lambda: 3.0}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    saved = types.SimpleNamespace(**vars(args))
    namespace["check_checkpoint_args"](saved)
    value = getattr(saved, option)
    setattr(saved, option, not value if isinstance(value, bool) else value + 1)
    with pytest.raises(AssertionError, match=option):
        namespace["check_checkpoint_args"](saved)


def teacher_saver(runtime, tmp_path, dp_size=2):
    return types.SimpleNamespace(
        dp_size=dp_size,
        cp_size=1,
        metadata_dict={
            "saver": {"k": 1, "p": None, "min_k": 1, "save_dtype": "fp32"},
            "identifiers": {"seed": 7},
        },
        save_dir=str(tmp_path),
    )


def teacher_cache(tmp_path, runtime, args, **kwargs):
    saver = teacher_saver(runtime, tmp_path)
    shared = runtime._dump_settings(saver)
    shared["contract"]["first_sample"] = kwargs.pop("first_sample", 0)
    return write_cache(tmp_path, metadata=shared, **kwargs)


@pytest.mark.parametrize("create_attention_mask", [False, True])
def test_teacher_and_student_use_effective_attention_mode(
    tmp_path, runtime_environment, create_attention_mask
):
    args, runtime, _ = runtime_environment
    args.reset_attention_mask = True
    args.create_attention_mask_in_dataloader = create_attention_mask
    metadata = runtime._dump_settings(teacher_saver(runtime, tmp_path))
    metadata["contract"]["first_sample"] = 0
    expected = "causal_eod_reset" if create_attention_mask else "causal"
    contract = metadata["contract"]
    assert contract["attention_mode"] == expected
    assert contract["prefix_stable"] is True
    assert set(contract) == {*codec.REQUIRED_KEYS, *codec.REPLAY_KEYS}
    assert set(metadata["recorded"]) == set(codec.RECORDED_KEYS)
    assert metadata["provenance"]["producer"] == "megatron"
    runtime._check_student_compatibility(args, metadata)
    args.create_attention_mask_in_dataloader = not create_attention_mask
    with pytest.raises(ValueError, match="attention_mode"):
        runtime._check_student_compatibility(args, metadata)


@pytest.mark.parametrize(
    "seq_length,prefix_stable,ok",
    [(4, True, True), (4, False, False), (8, False, True), (12, False, True)],
)
def test_prefix_shortening_is_checked_per_sample(tmp_path, seq_length, prefix_stable, ok):
    # Stored samples have 8 tokens; the producer's recorded seq_length plays no role.
    storage, metadata = write_cache(
        tmp_path, iterations=2, contract={"prefix_stable": prefix_stable, "seq_length": 64}
    )
    live = reader(storage, metadata)
    samples = live.samples([0, 1], 2)
    live.close()
    if ok:
        result = replay.collate_samples(samples, seq_length, prefix_stable=prefix_stable)
        assert result["tokens"].shape == (2, seq_length)
    else:
        with pytest.raises(ValueError, match="prefix-stable"):
            replay.collate_samples(samples, seq_length, prefix_stable=prefix_stable)


@pytest.mark.parametrize("column", [*arrow.INPUT_COLUMNS, "indices_high"])
def test_blocks_need_every_input_column_and_the_vocabulary_index_width(column):
    record = codec.finalize_record(
        codec.capture_batch(inputs([0])),
        torch.zeros(8, 1),
        torch.zeros(8, 1, dtype=torch.long),
        torch.zeros(8),
        isolated=False,
    )
    batch = arrow.samples_batch(*record[:3], None, vocab_size=1 << 17)
    arrow.decode_block(batch, vocab_size=1 << 17)
    batch = batch.drop_columns([column])
    message = "index encoding" if column == "indices_high" else "missing columns"
    with pytest.raises(ValueError, match=message):
        arrow.decode_block(batch, vocab_size=1 << 17)


def test_targets_beyond_the_student_vocabulary_are_masked(tmp_path, capsys):
    storage, metadata = write_cache(tmp_path, iterations=2)
    exact = reader(storage, metadata, student_vocab_size=1 << 17)
    assert exact.samples([0], 1)[0]["teacher_values"].min() > -1000
    exact.close()
    # Token IDs are id*100+position, so sample 1's targets (100..107) exceed a 104-entry vocab.
    small = reader(storage, metadata, student_vocab_size=104)
    sample = small.samples([0, 1], 2)[1]
    small.close()
    masked = sample["teacher_indices"].reshape(-1) >= 0
    values = sample["teacher_values"].reshape(-1)
    # Position 0 has no row (zero loss weight); rows 101..103 survive, 104..107 are masked.
    assert (values[3:] == -1000).all() and (values[:3] > -1000).all()
    assert small.masked_entries > 0 and masked.all()
    assert "beyond the student vocabulary" in capsys.readouterr().out


def test_megatron_hf_tokenizer_fingerprint_matches_other_producers(
    runtime_environment, monkeypatch
):
    _, runtime, _ = runtime_environment

    class HF:
        bos_token_id, eos_token_id, pad_token_id, unk_token_id = 1, 2, None, 0

        def get_vocab(self):
            return {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3}

        def __len__(self):
            return 4

    wrapper = types.SimpleNamespace(eod=2, _tokenizer=types.SimpleNamespace(tokenizer=HF()))
    monkeypatch.setattr(sys.modules["megatron.training"], "get_tokenizer", lambda: wrapper)
    assert runtime.tokenizer_identity() == codec.hf_tokenizer_identity(HF(), eod_id=2)


@pytest.mark.parametrize(
    "divisor,seq_length,ok",
    [
        (1, 8, True),
        (4, 8, True),  # Segments of 4 re-pad to 4 for a CP=2 student.
        (8, 8, False),  # A CP=4 student's padding no longer fits.
        (1, 16, True),  # Longer packs re-pad into more room.
        (8, 16, True),
        (1, 4, False),  # Two segments can't be shortened.
    ],
)
def test_packed_samples_are_checked_per_sample(tmp_path, divisor, seq_length, ok):
    # The producer recorded divisor 1 and a longer seq_length; neither is checked per file.
    storage, metadata = write_cache(
        tmp_path, packed=True, contract={"padding_divisor": 1, "seq_length": 64}
    )
    live = reader(storage, metadata)
    samples = live.samples([0, 1], 2)
    live.close()
    if ok:
        replay.collate_samples(samples, seq_length, packed=True, divisor=divisor)
    else:
        with pytest.raises(ValueError, match="student sequence length"):
            replay.collate_samples(samples, seq_length, packed=True, divisor=divisor)


@pytest.mark.parametrize("prefix_stable", [True, False])
def test_long_single_segment_packed_sample_is_prefix_shortened(prefix_stable):
    sample = {
        "tokens": torch.arange(8),
        "labels": torch.arange(1, 9),
        "position_ids": torch.arange(8),
        "loss_mask": torch.ones(8),
        "segments": torch.tensor([8]),
        "sample_id": 0,
    }
    if not prefix_stable:
        with pytest.raises(ValueError, match="prefix-stable"):
            replay.collate_samples([sample], 4, packed=True, prefix_stable=False)
        return
    result = replay.collate_samples([sample], 4, packed=True, divisor=2)
    assert result["tokens"].tolist() == [[0, 1, 2, 3]]
    assert result["cu_seqlens"].tolist() == [[0, 4]]


@pytest.mark.parametrize(
    "override,exit_interval,expected",
    [
        (None, None, (None, 128)),
        (4, 4, (4, 8)),
        (6, 4, (6, 8)),
        (124, 8, (124, 128)),
        (4, None, (4, 128)),
    ],
)
def test_dump_range_uses_override_and_exit_interval(
    runtime_environment, override, exit_interval, expected
):
    args, runtime, _ = runtime_environment
    args.logits_save_range_start = override
    args.exit_interval = exit_interval
    assert runtime.dump_range(args) == expected


def test_frozen_resume_uses_published_range_end(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.logits_save_dir = str(tmp_path)
    args.exit_interval = 4
    # Job B (iterations 4-8) has published 4-6; job A (0-4) has nothing yet.
    teacher_cache(tmp_path, runtime, args, start=4, iterations=6)
    args.override_ckpt_iteration = 4
    assert runtime.frozen_resume_iteration(args) == 6
    assert args.logits_save_range_start == 4
    args.override_ckpt_iteration = 6  # Replaced by load_checkpoint; the range start sticks.
    assert runtime.frozen_resume_iteration(args) == 6
    del args.logits_save_range_start
    args.override_ckpt_iteration = 0
    assert runtime.frozen_resume_iteration(args) == 0  # Job A starts its own range.
    del args.logits_save_range_start
    args.override_ckpt_iteration = None  # A single sequential writer stops at the hole.
    assert runtime.frozen_resume_iteration(args) == 0


def test_completed_range_exits_before_any_step(runtime_environment):
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.logits_load_inputs = False
    args.exit_interval = 4
    args.logits_save_range_start = 4
    args.consumed_train_samples = 7 * 8
    assert not runtime.before_train_step(args)
    args.consumed_train_samples = 8 * 8
    assert runtime.before_train_step(args)
    args.logits_save_range_start = None  # A sequential writer is bounded by --train-iters.
    assert not runtime.before_train_step(args)


def test_teacher_metadata_checks_cache_and_cleans_only_its_range(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.logits_save_dir = str(tmp_path)
    args.exit_interval = 4
    saver = teacher_saver(runtime, tmp_path)
    metadata = runtime.initialize_dump_metadata(saver)  # Empty cache.
    assert (metadata["published_through"], metadata["contract"]["first_sample"]) == (0, 0)
    teacher_cache(tmp_path, runtime, args, start=4, iterations=8)
    (tmp_path / "dp1__48-64.index.json").unlink()  # Job B's interrupted tail.
    args.logits_save_range_start = 4
    args.consumed_train_samples = 6 * 8
    metadata = runtime.initialize_dump_metadata(saver)
    assert (metadata["published_through"], metadata["range_end"]) == (48, 64)
    assert not storage_module.Storage(str(tmp_path)).list("dp*__48-64.*")
    assert len(storage_module.Storage(str(tmp_path)).list("dp*__32-48.*")) == 4
    args.consumed_train_samples = 8 * 8
    with pytest.raises(RuntimeError, match="outside this job's published range"):
        runtime.initialize_dump_metadata(saver)
    args.consumed_train_samples = 6 * 8
    args.micro_batch_size = 1
    with pytest.raises(RuntimeError, match="writer_mbs"):
        runtime.initialize_dump_metadata(saver)


def test_unfrozen_teacher_recovers_first_sample(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    dump_args(args)
    args.logits_save_dir = str(tmp_path)
    args.freeze_all_layers = False
    args.consumed_train_samples = 16
    saver = teacher_saver(runtime, tmp_path)
    assert runtime.initialize_dump_metadata(saver)["contract"]["first_sample"] == 16
    teacher_cache(tmp_path, runtime, args, first_sample=16, start=2, iterations=4)
    args.load = "resumed-training-checkpoint"
    args.consumed_train_samples = 24
    metadata = runtime.initialize_dump_metadata(saver)
    assert (metadata["contract"]["first_sample"], metadata["published_through"]) == (16, 32)
    assert metadata["recorded"]["teacher"] == "resumed-training-checkpoint"


def test_teacher_attempt_commit_discards_rerun_data(runtime_environment, monkeypatch):
    args, runtime, legacy = runtime_environment
    module = importlib.import_module(f"{_namespace}.v3_saver")
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "get_num_microbatches", lambda: 2)
    args.global_batch_size = 4
    saver = module.PairedLogitsSaver.__new__(module.PairedLogitsSaver)
    saver.tp_rank = saver.cp_rank = saver.dp_rank = 0
    saver.cp_size = saver.dp_size = 1
    saver._initialized = True
    saver.metadata_dict = {
        "published_through": 0,
        "contract": {"attention_mode": "causal", "logprob_vocab_size": 1 << 17},
    }
    saver._captured = []
    saver._pending_writes = collections.OrderedDict()
    saver._process_single_microbatch = lambda logits: (
        logits.half(),
        logits.long(),
        torch.zeros(logits.shape[:2]),
    )

    def forward(ids):
        source = inputs(ids)
        saver.capture(source)
        saver._forward_hook(
            types.SimpleNamespace(training=True), (), source["tokens"].T.unsqueeze(-1)
        )

    saver.begin_attempt()
    forward([0, 1])
    forward([2, 3])
    assert not saver._pending_writes
    saver.begin_attempt()
    forward([4, 5])
    forward([6, 7])
    saver.commit_attempt()
    record = saver._pending_writes[(0, 4)]
    batch = record["batch"]
    assert "sample_ids" not in record and "sample_id" not in batch.schema.names
    restored, _ = arrow.decode_block(batch.select(list(arrow.INPUT_COLUMNS)))
    assert restored["tokens"][0] == 400
    # Identity targets are not log probabilities, so inspect the columns directly.
    values, indices, tail = batch_targets(batch)
    weights = restored["loss_mask"]
    assert torch.equal(restored["tokens"].long()[weights != 0], indices.squeeze(-1))
    assert torch.equal(values.long(), indices)
    assert tail.dtype == np.float32 and tail.shape == values.shape[:1]


def test_saver_writes_files_readable_by_replay(tmp_path, runtime_environment, monkeypatch):
    args, runtime, _ = runtime_environment
    module = importlib.import_module(f"{_namespace}.v3_saver")
    metadata = settings()
    writes = {}
    for iteration in (0, 1):
        start, end = iteration * 8, (iteration + 1) * 8
        for rank in (0, 1):
            ids = [start + (mb * 2 + rank) * 2 + j for mb in range(2) for j in range(2)]
            record = make_record(codec.capture_batch(inputs(ids)), metadata["contract"])
            writes.setdefault(rank, {})[(start, end)] = {
                "start": start,
                "end": end,
                "batch": arrow.samples_batch(*record, vocab_size=1 << 17),
            }
    meta = json.dumps({**metadata, "published_through": 0, "range_end": 64})
    for rank in (0, 1):
        module.PairedLogitsSaver._write_batched_tar(
            str(tmp_path / storage_module.data_name(rank, 0, 16)), writes[rank], meta
        )
    storage = Storage(str(tmp_path))
    meta = storage_module.read_meta(storage, storage_module.FileName(0, 0, 16))
    assert meta == metadata  # Resume-only state is not stored.
    live = reader(storage, metadata)
    assert ids_of(live.samples(list(range(16)), 16)) == list(range(16))
    live.close()


def _loss_namespace(monkeypatch):
    """Execute the real shared LM/KD loss helpers without importing GPU startup."""
    path = Path(_package.__path__[0]) / "cached_logits_loss.py"
    names = {"topk_kl_div", "masked_loss_sum", "lm_loss_and_report", "add_kd_loss"}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    dist = types.SimpleNamespace(
        all_reduce=lambda *a, **kw: None,
        ReduceOp=torch.distributed.ReduceOp,
        ProcessGroup=torch.distributed.ProcessGroup,
    )
    namespace = {
        "torch": torch,
        "dist": dist,
        "dist_nn": None,
        "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3,
        "parallel_state": types.SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
        **{name: getattr(typing, name) for name in ("Dict", "Optional", "Tuple")},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    cached = types.ModuleType(f"{_namespace}.cached_logits_loss")
    for name in names:
        setattr(cached, name, namespace[name])
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    return cached


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_paired_loss_uses_explicit_targets_and_keeps_gradients(
    runtime_environment, monkeypatch, dtype
):
    args, runtime, legacy = runtime_environment
    cached = _loss_namespace(monkeypatch)
    logits = torch.randn(8, 2, 3, dtype=dtype, requires_grad=True)
    cached.get_student_logits_capture = lambda: types.SimpleNamespace(pop=lambda: logits)
    monkeypatch.delitem(sys.modules, f"{_namespace}.v3_loss", raising=False)
    module = importlib.import_module(f"{_namespace}.v3_loss")
    monkeypatch.setattr(module, "get_args", lambda: args)
    teacher = torch.randn(8, 2, 3).log_softmax(-1)
    sidecar = runtime.bind_student_logits(
        {
            "values": teacher,
            "indices": torch.arange(3).expand(8, 2, -1),
            "tail": codec.tail_logprob(teacher),
        }
    )
    mask = torch.ones(2, 8)
    mask[0, 0] = 0
    labels = torch.randint(3, (8, 2))
    lm_losses = (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1), reduction="none"
        )
        .reshape(8, 2)
        .T
    )
    total, count, report = module.paired_loss(
        mask, lm_losses, types.SimpleNamespace(training=True), sidecar
    )
    expected = (teacher.exp() * (teacher - logits.float().log_softmax(-1))).sum(-1).T
    assert torch.allclose(
        total,
        args.logits_load_kd_loss_alpha * (expected * mask).sum()
        + 0.1 * (lm_losses.float() * mask).sum(),
        atol=1e-5,
    )
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert count == 15
    assert list(report) == ["lm loss", "logits distillation loss", "total loss"]
    lm_only = module.paired_loss(mask, lm_losses, types.SimpleNamespace(training=False), None)
    assert list(lm_only[2]) == ["lm loss"]


@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize("missing_rows", [[1], [0, 1]])
def test_missing_kd_targets_keep_ce_loss_and_gradients(
    runtime_environment, monkeypatch, alpha, missing_rows
):
    args, _, _ = runtime_environment
    args.logits_load_kd_loss_alpha = alpha
    _loss_namespace(monkeypatch)
    monkeypatch.delitem(sys.modules, f"{_namespace}.v3_loss", raising=False)
    module = importlib.import_module(f"{_namespace}.v3_loss")
    monkeypatch.setattr(module, "get_args", lambda: args)
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 1.0, 0.0]]).log_softmax(-1)
    values = teacher.clone()
    values[missing_rows] = codec.LOGPROB_SENTINEL
    sample = {
        "tokens": torch.tensor([0, 1, 2], dtype=torch.int32),
        "labels": torch.tensor([1, 2, 0], dtype=torch.int32),
        "position_ids": torch.arange(3, dtype=torch.int32),
        "loss_mask": torch.tensor([1, 1, 0], dtype=torch.bool),
        "segments": torch.tensor([3]),
        "sample_id": 0,
        "teacher_values": values,
        "teacher_indices": torch.arange(3).expand(3, -1),
        "teacher_tail": codec.tail_logprob(values),
    }
    batch = replay.collate_samples([sample], 4)
    logits = torch.tensor(
        [[[0.0, 1.0, -1.0]], [[1.0, 0.0, -1.0]], [[-1.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    ce = torch.nn.functional.cross_entropy(
        logits[:, 0], batch["labels"][0], reduction="none"
    ).unsqueeze(0)
    total, count, report = module.paired_loss(
        batch["loss_mask"],
        ce,
        types.SimpleNamespace(training=True),
        {
            "logits": logits,
            "values": batch["_kd_values"][0],
            "indices": batch["_kd_indices"][0],
            "tail": batch["_kd_tail"][0],
        },
    )
    expected_kd = (teacher.exp() * (teacher - logits[:3, 0].log_softmax(-1))).sum(-1)
    expected_kd[missing_rows] = 0
    expected = (1 - alpha) * ce[0, :2].sum() + alpha * expected_kd[:2].sum()
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(report["lm loss"][0], ce[0, :2].sum())
    assert count == 2
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() > 0  # CE still trains the token with no KD entries.
    assert (logits.grad[2:] == 0).all()  # Original masking and inserted padding remain unscored.


def legacy_saver_methods():
    """Execute the inherited TP/top-P helpers without Megatron GPU imports."""
    path = Path(_package.__path__[0]) / "logits_saver.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    names = {
        "_process_single_microbatch",
        "_topk_logprobs",
        "_compute_global_topk",
        "_apply_topp_truncation",
    }
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_max_vocab_size" for t in n.targets)
    ]
    namespace = {
        "torch": torch,
        "dist": torch.distributed,
        "CACHED_LOGITS_INDEX_SENTINEL": -1,
        "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_only_v3_extracts_vocabularies_above_17_bits(runtime_environment):
    module = importlib.import_module(f"{_namespace}.v3_saver")
    helpers = legacy_saver_methods()
    method = helpers["_process_single_microbatch"]
    saver = types.SimpleNamespace(
        tp_size=1,
        k=2,
        p=None,
        _save_dtype=torch.float32,
        _max_vocab_size=helpers["_max_vocab_size"],
    )
    saver._topk_logprobs = types.MethodType(helpers["_topk_logprobs"], saver)
    logits = torch.arange((1 << 17) + 1, dtype=torch.float32).reshape(1, 1, -1)
    with pytest.raises(AssertionError, match="maximum supported"):
        method(saver, logits)
    saver._max_vocab_size = module.PairedLogitsSaver._max_vocab_size
    values, indices = method(saver, logits)
    assert indices.tolist() == [[[1 << 17, (1 << 17) - 1]]]
    restored, ids, _ = target_roundtrip(
        values.reshape(1, 2), indices.reshape(1, 2), vocab_size=logits.shape[-1]
    )
    assert torch.equal(ids, indices.reshape(1, 2))
    assert torch.equal(restored, values.reshape(1, 2))


@pytest.mark.parametrize("local_vocab_size", [4, 1 << 23])
def test_tp_transport_preserves_wide_indices(local_vocab_size, monkeypatch):
    method = legacy_saver_methods()["_compute_global_topk"]
    remote_id = 2 * local_vocab_size - 1

    def gather(tensor, outputs, dst, group):
        assert tensor.dtype == torch.float32
        outputs[0].copy_(tensor)
        outputs[1].copy_(torch.tensor([[[[6.0, -0.5, remote_id]]]], dtype=torch.float32))

    monkeypatch.setattr(torch.distributed, "gather", gather)
    saver = types.SimpleNamespace(tp_rank=0, tp_size=2, _tp_dst_rank_global=0, tp_group=None)
    values, ids = method(
        saver,
        torch.tensor([[[5.0]]]),
        torch.tensor([[[-1.0]]]),
        torch.tensor([[[1]]]),
        1,
        local_vocab_size,
    )
    assert ids.item() == remote_id
    assert values.item() == -0.5
    assert values.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_inherited_tp_top_p_survives_v3_codec(dtype, monkeypatch):
    helpers = legacy_saver_methods()
    full = torch.tensor([[[5.0, 3.0, 1.0, 6.0, 4.0, 2.0]], [[2.0, 4.0, 6.0, 1.0, 3.0, 5.0]]])
    chunks = full.chunk(2, dim=-1)
    lses = torch.stack([torch.logsumexp(c, -1, keepdim=True) for c in chunks])
    maximum = lses.max(0).values
    global_lse = torch.logsumexp(full, -1, keepdim=True)
    calls = []

    def reduce(tensor, op, group):
        calls.append(op)
        tensor.copy_(
            maximum if op == torch.distributed.ReduceOp.MAX else (lses - maximum).exp().sum(0)
        )

    def gather(tensor, outputs, dst, group):
        for rank, chunk in enumerate(chunks):
            values, indices = chunk.topk(3, dim=-1)
            outputs[rank].copy_(
                torch.stack((values, values - global_lse, (indices + rank * 3).float()), -1)
            )

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(torch.distributed, "gather", gather)
    saver = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_group=object(),
        _tp_dst_rank_global=7,
        _max_vocab_size=helpers["_max_vocab_size"],
        k=4,
        p=0.7,
        min_k=2,
        _save_dtype=dtype,
        _topp_kept_counts=[],
    )
    for name in ("_topk_logprobs", "_compute_global_topk", "_apply_topp_truncation"):
        setattr(saver, name, types.MethodType(helpers[name], saver))
    values, indices = helpers["_process_single_microbatch"](saver, chunks[0])
    assert calls == [torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.SUM]
    # The tail is the full-precision mass outside the kept nucleus, before downcasting.
    fp32_values, _ = saver._topk_logprobs(chunks[0])
    tail = codec.tail_logprob(fp32_values)
    expected, expected_ids = full.log_softmax(-1).topk(4, dim=-1)
    probs = expected.exp()
    keep = (probs.cumsum(-1) - probs < 0.7) | (torch.arange(4) < 2)
    assert torch.equal(indices, torch.where(keep, expected_ids, -1))
    assert torch.allclose(values, torch.where(keep, expected, -1e3).to(dtype), atol=1e-6)
    kept_mass = (full.softmax(-1).topk(4, dim=-1).values * keep).sum(-1)
    assert torch.allclose(tail, (1 - kept_mass).log(), atol=1e-6)
    restored, restored_ids, restored_tail = target_roundtrip(
        values.reshape(-1, 4), indices.reshape(-1, 4), tail.reshape(-1), vocab_size=6
    )
    assert torch.equal(restored_tail, tail.reshape(-1))
    assert torch.equal(restored, values.reshape(-1, 4))
    assert torch.equal(
        restored_ids[keep.reshape(-1, 4)], expected_ids.reshape(-1, 4)[keep.reshape(-1, 4)]
    )
    assert (restored[~keep.reshape(-1, 4)] == -1e3).all()
    assert values.shape[-1] == 4  # Different nucleus sizes never change collective shapes.


def test_top_p_metric_is_committed_only_for_accepted_attempt(runtime_environment, monkeypatch):
    args, _, _ = runtime_environment
    module = importlib.import_module(f"{_namespace}.v3_saver")
    args.curr_iteration = 12
    events = []
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(
        module,
        "get_tensorboard_writer",
        lambda: types.SimpleNamespace(add_scalar=lambda *a: events.append(a)),
    )
    saver = module.PairedLogitsSaver.__new__(module.PairedLogitsSaver)
    saver.tp_rank = saver.cp_rank = saver.dp_rank = 0
    saver._initialized = True
    saver.metadata_dict = {"published_through": 0}
    saver._captured = []
    saver._pending_writes = {}
    saver._topp_kept_counts = [99.0]
    saver.begin_attempt()
    saver._topp_kept_counts = [2.0, 4.0]
    saver._iteration_records = {"start": 0, "end": 4}
    saver.commit_attempt()
    assert events == [("avg-logprobs-kept", 3.0, 12)]
    assert not saver._topp_kept_counts


def test_msc_opt_in_and_worker_enable_are_preserved(runtime_environment, monkeypatch):
    _, _, _ = runtime_environment
    state = {"enabled": False}
    backend = types.SimpleNamespace()

    def enable():
        state["enabled"] = True

    def package():
        if not state["enabled"]:
            raise RuntimeError("pass --enable-msc")
        return backend

    msc = types.ModuleType("megatron.core.msc_utils")
    msc.MultiStorageClientFeature = types.SimpleNamespace(enable=enable, import_package=package)
    monkeypatch.setitem(sys.modules, msc.__name__, msc)
    with pytest.raises(RuntimeError, match="enable-msc"):
        Storage("msc://test/dump")
    calls = []
    backend.glob = lambda pattern: calls.append(pattern) or ["msc://test/dump/dp0__0-8.index.json"]
    module = importlib.import_module(f"{_namespace}.v3_saver")
    monkeypatch.setattr(module, "write_file", lambda *a: (b"", {}))
    monkeypatch.setattr(module, "publish", lambda *a: None)
    module.PairedLogitsSaver._write_batched_tar(
        "msc://test/dump/dp0__0-8.arrow",
        {0: {"start": 0, "end": 8, "batch": None}},
        "{}",
        msc_enabled=True,
    )
    assert state["enabled"]
    storage = Storage("msc://test/dump")
    assert storage_module.list_published(storage) == [storage_module.FileName(0, 0, 8)]
    storage_module.list_published(storage)
    assert len(calls) == 2  # Refreshes never reuse a stale application-level listing.


def test_v3_rejects_legacy_ignore_errors(runtime_environment):
    args, runtime, _ = runtime_environment
    args.logits_load_dir = "/test/dump"
    args.sequence_packing_scheduler = args.hybrid_context_parallel = args.use_varlen_dataset = False
    args.mtp_num_layers = None
    args.logits_load_ignore_errors = True
    with pytest.raises(ValueError, match="ignore-errors"):
        runtime.validate_options(args)


@pytest.mark.parametrize("packed", [False, True])
def test_cp_saver_reassembles_canonical_token_targets(runtime_environment, monkeypatch, packed):
    args, _, _ = runtime_environment
    module = importlib.import_module(f"{_namespace}.v3_saver")
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "get_num_microbatches", lambda: 1)
    captured = codec.capture_batch(inputs([0, 1], packed=packed))
    maps = [replay.token_map(captured, rank, 2) for rank in range(2)]
    full_ids = captured["tokens"].long().reshape(-1, 1)
    full_values = -(full_ids.float() + 1) / 1000  # Distinct log probabilities per token.
    full_tail = codec.tail_logprob(full_values)
    mapped = [replay.map_targets(full_values, full_ids, full_tail, m) for m in maps]
    parts = [maps, *([part[field] for part in mapped] for field in range(3))]
    calls = []

    def gather(tensor, buffers, dst, group):
        for output, expected in zip(buffers, parts[len(calls)]):
            output.copy_(expected)
        calls.append(tensor)

    monkeypatch.setattr(module.dist, "gather", gather)
    saver = module.PairedLogitsSaver.__new__(module.PairedLogitsSaver)
    saver.cp_rank = saver.dp_rank = 0
    saver.cp_size = 2
    saver.dp_size = 1
    saver.cp_group = object()
    saver._cp_dst_rank_global = 9
    saver.metadata_dict = {
        "contract": {
            "attention_mode": "isolated_segments" if packed else "causal",
            "logprob_vocab_size": 1 << 17,
        }
    }
    saver._captured = [(captured, maps[0], *mapped[0])]
    saver._serialize_attempt()
    values, indices, tail = batch_targets(saver._iteration_records["batch"])
    tail = arrow._tensor(tail)
    assert len(calls) == 4
    rows = captured["loss_mask"] != 0  # No trailing padding here, so only weights drop rows.
    assert torch.equal(values, full_values[rows])
    assert torch.equal(indices, full_ids[rows])
    assert torch.equal(tail, full_tail[rows])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tp_target_broadcast_preserves_shape_dtype_and_virtual_stage(
    runtime_environment, monkeypatch, dtype
):
    _, runtime, _ = runtime_environment
    mpu = sys.modules["megatron.core"].parallel_state
    stage = [False]
    mpu.is_pipeline_last_stage = (
        lambda **kw: stage[0] and kw["vp_stage"] == 2 and not kw["ignore_virtual"]
    )
    group = object()
    mpu.get_tensor_model_parallel_group = lambda: group
    mpu.get_tensor_model_parallel_src_rank = lambda: 7
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pinned = []

    def pin(tensor):
        pinned.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    header, tensors = [], []
    receive = [False]

    def broadcast_object(result, src, group):
        assert src == 7
        if receive[0]:
            result[:] = header
        else:
            header[:] = result

    def broadcast(tensor, src, group):
        assert src == 7
        if receive[0]:
            tensor.copy_(tensors.pop(0))
        else:
            tensors.append(tensor.clone())

    monkeypatch.setattr(runtime.dist, "broadcast_object_list", broadcast_object)
    monkeypatch.setattr(runtime.dist, "broadcast", broadcast)
    source = {
        "values": torch.randn(4, 2, 3).to(dtype),
        "indices": torch.arange(24).reshape(4, 2, 3),
        "tail": -torch.rand(4, 2),
        "sample_ids": [2, 3],
    }
    assert runtime.broadcast_targets(source, vp_stage=2) is None
    assert not header and not tensors
    stage[0] = True
    sent = runtime.broadcast_targets(source, vp_stage=2)
    receive[0] = True
    received = runtime.broadcast_targets(None, vp_stage=2)
    assert torch.equal(sent["values"], received["values"])
    assert received["values"].dtype == dtype
    assert torch.equal(sent["indices"], received["indices"])
    assert received["indices"].dtype == torch.int64
    assert torch.equal(sent["tail"], received["tail"])
    assert received["tail"].dtype == torch.float32
    assert received["sample_ids"] == [2, 3]
    assert len(pinned) == 3


def test_hybrid_padded_cp_mapping_uses_core_metadata_and_masks_inserted_padding(monkeypatch):
    path = Path(_package.__path__[0]).parents[1] / "core/context_parallel/layout.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_thd_zigzag_metadata"
    )
    namespace = {"torch": torch, "_THDZigzagMetadata": types.SimpleNamespace}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [node], type_ignores=[]), str(path), "exec"), namespace)
    module = types.ModuleType("megatron.core.context_parallel.layout")
    module._build_thd_zigzag_metadata = namespace[node.name]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    captured = codec.capture_batch(inputs([0, 1], length=8, packed=True))
    captured["cu_seqlens"] = [torch.tensor([0, 3, 8], dtype=torch.int32)] * 2
    full = torch.arange(16).reshape(-1, 1)
    seen = []
    for rank in range(2):
        mapping = replay.token_map(captured, rank, 2, hybrid_padded_zigzag=True, tp_alignment=4)
        values, indices, tail = replay.map_targets(
            full.float(), full, -full.float().squeeze(-1), mapping
        )
        assert (tail[mapping < 0] == 0).all()  # Inserted padding: no tail mass to match.
        assert mapping.numel() % 4 == 0
        assert (values.squeeze(-1)[mapping < 0] == -1e3).all()
        assert torch.equal(indices.squeeze(-1)[mapping >= 0], mapping[mapping >= 0])
        seen.extend(mapping[mapping >= 0].tolist())
    assert sorted(seen) == list(range(16))


def test_top_p_encoded_sentinel_is_excluded_from_kl_and_ghost_mass():
    path = Path(_package.__path__[0]) / "cached_logits_loss.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "topk_kl_div"
    )
    namespace = {
        "torch": torch,
        "dist": torch.distributed,
        "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3,
        "Optional": typing.Optional,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    # With a u16 encoding, -1 decodes as 65535, a real vocabulary ID at the limit.
    values, indices, tail = target_roundtrip(
        torch.tensor([[0.7, 0.0]]).log().clamp_min(-1e3),
        torch.tensor([[0, -1]]),
        vocab_size=1 << 16,
    )
    assert torch.allclose(tail, torch.tensor([0.3]).log())
    assert indices[0, 1] == 65535
    logits = torch.linspace(-2, 2, 1 << 16).reshape(1, 1, -1).requires_grad_()
    actual = namespace["topk_kl_div"](
        logits.clone(), values.unsqueeze(0), indices.unsqueeze(0), 1, 0, None
    )
    logprobs = logits.log_softmax(-1)
    p = logprobs[..., 0].exp()
    expected = 0.7 * (torch.tensor(0.7).log() - logprobs[..., 0]) + 0.3 * (
        torch.tensor(0.3).log() - (1 - p).log()
    )
    assert torch.allclose(actual, expected, atol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()
    # A stored tail replaces the derived ghost bucket.
    stored = namespace["topk_kl_div"](
        logits.detach().clone(),
        values.unsqueeze(0),
        indices.unsqueeze(0),
        1,
        0,
        None,
        teacher_tail_logprobs=tail.unsqueeze(0),
    )
    assert torch.allclose(stored, expected.detach(), atol=1e-5)


def test_v3_async_write_failure_sets_shared_event(runtime_environment, monkeypatch, tmp_path):
    module = importlib.import_module(f"{_namespace}.v3_saver")
    failures = []

    def broken(*args):
        raise OSError("failed publication")

    monkeypatch.setattr(module, "write_file", lambda *a: (b"", {}))
    monkeypatch.setattr(module, "publish", broken)
    event = types.SimpleNamespace(set=lambda: failures.append(True))
    with pytest.raises(OSError, match="failed publication"):
        module.PairedLogitsSaver._write_batched_tar(
            str(tmp_path / "dp0__0-8.arrow"),
            {0: {"start": 0, "end": 8, "batch": None}},
            "{}",
            failure_event=event,
        )
    assert failures == [True]


@pytest.mark.parametrize(
    "teacher_dp,teacher_mbs,teacher_gbs,student_dp,student_mbs",
    [(2, 2, 8, 3, 3), (3, 2, 12, 2, 3), (1, 3, 9, 4, 2), (4, 2, 16, 1, 3)],
)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_dp_resharding_with_rampup_shuffle_and_resume(
    tmp_path, teacher_dp, teacher_mbs, teacher_gbs, student_dp, student_mbs, packed, shuffle
):
    storage, metadata = write_cache(
        tmp_path, dp=teacher_dp, mbs=teacher_mbs, gbs=teacher_gbs, iterations=12, packed=packed
    )
    prototype = reader(storage, metadata, shuffle=shuffle, seed=7)
    expected_order = [sid for start, end in prototype.groups for sid in range(start, end)]
    prototype.close()
    # A valid old student checkpoint need not align to teacher iterations,
    # saved microbatches, or the new student's batch size.
    consumed = 5
    count = [1]

    def iterator(rank, cursor):
        return replay.ReplayIterator(
            reader(storage, metadata, shuffle=shuffle, seed=7),
            consumed=cursor,
            dp_rank=rank,
            dp_size=student_dp,
            micro_batch_size=student_mbs,
            seq_length=8,
            num_microbatches=lambda: count[0],
            prefetch=True,
        )

    iterators = [iterator(rank, consumed) for rank in range(student_dp)]
    try:
        for num_mb in (1, 2, 3):
            count[0] = num_mb
            seen = []
            for mb in range(num_mb):
                for rank, stream in enumerate(iterators):
                    batch = next(stream)
                    start = consumed + (mb * student_dp + rank) * student_mbs
                    expected = expected_order[start : start + student_mbs]
                    assert batch["_kd_sample_ids"] == expected
                    # Packed batches map to the flattened THD layout, others to [S, B].
                    targets = scored(batch["tokens"], batch["loss_mask"])
                    layout = targets.reshape(-1, 1) if packed else targets.T
                    assert torch.equal(layout, batch["_kd_indices"][0].squeeze(-1))
                    assert torch.equal(
                        batch["tokens"], torch.tensor(expected).unsqueeze(1) * 100 + torch.arange(8)
                    )
                    if packed:
                        assert torch.equal(
                            batch["cu_seqlens"],
                            torch.tensor([0, 4, 8], dtype=torch.int32).expand(student_mbs, -1),
                        )
                    seen.extend(expected)
            end = consumed + num_mb * student_dp * student_mbs
            assert seen == expected_order[consumed:end]
            assert all(stream.consumed == end for stream in iterators)
            consumed = end
        count[0] = 2
        for rank, stream in enumerate(iterators):
            resumed = iterator(rank, consumed)
            try:
                assert next(resumed)["_kd_sample_ids"] == next(stream)["_kd_sample_ids"]
            finally:
                resumed.close()
    finally:
        for stream in iterators:
            stream.close()


FIXTURES = Path(__file__).resolve().parent / "fixtures"


FIXTURE_VARIANTS = {
    # name: (write_cache kwargs)
    "u16_causal": dict(contract={"logprob_vocab_size": 4096}),  # No stored tail.
    "u16u8_packed_stripped": dict(
        packed=True, contract={"padding_divisor": 2}, source=tail_masked, gbs=4, mbs=1
    ),
    "u16_logprobs_with_tail": dict(contract={"logprob_vocab_size": 4096}, logprobs=True),
}


def build_fixture(root, name):
    return write_cache(root, iterations=2, bundle=2, **FIXTURE_VARIANTS[name])


@pytest.mark.parametrize("name", sorted(FIXTURE_VARIANTS))
def test_golden_fixtures_decode(tmp_path, name):
    """Committed V3 dumps that other producers/consumers (e.g. ModelOpt) must also decode.

    Regenerate with ``OFFLINE_KD_V3_REGEN_FIXTURES=1``; review the diff before committing.
    """
    import os
    import shutil

    golden = FIXTURES / name
    if os.environ.get("OFFLINE_KD_V3_REGEN_FIXTURES"):
        shutil.rmtree(golden, ignore_errors=True)
        build_fixture(golden, name)
    expected_storage, expected_meta = build_fixture(tmp_path, name)
    storage = Storage(str(golden))
    meta = storage_module.read_meta(storage, storage_module.list_published(storage)[0])
    codec.validate_metadata(meta)
    assert codec.mismatched_settings(meta, expected_meta) == ([], [])
    count = groups(storage, meta)[-1][1]
    got_reader, want_reader = reader(storage, meta), reader(expected_storage, expected_meta)
    try:
        got = got_reader.samples(list(range(count)), count)
        want = want_reader.samples(list(range(count)), count)
    finally:
        got_reader.close()
        want_reader.close()
    for a, b in zip(got, want):
        for key in (
            *codec.TOKEN_FIELDS,
            "segments",
            "teacher_values",
            "teacher_indices",
            "teacher_tail",
        ):
            assert torch.equal(a[key], b[key]), (name, key)
    first = storage_module.list_published(storage)[0]
    schema = pa.ipc.open_file(storage.path(first.data)).schema
    assert ("tail" in schema.names) == (name == "u16_logprobs_with_tail")
    if name == "u16_logprobs_with_tail":
        assert schema.field("tail").type == pa.large_list(pa.float32())
        tail = got[0]["teacher_tail"]
        assert torch.allclose(got[0]["teacher_values"].exp().sum(-1) + tail.exp(), torch.ones(1))
