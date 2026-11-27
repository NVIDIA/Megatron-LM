# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU format and bounded-I/O contracts for prepared-token mask companions."""

import multiprocessing
import random
import struct
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby

import numpy as np
import pytest

from examples.multimodal.data_loading.token_mask import TokenMaskFileStore, expand_token_mask
from megatron.energon.cache.file_store import ByteRangeStore
from megatron.energon.epathlib import EPath


def _write_mask(path, records, *, reserved=37):
    runs = bytearray()
    offsets = [0]
    for record in records:
        for count, flag in record:
            runs.extend(struct.pack("<iB3s", count, flag, b"\0\0\0"))
        offsets.append(len(runs) // 8)
    path.write_bytes(
        runs
        + struct.pack(f"<{len(offsets)}q", *offsets)
        + struct.pack("<8sQQQQ", b"MASKID\0\1", reserved, len(records), len(runs) // 8, len(runs))
    )


def test_record_reads_preserve_boolean_flags_and_source_identity(tmp_path):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(2, 1), (3, 0)], [], [(1, 0), (2, 1)]])
    store = TokenMaskFileStore(path)
    assert len(store) == 3
    assert expand_token_mask(store["0"][0], 5) == [True, True, False, False, False]
    assert expand_token_mask(store["1.mask"][0], 0) == []
    data, source = store["2.mask"]
    assert expand_token_mask(data, 3) == [False, True, True]
    assert source.index == 2
    assert str(source.dataset_path) == str(path)
    assert all(type(flag) is bool for flag in expand_token_mask(data, 3))
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: expand_token_mask(store["2"][0], 3), range(8)))
    assert outputs == [[False, True, True]] * 8


@pytest.mark.parametrize("record_count", [2, 1000])
def test_constructor_reads_are_bounded_independently_of_record_count(
    tmp_path, monkeypatch, record_count
):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 1)]] * record_count)
    sizes = []
    original = ByteRangeStore.read_range

    def counted(store, key, offset, size):
        sizes.append(size)
        return original(store, key, offset, size)

    monkeypatch.setattr(ByteRangeStore, "read_range", counted)
    store = TokenMaskFileStore(path)
    assert len(store) == record_count
    assert sum(sizes) <= 128


@pytest.mark.parametrize("contents", [b"", b"x" * 39, b"x" * 48])
def test_truncated_or_wrong_signature_is_rejected(tmp_path, contents):
    path = tmp_path / "bad.mask"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        TokenMaskFileStore(path)


@pytest.mark.parametrize("field,value", [(1, 99), (2, 99), (3, 99), (1, 2**64 - 1), (2, 2**64 - 1)])
def test_inconsistent_layout_is_rejected(tmp_path, field, value):
    path = tmp_path / "bad.mask"
    _write_mask(path, [[(2, 1)]])
    raw = bytearray(path.read_bytes())
    # Footer slots: reserved, record count, run count, byte index offset.
    struct.pack_into("<Q", raw, len(raw) - 32 + field * 8, value)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="layout"):
        TokenMaskFileStore(path)


@pytest.mark.parametrize(
    "key,error",
    [
        (key, ValueError)
        for key in ("name", "1.5", "", "1.tokens", "1.mask.extra", "+1", " 1", "1\n", "١", 1, None)
    ]
    + [("-1", IndexError), ("9", IndexError)],
)
def test_invalid_record_key_is_rejected(tmp_path, key, error):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 1)]])
    with pytest.raises(error):
        TokenMaskFileStore(path)[key]


@pytest.mark.parametrize(
    "count,flag,reserved",
    [(0, 1, b"\0\0\0"), (-1, 1, b"\0\0\0"), (1, 2, b"\0\0\0"), (1, 1, b"bad")],
)
def test_invalid_run_rows_are_rejected_on_lookup_and_expansion(tmp_path, count, flag, reserved):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 0)], [(1, 1)]])
    row = struct.pack("<iB3s", count, flag, reserved)
    raw = bytearray(path.read_bytes())
    raw[8:16] = row
    path.write_bytes(raw)
    store = TokenMaskFileStore(path)
    assert expand_token_mask(store["0"][0], 1) == [False]
    with pytest.raises(ValueError):
        store["1"]
    with pytest.raises(ValueError):
        expand_token_mask(row, 1)


@pytest.mark.parametrize("count", [-1, True, 1.0, None, "2", 3])
def test_expansion_requires_exact_nonnegative_integer_source_count(count):
    with pytest.raises(ValueError):
        expand_token_mask(struct.pack("<iB3s", 2, 1, b"\0\0\0"), count)


def test_partial_rows_are_rejected():
    with pytest.raises(ValueError):
        expand_token_mask(b"short", 1)


def test_empty_store_has_no_records(tmp_path):
    path = tmp_path / "empty.mask"
    _write_mask(path, [])
    store = TokenMaskFileStore(path)
    assert len(store) == store.document_count == 0
    with pytest.raises(IndexError):
        store["0.mask"]


def test_decimal_aliases_and_get_attach_original_source(tmp_path):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 0)], [(2, 1)]])
    store = TokenMaskFileStore(path)
    expected = store["1.mask"]
    for key in ("1", "01", "01.mask", "0001.mask"):
        assert store[key] == expected
    sample = {}
    assert store.get("01.mask", sample) == expected[0]
    (source,) = sample["__sources__"]
    assert source.dataset_path == EPath(path)
    assert source.index == 1
    assert source.shard_name == path.name
    assert source.file_names == ("1.mask",)
    assert store.get_path() == str(path)


@pytest.mark.parametrize("reserved", [0, 1, 2, 3, 1 << 63, 2**64 - 1])
def test_reserved_footer_values_do_not_change_records(tmp_path, reserved):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(2, 0)], [(1, 1)]], reserved=reserved)
    store = TokenMaskFileStore(path)
    assert len(store) == 2
    assert expand_token_mask(store["0"][0], 2) == [False, False]
    assert expand_token_mask(store["1"][0], 1) == [True]


@pytest.mark.parametrize("index,value", [(0, 1), (3, 2), (3, -1)])
def test_invalid_index_endpoints_fail_at_construction(tmp_path, index, value):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 0)], [(1, 1)], [(1, 0)]])
    raw = bytearray(path.read_bytes())
    struct.pack_into("<q", raw, 3 * 8 + index * 8, value)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="index endpoints"):
        TokenMaskFileStore(path)


@pytest.mark.parametrize("start,end", [(-1, 2), (4, 2), (2, 1)])
def test_invalid_interior_index_pairs_fail_on_access(tmp_path, start, end):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(1, 0)], [(1, 1)], [(1, 0)]])
    raw = bytearray(path.read_bytes())
    struct.pack_into("<qq", raw, 3 * 8 + 8, start, end)
    path.write_bytes(raw)
    store = TokenMaskFileStore(path)
    assert len(store) == 3
    with pytest.raises(ValueError, match="index pair"):
        store["1"]


@pytest.mark.parametrize("extra", [False, True])
def test_missing_or_extra_trailing_byte_is_rejected(tmp_path, extra):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(2, 1)]])
    raw = path.read_bytes()
    path.write_bytes(raw + b"x" if extra else raw[:-1])
    with pytest.raises(ValueError):
        TokenMaskFileStore(path)


@pytest.mark.parametrize("count", [2, np.int64(2)])
def test_integer_source_counts_expand_to_builtin_booleans(count):
    flags = expand_token_mask(struct.pack("<iB3s", 2, 1, b"\0\0\0"), count)
    assert flags == [True, True]
    assert all(type(flag) is bool for flag in flags)


@pytest.mark.parametrize("count", [False, True])
def test_boolean_source_counts_are_rejected_even_when_lengths_match(count):
    raw = struct.pack("<iB3s", 1, 1, b"\0\0\0") if count else b""
    with pytest.raises(ValueError, match="nonnegative integer"):
        expand_token_mask(raw, count)


def test_randomized_records_match_independent_token_flags(tmp_path):
    rng = random.Random(17)
    expected = [[], [False], [True] * 64]
    expected += [[bool(rng.getrandbits(1)) for _ in range(rng.randrange(128))] for _ in range(40)]
    records = [
        [(sum(1 for _ in group), int(flag)) for flag, group in groupby(flags)] for flags in expected
    ]
    path = tmp_path / "tokens.mask"
    _write_mask(path, records)
    store = TokenMaskFileStore(path)
    assert len(store) == len(expected)
    for index, flags in enumerate(expected):
        assert expand_token_mask(store[str(index)][0], len(flags)) == flags


def test_underlying_reads_stay_bounded_for_sparse_large_index(tmp_path, monkeypatch):
    path = tmp_path / "sparse.mask"
    runs, records = 100_000, 200_000
    with path.open("wb") as stream:
        stream.write(struct.pack("<iB3s", 2, 1, b"\0\0\0"))
        stream.seek(runs * 8)
        stream.write(struct.pack("<qq", 0, 1))
        stream.seek(runs * 8 + records * 8)
        stream.write(struct.pack("<q8sQQQQ", runs, b"MASKID\0\1", 37, records, runs, runs * 8))
    reads = []
    original_open = EPath.open

    class TrackedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def seek(self, offset):
            return self.stream.seek(offset)

        def read(self, size=-1):
            assert 0 <= size <= 64, f"unbounded mask read: {size}"
            reads.append((self.stream.tell(), size))
            return self.stream.read(size)

    def tracked_open(path, *args, **kwargs):
        return TrackedFile(original_open(path, *args, **kwargs))

    monkeypatch.setattr(EPath, "open", tracked_open)
    store = TokenMaskFileStore(path)
    assert sum(size for _, size in reads) <= 64
    reads.clear()
    assert expand_token_mask(store["0"][0], 2) == [True, True]
    assert sum(size for _, size in reads) <= 24
    assert all(offset < 8 or runs * 8 <= offset < runs * 8 + 16 for offset, _ in reads)


def _read_inherited_mask(store, connection):
    connection.send(expand_token_mask(store["1"][0], 3))
    connection.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_inherited_store_reads_in_forked_child(tmp_path):
    path = tmp_path / "tokens.mask"
    _write_mask(path, [[(2, 1)], [(1, 0), (2, 1)]])
    store = TokenMaskFileStore(path)
    assert expand_token_mask(store["0"][0], 2) == [True, True]
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_read_inherited_mask, args=(store, child))
    process.start()
    child.close()
    try:
        assert parent.poll(15), "child did not return its mask record"
        assert parent.recv() == [False, True, True]
        process.join(15)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(15)
