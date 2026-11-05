# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU format and bounded-I/O contracts for prepared-token mask companions."""

import multiprocessing
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from examples.multimodal.data_loading.token_mask import TokenMaskFileStore, expand_token_mask
from megatron.energon.cache.file_store import ByteRangeStore


def _write_mask(path, records):
    runs = bytearray()
    offsets = [0]
    for record in records:
        for count, flag in record:
            runs.extend(struct.pack("<iB3s", count, flag, b"\0\0\0"))
        offsets.append(len(runs) // 8)
    path.write_bytes(
        runs
        + struct.pack(f"<{len(offsets)}q", *offsets)
        + struct.pack("<8sQQQQ", b"MASKID\0\1", 37, len(records), len(runs) // 8, len(runs))
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


@pytest.mark.parametrize("field,value", [(1, 99), (2, 99), (3, 99)])
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
    "key,error", [("name", ValueError), ("1.5", ValueError), ("-1", IndexError), ("9", IndexError)]
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
def test_invalid_run_rows_are_rejected_before_expansion(count, flag, reserved):
    with pytest.raises(ValueError):
        expand_token_mask(struct.pack("<iB3s", count, flag, reserved), 1)


@pytest.mark.parametrize("count", [-1, True, 1.0, None, 3])
def test_expansion_requires_exact_nonnegative_integer_source_count(count):
    with pytest.raises(ValueError):
        expand_token_mask(struct.pack("<iB3s", 2, 1, b"\0\0\0"), count)


def test_partial_rows_are_rejected():
    with pytest.raises(ValueError):
        expand_token_mask(b"short", 1)


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
