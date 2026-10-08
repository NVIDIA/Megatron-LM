# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests of the v3 codec and replay, without Megatron/CUDA initialization.

Run with ``uv run --no-project --with torch --with numpy --with zstandard
--with pytest python -m pytest tests/offline_kd_cpu``. Load the deliberately
runtime-independent modules under a private namespace to avoid importing the
Megatron package's GPU initialization code.
"""

import argparse
import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_namespace = "offline_kd_cpu"
_package = types.ModuleType(_namespace)
_package.__path__ = [str(Path(__file__).resolve().parents[2] / "megatron/training/distillation")]
sys.modules[_namespace] = _package
codec = importlib.import_module(f"{_namespace}.v3_format")
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


def write_cache(
    root, dp=2, mbs=2, gbs=8, iterations=4, bundle=2, packed=False, extra_metadata=None
):
    storage = Storage(str(root))
    metadata = {
        "format_version": 3,
        "generation": "test-generation",
        "dp_size_save": dp,
        "mbs_save": mbs,
        "gbs_save": gbs,
        "first_sample": 0,
    }
    metadata.update(extra_metadata or {})
    storage.write_json(storage_module.CACHE_FILE, metadata)
    for begin in range(0, iterations, bundle):
        finish = min(begin + bundle, iterations)
        for rank in range(dp):
            records = []
            for iteration in range(begin, finish):
                ids = [
                    iteration * gbs + (mb * dp + rank) * mbs + j
                    for mb in range(gbs // (dp * mbs))
                    for j in range(mbs)
                ]
                captured = codec.capture_batch(inputs(ids, packed=packed))
                target_ids = captured["tokens"].to(torch.int64).reshape(-1, 1)
                targets = codec.pack_targets(target_ids.float(), target_ids)
                record_id = f'{metadata["generation"]}:dp{rank}:{iteration*gbs}-{(iteration+1)*gbs}'
                captured["record_id"] = targets["record_id"] = record_id
                captured["sample_ids"] = ids
                records.append(
                    {
                        "start": iteration * gbs,
                        "end": (iteration + 1) * gbs,
                        "sample_ids": ids,
                        "record_id": record_id,
                        "inputs": codec.encode(captured),
                        "targets": codec.encode(targets),
                    }
                )
            storage_module.write_shard(
                storage, f"dp{rank}__{begin*gbs}-{finish*gbs}.tar", metadata, records
            )
    return storage, metadata


def reader(storage, metadata, **kwargs):
    return replay.ReplayReader(
        storage, metadata, replay.discover_groups(storage, metadata), **kwargs
    )


def test_compact_inputs_preserve_labels_masks_and_positions():
    source = inputs([0, 1], packed=True)
    source["labels"][0, 0] = -100
    source["cu_seqlens"] = torch.tensor([[0, 4, 8, 8], [0, 2, 8, 8]])
    captured = codec.capture_batch(source)
    assert captured["tokens"].dtype == torch.int32
    assert captured["loss_mask"].dtype == torch.bool
    raw = codec.encode(captured)
    restored = codec.decode(raw, codec.digest(raw))
    codec.validate_inputs(restored)
    assert restored["labels"][0] == -100
    assert restored["cu_seqlens"][1].tolist() == [0, 2, 8]
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


def test_17_bit_indices_and_checksums():
    indices = torch.tensor([[0, 65535, 65536, 131071, -1]])
    values = torch.tensor([[0.0, -1.0, -2.0, -3.0, -1000.0]])
    encoded = codec.pack_targets(values, indices)
    restored_values, restored_indices = codec.unpack_targets(encoded)
    assert torch.equal(restored_values, values)
    assert torch.equal(restored_indices[:, :4], indices[:, :4])
    with pytest.raises(ValueError, match="vocabulary"):
        codec.pack_targets(values, indices + (1 << 17))
    raw = codec.encode(encoded)
    with pytest.raises(ValueError, match="checksum"):
        codec.decode(raw[:-1] + bytes([raw[-1] ^ 1]), codec.digest(raw))


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
        for _ in range(2 * gbs // (student_dp * student_mbs)):
            batch = next(iterator)
            assert torch.equal(batch["tokens"], batch["_kd_indices"].squeeze(-1))
            assert batch["tokens"].shape == (student_mbs, 4)
            all_ids.extend(batch["_kd_sample_ids"])
    assert sorted(all_ids) == list(range(24))


def test_rampup_changes_iteration_size_without_skips(tmp_path):
    storage, metadata = write_cache(tmp_path)
    count = [1]
    iterator = replay.ReplayIterator(
        reader(storage, metadata),
        consumed=0,
        dp_rank=0,
        dp_size=1,
        micro_batch_size=2,
        seq_length=8,
        num_microbatches=lambda: count[0],
    )
    assert next(iterator)["_kd_sample_ids"] == [0, 1]
    count[0] = 3
    assert [next(iterator)["_kd_sample_ids"] for _ in range(3)] == [[2, 3], [4, 5], [6, 7]]
    assert iterator.consumed == 8
    resumed = replay.ReplayIterator(
        reader(storage, metadata),
        consumed=8,
        dp_rank=0,
        dp_size=1,
        micro_batch_size=2,
        seq_length=8,
        num_microbatches=lambda: 3,
    )
    assert next(resumed)["_kd_sample_ids"] == [8, 9]


def test_shuffle_is_deterministic_and_group_granular(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=8)
    first = reader(storage, metadata, shuffle=True, seed=7)
    second = reader(storage, metadata, shuffle=True, seed=7)
    assert first.groups == second.groups
    assert first.groups != replay.discover_groups(storage, metadata)
    ids = [s["sample_id"] for s in first.samples(list(range(64)), 64)]
    assert sorted(ids) == list(range(64))
    for start in range(0, 64, 16):
        assert ids[start : start + 16] == list(range(ids[start], ids[start] + 16))


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


def test_missing_dp_group_is_not_published(tmp_path):
    storage, metadata = write_cache(tmp_path)
    (tmp_path / "dp1__0-16.tar.ready.json").unlink()
    assert replay.discover_groups(storage, metadata) == []
    with pytest.raises(RuntimeError, match="exhausted"):
        reader(storage, metadata).samples([0], 1)


def test_sequential_refresh_discovers_new_groups(tmp_path, monkeypatch):
    storage, metadata = write_cache(tmp_path, iterations=2)
    live = replay.ReplayReader(storage, metadata, [])
    assert live.samples([0], 1)[0]["sample_id"] == 0
    write_cache(tmp_path, iterations=4)
    assert live.samples([16], 17)[0]["sample_id"] == 16
    assert live.total_samples == 32


@pytest.mark.parametrize("incomplete", [False, True])
def test_sequential_exhaustion_refreshes_once(tmp_path, monkeypatch, incomplete):
    storage, metadata = write_cache(tmp_path, iterations=2)
    live = reader(storage, metadata)
    if incomplete:
        write_cache(tmp_path, iterations=4)
        (tmp_path / "dp1__16-32.tar.ready.json").unlink()
    calls = []
    discover = replay.discover_groups

    def counted(*args):
        calls.append(args)
        return discover(*args)

    monkeypatch.setattr(replay, "discover_groups", counted)
    live.ensure_available(16)
    assert not calls
    with pytest.raises(RuntimeError, match="exhausted at 16 samples; requested 17"):
        live.ensure_available(17)
    assert len(calls) == 1


def test_shuffle_exhaustion_does_not_extend_cache(tmp_path, monkeypatch):
    storage, metadata = write_cache(tmp_path, iterations=2)
    shuffled = reader(storage, metadata, shuffle=True)
    write_cache(tmp_path, iterations=4)

    def unexpected(*args):
        raise AssertionError("Shuffled loading must retain its original groups")

    monkeypatch.setattr(replay, "discover_groups", unexpected)
    with pytest.raises(RuntimeError, match="exhausted"):
        shuffled.ensure_available(17)


def test_sequential_refresh_rejects_changed_prefix(tmp_path):
    storage, metadata = write_cache(tmp_path, iterations=2)
    live = reader(storage, metadata)
    (tmp_path / "dp1__0-16.tar.ready.json").unlink()
    with pytest.raises(RuntimeError, match="Previously published"):
        live.ensure_available(17)


def test_published_shards_are_immutable_and_input_only_reads_skip_targets(tmp_path):
    storage, metadata = write_cache(tmp_path)
    group = replay.discover_groups(storage, metadata)[0]
    members = storage_module.read_members(storage, group[0], False)
    assert all(set(value) == {"inputs"} for value in members.values())
    descriptor = group[0]
    records = []
    for r in descriptor["records"]:
        records.append(
            {
                "start": r["start"],
                "end": r["end"],
                "sample_ids": r["sample_ids"],
                "record_id": r["record_id"],
                "inputs": b"replacement",
                "targets": b"replacement",
            }
        )
    with pytest.raises(RuntimeError, match="replace published"):
        storage_module.write_shard(storage, descriptor["tar"], metadata, records)


def test_packed_shortening_rejected_and_collation_preserves_boundaries(tmp_path):
    storage, metadata = write_cache(tmp_path, packed=True)
    samples = reader(storage, metadata).samples([0, 1], 2)
    with pytest.raises(ValueError, match="prefix shortening"):
        replay.collate_samples(samples, 4)
    result = replay.collate_samples(samples, 8)
    assert result["cu_seqlens"].tolist() == [[0, 4, 8], [0, 4, 8]]


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
        local_values, local_indices = replay.map_targets(values, indices, mapping)
        assert torch.equal(local_values.long(), local_indices)
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


def test_corrupt_sample_offsets_are_rejected():
    captured = codec.capture_batch(inputs([0]))
    captured["sample_offsets"][-1] = 100
    with pytest.raises(ValueError, match="tensor"):
        codec.validate_inputs(captured)


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
        logits_load_kd_loss_alpha=0.9,
        padded_vocab_size=3,
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
        is_pipeline_last_stage=lambda: True,
    )
    calculator = types.ModuleType("megatron.core.num_microbatches_calculator")
    calculator.get_num_microbatches = lambda: 2
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
    runtime = importlib.import_module(f"{_namespace}.v3_runtime")
    runtime._PLAN = None
    yield args, runtime, legacy_saver
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
    assert not hasattr(args, "logits_load_shuffle_seed")
    with pytest.raises(SystemExit):
        parser.parse_args(["--logits-load-shuffle-shards"])


@pytest.mark.parametrize("seed", [0, 7])
def test_shuffle_seed_requires_replay(runtime_environment, seed):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = seed
    args.logits_load_inputs = False
    with pytest.raises(ValueError, match="requires --logits-load-inputs"):
        runtime.validate_options(args)


def replay_cache(tmp_path, args, runtime, *, completed=True, generation="test-generation"):
    """Create a runtime-compatible cache, optionally publishing completion."""
    extra = {
        "generation": generation,
        "tokenizer": runtime.tokenizer_identity(),
        "seq_length": 8,
        "sft": False,
        "inter_document_masking": False,
        "reset_attention_mask": False,
        "padded_vocab_size": 3,
    }
    storage, metadata = write_cache(tmp_path, extra_metadata=extra)
    if completed:
        storage.write_json(
            storage_module.COMPLETE_FILE, {"generation": metadata["generation"], "end_sample": 32}
        )
    args.logits_load_dir = str(tmp_path)
    return storage, metadata


def test_loader_enables_shuffling_with_seed_zero(tmp_path, runtime_environment, monkeypatch):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = 0
    storage, metadata = replay_cache(tmp_path, args, runtime)
    write_cache(tmp_path, iterations=8, extra_metadata=metadata)
    storage.write_json(
        storage_module.COMPLETE_FILE, {"generation": metadata["generation"], "end_sample": 64}
    )
    runtime.initialize_replay(args)
    mpu = sys.modules["megatron.core"].parallel_state
    monkeypatch.setattr(mpu, "is_pipeline_first_stage", lambda: True, raising=False)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda: 1, raising=False)
    loader = runtime.build_replay_loader(args, consumed=0)
    iterator = iter(loader.dataset)
    expected = reader(storage, metadata, shuffle=True, seed=0)
    try:
        assert iterator.reader.groups == expected.groups
        assert iterator.reader.groups != replay.discover_groups(storage, metadata)
        assert next(iterator)["_kd_sample_ids"] == [
            s["sample_id"] for s in expected.samples([0, 1], 4)
        ]
    finally:
        iterator.close()


def test_completed_cache_resume_uses_seed_and_standard_sample_cursor(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    storage, metadata = replay_cache(tmp_path, args, runtime)
    before = set(storage.list("*"))
    runtime.initialize_replay(args)
    expected = [
        s["sample_id"] for s in reader(storage, metadata, shuffle=True, seed=7).samples([8, 9], 10)
    ]
    args.consumed_train_samples = 8
    runtime._PLAN = None
    runtime.initialize_replay(args)
    iterator = replay.ReplayIterator(
        reader(storage, metadata, shuffle=True, seed=7),
        consumed=args.consumed_train_samples,
        dp_rank=0,
        dp_size=1,
        micro_batch_size=2,
        seq_length=8,
        num_microbatches=lambda: 1,
    )
    assert next(iterator)["_kd_sample_ids"] == expected
    assert not hasattr(args, "offline_kd_cache_generation")
    assert not hasattr(args, "offline_kd_replay_state")
    assert set(storage.list("*")) == before  # Students need only read access to cache storage.


@pytest.mark.parametrize("seed", [0, 7])
def test_shuffling_requires_completed_cache(tmp_path, runtime_environment, seed):
    args, runtime, _ = runtime_environment
    replay_cache(tmp_path, args, runtime, completed=False)
    args.logits_load_shuffle_shards = seed
    with pytest.raises(RuntimeError, match="requires a completed cache"):
        runtime.initialize_replay(args)


@pytest.mark.parametrize("failure", ["generation", "range", "missing_dp"])
def test_completion_marker_must_match_published_cache(tmp_path, runtime_environment, failure):
    args, runtime, _ = runtime_environment
    storage, metadata = replay_cache(tmp_path, args, runtime)
    complete = storage.read_json(storage_module.COMPLETE_FILE)
    if failure == "generation":
        complete["generation"] = "wrong-generation"
    elif failure == "range":
        complete["end_sample"] += 8
    else:
        (tmp_path / "dp1__16-32.tar.ready.json").unlink()
    storage.write_json(storage_module.COMPLETE_FILE, complete)
    with pytest.raises(RuntimeError, match="completion marker disagrees"):
        runtime.initialize_replay(args)


def test_sequential_resume_discovers_new_groups_without_checkpoint_manifest(
    tmp_path, runtime_environment
):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = None
    storage, metadata = replay_cache(tmp_path, args, runtime, completed=False)
    runtime.initialize_replay(args)
    args.consumed_train_samples = 16
    write_cache(tmp_path, iterations=6, extra_metadata=metadata)
    runtime._PLAN = None
    runtime.initialize_replay(args)
    assert reader(storage, metadata).total_samples == 48
    assert reader(storage, metadata).samples([16], 17)[0]["sample_id"] == 16
    assert not hasattr(args, "offline_kd_replay_state")


def test_resume_validates_sample_cursor(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    replay_cache(tmp_path, args, runtime)
    runtime.initialize_replay(args)
    args.consumed_train_samples = 40
    runtime._PLAN = None
    with pytest.raises(RuntimeError, match="cursor exceeds"):
        runtime.initialize_replay(args)


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_can_switch_to_replacement_dump(tmp_path, runtime_environment, shuffle):
    args, runtime, _ = runtime_environment
    args.logits_load_shuffle_shards = 7 if shuffle else None
    original, metadata = replay_cache(tmp_path / "original", args, runtime)
    runtime.initialize_replay(args)
    expected = [
        s["sample_id"]
        for s in reader(original, metadata, shuffle=shuffle, seed=7).samples([8, 9], 10)
    ]
    args.consumed_train_samples = 8
    replacement, replacement_metadata = replay_cache(
        tmp_path / "replacement", args, runtime, generation="replacement-generation"
    )
    runtime._PLAN = None
    runtime.initialize_replay(args)
    assert runtime._PLAN["metadata"]["generation"] != metadata["generation"]
    iterator = replay.ReplayIterator(
        reader(replacement, replacement_metadata, shuffle=shuffle, seed=7),
        consumed=args.consumed_train_samples,
        dp_rank=0,
        dp_size=1,
        micro_batch_size=2,
        seq_length=8,
        num_microbatches=lambda: 1,
    )
    batch = next(iterator)
    assert batch["_kd_sample_ids"] == expected
    assert torch.equal(batch["tokens"], batch["_kd_indices"].squeeze(-1))
    assert not hasattr(args, "offline_kd_cache_generation")


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


def test_incomplete_groups_can_be_regenerated_without_replacing_published_data(tmp_path):
    storage, metadata = write_cache(tmp_path)
    (tmp_path / "dp1__16-32.tar.ready.json").unlink()
    immutable = (tmp_path / "dp0__0-16.tar.ready.json").read_bytes()
    storage_module.quarantine_unpublished(storage, 16)
    assert len(storage.list("*.ready.json")) == 2
    assert len(storage.list("*.aborted.*")) == 2
    assert (tmp_path / "dp0__0-16.tar.ready.json").read_bytes() == immutable
    write_cache(tmp_path)
    assert reader(storage, metadata).total_samples == 32


def test_teacher_metadata_restore_checks_publication_cursor(
    tmp_path, runtime_environment, monkeypatch
):
    args, runtime, _ = runtime_environment
    args.global_batch_size = 8
    args.load = "original-weights"
    args.iteration = 0
    args.logits_save_dir = str(tmp_path)
    utils = types.ModuleType(f"{_namespace}.utils")
    utils._broadcast_without_pp = lambda factory: factory()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    legacy_metadata = {"saver": {"k": 1}, "identifiers": {"seed": 7}}

    def initialize():
        saver = types.SimpleNamespace(
            dp_size=2, cp_size=1, metadata_dict=legacy_metadata, save_dir=str(tmp_path)
        )
        return runtime.initialize_dump_metadata(saver)

    metadata = initialize()
    assert metadata["published_through"] == 0
    metadata.pop("published_through")
    extra = {
        key: value
        for key, value in metadata.items()
        if key not in ("dp_size_save", "mbs_save", "gbs_save", "first_sample")
    }
    write_cache(tmp_path, extra_metadata=extra)
    storage_module.Storage(str(tmp_path)).write_json(
        storage_module.COMPLETE_FILE, {"generation": metadata["generation"], "end_sample": 32}
    )
    args.load = "resumed-training-checkpoint"
    args.consumed_train_samples = 16
    args.iteration = 2
    restored = initialize()
    assert restored["teacher_checkpoint"] == "original-weights"
    assert restored["published_through"] == 32
    assert not storage_module.Storage(str(tmp_path)).exists(storage_module.COMPLETE_FILE)
    args.consumed_train_samples = 40
    with pytest.raises(RuntimeError, match="outside the published"):
        initialize()


def test_teacher_attempt_commit_discards_rerun_data(runtime_environment, monkeypatch):
    args, runtime, legacy = runtime_environment
    module = importlib.import_module(f"{_namespace}.v3_saver")
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "get_num_microbatches", lambda: 2)
    saver = module.PairedLogitsSaver.__new__(module.PairedLogitsSaver)
    saver.tp_rank = saver.cp_rank = saver.dp_rank = 0
    saver.cp_size = saver.dp_size = 1
    saver._initialized = True
    saver.metadata_dict = {"generation": "test-generation", "published_through": 0}
    saver._captured = []
    saver._pending_writes = replay.OrderedDict()
    saver._process_single_microbatch = lambda logits: (logits.half(), logits.long())

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
    restored = codec.decode(record["inputs"], codec.digest(record["inputs"]))
    values, indices = codec.unpack_targets(
        codec.decode(record["targets"], codec.digest(record["targets"]))
    )
    assert restored["tokens"][0] == 400
    assert torch.equal(restored["tokens"].long(), indices.squeeze(-1))
    assert torch.equal(values.long(), indices)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_paired_loss_uses_explicit_targets_and_keeps_gradients(
    runtime_environment, monkeypatch, dtype
):
    args, runtime, legacy = runtime_environment
    # Execute the real legacy sparse KL function without importing GPU startup.
    path = Path(_package.__path__[0]) / "cached_logits_loss.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "topk_kl_div"
    )
    namespace = {"torch": torch, "dist": torch.distributed, "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cached = types.ModuleType(f"{_namespace}.cached_logits_loss")
    cached.topk_kl_div = namespace["topk_kl_div"]
    logits = torch.randn(8, 2, 3, dtype=dtype, requires_grad=True)
    cached.get_student_logits_capture = lambda: types.SimpleNamespace(pop=lambda: logits)
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    module = importlib.import_module(f"{_namespace}.v3_loss")
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **kw: None)
    teacher = torch.randn(8, 2, 3).log_softmax(-1)
    sidecar = runtime.bind_student_logits(
        {"values": teacher, "indices": torch.arange(3).expand(8, 2, -1)}
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
    assert "logits distillation loss" in report


def legacy_saver_methods():
    """Execute the inherited TP/top-P helpers without Megatron GPU imports."""
    path = Path(_package.__path__[0]) / "logits_saver.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    names = {"_process_single_microbatch", "_compute_global_topk", "_apply_topp_truncation"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "torch": torch,
        "dist": torch.distributed,
        "_MAX_VOCAB_SIZE": 1 << 17,
        "CACHED_LOGITS_INDEX_SENTINEL": -1,
        "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
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
        k=4,
        p=0.7,
        min_k=2,
        _save_dtype=dtype,
        _topp_kept_counts=[],
    )
    for name in ("_compute_global_topk", "_apply_topp_truncation"):
        setattr(saver, name, types.MethodType(helpers[name], saver))
    values, indices = helpers["_process_single_microbatch"](saver, chunks[0])
    assert calls == [torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.SUM]
    expected, expected_ids = full.log_softmax(-1).topk(4, dim=-1)
    probs = expected.exp()
    keep = (probs.cumsum(-1) - probs < 0.7) | (torch.arange(4) < 2)
    assert torch.equal(indices, torch.where(keep, expected_ids, -1))
    assert torch.allclose(values, torch.where(keep, expected, -1e3).to(dtype), atol=1e-6)
    encoded = codec.pack_targets(values.reshape(-1, 4), indices.reshape(-1, 4))
    raw = codec.encode(encoded)
    restored, restored_ids = codec.unpack_targets(codec.decode(raw, codec.digest(raw)))
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


def test_remote_refresh_lists_once_and_broadcasts_errors(runtime_environment, monkeypatch):
    _, runtime, _ = runtime_environment
    mpu = sys.modules["megatron.core"].parallel_state
    group = object()
    mpu.get_data_parallel_group = lambda **kw: group
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_process_group_ranks", lambda g: [3, 5])
    rank = [3]
    monkeypatch.setattr(runtime.dist, "get_rank", lambda: rank[0])
    payload = []

    def broadcast(result, src, group):
        assert src == 3
        if rank[0] == src:
            payload[:] = result
        else:
            result[:] = payload

    monkeypatch.setattr(runtime.dist, "broadcast_object_list", broadcast)
    listings = []
    monkeypatch.setattr(
        runtime, "discover_groups", lambda *a: listings.append(a) or [[{"test": True}]]
    )
    storage = types.SimpleNamespace(remote=True)
    assert runtime.refresh_replay_groups(storage, {}) == [[{"test": True}]]
    rank[0] = 5
    assert runtime.refresh_replay_groups(storage, {}) == [[{"test": True}]]
    assert len(listings) == 1
    rank[0] = 3

    def broken(*args):
        raise ValueError("broken descriptor")

    monkeypatch.setattr(runtime, "discover_groups", broken)
    with pytest.raises(RuntimeError, match="broken descriptor"):
        runtime.refresh_replay_groups(storage, {})
    rank[0] = 5
    with pytest.raises(RuntimeError, match="broken descriptor"):
        runtime.refresh_replay_groups(storage, {})


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
    backend.glob = lambda pattern: calls.append(pattern) or [
        "msc://test/dump/dp0__0-8.tar.ready.json"
    ]
    module = importlib.import_module(f"{_namespace}.v3_saver")
    monkeypatch.setattr(module, "write_shard", lambda *a: None)
    module.PairedLogitsSaver._write_batched_tar(
        "msc://test/dump/dp0__0-8.tar", {0: {}}, "{}", msc_enabled=True
    )
    assert state["enabled"]
    storage = Storage("msc://test/dump")
    assert storage.list("*.ready.json") == ["dp0__0-8.tar.ready.json"]
    storage.list("*.ready.json")
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
    full_values = full_ids.float()
    mapped = [replay.map_targets(full_values, full_ids, m) for m in maps]
    parts = [maps, [v for v, _ in mapped], [i for _, i in mapped]]
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
    saver.metadata_dict = {"generation": "test"}
    saver._captured = [(captured, maps[0], *mapped[0])]
    saver._serialize_attempt()
    raw = saver._iteration_records["targets"]
    values, indices = codec.unpack_targets(codec.decode(raw, codec.digest(raw)))
    assert len(calls) == 3
    assert torch.equal(values, full_values)
    assert torch.equal(indices, full_ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
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
    assert received["sample_ids"] == [2, 3]
    assert len(pinned) == 2


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
        values, indices = replay.map_targets(full.float(), full, mapping)
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
    namespace = {"torch": torch, "dist": torch.distributed, "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    # -1 encodes as 131071, which is a real vocabulary ID at the format limit.
    values, indices = codec.unpack_targets(
        codec.pack_targets(
            torch.tensor([[0.7, 0.0]]).log().clamp_min(-1e3), torch.tensor([[0, -1]])
        )
    )
    assert indices[0, 1] == 131071
    logits = torch.linspace(-2, 2, 1 << 17).reshape(1, 1, -1).requires_grad_()
    actual = namespace["topk_kl_div"](
        logits.clone(), values.unsqueeze(0), indices.unsqueeze(0), 1, 0, None, add_ghost_token=True
    )
    logprobs = logits.log_softmax(-1)
    p = logprobs[..., 0].exp()
    expected = 0.7 * (torch.tensor(0.7).log() - logprobs[..., 0]) + 0.3 * (
        torch.tensor(0.3).log() - (1 - p).log()
    )
    assert torch.allclose(actual, expected, atol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_v3_async_write_failure_sets_shared_event(runtime_environment, monkeypatch, tmp_path):
    module = importlib.import_module(f"{_namespace}.v3_saver")
    failures = []

    def broken(*args):
        raise OSError("failed publication")

    monkeypatch.setattr(module, "write_shard", broken)
    event = types.SimpleNamespace(set=lambda: failures.append(True))
    with pytest.raises(OSError, match="failed publication"):
        module.PairedLogitsSaver._write_batched_tar(
            str(tmp_path / "dp0__0-8.tar"), {0: {}}, "{}", failure_event=event
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
    expected_order = [
        sid
        for group in prototype.groups
        for sid in range(group[0]["records"][0]["start"], group[0]["records"][-1]["end"])
    ]
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
                    assert torch.equal(batch["tokens"], batch["_kd_indices"].squeeze(-1))
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
