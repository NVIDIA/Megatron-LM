# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests of the v3 codec and replay, without Megatron/CUDA initialization.

Run with ``uv run --no-project --with torch --with numpy --with zstandard
--with pytest python -m pytest tests/offline_kd_cpu``. Load the deliberately
runtime-independent modules under a private namespace to avoid importing the
Megatron package's GPU initialization code.
"""

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
batch_module = importlib.import_module(f"{_namespace}.v3_batch")
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
                record_id = f'test-generation:dp{rank}:{iteration*gbs}-{(iteration+1)*gbs}'
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


def test_follow_polls_then_observes_completion(tmp_path, monkeypatch):
    storage, metadata = write_cache(tmp_path, iterations=2)
    live = replay.ReplayReader(storage, metadata, [], follow=True, timeout=1, poll_interval=0.001)
    assert live.samples([0], 1)[0]["sample_id"] == 0
    storage.write_json(
        storage_module.COMPLETE_FILE, {"generation": metadata["generation"], "end_sample": 16}
    )
    with pytest.raises(RuntimeError, match="completed"):
        live.ensure_available(17)
    with pytest.raises(ValueError, match="shuffle"):
        replay.ReplayReader(storage, metadata, [], follow=True, shuffle=True)


def test_follow_timeout(tmp_path):
    storage, metadata = write_cache(tmp_path)
    (tmp_path / "dp1__0-16.tar.ready.json").unlink()
    live = replay.ReplayReader(
        storage, metadata, [], follow=True, timeout=0.001, poll_interval=0.001
    )
    with pytest.raises(TimeoutError):
        live.ensure_available(1)


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
    maps = [batch_module.token_map(captured, rank, 4, layout=layout) for rank in range(4)]
    positions = torch.cat([m.reshape(-1) for m in maps])
    assert sorted(positions.tolist()) == list(range(32))
    for mapping in maps:
        local_values, local_indices = batch_module.map_targets(values, indices, mapping)
        assert torch.equal(local_values.long(), local_indices)
        assert torch.equal(local_indices.squeeze(-1), captured["tokens"][mapping].long())


def test_packed_cp_padding_validation():
    captured = codec.capture_batch(inputs([0], length=8, packed=True))
    captured["cu_seqlens"] = [torch.tensor([0, 2, 8])]
    with pytest.raises(ValueError, match="padding"):
        batch_module.token_map(captured, 0, 2)


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
        mapping = batch_module.token_map(captured, rank, 4, layout=layout, per_sequence=True)
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
        logits_load_follow=False,
        logits_load_shuffle_shards=True,
        logits_load_shuffle_seed=7,
        logits_load_follow_timeout=1.0,
        logits_load_follow_poll_interval=0.001,
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


def test_checkpoint_snapshot_is_read_only_and_resume_ignores_new_groups(
    tmp_path, runtime_environment
):
    args, runtime, _ = runtime_environment
    extra = {
        "tokenizer": runtime.tokenizer_identity(),
        "seq_length": 8,
        "sft": False,
        "inter_document_masking": False,
        "reset_attention_mask": False,
        "padded_vocab_size": 3,
    }
    storage, metadata = write_cache(tmp_path, extra_metadata=extra)
    args.logits_load_dir = str(tmp_path)
    before = set(storage.list("*"))
    runtime.initialize_replay(args)
    args.consumed_train_samples = 8
    runtime.checkpoint_replay_state(args)
    assert set(storage.list("*")) == before  # Students need only read access to cache storage.
    state = args.offline_kd_replay_state
    write_cache(tmp_path, iterations=6, extra_metadata=extra)
    runtime._PLAN = None
    runtime.initialize_replay(args)
    assert runtime._PLAN["groups"] == state["groups"]
    assert (
        sum(len(r["sample_ids"]) for g in runtime._PLAN["groups"] for d in g for r in d["records"])
        == 32
    )
    runtime._PLAN = None
    args.logits_load_shuffle_seed += 1
    with pytest.raises(RuntimeError, match="ordering changed"):
        runtime.initialize_replay(args)


def test_checkpoint_detects_rewritten_publication(tmp_path, runtime_environment):
    args, runtime, _ = runtime_environment
    extra = {
        "tokenizer": runtime.tokenizer_identity(),
        "seq_length": 8,
        "sft": False,
        "inter_document_masking": False,
        "reset_attention_mask": False,
        "padded_vocab_size": 3,
    }
    storage, metadata = write_cache(tmp_path, extra_metadata=extra)
    args.logits_load_dir = str(tmp_path)
    runtime.initialize_replay(args)
    runtime.checkpoint_replay_state(args)
    name = storage.list("*.ready.json")[0]
    descriptor = storage.read_json(name)
    descriptor["records"][0]["inputs"]["sha256"] = "different"
    storage.write_json(name, descriptor)
    runtime._PLAN = None
    with pytest.raises(RuntimeError, match="data changed"):
        runtime.initialize_replay(args)


def test_frozen_progress_requires_every_dp_shard(tmp_path, runtime_environment):
    _, runtime, _ = runtime_environment
    storage, metadata = write_cache(tmp_path, extra_metadata={"first_iteration": 5})
    saver = types.SimpleNamespace(
        metadata_dict={**metadata, "published_through": 0}, save_dir=str(tmp_path)
    )
    assert runtime.durable_dump_iteration(saver, 100) == 9
    (tmp_path / "dp1__16-32.tar.ready.json").unlink()
    assert runtime.durable_dump_iteration(saver, 100) == 7
    assert runtime.durable_dump_iteration(saver, 6) == 6


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
    args.load = "resumed-training-checkpoint"
    args.consumed_train_samples = 16
    args.iteration = 2
    restored = initialize()
    assert restored["teacher_checkpoint"] == "original-weights"
    assert restored["published_through"] == 32
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
