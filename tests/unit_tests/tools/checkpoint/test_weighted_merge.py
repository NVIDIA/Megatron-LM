# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as torch_dcp

from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.dist_checkpointing.strategies import filesystem_async
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint import weighted_merge as weighted_merge_module
from tools.checkpoint.weighted_merge import (
    WeightedMergeError,
    checkpoint_coefficients,
    derive_start_iteration_from_token_window,
    ensure_process_group,
    filter_checkpoints_by_interval,
    get_valid_styles,
    iteration_dir_name,
    merge_same_layout_dcp_metadata_checkpoints,
    normalize_weights,
    output_checkpoint_dir,
    parse_weighted_inputs,
    resolve_checkpoint_dir,
    select_checkpoints_in_window,
    validate_min_checkpoints,
    validate_weights,
    write_latest_checkpointed_iteration,
)


@pytest.fixture
def process_group():
    already_initialized = dist.is_available() and dist.is_initialized()
    ensure_process_group()
    yield
    if not already_initialized and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def cpu_only_dcp_save(monkeypatch):
    # GPTModel construction asserts on NVTE_FLASH_ATTN/NVTE_FUSED_ATTN/
    # NVTE_UNFUSED_ATTN when they are set in the env (e.g. =0 in the CI
    # container). Unset them so CPU model fixtures build (mirrors the smoke).
    for _var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(_var, raising=False)
    if torch.cuda.is_available():
        return
    # DCP's mcore async writer synchronizes CUDA even for these CPU-only fixtures.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    if not filesystem_async.HAVE_PSUTIL:
        monkeypatch.setattr(filesystem_async, "_process_memory", lambda: 0)


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank_offsets():
    world_size = _world_size()
    return ((0, _rank(), world_size),) if world_size > 1 else ()


def _template(
    value=0.0,
    *,
    dtype=torch.float32,
    extra_value=0.0,
    include_bias=True,
    shape=(2, 2),
    device=None,
):
    rank_offsets = _rank_offsets()
    model_state_dict = {
        "weight": ShardedTensor.from_rank_offsets(
            "model.weight",
            torch.full(shape, value, dtype=dtype, device=device),
            *rank_offsets,
            replica_id=0,
        ),
        "decoder.layers.0._extra_state": ShardedTensor.from_rank_offsets(
            "model.decoder.layers.0._extra_state",
            torch.tensor([extra_value], dtype=torch.float32, device=device),
            *rank_offsets,
            replica_id=0,
        ),
    }
    if include_bias:
        model_state_dict["bias"] = ShardedTensor.from_rank_offsets(
            "model.bias",
            torch.full((2,), value + 1, dtype=dtype, device=device),
            *rank_offsets,
            replica_id=0,
        )
    return {"model": model_state_dict}


def _write_checkpoint(
    path, value, *, dtype=torch.float32, extra_value=0.0, iteration=0, shape=(2, 2)
):
    state_dict = _template(value, dtype=dtype, extra_value=extra_value, shape=shape)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _generated_gpt_model_state(value):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        add_bias_linear=True,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=16,
        max_sequence_length=8,
        pre_process=True,
        post_process=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return {
        key: sharded
        for key, sharded in model.sharded_state_dict(prefix="model.").items()
        if not key.endswith("._extra_state")
    }


def _write_generated_gpt_checkpoint(path, value):
    dist_checkpointing.save(_generated_gpt_model_state(value), str(path))


def _generated_moe_gpt_model_state(value):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        add_bias_linear=True,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(
            num_experts=2,
            moe_grouped_gemm=False,
        ),
        vocab_size=16,
        max_sequence_length=8,
        pre_process=True,
        post_process=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model.sharded_state_dict(prefix="model.")


def _write_generated_moe_gpt_checkpoint(path, value):
    dist_checkpointing.save(_generated_moe_gpt_model_state(value), str(path))


UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY = "decoder.layers.0.mlp.linear_fc1._extra_state"
UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY = "mtp.layers.0.eh_proj._extra_state"


def _unprefixed_gpt_like_model_state(value, *, rank_sharded=False):
    rank_offsets = _rank_offsets() if rank_sharded else ()
    return {
        "decoder.final_layernorm.weight": ShardedTensor.from_rank_offsets(
            "decoder.final_layernorm.weight",
            torch.full((3,), value, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "decoder.layers.0.mlp.linear_fc1.weight": ShardedTensor.from_rank_offsets(
            "decoder.layers.0.mlp.linear_fc1.weight",
            torch.full((2, 3), value + 1, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "embedding.word_embeddings.weight": ShardedTensor.from_rank_offsets(
            "embedding.word_embeddings.weight",
            torch.full((4, 3), value + 2, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "output_layer.weight": ShardedTensor.from_rank_offsets(
            "output_layer.weight",
            torch.full((4, 3), value + 3, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
    }


def _write_unprefixed_gpt_like_checkpoint(path, value):
    dist_checkpointing.save(_unprefixed_gpt_like_model_state(value), str(path))


def _unprefixed_gpt_like_byte_extra_template(extra_value=0):
    return {
        UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY: ShardedObject(
            UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY,
            _bytesio_state(extra_value),
            (_world_size(),),
            (_rank(),),
            replica_id=0,
        )
    }


def _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
    path, value, extra_value, *, rank_sharded=False
):
    state = _unprefixed_gpt_like_model_state(value, rank_sharded=rank_sharded)
    state[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY] = ShardedObject(
        UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY,
        _bytesio_state(extra_value),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_mtp_state(path, value, extra_value):
    state = _unprefixed_gpt_like_model_state(value)
    state["mtp.layers.0.eh_proj.weight"] = ShardedTensor.from_rank_offsets(
        "mtp.layers.0.eh_proj.weight",
        torch.full((2, 2), value + 4, dtype=torch.float32),
        replica_id=0,
    )
    state[UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY] = ShardedObject(
        UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY,
        _bytesio_state(extra_value),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(path, value):
    state = _unprefixed_gpt_like_model_state(value)
    state["optimizer._extra_state"] = ShardedObject(
        "optimizer._extra_state",
        _bytesio_state(123),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_optimizer_tensor(path, value):
    state = _unprefixed_gpt_like_model_state(value)
    state["optimizer.param_groups.0.lr"] = ShardedTensor.from_rank_offsets(
        "optimizer.param_groups.0.lr",
        torch.tensor([value], dtype=torch.float32),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _split_weight_factory(sharded_tensor):
    sharded_tensor_without_data = sharded_tensor.without_data()
    split_point = sharded_tensor.data.shape[0] // 2
    split_sections = (split_point, sharded_tensor.data.shape[0] - split_point)

    def build(key, tensor, replica_id, flattened_range):
        base = replace(
            sharded_tensor_without_data,
            key=key,
            data=tensor,
            dtype=tensor.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )
        chunks = []
        start = 0
        for name, length in zip(("left", "right"), split_sections):
            chunk = base.narrow(0, start, length)[0]
            chunk.key = f"{chunk.key}.{name}"
            chunks.append(chunk)
            start += length
        return chunks

    return ShardedTensorFactory(
        sharded_tensor.key,
        sharded_tensor.data,
        build,
        lambda chunks: torch.cat(chunks, dim=0),
        sharded_tensor.replica_id,
    )


def _factory_template(value=0.0, *, dtype=torch.float32, shape=(4, 2)):
    rank_offsets = _rank_offsets()
    weight = ShardedTensor.from_rank_offsets(
        "model.factory_weight",
        torch.full(shape, value, dtype=dtype),
        *rank_offsets,
        replica_id=0,
    )
    return {"model": {"weight": _split_weight_factory(weight)}}


def _write_factory_checkpoint(path, value, *, iteration=0, shape=(4, 2)):
    state_dict = _factory_template(value, shape=shape)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _load_checkpoint(path, *, dtype=torch.float32, shape=(2, 2)):
    return dist_checkpointing.load(_template(dtype=dtype, shape=shape), str(path))


def _full_public_dcp_state(path, *, dtype=torch.float32, shape=(2, 2)):
    world_size = _world_size()
    state_dict = {
        "model.weight": torch.empty((shape[0] * world_size, shape[1]), dtype=dtype),
        "model.bias": torch.empty((2 * world_size,), dtype=dtype),
        "model.decoder.layers.0._extra_state": torch.empty((world_size,), dtype=torch.float32),
    }
    torch_dcp.load(state_dict, checkpoint_id=str(path), no_dist=True)
    return state_dict


def _dcp_metadata_summary(path):
    metadata = torch_dcp.FileSystemReader(path).read_metadata()
    chunk_records = []
    for fqn, tensor_metadata in metadata.state_dict_metadata.items():
        for chunk in getattr(tensor_metadata, "chunks", []) or []:
            chunk_records.append(
                (
                    fqn,
                    tuple(int(offset) for offset in chunk.offsets),
                    tuple(int(size) for size in chunk.sizes),
                )
            )

    storage_data = getattr(metadata, "storage_data", {}) or {}
    storage_records = [
        (
            getattr(record, "relative_path", None),
            getattr(record, "offset", None),
            getattr(record, "length", None),
        )
        for record in storage_data.values()
    ]
    duplicate_chunk_offsets = {
        key: count
        for key, count in Counter((fqn, offsets) for fqn, offsets, _ in chunk_records).items()
        if count > 1
    }
    duplicate_storage_records = {
        key: count for key, count in Counter(storage_records).items() if count > 1
    }
    return {
        "chunk_records": chunk_records,
        "duplicate_chunk_offsets": duplicate_chunk_offsets,
        "duplicate_storage_records": duplicate_storage_records,
        "storage_file_count": len({str(record[0]) for record in storage_records}),
    }


def _bytesio_state(value):
    data = io.BytesIO()
    torch.save({"value": value}, data)
    data.seek(0)
    return data


def _decode_sharded_object_value(value):
    if value is None:
        return None
    if not isinstance(value, io.BytesIO):
        return value
    value.seek(0)
    payload = torch.load(value, weights_only=False)
    if isinstance(payload, list):
        assert len(payload) == 1
        payload = payload[0]
    if isinstance(payload, io.BytesIO):
        payload.seek(0)
        return torch.load(payload, weights_only=False)
    return payload


def _object_extra_template(extra_value=0):
    rank_offsets = _rank_offsets()
    return {
        "model": {
            "weight": ShardedTensor.from_rank_offsets(
                "model.weight",
                torch.zeros((2, 2), dtype=torch.float32),
                *rank_offsets,
                replica_id=0,
            ),
            "decoder.layers.0._extra_state": ShardedObject(
                "model.decoder.layers.0._extra_state",
                _bytesio_state(extra_value),
                (_world_size(),),
                (_rank(),),
                replica_id=0,
            ),
        }
    }


def _write_object_extra_checkpoint(path, value, extra_value, iteration):
    state_dict = _object_extra_template(extra_value)
    state_dict["model"]["weight"].data.fill_(value)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _multi_chunk_template(value=0.0, *, dtype=torch.float32):
    rank_offsets = _rank_offsets()
    return {
        "model0": {
            "weight": ShardedTensor.from_rank_offsets(
                "model0.weight", torch.full((2, 2), value, dtype=dtype), *rank_offsets, replica_id=0
            )
        },
        "model1": {
            "weight": ShardedTensor.from_rank_offsets(
                "model1.weight",
                torch.full((2, 2), value + 1, dtype=dtype),
                *rank_offsets,
                replica_id=0,
            )
        },
    }


def _write_multi_chunk_checkpoint(path, value, *, iteration=0):
    state_dict = _multi_chunk_template(value)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


# ---------------------------------------------------------------------------
# Mode-agnostic pure-math / selection / parsing tests.
# ---------------------------------------------------------------------------


def test_linear_coefficients_are_uniform():
    coeffs = checkpoint_coefficients([100, 200, 300, 400], "linear")
    assert list(coeffs) == [100, 200, 300, 400]
    assert all(abs(value - 0.25) < 1e-12 for value in coeffs.values())
    assert abs(sum(coeffs.values()) - 1.0) < 1e-12


def test_minus_sqrt_matches_discrete_difference():
    checkpoints = [10, 20, 30, 40]
    coeffs = checkpoint_coefficients(checkpoints, "minus-sqrt")
    decay = [1 - (index / len(checkpoints)) ** 0.5 for index in range(len(checkpoints))]
    expected = [
        1 - ((decay[1] - decay[2]) + (decay[2] - decay[3]) + decay[3]),
        decay[1] - decay[2],
        decay[2] - decay[3],
        decay[3],
    ]

    assert list(coeffs.values()) == pytest.approx(expected)
    assert abs(sum(coeffs.values()) - 1.0) < 1e-12


@pytest.mark.parametrize("style", get_valid_styles())
@pytest.mark.parametrize("n_checkpoints", [1, 2, 5])
def test_supported_coefficients_are_deterministic_and_normalized(style, n_checkpoints):
    checkpoints = list(range(n_checkpoints))
    coeffs_a = checkpoint_coefficients(checkpoints, style, seed=123)
    coeffs_b = checkpoint_coefficients(checkpoints, style, seed=123)

    assert coeffs_a == coeffs_b
    assert list(coeffs_a) == checkpoints
    assert sum(coeffs_a.values()) == pytest.approx(1.0)


def test_modifiers_are_deterministic():
    checkpoints = [1, 2, 3, 4, 5]
    normal = list(checkpoint_coefficients(checkpoints, "minus-sqrt").values())
    reversed_coeffs = list(checkpoint_coefficients(checkpoints, "minus-sqrt__reverse").values())
    scrambled_a = checkpoint_coefficients(checkpoints, "minus-sqrt__scramble", seed=17)
    scrambled_b = checkpoint_coefficients(checkpoints, "minus-sqrt__scramble", seed=17)

    assert reversed_coeffs == list(reversed(normal))
    assert scrambled_a == scrambled_b
    assert sorted(scrambled_a.values()) == sorted(normal)


def test_parse_weighted_inputs_uses_last_colon_for_weight():
    paths, weights = parse_weighted_inputs(["/tmp/with:colon/iter_0000010:0.75"])
    assert str(paths[0]) == "/tmp/with:colon/iter_0000010"
    assert weights == [0.75]


def test_manual_weight_policy_warnings():
    assert weighted_merge_module._manual_weight_warnings([0.25, 0.75], normalize=False) == []
    assert weighted_merge_module._manual_weight_warnings([2.0, -1.0], normalize=False) == [
        "WARNING: manual merge weights include negative values; this is allowed for "
        "subtractive merges but can produce outputs outside the input checkpoint range."
    ]

    unnormalized = weighted_merge_module._manual_weight_warnings([0.25, 0.25], normalize=False)
    assert len(unnormalized) == 1
    assert "sum to 0.5 without --normalize" in unnormalized[0]

    normalized = weighted_merge_module._manual_weight_warnings([0.25, 0.25], normalize=True)
    assert normalized == []


def test_invalid_pure_inputs_raise(tmp_path):
    with pytest.raises(WeightedMergeError, match="PATH:WEIGHT"):
        parse_weighted_inputs([str(tmp_path / "iter_0000010")])
    with pytest.raises(WeightedMergeError, match="Invalid weight"):
        parse_weighted_inputs([f"{tmp_path / 'iter_0000010'}:not-a-float"])
    with pytest.raises(WeightedMergeError, match="Unknown coefficient schedule"):
        checkpoint_coefficients([1, 2], "unknown")
    with pytest.raises(WeightedMergeError, match="Unknown coefficient modifier"):
        checkpoint_coefficients([1, 2], "linear__unknown")
    with pytest.raises(WeightedMergeError, match="Weight sum"):
        normalize_weights([1.0, -1.0])
    with pytest.raises(WeightedMergeError, match="finite"):
        normalize_weights([1.0, float("nan")])
    with pytest.raises(WeightedMergeError, match="finite"):
        validate_weights([1.0, float("inf")])
    with pytest.raises(WeightedMergeError, match="min_checkpoints"):
        validate_min_checkpoints(2, 0)
    with pytest.raises(WeightedMergeError, match="at least 3"):
        validate_min_checkpoints(2, 3)
    with pytest.raises(WeightedMergeError, match="does not match requested iteration"):
        output_checkpoint_dir(tmp_path / "iter_0000001", 2)
    with pytest.raises(WeightedMergeError, match="not a distributed checkpoint"):
        resolve_checkpoint_dir(tmp_path)


def test_select_checkpoints_preserves_target_and_applies_interval(tmp_path):
    for iteration in [100, 150, 210, 260, 300]:
        (tmp_path / iteration_dir_name(iteration)).mkdir()

    selected = select_checkpoints_in_window(
        tmp_path, start_iteration=100, end_iteration=300, min_iteration_interval=100
    )

    assert selected == [150, 300]


def test_token_window_selection_uses_ceil_and_requires_target(tmp_path):
    for iteration in [0, 10, 20, 30]:
        (tmp_path / iteration_dir_name(iteration)).mkdir()

    start = derive_start_iteration_from_token_window(
        end_iteration=30, token_window_btok=1, seq_length=128, global_batch_size=1_000_000
    )
    selected = select_checkpoints_in_window(
        tmp_path,
        start_iteration=None,
        end_iteration=30,
        token_window_btok=1,
        seq_length=128,
        global_batch_size=1_000_000,
    )

    assert start == 22
    assert selected == [30]
    with pytest.raises(WeightedMergeError, match="Target iteration"):
        select_checkpoints_in_window(tmp_path, start_iteration=0, end_iteration=40)
    with pytest.raises(WeightedMergeError, match="requires seq_length and global_batch_size"):
        select_checkpoints_in_window(
            tmp_path, start_iteration=None, end_iteration=30, token_window_btok=1
        )


def test_filter_checkpoints_by_interval_keeps_last_checkpoint():
    assert filter_checkpoints_by_interval([100, 180, 240, 300], 100) == [180, 300]


# ---------------------------------------------------------------------------
# Metadata-driven same-layout merge tests (the sole merge path).
# ---------------------------------------------------------------------------


def test_metadata_same_layout_merge_round_trip_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    shape = (5, 2)

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=30,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert result.averaged_tensors == 2
        assert result.copied_extra_states == 1
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert torch.equal(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))

        dcp_loaded = {
            "model.weight": torch.empty(shape, dtype=torch.float32),
            "model.bias": torch.empty((2,), dtype=torch.float32),
            "model.decoder.layers.0._extra_state": torch.empty((1,), dtype=torch.float32),
        }
        torch_dcp.load(dcp_loaded, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(dcp_loaded["model.weight"], torch.full(shape, 4.0))
        assert torch.equal(dcp_loaded["model.bias"], torch.full((2,), 5.0))
        assert torch.equal(dcp_loaded["model.decoder.layers.0._extra_state"], torch.tensor([999.0]))

        source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
        output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        for fqn in (
            "model.weight",
            "model.bias",
            "model.decoder.layers.0._extra_state",
        ):
            source_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in source_metadata.state_dict_metadata[fqn].chunks
            ]
            output_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in output_metadata.state_dict_metadata[fqn].chunks
            ]
            assert output_chunks == source_chunks

        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["implementation_mode"] == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert provenance["extra_state_source_index"] == 1


def test_metadata_same_layout_balanced_work_retiles_single_chunk_tensors(
    tmp_path_dist_ckpt, process_group
):
    if _world_size() != 1:
        pytest.skip("balanced virtual tiling coverage uses single-rank fixture metadata")

    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_balanced_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_balanced_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_balanced_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=30,
            extra_state_source_index=1,
            balanced_work=True,
            target_chunk_bytes=16,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert torch.equal(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))

        source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
        output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        source_weight_chunks = [
            (tuple(chunk.offsets), tuple(chunk.sizes))
            for chunk in source_metadata.state_dict_metadata["model.weight"].chunks
        ]
        output_weight_chunks = sorted(
            (tuple(chunk.offsets), tuple(chunk.sizes))
            for chunk in output_metadata.state_dict_metadata["model.weight"].chunks
        )

        assert source_weight_chunks == [((0, 0), shape)]
        assert output_weight_chunks == [
            ((0, 0), (2, 2)),
            ((2, 0), (2, 2)),
            ((4, 0), (1, 2)),
        ]
        assert result.balanced_work is True
        assert result.target_chunk_bytes == 16
        assert result.plan_tensor_chunks_by_rank == (5,)

        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["balanced_work"] is True
        assert provenance["target_chunk_bytes"] == 16


def test_metadata_same_layout_generated_gpt_round_trip_cpu_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1)
    try:
        with (
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_b") as ckpt_b,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_out") as output_root,
        ):
            _write_generated_gpt_checkpoint(ckpt_a, 1.0)
            _write_generated_gpt_checkpoint(ckpt_b, 5.0)

            result = merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=40,
            )

            assert result.output_dir == output_root / "iter_0000040"
            assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert result.averaged_tensors >= 15
            assert result.copied_extra_states == 0
            assert result.memory_estimate.template_devices == ("cpu",)

            output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
            tensor_state = {
                fqn: torch.empty(
                    tuple(int(dim) for dim in metadata.size),
                    dtype=metadata.properties.dtype,
                )
                for fqn, metadata in output_metadata.state_dict_metadata.items()
            }
            assert any("model.decoder.layers" in fqn for fqn in tensor_state)
            assert "model.embedding.word_embeddings.weight" in tensor_state
            assert "model.output_layer.weight" in tensor_state
            assert not any(fqn.endswith("._extra_state") for fqn in tensor_state)

            torch_dcp.load(tensor_state, checkpoint_id=str(result.output_dir), no_dist=True)
            for fqn, tensor in tensor_state.items():
                assert torch.equal(tensor, torch.full_like(tensor, 4.0)), fqn

            source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
            for fqn, output_tensor_metadata in output_metadata.state_dict_metadata.items():
                source_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in source_metadata.state_dict_metadata[fqn].chunks
                ]
                output_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in output_tensor_metadata.chunks
                ]
                assert output_chunks == source_chunks
    finally:
        ps.destroy_model_parallel()


def test_metadata_same_layout_generated_moe_gpt_round_trip_cpu_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1)
    try:
        with (
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_b") as ckpt_b,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_out") as output_root,
        ):
            _write_generated_moe_gpt_checkpoint(ckpt_a, 1.0)
            _write_generated_moe_gpt_checkpoint(ckpt_b, 5.0)

            result = merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=41,
                extra_state_source_index=1,
            )

            assert result.output_dir == output_root / "iter_0000041"
            assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert result.averaged_tensors == 19
            # All in-root byte/object _extra_state entries are copied from the
            # selected source (the metadata path does no object-entry filtering).
            assert result.copied_extra_states == 7
            assert result.memory_estimate.template_devices == ("cpu",)

            output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
            object_metadata_keys = [
                fqn
                for fqn, metadata in output_metadata.state_dict_metadata.items()
                if not hasattr(metadata, "size")
            ]
            assert len(object_metadata_keys) == 7
            assert all("_extra_state" in fqn for fqn in object_metadata_keys)
            tensor_state = {
                fqn: torch.empty(
                    tuple(int(dim) for dim in metadata.size),
                    dtype=metadata.properties.dtype,
                )
                for fqn, metadata in output_metadata.state_dict_metadata.items()
                if hasattr(metadata, "size")
            }
            assert any(fqn.endswith(".mlp.router.weight") for fqn in tensor_state)
            assert any(fqn.endswith(".mlp.router.bias") for fqn in tensor_state)
            expert_tensor_keys = [fqn for fqn in tensor_state if ".mlp.experts." in fqn]
            assert len(expert_tensor_keys) == 4
            assert not any(fqn.endswith("._extra_state") for fqn in tensor_state)

            torch_dcp.load(tensor_state, checkpoint_id=str(result.output_dir), no_dist=True)
            for fqn, tensor in tensor_state.items():
                assert torch.equal(tensor, torch.full_like(tensor, 4.0)), fqn

            source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
            for fqn in tensor_state:
                output_tensor_metadata = output_metadata.state_dict_metadata[fqn]
                source_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in source_metadata.state_dict_metadata[fqn].chunks
                ]
                output_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in output_tensor_metadata.chunks
                ]
                assert output_chunks == source_chunks
    finally:
        ps.destroy_model_parallel()


def test_metadata_same_layout_unprefixed_gpt_model_tensor_roots_round_trip(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=50,
        )

        assert result.output_dir == output_root / "iter_0000050"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 0

        public_state = {
            "decoder.final_layernorm.weight": torch.empty((3,), dtype=torch.float32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.empty((2, 3), dtype=torch.float32),
            "embedding.word_embeddings.weight": torch.empty((4, 3), dtype=torch.float32),
            "output_layer.weight": torch.empty((4, 3), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)

        assert torch.equal(public_state["decoder.final_layernorm.weight"], torch.full((3,), 4.0))
        assert torch.equal(
            public_state["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), 5.0),
        )
        assert torch.equal(public_state["embedding.word_embeddings.weight"], torch.full((4, 3), 6.0))
        assert torch.equal(public_state["output_layer.weight"], torch.full((4, 3), 7.0))


def test_metadata_same_layout_copies_unprefixed_byte_extra_state_from_selected_source(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_b, 5.0, 999)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=51,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000051"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 1

        public_state = {
            "decoder.final_layernorm.weight": torch.empty((3,), dtype=torch.float32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.empty((2, 3), dtype=torch.float32),
            "embedding.word_embeddings.weight": torch.empty((4, 3), dtype=torch.float32),
            "output_layer.weight": torch.empty((4, 3), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(public_state["decoder.final_layernorm.weight"], torch.full((3,), 4.0))
        assert torch.equal(
            public_state["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), 5.0),
        )
        assert torch.equal(public_state["embedding.word_embeddings.weight"], torch.full((4, 3), 6.0))
        assert torch.equal(public_state["output_layer.weight"], torch.full((4, 3), 7.0))

        object_key = f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_0_1"
        loaded = dist_checkpointing.load(
            {object_key: ShardedObject.empty_from_unique_key(object_key)},
            str(result.output_dir),
            validate_access_integrity=False,
        )
        assert _decode_sharded_object_value(loaded[object_key]) == {"value": 999}

        normal_loaded = dist_checkpointing.load(
            _unprefixed_gpt_like_byte_extra_template(),
            str(result.output_dir),
        )
        normal_extra_state = normal_loaded[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY]
        normal_extra_state.seek(0)
        assert torch.load(normal_extra_state, weights_only=False) == {"value": 999}


def test_metadata_same_layout_accepts_unprefixed_mtp_tensor_and_byte_extra_state(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_mtp_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint_with_mtp_state(ckpt_b, 5.0, 999)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=52,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000052"
        assert result.averaged_tensors == 5
        assert result.copied_extra_states == 1

        public_state = {
            "mtp.layers.0.eh_proj.weight": torch.empty((2, 2), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(
            public_state["mtp.layers.0.eh_proj.weight"],
            torch.full((2, 2), 8.0),
        )

        object_key = f"{UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY}/shard_0_1"
        loaded = dist_checkpointing.load(
            {object_key: ShardedObject.empty_from_unique_key(object_key)},
            str(result.output_dir),
            validate_access_integrity=False,
        )
        assert _decode_sharded_object_value(loaded[object_key]) == {"value": 999}


def test_metadata_same_layout_rejects_byte_objects_outside_model_roots(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(ckpt_b, 5.0)

        with pytest.raises(WeightedMergeError, match="byte/object DCP entries outside model roots"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=53,
            )


def test_metadata_same_layout_rejects_mismatched_byte_extra_state_keys(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        with pytest.raises(
            WeightedMergeError,
            match="identical byte/object _extra_state key sets",
        ):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=54,
            )


def test_metadata_same_layout_rejects_unprefixed_non_model_tensor_root(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_optimizer_tensor(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        with pytest.raises(WeightedMergeError, match="non-model DCP tensor keys"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=52,
            )


def test_metadata_same_layout_cli_dispatch_skips_megatron_parser(tmp_path, monkeypatch):
    ckpt_a = tmp_path / "iter_0000001"
    ckpt_b = tmp_path / "iter_0000002"
    output_root = tmp_path / "merged"
    calls = {}

    def fail_megatron_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not initialize Megatron")

    def fake_merge(input_paths, weights, output_root_arg, **kwargs):
        calls["input_paths"] = input_paths
        calls["weights"] = weights
        calls["output_root"] = output_root_arg
        calls["kwargs"] = kwargs
        return weighted_merge_module.MergeResult(
            output_dir=Path(output_root_arg) / "iter_0000030",
            input_dirs=tuple(Path(path) for path in input_paths),
            weights=tuple(weights),
            timings=weighted_merge_module.MergeTimings(),
            averaged_tensors=2,
            copied_extra_states=1,
            backend="torch_dist",
            implementation_mode=weighted_merge_module.METADATA_SAME_LAYOUT_MODE,
        )

    monkeypatch.setattr(weighted_merge_module, "ensure_process_group", fail_megatron_path)
    monkeypatch.setattr(
        weighted_merge_module,
        "merge_same_layout_dcp_metadata_checkpoints",
        fake_merge,
    )

    args = weighted_merge_module._parse_metadata_same_layout_args(
        [
            "--merge-inputs",
            f"{ckpt_a}:0.25",
            f"{ckpt_b}:0.75",
            "--merge-output",
            str(output_root),
            "--output-iteration",
            "30",
            "--extra-state-source-index",
            "1",
            "--ckpt-format",
            "torch_dist",
        ]
    )
    result = weighted_merge_module._run_metadata_same_layout_cli(args)

    assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
    assert calls["input_paths"] == [ckpt_a, ckpt_b]
    assert calls["weights"] == [0.25, 0.75]
    assert calls["output_root"] == str(output_root)
    assert calls["kwargs"]["output_iteration"] == 30
    assert calls["kwargs"]["extra_state_source_index"] == 1


def test_metadata_same_layout_cli_rejects_unsupported_template_options(tmp_path):
    base_args = [
        "--merge-inputs",
        f"{tmp_path / 'iter_0000001'}:1.0",
        "--merge-output",
        str(tmp_path / "merged"),
    ]
    with pytest.raises(WeightedMergeError, match="--merge-window-btoks is not supported"):
        weighted_merge_module._parse_metadata_same_layout_args(
            base_args + ["--end-checkpoint", "2", "--merge-window-btoks", "1"]
        )
    with pytest.raises(WeightedMergeError, match="--no-atomic-merge-output"):
        weighted_merge_module._parse_metadata_same_layout_args(
            base_args + ["--no-atomic-merge-output"]
        )


def test_metadata_same_layout_rejects_no_atomic_output(tmp_path_dist_ckpt, process_group):
    with pytest.raises(WeightedMergeError, match="requires atomic output publication"):
        merge_same_layout_dcp_metadata_checkpoints(
            [tmp_path_dist_ckpt / "missing"],
            [1.0],
            tmp_path_dist_ckpt / "merged",
            atomic_output=False,
        )


def test_metadata_same_layout_rejects_existing_output_overwrite_for_crash_safety(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_b") as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_out"
        ) as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)
        final_dir = output_root / "iter_0000030"
        final_dir.mkdir()
        _write_checkpoint(final_dir, 9.0, iteration=29)
        write_latest_checkpointed_iteration(final_dir, 29)

        with pytest.raises(WeightedMergeError, match="crash-atomic"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=30,
                overwrite_output=True,
            )

        restored = _load_checkpoint(final_dir)
        torch.testing.assert_close(restored["model"]["weight"], torch.full((2, 2), 9.0))
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "29"
        assert not list(output_root.glob(".iter_0000030.old-*"))
        assert not list(output_root.glob(".iter_0000030.tmp-*"))


def test_metadata_same_layout_rejects_mismatched_metadata(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, shape=(2, 2), iteration=1)
        _write_checkpoint(ckpt_b, 5.0, shape=(3, 2), iteration=2)

        with pytest.raises(WeightedMergeError, match="Shape mismatch.*metadata-same-layout"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root / "merged",
            )

        assert not (output_root / "merged").exists()


def test_metadata_same_layout_two_rank_product_round_trip_public_metadata(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    if _world_size() != 2:
        pytest.skip("two-rank metadata same-layout coverage requires torchrun --nproc_per_node=2")

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_a", sync=True
        ) as ckpt_a,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_b", sync=True
        ) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_checkpoint(ckpt_a, 1.0 + rank, extra_value=111.0 + rank, iteration=1)
        _write_checkpoint(ckpt_b, 5.0 + (2 * rank), extra_value=999.0 + rank, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=30,
            extra_state_source_index=1,
        )

        loaded = _load_checkpoint(result.output_dir)
        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert result.world_size == 2
        assert result.averaged_tensors == 2
        assert result.copied_extra_states == 1
        assert result.memory_estimate.template_devices == ("cpu",)

        expected_rank_weight = 4.0 + (1.75 * rank)
        assert torch.equal(loaded["model"]["weight"], torch.full((2, 2), expected_rank_weight))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), expected_rank_weight + 1.0))
        assert torch.equal(
            loaded["model"]["decoder.layers.0._extra_state"],
            torch.tensor([999.0 + rank]),
        )

        public = _full_public_dcp_state(result.output_dir)
        expected_public_weight = torch.cat(
            [torch.full((2, 2), 4.0), torch.full((2, 2), 5.75)], dim=0
        )
        expected_public_bias = torch.tensor([5.0, 5.0, 6.75, 6.75])
        expected_public_extra = torch.tensor([999.0, 1000.0])
        assert torch.equal(public["model.weight"], expected_public_weight)
        assert torch.equal(public["model.bias"], expected_public_bias)
        assert torch.equal(public["model.decoder.layers.0._extra_state"], expected_public_extra)

        source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
        output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        for fqn in (
            "model.weight",
            "model.bias",
            "model.decoder.layers.0._extra_state",
        ):
            source_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in source_metadata.state_dict_metadata[fqn].chunks
            ]
            output_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in output_metadata.state_dict_metadata[fqn].chunks
            ]
            assert output_chunks == source_chunks

        metadata_summary = _dcp_metadata_summary(result.output_dir)
        assert not metadata_summary["duplicate_chunk_offsets"]
        assert not metadata_summary["duplicate_storage_records"]
        assert metadata_summary["storage_file_count"] == 2
        assert len(metadata_summary["chunk_records"]) == 6

        if _rank() == 0:
            common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
            provenance = common_state["weighted_merge_provenance"]
            assert provenance["implementation_mode"] == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert provenance["extra_state_source_index"] == 1
            assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"
            assert len(list(result.output_dir.glob("*.distcp"))) == 2


def test_metadata_same_layout_two_rank_byte_extra_state_round_trip(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    if _world_size() != 2:
        pytest.skip("two-rank metadata same-layout byte coverage requires torchrun --nproc_per_node=2")

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_a", sync=True
        ) as ckpt_a,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_b", sync=True
        ) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
            ckpt_a,
            1.0 + rank,
            111 + rank,
            rank_sharded=True,
        )
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
            ckpt_b,
            5.0 + (2 * rank),
            999 + rank,
            rank_sharded=True,
        )

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=31,
            extra_state_source_index=1,
        )

        load_template = _unprefixed_gpt_like_model_state(0.0, rank_sharded=True)
        load_template.update(_unprefixed_gpt_like_byte_extra_template())
        loaded = dist_checkpointing.load(load_template, str(result.output_dir))

        assert result.output_dir == output_root / "iter_0000031"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 2
        expected_base = 4.0 + (1.75 * rank)
        assert torch.equal(
            loaded["decoder.final_layernorm.weight"],
            torch.full((3,), expected_base),
        )
        assert torch.equal(
            loaded["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), expected_base + 1.0),
        )
        assert torch.equal(
            loaded["embedding.word_embeddings.weight"],
            torch.full((4, 3), expected_base + 2.0),
        )
        assert torch.equal(
            loaded["output_layer.weight"],
            torch.full((4, 3), expected_base + 3.0),
        )
        loaded_extra_state = loaded[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY]
        loaded_extra_state.seek(0)
        assert torch.load(loaded_extra_state, weights_only=False) == {"value": 999 + rank}

        metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        byte_keys = sorted(
            str(fqn)
            for fqn, entry in metadata.state_dict_metadata.items()
            if type(entry).__name__ == "BytesStorageMetadata"
        )
        assert byte_keys == [
            f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_0_2",
            f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_1_2",
        ]
        metadata_summary = _dcp_metadata_summary(result.output_dir)
        assert not metadata_summary["duplicate_chunk_offsets"]
        assert not metadata_summary["duplicate_storage_records"]
        assert metadata_summary["storage_file_count"] == 2

        if _rank() == 0:
            assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "31"
            assert len(list(result.output_dir.glob("*.distcp"))) == 2


# ---------------------------------------------------------------------------
# Correctness tests rewritten onto the metadata-driven merge path.
# ---------------------------------------------------------------------------


def test_merge_rejects_existing_output_directory_by_default(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_exists_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_exists_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        output_dir = output_root / "merged"
        output_dir.mkdir()

        with pytest.raises(WeightedMergeError, match="Output directory already exists|crash-atomic"):
            merge_same_layout_dcp_metadata_checkpoints([ckpt_a], [1.0], output_dir)


def test_merge_supports_multiple_model_chunks(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_out") as output_root,
    ):
        _write_multi_chunk_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_multi_chunk_checkpoint(ckpt_b, 5.0, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
        )
        loaded = dist_checkpointing.load(_multi_chunk_template(), str(result.output_dir))

        assert torch.allclose(loaded["model0"]["weight"], torch.full((2, 2), 4.0))
        assert torch.allclose(loaded["model1"]["weight"], torch.full((2, 2), 5.0))


def test_merge_save_dtype_policy(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.float16, iteration=1)
        _write_checkpoint(ckpt_b, 3.0, dtype=torch.float16, iteration=2)

        same = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "same", save_dtype="same"
        )
        fp32 = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "fp32", save_dtype="float32"
        )
        bf16 = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "bf16", save_dtype="bfloat16"
        )
        fp16 = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "fp16", save_dtype="float16"
        )

        assert (
            _load_checkpoint(same.output_dir, dtype=torch.float16)["model"]["weight"].dtype
            == torch.float16
        )
        assert (
            _load_checkpoint(fp32.output_dir, dtype=torch.float32)["model"]["weight"].dtype
            == torch.float32
        )
        assert (
            _load_checkpoint(bf16.output_dir, dtype=torch.bfloat16)["model"]["weight"].dtype
            == torch.bfloat16
        )
        assert (
            _load_checkpoint(fp16.output_dir, dtype=torch.float16)["model"]["weight"].dtype
            == torch.float16
        )


def test_merge_accumulates_fp16_inputs_in_fp32(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 4096.0, dtype=torch.float16, iteration=1)
        _write_checkpoint(ckpt_b, 1.0, dtype=torch.float16, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            save_dtype="float32",
        )
        loaded = _load_checkpoint(result.output_dir, dtype=torch.float32)

        assert loaded["model"]["weight"].dtype == torch.float32
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 2048.5))


def test_merge_rejects_dtype_mismatch_with_same_policy(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.float32, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, dtype=torch.float16, iteration=2)

        with pytest.raises(WeightedMergeError, match="Dtype mismatch"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                save_dtype="same",
            )


def test_merge_rejects_non_floating_tensors(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1, dtype=torch.int64, iteration=1)
        _write_checkpoint(ckpt_b, 2, dtype=torch.int64, iteration=2)

        with pytest.raises(WeightedMergeError, match="non-floating"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
            )


def test_merge_rejects_shape_mismatch(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, shape=(2, 2), iteration=1)
        _write_checkpoint(ckpt_b, 2.0, shape=(3, 2), iteration=2)

        with pytest.raises(WeightedMergeError, match="Shape mismatch|shape|model.weight"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
            )


def test_merge_rejects_incompatible_checkpoint_keys(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_good") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_bad") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_bad_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        state_dict = _template(2.0, include_bias=False)
        state_dict["args"] = SimpleNamespace(iteration=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = 2
        dist_checkpointing.save(state_dict, str(ckpt_b))

        with pytest.raises(WeightedMergeError, match="identical model tensor key sets"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged"
            )


def test_merge_rejects_checkpoint_format_mismatch(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)
        (ckpt_b / "metadata.json").write_text(
            '{"sharded_backend": "different_backend", "sharded_backend_version": 1, '
            '"common_backend": "torch", "common_backend_version": 1}',
            encoding="utf-8",
        )

        with pytest.raises(WeightedMergeError, match="Checkpoint format mismatch"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged"
            )


def test_merge_rejects_unsupported_checkpoint_format(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unsupported_format") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unsupported_format_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        (ckpt_a / "metadata.json").write_text(
            '{"sharded_backend": "fsdp_dtensor", "sharded_backend_version": 1, '
            '"common_backend": "torch", "common_backend_version": 1}',
            encoding="utf-8",
        )

        with pytest.raises(WeightedMergeError, match="torch_dist checkpoints only"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a], [1.0], output_root / "merged"
            )


def test_merge_rejects_missing_metadata(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)
        (ckpt_b / "metadata.json").unlink()

        with pytest.raises(WeightedMergeError, match="not a distributed checkpoint|metadata"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged"
            )


def test_merge_rejects_argument_validation_errors(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_arg_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_arg_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)

        with pytest.raises(WeightedMergeError, match="At least one input"):
            merge_same_layout_dcp_metadata_checkpoints([], [], output_root)
        with pytest.raises(WeightedMergeError, match="input paths but"):
            merge_same_layout_dcp_metadata_checkpoints([ckpt_a], [0.5, 0.5], output_root)
        with pytest.raises(WeightedMergeError, match="Unsupported save dtype"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a], [1.0], output_root, save_dtype="fp8"
            )


def test_object_extra_state_is_copied(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_out") as output_root,
    ):
        _write_object_extra_checkpoint(ckpt_a, 1.0, extra_value=111, iteration=1)
        _write_object_extra_checkpoint(ckpt_b, 3.0, extra_value=999, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
        )
        loaded = dist_checkpointing.load(_object_extra_template(), str(result.output_dir))

        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 2.0))
        loaded["model"]["decoder.layers.0._extra_state"].seek(0)
        assert torch.load(loaded["model"]["decoder.layers.0._extra_state"]) == {"value": 111}


def test_extra_state_source_index_can_be_selected(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1)
        _write_checkpoint(ckpt_b, 3.0, extra_value=999.0, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            extra_state_source_index=1,
        )
        loaded = _load_checkpoint(result.output_dir)

        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))


def test_merge_minus_sqrt_schedule_three_checkpoints_matches_fp32_weighted_sum(
    tmp_path_dist_ckpt, process_group
):
    values = [1.0, 2.0, 4.0]
    coefficients = list(checkpoint_coefficients([1, 2, 3], "minus-sqrt").values())
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_minus_sqrt_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_minus_sqrt_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_minus_sqrt_c") as ckpt_c,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_minus_sqrt_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, values[0], extra_value=111.0, iteration=1)
        _write_checkpoint(ckpt_b, values[1], extra_value=222.0, iteration=2)
        _write_checkpoint(ckpt_c, values[2], extra_value=333.0, iteration=3)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b, ckpt_c],
            coefficients,
            output_root / "merged",
            merge_style="minus-sqrt",
        )
        loaded = _load_checkpoint(result.output_dir)

        expected_weight = sum(coef * value for coef, value in zip(coefficients, values))
        expected_bias = sum(coef * (value + 1.0) for coef, value in zip(coefficients, values))
        assert result.averaged_tensors == 2
        assert torch.allclose(
            loaded["model"]["weight"], torch.full((2, 2), expected_weight), atol=1e-6
        )
        assert torch.allclose(
            loaded["model"]["bias"], torch.full((2,), expected_bias), atol=1e-6
        )
        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["merge_style"] == "minus-sqrt"
        assert provenance["weights"] == pytest.approx(coefficients)


def test_merge_negative_manual_weights_round_trip(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_negative_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_negative_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_negative_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 2.0, extra_value=111.0, iteration=1)
        _write_checkpoint(ckpt_b, 4.0, extra_value=999.0, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [1.5, -0.5],
            output_root / "merged",
        )
        loaded = _load_checkpoint(result.output_dir)

        # weight = 1.5 * 2.0 - 0.5 * 4.0 = 1.0; bias = 1.5 * 3.0 - 0.5 * 5.0 = 2.0
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 1.0), atol=1e-6)
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 2.0), atol=1e-6)
        assert torch.equal(
            loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0])
        )
        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        assert common_state["weighted_merge_provenance"]["weights"] == [1.5, -0.5]


def test_cli_argparse_surface_merges_checkpoint(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cli_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cli_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cli_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        # Exercise the real metadata CLI surface the same way main() does.
        args = weighted_merge_module._parse_metadata_same_layout_args(
            [
                "--merge-inputs",
                f"{ckpt_a}:0.25",
                f"{ckpt_b}:0.75",
                "--merge-output",
                str(output_root),
                "--output-iteration",
                "30",
            ]
        )

        assert args.merge_save_dtype == "same"

        result = weighted_merge_module._run_metadata_same_layout_cli(args)

        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        loaded = _load_checkpoint(result.output_dir)
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(
            loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0])
        )
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"
