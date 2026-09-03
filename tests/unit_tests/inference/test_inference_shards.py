# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shard spec parsing/validation (CPU; no torch.distributed)."""

from unittest.mock import Mock

import pytest

from megatron.core.inference import shards as shards_module
from megatron.core.inference.disaggregation.coordinator_setup import validate_disaggregation_shards
from megatron.core.inference.shards_spec import (
    InferenceShardAssignment,
    InferenceShardSpec,
    normalize_shard_specs,
    parse_inference_shards_spec,
    resolve_inference_shard,
)


def test_shard_builder_keeps_replica_process_group(monkeypatch):
    pg_collections = [Mock(name="prefill_pgc"), Mock(name="decode_pgc")]
    shard_groups = [Mock(name="prefill_group"), Mock(name="decode_group")]
    monkeypatch.setattr(shards_module.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(
        shards_module, "build_inference_pg_collection", Mock(side_effect=pg_collections)
    )
    new_group = Mock(side_effect=shard_groups)
    monkeypatch.setattr(shards_module.dist, "new_group", new_group)

    shards = shards_module.build_inference_pg_collections_for_shards(
        4,
        [
            InferenceShardSpec(tp=1, dp=2, role="prefill"),
            InferenceShardSpec(tp=1, dp=2, role="decode"),
        ],
    )

    assert shards[0].pg_collection is pg_collections[0]
    assert shards[0].process_group is shard_groups[0]
    assert shards[1].pg_collection is None
    assert shards[1].process_group is None
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 1], [2, 3]]


def test_shard_spec_objects_match_string_parsing():
    objs = [InferenceShardSpec(tp=2, role="prefill"), InferenceShardSpec(tp=1, dp=2, role="decode")]
    assert normalize_shard_specs(objs, 4) == parse_inference_shards_spec(
        "tp=2,role=prefill+tp=1,dp=2,role=decode", 4
    )
    # expt_tp defaults to tp; raw dicts also accepted
    assert InferenceShardSpec(tp=4).to_dict()["expt_tp"] == 4
    assert normalize_shard_specs([{"tp": 1, "role": "prefill"}, {"tp": 1, "role": "decode"}], 2)
    # bad role rejected at construction
    with pytest.raises(ValueError):
        InferenceShardSpec(tp=1, role="both")


@pytest.mark.parametrize(
    "rank,index,rank_offset,role",
    [(0, 0, 0, "prefill"), (1, 0, 0, "prefill"), (2, 1, 2, "decode"), (3, 1, 2, "decode")],
)
def test_resolve_inference_shard(rank, index, rank_offset, role):
    assignment = resolve_inference_shard(
        "tp=1,dp=2,role=prefill+tp=2,role=decode", world_size=4, rank=rank
    )
    assert assignment == InferenceShardAssignment(
        index=index,
        spec=InferenceShardSpec(
            tp=1 if role == "prefill" else 2, dp=2 if role == "prefill" else 1, role=role
        ),
        rank_offset=rank_offset,
        shard_count=2,
    )


@pytest.mark.parametrize("rank", [-1, 4])
def test_resolve_inference_shard_rejects_out_of_range_rank(rank):
    with pytest.raises(ValueError, match="rank must be in"):
        resolve_inference_shard("tp=2+tp=2", world_size=4, rank=rank)


def test_validate_disaggregation_shards_owns_role_validation():
    specs = validate_disaggregation_shards("tp=1,role=prefill+tp=1,role=decode", world_size=2)
    assert [spec.role for spec in specs] == ["prefill", "decode"]

    with pytest.raises(ValueError, match="every disaggregated shard"):
        validate_disaggregation_shards("tp=1+tp=1,role=decode", world_size=2)
    with pytest.raises(ValueError, match="at least one prefill and one decode"):
        validate_disaggregation_shards("tp=1,role=prefill+tp=1,role=prefill", world_size=2)


# --------------------------------------------------------------------------
# spec parsing
# --------------------------------------------------------------------------
def test_parse_defaults_and_dp_and_role():
    specs = parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=4)
    # parser returns InferenceShardSpec with defaults filled (expt_tp -> tp).
    assert specs[0] == InferenceShardSpec(tp=2, role="prefill")
    assert specs[1] == InferenceShardSpec(tp=1, dp=2, role="decode")
    # dict form (serialization / external consumers) carries the resolved keys.
    assert specs[0].to_dict() == {
        "tp": 2,
        "pp": 1,
        "ep": 1,
        "dp": 1,
        "expt_tp": 2,
        "role": "prefill",
    }


def test_parse_partitions_world_with_dp():
    # world must equal sum(tp*pp*dp): 2 + (1*1*2) = 4
    parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=4)
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=5)


def test_parse_rejects_bad_role_and_unknown_key():
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=1,role=both", world_size=1)
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=1,foo=2", world_size=1)


def test_plus_and_semicolon_separators_equivalent():
    a = parse_inference_shards_spec("tp=2,role=prefill+tp=1,role=decode", world_size=3)
    b = parse_inference_shards_spec("tp=2,role=prefill;tp=1,role=decode", world_size=3)
    assert a == b


def test_cp_accepted_only_when_one():
    # cp is a recognized key (clear error, not "unknown key") but must be 1:
    # inference shards don't context-parallelize.
    assert parse_inference_shards_spec("tp=2,cp=1", world_size=2) == [
        InferenceShardSpec(tp=2, cp=1)
    ]
    with pytest.raises(ValueError):
        InferenceShardSpec(tp=1, cp=2)
    with pytest.raises(ValueError):
        parse_inference_shards_spec("tp=1,cp=2", world_size=1)
