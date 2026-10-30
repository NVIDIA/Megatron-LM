# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shard spec validation and shard-specific serving/RL model construction."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.inference import shards as shards_module
from megatron.core.inference.disaggregation.coordinator_setup import validate_disaggregation_shards
from megatron.core.inference.shards_spec import (
    InferenceShardAssignment,
    InferenceShardSpec,
    normalize_shard_specs,
    parse_inference_shards_spec,
    resolve_inference_shard,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import unwrap_model
from megatron.training.argument_utils import tokenizer_config_from_args
from megatron.training.config import RLConfig
from tests.unit_tests.test_utilities import Utils


def test_serving_and_rl_build_shard_models_with_global_parallel_args(monkeypatch, run_config):
    from megatron.inference import utils
    from megatron.rl.inference import disagg
    from megatron.training.arguments import validate_args
    from tests.unit_tests.test_argument_utils import _minimal_training_args

    # The TP=2 training configuration requires this on Hopper and earlier.
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    Utils.initialize_model_parallel()
    try:
        args = _minimal_training_args(monkeypatch)
        args.freeze_base_model_for_mtp = False
        args.gradient_accumulation_fusion = False
        args.world_size = max(4, Utils.world_size)
        args.tensor_model_parallel_size = 2
        args.pipeline_model_parallel_size = 2
        args.expert_tensor_parallel_size = 2
        args.bf16 = True
        args = validate_args(args)
        args.model_provider = "gpt"
        args.padded_vocab_size = 1024
        args.load = "unused-test-checkpoint"
        args.inference_ckpt_non_strict = False
        args.inference_dynamic_batching_enable_prefix_caching = True
        args.inference_shards = "tp=1,role=prefill+tp=1,role=decode"
        args.use_tp_pp_dp_mapping = False
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(123)
        monkeypatch.setattr(utils, "get_args", lambda: args)
        load = Mock()
        monkeypatch.setattr(utils, "load_checkpoint", load)
        tokenizer_cfg = tokenizer_config_from_args(args)
        run_config.tokenizer = tokenizer_cfg
        serving = utils.get_model_for_inference(pg_collection=pg, checkpoint_group=pg.mp)

        # Exercise the separate RL builder too, starting with the training config.
        training_config = utils.get_model_builder(args, tokenizer_cfg)._model_config
        shard = SimpleNamespace(pg_collection=pg, spec=InferenceShardSpec(tp=1, role="decode"))
        monkeypatch.setattr(
            disagg, "build_inference_pg_collections_for_shards", lambda **kw: [shard]
        )
        rl = disagg.build_disagg_inference_model(
            args,
            training_config.transformer,
            cfg_container=SimpleNamespace(model=training_config, rl=RLConfig()),
            model_alloc_ctx=nullcontext(),
        )[0]
        for model in (serving, rl):
            config = unwrap_model(model).config
            assert config.tensor_model_parallel_size == 1
            assert config.pipeline_model_parallel_size == 1
            assert config.expert_tensor_parallel_size == 1
            assert len(unwrap_model(model).decoder.layers) == args.num_layers
            tokens = torch.tensor([[1, 2, 3]], device="cuda")
            with torch.no_grad():
                output = model(tokens, torch.arange(3, device="cuda")[None], attention_mask=None)
            assert output.shape == (1, 3, 1024)
            assert torch.isfinite(output).all()
        assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == 2
        assert training_config.transformer.tensor_model_parallel_size == 2
        assert load.call_args.kwargs["tp_group"] is pg.tp
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "field",
    [
        "rl_inference_tensor_model_parallel_size",
        "rl_inference_pipeline_model_parallel_size",
        "rl_inference_expert_model_parallel_size",
        "rl_inference_expert_tensor_model_parallel_size",
        "rl_verify_model_weights_swap",
        "refit_method",
    ],
)
def test_disagg_rl_rejects_ignored_overrides(field):
    from megatron.rl.inference.disagg import build_disagg_inference_model

    config = RLConfig()
    value = (
        "nixl"
        if field == "refit_method"
        else True if field == "rl_verify_model_weights_swap" else 1
    )
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        build_disagg_inference_model(
            None, None, cfg_container=SimpleNamespace(rl=config), model_alloc_ctx=None
        )


@pytest.mark.parametrize("mismatch", [None, "groups", "model"])
def test_prebuilt_disagg_checks_groups_and_model_config(monkeypatch, mismatch):
    from megatron.core.inference.disaggregation import coordinator_setup as setup

    monkeypatch.setattr(setup.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(setup.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(setup, "get_pg_size", lambda group: group)
    monkeypatch.setattr(setup, "get_pg_rank", lambda group: 0)
    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    engine = SimpleNamespace(
        context=SimpleNamespace(
            enable_prefix_caching=True,
            config=SimpleNamespace(disaggregation_shards="tp=1,role=prefill+tp=1,role=decode"),
        ),
        pg_collection=SimpleNamespace(
            tp=2 if mismatch == "groups" else 1, pp=1, dp=1, ep=1, cp=1, expt_tp=1
        ),
        controller=SimpleNamespace(
            inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=config))
        ),
        set_disaggregation_config=Mock(),
    )
    if mismatch == "model":
        config.pipeline_model_parallel_size = 2
    if mismatch:
        with pytest.raises(ValueError, match="parallelism mismatch"):
            setup.configure_prebuilt_disagg_engine(engine)
        engine.set_disaggregation_config.assert_not_called()
    else:
        setup.configure_prebuilt_disagg_engine(engine)
        engine.set_disaggregation_config.assert_called_once()


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


def test_checkpoint_prefix_warning_uses_replica_group(monkeypatch, caplog):
    from megatron.inference import utils

    group = Mock(name="checkpoint_group")
    args = SimpleNamespace(ckpt_step=1, load="checkpoint", checkpoint_model_prefix="model.")
    world_size = Mock(return_value=1)
    rank = Mock(return_value=0)
    gather = Mock(side_effect=lambda output, keys, **kwargs: output.__setitem__(0, keys))
    monkeypatch.setattr(utils.torch.distributed, "get_world_size", world_size)
    monkeypatch.setattr(utils.torch.distributed, "get_rank", rank)
    monkeypatch.setattr(utils.torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(utils, "get_loaded_iteration", lambda: 1)
    monkeypatch.setattr(
        utils.dist_checkpointing,
        "load_tensors_metadata",
        lambda path: {"model.weight": None, "model.unused": None},
    )

    utils._warn_on_unloaded_checkpoint_tensors(args, {"model.weight"}, group)

    world_size.assert_called_once_with(group)
    rank.assert_called_once_with(group)
    gather.assert_called_once_with([["model.weight"]], ["model.weight"], group=group)
    assert "model.unused" in caplog.text


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
