# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full Qwen3.5 module costs, fixed user inputs and runtime model selection."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import data_schedule
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.models._benchmark import model_e2e
from megatron.core.datasets.dcp_runtime_scheduler.models.full35b.gdn_features import (
    predict as frozen_gdn_predict,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.full35b.geometry import (
    dot,
    fa_features,
    pack_geometry,
)
from megatron.core.datasets.dcp_runtime_scheduler.models.qwen35_full import Qwen35FullCostModel
from megatron.core.datasets.dcp_runtime_scheduler.models.selection import (
    validate_full_qwen35_config,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightOptions,
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry
from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import build_timeline
from megatron.core.packed_seq_params import resolve_thd_tail_padding_policy
from megatron.core.transformer import transformer_config
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

FROZEN = json.loads(Path(__file__).with_name("dcp_qwen35_full_frozen_predictions.json").read_text())


def _geometry(allowed=None):
    return RuntimePlanGeometry(
        world_size=32,
        tokens_per_rank=16384,
        ep_size=32,
        allowed_cp_sizes=tuple(range(1, 33)) if allowed is None else allowed,
        dataset_alignment=64,
        max_group_seqs=64,
    )


def _non_attention(forward=0.5, backward=1.0, **options):
    return dict(forward_ms_per_layer=forward, backward_ms_per_layer=backward, **options)


def _model(geometry=None, **options):
    options.setdefault("hardware", "gb200")
    options.setdefault("non_attention", _non_attention())
    return Qwen35FullCostModel(_geometry() if geometry is None else geometry, **options)


@pytest.fixture(scope="module")
def frozen_model():
    return _model()


@pytest.mark.parametrize(
    "case",
    FROZEN["cases"],
    ids=lambda row: f"{row['kind']}-{row['phase']}-cp{row['cp']}-cache{row['cache_pass']}",
)
def test_attention_heads_match_frozen_explicit_physical_points(case, frozen_model):
    point = {
        "cp": case["cp"],
        "logical_lengths": case["lengths"],
        "storage_lengths": case["storage_lengths"],
    }
    if case["kind"] == "fa":
        head = frozen_model.attention["models_fa_module"][str(case["cp"])][case["phase"]]
        actual = dot(head["coefficients"], fa_features(point))
    else:
        branch = case["phase"] + "_cache" + str(case["cache_pass"])
        head = frozen_model.attention["models_gdn_module"][branch]
        actual = frozen_gdn_predict(head["coefficients"], point, case["phase"])
    assert actual == pytest.approx(case["predicted_ms"], rel=2e-13, abs=1e-12)


def test_runtime_raw_lengths_use_canonical_storage_instead_of_calibration_storage(frozen_model):
    lengths, cp = (12472, 12492), 2
    canonical = pack_geometry(lengths, cp)
    assert canonical["storage_lengths"] == [12480, 20288]
    reference = next(
        case
        for case in FROZEN["cases"]
        if case["kind"] == "gdn"
        and case["cp"] == cp
        and case["phase"] == "backward"
        and case["cache_pass"] == 0
    )
    assert reference["lengths"] == list(lengths)
    assert reference["storage_lengths"] == [16384, 16384]
    for branch in (0, 1):
        predictions = frozen_model.attention_components(lengths, cp, cache_pass=branch)
        for phase in ("forward", "backward"):
            fa_head = frozen_model.attention["models_fa_module"][str(cp)][phase]
            assert predictions["fa"]["module"][phase] == pytest.approx(
                dot(fa_head["coefficients"], fa_features(canonical)), rel=2e-13, abs=1e-12
            )
            gdn_head = frozen_model.attention["models_gdn_module"][phase + "_cache" + str(branch)]
            assert predictions["gdn"]["module"][phase] == pytest.approx(
                frozen_gdn_predict(gdn_head["coefficients"], canonical, phase), rel=2e-13, abs=1e-12
            )
        if branch == 0:
            assert predictions["gdn"]["module"]["backward"] != pytest.approx(
                reference["predicted_ms"], rel=2e-13, abs=1e-12
            )


def test_full_model_contains_every_decoder_layer_and_the_fa_mtp(frozen_model):
    layers = frozen_model.layers
    assert frozen_model.layer_count == len(layers) == 41
    assert [layer["kind"] for layer in layers] == ["gdn", "gdn", "gdn", "fa"] * 10 + ["fa"]
    assert len({layer["name"] for layer in layers}) == 41
    assert "decoder.layers.39" in layers[-2]["name"]
    assert "mtp.layers.0" in layers[-1]["name"]
    program = frozen_model.timing()["execution_program"][0]
    forward = [op["cost"] for op in program["forward"] if ":attention:" in op.get("cost", "")]
    backward = [op["cost"] for op in program["backward"] if ":attention:" in op.get("cost", "")]
    assert forward == [layer["name"] + ":attention:forward" for layer in layers]
    assert backward == [layer["name"] + ":attention:backward" for layer in reversed(layers)]


def test_full_module_costs_are_not_advertised_as_core_predictions(frozen_model):
    metadata = frozen_model.metadata()
    assert metadata["layer_count"] == 41
    assert metadata["gdn_feature_schema"] == "GDN_CP_BRANCH_ACTIVATION_V2"
    assert metadata["core_prediction_available"] is False
    assert metadata["whole_attention_module_includes_projections"] is True
    assert metadata["non_attention_density_regression_loaded"] is False
    assert metadata["e2e_residual_fitted"] is False
    assert frozen_model.source_hashes["source/attention.json"] == FROZEN["source_sha256"]
    assert frozen_model.source_hashes["source_candidate.json"] == FROZEN["candidate_sha256"]
    with pytest.raises(ValueError, match="core|module"):
        frozen_model.attention_components((1025,), 1, scope="core")


def test_metadata_reports_accuracy_gates_without_status_labels(frozen_model):
    metadata = frozen_model.metadata()
    # The held-out gate results of the source fits are facts about the parameters.
    assert metadata["accuracy"] == {
        "FA_v1_module_heads_gate_pass": True,
        "GDN_v2_new_heldout_gate_pass": False,
        "attention_module_heldout_gate_pass": False,
        "density_heldout_gate_pass": False,
    }
    assert metadata["scientific_accuracy_passed"] is False
    # Labels of the source's own release process are not part of a plan record.
    for label in ("provisional", "runtime_ready", "factory_admission", "source_scientific_status"):
        assert label not in metadata
    text = json.dumps(metadata)
    for word in ("CANDIDATE", "PENDING", "runtime_ready", "factory_admission"):
        assert word not in text
    # The layer types a per-type non_attention mapping in the runtime file may name.
    assert metadata["non_attention_mapping_keys"] == ["gdn", "gqa", "mtp"]


def test_metadata_reports_the_executed_tail_policy_next_to_the_source_one():
    # The features evaluate every pack in the source layout: the tail extends the last
    # sequence, so one sequence fills the whole CP1 capacity and no dummy sequence appears.
    assert pack_geometry((1025,), 1)["storage_lengths"] == [16384]
    default = _model().metadata()
    assert default["source_tail_padding"] == "extend_last"
    assert default["source_execution_domain"]["geometry_policy"]["tail_padding"] == "extend_last"
    assert default["executed_tail_padding"] == "append_dummy_seq"
    same = _model(tail_padding="extend_last")
    assert same.metadata()["executed_tail_padding"] == "extend_last"
    # Reporting the executed policy changes no prediction.
    lengths, ranks = (32769, 1025), tuple(range(8))
    assert same.program_costs(lengths, 8, ranks=ranks) == _model().program_costs(
        lengths, 8, ranks=ranks
    )
    with pytest.raises(ValueError, match="tail_padding must be one of"):
        _model(tail_padding="separate_sequence")


def test_attention_cache_branches_follow_the_original_per_layer_program(frozen_model):
    lengths, cp = (32769, 1025), 8
    costs = frozen_model.program_costs(lengths, cp, ranks=tuple(range(cp)))
    branches = {
        branch: frozen_model.attention_components(lengths, cp, cache_pass=branch)
        for branch in (0, 1)
    }
    for layer in frozen_model.layers:
        for phase in ("forward", "backward"):
            branch = (
                frozen_model.cache["by_cp"][str(cp)][layer["name"]][phase]
                if layer["kind"] == "gdn"
                else 0
            )
            expected = branches[branch][layer["kind"]]["module"][phase]
            assert costs[layer["name"] + ":attention:" + phase] == expected


def test_one_cp32_pack_charges_modules_fixed_remainders_and_tail_once():
    geometry = _geometry((32,))
    model = _model(geometry, non_attention=_non_attention(0.5, 1.0, iteration_tail_ms=7.0))
    lengths, ranks = (32769, 1025), tuple(range(32))
    costs = model.program_costs(lengths, 32, ranks=ranks)
    attention = sum(value for name, value in costs.items() if ":attention:" in name)
    group = (
        0,
        32,
        model.forward(lengths, 32, ranks=ranks),
        model.backward(lengths, 32, ranks=ranks),
        costs,
    )
    predicted = build_timeline(model, geometry).evaluate([[group]])
    assert predicted == pytest.approx(attention + 41 * 1.5 + 7.0)


def test_family_inputs_change_only_fixed_non_attention_costs():
    lengths, ranks = (32769, 1025), tuple(range(8))
    zero = _model(non_attention=_non_attention(0.0, 0.0))
    measured = _model(
        non_attention=_non_attention({"gdn": 0.5, "fa": 1.0}, {"gdn": 1.5, "fa": 3.0})
    )
    before = zero.program_costs(lengths, 8, ranks=ranks)
    after = measured.program_costs(lengths, 8, ranks=ranks)
    for layer in measured.layers:
        forward, backward = (0.5, 1.5) if layer["kind"] == "gdn" else (1.0, 3.0)
        for phase, expected in (("forward", forward), ("backward", backward)):
            prefix = layer["name"]
            assert after[prefix + ":attention:" + phase] == before[prefix + ":attention:" + phase]
            assert before[prefix + ":nonattention:" + phase] == 0.0
            assert after[prefix + ":nonattention:" + phase] == expected


def test_named_layer_inputs_can_distinguish_the_mtp(frozen_model):
    forward = {layer["name"]: 1.0 for layer in frozen_model.layers}
    backward = {layer["name"]: 2.0 for layer in frozen_model.layers}
    forward[frozen_model.layers[-1]["name"]] = 3.0
    backward[frozen_model.layers[-1]["name"]] = 5.0
    model = _model(non_attention=_non_attention(forward, backward))
    costs = model.program_costs((1025,), 1, ranks=(0,))
    assert costs[model.layers[0]["name"] + ":nonattention:forward"] == 1.0
    assert costs[model.layers[0]["name"] + ":nonattention:backward"] == 2.0
    assert costs[model.layers[-1]["name"] + ":nonattention:forward"] == 3.0
    assert costs[model.layers[-1]["name"] + ":nonattention:backward"] == 5.0


@pytest.mark.parametrize("named_override", [False, True])
def test_mtp_inputs_override_only_the_final_fa_layer(frozen_model, named_override):
    forward = {"gdn": 0.5, "fa": 1.0, "mtp": 3.0}
    backward = {"gdn": 1.5, "fa": 3.0, "mtp": 7.0}
    if named_override:
        forward[frozen_model.layers[-1]["name"]] = 5.0
        backward[frozen_model.layers[-1]["name"]] = 11.0
    model = _model(non_attention=_non_attention(forward, backward))
    costs = model.program_costs((1025,), 1, ranks=(0,))
    for layer in model.layers[:-1]:
        expected = (0.5, 1.5) if layer["kind"] == "gdn" else (1.0, 3.0)
        for phase, value in zip(("forward", "backward"), expected):
            assert costs[layer["name"] + ":nonattention:" + phase] == value
    mtp = model.layers[-1]["name"]
    assert costs[mtp + ":nonattention:forward"] == (5.0 if named_override else 3.0)
    assert costs[mtp + ":nonattention:backward"] == (11.0 if named_override else 7.0)


def test_exported_e2e_qwen35_layout_is_the_full_decoder_and_mtp():
    config = model_e2e.qwen35_35b(world_size=32, ep_size=32)
    assert config.layer_types == ("gdn", "gdn", "gdn", "gqa") * 10 + ("gqa",)
    assert len(config.layer_types) == 41
    assert config.layer_types.count("gdn") == 30
    assert config.layer_types.count("gqa") == 11


def test_fixed_non_attention_inputs_remain_required():
    with pytest.raises(TypeError):
        Qwen35FullCostModel(_geometry())
    with pytest.raises((TypeError, ValueError)):
        _model(non_attention={}).timing()


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("value", [None, True, -1.0, float("nan"), float("inf"), {"fa": 1.0}])
def test_invalid_or_incomplete_non_attention_inputs_are_rejected(phase, value):
    inputs = _non_attention()
    inputs[phase + "_ms_per_layer"] = value
    with pytest.raises((TypeError, ValueError)):
        _model(non_attention=inputs).timing()


@pytest.mark.parametrize(
    "changes,message",
    [
        # The first differing setting is named with the measured and the actual value.
        ({"world_size": 64}, "were measured with DPxCP=32, got 64"),
        ({"world_size": 64, "ep_size": 4}, "were measured with DPxCP=32, got 64"),
        ({"ep_size": 4}, "were measured with expert_model_parallel_size=32, got 4"),
        ({"tokens_per_rank": 8192}, "were measured with max_seqlen_per_dp_cp_rank=16384, got 8192"),
        ({"dataset_alignment": 32}, "were measured with dataset_alignment=64, got 32"),
        ({"max_group_seqs": 65}, "cover at most 64 sequences per CP group, got 65"),
    ],
)
def test_full_model_rejects_a_different_measured_geometry(changes, message):
    geometry = replace(_geometry(), **changes)
    with pytest.raises(ValueError) as caught:
        _model(geometry)
    assert message in str(caught.value)


def test_full_model_rejects_other_cp_sizes_and_another_geometry_object():
    with pytest.raises(ValueError, match=r"cover CP groups of 1 to 32 ranks, got \[True, 2\]"):
        _model(replace(_geometry(), allowed_cp_sizes=(True, 2)))
    with pytest.raises(ValueError, match="were bound to a different geometry"):
        _model().check_domain(_geometry((1, 2)))


@pytest.mark.parametrize("cap", [None, 1, 63, 64])
def test_full_model_accepts_any_sequence_cap_within_its_64_sequence_domain(cap):
    # A 64-entry THD buffer keeps 63 real sequences when a dummy tail sequence is appended.
    geometry = replace(_geometry(), max_group_seqs=cap)
    model = _model(geometry)
    model.check_domain(geometry, None)
    assert model.forward((1025, 511), 32, ranks=tuple(range(32))) > 0


@pytest.mark.parametrize("hardware", ["h100", "gb300", "GB200", None])
def test_full_model_does_not_silently_transfer_hardware(hardware):
    with pytest.raises(ValueError) as caught:
        _model(hardware=hardware)
    assert str(caught.value) == f'"hardware" must be "gb200" for full Qwen3.5, got {hardware!r}'


@pytest.mark.parametrize(
    "lengths,cp,ranks",
    [
        ((1025,), 0, ()),
        ((1025,), 33, tuple(range(33))),
        ((0,), 1, (0,)),
        ((True,), 1, (0,)),
        ((), 1, (0,)),
        ((16385,), 1, (0,)),
        ((1,) * 65, 1, (0,)),
        ((1025,), 2, (0, 2)),
        ((1025,), 2, (0, 0)),
        ((1025,), 2, (31, 32)),
        ((1025,), 1, None),
    ],
)
def test_full_model_rejects_invalid_queries(frozen_model, lengths, cp, ranks):
    with pytest.raises(ValueError):
        frozen_model.program_costs(lengths, cp, ranks=ranks)


# --------------------------------------------------------------------------------------
# The training configuration: the real config class, validated the way training does it
# --------------------------------------------------------------------------------------
def _training_config(monkeypatch, tmp_path, **overrides):
    """TransformerConfig of a full Qwen3.5-35B-A3B run with dynamic CP on 32 ranks."""
    # TransformerConfig accepts sequence packing only with Transformer Engine >= 2.9. The
    # configuration checks under test do not depend on the installed version, so answer its
    # version predicate here and keep these tests running on images with an older one.
    monkeypatch.setattr(transformer_config, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config, "is_te_min_version", lambda _version: True)
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(dict(type="dcp_benchmark", hardware="gb200", non_attention=_non_attention()))
    )
    values = dict(
        # Model.
        num_layers=40,
        mtp_num_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        qk_layernorm=True,
        attention_output_gate=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="gdn",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_moe_experts=256,
        moe_token_dispatcher_type="alltoall",
        # Execution: BF16, EP32, dynamic CP over P2P with 16384 tokens stored on every rank.
        bf16=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=32,
        context_parallel_size=32,
        dynamic_context_parallel=True,
        # TransformerConfig accepts dynamic CP only with the per-token loss.
        calculate_per_token_loss=True,
        min_dynamic_context_parallel_size=1,
        linear_cp_mode="chunkwise",
        cp_partition_mode="contiguous",
        cp_comm_type="p2p",
        max_seqlen_per_dp_cp_rank=16384,
        pad_packed_seq_alignment="max",
        thd_tail_padding_policy="append_dummy_seq",
        thd_max_packed_sequences=64,
        use_cpu_initialization=True,
        dynamic_cp_runtime_config=str(path),
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _start_up(monkeypatch, config, world_size=32):
    """Run the validation training performs on every rank before the first step."""
    single = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    parent = SimpleNamespace(rank=lambda: 0, size=lambda: world_size)
    # This process stands in for the DPxCP leader of the coverage broadcast.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda *args: 0)
    monkeypatch.setattr(prefetch, "_count_over_group", lambda flag, group: int(bool(flag)))
    data_schedule.validate_dynamic_cp_runtime(config, SimpleNamespace(dp_cp=parent, pp=single))
    return config._dynamic_cp_runtime["spec"]


def test_real_training_configuration_passes_start_up_validation(monkeypatch, tmp_path):
    config = _training_config(monkeypatch, tmp_path)
    # What TransformerConfig resolves for this run; the model check has to agree with it.
    assert config.experimental_attention_variant == "gdn"
    assert config.sequence_packing_scheduler == "default_dynamic_cp"
    assert config.linear_cp_mode == "chunkwise" and config.cp_partition_mode == "contiguous"
    assert config.linear_cp_layout == "zigzag"  # An unset linear layout never says contiguous.
    assert config.attention_backend is AttnBackend.auto
    spec = _start_up(monkeypatch, config)
    assert spec["model"] == "qwen35_full"
    geometry = spec["geometry"]
    assert geometry["allowed_cp_sizes"] == (1, 2, 4, 8, 16, 32)
    assert (geometry["world_size"], geometry["ep_size"]) == (32, 32)
    assert (geometry["tokens_per_rank"], geometry["dataset_alignment"]) == (16384, 64)
    # thd_max_packed_sequences=64 keeps one entry for the appended dummy tail sequence.
    assert geometry["max_group_seqs"] == 63
    # The tail policy of this run reaches the cost model, which reports it in its metadata.
    assert spec["model_options"]["tail_padding"] == "append_dummy_seq"
    assert config._dynamic_cp_runtime["coverage_checked"]


@pytest.mark.parametrize(
    "overrides,allowed",
    [
        ({"thd_tail_padding_policy": None}, (1, 2, 4, 8, 16, 32)),
        ({"pad_packed_seq_alignment": 16384}, (1, 2, 4, 8, 16, 32)),
        ({"thd_max_packed_sequences": None}, (1, 2, 4, 8, 16, 32)),
        ({"context_parallel_size": 1}, (1, 2, 4, 8, 16, 32)),
        ({"attention_backend": AttnBackend.fused}, (1, 2, 4, 8, 16, 32)),
        ({"linear_cp_mode": None}, (1, 2, 4, 8, 16, 32)),
        ({"min_dynamic_context_parallel_size": 8}, (8, 16, 32)),
    ],
)
def test_start_up_accepts_the_executions_the_parameters_describe(
    monkeypatch, tmp_path, overrides, allowed
):
    spec = _start_up(monkeypatch, _training_config(monkeypatch, tmp_path, **overrides))
    assert spec["model"] == "qwen35_full"
    assert spec["geometry"]["allowed_cp_sizes"] == allowed
    assert spec["model_options"]["tail_padding"] == "append_dummy_seq"


@pytest.mark.parametrize(
    "overrides,message",
    [
        # Execution the module parameters do not describe.
        ({"cp_partition_mode": "zigzag"}, "set cp_partition_mode='contiguous' (got 'zigzag')"),
        (
            {"pad_packed_seq_alignment": None, "thd_tail_padding_policy": None},
            "set pad_packed_seq_alignment to 'max' or 16384 (got None)",
        ),
        (
            {"pad_packed_seq_alignment": 128},
            "set pad_packed_seq_alignment to 'max' or 16384 (got 128)",
        ),
        (
            {
                "linear_cp_mode": "headwise",
                "cp_partition_mode": "zigzag",
                "context_parallel_size": 1,
            },
            "set linear_cp_mode='chunkwise' (got 'headwise')",
        ),
        (
            {"gdn_kernel_backend": "torch"},
            "require the FLA GDN backend: gdn_kernel_backend='fla', got 'torch'",
        ),
        ({"attention_backend": AttnBackend.flash}, "'auto' or 'fused' (got 'flash')"),
        ({"attention_backend": AttnBackend.unfused}, "'auto' or 'fused' (got 'unfused')"),
        (
            {"cp_comm_type": "a2a"},
            "require P2P attention communication: cp_comm_type='p2p', got 'a2a'",
        ),
        (
            {"recompute_granularity": "selective"},
            "eager execution without recomputation: cuda_graph_impl='none' and "
            "recompute_granularity unset, got 'none' and 'selective'",
        ),
        (
            {"bf16": False},
            "describe BF16 execution: bf16=True without fp8 or fp4, got {'bf16': False, ",
        ),
        ({"attention_dropout": 0.1}, "require attention_dropout=0.0, got 0.1"),
        ({"attention_dropout": True}, "require attention_dropout=0.0, got True"),
        # The parallel geometry: the first differing setting, measured and actual value.
        (
            {"expert_model_parallel_size": 4},
            "does not match the training configuration: Full Qwen3.5-35B parameters were "
            "measured with expert_model_parallel_size=32, got 4",
        ),
        (
            {"max_seqlen_per_dp_cp_rank": 8192},
            "does not match the training configuration: Full Qwen3.5-35B parameters were "
            "measured with max_seqlen_per_dp_cp_rank=16384, got 8192",
        ),
        # A different model.
        ({"num_layers": 4}, "require num_layers=40, got 4"),
        ({"mtp_num_layers": 0}, "require mtp_num_layers=1, got 0"),
        ({"num_attention_heads": 32}, "require num_attention_heads=16, got 32"),
        ({"num_query_groups": 4}, "require num_query_groups=2, got 4"),
        ({"hidden_size": 4096}, "require hidden_size=2048, got 4096"),
        ({"qk_layernorm": False}, "require qk_layernorm=True, got False"),
        ({"attention_output_gate": False}, "require attention_output_gate=True, got False"),
        ({"linear_conv_kernel_dim": 3}, "require linear_conv_kernel_dim=4, got 3"),
        (
            {"linear_attention_freq": 3},
            "require linear_attention_freq=4 (30 GDN and 10 FA decoder layers in 3:1 order), "
            "got 3",
        ),
        (
            {"linear_attention_freq": [1, 1, 0, 1] * 10},
            "require linear_attention_freq=4 (30 GDN and 10 FA decoder layers in 3:1 order), "
            "got [1, 1, 0, 1, ",
        ),
        ({"linear_num_key_heads": 8}, "require linear_num_key_heads=16, got 8"),
        ({"linear_num_value_heads": 16}, "require linear_num_value_heads=32, got 16"),
        (
            {"mtp_use_repeated_layer": True},
            "require a distinct FA MTP layer: mtp_use_repeated_layer=False",
        ),
    ],
)
def test_start_up_rejects_a_training_configuration_the_parameters_do_not_describe(
    monkeypatch, tmp_path, overrides, message
):
    config = _training_config(monkeypatch, tmp_path, **overrides)
    with pytest.raises(ValueError) as caught:
        _start_up(monkeypatch, config)
    assert message in str(caught.value)
    assert str(config.dynamic_cp_runtime_config) in str(caught.value)


def test_start_up_names_dpxcp_and_the_hardware_key(monkeypatch, tmp_path):
    config = _training_config(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as caught:
        _start_up(monkeypatch, config, world_size=64)
    assert (
        "does not match the training configuration: Full Qwen3.5-35B parameters were "
        "measured with DPxCP=32, got 64"
    ) in str(caught.value)
    # "hardware" is a key of the runtime file, so its fault is the file's.
    config = _training_config(monkeypatch, tmp_path)
    path = Path(config.dynamic_cp_runtime_config)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), hardware="GB200")))
    with pytest.raises(ValueError) as caught:
        _start_up(monkeypatch, config)
    assert "\"hardware\" must be \"gb200\" for full Qwen3.5, got 'GB200'" in str(caught.value)
    assert "training configuration" not in str(caught.value)


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("tensor_model_parallel_size", 2, "require TP1/PP1: tensor_model_parallel_size=1, got 2"),
        (
            "pipeline_model_parallel_size",
            2,
            "require TP1/PP1: pipeline_model_parallel_size=1, got 2",
        ),
        (
            "virtual_pipeline_model_parallel_size",
            2,
            "require VPP1: virtual_pipeline_model_parallel_size unset, got 2",
        ),
        (
            "cuda_graph_impl",
            "transformer_engine",
            "recompute_granularity unset, got 'transformer_engine' and None",
        ),
        ("cuda_graph_impl", "full_iteration", "eager execution without recomputation"),
        ("recompute_granularity", "full", "recompute_granularity unset, got 'none' and 'full'"),
        ("fp8", "hybrid", "bf16=True without fp8 or fp4, got {'bf16': True, 'fp8': 'hybrid', "),
        ("is_hybrid_model", True, "require the GPT GDN/GQA layer configuration"),
    ],
)
def test_model_check_rejects_settings_the_config_class_does_not_combine_with_this_run(
    monkeypatch, tmp_path, name, value, message
):
    # TransformerConfig itself refuses these together with this configuration or needs more
    # settings for them, so change one attribute on a copy of the real post-init values.
    values = SimpleNamespace(**vars(_training_config(monkeypatch, tmp_path)))
    validate_full_qwen35_config(values)
    setattr(values, name, value)
    with pytest.raises(ValueError) as caught:
        validate_full_qwen35_config(values)
    assert message in str(caught.value)


def test_another_gdn_variant_is_not_priced_with_the_full_qwen35_parameters(monkeypatch, tmp_path):
    # TransformerConfig builds no gdn2 model with dynamic CP, so change a copy of the real
    # values. With the tail policy and the attention backend the ordinary-GQA parameters
    # describe, the variant is the only setting left for the ordinary route to reject.
    values = SimpleNamespace(**vars(_training_config(monkeypatch, tmp_path)))
    values.experimental_attention_variant = "gdn2"
    values.thd_tail_padding_policy = "extend_last"
    values.attention_backend = AttnBackend.fused
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(values, 32)
    assert (
        "does not match the training configuration: Ordinary GQA parameters cannot represent "
        "another attention architecture"
    ) in str(caught.value)


def test_single_json_auto_selects_full_qwen35_from_training_configuration(monkeypatch, tmp_path):
    config = _training_config(monkeypatch, tmp_path)
    spec = prefetch.runtime_spec(config, 32)
    assert spec["model"] == "qwen35_full"
    assert spec["type"] == "dcp_benchmark" and "solver" not in spec
    assert spec["geometry"]["ep_size"] == 32
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8, 16, 32)
    assert "query_heads" not in spec["model_options"]
    assert "layer_count" not in spec["model_options"]
    assert isinstance(
        make_model(spec["model"], _geometry(), **spec["model_options"]), Qwen35FullCostModel
    )


@pytest.mark.parametrize("max_num_seqs,cap", [(None, 64), (64, 64), (63, 63)])
def test_full_qwen35_runtime_accepts_the_real_sequence_cap(
    monkeypatch, tmp_path, max_num_seqs, cap
):
    # thd_max_packed_sequences=64 with an appended dummy tail leaves 63 real sequences.
    spec = prefetch.runtime_spec(
        _training_config(monkeypatch, tmp_path), 32, max_num_seqs=max_num_seqs
    )
    assert spec["model"] == "qwen35_full"
    assert spec["geometry"]["max_group_seqs"] == cap


def test_full_model_internal_factory_name_is_not_a_public_scheduler_type(monkeypatch, tmp_path):
    config = _training_config(monkeypatch, tmp_path)
    path = Path(config.dynamic_cp_runtime_config)
    payload = json.loads(path.read_text())
    payload["type"] = "qwen35_full"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="'type' must be one of") as caught:
        prefetch.runtime_spec(config, 32)
    assert "'dcp_benchmark'" in str(caught.value)


def _with_non_attention(config, forward, backward):
    path = Path(config.dynamic_cp_runtime_config)
    payload = json.loads(path.read_text())
    payload["non_attention"] = _non_attention(forward, backward)
    path.write_text(json.dumps(payload))
    return config


def test_json_layer_types_are_gdn_and_gqa_with_an_optional_mtp(monkeypatch, tmp_path):
    config = _with_non_attention(
        _training_config(monkeypatch, tmp_path),
        {"gdn": 0.5, "gqa": 1.0, "mtp": 3.0},
        {"gdn": 1.5, "gqa": 3.0},
    )
    spec = prefetch.runtime_spec(config, 32)
    model = make_model(spec["model"], _geometry(), **spec["model_options"])
    costs = model.program_costs((1025,), 1, ranks=(0,))
    names = [layer["name"] for layer in model.layers]
    assert costs[names[0] + ":nonattention:forward"] == 0.5  # GDN
    assert costs[names[3] + ":nonattention:forward"] == 1.0  # Full attention
    assert costs[names[-1] + ":nonattention:forward"] == 3.0  # MTP
    assert costs[names[-1] + ":nonattention:backward"] == 3.0  # MTP falls back to gqa


@pytest.mark.parametrize(
    "forward,got",
    [
        ({"gdn": 0.5, "fa": 1.0}, "['fa', 'gdn']"),
        ({"gdn": 0.5}, "['gdn']"),
        ({"gqa": 1.0}, "['gqa']"),
        ({"gdn": 0.5, "gqa": 1.0, "mlp": 2.0}, "['gdn', 'gqa', 'mlp']"),
        (
            {"gdn": 0.5, "gqa": 1.0, "layer:0:module.module.decoder.layers.0": 2.0},
            "['gdn', 'gqa', 'layer:0:module.module.decoder.layers.0']",
        ),
    ],
)
def test_json_names_the_layer_types_the_full_model_expects(monkeypatch, tmp_path, forward, got):
    config = _with_non_attention(_training_config(monkeypatch, tmp_path), forward, 1.0)
    with pytest.raises(ValueError) as caught:
        prefetch.runtime_spec(config, 32)
    assert (
        "non_attention.forward_ms_per_layer must be one number, or one number per layer type "
        f"['gdn', 'gqa'] (optionally also ['mtp']); got {got}"
    ) in str(caught.value)
    assert "training configuration" not in str(caught.value)


def test_full_model_prices_every_allowed_cp_size(monkeypatch, tmp_path):
    spec = prefetch.runtime_spec(_training_config(monkeypatch, tmp_path), 32)
    assert spec["geometry"]["allowed_cp_sizes"] == (1, 2, 4, 8, 16, 32)
    assert prefetch._coverage_error(spec) is None


def test_full_model_json_requires_explicit_hardware(monkeypatch, tmp_path):
    config = _training_config(monkeypatch, tmp_path)
    path = Path(config.dynamic_cp_runtime_config)
    payload = json.loads(path.read_text())
    del payload["hardware"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hardware"):
        prefetch.runtime_spec(config, 32)


@pytest.mark.parametrize("frequency", [4, [1, 1, 1, 0] * 10])
@pytest.mark.parametrize("variant", ["gdn", "gated_delta_net"])
def test_full_model_recognizes_equivalent_training_layer_layouts(
    monkeypatch, tmp_path, frequency, variant
):
    config = _training_config(
        monkeypatch,
        tmp_path,
        linear_attention_freq=frequency,
        experimental_attention_variant=variant,
    )
    assert prefetch.runtime_spec(config, 32)["model"] == "qwen35_full"


def test_real_worker_loads_full_41_layer_model_without_cuda_or_external_packages(
    monkeypatch, tmp_path
):
    config = _training_config(monkeypatch, tmp_path, min_dynamic_context_parallel_size=32)
    spec = prefetch.runtime_spec(config, 32)
    spec["solver_options"] = dict(max_scored_candidates=1, time_budget_s=0.5)
    worker = prefetch._CpuPlanner(spec)
    try:
        result = worker.solve([32768], [32768], [[[0]] * 32])
    finally:
        worker.close()
    assert result["predicted_ms"] > 0
    assert result["model_metadata"]["name"] == "dcp_benchmark_qwen35_full35b"
    assert result["model_metadata"]["hardware"] == "gb200"
    assert result["model_metadata"]["nvlink_domain_size"] == 72
    assert result["model_metadata"]["layer_count"] == 41
    # Every plan record states the tail policy the run executes and the source's own.
    assert resolve_thd_tail_padding_policy(config) == "append_dummy_seq"
    assert result["model_metadata"]["executed_tail_padding"] == "append_dummy_seq"
    assert result["model_metadata"]["source_tail_padding"] == "extend_last"
    assert not result["worker_cuda_initialized"]
    assert not result["worker_distributed_initialized"]
    assert not result["external_scheduler_imported"]
    assert "lightweight" in result["solver_entry"]


def test_full_model_runs_through_the_lightweight_solver():
    geometry = _geometry((32,))
    model = _model(geometry)
    planner = LightweightRuntimePlanner(
        geometry, model, LightweightOptions(max_scored_candidates=1)
    )
    plan = planner.plan_padded({0: 32768}, logical_lengths={0: 32768}, incumbent_rows=[[[0]] * 32])
    assert plan.predicted_ms > 0
    assert plan.rows == [[[0]] * 32]


def test_full_model_plans_within_a_63_sequence_cap():
    geometry = replace(_geometry((32,)), max_group_seqs=63)
    planner = LightweightRuntimePlanner(
        geometry, _model(geometry), LightweightOptions(max_scored_candidates=1)
    )
    lengths = dict.fromkeys(range(64), 64)
    plan = planner.plan_padded(lengths, logical_lengths=lengths)
    assert plan.microbatches == 2
    assert max(len(ids) for row in plan.groups for _, _, ids in row) <= 63
