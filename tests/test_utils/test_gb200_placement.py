# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only checks for CI placement; never import the GPU test modules."""

import ast
import enum
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests/unit_tests/gb200_placement.py"
spec = importlib.util.spec_from_file_location("gb200_placement", MODULE_PATH)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


class Recipe(str, enum.Enum):
    tensorwise = "tensorwise"
    delayed = "delayed"
    mxfp8 = "mxfp8"


def selected(path, node="test_case", **params):
    return placement.is_gb200_candidate("tests/unit_tests/" + path, node, params)


def test_complete_bucket_manifest_matches_approved_jobs():
    assert placement.FULL_BUCKETS == tuple(
        "tests/unit_tests/" + path
        for path in (
            "ssm/test_gated_delta_net2.py",
            "ssm/test_gated_delta_net2_parallel.py",
            "ssm/test_gated_delta_net_recompute.py",
            "ssm/test_gated_delta_net_padding.py",
            "ssm/test_gated_delta_net_parallel.py",
            "inference/test_coordinator_sampling_pairwise.py",
            "inference/test_coordinator_media_pairwise.py",
            "inference/test_coordinator_execution_pairwise.py",
            "inference/test_coordinator_schedule_pairwise.py",
            "ops/**/*.py",
            "test_model_configs.py",
        )
    )
    assert len(placement.NEW_GB200_BUCKETS) == len(set(placement.NEW_GB200_BUCKETS)) == 18


def test_host_launcher_loads_placement_without_torch():
    # A fresh interpreter catches package __init__ imports hidden by the test
    # process's module cache. The host launcher runs before the GPU container.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys

sys.modules["torch"] = None
from tests.test_utils.python_scripts import launch_nemo_run_workload

assert "tests.unit_tests" not in sys.modules
print(json.dumps(launch_nemo_run_workload.NEW_GB200_BUCKETS))
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == list(placement.NEW_GB200_BUCKETS)


def test_candidate_manifest_matches_source_files_and_is_independent_of_cwd(tmp_path, monkeypatch):
    expected = placement.candidate_files()
    assert expected
    assert all((ROOT / path).is_file() for path in expected)
    assert all(Path(path).name.startswith("test_") for path in expected)
    assert all("distributed/mfsdp" not in path for path in expected)
    expected_ops = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests/unit_tests/ops").rglob("test_*.py")
    }
    assert expected_ops <= expected
    monkeypatch.chdir(tmp_path)
    assert placement.candidate_files() == expected
    for path in expected_ops:
        assert placement.is_gb200_candidate(path, "test_case", {})
        assert placement.is_gb200_candidate(str(ROOT / path), "test_case", {})
    assert not placement.is_gb200_candidate(str(tmp_path / "test_case.py"), "test_case", {})
    assert not placement.is_gb200_candidate("../test_case.py", "test_case", {})
    assert not selected("ops/helpers.py")
    assert not selected("test_unknown.py")


@pytest.mark.parametrize("path", sorted(placement.WHOLE_FILES))
def test_approved_whole_files_are_selected(path):
    assert selected(path)


@pytest.mark.parametrize(
    "path,node,params,expected",
    [
        ("ssm/test_kda.py", "test_kda_forward_backward[True]", {}, True),
        ("ssm/test_kda.py", "test_parallel_kda_correctness", {}, True),
        ("ssm/test_kda.py", "test_kda_fp8_parameter_initialization", {}, False),
        ("ssm/test_gated_delta_product.py", "test_in_proj_recompute_parity", {}, True),
        ("ssm/test_gated_delta_product.py", "test_in_proj_recompute_rejects_qkv_offload", {}, True),
        (
            "ssm/test_gated_delta_product.py",
            "test_in_proj_recompute_preserves_norm_gate_view[renamed-id]",
            {"fp8_recipe": None},
            True,
        ),
        (
            "ssm/test_gated_delta_product.py",
            "test_in_proj_recompute_preserves_norm_gate_view[None]",
            {"fp8_recipe": "mxfp8"},
            False,
        ),
        (
            "ssm/test_gated_delta_product.py",
            "test_in_proj_recompute_preserves_norm_gate_view",
            {},
            False,
        ),
        ("ssm/test_hybrid_block.py", "TestHybridBlock::test_gpu_forward", {}, True),
        (
            "ssm/test_hybrid_block.py",
            "TestHybridBlock::test_two_stage_attention_matches_atomic_forward_bitwise[opaque]",
            {"compute_config": {"bf16": True}},
            True,
        ),
        (
            "ssm/test_hybrid_block.py",
            "TestHybridBlock::test_shortcut_pair_eager_forward_backward[attention]",
            {"compute_config": {"bf16": True, "fp8": "e4m3"}},
            False,
        ),
        (
            "ssm/test_hybrid_block.py",
            "TestHybridBlock::test_shortcut_pair_eager_forward_backward",
            {},
            False,
        ),
        ("ssm/test_hybrid_block.py", "test_unapproved_helper", {}, False),
        (
            "ssm/test_gdp_dynamic_inference.py",
            "TestGDPDynamicInference::test_decode_step_matches_recompute",
            {},
            True,
        ),
        (
            "ssm/test_gdp_dynamic_inference.py",
            "TestGDPDynamicInferenceEngine::test_engine_runs_to_completion",
            {},
            True,
        ),
        (
            "ssm/test_gdp_dynamic_inference.py",
            "TestGDPDynamicInferenceEngine::test_engine_greedy_matches_golden",
            {},
            False,
        ),
        (
            "ssm/test_gdp_dynamic_inference.py",
            "TestGDPSpeculativeDecode::test_cuda_graph_capture_and_replay",
            {},
            False,
        ),
        (
            "inference/engines/test_dynamic_engine.py",
            "TestDynamicInferenceEngine::test_simple[gpt]",
            {},
            False,
        ),
        (
            "inference/engines/test_dynamic_engine.py",
            "TestDynamicInferenceEngine::test_fp8_inference[gpt]",
            {},
            True,
        ),
        ("transformer/test_cuda_graphs.py", "TestPartialCudaGraph::test_case", {}, False),
        ("transformer/test_cuda_graphs.py", "TestCudaGraphReplay::test_case", {}, True),
        ("transformer/test_attention.py", "TestSelfAttention::test_case", {}, False),
        ("transformer/test_attention.py", "TestParallelAttention::test_case", {}, True),
        (
            "transformer/test_transformer_layer.py",
            "TestMHCRecomputeMemorySaving::test_case",
            {},
            False,
        ),
        (
            "transformer/test_transformer_layer.py",
            "TestParallelTransformerLayer::test_sharded_state_dict[opaque]",
            {"tp_pp": (2, 2)},
            True,
        ),
        (
            "transformer/test_transformer_layer.py",
            "TestParallelTransformerLayer::test_sharded_state_dict[opaque]",
            {"tp_pp": (4, 2)},
            False,
        ),
        (
            "transformer/test_transformer_layer.py",
            "TestParallelTransformerLayer::test_sharded_state_dict[opaque]",
            {"tp_pp": (8, 1)},
            False,
        ),
        (
            "transformer/test_transformer_layer.py",
            "TestParallelTransformerLayer::test_sharded_state_dict[opaque]",
            {},
            False,
        ),
        ("models/test_hybrid_model.py", "TestHybridModel::test_forward", {}, True),
        ("models/test_hybrid_model.py", "test_hybrid_model_with_custom_process_groups", {}, False),
        (
            "models/test_hybrid_model.py",
            "TestHybridWithDynamicInference::test_dynamic_inference_padding_with_fp8",
            {},
            False,
        ),
        (
            "generalized_tensor_parallel/test_gtp_muon_dcp.py",
            "TestGTPMuonDCP::test_gtp_muon_moe_save_load",
            {},
            True,
        ),
        (
            "generalized_tensor_parallel/test_gtp_muon_dcp.py",
            "TestGTPMuonDCP::test_gtp_muon_moe_native_fp8_save_load",
            {},
            False,
        ),
        ("generalized_tensor_parallel/test_gtp_symmetric_memory.py", "test_case", {}, False),
        ("distributed/mfsdp_v2/test_fully_shard.py", "test_case", {}, False),
        ("pipeline_parallel/test_fine_grained_activation_offloading.py", "test_case", {}, False),
    ],
)
def test_approved_method_boundaries(path, node, params, expected):
    assert selected(path, node, **params) is expected


@pytest.mark.parametrize("recompute_method", ["uniform", "block"])
@pytest.mark.parametrize("learned_output_contract", [False, True])
@pytest.mark.parametrize(
    "precision,autocast_forward,expected",
    [
        ("fp32", False, False),
        ("bf16_fp32_mixing", False, True),
        ("bf16_fused", False, True),
        ("bf16_fp32_mixing", True, True),
    ],
)
def test_hybrid_mhc_recompute_keeps_failed_fp32_cases_on_h100(
    recompute_method, learned_output_contract, precision, autocast_forward, expected
):
    assert (
        selected(
            "models/test_hybrid_mhc.py",
            "TestHybridStackMHC::test_full_recompute_matches_forward_backward[opaque]",
            recompute_method=recompute_method,
            learned_output_contract=learned_output_contract,
            precision=precision,
            autocast_forward=autocast_forward,
        )
        is expected
    )


@pytest.mark.parametrize("recompute_method", ["uniform", "block"])
def test_hybrid_mhc_real_moe_recompute_stays_on_h100(recompute_method):
    assert not selected(
        "models/test_hybrid_mhc.py",
        "TestHybridStackMHC::test_full_recompute_real_moe_matches_forward_backward[opaque]",
        recompute_method=recompute_method,
    )


def test_hybrid_mhc_other_cases_still_run_on_gb200():
    path = "models/test_hybrid_mhc.py"
    assert selected(path, "test_mhc_mtp_requires_hybrid_contract")
    assert selected(path, "TestHybridStackMHC::test_fused_bf16_forward_backward")
    assert selected(
        path,
        "TestHybridStackMHC::test_forward_backward[False-full]",
        learned_output_contract=False,
        recompute_kwargs={
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    )
    assert not selected(path, "TestHybridStackMHC::test_full_recompute_matches_forward_backward")


@pytest.mark.parametrize("implementation", ["local", "inference_optimized"])
@pytest.mark.parametrize("sizes", [(1, 1, 1), (2, 2, 1), (2, 1, 2), (2, 2, 2)])
def test_dynamic_parallel_inference_preserves_four_rank_limit(implementation, sizes):
    path = "inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py"
    params = dict(zip(("tp_size", "pp_size", "ep_size"), sizes))
    params["transformer_impl"] = implementation
    expected = implementation == "local" and sizes != (2, 2, 2)
    assert (
        selected(path, "TestDynamicInferenceEngineParallel::test_parallel_inference", **params)
        is expected
    )


@pytest.mark.parametrize(
    "node,params,expected",
    [
        ("TestChunkedPrefillCudaGraphs::test_chunked_prefill_cuda_graphs", {}, True),
        ("TestDynamicInferenceEngineParallel::test_gdn_tensor_parallel", {}, True),
        ("TestDynamicInferenceEngineParallel::test_sequence_parallel_fp8_inference", {}, False),
        ("TestDynamicInferenceEngineParallel::test_mtp_kv_cache_inference_optimized", {}, False),
        (
            "TestDynamicInferenceEngineParallel::test_mtp_kv_cache_inference_optimized_ep_prefix_caching",
            {},
            False,
        ),
        (
            "TestDynamicInferenceEngineParallel::test_mtp_kv_cache_chunked_prefill[opaque]",
            {"transformer_impl": "local", "ep_size": 2, "dispatcher": "nccl"},
            True,
        ),
        (
            "TestDynamicInferenceEngineParallel::test_mtp_kv_cache_chunked_prefill[local-ep]",
            {"transformer_impl": "inference_optimized", "ep_size": 2, "dispatcher": "nvls"},
            False,
        ),
        ("TestDynamicInferenceEngineParallel::test_parallel_inference", {}, False),
        ("TestUnapproved::test_case", {}, False),
    ],
)
def test_dynamic_engine_special_cases(node, params, expected):
    assert (
        selected(
            "inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py", node, **params
        )
        is expected
    )


@pytest.mark.parametrize(
    "file,method",
    [
        ("test_schedule_chunk_1f1b.py", "TestA2AOverlap::test_1f1b_schedule_model_chunk"),
        ("test_schedule_layer_1f1b.py", "TestA2AOverlap::test_transformer_layer_overlap"),
        (
            "test_delay_wgrad_compute.py",
            "TestDelayWgradCompute::test_overlap_dispatch_backward_with_experts_wgrad",
        ),
    ],
)
@pytest.mark.parametrize(
    "flag,expected",
    [
        (None, True),
        (("e4m3", Recipe.tensorwise), True),
        (("e4m3", Recipe.mxfp8), False),
        (("e4m3", Recipe.delayed), False),
    ],
)
def test_a2a_precision_selection_uses_values_not_generated_ids(file, method, flag, expected):
    assert selected("a2a_overlap/" + file, method + "[fp8_flag0]", fp8_flag=flag) is expected


def test_a2a_retains_zero_copy_and_eight_rank_padding():
    assert not selected(
        "a2a_overlap/test_schedule_layer_1f1b.py",
        "TestA2AOverlap::test_transformer_layer_overlap_zero_copy",
        wire_dtype="bf16",
    )
    assert not selected(
        "a2a_overlap/test_delay_wgrad_compute.py",
        "TestDelayWgradCompute::test_overlap_dispatch_backward_with_experts_wgrad_with_fsdp",
    )
    for size in (1, 2, 4, 8):
        assert selected(
            "a2a_overlap/test_schedule_chunk_1f1b.py",
            "TestA2AOverlapPaddingMask::test_1f1b_schedule_model_chunk_with_padding_mask",
            tp_size=size,
        ) is (size <= 4)


def test_unreviewed_classes_and_missing_precision_stay_on_h100():
    for path in placement.APPROVED_CLASSES:
        assert not selected(path, "TestNewHardwareCoverage::test_case")
    for path, method in (
        ("test_schedule_chunk_1f1b.py", "TestA2AOverlap::test_1f1b_schedule_model_chunk"),
        ("test_schedule_layer_1f1b.py", "TestA2AOverlap::test_transformer_layer_overlap"),
        (
            "test_delay_wgrad_compute.py",
            "TestDelayWgradCompute::test_overlap_dispatch_backward_with_experts_wgrad",
        ),
    ):
        assert not selected("a2a_overlap/" + path, method)
        assert not selected("a2a_overlap/" + path, "TestA2AOverlap::test_new_case", fp8_flag=None)
    assert selected(
        "a2a_overlap/test_schedule_layer_1f1b.py",
        "TestA2AOverlap::test_transformer_layer_overlap_dense",
    )


@pytest.mark.parametrize("method", ["alltoall", "deepep", "hybridep", "ncclep"])
@pytest.mark.parametrize(
    "layout,move",
    [
        ((False, False, False), False),
        ((True, False, False), True),
        ((False, True, False), True),
        ((True, True, False), True),
        ((False, True, True), True),
        ((True, True, True), False),
    ],
)
def test_moe_retains_representative_h100_layouts(method, layout, move):
    params = dict(zip(("single_grouped_weight", "use_bias", "single_grouped_bias"), layout))
    assert selected(
        "transformer/moe/test_grouped_tensor_dispatcher_numerics.py",
        f"TestGroupedTensorDispatcherNumerics::test_{method}_grouped_tensor_moe_parity[opaque]",
        **params,
    ) is (move and method != "ncclep")


def test_partial_selection_names_still_exist_in_current_sources():
    # Catch stale/renamed selectors without importing GPU-dependent test modules.
    tree = ast.parse(MODULE_PATH.read_text())
    referenced = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value.startswith("test_") or node.value.startswith("Test"))
        and "/" not in node.value
        and not node.value.endswith(".py")
        and node.value != "test_"
    }
    definitions = set()
    for path in placement.PARTIAL_FILES:
        source = ast.parse((ROOT / "tests/unit_tests" / path).read_text())
        definitions.update(
            node.name
            for node in ast.walk(source)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
    assert referenced <= definitions


@pytest.mark.parametrize("platform", ["h100", "gb200"])
@pytest.mark.parametrize("enabled", [True, False])
def test_matrix_filter_cli_preserves_residual_jobs_and_rollback(platform, enabled):
    matrix = [
        {"bucket": bucket, "metadata": {"index": index}}
        for index, bucket in enumerate(placement.NEW_GB200_BUCKETS)
    ] + [
        {"bucket": "tests/unit_tests/generalized_tensor_parallel/**/*.py", "existing": True},
        {"bucket": "tests/unit_tests/distributed/mfsdp_v2/**/*.py", "existing": True},
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "filter-buckets",
            "--platform",
            platform,
            "--enabled",
            str(enabled).lower(),
        ],
        input=json.dumps(matrix),
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    if platform == "h100" and enabled:
        expected = [entry for entry in matrix if entry["bucket"] not in placement.FULL_BUCKETS]
    elif platform == "gb200" and not enabled:
        expected = matrix[-2:]
    else:
        expected = matrix
    assert actual == expected
