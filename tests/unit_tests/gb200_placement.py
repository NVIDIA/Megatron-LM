# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement of portable unit tests on the existing four-GPU GB200 lane.

This module deliberately has no GPU or pytest dependencies: matrix generation,
file discovery, and pytest collection must use the same selection. Placement is
additive until the caller enables the migration; existing platform markers and
device-support skips remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "tests/unit_tests/"

# These complete H100 jobs can disappear once exclusive placement is enabled.
# Other buckets contain retained H100 cases and must remain in its matrix.
FULL_BUCKETS = (
    PREFIX + "ssm/test_gated_delta_net2.py",
    PREFIX + "ssm/test_gated_delta_net2_parallel.py",
    PREFIX + "ssm/test_gated_delta_net_recompute.py",
    PREFIX + "ssm/test_gated_delta_net_padding.py",
    PREFIX + "ssm/test_gated_delta_net_parallel.py",
    PREFIX + "inference/test_coordinator_sampling_pairwise.py",
    PREFIX + "inference/test_coordinator_media_pairwise.py",
    PREFIX + "inference/test_coordinator_execution_pairwise.py",
    PREFIX + "inference/test_coordinator_schedule_pairwise.py",
    PREFIX + "ops/**/*.py",
    PREFIX + "test_model_configs.py",
)

# The existing GB200 buckets are never removed by the migration rollback.
NEW_GB200_BUCKETS = FULL_BUCKETS + (
    PREFIX + "ssm/**/*.py",
    PREFIX + "inference/engines/test_dynamic_engine.py",
    PREFIX + "inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py",
    PREFIX + "transformer/**/*.py",
    PREFIX + "a2a_overlap/**/*.py",
    PREFIX + "models/**/*.py",
    PREFIX + "pipeline_parallel/**/*.py",
)

WHOLE_FILES = frozenset(
    (
        "ssm/test_kda_gate_precision.py",
        "ssm/test_gdp_packed_seq.py",
        "transformer/test_mhc_precision.py",
        "transformer/test_mhc_projection_recompute.py",
        "transformer/test_hyper_connection_recompute.py",
        "models/test_puzzle_config.py",
        "models/test_wide_residual_mtp_mimo.py",
        "generalized_tensor_parallel/test_attention_gtp.py",
        "generalized_tensor_parallel/test_gtp_grad_correctness.py",
        "generalized_tensor_parallel/test_gtp_cp_overlap.py",
        "generalized_tensor_parallel/test_gtp_loss_correctness.py",
        "generalized_tensor_parallel/test_gtp_muon.py",
        "generalized_tensor_parallel/test_gtp_muon_qkv.py",
        "generalized_tensor_parallel/test_wide_shortcut_parallel.py",
        "pipeline_parallel/test_pp_mhc_compatibility.py",
        "pipeline_parallel/test_pp_mhc_dsv4_shapes.py",
    )
)

PARTIAL_FILES = frozenset(
    (
        "ssm/test_kda.py",
        "ssm/test_gated_delta_product.py",
        "ssm/test_hybrid_block.py",
        "ssm/test_gdp_dynamic_inference.py",
        "inference/engines/test_dynamic_engine.py",
        "inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py",
        "transformer/test_cuda_graphs.py",
        "transformer/test_attention.py",
        "transformer/test_transformer_layer.py",
        "a2a_overlap/test_schedule_chunk_1f1b.py",
        "a2a_overlap/test_schedule_layer_1f1b.py",
        "a2a_overlap/test_delay_wgrad_compute.py",
        "transformer/moe/test_grouped_tensor_dispatcher_numerics.py",
        "models/test_hybrid_mhc.py",
        "models/test_hybrid_model.py",
        "generalized_tensor_parallel/test_gtp_muon_dcp.py",
    )
)

# These files were approved broadly, with the exclusions below. New test
# classes still require an audit instead of inheriting exclusive placement.
APPROVED_CLASSES = {
    "inference/engines/test_dynamic_engine.py": {
        "TestDynamicInferenceEngine",
        "TestGDNDynamicInferenceEngine",
    },
    "transformer/test_cuda_graphs.py": {
        "TestCudaGraphConfigAndArguments",
        "TestCudaGraphReplay",
        "TestParallelTransformerBlockCudagraphs",
        "TestPackedSeqCudagraphs",
        "TestLLaVACudaGraph",
        "TestParallelHybridBlockCudagraphs",
        "TestTECudaGraphHelper",
        "TestCheckpointParameterDiscovery",
        "TestRepeatedParameterCapture",
        "TestInlineCaptureManager",
        "TestSkipFp8WeightUpdateTensor",
    },
    "transformer/test_attention.py": {
        "TestParallelAttention",
        "TestClipQK",
        "TestFlashDecodeSoftcapPlumbing",
        "TestFlashAttention3SoftcapWrapper",
        "TestFlashDecodeAndPrefillSoftcap",
        "TestFlashMLASoftcapRejected",
    },
    "transformer/test_transformer_layer.py": {
        "TestParallelTransformerLayer",
        "TestTransformerLayerCudaGraphManagers",
        "TestTransformerLayerWithHyperConnectionRecompute",
        "TestMHCWithCudaGraph",
        "TestMHCWithOffloading",
    },
    "models/test_hybrid_mhc.py": {"TestHybridStackMHC"},
    "models/test_hybrid_model.py": {
        "TestHybridModel",
        "TestHybridQKLayernorm",
        "TestHybridMLAQKLayernorm",
        "TestHybridDSAQKLayernorm",
        "TestMLAQKNormSpecValidation",
        "TestMLAQKNormResolution",
        "TestDSAQKNormResolution",
        "TestMLADownProjFusion",
        "TestHybridModelWithYarn",
    },
}


def _relative_path(path: str) -> str:
    """Normalize collected absolute paths without accepting another checkout."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return ""


def candidate_files() -> set[str]:
    """Return repository-relative test files needed to collect GB200 candidates."""
    files = {PREFIX + path for path in WHOLE_FILES | PARTIAL_FILES}
    for bucket in FULL_BUCKETS:
        if "*" in bucket:
            files.update(
                path.relative_to(ROOT).as_posix()
                for path in ROOT.glob(bucket)
                if path.is_file() and path.name.startswith("test_")
            )
        else:
            files.add(bucket)
    return files


def _bf16_or_tensorwise(params: dict) -> bool:
    # Do not depend on pytest's fp8_flag0 IDs, or treat missing precision
    # metadata as a BF16 case. Fixed-precision methods are selected separately.
    if "fp8_flag" not in params:
        return False
    flag = params["fp8_flag"]
    return flag is None or (
        isinstance(flag, (tuple, list)) and len(flag) == 2 and flag[1] == "tensorwise"
    )


def is_gb200_candidate(path: str, node_name: str, params: dict) -> bool:
    """Select a collected item using semantic parameters, not generated IDs.

    ``node_name`` is the nodeid suffix after ``file::`` and ``params`` is
    ``item.callspec.params`` (or an empty dictionary for unparametrized tests).
    """
    relative = _relative_path(path)
    if not relative.startswith(PREFIX):
        return False
    path = relative[len(PREFIX) :]
    if relative in FULL_BUCKETS or (
        path.startswith("ops/") and Path(path).name.startswith("test_") and path.endswith(".py")
    ):
        return True
    if path in WHOLE_FILES:
        return True
    if path not in PARTIAL_FILES:
        return False

    parts = node_name.split("[", 1)[0].split("::")
    method = parts[-1]
    cls = parts[-2] if len(parts) > 1 else ""
    if cls and path in APPROVED_CLASSES and cls not in APPROVED_CLASSES[path]:
        return False

    if path == "ssm/test_kda.py":
        return not cls and method in {
            "test_kda_forward_backward",
            "test_kda_packed_matches_unpacked_cp1",
            "test_kda_selective_recompute",
            "test_kda_cp2_packed_with_physical_padding_matches_cp1",
            "test_parallel_kda_correctness",
        }
    if path == "ssm/test_gated_delta_product.py":
        return not cls and (
            method
            in {"test_in_proj_recompute_parity", "test_in_proj_recompute_rejects_qkv_offload"}
            or (
                method == "test_in_proj_recompute_preserves_norm_gate_view"
                and "fp8_recipe" in params
                and params["fp8_recipe"] is None
            )
        )
    if path == "ssm/test_hybrid_block.py":
        if cls != "TestHybridBlock":
            return False
        if method in {
            "test_gpu_forward",
            "test_recompute",
            "test_wide_residual_gpu_forward_and_backward",
            "test_wide_shortcut_replay_matches_eager_forward_backward",
            "test_gdn_gpu_forward",
        }:
            return True
        return (
            method
            in {
                "test_two_stage_attention_matches_atomic_forward_bitwise",
                "test_shortcut_pair_eager_forward_backward",
            }
            and isinstance(params.get("compute_config"), dict)
            and params["compute_config"].get("fp8") is None
        )
    if path == "ssm/test_gdp_dynamic_inference.py":
        return cls == "TestGDPDynamicInference" or (
            cls == "TestGDPDynamicInferenceEngine"
            and method in {"test_engine_runs_to_completion", "test_generate_over_multiple_prompts"}
        )
    if path == "inference/engines/test_dynamic_engine.py":
        return (cls, method) != ("TestDynamicInferenceEngine", "test_simple")
    if path == "inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py":
        if cls == "TestChunkedPrefillCudaGraphs":
            return True
        if cls != "TestDynamicInferenceEngineParallel":
            return False
        if method == "test_parallel_inference":
            sizes = [params.get(name) for name in ("tp_size", "pp_size", "ep_size")]
            return (
                params.get("transformer_impl") == "local"
                and all(type(size) is int and size > 0 for size in sizes)
                and sizes[0] * sizes[1] * sizes[2] <= 4
            )
        if method == "test_mtp_kv_cache_chunked_prefill":
            return (
                params.get("transformer_impl") == "local"
                and params.get("ep_size") == 2
                and params.get("dispatcher") == "nccl"
            )
        return method != "test_sequence_parallel_fp8_inference" and not method.startswith(
            "test_mtp_kv_cache_inference_optimized"
        )
    if path == "transformer/test_cuda_graphs.py":
        return cls != "TestPartialCudaGraph"
    if path == "transformer/test_attention.py":
        return cls != "TestSelfAttention"
    if path == "transformer/test_transformer_layer.py":
        if cls == "TestMHCRecomputeMemorySaving":
            return False
        if (cls, method) == ("TestParallelTransformerLayer", "test_sharded_state_dict"):
            return params.get("tp_pp") in ((1, 1), (2, 2))
        return True
    if path == "a2a_overlap/test_schedule_chunk_1f1b.py":
        if cls == "TestA2AOverlapPaddingMask":
            return method == "test_1f1b_schedule_model_chunk_with_padding_mask" and params.get(
                "tp_size"
            ) in (1, 2, 4)
        if cls != "TestA2AOverlap":
            return False
        if method in {
            "test_full_recompute_preserves_mtp_input_mask",
            "test_full_recompute_segment_boundaries_are_graph_free",
        }:
            return True
        return method in {
            "test_1f1b_schedule_model_chunk",
            "test_1f1b_schedule_model_chunk_full_recompute",
            "test_1f1b_schedule_model_chunk_full_recompute_multi_microbatch",
            "test_full_recompute_same_model_steady_state",
        } and _bf16_or_tensorwise(params)
    if path == "a2a_overlap/test_schedule_layer_1f1b.py":
        if cls != "TestA2AOverlap":
            return False
        if method in {
            "test_transformer_layer_overlap_dense",
            "test_transformer_layer_overlap_shared_expert",
            "test_transformer_layer_overlap_early_attn_memory_release",
        }:
            return True
        return method in {
            "test_transformer_layer_overlap",
            "test_mtp_layer_overlap",
        } and _bf16_or_tensorwise(params)
    if path == "a2a_overlap/test_delay_wgrad_compute.py":
        return (
            cls == "TestDelayWgradCompute"
            and method == "test_overlap_dispatch_backward_with_experts_wgrad"
            and _bf16_or_tensorwise(params)
        )
    if path == "transformer/moe/test_grouped_tensor_dispatcher_numerics.py":
        return (
            cls == "TestGroupedTensorDispatcherNumerics"
            and method
            in {
                "test_alltoall_grouped_tensor_moe_parity",
                "test_deepep_grouped_tensor_moe_parity",
                "test_hybridep_grouped_tensor_moe_parity",
            }
            and tuple(
                params.get(name)
                for name in ("single_grouped_weight", "use_bias", "single_grouped_bias")
            )
            in (
                (True, False, False),
                (False, True, False),
                (True, True, False),
                (False, True, True),
            )
        )
    if path == "models/test_hybrid_mhc.py":
        # Keep the six FP32 forward-parity failures from GB200 run 41279258383
        # on H100 until their recompute numerical differences are understood.
        if (cls, method) == ("TestHybridStackMHC", "test_full_recompute_matches_forward_backward"):
            return params.get("precision") in ("bf16_fp32_mixing", "bf16_fused")
        return (cls, method) != (
            "TestHybridStackMHC",
            "test_full_recompute_real_moe_matches_forward_backward",
        )
    if path == "models/test_hybrid_model.py":
        return cls != "TestHybridWithDynamicInference" and method != (
            "test_hybrid_model_with_custom_process_groups"
        )
    if path == "generalized_tensor_parallel/test_gtp_muon_dcp.py":
        return (cls, method) == ("TestGTPMuonDCP", "test_gtp_muon_moe_save_load")
    return False


def active_buckets(buckets: list[str], platform: str, enabled: bool) -> list[str]:
    """Return the active recipe buckets, including rollback to the H100 suite."""
    exclusions = ()
    if platform == "h100" and enabled:
        exclusions = FULL_BUCKETS
    elif platform == "gb200" and not enabled:
        exclusions = NEW_GB200_BUCKETS
    return [bucket for bucket in buckets if bucket not in exclusions]


def filter_buckets(matrix: list[dict], platform: str, enabled: bool) -> list[dict]:
    """Preserve matrix metadata while pruning migrated or disabled jobs."""
    active = set(active_buckets([entry["bucket"] for entry in matrix], platform, enabled))
    return [entry for entry in matrix if entry["bucket"] in active]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("filter-buckets")
    command.add_argument("--platform", choices=("h100", "gb200"), required=True)
    command.add_argument("--enabled", choices=("true", "false"), required=True)
    args = parser.parse_args()
    matrix = json.load(sys.stdin)
    json.dump(filter_buckets(matrix, args.platform, args.enabled == "true"), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
