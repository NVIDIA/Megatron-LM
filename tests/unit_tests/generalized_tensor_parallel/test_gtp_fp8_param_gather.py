# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP + MXFP8 --fp8-param-gather / --reuse-grad-buf-for-mxfp8-param-ag correctness.

Asserts the two MXFP8 param-gather knobs don't change training: a GTP (weight-remat=2) loss
trajectory with the knobs on must match the same run with them off. Reuses the full DDP +
DistributedOptimizer harness from ``test_fp8_param.py::TestFP8Param`` by composition (imported
under a non-``Test*`` alias so pytest doesn't re-collect it), flipping GTP on via
``tensor_parallel_num_weight_shards`` (= tp x gtp_weight_remat_size).
"""

import pytest
import torch

from megatron.core.tensor_parallel.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.utils import is_te_min_version
from megatron.training.utils import get_device_arch_version

# Non-"Test*" alias so pytest does not re-collect the whole TestFP8Param suite here (wrong
# world/DP config + global-state pollution); reused by composition only.
from tests.unit_tests.test_fp8_param import TestFP8Param as _FP8ParamHarness
from tests.unit_tests.test_fp8_param import fp8_available, reason_for_no_fp8


class TestGTPFp8ParamGather:
    """GTP weight-remat=2 loss-trajectory parity for the MXFP8 param-gather knobs."""

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("dp_overlap", [(False, False), (True, True)])
    # (tp_size, num_weight_shards, min_gpus): tp2 case guards a TP/GTP axis-order inversion in
    # native-FP8 init (real TE TP divide + sequence_parallel x GTP2).
    @pytest.mark.parametrize("tp_case", [(1, 2, 2), (2, 4, 4)])
    def test_gtp_mxfp8_fp8_param_gather(self, dp_overlap, tp_case):
        """GTP weight-remat=2: fp8-param loss must track pure-BF16 loss within MXFP8 noise.

        A frozen fp8 forward weight (optimizer updates not reaching the native fp8 shard) instead
        leaves fp8 flat while BF16 descends (~1+ gap). dp_overlap=(overlap_param_gather,
        overlap_grad_reduce); the overlap leg exercises the ``_copy_main_params_to_param_buffer``
        path GTP hooks for --reuse-grad-buf-for-mxfp8-param-ag.
        """
        tp_size, num_shards, min_gpus = tp_case
        if torch.cuda.device_count() < min_gpus:
            pytest.skip(f"Requires {min_gpus} CUDA devices for TP{tp_size} x GTP weight-remat=2")

        harness = _FP8ParamHarness()
        harness.setup_method(None)
        # num-microbatches uses data_parallel_size = world/tp (gtp is a DP sub-axis).
        harness.micro_batch_size = 1
        try:
            common = dict(
                tp_size=tp_size,
                global_batch_size=4,
                overlap_param_gather=dp_overlap[0],
                overlap_grad_reduce=dp_overlap[1],
                tensor_parallel_num_weight_shards=num_shards,  # tp * N => gtp_weight_remat_size=N
                # Untie: the tied path feeds the GTP-sharded embedding into a Megatron-native
                # ColumnParallelLinear, which does no GTP all-gather (TE-only) and fails its check.
                untie_embeddings_and_output_weights=True,
            )
            loss_fp8 = harness._run_test_helper(recipe="mxfp8", fp8_param_gather=True, **common)
            # Pure BF16 GTP reference: fp8=None overrides the harness default (recipe inert).
            loss_bf16 = harness._run_test_helper(
                recipe="delayed", fp8_param_gather=False, fp8=None, **common
            )
            # Max drift ~0.03 over 100 steps (MXFP8 noise); 0.05 stays above it and trips on the
            # ~1+ frozen-weight gap.
            diff = (loss_fp8 - loss_bf16).abs().max().item()
            assert diff < 0.05, (
                f"GTP+mxfp8 fp8-param-gather loss diverges from pure-BF16 GTP baseline "
                f"(max per-step |diff|={diff:.4f}; fp8: {loss_fp8[0]:.3f}->{loss_fp8[-1]:.3f}, "
                f"bf16: {loss_bf16[0]:.3f}->{loss_bf16[-1]:.3f})."
            )
        finally:
            harness.teardown_method(None)
            # Restore GTP_CONFIG defaults mutated by the mxfp8 arg setup.
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_gtp_mxfp8_moe_fp8_param_gather(self):
        """MoE grouped-expert (TEGroupedLinear) native-FP8 GTP: loss must track pure-BF16 GTP.

        Covers the EGTP-sharded expert weights built as native MXFP8 shards under
        --fp8-param-gather — the gap the dense test (attention + dense MLP) leaves. Same parity
        assertion as the dense case.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices for EP=2 x GTP weight-remat=2 MoE")

        harness = _FP8ParamHarness()
        harness.setup_method(None)
        harness.micro_batch_size = 1
        try:
            common = dict(
                tp_size=1,
                global_batch_size=4,
                overlap_param_gather=True,
                overlap_grad_reduce=True,
                tensor_parallel_num_weight_shards=2,  # tp=1 * 2 => gtp_weight_remat_size=2 (EGTP)
                untie_embeddings_and_output_weights=True,
                # MoE grouped experts (mirror test_mxfp8_moe), EP=2.
                num_experts=2,
                moe_grouped_gemm=True,
                expert_model_parallel_size=2,
                moe_token_dispatcher_type="alltoall",
                moe_router_topk=1,
                moe_router_pre_softmax=True,
                moe_router_load_balancing_type="none",
                moe_aux_loss_coeff=0.0,
                moe_ffn_hidden_size=128,
            )
            loss_fp8 = harness._run_test_helper(recipe="mxfp8", fp8_param_gather=True, **common)
            loss_bf16 = harness._run_test_helper(
                recipe="delayed", fp8_param_gather=False, fp8=None, **common
            )
            diff = (loss_fp8 - loss_bf16).abs().max().item()
            assert diff < 0.05, (
                f"GTP+mxfp8 MoE fp8-param-gather loss diverges from pure-BF16 GTP baseline "
                f"(max per-step |diff|={diff:.4f}; fp8: {loss_fp8[0]:.3f}->{loss_fp8[-1]:.3f}, "
                f"bf16: {loss_bf16[0]:.3f}->{loss_bf16[-1]:.3f})."
            )
        finally:
            harness.teardown_method(None)
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_gtp_mxfp8_save_does_not_perturb_training(self):
        """A checkpoint save must NOT mutate the live weights.

        Runs GTP+mxfp8+fp8-param-gather twice with identical seeds — once driving the production
        save path mid-training (force_param_sync + sharded_state_dict), once without — and requires
        matching loss trajectories. overlap_param_gather=True makes should_disable_forward_pre_hook
        True so force_param_sync actually runs; passing the optimizer copies FP32 masters into the
        param buffer first, so the copy-back re-quantizes the GTP native-FP8 shard from masters (not
        stale grad scratch). Guards the historical post-save loss spike (seen at a55b) — a save
        side-effect test_gtp_dcp can't see (it never trains after saving).
        """
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires at least 2 CUDA devices for GTP weight-remat=2")

        common = dict(
            tp_size=1,
            recipe="mxfp8",
            fp8_param_gather=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            global_batch_size=4,
            tensor_parallel_num_weight_shards=2,
            untie_embeddings_and_output_weights=True,
        )
        try:
            h1 = _FP8ParamHarness()
            h1.setup_method(None)
            h1.micro_batch_size = 1
            loss_baseline = h1._run_test_helper(**common)
            h1.teardown_method(None)

            h2 = _FP8ParamHarness()
            h2.setup_method(None)
            h2.micro_batch_size = 1
            loss_saved = h2._run_test_helper(save_at_steps=(5, 10, 15), **common)
            h2.teardown_method(None)

            diff = (loss_baseline - loss_saved).abs()
            worst = diff.max().item()
            # Save runs a real MXFP8 force_param_sync (not bit-exact vs no-save), so allow re-gather
            # noise (~0.03/100 steps); 0.1 clears it and still catches the pre-fix O(10) spike.
            assert worst < 0.1, (
                f"Checkpoint save perturbed training (max per-step |diff|={worst:.4f} at step "
                f"{int(diff.argmax())}); the forced pre-save param-sync is corrupting live FP8 "
                f"weights. saved-run around first save: {loss_saved[4:8].tolist()}"
            )
        finally:
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)
