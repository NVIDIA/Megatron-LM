# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP + MXFP8 --fp8-param-gather / --reuse-grad-buf-for-mxfp8-param-ag correctness.

Asserts the two MXFP8 param-gather knobs don't change training: a GTP (weight-remat=2) loss
trajectory with the knobs on must match the same run with them off. Reuses the full DDP +
DistributedOptimizer harness from ``test_fp8_param.py::TestFP8Param`` by composition (imported
under a non-``Test*`` alias so pytest doesn't re-collect it), flipping GTP on via
``tensor_parallel_num_weight_shards`` (= tp x gtp_weight_remat_size).
"""

import math

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.fp8_utils import dequantize_fp8_tensor, is_mxfp8tensor
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    is_managed_by_layer_wise_optimizer,
)
from megatron.core.ssm.gated_delta_product import (
    HAVE_EINOPS,
    HAVE_FLA,
    HAVE_MAMBA_SSM,
    GatedDeltaProductMixer,
    causal_conv1d_fn,
    check_fla_sequence_packing_support,
)
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    dequantize_gtp_native_fp8,
    is_gtp_param,
)
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.utils import get_device_arch_version

# Non-"Test*" alias so pytest does not re-collect the whole TestFP8Param suite here (wrong
# world/DP config + global-state pollution); reused by composition only.
from tests.unit_tests.test_fp8_param import TestFP8Param as _FP8ParamHarness
from tests.unit_tests.test_fp8_param import fp8_available, reason_for_no_fp8
from tests.unit_tests.test_utilities import Utils

HAVE_FLA_SEQUENCE_PACKING, FLA_SEQUENCE_PACKING_REASON = check_fla_sequence_packing_support()
HAVE_GDP_DEPS = all(
    (HAVE_MAMBA_SSM, HAVE_EINOPS, HAVE_FLA, causal_conv1d_fn is not None, HAVE_FLA_SEQUENCE_PACKING)
)


def _gdp_moe_test_args(overlap, *, num_weight_shards):
    """Return the shared real-GDP + grouped-MoE training configuration."""

    return dict(
        tp_size=1,
        num_steps=4,
        num_layers=2,
        hybrid_layer_pattern="ME",
        spec=["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"],
        position_embedding_type="none",
        padded_vocab_size=512,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        # Keep d_inner=256 while making GDP's logical in-proj/dgrad K
        # 4 * (d_inner + groups * state_dim + heads) = 2080, which is
        # divisible by the MXFP8 block size (32). GTP storage remains padded when enabled.
        mamba_num_heads=8,
        mamba_head_dim=32,
        mamba_num_groups=2,
        mamba_state_dim=128,
        gdp_num_householder=3,
        gdp_cutedsl_kernel=False,
        num_experts=2,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_ffn_hidden_size=256,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        expert_tensor_parallel_num_weight_shards=1,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        optimizer="muon",
        muon_scalar_optimizer="adam",
        muon_momentum=0.9,
        muon_scale_mode="spectral",
        muon_num_ns_steps=5,
        muon_coefficient_type="quintic",
        muon_tp_mode="duplicated",
        lr=1e-3,
        clip_grad=0.0,
        global_batch_size=4,
        tensor_parallel_num_weight_shards=num_weight_shards,
        use_layer_wise_param_layout=True,
        untie_embeddings_and_output_weights=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
    )


class _GDPAdamWMuonHarness(_FP8ParamHarness):
    """Validate the real GDP-AdamW + MoE-Muon ownership and sync paths."""

    @staticmethod
    def _dequantize(param):
        with torch.no_grad():
            if is_mxfp8tensor(param):
                if is_gtp_param(param):
                    value = dequantize_gtp_native_fp8(param)
                else:
                    value = dequantize_fp8_tensor(param)
            else:
                value = param.float()
            return value.detach().clone()

    @staticmethod
    def _find_buffer(model, param):
        matches = [
            buffer
            for buffer in model.buffers + model.expert_parallel_buffers
            if any(buffer_param is param for buffer_param in buffer.params)
        ]
        assert len(matches) == 1, f"Expected one DDP buffer for parameter, found {len(matches)}"
        return matches[0]

    def _on_model_built(self, model_chunks, optimizer, args):
        assert args.use_layer_wise_param_layout
        assert args.expert_gtp_weight_remat_size == 1
        gtp_enabled = args.gtp_weight_remat_size > 1
        native_mxfp8 = args.fp8_param_gather
        assert args.fp8_recipe == "mxfp8"
        assert args.fp8 is not None
        if gtp_enabled:
            assert native_mxfp8, "GTP with the MXFP8 recipe requires FP8 parameter gather"
        if native_mxfp8:
            assert args.reuse_grad_buf_for_mxfp8_param_ag
        else:
            assert not args.reuse_grad_buf_for_mxfp8_param_ag
        assert args.muon_scalar_optimizer == "adam"
        assert optimizer.config.decoupled_weight_decay, "Scalar Adam must use AdamW semantics"

        model = model_chunks[0]
        core_model = unwrap_model(model)
        assert core_model.decoder.layer_type_list == ["M", "E"]

        gdp_mixers = [
            module for module in core_model.modules() if isinstance(module, GatedDeltaProductMixer)
        ]
        assert len(gdp_mixers) == 1, f"Expected one GDP mixer, found {len(gdp_mixers)}"
        gdp = gdp_mixers[0]
        in_proj_width = (1 + gdp.num_householder) * (
            gdp.d_inner + gdp.ngroups * gdp.d_state + gdp.nheads
        )
        assert (
            in_proj_width % 32 == 0
        ), f"GDP in-proj width {in_proj_width} is incompatible with MXFP8's 32-element blocks"
        in_proj = gdp.in_proj.weight
        out_proj = gdp.out_proj.weight

        named_params = dict(core_model.named_parameters())
        expert_name = "decoder.layers.1.mlp.experts.linear_fc1.weight0"
        assert expert_name in named_params, f"Missing grouped-MoE parameter {expert_name}"
        expert_weight = named_params[expert_name]

        # These are production attributes: the test does not inject optimizer routing.
        assert getattr(in_proj, "use_muon", True) is False
        assert _is_muon_excluded(in_proj)
        assert not is_managed_by_layer_wise_optimizer(in_proj)
        assert getattr(in_proj, "is_managed_by_layer_wise_optimizer", None) is False

        for label, param in (("GDP out_proj", out_proj), ("MoE expert", expert_weight)):
            assert not _is_muon_excluded(param), f"{label} must remain on Muon"
            assert is_managed_by_layer_wise_optimizer(param)
            assert getattr(param, "is_managed_by_layer_wise_optimizer", None) is True

        assert is_gtp_param(in_proj) == gtp_enabled
        assert is_gtp_param(out_proj) == gtp_enabled
        assert not is_gtp_param(expert_weight)
        assert getattr(expert_weight, "allreduce", True) is False
        for label, param in (
            ("GDP in_proj", in_proj),
            ("GDP out_proj", out_proj),
            ("MoE expert", expert_weight),
        ):
            assert (
                is_mxfp8tensor(param) == native_mxfp8
            ), f"{label} storage does not match fp8_param_gather={args.fp8_param_gather}"

        layerwise_optimizers = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, LayerWiseDistributedOptimizer)
        ]
        distributed_optimizers = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, DistributedOptimizer)
        ]
        assert len(layerwise_optimizers) == 1
        assert len(distributed_optimizers) == 1
        layerwise_optimizer = layerwise_optimizers[0]
        distributed_optimizer = distributed_optimizers[0]

        assert any(
            param is in_proj
            for group in distributed_optimizer.model_float16_groups
            for param in group
        ), "GDP in_proj must be owned by the scalar AdamW DistributedOptimizer"
        assert any(
            param is out_proj
            for owner_params in (layerwise_optimizer.dp_cp_params_list or [])
            for param in owner_params
        ), "GDP out_proj must be owned by LayerWise/Muon"
        assert any(
            param is expert_weight
            for owner_params in (layerwise_optimizer.expt_dp_params_list or [])
            for param in owner_params
        ), "MoE expert weight must be owned by LayerWise/Muon"

        in_proj_buffer = self._find_buffer(model, in_proj)
        out_proj_buffer = self._find_buffer(model, out_proj)
        expert_buffer = self._find_buffer(model, expert_weight)
        expected_buffer_dtype = torch.uint8 if native_mxfp8 else torch.bfloat16
        assert in_proj_buffer.param_dtype == expected_buffer_dtype
        assert out_proj_buffer.param_dtype == expected_buffer_dtype
        assert expert_buffer.param_dtype == expected_buffer_dtype
        assert (
            in_proj_buffer is not out_proj_buffer
        ), "AdamW GDP in_proj and Muon GDP out_proj require distinct DDP buffers"
        assert in_proj_buffer.data_parallel_group.size() == args.data_parallel_size
        assert out_proj_buffer.data_parallel_group.size() == args.data_parallel_size
        assert expert_buffer.data_parallel_group.size() == 2

        # Subset sync classifies a whole bucket group from its first bucket, so mixed ownership is
        # never legal. This is the invariant guarded by the partition_buckets change.
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            owners = {
                getattr(param, "is_managed_by_layer_wise_optimizer", False)
                for bucket in bucket_group.buckets
                for param in bucket.params
            }
            assert len(owners) == 1, f"Bucket group mixes optimizer ownership: {owners}"

        in_proj_group_index, in_proj_group_order = (
            distributed_optimizer.model_param_group_index_map[in_proj]
        )
        in_proj_main = distributed_optimizer.optimizer.param_groups[in_proj_group_index]["params"][
            in_proj_group_order
        ]

        tracked = [
            ("GDP in_proj", in_proj, in_proj_buffer),
            ("GDP out_proj", out_proj, out_proj_buffer),
            ("MoE expert", expert_weight, expert_buffer),
        ]
        self._tracked_params = [
            (label, param, buffer, self._dequantize(param)) for label, param, buffer in tracked
        ]
        self._tracked_masters = [
            ("GDP in_proj AdamW master", in_proj_main, in_proj_main.detach().clone())
        ]
        for label, param, _ in tracked[1:]:
            main_param = getattr(param, "main_param", None)
            if main_param is not None:
                self._tracked_masters.append(
                    (f"{label} Muon master", main_param, main_param.detach().clone())
                )
        self._runtime_validation_ran = False

    def _on_forward_complete(self, model_chunks, optimizer, args, step, num_steps):
        if args.overlap_param_gather:
            hooks_enabled = bool(model_chunks[0].remove_forward_pre_hook_handles)
            assert hooks_enabled == (step > 0), (
                "Parameter-gather pre-hook lifecycle differs from production: "
                f"step={step}, hooks_enabled={hooks_enabled}"
            )

        if step != num_steps - 1:
            return

        failures = []
        for label, main_param, initial_main in self._tracked_masters:
            if torch.equal(main_param, initial_main):
                failures.append(f"{label} did not update")

        for label, param, buffer, initial_value in self._tracked_params:
            current_value = self._dequantize(param)
            if not torch.isfinite(current_value).all():
                failures.append(f"{label} contains NaN/Inf")
            if torch.equal(current_value, initial_value):
                failures.append(f"{label} forward weight did not update")

            initial_norm = initial_value.float().norm()
            current_norm = current_value.float().norm()
            if not (current_norm > initial_norm * 0.1 and current_norm < initial_norm * 10.0):
                failures.append(
                    f"{label} norm changed implausibly: {initial_norm.item():.4f} -> "
                    f"{current_norm.item():.4f}"
                )

            replicas = [
                torch.empty_like(current_value) for _ in range(buffer.data_parallel_group.size())
            ]
            torch.distributed.all_gather(
                replicas, current_value.contiguous(), group=buffer.data_parallel_group
            )
            if any(not torch.equal(replicas[0], replica) for replica in replicas[1:]):
                failures.append(f"{label} differs across its data-parallel replicas")

        # Make every rank fail together so teardown barriers cannot hang on a rank-local error.
        failure_flag = torch.tensor(bool(failures), dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
        if failure_flag.item():
            if not failures:
                failures.append("runtime validation failed on another rank")
            raise AssertionError("; ".join(failures))
        self._runtime_validation_ran = True


class TestGTPFp8ParamGather:
    """MXFP8 parameter-gather tests, including GTP and exact same-recipe parity."""

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

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    @pytest.mark.skipif(
        not HAVE_GDP_DEPS,
        reason=(
            FLA_SEQUENCE_PACKING_REASON or "GDP requires mamba-ssm, einops, FLA, and causal-conv1d"
        ),
    )
    @pytest.mark.parametrize("overlap", [False, True])
    def test_gtp_gdp_adamw_moe_muon_mxfp8_sync(self, overlap):
        """GTP2 GDP-AdamW + MoE-Muon MXFP8 weights must update and stay synchronized."""
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for GTP2 x DP2 and EP2 x EDP2")

        harness = _GDPAdamWMuonHarness()
        harness.setup_method(None)
        harness.seq_length = 128
        harness.micro_batch_size = 1
        try:
            losses = harness._run_test_helper(
                recipe="mxfp8",
                fp8_param_gather=True,
                **_gdp_moe_test_args(overlap, num_weight_shards=2),
            )
            assert harness._runtime_validation_ran

            all_ranks_finite = torch.tensor(
                int(torch.isfinite(losses).all().item()), dtype=torch.int32, device="cuda"
            )
            torch.distributed.all_reduce(all_ranks_finite, op=torch.distributed.ReduceOp.MIN)
            assert all_ranks_finite.item(), f"GTP GDP+MoE loss contains NaN/Inf: {losses}"
        finally:
            harness.teardown_method(None)
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    @pytest.mark.parametrize("overlap", [False, True])
    def test_muon_layout_and_mxfp8_param_gather_parity(self, overlap):
        """Padded LayerWise MXFP8 sync must match the legacy LayerWise path.

        Both runs are configured with Muon, GTP2 x DP2, MXFP8 primary weights,
        FP8 parameter gather, and grad-buffer reuse. The legacy LayerWise gather
        is the oracle; the padded-layout run exercises the reused DDP grad buffer
        and is the regression target. Overlap selects synchronous step-time gather
        or asynchronous dispatch from the next forward pre-hook.
        """
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for GTP2 x DP2")

        harness = _FP8ParamHarness()
        harness.setup_method(None)
        harness.seq_length = 64
        harness.micro_batch_size = 1
        try:
            common = dict(
                num_steps=12,
                num_layers=1,
                padded_vocab_size=512,
                ffn_hidden_size=256,
                global_batch_size=4,
                optimizer="muon",
                muon_scalar_optimizer="adam",
                muon_momentum=0.9,
                muon_scale_mode="spectral",
                muon_num_ns_steps=5,
                muon_coefficient_type="quintic",
                muon_tp_mode="duplicated",
                lr=1e-3,
                clip_grad=0.0,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                tensor_parallel_num_weight_shards=2,
                untie_embeddings_and_output_weights=True,
                overlap_param_gather=overlap,
                overlap_grad_reduce=overlap,
            )

            loss_legacy = harness._run_test_helper(
                tp_size=1,
                recipe="mxfp8",
                fp8_param_gather=True,
                use_layer_wise_param_layout=False,
                **common,
            )
            loss_padded = harness._run_test_helper(
                tp_size=1,
                recipe="mxfp8",
                fp8_param_gather=True,
                use_layer_wise_param_layout=True,
                **common,
            )

            max_diff = torch.tensor(float((loss_legacy - loss_padded).abs().max()), device="cuda")
            # Compare the worst local GTP/DP trajectory, rather than rank 0 alone.
            # Map local non-finite values before MAX so NCCL cannot hide a NaN from one rank.
            max_diff.nan_to_num_(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
            torch.distributed.all_reduce(max_diff, op=torch.distributed.ReduceOp.MAX)
            diff = max_diff.item()
            tolerance = 2e-3
            assert math.isfinite(diff) and diff < tolerance, (
                f"Padded LayerWise MXFP8 loss diverges from the legacy path "
                f"(overlap={overlap}, max per-step |diff|={diff:.6f}, tolerance={tolerance})."
            )
        finally:
            harness.teardown_method(None)
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    @pytest.mark.skipif(
        not HAVE_GDP_DEPS,
        reason=(
            FLA_SEQUENCE_PACKING_REASON or "GDP requires mamba-ssm, einops, FLA, and causal-conv1d"
        ),
    )
    @pytest.mark.parametrize("overlap", [False, True])
    def test_gdp_adamw_moe_muon_mxfp8_param_gather(self, overlap, monkeypatch):
        """GDP-AdamW + MoE-Muon must agree with MXFP8 parameter gather ON and OFF.

        ``ME`` builds a GDP mixer followed by a grouped-MoE layer. GDP itself marks its
        ``in_proj.weight`` ``use_muon=False``, which routes it to scalar AdamW; GDP ``out_proj``
        and the two-dimensional expert weights remain on LayerWise/Muon. The padded layout then
        creates separate AdamW-owned and Muon-owned buffers. For each overlap mode, the native
        MXFP8 parameter-gather/reuse trajectory must remain close to the same model using BF16
        primary weights, while both runs use the same MXFP8 compute recipe. In particular, the
        first post-update loss verifies that LayerWise/Muon initialized its FP32 masters from
        TE's preserved high-precision values instead of dequantized MXFP8 weights.

        Weight rematerialization is intentionally disabled: active GTP requires MXFP8 parameter
        gather and cannot form this exact ON/OFF comparison. The original reused-grad-buffer bug
        does not require GTP, and remains exercised by the ON leg's LayerWise DP all-gather.
        """
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for DP4 and EP2 x EDP2")

        # GDP's channels-last causal-conv backward normally accumulates dweight with atomicAdd.
        # Its launch-order-dependent rounding becomes visible when gradient reduce is overlapped,
        # even between two otherwise identical BF16 runs. causal-conv1d 1.6.2 reads this switch on
        # every backward and uses a deterministic workspace reduction, keeping this test focused
        # on fp8-param-gather/reuse instead of an unrelated convolution-kernel scheduling effect.
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")

        harness = _GDPAdamWMuonHarness()
        harness.setup_method(None)
        harness.seq_length = 128
        harness.micro_batch_size = 1
        try:
            common = _gdp_moe_test_args(overlap, num_weight_shards=1)

            loss_fp8_param_gather_on = harness._run_test_helper(
                recipe="mxfp8", fp8_param_gather=True, **common
            )
            assert harness._runtime_validation_ran

            loss_fp8_param_gather_off = harness._run_test_helper(
                recipe="mxfp8", fp8_param_gather=False, **common
            )
            assert harness._runtime_validation_ran

            local_trajectories = torch.stack(
                (loss_fp8_param_gather_on, loss_fp8_param_gather_off)
            ).cuda()
            gathered_trajectories = [
                torch.empty_like(local_trajectories)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_trajectories, local_trajectories)
            trajectories = torch.stack(gathered_trajectories)
            per_rank_step_diff = (trajectories[:, 0] - trajectories[:, 1]).abs()
            per_rank_step_diff.nan_to_num_(
                nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            )
            atol = 1e-4
            rtol = 1e-3
            allowed_diff = atol + rtol * trajectories[:, 1].abs()
            error_ratio = per_rank_step_diff / allowed_diff
            error_ratio.nan_to_num_(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            worst_index = int(error_ratio.argmax().item())
            worst_rank, worst_step = divmod(worst_index, per_rank_step_diff.shape[1])
            diff = per_rank_step_diff[worst_rank, worst_step].item()
            tolerance = allowed_diff[worst_rank, worst_step].item()
            assert torch.isfinite(trajectories).all() and diff <= tolerance, (
                "GDP-AdamW + MoE-Muon loss differs with MXFP8 parameter gather ON versus OFF "
                f"(overlap={overlap}, |diff|={diff:.6f}, allowed={tolerance:.6f}, "
                f"atol={atol}, rtol={rtol}, worst_rank={worst_rank}, "
                f"worst_step={worst_step}; fp8-param-gather-ON: "
                f"{trajectories[worst_rank, 0].tolist()}, fp8-param-gather-OFF: "
                f"{trajectories[worst_rank, 1].tolist()})."
            )
        finally:
            harness.teardown_method(None)

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
