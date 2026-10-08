# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Blockwise FP8 gather ON/OFF parity for GDP-AdamW and MoE-Muon, without GTP."""

import pytest
import torch

from megatron.core.fp8_utils import (
    dequantize_fp8_tensor,
    is_blockwise_float8tensor,
    is_float8tensor,
)
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
from megatron.core.utils import is_te_min_version, unwrap_model
from tests.unit_tests.test_fp8_param import TestFP8Param as _FP8ParamHarness
from tests.unit_tests.test_fp8_param import fp8_available, reason_for_no_fp8
from tests.unit_tests.test_fp8_param_gather_policy import (
    assert_param_storage_policy,
    require_fp8_recipe,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200

HAVE_FLA_SEQUENCE_PACKING, FLA_SEQUENCE_PACKING_REASON = check_fla_sequence_packing_support()
HAVE_GDP_DEPS = all(
    (HAVE_MAMBA_SSM, HAVE_EINOPS, HAVE_FLA, causal_conv1d_fn is not None, HAVE_FLA_SEQUENCE_PACKING)
)


def _gdp_moe_test_args(overlap, *, use_layer_wise_param_layout):
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
        # GDP sets d_inner=heads*head_dim independently of hidden_size.
        # Blockwise requires in-proj width4*(1024+2*128+32)=5248 to be 128-aligned.
        mamba_num_heads=32,
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
        moe_router_skip_muon=False,
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
        tensor_parallel_num_weight_shards=1,
        use_layer_wise_param_layout=use_layer_wise_param_layout,
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
            if is_blockwise_float8tensor(param):
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
        assert isinstance(args.use_layer_wise_param_layout, bool)
        assert args.expert_gtp_weight_remat_size == 1
        assert args.gtp_weight_remat_size == 1
        native_fp8 = args.fp8_param_gather
        assert args.fp8_recipe == "blockwise"
        assert args.fp8 is not None
        assert (
            not args.reuse_grad_buf_for_mxfp8_param_ag
        ), "Blockwise Muon reuse must be implicit, without enabling the MXFP8-only flag"
        assert args.muon_scalar_optimizer == "adam"
        assert optimizer.config.decoupled_weight_decay, "Scalar Adam must use AdamW semantics"

        model = model_chunks[0]
        counts = assert_param_storage_policy(model, args)
        if native_fp8:
            assert counts["adam_blockwise"] and counts["muon_fp8"]
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
        block_size = 128
        assert in_proj_width % block_size == 0, (
            f"GDP in-proj width {in_proj_width} is incompatible with "
            f"{args.fp8_recipe}'s {block_size}-element blocks"
        )
        in_proj = gdp.in_proj.weight
        out_proj = gdp.out_proj.weight

        named_params = dict(core_model.named_parameters())
        expert_name = "decoder.layers.1.mlp.experts.linear_fc1.weight0"
        assert expert_name in named_params, f"Missing grouped-MoE parameter {expert_name}"
        expert_weight = named_params[expert_name]
        router_weight = named_params["decoder.layers.1.mlp.router.weight"]
        assert not is_float8tensor(router_weight)
        assert getattr(
            router_weight, "is_managed_by_layer_wise_optimizer", False
        ), "GDP integration must retain a high-precision Muon router alongside FP8 matrices"

        # These are production attributes: the test does not inject optimizer routing.
        assert getattr(in_proj, "use_muon", True) is False
        assert _is_muon_excluded(in_proj)
        assert not is_managed_by_layer_wise_optimizer(in_proj)
        assert getattr(in_proj, "is_managed_by_layer_wise_optimizer", None) is False

        for label, param in (("GDP out_proj", out_proj), ("MoE expert", expert_weight)):
            assert not _is_muon_excluded(param), f"{label} must remain on Muon"
            assert is_managed_by_layer_wise_optimizer(param)
            assert getattr(param, "is_managed_by_layer_wise_optimizer", None) is True

        assert getattr(expert_weight, "allreduce", True) is False
        for label, param in (
            ("GDP in_proj", in_proj),
            ("GDP out_proj", out_proj),
            ("MoE expert", expert_weight),
        ):
            is_recipe_tensor = is_blockwise_float8tensor(param)
            assert is_recipe_tensor == native_fp8, (
                f"{label} storage does not match recipe={args.fp8_recipe}, "
                f"fp8_param_gather={args.fp8_param_gather}"
            )

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
        expected_adam_dtype = torch.uint8 if native_fp8 else torch.bfloat16
        expected_muon_dtype = (
            expected_adam_dtype if args.use_layer_wise_param_layout else torch.bfloat16
        )
        assert in_proj_buffer.param_dtype == expected_adam_dtype
        assert out_proj_buffer.param_dtype == expected_muon_dtype
        assert expert_buffer.param_dtype == expected_muon_dtype
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


class TestMuonBlockwiseFP8ParamGather:
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
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
    @pytest.mark.parametrize("use_layer_wise_param_layout", [False, True])
    def test_gdp_adamw_moe_muon_blockwise_param_gather(
        self, overlap, use_layer_wise_param_layout, monkeypatch
    ):
        """GDP-AdamW + MoE-Muon must agree with blockwise parameter gather ON and OFF.

        ``ME`` builds a GDP mixer followed by a grouped-MoE layer. GDP itself marks its
        ``in_proj.weight`` ``use_muon=False``, which routes it to scalar AdamW; GDP ``out_proj``
        and the two-dimensional expert weights remain on LayerWise/Muon. Both layouts
        create separate AdamW-owned and Muon-owned buffers. For each overlap mode, the native
        FP8 parameter-gather/reuse trajectory must remain close to the same model using BF16
        primary weights, while both runs use the same compute recipe. In particular, the
        first post-update loss verifies that LayerWise/Muon initialized its FP32 masters from
        TE's preserved high-precision values instead of dequantized FP8 weights.

        Weight rematerialization is disabled. Muon FP8 buffers must implicitly reuse gradient
        storage, while AdamW keeps its native FP8 transport and BF16 weights remain persistent.
        Neither leg enables the MXFP8 reuse flag, and both use blockwise compute.
        """
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for DP4 and EP2 x EDP2")
        require_fp8_recipe("blockwise")

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
            common = _gdp_moe_test_args(
                overlap, use_layer_wise_param_layout=use_layer_wise_param_layout
            )

            loss_fp8_param_gather_on = harness._run_test_helper(
                recipe="blockwise", fp8_param_gather=True, **common
            )
            assert harness._runtime_validation_ran

            loss_fp8_param_gather_off = harness._run_test_helper(
                recipe="blockwise", fp8_param_gather=False, **common
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
                "GDP-AdamW + MoE-Muon loss differs with FP8 parameter gather ON versus OFF "
                f"(recipe=blockwise, layout={use_layer_wise_param_layout}, overlap={overlap}, "
                f"|diff|={diff:.6f}, allowed={tolerance:.6f}, "
                f"atol={atol}, rtol={rtol}, worst_rank={worst_rank}, "
                f"worst_step={worst_step}; fp8-param-gather-ON: "
                f"{trajectories[worst_rank, 0].tolist()}, fp8-param-gather-OFF: "
                f"{trajectories[worst_rank, 1].tolist()})."
            )
        finally:
            harness.teardown_method(None)
