# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for EGTP_remat + MoE correctness.

Test groups
-----------
TestMoEEGTPCorrectness  - router expert-bias sync across gtp_remat peers; MXFP8 expert override
                          under NVFP4 with EGTP_remat. EGTP expert gradient values are checked in
                          test_gtp_grad_correctness.py.
TestMoEEGTPModuleGroupedTensor
                        - BF16 grouped experts with the op-fuser disabled must select the
                          module grouped-tensor path or its old-TE fallback and finish backward.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTPShardedParam,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _run_distributed


def _worker_expert_bias_gtp_inclusive(rank, world_size, port):
    """Router expert-bias must stay identical across gtp_remat peers.

    The aux-loss-free balancer (``get_updated_expert_bias``) all-reduces per-expert token counts
    over the router's tp_dp_cp group, then sign-updates the replicated expert_bias. gtp_remat peers
    hold DISTINCT tokens but share the replicated router, so the reduction must span gtp_remat.
    ``get_tensor_and_data_parallel_group`` therefore spans gtp_remat (like dp); over a gtp-EXCLUDED
    group the peers reduce different token sums and the bias diverges -> routing instability
    (loss spikes). world=4 -> tp1 * gtp_remat2 * dp2.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
    from megatron.core.utils import get_pg_size

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    gtp_group = ps.get_gtp_weight_remat_group()
    tp_dp_cp = ps.get_tensor_and_data_parallel_group(with_context_parallel=True)  # spans gtp_remat
    # gtp-EXCLUDED replicate group (explicit, since get_data_parallel_group now defaults inclusive).
    replicate = ps.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=False)

    num_experts = 8
    torch.manual_seed(1000 + rank)  # distinct per-rank tokens (distinct data on each rank)
    base = torch.randint(0, 100, (num_experts,), device="cuda").float()

    def _max_bias_diff_across_gtp(group):
        bias = torch.zeros(num_experts, device="cuda")  # identical start on every rank
        updated = get_updated_expert_bias(base.clone(), bias, 0.01, tp_dp_cp_group=group)
        buf = [torch.zeros_like(updated) for _ in range(gtp_group.size())]
        dist.all_gather(buf, updated, group=gtp_group)
        return max((buf[i] - buf[0]).abs().max().item() for i in range(gtp_group.size()))

    # tp_dp_cp must span gtp_remat (size = replicate_dp_cp x gtp_remat at tp=1).
    spans_gtp = get_pg_size(tp_dp_cp) == get_pg_size(replicate) * get_pg_size(gtp_group)
    diff_excluded = _max_bias_diff_across_gtp(replicate)
    diff_included = _max_bias_diff_across_gtp(tp_dp_cp)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()

    if rank == 0:
        assert spans_gtp, "tp_dp_cp group must span the gtp_remat axis (like dp)"
        # gtp-excluded reduction reproduces the bug; tp_dp_cp (spans gtp_remat) keeps peers in sync.
        assert (
            diff_excluded > 0
        ), f"gtp-excluded group should diverge across gtp_remat peers, got {diff_excluded}"
        assert (
            diff_included == 0
        ), f"tp_dp_cp must keep expert_bias identical across gtp_remat peers, got {diff_included}"


# ---------------------------------------------------------------------------
# EGTP_remat module grouped-tensor compatibility with a BF16 precision override
# ---------------------------------------------------------------------------


def _bf16_override_recipe(pattern):
    """A te-precision-config recipe forcing `pattern` to high precision."""
    from megatron.core.quantization.quant_config import RecipeConfig

    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                }
            },
            "matchers": {
                "bf16": {"type": "glob", "enabled": True, "pattern": pattern, "config": "bf16"}
            },
        }
    )


def _worker_egtp_module_grouped_tensor_backward(rank, world_size, port, egtp_remat_size):
    """Exercise the module backend selection independently of the op-fuser precision path.

    Request grouped tensors explicitly with the op-fuser disabled. A TE build with #3517
    supports GTP-sharded weights on this path; older builds must fall back to split-quantize.
    EGTP_remat=1 is the unsharded control, and 2 exercises the compatibility guard.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe import experts as experts_mod
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN, FFN, NUM_EXPERTS, SEQ = 512, 256, 4, 16
    dtype = torch.bfloat16
    # A module path the production `*mtp.layers.*` matcher hits, so the override resolves by
    # module path exactly as it does in production, without building a full MTP stack.
    name = "mtp.layers.0.mlp"

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_gtp_remat_size=egtp_remat_size,
    )
    model_parallel_cuda_manual_seed(42)

    config = TransformerConfig(
        num_attention_heads=8,
        num_layers=1,
        hidden_size=HIDDEN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        params_dtype=dtype,
        bf16=True,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_transformer_engine_op_fuser=False,
        moe_use_grouped_tensor=True,
        gradient_accumulation_fusion=True,  # so TE builds main_grad_funcs at all
        quant_recipe=_bf16_override_recipe("*mtp.layers.*"),
    )

    # MoELayer's default collection has no expt_gtp_remat entry; EGTP_remat sharding keys off it.
    moe_pg = get_default_pg_collection()
    moe_pg.expt_gtp_remat = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['expt_gtp_remat']
    ).expt_gtp_remat
    assert moe_pg.expt_gtp_remat.size() == egtp_remat_size

    moe_spec = get_moe_module_spec(use_te=True, num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
    layer = moe_spec(config, layer_number=1, pg_collection=moe_pg, name=name).cuda()

    try:
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)
        # The module path is explicitly selected, rather than relying on a BF16 override to
        # disable the op-fuser. Both linears must still resolve the MTP precision override.
        assert not config.use_transformer_engine_op_fuser
        assert not experts._with_fused_impl
        assert config.moe_use_grouped_tensor
        assert not experts.linear_fc1.will_execute_quantized(True)
        assert not experts.linear_fc2.will_execute_quantized(True)

        # Ingredient 2: EGTP_remat > 1 shards the expert weights. Only then may the guard fire;
        # leaving an unsharded grouped-tensor run alone is what keeps the flag usable at large.
        sharded = any(isinstance(p, GTPShardedParam) for p in experts.parameters())
        assert sharded == (egtp_remat_size > 1)
        # A TE that handles sharded weights on its grouped-tensor path keeps the layer there.
        expect_grouped_tensor = (
            not sharded or experts_mod._te_grouped_tensor_supports_sharded_weights()
        )
        assert experts._use_grouped_tensor == expect_grouped_tensor
        assert experts.linear_fc1.use_grouped_tensor == expect_grouped_tensor
        assert experts.linear_fc2.use_grouped_tensor == expect_grouped_tensor

        # DDP would allocate these; the wgrad closures are what the bug corrupts.
        for p in experts.parameters():
            if p.requires_grad and not hasattr(p, "main_grad"):
                p.main_grad = torch.zeros_like(p.data, dtype=torch.float32)

        x = torch.randn((SEQ, 1, HIDDEN), dtype=dtype, device="cuda", requires_grad=True)
        output, _ = layer(x)
        output.mean().backward()  # pre-fix, egtp_remat_size=2 raises AttributeError on main_grad

        assert any(
            p.main_grad.abs().sum() > 0 for p in experts.parameters() if hasattr(p, "main_grad")
        ), "no expert wgrad reached main_grad"
    finally:
        # An assertion above would otherwise leave the EGTP_remat groups installed.
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _mxfp8_override_recipe(pattern):
    """A te-precision-config recipe keeping `pattern` quantized, but with MXFP8."""
    from megatron.core.quantization.quant_config import RecipeConfig

    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "mxfp8": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {"fp8_quantization_recipe": "mxfp8"},
                }
            },
            "matchers": {
                "mxfp8": {"type": "glob", "enabled": True, "pattern": pattern, "config": "mxfp8"}
            },
        }
    )


def _te_fused_grouped_mlp_quantizes_gathered_bf16() -> bool:
    """Whether TE's fused grouped MLP quantizes a distributed weight it re-gathers as BF16.

    A capability probe rather than a version check: the fix landed with the helper below and is
    not tied to a TE release yet. Without it, EGTP_remat-sharded experts under a quantized
    override raise AttributeError on ``_columnwise_data`` in the fused backward.
    """
    try:
        from transformer_engine.pytorch.ops.fused import grouped_mlp
    except ImportError:
        return False
    return hasattr(grouped_mlp, "_quantize_unquantized_weights")


def _worker_egtp_quantized_precision_override_backward(rank, world_size, port, egtp_remat_size):
    """EGTP_remat experts kept *fused* by an MXFP8 override under a global NVFP4 recipe.

    Production repro: MTP experts routed to MXFP8 by ``--te-precision-config-file`` while the
    run is NVFP4. The fused call must (a) enter the module's MXFP8 autocast rather than inherit
    NVFP4 and (b) with EGTP_remat sharding, have TE quantize the BF16 shards it re-gathers in backward
    (the override creates no native-quantized params). Pre-fix this raised AttributeError in
    the fused backward: ``_amax_columnwise`` without (a), ``_columnwise_data`` without (b).
    Once TE has (b), missing (a) no longer crashes but silently runs NVFP4, hence the quantizer
    assertion below. ``egtp_remat_size=1`` is the unsharded control.
    """
    from transformer_engine.common.recipe import NVFP4BlockScaling
    from transformer_engine.pytorch import MXFP8Quantizer

    from megatron.core import parallel_state as ps
    from megatron.core.activations import squared_relu
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN, FFN, NUM_EXPERTS, SEQ = 512, 256, 4, 16
    dtype = torch.bfloat16
    name = "mtp.layers.0.mlp"

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_gtp_remat_size=egtp_remat_size,
    )
    model_parallel_cuda_manual_seed(42)

    config = TransformerConfig(
        num_attention_heads=8,
        num_layers=1,
        hidden_size=HIDDEN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        add_bias_linear=False,
        # Production expert path (fused weighted squared-ReLU): forms the CuTeDSL fused op
        # without the 32-wide GLU interleave the GLU variant would need.
        gated_linear_unit=False,
        activation_func=squared_relu,
        use_fused_weighted_squared_relu=True,
        bias_activation_fusion=False,
        params_dtype=dtype,
        bf16=True,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        quant_recipe=_mxfp8_override_recipe("*mtp.layers.*"),
    )

    moe_pg = get_default_pg_collection()
    moe_pg.expt_gtp_remat = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['expt_gtp_remat']
    ).expt_gtp_remat
    assert moe_pg.expt_gtp_remat.size() == egtp_remat_size

    moe_spec = get_moe_module_spec(use_te=True, num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
    layer = moe_spec(config, layer_number=1, pg_collection=moe_pg, name=name).cuda()

    try:
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)
        # The configured op-fuser path must execute the module's MXFP8 recipe.
        assert experts._with_fused_impl
        sharded = any(isinstance(p, GTPShardedParam) for p in experts.parameters())
        assert sharded == (egtp_remat_size > 1)

        for p in experts.parameters():
            if p.requires_grad and not hasattr(p, "main_grad"):
                p.main_grad = torch.zeros_like(p.data, dtype=torch.float32)

        x = torch.randn((SEQ, 1, HIDDEN), dtype=dtype, device="cuda", requires_grad=True)
        with fp8_autocast(enabled=True, fp8_recipe=NVFP4BlockScaling()):
            output, _ = layer(x)
        # The CuTeDSL fused op must actually have formed (after the lazy first forward); the
        # basic-op fallback would not exercise the fused weight plumbing this test is about.
        (sequential,) = experts._fused_ops
        selected = [
            type(entry[0] if isinstance(entry, (tuple, list)) else entry).__name__
            for group in sequential._module_groups
            for entry in group._forward_ops
        ]
        assert any(name.startswith("GroupedMLP_CuTeGEMM") for name in selected), selected
        # ... and it quantized with the module's MXFP8 recipe, not the ambient NVFP4 one. With a
        # TE that quantizes re-gathered BF16 shards, inheriting NVFP4 here would not crash; it
        # would silently run these experts in the wrong precision.
        fused_op = next(
            entry[0] if isinstance(entry, (tuple, list)) else entry
            for group in sequential._module_groups
            for entry in group._forward_ops
            if type(entry[0] if isinstance(entry, (tuple, list)) else entry).__name__.startswith(
                "GroupedMLP_CuTeGEMM"
            )
        )
        for fc in (fused_op.basic_ops[0], fused_op.basic_ops[-1]):
            assert isinstance(fc.get_quantizer("forward", 1), MXFP8Quantizer), type(
                fc.get_quantizer("forward", 1)
            )
        output.float().mean().backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
        wait_for_gtp_grad_reduction_on_current_stream()
        expert_grads = [p.main_grad for p in experts.parameters() if p.requires_grad]
        assert expert_grads
        assert all(torch.isfinite(grad).all() for grad in expert_grads)
        assert any(
            grad.abs().sum() > 0 for grad in expert_grads
        ), "no expert wgrad reached main_grad"
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestMoEEGTPCorrectness:
    def test_expert_bias_gtp_inclusive(self):
        """expert_bias stays synced across gtp_remat peers only with the gtp-inclusive group."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_expert_bias_gtp_inclusive, 4)

    @pytest.mark.launch_on_gb200
    @pytest.mark.parametrize("egtp_remat_size", (1, 2))
    def test_egtp_quantized_precision_override_backward(self, request, egtp_remat_size):
        """MXFP8 override under global NVFP4 keeps experts fused and survives backward with EGTP_remat."""
        import transformer_engine.pytorch as te

        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        for probe in (te.is_mxfp8_available, te.is_nvfp4_available):
            available, reason = probe(return_reason=True)
            if not available:
                pytest.skip(reason)
        if egtp_remat_size > 1 and not _te_fused_grouped_mlp_quantizes_gathered_bf16():
            pytest.skip(
                "Sharded experts under a quantized override need a TE whose fused grouped MLP "
                "quantizes re-gathered BF16 weights (ops/fused/grouped_mlp.py "
                "_quantize_unquantized_weights)"
            )
        request.getfixturevalue("require_te_fused_srelu_mlp")
        _run_distributed(_worker_egtp_quantized_precision_override_backward, 4, egtp_remat_size)


class TestMoEEGTPModuleGroupedTensor:
    @pytest.mark.parametrize("egtp_remat_size", (1, 2))
    def test_egtp_module_grouped_tensor_backward(self, egtp_remat_size):
        """Select the supported module backend and complete BF16 backward with EGTP=1 or 2."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_egtp_module_grouped_tensor_backward, 4, egtp_remat_size)

    def test_egtp_module_grouped_tensor_backward_te_fallback(self, monkeypatch):
        """An old TE must disable grouped tensors on both linears and still finish backward.

        Without this the split-quantize branch of the guard goes uncovered on a TE that does
        support sharded weights.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        from megatron.core.transformer.moe import experts as experts_mod

        monkeypatch.setattr(
            experts_mod, "_te_grouped_tensor_supports_sharded_weights", lambda: False
        )
        _run_distributed(_worker_egtp_module_grouped_tensor_backward, 4, 2)
