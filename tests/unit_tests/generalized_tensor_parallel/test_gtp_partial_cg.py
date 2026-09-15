# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness with local partial CUDA graphs.

This is the local-CUDA-graph counterpart of ``test_gtp_loss_correctness.py``. It compares eager
execution with per-module and coalesced local CUDA graphs under the same GTP2 x DP2 topology. It
verifies outputs, gradients, the complete loss trajectory, and global gradient norm, including
repeated replays of one backward.
"""

import copy
import gc

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import (
    HAVE_GTP,
    GTPChain,
    classify_gtp_remat_chains,
    deregister_and_clear_gtp_symm_pools,
    register_gtp_symm_pool,
    wait_for_gtp_grad_reduction_on_current_stream,
)

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _worker_gtp_partial_cg_correctness(
    rank,
    world_size,
    port,
    partial_cg_modules,
    opt_in_modules,
    coalesce_partial_captures,
    use_symmetric_memory,
    repeated_span_uses,
):
    """Compare eager and local CUDA graphs with GTP2 x DP2."""
    del port
    gtp_module._GTP_PARAMS.clear()

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel import param_is_not_gtp_duplicate
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.cuda_graphs import (
        _CudagraphGlobalRecord,
        create_cudagraphs,
        delete_cuda_graphs,
    )
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    latent_projection_case = "moe_latent_proj" in opt_in_modules
    span_layer_pattern = "MEM*E"
    hidden = 256 if latent_projection_case else 4096
    num_heads = 8 if latent_projection_case else 32
    ffn_hidden = 512 if latent_projection_case else 16384
    # Use multiple layers to exercise repeated local CUDA-graph execution with GTP parameters.
    num_layers = (
        len(span_layer_pattern)
        if coalesce_partial_captures
        else (1 if latent_projection_case else 4)
    )
    sequence_length = 16 if latent_projection_case else 32
    batch_size = 1
    learning_rate = 0.01
    steps = 10
    dtype = torch.bfloat16
    gtp_degree = 2
    dp_degree = 2
    assert world_size == gtp_degree * dp_degree

    def make_config(*, partial_cg=False):
        moe_options = {}
        if latent_projection_case:
            moe_options = {
                "num_moe_experts": 2,
                "moe_router_topk": 1,
                "moe_router_pre_softmax": True,
                "moe_ffn_hidden_size": ffn_hidden,
                "moe_grouped_gemm": True,
                "moe_token_dispatcher_type": "allgather",
                "moe_aux_loss_coeff": 0.0,
                "moe_latent_size": 128,
            }
        return TransformerConfig(
            num_attention_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden,
            ffn_hidden_size=ffn_hidden,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            # Repeated use changes expert-GEMM launch ordering. Disable TE's fused accumulation
            # there so this strict comparison measures CUDA-graph correctness, not eager jitter.
            gradient_accumulation_fusion=latent_projection_case and repeated_span_uses == 1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_weight_remat_size=gtp_degree,
            gtp_remat_opt_in_modules=opt_in_modules,
            cuda_graph_impl="local" if partial_cg else "none",
            cuda_graph_modules=partial_cg_modules if partial_cg else [],
            cuda_graph_coalesce_partial_captures=(partial_cg and coalesce_partial_captures),
            cuda_graph_warmup_steps=2,
            **moe_options,
        )

    def make_layer_stack(config, pg_collection):
        spec = copy.deepcopy(
            get_gpt_layer_with_transformer_engine_spec(
                num_experts=2 if latent_projection_case else None,
                moe_grouped_gemm=latent_projection_case,
            )
        )
        if latent_projection_case:
            spec.submodules.input_layernorm = IdentityOp
            spec.submodules.self_attention = IdentityOp
            spec.submodules.self_attn_bda = IdentityFuncOp
            if coalesce_partial_captures:
                span_submodules = copy.deepcopy(hybrid_stack_spec.submodules)
                span_submodules.moe_layer = ModuleSpec(
                    module=MoETransformerLayer, submodules=spec.submodules
                )
                return HybridStack(
                    config,
                    span_submodules,
                    layer_config_list=validate_segment_layers(span_layer_pattern, config),
                    pp_layer_offset=0,
                    post_layer_norm=False,
                    pg_collection=pg_collection,
                )
            return torch.nn.ModuleList(
                [
                    MoETransformerLayer(
                        config,
                        spec.submodules,
                        layer_number=1,
                        pg_collection=pg_collection,
                        name="decoder.layers.0",
                    )
                ]
            )

        spec.submodules.pre_mlp_layernorm = IdentityOp
        spec.submodules.mlp = IdentityOp
        spec.submodules.mlp_bda = IdentityFuncOp
        return torch.nn.ModuleList(
            [
                spec.module(
                    config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection
                )
                for i in range(num_layers)
            ]
        )

    def get_cudagraph_managers(layers):
        if coalesce_partial_captures:
            return [span.cudagraph_manager for span in layers._cuda_graph_spans]
        if latent_projection_case:
            return [
                manager
                for layer in layers
                for manager in (layer.cudagraph_manager_router, layer.cudagraph_manager_postprocess)
            ]
        return [layer.cudagraph_manager for layer in layers]

    def get_latent_params(layers):
        return [
            param
            for name, param in layers.named_parameters()
            if "fc1_latent_proj.weight" in name or "fc2_latent_proj.weight" in name
        ]

    def make_pg_collection():
        if latent_projection_case:
            return ProcessGroupCollection.use_mpu_process_groups()
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "gtp_remat"])

    def run_step(layers, x):
        span_use_outputs = []
        with fp8_autocast(enabled=False):
            if coalesce_partial_captures:
                for _ in range(repeated_span_uses):
                    x = layers(x, attention_mask=None)
                    span_use_outputs.append(x.detach().float().cpu().clone())
            else:
                for layer in layers:
                    x, _ = layer(x, attention_mask=None)
        output = x.detach().float().cpu().clone() if coalesce_partial_captures else None
        loss = x.float().mean()
        return output, loss, span_use_outputs

    def reset_grad_state(layers):
        for param in layers.parameters():
            if hasattr(param, "main_grad"):
                param.main_grad.zero_()
            param.grad = None
            # DDP resets this before every local-CG training iteration.
            param.grad_added_to_main_grad = False

    def initialize_main_grads(layers):
        for param in layers.parameters():
            if not hasattr(param, "main_grad"):
                param.main_grad = torch.zeros_like(param)
            param.grad_added_to_main_grad = False

    def make_replica_input(seed, replica_rank):
        # This focused test does not instantiate DDP. Keep each GTP group on one microbatch so
        # replicated parameters stay synchronized, while the two DP replicas exercise different
        # trajectories.
        torch.manual_seed(seed + replica_rank)
        return torch.randn(sequence_length, batch_size, hidden, dtype=dtype, device="cuda")

    def get_param_grad(param):
        if isinstance(param, GTPShardedParam) or getattr(param, "grad_added_to_main_grad", False):
            return param.main_grad
        return param.grad if param.grad is not None else param.main_grad

    def global_grad_norm(layers, grad_stats_group):
        """Mirror Megatron's GTP duplicate filtering and global L2-norm reduction."""
        grads = []
        for param in layers.parameters():
            if not param_is_not_gtp_duplicate(param):
                continue
            grad = get_param_grad(param)
            assert grad is not None
            # Hybrid stacks mix FP32 Mamba state grads with BF16 model grads. Apex's single
            # multi-tensor L2 invocation expects one dtype, so normalize this test input to FP32.
            grads.append(grad.detach().float())
        return float(get_grad_norm_fp32(grads, grad_stats_parallel_group=grad_stats_group))

    def snapshot_param_grads(layers):
        grads = {}
        for name, param in layers.named_parameters():
            grad = get_param_grad(param)
            assert grad is not None, f"Missing gradient for {name}"
            grads[name] = grad.detach().float().cpu().clone()
        return grads

    def assert_param_grads_close(actual, expected):
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=1e-6,
                rtol=5e-3,
                msg=lambda message: f"Gradient mismatch for {name}: {message}",
            )

    def apply_sgd_step(layers, gtp_size):
        with torch.no_grad():
            for param in layers.parameters():
                if isinstance(param, GTPShardedParam):
                    param.data.sub_((learning_rate / gtp_size) * param.main_grad)
                else:
                    grad = get_param_grad(param)
                    param.data.sub_(learning_rate * grad)
                    param.grad = None

    # Eager reference: GTP2 x DP2.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_degree
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = make_pg_collection()
    if use_symmetric_memory:
        register_gtp_symm_pool(ps.get_gtp_weight_remat_group())
    eager_config = make_config()
    eager = make_layer_stack(eager_config, pg_collection).cuda()
    eager_gtp_group = ps.get_gtp_weight_remat_group()
    eager_dp_group = ps.get_data_parallel_group(with_gtp_remat=False)
    eager_dp_rank = eager_dp_group.rank()
    assert eager_gtp_group.size() == gtp_degree
    assert eager_dp_group.size() == dp_degree
    assert any(isinstance(param, GTPShardedParam) for param in eager.parameters())
    if latent_projection_case:
        eager_latent_params = get_latent_params(eager)
        expected_latent_params = 2 * span_layer_pattern.count(Symbols.MOE)
        if not coalesce_partial_captures:
            expected_latent_params = 2
        assert len(eager_latent_params) == expected_latent_params
        assert all(isinstance(param, GTPShardedParam) for param in eager_latent_params)
    initialize_main_grads(eager)
    saved_local_weights = {name: param.data.clone() for name, param in eager.named_parameters()}

    eager_losses = []
    eager_grad_norms = []
    eager_outputs = []
    eager_input_grads = []
    eager_param_grads = []
    for step in range(steps):
        reset_grad_state(eager)
        x = make_replica_input(step * world_size, eager_dp_rank)
        x.requires_grad_()
        output, loss, _ = run_step(eager, x)
        eager_losses.append(loss.item())
        loss.backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        eager_grad_norms.append(global_grad_norm(eager, eager_gtp_group))
        if coalesce_partial_captures:
            eager_outputs.append(output)
            eager_input_grads.append(x.grad.detach().float().cpu().clone())
            eager_param_grads.append(snapshot_param_grads(eager))
        apply_sgd_step(eager, eager_gtp_group.size())

    del eager, loss, output, x
    torch.cuda.synchronize()
    if use_symmetric_memory:
        deregister_and_clear_gtp_symm_pools()
    ps.destroy_model_parallel()
    gtp_module.reset_gtp_state()
    gtp_module._GTP_PARAMS.clear()

    # Optimized path: the same GTP2 x DP2 topology with local CUDA graphs.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_degree
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(42)
    pg_collection = make_pg_collection()
    if use_symmetric_memory:
        register_gtp_symm_pool(ps.get_gtp_weight_remat_group())
    partial_cg_config = make_config(partial_cg=True)
    partial_cg = make_layer_stack(partial_cg_config, pg_collection).cuda()
    if coalesce_partial_captures:
        assert all(not hasattr(layer, "cudagraph_manager") for layer in partial_cg.layers)
        for layer in partial_cg.layers:
            if isinstance(layer, MoETransformerLayer):
                assert not hasattr(layer, "cudagraph_manager_router")
                assert not hasattr(layer, "cudagraph_manager_postprocess")
    classify_gtp_remat_chains(
        partial_cg,
        cuda_graph_modules=partial_cg_config.cuda_graph_modules,
        cuda_graph_impl=partial_cg_config.cuda_graph_impl,
    )

    gtp_group = ps.get_gtp_weight_remat_group()
    dp_group = ps.get_data_parallel_group(with_gtp_remat=False)
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    dp_rank = dp_group.rank()
    assert gtp_size == gtp_degree
    assert dp_group.size() == dp_degree
    assert dp_rank == eager_dp_rank
    gtp_params = [param for param in partial_cg.parameters() if isinstance(param, GTPShardedParam)]
    assert gtp_params, "GTP not active: no GTPShardedParam found"
    if latent_projection_case:
        latent_params = get_latent_params(partial_cg)
        expected_chain = (
            GTPChain.GRAPHED.value
            if "moe_router" in partial_cg_modules
            else GTPChain.UNGRAPHED.value
        )
        assert all(param.chain_id == expected_chain for param in latent_params)
    assert any(param.chain_id == GTPChain.GRAPHED.value for param in gtp_params)
    for name, param in partial_cg.named_parameters():
        param.data.copy_(saved_local_weights[name])
    # Production captures after DDP maps every parameter into a main-grad buffer and initializes
    # the fused-accumulation marker. Mirror those invariants without pulling the full DDP stack into
    # this focused GTP/CUDA-graph test.
    initialize_main_grads(partial_cg)

    partial_cg_losses = []
    partial_cg_grad_norms = []
    partial_cg_outputs = []
    partial_cg_input_grads = []
    partial_cg_param_grads = []
    try:
        # Record one eager backward, then replay the same input and weights. This isolates the
        # graph execution path: model state, GTP topology, and reduction order are unchanged.
        reset_grad_state(partial_cg)
        eager_probe_x = make_replica_input(1234, dp_rank)
        eager_probe_x.requires_grad_()
        eager_probe_output, eager_probe_loss, eager_probe_span_outputs = run_step(
            partial_cg, eager_probe_x
        )
        eager_probe_loss.backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        eager_grad_norm = global_grad_norm(partial_cg, gtp_group)
        eager_probe_loss_value = eager_probe_loss.item()
        if coalesce_partial_captures:
            eager_probe_output_value = eager_probe_output
            eager_probe_input_grad = eager_probe_x.grad.detach().float().cpu().clone()
            eager_probe_param_grads = snapshot_param_grads(partial_cg)
        pre_capture_param_grads = snapshot_param_grads(partial_cg)
        del eager_probe_loss, eager_probe_output, eager_probe_x

        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created
        post_capture_param_grads = snapshot_param_grads(partial_cg)
        assert post_capture_param_grads.keys() == pre_capture_param_grads.keys()
        for name in pre_capture_param_grads:
            torch.testing.assert_close(
                post_capture_param_grads[name],
                pre_capture_param_grads[name],
                atol=0,
                rtol=0,
                msg=lambda message: f"Capture changed finalized gradient for {name}: {message}",
            )
        managers = get_cudagraph_managers(partial_cg)
        assert all(len(manager.cudagraph_runners) == repeated_span_uses for manager in managers)
        runners = [manager.cudagraph_runners[0] for manager in managers]
        assert any(runner.gtp_remat for runner in runners)
        if coalesce_partial_captures:
            assert partial_cg._cuda_graph_span_plan is not None
            expected_span_count = 3 if "moe_router" in partial_cg_modules else 2
            assert len(partial_cg._cuda_graph_spans) == expected_span_count
            assert all(runner.is_hybrid_cuda_graph_span for runner in runners)

        replay_grad_norms = []
        replay_losses = []
        for _ in range(3):
            reset_grad_state(partial_cg)
            replay_x = make_replica_input(1234, dp_rank).requires_grad_()
            replay_output, replay_loss, replay_span_outputs = run_step(partial_cg, replay_x)
            replay_loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()
            replay_losses.append(replay_loss.item())
            replay_grad_norms.append(global_grad_norm(partial_cg, gtp_group))
            if coalesce_partial_captures:
                for span_use, (actual, expected) in enumerate(
                    zip(replay_span_outputs, eager_probe_span_outputs, strict=True)
                ):
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=1e-6,
                        rtol=5e-3,
                        msg=lambda message, span_use=span_use: (
                            f"Span-stack use {span_use} output mismatch: {message}"
                        ),
                    )
                torch.testing.assert_close(
                    replay_output, eager_probe_output_value, atol=1e-6, rtol=5e-3
                )
                torch.testing.assert_close(
                    replay_x.grad.detach().float().cpu(),
                    eager_probe_input_grad,
                    atol=1e-6,
                    rtol=5e-3,
                )
                assert_param_grads_close(snapshot_param_grads(partial_cg), eager_probe_param_grads)

        replay_grad_norms_tensor = torch.tensor(replay_grad_norms)
        assert torch.isfinite(replay_grad_norms_tensor).all()
        torch.testing.assert_close(
            replay_grad_norms_tensor,
            torch.full_like(replay_grad_norms_tensor, eager_grad_norm),
            atol=1e-6,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            torch.tensor(replay_losses),
            torch.full((len(replay_losses),), eager_probe_loss_value),
            atol=1e-6,
            rtol=5e-3,
        )
        if gtp_rank == 0:
            print(
                f"[partial-CG grad norm, DP replica {dp_rank}] "
                f"eager={eager_grad_norm:.6f} replays={replay_grad_norms}",
                flush=True,
            )

        del replay_loss, replay_output, replay_x

        for step in range(steps):
            reset_grad_state(partial_cg)
            x = make_replica_input(step * world_size, dp_rank)
            x.requires_grad_()
            output, loss, _ = run_step(partial_cg, x)
            partial_cg_losses.append(loss.item())
            loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()
            partial_cg_grad_norms.append(global_grad_norm(partial_cg, gtp_group))
            if coalesce_partial_captures:
                partial_cg_outputs.append(output)
                partial_cg_input_grads.append(x.grad.detach().float().cpu().clone())
                partial_cg_param_grads.append(snapshot_param_grads(partial_cg))
            apply_sgd_step(partial_cg, gtp_size)
        del loss, output, x
    finally:
        torch.cuda.synchronize()
        managers = get_cudagraph_managers(partial_cg)
        for manager in managers:
            for runner in manager.cudagraph_runners:
                if runner.fwd_graph is not None:
                    runner.fwd_graph.reset()
                if runner.bwd_graph is not None:
                    runner.bwd_graph.reset()
        delete_cuda_graphs()
        for manager in managers:
            manager.cudagraph_runners.clear()
        if use_symmetric_memory:
            deregister_and_clear_gtp_symm_pools()
        gtp_module.reset_gtp_state()
        gc.collect()
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        gtp_module._GTP_PARAMS.clear()

    if rank == 0:
        for step, (eager_loss, partial_cg_loss) in enumerate(zip(eager_losses, partial_cg_losses)):
            print(
                f"Step {step:2d}: eager={eager_loss:.6f}  partial_cg={partial_cg_loss:.6f}",
                flush=True,
            )
    torch.testing.assert_close(
        torch.tensor(partial_cg_losses), torch.tensor(eager_losses), atol=1e-6, rtol=5e-3
    )
    torch.testing.assert_close(
        torch.tensor(partial_cg_grad_norms), torch.tensor(eager_grad_norms), atol=1e-6, rtol=5e-3
    )
    if coalesce_partial_captures:
        for actual, expected in zip(partial_cg_outputs, eager_outputs, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=5e-3)
        for actual, expected in zip(partial_cg_input_grads, eager_input_grads, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=5e-3)
        for actual, expected in zip(partial_cg_param_grads, eager_param_grads, strict=True):
            assert_param_grads_close(actual, expected)


class TestGTPPartialCGCorrectness:
    @pytest.mark.parametrize(
        "partial_cg_modules,opt_in_modules,coalesce_partial_captures,use_symmetric_memory,"
        "repeated_span_uses",
        [
            pytest.param(["attn"], [], False, False, 1, id="attention"),
            pytest.param(
                ["moe_router"],
                ["moe_latent_proj"],
                False,
                False,
                1,
                id="moe-router-latent-projections",
            ),
            pytest.param(
                ["mamba", "attn", "moe_router"],
                ["moe_latent_proj"],
                True,
                False,
                1,
                id="mamba-attn-moe-router-coalesced",
            ),
            pytest.param(
                ["mamba", "attn"], ["moe_latent_proj"], True, False, 1, id="mamba-attn-coalesced"
            ),
            pytest.param(
                ["mamba", "attn"],
                ["moe_latent_proj"],
                True,
                False,
                2,
                id="mamba-attn-coalesced-repeated-use",
            ),
            pytest.param(
                ["moe_router"],
                ["moe_latent_proj"],
                False,
                True,
                1,
                id="moe-router-latent-projections-symmetric-memory",
            ),
            pytest.param(
                ["mamba", "attn", "moe_router"],
                ["moe_latent_proj"],
                True,
                True,
                1,
                id="mamba-attn-moe-router-coalesced-symmetric-memory",
            ),
            pytest.param(
                ["mamba", "attn"],
                ["moe_latent_proj"],
                True,
                True,
                2,
                id="mamba-attn-coalesced-repeated-use-symmetric-memory",
            ),
        ],
    )
    def test_gtp_partial_cg_loss_and_grad_norm_match_eager(
        self,
        partial_cg_modules,
        opt_in_modules,
        coalesce_partial_captures,
        use_symmetric_memory,
        repeated_span_uses,
    ):
        """Local-CG outputs, gradients, and optimization trajectory must match eager."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(
            _worker_gtp_partial_cg_correctness,
            4,
            partial_cg_modules,
            opt_in_modules,
            coalesce_partial_captures,
            use_symmetric_memory,
            repeated_span_uses,
        )
