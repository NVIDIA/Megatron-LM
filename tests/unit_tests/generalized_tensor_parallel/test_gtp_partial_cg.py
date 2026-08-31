# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness with local partial CUDA graphs.

This is the local-CUDA-graph counterpart of ``test_gtp_loss_correctness.py``. It compares eager
execution with local CUDA graphs under the same GTP topology, including FP32-accumulation RS. It
verifies the complete loss trajectory and global gradient norm across repeated graph replays.
"""

import copy
import gc

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import (
    GTP_CONFIG,
    HAVE_GTP,
    GTPChain,
    classify_gtp_remat_chains,
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
    gtp_degree=2,
    fp32_accumulation=False,
):
    """Compare eager and local CUDA graphs with the requested GTP topology."""
    del port
    gtp_module._GTP_PARAMS.clear()
    saved_fp32_accumulation = GTP_CONFIG.reduce_scatter_with_fp32_accumulation
    GTP_CONFIG.reduce_scatter_with_fp32_accumulation = fp32_accumulation

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
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
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    latent_projection_case = "moe_latent_proj" in opt_in_modules
    hidden = 256 if latent_projection_case else 4096
    num_heads = 8 if latent_projection_case else 32
    ffn_hidden = 512 if latent_projection_case else 16384
    # Use multiple layers to exercise repeated local CUDA-graph execution with GTP parameters.
    num_layers = 1 if latent_projection_case else 4
    sequence_length = 16 if latent_projection_case else 32
    batch_size = 1
    learning_rate = 0.01
    steps = 10
    dtype = torch.bfloat16
    assert world_size % gtp_degree == 0
    dp_degree = world_size // gtp_degree
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
            gradient_accumulation_fusion=latent_projection_case,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_weight_remat_size=gtp_degree,
            gtp_remat_opt_in_modules=opt_in_modules,
            cuda_graph_impl="local" if partial_cg else "none",
            cuda_graph_modules=partial_cg_modules if partial_cg else [],
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
        with fp8_autocast(enabled=False):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
        return x.mean()

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

    def global_grad_norm(layers, grad_stats_group):
        """Mirror Megatron's GTP duplicate filtering and global L2-norm reduction."""
        grads = []
        for param in layers.parameters():
            if not param_is_not_gtp_duplicate(param):
                continue
            if isinstance(param, GTPShardedParam):
                grad = param.main_grad
            else:
                grad = param.grad if param.grad is not None else param.main_grad
            assert grad is not None
            grads.append(grad)
        return float(get_grad_norm_fp32(grads, grad_stats_parallel_group=grad_stats_group))

    def apply_sgd_step(layers, gtp_size):
        with torch.no_grad():
            for param in layers.parameters():
                if isinstance(param, GTPShardedParam):
                    param.data.sub_((learning_rate / gtp_size) * param.main_grad)
                else:
                    grad = param.grad if param.grad is not None else param.main_grad
                    param.data.sub_(learning_rate * grad)
                    param.grad = None

    # Eager reference with the requested GTP x DP topology.
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_degree
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = make_pg_collection()
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
        assert len(eager_latent_params) == 2
        assert all(isinstance(param, GTPShardedParam) for param in eager_latent_params)
    initialize_main_grads(eager)
    saved_local_weights = {name: param.data.clone() for name, param in eager.named_parameters()}

    eager_losses = []
    eager_grad_norms = []
    for step in range(steps):
        reset_grad_state(eager)
        x = make_replica_input(step * world_size, eager_dp_rank)
        x.requires_grad_()
        loss = run_step(eager, x)
        eager_losses.append(loss.item())
        loss.backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        eager_grad_norms.append(global_grad_norm(eager, eager_gtp_group))
        apply_sgd_step(eager, eager_gtp_group.size())

    del eager, loss, x
    torch.cuda.synchronize()
    ps.destroy_model_parallel()
    gtp_module.reset_gtp_state()
    gtp_module._GTP_PARAMS.clear()

    # Optimized path: the same topology with local CUDA graphs.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_degree
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(42)
    pg_collection = make_pg_collection()
    partial_cg_config = make_config(partial_cg=True)
    partial_cg = make_layer_stack(partial_cg_config, pg_collection).cuda()
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
    params_in_scope = get_latent_params(partial_cg) if latent_projection_case else gtp_params
    assert all(param.chain_id == GTPChain.GRAPHED.value for param in params_in_scope)
    for name, param in partial_cg.named_parameters():
        param.data.copy_(saved_local_weights[name])
    # Production captures after DDP maps every parameter into a main-grad buffer and initializes
    # the fused-accumulation marker. Mirror those invariants without pulling the full DDP stack into
    # this focused GTP/CUDA-graph test.
    initialize_main_grads(partial_cg)

    partial_cg_losses = []
    partial_cg_grad_norms = []
    try:
        # Record one eager backward, then replay the same input and weights. This isolates the
        # graph execution path: model state, GTP topology, and reduction order are unchanged.
        reset_grad_state(partial_cg)
        eager_probe_x = make_replica_input(1234, dp_rank)
        eager_probe_x.requires_grad_()
        eager_probe_loss = run_step(partial_cg, eager_probe_x)
        eager_probe_loss.backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        eager_grad_norm = global_grad_norm(partial_cg, gtp_group)
        eager_probe_loss_value = eager_probe_loss.item()
        del eager_probe_loss, eager_probe_x
        reset_grad_state(partial_cg)

        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created
        managers = get_cudagraph_managers(partial_cg)
        assert all(len(manager.cudagraph_runners) == 1 for manager in managers)
        runners = [manager.cudagraph_runners[0] for manager in managers]
        assert any(runner.gtp_remat for runner in runners)
        assert any(runner.persistent_buffer_state.capacities for runner in runners)
        if fp32_accumulation:
            assert any(
                any(
                    dtype == torch.float32 for _, dtype in runner.persistent_buffer_state.capacities
                )
                for runner in runners
            )

        replay_grad_norms = []
        replay_losses = []
        for _ in range(3):
            reset_grad_state(partial_cg)
            replay_x = make_replica_input(1234, dp_rank).requires_grad_()
            replay_loss = run_step(partial_cg, replay_x)
            replay_loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()
            replay_losses.append(replay_loss.item())
            replay_grad_norms.append(global_grad_norm(partial_cg, gtp_group))

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

        del replay_loss, replay_x

        for step in range(steps):
            reset_grad_state(partial_cg)
            x = make_replica_input(step * world_size, dp_rank)
            x.requires_grad_()
            loss = run_step(partial_cg, x)
            partial_cg_losses.append(loss.item())
            loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()
            partial_cg_grad_norms.append(global_grad_norm(partial_cg, gtp_group))
            apply_sgd_step(partial_cg, gtp_size)
        del loss, x
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
        gc.collect()
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        gtp_module.reset_gtp_state()
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation = saved_fp32_accumulation
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


class TestGTPPartialCGCorrectness:
    @pytest.mark.parametrize(
        "partial_cg_modules,opt_in_modules",
        [
            pytest.param(["attn"], [], id="attention"),
            pytest.param(["moe_router"], ["moe_latent_proj"], id="moe-router-latent-projections"),
        ],
    )
    def test_gtp_partial_cg_loss_and_grad_norm_match_eager(
        self, partial_cg_modules, opt_in_modules
    ):
        """Local-CG loss trajectory and global grad norm must match eager execution."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_partial_cg_correctness, 4, partial_cg_modules, opt_in_modules)

    def test_gtp_fp32_accumulation_partial_cg_matches_eager(self):
        """GTP4 FP32-accumulation workspaces must remain stable across graph replays."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_partial_cg_correctness, 4, ["attn"], [], 4, True)
