# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness with local partial CUDA graphs.

This is the local-CUDA-graph counterpart of ``test_gtp_loss_correctness.py``. Under the same
GTP2 x DP2 topology it compares eager execution with local CUDA graphs over two scopes: dense
attention layers, and an MoE router whose latent projections are GTP-sharded. The loss and global
gradient norm must match at every step, starting with the first replay after capture.
"""

import copy

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import (
    HAVE_GTP,
    classify_gtp_remat_chains,
    is_graphed_chain,
    wait_for_gtp_grad_reduction_on_current_stream,
)

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _run_distributed


def _worker_gtp_partial_cg_correctness(rank, world_size, port, partial_cg_modules, opt_in_modules):
    """Compare eager and local CUDA graphs with GTP2 x DP2."""
    del port

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
    hidden = 256
    num_heads = 8
    # Two attention layers make each graph prefetch into, and reduce from, a neighbouring graph.
    num_layers = 1 if latent_projection_case else 2
    sequence_length = 16 if latent_projection_case else 32
    batch_size = 1
    learning_rate = 0.01
    steps = 4
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
                "moe_ffn_hidden_size": 512,
                "moe_grouped_gemm": True,
                "moe_token_dispatcher_type": "allgather",
                "moe_aux_loss_coeff": 0.0,
                "moe_latent_size": 128,
            }
        return TransformerConfig(
            num_attention_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden,
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
            grads.append(trained_grad(param))
        return float(get_grad_norm_fp32(grads, grad_stats_parallel_group=grad_stats_group))

    def trained_grad(param):
        """The gradient DDP would reduce for ``param``.

        GTP params and params with fused wgrad accumulation keep the real gradient in main_grad;
        their ``.grad`` is TE's uninitialized dummy (DDP discards it, see grad_added_to_main_grad).
        The GTP reduce-scatter already returns the gtp mean, so main_grad is the full gradient.
        """
        if isinstance(param, GTPShardedParam) or param.grad_added_to_main_grad:
            return param.main_grad
        return param.grad if param.grad is not None else param.main_grad

    def apply_sgd_step(layers):
        with torch.no_grad():
            for param in layers.parameters():
                param.data.sub_(learning_rate * trained_grad(param))
                param.grad = None

    # Eager reference: GTP2 x DP2.
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
        apply_sgd_step(eager)

    del eager, loss, x
    torch.cuda.synchronize()
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
    dp_rank = dp_group.rank()
    assert gtp_size == gtp_degree
    assert dp_group.size() == dp_degree
    assert dp_rank == eager_dp_rank
    gtp_params = [param for param in partial_cg.parameters() if isinstance(param, GTPShardedParam)]
    assert gtp_params, "GTP not active: no GTPShardedParam found"
    params_in_scope = get_latent_params(partial_cg) if latent_projection_case else gtp_params
    assert params_in_scope
    assert all(is_graphed_chain(param.chain_id) for param in params_in_scope)
    for name, param in partial_cg.named_parameters():
        param.data.copy_(saved_local_weights[name])
    # Production captures after DDP maps every parameter into a main-grad buffer and initializes
    # the fused-accumulation marker. Mirror those invariants without pulling the full DDP stack into
    # this focused GTP/CUDA-graph test.
    initialize_main_grads(partial_cg)

    partial_cg_losses = []
    partial_cg_grad_norms = []
    try:
        # One eager step records the graph order; it does not update the weights. Every step of
        # the trajectory below then replays, so its first step checks the first replay.
        reset_grad_state(partial_cg)
        run_step(partial_cg, make_replica_input(0, dp_rank).requires_grad_()).backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        create_cudagraphs()
        assert _CudagraphGlobalRecord.cudagraph_created
        managers = get_cudagraph_managers(partial_cg)
        assert all(len(manager.cudagraph_runners) == 1 for manager in managers)
        assert any(manager.cudagraph_runners[0].gtp_remat for manager in managers)

        for step in range(steps):
            reset_grad_state(partial_cg)
            x = make_replica_input(step * world_size, dp_rank)
            x.requires_grad_()
            loss = run_step(partial_cg, x)
            partial_cg_losses.append(loss.item())
            loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()
            partial_cg_grad_norms.append(global_grad_norm(partial_cg, gtp_group))
            apply_sgd_step(partial_cg)
        del loss, x
    finally:
        delete_cuda_graphs()
        for manager in get_cudagraph_managers(partial_cg):
            manager.cudagraph_runners.clear()
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()

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
            # On H100 CI the first replay after capture differed from eager in the full GTP bucket;
            # it never reproduced on GB300. Cause open.
            pytest.param(["attn"], [], id="attention", marks=pytest.mark.flaky_in_dev),
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
