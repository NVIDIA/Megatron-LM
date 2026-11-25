# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Partial CUDA graphs composed with real GPT/Hybrid MoE training and MFSDP v2."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from transformer_engine.pytorch import graph as te_graph

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    RuntimePolicy,
    fully_shard_context,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_groups(distributed_setup):
    """Use EP2 on multiple ranks and EP1 for single-rank alias-buffer coverage."""
    if distributed_setup.world_size > 1 and distributed_setup.world_size % 2:
        pytest.skip("Requires one rank or an even world size.")
    destroy_moe_metrics_tracker()
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=min(2, distributed_setup.world_size)
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    init_num_microbatches_calculator(
        rank=distributed_setup.rank,
        global_batch_size=2 * distributed_setup.world_size,
        micro_batch_size=1,
        data_parallel_size=distributed_setup.world_size,
    )
    yield groups
    destroy_moe_metrics_tracker()
    destroy_num_microbatches_calculator()
    destroy_model_parallel()


PARTIAL_SCOPES = ("attn", "mlp", "moe_router", "moe_preprocess")


def _moe_config(mla=False, hybrid=False):
    config_type = MLATransformerConfig if mla else TransformerConfig
    mla_options = (
        dict(q_lora_rank=32, kv_lora_rank=32, qk_head_dim=16, qk_pos_emb_head_dim=16, v_head_dim=16)
        if mla
        else {}
    )
    return config_type(
        num_layers=4 if hybrid else 2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_layer_freq=[0, 0, 0, 1] if hybrid else [0, 1],
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        multi_latent_attention=mla,
        **mla_options,
    )


def _build_moe_model(config, process_groups, hybrid, capture_modules, strategy, graphs):
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
    model_config = replace(
        config,
        expert_model_parallel_size=process_groups.ep.size(),
        cuda_graph_impl="transformer_engine" if graphs else "none",
        cuda_graph_modules=capture_modules,
    )
    model_type = HybridModel if hybrid else GPTModel
    model_options = (
        dict(hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*-*E")
        if hybrid
        else dict(
            transformer_layer_spec=get_gpt_decoder_block_spec(
                model_config, use_transformer_engine=True
            )
        )
    )
    model = (
        model_type(
            config=model_config,
            vocab_size=128,
            max_sequence_length=16,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            pg_collection=process_groups,
            **model_options,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    if not hybrid:
        assert not model.decoder.layers[0].is_moe_layer
        assert model.decoder.layers[1].is_moe_layer
    model = FullyShardedDataParallel(
        config=model_config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy=strategy,
            megatron_fsdp_main_grads_dtype=torch.float32,
        ),
        module=model,
        pg_collection=process_groups,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=1e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        ),
        [model],
    )
    optimizer.reload_model_params()
    return model, optimizer


def _train_moe_step(model, optimizer, batch, *, snapshot=True):
    """Train one step and optionally return its loss and training state."""
    model.train()
    model.zero_grad_buffer()
    optimizer.zero_grad()
    positions = torch.arange(16, device="cuda").unsqueeze(0)
    mask = ~torch.ones((1, 1, 16, 16), device="cuda", dtype=torch.bool).tril()
    iterator = iter(batch)

    def forward():
        tokens = next(iterator)
        return model(tokens[:, :-1], positions, mask, labels=tokens[:, 1:]).float().mean()

    losses = run_forward_backward_on_microbatches(
        model, forward, num_microbatches=2, loss_scale=0.5
    )
    gradients = [p.grad.to_local().detach().cpu() for p in model.parameters()] if snapshot else []
    success, _, _ = optimizer.step()
    assert success
    if not snapshot:
        return
    return (
        torch.stack(losses).cpu(),
        gradients,
        [p.to_local().detach().cpu() for p in model.parameters()],
        [
            state[key].to_local().detach().cpu()
            for state in optimizer.optimizer.state.values()
            for key in ("exp_avg", "exp_avg_sq")
        ],
        [
            torch.as_tensor(group.get("step", 0)).detach().cpu().clone()
            for group in optimizer.optimizer.param_groups
        ],
    )


def _rng_contents():
    """Read state contents, rather than live graph-safe generator handles."""
    tracker = get_cuda_rng_tracker().get_states()
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        {name: state.get_state().clone() for name, state in tracker.items()},
    )


@pytest.mark.parametrize(
    "hybrid,mla,strategy,scopes,separate_streams",
    [
        pytest.param(False, False, "optim_grads_params", PARTIAL_SCOPES, False, id="gpt-zero3"),
        pytest.param(False, True, "optim_grads", ("attn", "moe_router"), True, id="mla-zero2"),
        pytest.param(True, False, "optim", PARTIAL_SCOPES, True, id="hybrid-zero1"),
    ],
)
def test_partial_cuda_graph_moe_matches_eager(
    process_groups, hybrid, mla, strategy, scopes, separate_streams, monkeypatch
):
    """Check MoE training parity, communication streams and retained weight lifetime."""
    config = replace(_moe_config(mla, hybrid), attention_dropout=0.1, hidden_dropout=0.1)
    recapture = strategy == "optim_grads_params"
    torch.manual_seed(4321 + torch.distributed.get_rank())
    batches = [
        torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(7 if recapture else 5)
    ]

    def run(graphs):
        context = FsdpContext(
            runtime_policy=RuntimePolicy(overlap_allgather_reduction=separate_streams)
        )
        streams = (
            context.allgather_stream,
            context.inner_reduction_stream,
            context.outer_reduction_stream,
        )
        assert all(stream != torch.cuda.current_stream() for stream in streams)
        assert (streams[0] != streams[1]) == separate_streams
        with fully_shard_context(context):
            model, optimizer = _build_moe_model(
                config, process_groups, hybrid, list(scopes), strategy, graphs
            )
        groups = [
            group for layer in model.module.decoder.layers for group in layer.parameter_groups
        ]
        helper = None
        pointers = {}
        history = []
        try:
            for step, batch in enumerate(batches):
                if graphs and (step == 3 or (recapture and step == 6)):
                    helper = TECudaGraphHelper(
                        [model], model.config, 16, 1, [optimizer], process_groups
                    )
                    if recapture and step == 3:
                        with monkeypatch.context() as patch:
                            patch.setattr(model.config, "context_parallel_size", 2)
                            with pytest.raises(ValueError, match="TP=PP=CP=1"):
                                helper.create_cudagraphs()
                        assert not cuda_graphs.is_graph_capturing()
                    rng = _rng_contents()
                    helper.create_cudagraphs()
                    torch.testing.assert_close(_rng_contents(), rng, rtol=0, atol=0)
                    assert helper.graphs_created()
                    if recapture and step == 3:
                        duplicate = TECudaGraphHelper(
                            [model], model.config, 16, 1, [optimizer], process_groups
                        )
                        with pytest.raises(RuntimeError, match="Delete existing CUDA graphs"):
                            duplicate.create_cudagraphs()
                    pointers = {
                        group: group._unsharded_model_weight.local_buffer.data_ptr()
                        for layer in helper.flattened_callables
                        for group in layer.parameter_groups
                    }
                elif graphs and recapture and step == 5:
                    helper.delete_cuda_graphs()
                    assert all(not layer.te_cuda_graphs for layer in helper.flattened_callables)
                    assert all(not group._retain_unsharded_storage for group in groups)
                    for group in pointers:
                        buffer = group._unsharded_model_weight
                        if not buffer.is_alias_of(group.model_weight):
                            assert not buffer.is_allocated
                    helper = None
                    pointers.clear()

                with (
                    torch.profiler.profile()
                    if helper is not None and step in (3, 6)
                    else nullcontext()
                ) as profiler:
                    history.append(_train_moe_step(model, optimizer, batch))
                if profiler is not None:
                    # Two microbatches, with a forward and backward launch for each captured unit.
                    assert sum(event.name == "cudaGraphLaunch" for event in profiler.events()) == (
                        4 * len(helper.flattened_callables)
                    )
                for group in groups:
                    assert not group._unshard_events
                    buffer = group._unsharded_model_weight
                    if buffer.local_buffer.numel():
                        assert buffer.is_allocated == (
                            (group in pointers) or buffer.is_alias_of(group.model_weight)
                        )
                    if group in pointers:
                        assert buffer.local_buffer.data_ptr() == pointers[group]
                assert streams == (
                    context.allgather_stream,
                    context.inner_reduction_stream,
                    context.outer_reduction_stream,
                )
            return history
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["retry", "cleanup-error"])
def test_partial_capture_rollback(process_groups, cleanup_fails, monkeypatch):
    """Restore capture state and surface the original error even if cleanup raises."""
    config = replace(_moe_config(), attention_dropout=0.1, hidden_dropout=0.1)
    torch.manual_seed(92)
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(4)]

    def warm_up(graphs):
        model, optimizer = _build_moe_model(
            config, process_groups, False, list(PARTIAL_SCOPES), "optim_grads_params", graphs
        )
        for batch in batches[:3]:
            _train_moe_step(model, optimizer, batch, snapshot=False)
        return model, optimizer

    if not cleanup_fails:
        eager, eager_optimizer = warm_up(False)
        expected = _train_moe_step(eager, eager_optimizer, batches[-1])
        del eager, eager_optimizer
    model, optimizer = warm_up(True)
    helper = TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
    layers = helper.flattened_callables
    rng = _rng_contents()
    hook_names = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    hooks = [{name: dict(getattr(layer, name)) for name in hook_names} for layer in layers]
    original_calls = [type(layer).__call__ for layer in layers]
    original_start = helper._start_capturing

    def fail(*args, **kwargs):
        raise RuntimeError("controlled capture failure")

    def fail_after_start():
        original_start()
        fail()

    def fail_cleanup():
        raise RuntimeError("controlled cleanup failure")

    # Reuse the idle model to cover failures before and after graph installation.
    failures = [("_finish_capturing", fail)]
    if not cleanup_fails:
        failures.insert(0, ("_start_capturing", fail_after_start))
    try:
        if cleanup_fails:
            layer = layers[0]
            reshard = layer.reshard

            def fail_on_capture_exit():
                # Allow the helper's subsequent buffer-release attempt to finish.
                layer.reshard = reshard
                raise RuntimeError("controlled reshard failure")

            # Use TE's real caller-hook rejection as the primary failure.
            with (
                layer.register_forward_pre_hook(lambda *_: None) as handle,
                monkeypatch.context() as patch,
            ):
                patch.setattr(layer, "reshard", fail_on_capture_exit)
                with pytest.raises(RuntimeError, match="Modules must not have hooks") as error:
                    helper.create_cudagraphs()
                assert any("controlled reshard failure" in note for note in error.value.__notes__)
                assert handle.id in layer._forward_pre_hooks

        for method, inject_failure in failures:
            with monkeypatch.context() as patch:
                patch.setattr(helper, method, inject_failure)
                if cleanup_fails:
                    patch.setattr(helper, "_reset_after_capture", fail_cleanup)
                with pytest.raises(RuntimeError, match="controlled capture failure") as error:
                    helper.create_cudagraphs()
            if cleanup_fails:
                assert any("controlled cleanup failure" in note for note in error.value.__notes__)
            assert not (cuda_graphs.is_graph_capturing() or te_graph.is_graph_capturing())
            assert not helper.graphs_created()
            torch.testing.assert_close(_rng_contents(), rng, rtol=0, atol=0)
            for layer, original_hooks, original_call in zip(layers, hooks, original_calls):
                assert type(layer).__call__ is original_call
                assert all(getattr(layer, name) == original_hooks[name] for name in hook_names)
                assert not layer._fsdp_cuda_graph_capturing
                assert not layer.te_cuda_graphs
                assert all(not group._retain_unsharded_storage for group in layer.parameter_groups)

        if not cleanup_fails:
            helper.create_cudagraphs()
            assert helper.graphs_created()
            actual = _train_moe_step(model, optimizer, batches[-1])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if helper.graphs_created():
            helper.delete_cuda_graphs()
        cuda_graphs._set_capture_end()
