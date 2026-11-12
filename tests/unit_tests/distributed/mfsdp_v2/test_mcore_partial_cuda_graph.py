# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Partial CUDA graphs composed with real GPT/Hybrid MoE training and MFSDP v2."""

import gc
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
import torch.distributed.checkpoint as dcp
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
from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_groups(distributed_setup):
    """Use EP2 and independent expert-DP shards on four ranks."""
    if distributed_setup.world_size % 2:
        pytest.skip("Requires an even world size.")
    destroy_moe_metrics_tracker()
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
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


# Cover each scope and sharding mode without multiplying independent dimensions.
@pytest.mark.parametrize(
    "hybrid,mla,strategy,capture_modules",
    [
        pytest.param(
            False, False, "optim_grads_params", ["attn"], id="attn-optim_grads_params-attention-gpt"
        ),
        pytest.param(
            False, False, "optim_grads_params", ["mlp"], id="mlp-optim_grads_params-attention-gpt"
        ),
        pytest.param(
            False,
            False,
            "optim_grads_params",
            ["moe_router"],
            id="router-optim_grads_params-attention-gpt",
        ),
        pytest.param(
            False,
            False,
            "optim_grads_params",
            ["attn", "mlp", "moe_router"],
            id="attn-mlp-router-optim_grads_params-attention-gpt",
        ),
        pytest.param(
            False,
            False,
            "optim_grads_params",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads_params-attention-gpt",
        ),
        pytest.param(
            False,
            False,
            "optim",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim-attention-gpt",
        ),
        pytest.param(
            False,
            False,
            "optim_grads",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads-attention-gpt",
        ),
        pytest.param(
            False,
            True,
            "optim_grads_params",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads_params-mla-gpt",
        ),
        pytest.param(
            True,
            False,
            "optim_grads_params",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads_params-attention-hybrid",
        ),
        pytest.param(
            True,
            False,
            "optim",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim-attention-hybrid",
        ),
        pytest.param(
            True,
            False,
            "optim_grads",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads-attention-hybrid",
        ),
        pytest.param(
            True,
            True,
            "optim_grads_params",
            ["attn", "mlp", "moe_router", "moe_preprocess"],
            id="attn-mlp-router-preprocess-optim_grads_params-mla-hybrid",
        ),
    ],
)
def test_partial_cuda_graph_moe_matches_eager(
    process_groups, mla, strategy, hybrid, capture_modules, monkeypatch
):
    """Compare actual language-model loss and every parameter update across capture."""
    config = _moe_config(mla, hybrid)

    def build(graphs):
        return _build_moe_model(config, process_groups, hybrid, capture_modules, strategy, graphs)

    torch.manual_seed(4321 + torch.distributed.get_rank())
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(8)]
    positions = torch.arange(16, device="cuda").unsqueeze(0)
    attention_mask = ~torch.ones((1, 1, 16, 16), device="cuda", dtype=torch.bool).tril()

    def run(graphs):
        model, optimizer = build(graphs)
        helper = (
            TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
            if graphs
            else None
        )
        losses = []
        updates = []
        gradients = []
        moments = []
        captured_pointers = {}
        try:
            for step, batch in enumerate(batches):
                if helper is not None and step == 3:
                    if not mla and strategy == "optim_grads_params":
                        layers = helper.flattened_callables
                        # Unsupported compositions must fail before global capture starts.
                        with monkeypatch.context() as patch:
                            patch.setattr(model.config, "context_parallel_size", 2)
                            with pytest.raises(ValueError, match="TP=PP=CP=1"):
                                helper.create_cudagraphs()
                        assert not cuda_graphs.is_graph_capturing()

                        # Preserve user hooks when TE rejects them, then retry normally.
                        user_hook = layers[0].register_forward_pre_hook(lambda module, inputs: None)
                        hook_names = (
                            "_forward_pre_hooks",
                            "_forward_hooks",
                            "_backward_pre_hooks",
                            "_backward_hooks",
                        )
                        hooks = [
                            {name: dict(getattr(layer, name)) for name in hook_names}
                            for layer in layers
                        ]
                        with pytest.raises(RuntimeError, match="Modules must not have hooks"):
                            helper.create_cudagraphs()
                        assert not cuda_graphs.is_graph_capturing()
                        for layer, original_hooks in zip(layers, hooks):
                            for name in hook_names:
                                assert getattr(layer, name) == original_hooks[name]
                            assert not layer._fsdp_cuda_graph_capturing
                            assert all(
                                not group._retain_unsharded_storage
                                for group in layer.parameter_groups
                            )
                        user_hook.remove()
                    snapshot = [p.to_local().clone() for p in model.parameters()]
                    helper.create_cudagraphs()
                    assert helper.graphs_created()
                    for parameter, original in zip(model.parameters(), snapshot):
                        torch.testing.assert_close(parameter.to_local(), original, rtol=0, atol=0)
                    captured_pointers = {
                        id(group): group._unsharded_model_weight.local_buffer.data_ptr()
                        for layer in helper.flattened_callables
                        for group in layer.parameter_groups
                    }
                if helper is not None and step == 6:
                    helper.delete_cuda_graphs()
                    for layer in model.module.decoder.layers:
                        for group in layer.parameter_groups:
                            assert not group._retain_unsharded_storage
                            if strategy == "optim_grads_params":
                                assert not group._unsharded_model_weight.is_allocated
                    captured_pointers.clear()
                model.zero_grad_buffer()
                optimizer.zero_grad()
                iterator = iter(batch)

                def forward():
                    tokens = next(iterator)
                    ids, labels = tokens[:, :-1], tokens[:, 1:]
                    model.eval()
                    with torch.no_grad():
                        expected = (
                            model(ids, positions, attention_mask, labels=labels).float().mean()
                        )
                    model.train()
                    loss = model(ids, positions, attention_mask, labels=labels).float().mean()
                    torch.testing.assert_close(
                        loss.detach(),
                        expected,
                        rtol=1e-5,
                        atol=1e-6,
                        msg=lambda msg: f'graphs={graphs}, step={step}: {msg}',
                    )
                    return loss

                with (
                    torch.profiler.profile() if graphs and step == 4 else nullcontext()
                ) as profiler:
                    values = run_forward_backward_on_microbatches(
                        model, forward, num_microbatches=2, loss_scale=0.5
                    )
                if profiler is not None:
                    # One launch per captured unit and microbatch, in each direction.
                    assert sum(event.name == "cudaGraphLaunch" for event in profiler.events()) == (
                        2 * 2 * len(helper.flattened_callables)
                    )
                gradients.append([p.grad.to_local().detach().clone() for p in model.parameters()])
                for layer in model.module.decoder.layers:
                    for group in layer.parameter_groups:
                        if id(group) in captured_pointers:
                            assert (
                                group._unsharded_model_weight.local_buffer.data_ptr()
                                == captured_pointers[id(group)]
                            )
                success, _, _ = optimizer.step()
                assert success
                losses.append(torch.stack(values).mean())
                updates.append([p.to_local().detach().clone() for p in model.parameters()])
                moments.append(
                    [
                        state[key].to_local().detach().clone()
                        for state in optimizer.optimizer.state.values()
                        for key in ("exp_avg", "exp_avg_sq")
                    ]
                )
            return torch.stack(losses), gradients, updates, moments
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()

    expected = run(False)
    actual = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _train_moe_step(model, optimizer, batch):
    """Return actual LM loss, gradients, updated weights, and Adam moments."""
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
    gradients = [p.grad.to_local().detach().cpu() for p in model.parameters()]
    success, _, _ = optimizer.step()
    assert success
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


@pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
@pytest.mark.parametrize(
    "separate_communication_streams", [False, True], ids=["shared", "separate"]
)
def test_partial_capture_preserves_communication_streams(
    process_groups, hybrid, separate_communication_streams
):
    """Preserve the existing FSDP stream policy and training state across capture.

    GPU timing depends on workload and hardware. Check correctness with both
    communication policies here; establish actual overlap with a training trace.
    """
    config = replace(_moe_config(hybrid=hybrid), attention_dropout=0.1, hidden_dropout=0.1)
    torch.manual_seed(6791 + torch.distributed.get_rank())
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(6)]

    def run(graphs):
        context = FsdpContext(
            runtime_policy=RuntimePolicy(
                caller_managed_grad_sync=True,
                overlap_allgather_reduction=separate_communication_streams,
            )
        )
        streams = (
            context.allgather_stream,
            context.inner_reduction_stream,
            context.outer_reduction_stream,
        )
        assert all(stream != torch.cuda.current_stream() for stream in streams)
        assert (streams[0] != streams[1]) == separate_communication_streams
        with fully_shard_context(context):
            model, optimizer = _build_moe_model(
                config,
                process_groups,
                hybrid,
                ["attn", "mlp", "moe_router", "moe_preprocess"],
                "optim_grads_params",
                graphs,
            )
        helper = (
            TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
            if graphs
            else None
        )
        history = []
        try:
            for step, batch in enumerate(batches):
                if helper is not None and step == 3:
                    helper.create_cudagraphs()
                if helper is not None and step == 5:
                    helper.delete_cuda_graphs()
                history.append(_train_moe_step(model, optimizer, batch))
                assert streams == (
                    context.allgather_stream,
                    context.inner_reduction_stream,
                    context.outer_reduction_stream,
                )
            return history
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()
            destroy_moe_metrics_tracker()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)


def _rng_contents():
    """Read state contents, rather than live graph-safe generator handles."""
    tracker = get_cuda_rng_tracker().get_states()
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        {name: state.get_state().clone() for name, state in tracker.items()},
    )


@pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
@pytest.mark.parametrize("failure", ["inputs", "finish"])
def test_partial_capture_failure_retry(process_groups, hybrid, failure, monkeypatch):
    """Failed real MoE capture must restore its lifecycle before a clean retry."""
    config = replace(_moe_config(hybrid=hybrid), attention_dropout=0.1, hidden_dropout=0.1)
    scopes = ["attn", "mlp", "moe_router", "moe_preprocess"]
    torch.manual_seed(92)
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(4)]

    def run(graphs):
        model, optimizer = _build_moe_model(
            config, process_groups, hybrid, scopes, "optim_grads_params", graphs
        )
        helper = (
            TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
            if graphs
            else None
        )
        try:
            for batch in batches[:3]:
                _train_moe_step(model, optimizer, batch)
            if helper is not None:
                before_rng = _rng_contents()
                layers = helper.flattened_callables
                original_calls = {type(layer): type(layer).__call__ for layer in layers}
                hook_names = (
                    "_forward_pre_hooks",
                    "_forward_hooks",
                    "_backward_pre_hooks",
                    "_backward_hooks",
                )
                hooks = [
                    (module, {name: dict(getattr(module, name)) for name in hook_names})
                    for module in model.modules()
                ]
                gc_enabled = gc.isenabled()

                def fail(*args, **kwargs):
                    raise RuntimeError("controlled capture failure")

                with monkeypatch.context() as patch:
                    if failure == "inputs":
                        patch.setattr(helper, "_get_cuda_graph_input_data", fail)
                    elif failure == "finish":
                        patch.setattr(helper, "_finish_capturing", fail)
                    with pytest.raises(RuntimeError, match="controlled capture failure"):
                        helper.create_cudagraphs()
                assert not cuda_graphs.is_graph_capturing()
                for module_type, original_call in original_calls.items():
                    assert module_type.__call__ is original_call
                assert not te_graph.is_graph_capturing()
                assert not helper.graphs_created()
                assert not helper._capture_finished
                torch.testing.assert_close(_rng_contents(), before_rng, rtol=0, atol=0)
                assert gc.isenabled() == gc_enabled
                for module, original_hooks in hooks:
                    for name in hook_names:
                        assert getattr(module, name) == original_hooks[name], (module, name)
                for layer in layers:
                    assert not layer._fsdp_cuda_graph_capturing
                    assert not layer.cuda_graphs
                    assert all(
                        not group._retain_unsharded_storage for group in layer.parameter_groups
                    )
                helper.create_cudagraphs()
            return _train_moe_step(model, optimizer, batches[-1])
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()
            # Keep a failing test from poisoning unrelated cases in this process.
            cuda_graphs._set_capture_end()

    expected = run(False)
    actual = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
@pytest.mark.parametrize("attention_dropout,hidden_dropout", [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_partial_capture_dropout_rng(process_groups, hybrid, attention_dropout, hidden_dropout):
    """Compare stochastic training and capture RNG contents to the same eager path."""
    config = replace(
        _moe_config(hybrid=hybrid),
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
    )
    scopes = ["attn", "mlp", "moe_router", "moe_preprocess"]
    torch.manual_seed(93)
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(6)]

    def run(graphs):
        model, optimizer = _build_moe_model(
            config, process_groups, hybrid, scopes, "optim_grads_params", graphs
        )
        helper = (
            TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
            if graphs
            else None
        )
        history = []
        try:
            for index, batch in enumerate(batches):
                if helper is not None and index == 3:
                    before = _rng_contents()
                    helper.create_cudagraphs()
                    torch.testing.assert_close(_rng_contents(), before, rtol=0, atol=0)
                if index == 4:
                    before = _rng_contents()
                    model.eval()
                    positions = torch.arange(16, device="cuda").unsqueeze(0)
                    mask = ~torch.ones((1, 1, 16, 16), device="cuda", dtype=torch.bool).tril()
                    with torch.no_grad():
                        model(batch[0, :, :-1], positions, mask, labels=batch[0, :, 1:])
                    torch.testing.assert_close(_rng_contents(), before, rtol=0, atol=0)
                before = _rng_contents()
                history.append(_train_moe_step(model, optimizer, batch))
                after = _rng_contents()
                assert not torch.equal(after[1], before[1]) or any(
                    not torch.equal(after[2][name], before[2][name]) for name in before[2]
                )
            return history
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)


@pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
def test_partial_capture_delete_recapture(process_groups, hybrid, tmp_path_dist_ckpt):
    """Reuse MoE/Adam across graph lifetimes, native checkpoint restore and eager intervals."""
    config = _moe_config(hybrid=hybrid)
    scopes = ["attn", "mlp", "moe_router", "moe_preprocess"]
    torch.manual_seed(94)
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(12)]

    def run(graphs):
        model, optimizer = _build_moe_model(
            config, process_groups, hybrid, scopes, "optim_grads_params", graphs
        )
        helper = None
        history = []
        released_memory = []
        try:
            for index, batch in enumerate(batches):
                if graphs and index in (3, 6, 9):
                    helper = TECudaGraphHelper(
                        [model], model.config, 16, 1, [optimizer], process_groups
                    )
                    helper.create_cudagraphs()
                    pointers = {
                        id(group): group._unsharded_model_weight.local_buffer.data_ptr()
                        for layer in helper.flattened_callables
                        for group in layer.parameter_groups
                    }
                history.append(_train_moe_step(model, optimizer, batch))
                if helper is not None and helper.graphs_created():
                    for layer in helper.flattened_callables:
                        for group in layer.parameter_groups:
                            assert (
                                group._unsharded_model_weight.local_buffer.data_ptr()
                                == pointers[id(group)]
                            )
                    if index in (4, 7, 10):
                        helper.delete_cuda_graphs()
                        for layer in helper.flattened_callables:
                            assert not layer.cuda_graphs
                            for group in layer.parameter_groups:
                                assert not group._retain_unsharded_storage
                                assert not group._unsharded_model_weight.is_allocated
                        helper = None
                        if index == 4:
                            with TempNamedDir(
                                tmp_path_dist_ckpt / "partial_graph_restart", sync=True
                            ) as path:
                                state = {
                                    "model": model.state_dict_for_save_checkpoint(),
                                    "optimizer": optimizer.sharded_state_dict({}),
                                }
                                state = preprocess_fsdp_dtensor_state_dict(
                                    model.config, state, model
                                )
                                dcp.save(state, checkpoint_id=path)
                                with torch.no_grad():
                                    for parameter in model.parameters():
                                        parameter.zero_()
                                    for values in optimizer.optimizer.state.values():
                                        for value in values.values():
                                            if torch.is_tensor(value):
                                                value.zero_()
                                    for group in optimizer.optimizer.param_groups:
                                        for parameter in group["params"]:
                                            parameter.zero_()
                                        if "step" in group:
                                            group["step"] = 0
                                raw_model_state = model.state_dict_for_save_checkpoint()
                                state = {
                                    "model": raw_model_state.copy(),
                                    "optimizer": optimizer.sharded_state_dict({}, is_loading=True),
                                }
                                state = preprocess_fsdp_dtensor_state_dict(
                                    model.config, state, model
                                )
                                dcp.load(state, checkpoint_id=path)
                                model.load_state_dict(raw_model_state)
                                optimizer.load_state_dict(state["optimizer"])
                        gc.collect()
                        torch.cuda.synchronize()
                        # Match test_memory: isolate FSDP storage from per-stream library caches.
                        torch._C._cuda_clearCublasWorkspaces()
                        released_memory.append(torch.cuda.memory_allocated())
            if graphs:
                assert len(released_memory) == 3
                assert released_memory[-1] <= released_memory[-2], released_memory
            return history
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)
