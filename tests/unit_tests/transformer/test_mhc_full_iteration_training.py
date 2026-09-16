# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real mHC training parity for the full-iteration graph runner.

The default runs PP1, independently of the mHC pipeline-support PR. After that
PR is present, the same tests exercise PP2/VPP2 without changing validation:

    MHC_CG_PP_SIZE=2 MHC_CG_VP_SIZE=2 torchrun --nproc-per-node=4 -m pytest \
        tests/unit_tests/transformer/test_mhc_full_iteration_training.py

Combined eight-GPU examples (use the same eager/graph comparison):

    MHC_CG_PP_SIZE=2 MHC_CG_VP_SIZE=2 MHC_CG_TP_SIZE=2 MHC_CG_EP_SIZE=2 \
        torchrun --nproc-per-node=8 -m pytest -k 'hybrid_moe_mtp and bf16' \
        tests/unit_tests/transformer/test_mhc_full_iteration_training.py
    MHC_CG_PP_SIZE=2 MHC_CG_CP_SIZE=2 MHC_CG_EP_SIZE=2 \
        torchrun --nproc-per-node=8 -m pytest -k 'hybrid_moe_mtp and bf16' \
        tests/unit_tests/transformer/test_mhc_full_iteration_training.py

TP > 1 enables SP; expert TP equals TP. EP applies to MoE cases. CP uses fixed
SBHD zigzag shards and BF16 fused attention; the FP32 cases remain CP1 checks.

Every case runs three eager warmups, the capture iteration, and four replay
iterations, using real DDP gradient synchronization and an optimizer step.
"""

import os
import sys
import traceback

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_all_rng_states, model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_model_config,
)
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

_SEQ_LENGTH = 16
_VOCAB_SIZE = 128
_NUM_MICROBATCHES = 4
_WARMUP_STEPS = 3
_TRAIN_STEPS = 8


def _snapshot_rng():
    """Copy state bytes, preserving the generator objects registered with graphs."""
    return (
        torch.cuda.get_rng_state().clone(),
        {name: generator.get_state().clone() for name, generator in get_all_rng_states().items()},
    )


def _restore_rng(snapshot):
    default_state, tracker_states = snapshot
    torch.cuda.set_rng_state(default_state)
    generators = get_all_rng_states()
    assert generators.keys() == tracker_states.keys()
    for name, state in tracker_states.items():
        generators[name].set_state(state)


def _build_models(
    model_kind,
    dtype,
    graph_impl,
    pp_size,
    vp_size,
    dropout,
    *,
    te_graph_modules=None,
    tp_size=1,
    cp_size=1,
    ep_size=1,
):
    is_hybrid = model_kind.startswith("hybrid")
    is_moe = "moe" in model_kind
    has_mtp = "mtp" in model_kind
    static_moe = is_moe and te_graph_modules is None
    num_chunks = vp_size or 1
    num_layers = 2 * pp_size * num_chunks
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        microbatch_group_size_per_vp_stage=pp_size,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        use_fused_mhc=True,
        cuda_graph_impl=graph_impl,
        cuda_graph_modules=te_graph_modules if graph_impl == "transformer_engine" else [],
        use_te_rng_tracker=graph_impl == "transformer_engine",
        cuda_graph_warmup_steps=_WARMUP_STEPS,
        attention_backend=AttnBackend.fused if cp_size > 1 else AttnBackend.unfused,
        cp_comm_type="p2p" if cp_size > 1 else None,
        attention_dropout=dropout,
        hidden_dropout=dropout,
        normalization="RMSNorm",
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        deallocate_pipeline_outputs=True,
        batch_p2p_comm=True,
        batch_p2p_sync=True,
        is_hybrid_model=is_hybrid,
        mtp_num_layers=1 if has_mtp else None,
        mtp_loss_scaling_factor=0.2,
        num_moe_experts=(2 if static_moe else 4) if is_moe else None,
        moe_ffn_hidden_size=128 if is_moe else None,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=1.0 if static_moe else None,
        moe_pad_expert_input_to_capacity=static_moe,
        moe_grouped_gemm=is_moe,
        finalize_model_grads_func=finalize_model_grads,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    assert (groups.tp.size(), groups.pp.size(), groups.cp.size(), groups.ep.size()) == (
        tp_size,
        pp_size,
        cp_size,
        ep_size,
    )
    assert groups.expt_tp.size() == tp_size
    models = []
    for chunk_index in range(num_chunks):
        vp_stage = chunk_index if vp_size is not None else None
        pre_process = groups.pp.rank() == 0 and chunk_index == 0
        post_process = groups.pp.rank() == pp_size - 1 and chunk_index == num_chunks - 1
        kwargs = dict(
            config=config,
            vocab_size=_VOCAB_SIZE,
            max_sequence_length=_SEQ_LENGTH,
            pre_process=pre_process,
            post_process=post_process,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
            pg_collection=groups,
            vp_stage=vp_stage,
        )
        if is_hybrid:
            segment = "*E" if is_moe else "*-"
            pattern = "|".join([segment] * (pp_size * num_chunks))
            if has_mtp:
                pattern += "/" + segment
            model = HybridModel(
                hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern=pattern, **kwargs
            )
        else:
            spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=config.num_moe_experts, moe_grouped_gemm=is_moe
            )
            spec.module = HyperConnectionTransformerLayer
            spec.submodules.self_attention_hyper_connection = HyperConnectionModule
            spec.submodules.mlp_hyper_connection = HyperConnectionModule
            model = GPTModel(transformer_layer_spec=spec, **kwargs)
        model = model.cuda()
        if dtype == torch.bfloat16:
            model = Float16Module(config, model)
        models.append(model)
    wrapped = _ddp_wrap(
        models,
        data_parallel_random_init=False,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            check_for_nan_in_grad=False,
            check_for_large_grads=False,
        ),
        overlap_param_gather_with_optimizer_step=False,
        pg_collection=groups,
    )
    config.no_sync_func = [model.no_sync for model in wrapped] if vp_size else wrapped[0].no_sync
    return wrapped, groups


def _named_parameters(models):
    return {
        f"chunk_{index}.{name}": parameter
        for index, model in enumerate(models)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _make_batch(iteration, microbatch, data_parallel_rank=0):
    # No RNG consumption from the data loader: dropout's state trajectory can be
    # compared independently from changed inputs on every training iteration.
    positions = torch.arange(_SEQ_LENGTH, device="cuda").unsqueeze(0)
    # Different DP replicas must contribute different gradients. CP peers use
    # the same DP rank, then split this global sequence in the forward step.
    tokens = (positions + 7 * iteration + 3 * microbatch + 17 * data_parallel_rank) % _VOCAB_SIZE
    return {
        "tokens": tokens,
        "labels": (tokens + 1 + iteration) % _VOCAB_SIZE,
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "position_ids": positions,
        "attention_mask": None,
    }


def _training_loss(losses, loss_mask):
    """Use the native sum/token-count contract for CP-correct gradient scaling."""
    losses = losses.reshape(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss_sum = (losses * loss_mask).sum()
    num_tokens = loss_mask.sum().detach().clone().to(torch.int)
    # The schedule normalizes the backward loss in place. Keep an independent
    # local mean for reporting, and avoid its legacy two-value CP multiplier.
    report = {"loss": loss_sum.detach().clone() / num_tokens.clamp(min=1)}
    return loss_sum, num_tokens, report


def _run_training(models, groups, use_graph, pp_size):
    parameters = _named_parameters(models)
    optimizer = torch.optim.SGD(list(parameters.values()), lr=0.02, momentum=0.1)
    forward_backward = get_forward_backward_func()
    p2p_communicator = (
        P2PCommunicator(pp_group=groups.pp, config=get_model_config(models[0]))
        if pp_size > 1
        else None
    )
    python_forward_calls = 0
    gradients_seen = set()
    gradient_hooks = [
        parameter.register_hook(lambda gradient, name=name: gradients_seen.add(name))
        for name, parameter in parameters.items()
    ]

    def forward_step(data_iterator, model):
        nonlocal python_forward_calls
        python_forward_calls += 1
        module = model.module
        if isinstance(module, Float16Module):
            module = module.module
        batch = get_batch_on_this_tp_rank(
            batch=next(data_iterator),
            has_cu_seqlens=False,
            is_hybrid_cp=False,
            create_attention_mask_in_dataloader=False,
            broadcast_src_rank=torch.distributed.get_global_rank(groups.tp, 0),
            broadcast_group=groups.tp,
            cp_size=groups.cp.size(),
            tp_rank=groups.tp.rank(),
            micro_batch_size=1,
            seq_length=_SEQ_LENGTH,
            mtp_on_this_rank=getattr(module, "mtp_process", False),
            pipeline_model_parallel_size=pp_size,
            is_pipeline_first_stage=module.pre_process,
            is_pipeline_last_stage=module.post_process,
        )
        if groups.cp.size() > 1:
            batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=groups.cp)
            if batch.get("tokens") is not None:
                assert batch["tokens"].shape[1] == _SEQ_LENGTH // groups.cp.size()
        output = model(
            batch["tokens"], batch["position_ids"], batch["attention_mask"], labels=batch["labels"]
        )

        def loss_func(losses):
            return _training_loss(losses, batch["loss_mask"])

        return output, loss_func

    wrapper = FullCudaGraphWrapper(forward_backward, cuda_graph_warmup_steps=_WARMUP_STEPS)
    run_step = wrapper if use_graph else forward_backward
    history = []
    captured_graph = None
    try:
        for iteration in range(_TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)
            for model in models:
                model.zero_grad_buffer()
            batches = [
                _make_batch(iteration, index, data_parallel_rank=groups.dp.rank())
                for index in range(_NUM_MICROBATCHES)
            ]
            data_iterators = [iter([batch.copy() for batch in batches]) for _ in models]
            losses = run_step(
                forward_step_func=forward_step,
                data_iterator=data_iterators,
                model=models,
                num_microbatches=_NUM_MICROBATCHES,
                seq_length=_SEQ_LENGTH,
                micro_batch_size=1,
                forward_only=False,
                p2p_communicator=p2p_communicator,
                pg_collection=groups,
            )
            # The actual schedule calls finalize_model_grads inside the captured
            # iteration; inspect gradients only after those collectives finish.
            torch.cuda.synchronize()
            assert (
                gradients_seen == parameters.keys()
            ), f"Parameters without a backward gradient: {parameters.keys() - gradients_seen}"
            gradients = {}
            for name, parameter in parameters.items():
                gradient = getattr(parameter, "main_grad", parameter.grad)
                assert gradient is not None, f"Missing synchronized gradient: {name}"
                assert torch.isfinite(gradient).all(), f"Non-finite gradient: {name}"
                gradients[name] = gradient.detach().cpu().clone()
                parameter.grad = gradient.to(dtype=parameter.dtype).clone()
            optimizer.step()
            history.append(
                {
                    "losses": [loss["loss"].detach().cpu().clone() for loss in losses],
                    "gradients": gradients,
                    "parameters": {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in parameters.items()
                    },
                }
            )
            if use_graph and iteration >= _WARMUP_STEPS:
                graph = FullCudaGraphWrapper.cuda_graph["training"]
                assert graph is not None
                if captured_graph is None:
                    captured_graph = graph
                assert graph is captured_graph
                assert python_forward_calls == (_WARMUP_STEPS + 1) * _NUM_MICROBATCHES * len(models)
        if use_graph:
            assert wrapper.curr_iter("training") == _TRAIN_STEPS
        return history, _snapshot_rng()
    except BaseException:
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        raise
    finally:
        for hook in gradient_hooks:
            hook.remove()
        wrapper.reset_cuda_graph()
        StaticBufferLoader.static_buffers = {"training": [], "validation": []}


def _assert_training_close(actual, expected, dtype):
    # Use PyTorch's BF16 defaults for BF16 compute, including FP32 DDP
    # accumulation buffers. Keep the original values instead of rounding those
    # buffers to BF16 before comparison.
    tolerance = (
        {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1.6e-2, "atol": 1e-5}
    )
    assert len(actual) == len(expected) == _TRAIN_STEPS
    for iteration, (actual_step, expected_step) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(actual_step["losses"], expected_step["losses"], **tolerance)
        for field in ("gradients", "parameters"):
            assert actual_step[field].keys() == expected_step[field].keys()
            for name, value in actual_step[field].items():
                # BF16 parameter gradients are held in FP32 DDP buffers. Compare
                # at the compute dtype's default tolerance, rather than treating
                # that storage choice as a claim of FP32 compute accuracy.
                reference = expected_step[field][name]
                torch.testing.assert_close(
                    value,
                    reference,
                    **tolerance,
                    msg=lambda message: f"iteration={iteration} {field}.{name}: {message}",
                )


def _initialize_test_parallelism(model_kind, dtype):
    """Read the integration topology and initialize its real process groups."""
    pp_size = int(os.environ.get("MHC_CG_PP_SIZE", "1"))
    vp_size = int(os.environ.get("MHC_CG_VP_SIZE", "0")) or None
    tp_size = int(os.environ.get("MHC_CG_TP_SIZE", "1"))
    cp_size = int(os.environ.get("MHC_CG_CP_SIZE", "1"))
    requested_ep_size = int(os.environ.get("MHC_CG_EP_SIZE", "1"))
    assert min(pp_size, tp_size, cp_size, requested_ep_size) > 0
    ep_size = requested_ep_size if "moe" in model_kind else 1
    dense_model_parallel_size = pp_size * tp_size * cp_size
    assert Utils.world_size % dense_model_parallel_size == 0
    assert Utils.world_size % (pp_size * tp_size * ep_size) == 0
    assert vp_size is None or (vp_size > 0 and pp_size > 1), "VPP requires PP > 1"
    assert _SEQ_LENGTH % (2 * cp_size * tp_size) == 0
    assert (
        cp_size == 1 or dtype == torch.bfloat16
    ), "CP integration uses BF16 fused attention; select the bf16 test cases."
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=tp_size,
    )
    dp_size = Utils.world_size // dense_model_parallel_size
    init_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        global_batch_size=_NUM_MICROBATCHES * dp_size,
        micro_batch_size=1,
        data_parallel_size=dp_size,
        decrease_batch_size_if_needed=False,
    )
    return dict(pp_size=pp_size, vp_size=vp_size, tp_size=tp_size, cp_size=cp_size, ep_size=ep_size)


def _check_training_parity(model_kind, dtype, dropout):
    parallel_sizes = _initialize_test_parallelism(model_kind, dtype)
    pp_size = parallel_sizes["pp_size"]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
        eager_models, groups = _build_models(
            model_kind, dtype, "none", dropout=dropout, **parallel_sizes
        )
        graph_models, _ = _build_models(
            model_kind, dtype, "full_iteration", dropout=dropout, **parallel_sizes
        )
        eager_parameters = _named_parameters(eager_models)
        graph_parameters = _named_parameters(graph_models)
        assert eager_parameters.keys() == graph_parameters.keys()
        with torch.no_grad():
            for name, parameter in graph_parameters.items():
                parameter.copy_(eager_parameters[name])
        initial_parameters = {
            name: value.detach().cpu().clone() for name, value in eager_parameters.items()
        }
        initial_rng = _snapshot_rng()
        expected, expected_rng = _run_training(eager_models, groups, False, pp_size)
        _restore_rng(initial_rng)
        actual, actual_rng = _run_training(graph_models, groups, True, pp_size)
        _assert_training_close(actual, expected, dtype)
        assert any(
            not torch.equal(value, actual[-1]["parameters"][name])
            for name, value in initial_parameters.items()
        ), "The test must execute optimizer updates"
        mhc_names = [name for name in graph_parameters if "hyper_connection" in name]
        assert mhc_names, "The selected model must contain real mHC parameters"
        assert any(actual[-1]["gradients"][name].count_nonzero() > 0 for name in mhc_names)
        if "mtp" in model_kind and groups.pp.rank() == pp_size - 1:
            mtp_names = [name for name in graph_parameters if ".mtp." in name]
            assert mtp_names
            assert any(actual[-1]["gradients"][name].count_nonzero() > 0 for name in mtp_names)
        if dropout:
            # This catches replaying one captured mask forever even if loss and
            # parameter updates happen to remain non-trivial.
            torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
            # Sequence-parallel dropout advances the tracked model-parallel
            # generator; the default generator may remain unchanged.
            default_rng_advanced = not torch.equal(initial_rng[0], actual_rng[0])
            tracked_rng_advanced = any(
                not torch.equal(state, actual_rng[1][name])
                for name, state in initial_rng[1].items()
            )
            assert default_rng_advanced or tracked_rng_advanced, "Dropout must advance CUDA RNG"
    except BaseException:
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        StaticBufferLoader.static_buffers = {"training": [], "validation": []}
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "model_kind",
    ["gpt_dense", "gpt_moe", "hybrid_dense", "hybrid_moe", "hybrid_mtp", "hybrid_moe_mtp"],
)
def test_mhc_full_iteration_training_parity(model_kind, dtype):
    """Changed batches, all synchronized gradients, and optimizer updates match eager."""
    _check_training_parity(model_kind, dtype, dropout=0.0)


@pytest.mark.launch_on_gb200
def test_mhc_full_iteration_training_with_dropout():
    """Nonzero dropout uses registered graph-safe RNG generators across replays."""
    _check_training_parity("gpt_dense", torch.float32, dropout=0.1)
