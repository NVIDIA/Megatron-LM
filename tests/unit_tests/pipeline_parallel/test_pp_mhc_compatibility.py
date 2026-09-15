# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""PP/VPP mHC parity against the same model and parameters at PP1.

Run with torchrun --nproc-per-node=8 -m pytest <this file>.
The schedules, P2P communication, DDP gradient finalization, TE layers, and
Hybrid MTP are real; only the GPT spec explicitly selects main's mHC layer.
"""

import gc
import re
import zlib
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    _get_pipeline_hidden_size,
    get_forward_backward_func,
    get_tensor_shapes,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from megatron.core.utils import get_batch_on_this_cp_rank
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("enabled,streams", [(False, 4), (True, 1), (True, 2), (True, 4)])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("variable", [False, True])
def test_mhc_pipeline_tensor_shapes(enabled, streams, pp_size, tp_size, cp_size, variable):
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        context_parallel_size=cp_size,
        variable_seq_lengths=variable,
        enable_mhc_connections=enabled,
        mhc_num_residual_streams=streams,
    )
    width = 64 * (streams if enabled and pp_size > 1 else 1)
    assert _get_pipeline_hidden_size(config) == width
    # decoder_seq_length takes precedence; SP/CP affect only the sequence axis.
    shape = get_tensor_shapes(
        seq_length=64,
        decoder_seq_length=32,
        micro_batch_size=2,
        config=config,
        tp_group=SimpleNamespace(size=lambda: tp_size),
        cp_group=SimpleNamespace(size=lambda: cp_size),
    )
    assert shape == ([()] if variable else [(32 // tp_size // cp_size, 2, width)])


def test_mhc_pipeline_rejects_ep_overlap():
    with pytest.raises(NotImplementedError, match="overlap_moe_expert_parallel_comm"):
        TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            enable_mhc_connections=True,
            num_moe_experts=2,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
        )


@pytest.mark.internal
@pytest.mark.parametrize("pp_size", [2, 4])
@pytest.mark.parametrize("variable", [False, True])
def test_mhc_batched_p2p_preserves_directions(pp_size, variable):
    """Activation and gradient messages must stay distinct when peers coincide."""
    if Utils.world_size % pp_size != 0:
        pytest.skip("Requires a world size divisible by PP")
    try:
        Utils.initialize_model_parallel(pipeline_model_parallel_size=pp_size)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=pp_size,
            pipeline_dtype=torch.float32,
            enable_mhc_connections=True,
            mhc_num_residual_streams=2,
            batch_p2p_comm=True,
            variable_seq_lengths=variable,
        )
        communicator = P2PCommunicator(groups.pp, config)
        shape = (16, 1, _get_pipeline_hidden_size(config))
        rank = groups.pp.rank()
        activation = torch.full(shape, float(rank + 1), device="cuda")
        backward_shape = (32, 1, shape[-1]) if variable else shape
        gradient = torch.full(backward_shape, float(rank + 101), device="cuda")
        received_activation, received_gradient = (
            communicator.send_forward_backward_recv_forward_backward(
                activation, gradient, recv_prev=True, recv_next=True, tensor_shape=shape
            )
        )
        torch.testing.assert_close(
            received_activation, torch.full_like(activation, (rank - 1) % pp_size + 1)
        )
        torch.testing.assert_close(
            received_gradient, torch.full_like(gradient, (rank + 1) % pp_size + 101)
        )
    finally:
        Utils.destroy_model_parallel()


def _canonical_name(model, name):
    """Map rank-local decoder layers onto their global PP1 parameter names."""
    match = re.match(r"decoder\.layers\.(\d+)\.(.*)", name)
    if match:
        layer = model.decoder.layers[int(match[1])]
        return f"decoder.layers.{layer.layer_number - 1}.{match[2]}"
    return name


def _initialize_parameters(model):
    # Initialization is independent of rank-local construction order and PP
    # partition. Replicated parameters are identical on every TP rank too.
    for name, param in model.named_parameters():
        canonical = _canonical_name(model, name)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if not getattr(param, "tensor_model_parallel", False):
            tp_rank = 0
        generator = torch.Generator().manual_seed(zlib.crc32(f"{canonical}:{tp_rank}".encode()))
        values = torch.randn(param.shape, generator=generator) * 0.02
        if "norm" in canonical and canonical.endswith("weight"):
            values += 1.0
        if canonical.endswith("hc_head_scale") or canonical.endswith("alpha"):
            values.fill_(0.01)
        with torch.no_grad():
            param.copy_(values)


def _make_config(pp_size, vp_size, *, tp_size, cp_size, recompute, variable, standalone, mtp):
    bf16 = cp_size > 1
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        kv_channels=16,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        pipeline_model_parallel_layout="E|tt|tt|L" if standalone and pp_size > 1 else None,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iterations=3,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["mhc"] if recompute else None,
        mhc_recompute_layer_num=2 if recompute else None,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused if bf16 else AttnBackend.unfused,
        gradient_accumulation_fusion=False,
        deallocate_pipeline_outputs=True,
        batch_p2p_comm=True,
        variable_seq_lengths=variable,
        mtp_num_layers=1 if mtp else None,
        mtp_loss_scaling_factor=0.1,
        normalization="RMSNorm",
    )
    config.finalize_model_grads_func = finalize_model_grads
    return config


def _make_models(config, kind, empty):
    models = []
    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size
    for chunk in range(vp_size or 1):
        vp_stage = chunk if vp_size is not None else None
        pre_process = parallel_state.is_pipeline_first_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
        post_process = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
        kwargs = dict(
            config=config,
            vocab_size=64,
            max_sequence_length=32,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        )
        if kind == "gpt":
            spec = get_gpt_layer_with_transformer_engine_spec()
            spec.module = HyperConnectionTransformerLayer
            spec.submodules.self_attention_hyper_connection = HyperConnectionModule
            spec.submodules.mlp_hyper_connection = HyperConnectionModule
            model = GPTModel(transformer_layer_spec=spec, **kwargs)
        else:
            pattern = "*-||*-|" if empty and pp_size > 1 else "*-*-"
            if config.mtp_num_layers:
                pattern += "/-"
            model = HybridModel(
                hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern=pattern, **kwargs
            )
        model = convert_module_to_dtype_except_fp32_marked(
            model.cuda(), config.params_dtype
        ).train()
        _initialize_parameters(model)
        models.append(model)
    return models


def _batches(variable, cp_group):
    for microbatch in range(4):
        length = (16 if microbatch % 2 == 0 else 32) if variable else 16
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        tokens = (positions + 3 * microbatch) % 64
        batch = dict(tokens=tokens, labels=(tokens + 1) % 64, position_ids=positions)
        yield get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=cp_group)


def _run_schedule(config, models):
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    wrapped = [
        DistributedDataParallel(
            config=config,
            # Match native BF16 training: accumulate and reduce main grads in
            # FP32, even when the parameter and pipeline activation are BF16.
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False
            ),
            module=model,
        )
        for model in models
    ]
    for chunk in wrapped:
        chunk.zero_grad_buffer()
    seen_shapes = []
    grad_seen = set()
    hooks = []
    for model in models:
        for name, param in model.named_parameters():
            key = _canonical_name(model, name)
            hooks.append(param.register_hook(lambda grad, key=key: grad_seen.add(key)))

    def forward_step(iterator, model):
        batch = next(iterator)
        output = model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            attention_mask=None,
        )
        if not model.module.post_process:
            assert output.shape[-1] == config.hidden_size * config.mhc_num_residual_streams
            seen_shapes.append(tuple(output.shape))

        def loss_func(losses):
            # Use the native local-sum/token-count contract. The legacy
            # two-value contract would add an extra CP factor to a local mean.
            loss_sum = losses.float().sum()
            num_tokens = torch.tensor(losses.numel(), dtype=torch.int, device=losses.device)
            return loss_sum, num_tokens, {"loss": loss_sum.detach().clone() / num_tokens}

        return output, loss_func

    iterators = [iter(_batches(config.variable_seq_lengths, pg_collection.cp)) for _ in models]
    schedule = get_forward_backward_func()
    losses = schedule(
        forward_step_func=forward_step,
        data_iterator=iterators if len(models) > 1 else iterators[0],
        model=wrapped if len(models) > 1 else wrapped[0],
        num_microbatches=4,
        seq_length=32 if config.variable_seq_lengths else 16,
        micro_batch_size=1,
        forward_only=False,
    )
    if any(not model.post_process for model in models):
        assert len(seen_shapes) == 4 * sum(not model.post_process for model in models)
    local_loss = torch.zeros(4, device="cuda")
    if losses:
        local_loss.copy_(torch.stack([item["loss"] for item in losses]))
    torch.distributed.broadcast(
        local_loss,
        src=parallel_state.get_pipeline_model_parallel_last_rank(),
        group=pg_collection.pp,
    )
    grads = {}
    for model in models:
        for name, param in model.named_parameters():
            assert _canonical_name(model, name) in grad_seen, name
            assert hasattr(param, "main_grad"), name
            assert param.main_grad.dtype == torch.float32, name
            assert torch.isfinite(param.main_grad).all(), name
            key = _canonical_name(model, name)
            assert key not in grads, key
            grads[key] = param.main_grad.detach().float().cpu().clone()
    for hook in hooks:
        hook.remove()
    all_names = [None] * pg_collection.pp.size()
    torch.distributed.all_gather_object(all_names, list(grads), group=pg_collection.pp)
    return local_loss.cpu(), grads, set().union(*map(set, all_names))


@pytest.mark.internal
@pytest.mark.parametrize("recompute", [False, True], ids=["eager", "recompute_mhc"])
@pytest.mark.parametrize(
    "kind,vp_size,tp_size,cp_size,variable,standalone,empty,mtp",
    [
        ("gpt", None, 1, 1, False, False, False, False),
        ("gpt", 2, 1, 1, False, False, False, False),
        ("gpt", 2, 1, 1, False, True, False, False),
        ("hybrid", 2, 1, 1, False, False, True, False),
        ("hybrid", None, 2, 1, False, False, False, True),
        ("gpt", None, 1, 2, True, False, False, False),
        ("gpt", 2, 1, 2, True, False, False, False),
    ],
    ids=[
        "pp2",
        "vpp2",
        "embedding_loss",
        "empty_hybrid",
        "tp_sp_mtp",
        "cp_variable",
        "vpp_cp_variable",
    ],
)
def test_mhc_schedule_matches_pp1(
    kind, vp_size, tp_size, cp_size, variable, standalone, empty, mtp, recompute
):
    if Utils.world_size % (2 * tp_size * cp_size) != 0:
        pytest.skip("Requires a world size divisible by 2 * TP * CP")
    kwargs = dict(
        tp_size=tp_size,
        cp_size=cp_size,
        recompute=recompute,
        variable=variable,
        standalone=standalone,
        mtp=mtp,
    )
    snapshots = []
    try:
        for pp_size, vp in [(1, None), (2, vp_size)]:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
                virtual_pipeline_model_parallel_size=vp,
                context_parallel_size=cp_size,
            )
            model_parallel_cuda_manual_seed(123)
            config = _make_config(pp_size, vp, **kwargs)
            models = _make_models(config, kind, empty)
            snapshots.append(_run_schedule(config, models))
            del models
            gc.collect()
            Utils.destroy_model_parallel()
    finally:
        Utils.destroy_model_parallel()
    (ref_loss, ref_grads, ref_names), (loss, grads, names) = snapshots
    assert names == ref_names
    tolerance = dict(rtol=1.6e-2, atol=1e-5) if cp_size > 1 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        loss,
        ref_loss,
        msg=lambda message: f"Microbatch losses {loss.tolist()} vs PP1 {ref_loss.tolist()}: {message}",
        **tolerance,
    )
    gradient_errors = []
    for name, grad in grads.items():
        assert name in ref_grads, name
        reference = ref_grads[name]
        try:
            torch.testing.assert_close(grad, reference, **tolerance)
        except AssertionError as error:
            difference = grad - reference
            gradient_errors.append(
                f"{name}: max_abs={difference.abs().max().item():.8g}, "
                f"reference_max={reference.abs().max().item():.8g}, "
                f"difference_l2={difference.norm().item():.8g}, "
                f"reference_l2={reference.norm().item():.8g}\n{error}"
            )
    assert not gradient_errors, "\n\n".join(gradient_errors)
