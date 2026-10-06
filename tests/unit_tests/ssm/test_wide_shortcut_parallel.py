# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward/backward parity for wide ShortcutMoE scheduling, replay, and parallelism."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import wide_residual_hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.test_utilities import Utils

_HIDDEN = 256
_SEQUENCE = 64
_BATCH = 2
_STREAMS = 3
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def _weight_group(parameter, pg_collection):
    return (
        pg_collection.gtp_remat
        if getattr(parameter, "allreduce", True)
        else pg_collection.expt_gtp_remat
    )


def _is_gtp_parameter(parameter):
    if not HAVE_GTP:
        return False
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam

    return isinstance(parameter, GTPShardedParam)


def _full_tensors(module, tensors, pg_collection):
    """Reconstruct GTP shards, but keep each rank's EP-local expert identity."""

    result = {}
    for name, parameter in module.named_parameters():
        value = tensors[name].detach().contiguous()
        if _is_gtp_parameter(parameter):
            group = _weight_group(parameter, pg_collection)
            shards = [torch.empty_like(value) for _ in range(group.size())]
            dist.all_gather(shards, value, group=group)
            value = torch.cat(shards, dim=0)
            if parameter.pad_length:
                value = value[: -parameter.pad_length]
        result[name] = value.float().cpu().clone()
    return result


def _build_stack(
    pg_collection,
    *,
    compute_symbol,
    fp32_residual,
    overlap,
    replay,
    tp_size,
    cp_size,
    gtp_size,
    wide_input=False,
):
    config = TransformerConfig(
        hidden_size=_HIDDEN,
        num_layers=2,
        num_attention_heads=8,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp32_residual_connection=fp32_residual,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=tp_size > 1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        tensor_parallel_num_weight_shards=tp_size * gtp_size,
        expert_tensor_parallel_num_weight_shards=gtp_size,
        num_moe_experts=2,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_shortcut_connection=True,
        moe_shortcut_parallel=overlap,
        moe_shortcut_post_norm=True,
        moe_shared_expert_intermediate_size=256,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        recompute_granularity="selective" if replay else None,
        recompute_modules=["residual_stream"] if replay else None,
        residual_stream_recompute_num_layers=1 if replay else None,
        wide_residual=WideResidualConfig(num_streams=_STREAMS, learned_retention=True),
    )
    if gtp_size > 1:
        from megatron.core.tensor_parallel.gtp_api import configure_gtp_remat_from_recipe

        configure_gtp_remat_from_recipe()
    model = (
        HybridStack(
            config=config,
            submodules=wide_residual_hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers(compute_symbol + Symbols.MOE, config),
            pre_process=not wide_input,
            post_layer_norm=False,
            pg_collection=pg_collection,
        )
        .cuda()
        .to(dtype=config.params_dtype)
    )
    assert model._residual_stream_atomic_layer_pairs == ((0, 1),)
    if cp_size > 1:
        assert model._cp_layout_manager is not None
    return model


def _load_weights(module, saved, pg_collection):
    assert saved.keys() == dict(module.named_parameters()).keys()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = saved[name]
            if _is_gtp_parameter(parameter):
                from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                    gtp_remat_slice_rows,
                )

                group = _weight_group(parameter, pg_collection)
                value = gtp_remat_slice_rows(value, group)
            assert value.shape == parameter.shape, name
            parameter.copy_(value)


def _synchronize_weights(module, pg_collection):
    """Replicate weights within their owner groups, without equating different experts."""

    for parameter in module.parameters():
        group = (
            pg_collection.dp_cp_gtp_remat
            if getattr(parameter, "allreduce", True)
            else pg_collection.expt_dp_gtp_remat
        )
        dist.broadcast(parameter.data, src=dist.get_process_group_ranks(group)[0], group=group)


def _local_batch(pg_collection, batch_index, *, sequence_parallel, wide_input):
    """Different DP/GTP samples; identical samples across a sample's TP/CP peers."""

    generator = torch.Generator(device="cuda").manual_seed(
        456 + batch_index * 100 + pg_collection.dp_gtp_remat.rank()
    )
    hidden = torch.randn(
        _SEQUENCE,
        _BATCH,
        _HIDDEN * (_STREAMS if wide_input else 1),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    target = torch.randn(
        _SEQUENCE, _BATCH, _HIDDEN, dtype=torch.float32, device="cuda", generator=generator
    )
    hidden = hidden.chunk(pg_collection.cp.size(), dim=0)[pg_collection.cp.rank()]
    target = target.chunk(pg_collection.cp.size(), dim=0)[pg_collection.cp.rank()]
    if sequence_parallel:
        hidden = hidden.chunk(pg_collection.tp.size(), dim=0)[pg_collection.tp.rank()]
        target = target.chunk(pg_collection.tp.size(), dim=0)[pg_collection.tp.rank()]
    return hidden.contiguous().requires_grad_(True), target.contiguous()


def _run_forward_backward(module, pg_collection):
    """Compare two distinct batches at fixed weights through real DDP finalization."""

    module.train()
    ddp = DistributedDataParallel(
        module.config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
        ),
        module,
        pg_collection=pg_collection,
    )
    if pg_collection.gtp_remat.size() > 1 or pg_collection.expt_gtp_remat.size() > 1:
        from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains

        classify_gtp_remat_chains(ddp)
    records = []
    write_dtypes = []
    read_count = 0

    def record_write(_module, _args, kwargs, output):
        if kwargs.get("operation") == "write":
            write_dtypes.append(output.dtype)

    def record_shortcut_read(_module, _args, _output):
        nonlocal read_count
        read_count += 1

    pair = module.layers[0]
    write_handle = pair.moe_layer.residual_connection_mlp.register_forward_hook(
        record_write, with_kwargs=True
    )
    read_handle = pair.shortcut_residual_read.register_forward_hook(record_shortcut_read)
    try:
        for batch_index in range(2):
            reads_before = read_count
            ddp.zero_grad_buffer()
            hidden, target = _local_batch(
                pg_collection,
                batch_index,
                sequence_parallel=module.config.sequence_parallel,
                wide_input=not module.pre_process,
            )
            if not module.pre_process:
                module.set_input_tensor(hidden)
            model_parallel_cuda_manual_seed(789 + batch_index)
            output = ddp(hidden, attention_mask=None)
            loss = (output.float() - target).square().mean()
            loss.backward()
            finalize_model_grads([ddp], pg_collection=pg_collection)
            torch.cuda.synchronize()
            replay = module.config.recompute_granularity == "selective"
            assert read_count - reads_before == (2 if replay else 1)
            gradients = {}
            for name, parameter in module.named_parameters():
                assert parameter.main_grad is not None, name
                assert torch.isfinite(parameter.main_grad).all(), name
                gradients[name] = parameter.main_grad
                if "residual_connection" in name or "shortcut_residual_read" in name:
                    assert not _is_gtp_parameter(parameter), name
                    active = parameter.main_grad.flatten()[:_STREAMS]
                    assert torch.count_nonzero(active) > 0, name
                if ".mlp.experts." in name:
                    assert torch.count_nonzero(parameter.main_grad) > 0, name

            records.append(
                (
                    output.detach().float().cpu(),
                    hidden.grad.detach().float().cpu(),
                    _full_tensors(module, gradients, pg_collection),
                )
            )
    finally:
        write_handle.remove()
        read_handle.remove()
    assert write_dtypes
    expected_dtype = torch.float32 if module.config.fp32_residual_connection else torch.bfloat16
    assert all(dtype == expected_dtype for dtype in write_dtypes)
    return records


def _assert_parity(actual, expected, *, gtp=False):
    for batch_index, (actual_batch, expected_batch) in enumerate(
        zip(actual, expected, strict=True)
    ):
        for kind, actual_value, expected_value in zip(
            ("output", "input gradient", "main_grad"), actual_batch, expected_batch, strict=True
        ):
            if isinstance(actual_value, dict):
                assert actual_value.keys() == expected_value.keys()
                pairs = actual_value.items()
            else:
                pairs = [(kind, actual_value)]
                expected_value = {kind: expected_value}
            for name, value in pairs:
                if gtp:
                    # BF16 GTP reductions change cancellation near zero. Bound the error
                    # relative to each tensor's peak AND norm, not each individual element.
                    reference = expected_value[name]
                    atol, rtol = 0.02 * reference.abs().max().item() + 1e-10, 0.0
                    reference_norm = reference.norm().item()
                    error_norm = (value - reference).norm().item()
                    assert error_norm <= 0.02 * reference_norm + 1e-10, (
                        f"batch {batch_index}, {kind}, {name}: error {error_norm} exceeds "
                        f"2% of reference norm {reference_norm}"
                    )
                else:
                    atol, rtol = 1e-6, 1e-5
                torch.testing.assert_close(
                    value,
                    expected_value[name],
                    atol=atol,
                    rtol=rtol,
                    msg=lambda detail, name=name: f"batch {batch_index}, {kind}, {name}: {detail}",
                )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "compute_symbol", [Symbols.ATTENTION, Symbols.MAMBA], ids=["attention", "mamba"]
)
@pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16-residual", "fp32-residual"])
@pytest.mark.parametrize(
    ("tp_size", "cp_size"), [(1, 1), (2, 1), (1, 2)], ids=["ep2", "tp2-ep2", "cp2-ep2"]
)
def test_wide_shortcut_forward_backward_matches_eager_serial(
    compute_symbol, fp32_residual, tp_size, cp_size
):
    if _WORLD_SIZE < 2 * tp_size * cp_size or _WORLD_SIZE % (2 * tp_size * cp_size):
        pytest.skip("Requires WORLD_SIZE divisible by 2 * TP * CP")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
    )
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        saved = None
        reference = None
        for overlap, replay in ((False, False), (True, False), (False, True), (True, True)):
            model_parallel_cuda_manual_seed(123)
            torch.manual_seed(123 + pg_collection.ep.rank())
            module = _build_stack(
                pg_collection,
                compute_symbol=compute_symbol,
                fp32_residual=fp32_residual,
                overlap=overlap,
                replay=replay,
                tp_size=tp_size,
                cp_size=cp_size,
                gtp_size=1,
            )
            if saved is None:
                _synchronize_weights(module, pg_collection)
                saved = _full_tensors(module, dict(module.named_parameters()), pg_collection)
            _load_weights(module, saved, pg_collection)
            actual = _run_forward_backward(module, pg_collection)
            if reference is None:
                reference = actual
            else:
                _assert_parity(actual, reference)
            del module
    finally:
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()


def _run_wide_shortcut_gtp_parity(compute_symbol):
    """Shared worker collected by the four-GPU GTP test bucket."""

    if _WORLD_SIZE != 4:
        pytest.skip("Run with torchrun --nproc-per-node=4")
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import reset_gtp_state

    saved = reference = gtp_reference = expert_rank = None
    try:
        for gtp_size, overlap, replay in (
            (1, False, False),
            (2, False, False),
            (2, True, False),
            (2, False, True),
            (2, True, True),
        ):
            reset_gtp_state()
            Utils.initialize_model_parallel(
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=1,
                gtp_remat_size=gtp_size,
                expert_gtp_remat_size=gtp_size,
            )
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            if expert_rank is None:
                expert_rank = pg_collection.ep.rank()
            assert pg_collection.ep.rank() == expert_rank
            model_parallel_cuda_manual_seed(123)
            torch.manual_seed(123 + pg_collection.ep.rank())
            module = _build_stack(
                pg_collection,
                compute_symbol=compute_symbol,
                fp32_residual=True,
                overlap=overlap,
                replay=replay,
                tp_size=1,
                cp_size=1,
                gtp_size=gtp_size,
                # Identical initial streams make pre-norm read gradients nearly zero.
                # Independent wide inputs exercise well-conditioned controller gradients.
                wide_input=True,
            )
            if saved is None:
                _synchronize_weights(module, pg_collection)
                saved = _full_tensors(module, dict(module.named_parameters()), pg_collection)
            _load_weights(module, saved, pg_collection)
            if gtp_size > 1:
                sharded = [
                    parameter for parameter in module.parameters() if _is_gtp_parameter(parameter)
                ]
                assert any(getattr(parameter, "allreduce", True) for parameter in sharded)
                assert any(not getattr(parameter, "allreduce", True) for parameter in sharded)
            actual = _run_forward_backward(module, pg_collection)
            if reference is None:
                reference = actual
            else:
                _assert_parity(actual, reference, gtp=True)
                if gtp_reference is None:
                    gtp_reference = actual
                else:
                    _assert_parity(actual, gtp_reference)
            del module
            torch.cuda.synchronize()
            Utils.destroy_model_parallel()
    finally:
        reset_gtp_state()
        Utils.destroy_model_parallel()
