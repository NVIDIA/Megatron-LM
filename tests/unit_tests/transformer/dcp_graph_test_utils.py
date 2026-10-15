# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared inputs and checks for DCP graph numerical regressions."""

from contextlib import contextmanager

import pytest
import torch

from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.cuda_graphs import (
    set_current_microbatch,
    te_supports_graph_memory_slots,
)
from tests.unit_tests.test_utilities import Utils

requires_te_graph_memory_slots = pytest.mark.skipif(
    not te_supports_graph_memory_slots(),
    reason="dynamic-CP graph banks need NVIDIA/TransformerEngine#3353 (_graph_memory_slots)",
)


@contextmanager
def dcp_test_context(monkeypatch, *, num_microbatches=1, **parallel_sizes):
    """Initialize native DCP groups and deterministic fused attention for one test."""
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        context_parallel_size=1, dynamic_context_parallel=True, **parallel_sizes
    )
    try:
        for name, value in (
            ("NVTE_FUSED_ATTN", "1"),
            ("NVTE_FLASH_ATTN", "0"),
            ("NVTE_UNFUSED_ATTN", "0"),
            ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
        ):
            monkeypatch.setenv(name, value)
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=num_microbatches * groups.dp.size(),
            micro_batch_size=1,
            data_parallel_size=groups.dp.size(),
        )
        yield groups
    finally:
        torch.cuda.synchronize()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


def dcp_batch(config, groups, cp_size, local_lengths, seed, **packing_options):
    """Make a deterministic packed batch through the production CP slicer."""
    lengths = [length * cp_size for length in local_lengths]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, 128, (sum(lengths) + 1,), generator=generator).cuda()
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    raw = dict(
        tokens=tokens[:-1].contiguous(),
        labels=tokens[1:].contiguous(),
        loss_mask=torch.ones(sum(lengths), dtype=torch.float32, device="cuda"),
        position_ids=torch.cat([torch.arange(length) for length in lengths]).cuda(),
        cu_seqlens=cu,
        cu_seqlens_padded=cu.clone(),
        max_seqlen=max(lengths),
        local_cp_size=cp_size,
    )
    values = get_batch_on_this_rank_for_sequence_packing(
        iter([raw]),
        dynamic_cp=True,
        mtp_on_this_rank=bool(config.mtp_num_layers),
        pg_collection=groups,
        config=config,
        **packing_options,
    )
    return dict(
        zip(
            (
                "input_ids",
                "labels",
                "loss_mask",
                "attention_mask",
                "position_ids",
                "packed_seq_params",
                "padding_mask",
            ),
            values,
        )
    )


def dcp_train_step(model, batch, microbatch, *, cpu=False):
    """Snapshot losses and leaf-owned gradients before another graph can overwrite them."""
    set_current_microbatch(model, microbatch)
    for parameter in model.parameters():
        # AccumulateGrad must not adopt TE's recyclable static dgrad allocation.
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        else:
            parameter.grad.zero_()

    def snapshot(tensor):
        value = tensor.detach().float()
        return (value.cpu() if cpu else value).clone()

    loss = model(**batch)
    values = snapshot(loss)
    # A contiguous CP shard may be entirely padding. Normalize by the CP-wide
    # token count, so a valid global pack never produces a local 0/0 loss.
    num_tokens = batch["loss_mask"].sum()
    cp_group = batch["packed_seq_params"].cp_group
    if cp_group is not None and cp_group.size() > 1:
        torch.distributed.all_reduce(num_tokens, group=cp_group)
    assert num_tokens > 0, "the full CP pack must contain valid tokens"
    ((loss.float() * batch["loss_mask"]).sum() / num_tokens).backward()
    gradients = {
        name: snapshot(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return values, gradients


def track_graph_replays(monkeypatch):
    """Record real CUDA graph launches, including backward replay."""
    launches = []
    original = torch.cuda.CUDAGraph.replay

    def replay(graph, *args, **kwargs):
        launches.append(graph)
        return original(graph, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    return launches


def assert_gradients_close(actual, expected, microbatch):
    """Apply the same strict per-parameter norm bound in every DCP graph test."""
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        error = (actual[name] - reference).norm()
        assert error <= 2e-2 * reference.norm() + 1e-6, (name, microbatch, error.item())
