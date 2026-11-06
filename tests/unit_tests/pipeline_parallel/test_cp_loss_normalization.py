# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Context-parallel normalization of masked losses that report a token count.

With ``calculate_per_token_loss=False`` the CP ranks hold shards of the same sequences, so each
microbatch loss must be divided by its valid-token count summed over the CP group. Dividing by
the CP-local count optimizes the mean of the CP-local means instead of the global masked mean.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    forward_step_calc_loss,
    get_forward_backward_func,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from tests.unit_tests.test_utilities import Utils


class _TokenModel(MegatronModule):
    """Two token-wise projections; the first and last pipeline stages each hold one of them.

    Stages in between (including virtual ones) have no parameters and pass activations through.
    """

    def __init__(self, config, pre_process, post_process):
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = False
        self.input_tensor = None
        self.first_layer = torch.nn.Linear(4, 4).cuda() if pre_process else None
        self.last_layer = torch.nn.Linear(4, 4).cuda() if post_process else None
        with torch.no_grad():
            for index, layer in enumerate((self.first_layer, self.last_layer)):
                if layer is not None:
                    weight, bias = _initial_parameters(index)
                    layer.weight.copy_(weight)
                    layer.bias.copy_(bias)

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor[0]

    def forward(self, tokens):
        hidden = tokens if self.pre_process else self.input_tensor
        if self.first_layer is not None:
            hidden = torch.tanh(self.first_layer(hidden))
        if self.last_layer is not None:
            hidden = self.last_layer(hidden)
        return hidden


def _initial_parameters(index):
    weight = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    return (weight - 7 + index) / 31, torch.arange(4, device="cuda") / 17.0 - index / 13


def _batch(dp_rank, microbatch, mask_kind):
    """Return the full [s=8, b=1, h=4] tokens, targets and [s, b] loss mask of a microbatch."""
    tokens = torch.arange(32, dtype=torch.float32, device="cuda").reshape(8, 1, 4)
    tokens = torch.sin(tokens / 5 + dp_rank * 0.7 + microbatch * 0.4)
    targets = torch.cos(tokens * 1.3 + microbatch + dp_rank)
    masks = {
        "full": [1] * 8,
        "uneven": [1, 1, 1, 0, 0, 0, 0, 1],
        "one_target": [0, 0, 1, 0, 0, 0, 0, 0],
        "empty": [0] * 8,
        "varying": [1] * (1 + (dp_rank + microbatch * 3) % 8)
        + [0] * (7 - (dp_rank + microbatch * 3) % 8),
    }
    mask = torch.tensor(masks[mask_kind], device="cuda", dtype=torch.float32).view(8, 1)
    return tokens, targets, mask


def _cp_slice(value, cp_size, cp_rank):
    """Load-balanced CP split along the sequence dimension, as used by Megatron-LM."""
    chunks = value.chunk(2 * cp_size, dim=0)
    return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=0)


def _loss(targets, mask, output):
    per_token = (output - targets).square().mean(dim=-1)
    loss = (per_token * mask).sum()
    count = mask.sum().detach().to(torch.int)
    return loss, count, {"loss": torch.stack((loss.detach(), count.float()))}


def _reference(dp_size, num_microbatches, mask_kind, per_token):
    """Gradients of the same objective computed on unsharded sequences on one rank."""
    params = [value.detach().requires_grad_() for i in range(2) for value in _initial_parameters(i)]
    total_loss = torch.zeros((), device="cuda")
    total_count = torch.zeros((), device="cuda")
    for dp_rank in range(dp_size):
        for microbatch in range(num_microbatches):
            tokens, targets, mask = _batch(dp_rank, microbatch, mask_kind)
            output = F.linear(torch.tanh(F.linear(tokens, *params[:2])), *params[2:])
            loss, count, _ = _loss(targets, mask, output)
            if per_token:
                total_loss = total_loss + loss
            else:
                total_loss = total_loss + loss / count.clamp(min=1) / dp_size / num_microbatches
            total_count += count
    total_loss.backward()
    if per_token:
        # Divide the gradients after backward, as finalize_model_grads does. Dividing the loss
        # first changes the inputs of the backward GEMMs, which round differently under TF32.
        for param in params:
            param.grad /= total_count.clamp(min=1)
    return params


@pytest.fixture(
    scope="module",
    params=[(1, 1, None), (2, 1, None), (4, 1, None), (1, 2, None), (2, 2, None), (2, 2, 2)],
    ids=lambda p: "cp{}-pp{}-vpp{}".format(*p),
)
def parallel_setup(request):
    """Initialize one (CP, PP, VPP) topology and reuse it across the tests of this module."""
    cp_size, pp_size, vp_size = request.param
    if Utils.world_size % (cp_size * pp_size):
        pytest.skip("The CP x PP size must divide the world size")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        context_parallel_size=cp_size,
    )
    try:
        yield ProcessGroupCollection.use_mpu_process_groups(), vp_size
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("mask_kind", ["full", "uneven", "one_target", "empty", "varying"])
@pytest.mark.parametrize("per_token", [False, True])
def test_masked_loss_gradients_match_unsharded_reference(parallel_setup, mask_kind, per_token):
    """The schedule, DDP and grad finalization must reproduce the unsharded gradients."""
    pg_collection, vp_size = parallel_setup
    cp_size, pp_size = pg_collection.cp.size(), pg_collection.pp.size()
    pp_rank, dp_rank = pg_collection.pp.rank(), pg_collection.dp.rank()
    cp_rank = pg_collection.cp.rank()
    # Covers the in-loop and final microbatches of every schedule, and warmup, steady state
    # and cooldown of the pipelined ones.
    num_microbatches = 3
    num_chunks = vp_size or 1
    config = TransformerConfig(
        num_layers=4,
        hidden_size=4,
        num_attention_heads=1,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        microbatch_group_size_per_vp_stage=num_microbatches if vp_size else None,
        context_parallel_size=cp_size,
        calculate_per_token_loss=per_token,
        pipeline_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        finalize_model_grads_func=finalize_model_grads,
        deallocate_pipeline_outputs=False,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )
    modules = [
        _TokenModel(
            config,
            pre_process=pp_rank == 0 and chunk == 0,
            post_process=pp_rank == pp_size - 1 and chunk == num_chunks - 1,
        )
        for chunk in range(num_chunks)
    ]
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=False)
    model = [
        DistributedDataParallel(config, ddp_config, module, pg_collection=pg_collection)
        for module in modules
    ]
    for chunk in model:
        chunk.zero_grad_buffer()
    # One iterator per model chunk: every chunk reads the targets and mask of each microbatch.
    data_iterator = [
        iter(
            tuple(_cp_slice(value, cp_size, cp_rank) for value in _batch(dp_rank, i, mask_kind))
            for i in range(num_microbatches)
        )
        for _ in model
    ]

    def forward_step(data_iterator, wrapped_model):
        tokens, targets, mask = next(data_iterator)
        return wrapped_model(tokens), partial(_loss, targets, mask)

    metrics = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=8,
        micro_batch_size=1,
        p2p_communicator=P2PCommunicator(pg_collection.pp, config),
        pg_collection=pg_collection,
    )

    reference = _reference(pg_collection.dp.size(), num_microbatches, mask_kind, per_token)
    for module in modules:
        for name, param in module.named_parameters():
            index = (0 if name.startswith("first_layer") else 2) + int(name.endswith("bias"))
            torch.testing.assert_close(param.main_grad, reference[index].grad, rtol=2e-5, atol=2e-7)
    if modules[-1].post_process:
        assert len(metrics) == num_microbatches
        for i, metric in enumerate(metrics):
            local_mask = _cp_slice(_batch(dp_rank, i, mask_kind)[2], cp_size, cp_rank)
            # The reduced denominator must not leak into the reported, CP-local count.
            torch.testing.assert_close(metric["loss"][1], local_mask.sum())


def _calc_loss(count, output, per_token, **kwargs):
    metrics = []
    loss, returned_count = forward_step_calc_loss(
        model=None,
        output_tensor=output,
        loss_func=lambda value: (value.clone(), count, {"count": count}),
        config=SimpleNamespace(calculate_per_token_loss=per_token, timers=None),
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=3,
        forward_data_store=metrics,
        **kwargs,
    )
    return loss, returned_count, metrics


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("passed_args", ["all", "cp_group_only", "none"])
def test_loss_is_normalized_over_cp_group(parallel_setup, per_token, passed_args):
    """Each CP rank divides by the CP-summed count; the returned count stays local."""
    pg_collection, vp_size = parallel_setup
    cp_group = pg_collection.cp
    if cp_group.size() == 1 or vp_size is not None:
        pytest.skip("Requires context parallelism; independent of virtual pipelining")
    if passed_args == "all":
        kwargs = dict(cp_group_size=cp_group.size(), is_last_stage=True, cp_group=cp_group)
    elif pg_collection.pp.size() > 1:
        pytest.skip("The parallel_state fallback is checked on the last pipeline stage only")
    elif passed_args == "cp_group_only":
        # The combined 1F1B loss step passes the CP group but no sizes.
        kwargs = dict(cp_group=cp_group)
    else:
        # Direct callers of the combined 1F1B step that pass no groups use the global CP group.
        kwargs = {}
    count = torch.tensor(Utils.rank + 1, dtype=torch.int, device="cuda")
    output = torch.tensor(float(Utils.rank + 2), device="cuda", requires_grad=True)

    loss, returned_count, metrics = _calc_loss(count, output, per_token, **kwargs)

    cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    scale = 1.0 if per_token else cp_group.size() / sum(rank + 1 for rank in cp_ranks) / 3
    torch.testing.assert_close(loss, output.detach() * scale)
    loss.backward()
    torch.testing.assert_close(output.grad, torch.full_like(output, scale))
    assert returned_count is count and metrics[0]["count"] is count
    assert count.item() == Utils.rank + 1


def test_loss_requires_matching_cp_group(parallel_setup):
    """A CP size > 1 without a matching CP group must fail instead of mis-normalizing."""
    pg_collection, vp_size = parallel_setup
    cp_size = pg_collection.cp.size()
    if cp_size == 1 or vp_size is not None:
        pytest.skip("Requires context parallelism; independent of virtual pipelining")
    count = torch.tensor(1, dtype=torch.int, device="cuda")
    output = torch.tensor(2.0, device="cuda")
    with pytest.raises(ValueError, match="cp_group must be provided"):
        _calc_loss(count, output, per_token=False, cp_group_size=cp_size, is_last_stage=True)
    with pytest.raises(ValueError, match="cp_group has"):
        _calc_loss(
            count,
            output,
            per_token=False,
            cp_group_size=2 * cp_size,
            is_last_stage=True,
            cp_group=pg_collection.cp,
        )
