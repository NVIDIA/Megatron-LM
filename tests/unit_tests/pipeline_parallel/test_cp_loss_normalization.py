# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare masked-loss schedule gradients with an unsharded training step."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
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
    """Two token-wise projections, optionally split across pipeline stages."""

    def __init__(self, config, pp_rank):
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pp_rank == 0
        self.post_process = pp_rank == config.pipeline_model_parallel_size - 1
        self.share_embeddings_and_output_weights = False
        self.input_tensor = None
        self.first_layer = torch.nn.Linear(4, 4).cuda() if self.pre_process else None
        self.last_layer = torch.nn.Linear(4, 4).cuda() if self.post_process else None
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
    chunks = value.chunk(2 * cp_size, dim=0)
    return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=0)


def _loss(targets, mask, output):
    per_token = (output - targets).square().mean(dim=-1)
    loss = (per_token * mask).sum()
    count = mask.sum().detach().to(torch.int)
    return loss, count, {"loss": torch.stack((loss.detach(), count))}


def _reference(dp_size, num_microbatches, mask_kind, per_token):
    params = [value.detach().requires_grad_() for i in range(2) for value in _initial_parameters(i)]
    total_loss = torch.zeros((), device="cuda")
    total_count = torch.zeros((), device="cuda")
    for dp_rank in range(dp_size):
        for microbatch in range(num_microbatches):
            tokens, targets, mask = _batch(dp_rank, microbatch, mask_kind)
            output = F.linear(torch.tanh(F.linear(tokens, *params[:2])), *params[2:])
            loss, count, _ = _loss(targets, mask, output)
            total_loss = total_loss + (
                loss if per_token else loss / count.clamp(min=1) / dp_size / num_microbatches
            )
            total_count += count
    if per_token:
        total_loss = total_loss / total_count.clamp(min=1)
    total_loss.backward()
    return params


@pytest.fixture(scope="module", params=[(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)])
def parallel_groups(request):
    """Reuse each topology across masks to avoid repeated communicator creation."""
    cp_size, pp_size = request.param
    if Utils.world_size % (cp_size * pp_size):
        pytest.skip("The CP/PP configuration must divide the distributed world size")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp_size, context_parallel_size=cp_size
    )
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("mask_kind", ["full", "uneven", "one_target", "empty", "varying"])
@pytest.mark.parametrize("num_microbatches", [1, 3])
@pytest.mark.parametrize("per_token", [False, True])
def test_masked_loss_training_step(parallel_groups, mask_kind, num_microbatches, per_token):
    """Real schedule, DDP, finalization and SGD must match the unsplit objective."""
    pg_collection = parallel_groups
    cp_size, pp_size = pg_collection.cp.size(), pg_collection.pp.size()
    pp_rank = pg_collection.pp.rank()
    dp_rank = pg_collection.dp.rank()
    cp_rank = pg_collection.cp.rank()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        calculate_per_token_loss=per_token,
        pipeline_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        finalize_model_grads_func=finalize_model_grads,
        deallocate_pipeline_outputs=False,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )
    module = _TokenModel(config, pp_rank)
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=False),
        module,
        pg_collection=pg_collection,
    )
    model.zero_grad_buffer()
    batches = iter(
        tuple(_cp_slice(value, cp_size, cp_rank) for value in _batch(dp_rank, i, mask_kind))
        for i in range(num_microbatches)
    )

    def forward_step(data_iterator, wrapped_model):
        tokens, targets, mask = next(data_iterator)
        return wrapped_model(tokens), partial(_loss, targets, mask)

    metrics = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=batches,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=8,
        micro_batch_size=1,
        p2p_communicator=P2PCommunicator(pg_collection.pp, config),
        pg_collection=pg_collection,
    )
    reference = _reference(pg_collection.dp.size(), num_microbatches, mask_kind, per_token)
    for name, param in module.named_parameters():
        offset = 0 if name.startswith("first_layer") else 2
        expected = reference[offset + int(name.endswith("bias"))]
        torch.testing.assert_close(param.main_grad, expected.grad, rtol=2e-5, atol=2e-7)
        # Compare an actual optimizer update using the finalized Core gradient.
        param.grad = param.main_grad.clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    for name, param in module.named_parameters():
        offset = 0 if name.startswith("first_layer") else 2
        expected = reference[offset + int(name.endswith("bias"))]
        torch.testing.assert_close(param, expected - 0.1 * expected.grad, rtol=2e-5, atol=2e-7)
    if module.post_process:
        assert len(metrics) == num_microbatches
        for i, metric in enumerate(metrics):
            local_mask = _cp_slice(_batch(dp_rank, i, mask_kind)[2], cp_size, cp_rank)
            # Reducing a denominator must not overwrite local metric/token-count storage.
            torch.testing.assert_close(metric["loss"][1], local_mask.sum())


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("explicit_group", [False, True])
def test_loss_preserves_local_count_and_uses_cp_group(per_token, explicit_group):
    """Independent CP groups keep distinct denominators and local count aliases."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("Requires an even number of ranks")
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        count = torch.tensor(Utils.rank + 1, dtype=torch.int, device="cuda")
        original_count = count.clone()
        output = torch.tensor(float(Utils.rank + 2), device="cuda", requires_grad=True)
        config = SimpleNamespace(calculate_per_token_loss=per_token, timers=None)
        metrics = []
        with patch.object(
            parallel_state, "get_context_parallel_group", return_value=cp_group
        ) as get_group:
            loss, returned_count = forward_step_calc_loss(
                model=None,
                output_tensor=output,
                loss_func=lambda value: (value.clone(), count, {"count": count}),
                config=config,
                vp_stage=None,
                collect_non_loss_data=False,
                num_microbatches=3,
                forward_data_store=metrics,
                cp_group_size=2,
                is_last_stage=True,
                cp_group=cp_group if explicit_group else None,
            )
            assert get_group.call_count == int(not per_token and not explicit_group)
        ranks = torch.distributed.get_process_group_ranks(cp_group)
        scale = 1.0 if per_token else 2.0 / sum(rank + 1 for rank in ranks) / 3
        torch.testing.assert_close(loss, output.detach() * scale)
        loss.backward()
        torch.testing.assert_close(output.grad, torch.full_like(output, scale))
        assert returned_count is count and metrics[0]["count"] is count
        torch.testing.assert_close(count, original_count)
    finally:
        Utils.destroy_model_parallel()
