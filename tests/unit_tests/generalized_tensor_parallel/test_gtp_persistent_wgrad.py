# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded persistent scratch, including a repeated writer with an outstanding GTP4 RS."""

import os

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel import gtp_cuda_graphs
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="requires four ranks"),
]


@pytest.mark.parametrize("async_reduction", [False, True], ids=["sync", "async"])
def test_persistent_wgrad_reuse_preserves_every_reduction(monkeypatch, async_reduction):
    """Full gradients, padding and exactly-once completion survive shared/repeated writers."""
    Utils.initialize_distributed()
    group = torch.distributed.group.WORLD
    monkeypatch.setattr(gtp_cuda_graphs, "_WGRAD_RINGS", {})
    monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", async_reduction)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    # Both FC roles deliberately have the same shape; each has two independent experts.
    layers = []
    expected, completions = {}, {}
    for layer in range(3):
        roles = []
        for role in ("fc1", "fc2"):
            weights = [
                gtp.GTPShardedParam(torch.zeros(33, 16, dtype=torch.bfloat16, device="cuda"))
                for _ in range(2)
            ]
            for expert, weight in enumerate(weights):
                weight.group, weight.pad_length, weight.expert_idx = group, 2, expert
                weight.chain_id = f"GTP_remat_grouped_{role}_ungraphed"
                weight.is_routed_expert = True
                weight._debug_name = f"layers.{layer}.{role}.weight{expert}"
                weight.main_grad = torch.full_like(weight, 0.5)
                weight._double_buffer_parity()  # Normally assigned by the forward gathers.
                expected[id(weight)] = weight.main_grad.clone()
                completions[id(weight)] = 0

                def completed(weight=weight):
                    completions[id(weight)] += 1

                weight.register_grad_accum_hook(None, completed)
            weights[0].weight_list = weights
            roles.append(weights)
        layers.append(roles)
    for previous, current in zip(layers, layers[1:]):
        for prev_weights, weights in zip(previous, current):
            prev_weights[0].next_w, weights[0].prev_w = weights[0], prev_weights[0]

    pointers = {}
    try:
        # Repeating the tail before the cascade drains its first RS exercises early reuse.
        for step in range(3):
            for layer_index in (2, 2, 1, 0):
                for role_index in (1, 0):
                    weights = layers[layer_index][role_index]
                    if async_reduction:
                        with torch.cuda.stream(gtp.get_rs_stream(weights[0].chain_id, group)):
                            torch.cuda._sleep(1_000_000)
                    grads = []
                    for weight in weights:
                        grad = weight.get_wgrad_tensor(persistent=True)
                        prior = pointers.setdefault(id(weight), grad.data_ptr())
                        assert grad.data_ptr() == prior
                        if step == 1:
                            # Foreign gradients must copy into the same padded ring storage.
                            grad = torch.empty_like(grad)
                        base = (
                            torch.arange(130 * 16, device="cuda").view(130, 16) % 11 - 5
                        ).float() / 4 + (step + layer_index + weight.expert_idx) / 8
                        grad.copy_(base + torch.distributed.get_rank() / 4)
                        mean = torch.nn.functional.pad(base + 3 / 8, (0, 0, 0, 2))
                        # Mean 250.25 + main_grad 0.5 rounds to 251 once, or 250.5 if the
                        # RS output is first rounded to BF16. Padding must still stay zero.
                        grad[:, 0] = 251 if torch.distributed.get_rank() == 3 else 250
                        mean[:130, 0] = 250.25
                        expected[id(weight)].add_(mean.chunk(4)[torch.distributed.get_rank()])
                        grads.append(grad)
                    weights[0].finalize_group_grads(grads)
            torch.cuda.synchronize()
            for layer in layers:
                for weights in layer:
                    for weight in weights:
                        torch.testing.assert_close(
                            weight.main_grad, expected[id(weight)], rtol=0, atol=0
                        )
                        assert not torch.count_nonzero(weight._gtp_wgrad_ring_slot.tensor[130:])
                        calls_per_step = 2 if layer is layers[-1] else 1
                        assert completions[id(weight)] == (step + 1) * calls_per_step
            # Two buffers per role/expert, independent of the three-layer model depth.
            assert len(set(pointers.values())) == 8
            assert len(gtp_cuda_graphs._WGRAD_RINGS) == 8
    finally:
        torch.cuda.synchronize()
        for layer in layers:
            for weights in layer:
                weights[0]._wait_reduce_scatter(finalize_grad=True)
