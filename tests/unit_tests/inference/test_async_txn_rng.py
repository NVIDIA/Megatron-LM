# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.async_txn import RequestRNGStore
from megatron.core.inference.sampling.torch_sampling import TorchSampling


class FakeContext:
    def __init__(self, request_ids, rng_store):
        self.total_request_count = len(request_ids)
        self.paused_request_count = 0
        self.request_ids = torch.tensor(request_ids, dtype=torch.int32, device='cpu')
        self.request_rng_store = rng_store
        self.active_request_metadata = {
            "temperature": torch.ones(len(request_ids), dtype=torch.float32, pin_memory=True),
            "top_k": torch.zeros(len(request_ids), dtype=torch.int32, pin_memory=True),
            "top_p": torch.zeros(len(request_ids), dtype=torch.float32, pin_memory=True),
        }


def test_request_rng_store_is_keyed_by_request_id():
    store_a = RequestRNGStore(123, device='cpu')
    store_b = RequestRNGStore(123, device='cpu')

    first = torch.rand(3, generator=store_a.get(7))
    second = torch.rand(3, generator=store_b.get(7))
    other = torch.rand(3, generator=store_b.get(8))

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_removing_finished_request_does_not_reset_survivor_rng():
    store = RequestRNGStore(123, device='cpu')
    expected_first = torch.rand(1, generator=store.get(2))
    store.remove(1)
    expected_second = torch.rand(1, generator=store.get(2))

    replay = RequestRNGStore(123, device='cpu')
    replay_first = torch.rand(1, generator=replay.get(2))
    replay_second = torch.rand(1, generator=replay.get(2))

    assert torch.equal(expected_first, replay_first)
    assert torch.equal(expected_second, replay_second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampling test requires GPU")
def test_torch_sampling_uses_request_rng_when_store_is_present():
    device = torch.device("cuda", torch.cuda.current_device())
    store_a = RequestRNGStore(555, device=device)
    store_b = RequestRNGStore(555, device=device)
    context = FakeContext([101, 202], store_a)
    sampler = TorchSampling(torch.Generator(device=device), vocab_size=5)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]], device='cuda')

    first = sampler.sample_kernel(logits, 2, context)
    context_reordered = FakeContext([202, 101], store_b)
    second = sampler.sample_kernel(logits.flip(0), 2, context_reordered)

    assert first[0].item() == second[1].item()
    assert first[1].item() == second[0].item()
