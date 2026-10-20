# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Instance-owned RoPE/YaRN caches shared by eager execution and CUDA graphs."""

import gc
import weakref

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(params=[RotaryEmbedding, YarnRotaryEmbedding], ids=["rope", "yarn"])
def rotary(request):
    Utils.initialize_model_parallel(1, 1)
    kwargs = (
        {"original_max_position_embeddings": 16} if request.param is YarnRotaryEmbedding else {}
    )
    module = request.param(kv_channels=8, rotary_percent=1.0, **kwargs)
    yield module
    Utils.destroy_model_parallel()


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


def test_cache_reuses_normalized_arguments_and_separates_offsets(rotary):
    output = rotary(16)
    assert rotary(max_seq_len=16, offset=0, packed_seq=False, cp_group=rotary.cp_group) is output
    offset_output = rotary(16, offset=1)
    assert _tensor(offset_output) is not _tensor(output)
    torch.testing.assert_close(_tensor(offset_output), _tensor(rotary(17))[1:], rtol=0, atol=0)


def test_cache_distinguishes_cp_groups_and_packing(rotary):
    group = torch.distributed.group.WORLD
    if group.size() < 2:
        pytest.skip("Requires at least two ranks.")
    full = _tensor(rotary(16))
    sharded = _tensor(rotary(16, cp_group=group))
    packed = _tensor(rotary(16, packed_seq=True, cp_group=group))
    assert sharded.shape[0] == full.shape[0] // group.size()
    torch.testing.assert_close(packed, full, rtol=0, atol=0)
    assert _tensor(rotary(16, cp_group=group)) is sharded


def test_inference_cache_can_be_reused_by_training_backward(rotary):
    with torch.inference_mode():
        output = rotary(16)
    embedding = _tensor(output)
    assert not embedding.is_inference()
    assert rotary(16) is output
    # Multiplication saves the constant for backward, which rejects inference tensors.
    activations = torch.randn_like(embedding, requires_grad=True)
    (activations * embedding).sum().backward()
    torch.testing.assert_close(activations.grad, embedding, rtol=0, atol=0)


def test_cache_is_bounded_and_clear_releases_ordinary_entries(rotary):
    capacity = rotary.CACHE_CAPACITY
    first = weakref.ref(_tensor(rotary(1)))
    for seq_len in range(2, capacity + 2):
        rotary(seq_len)
    gc.collect()
    assert first() is None
    assert len(rotary._embedding_cache) == capacity
    last = weakref.ref(_tensor(rotary(capacity + 1)))
    rotary.clear_cache()
    gc.collect()
    assert last() is None
    assert not rotary._embedding_cache


def test_cache_does_not_keep_embedding_instance_alive(rotary):
    # Unlike a method-level functools cache, the cache must not retain its owner.
    other = type(rotary)(kv_channels=8, rotary_percent=1.0)
    owner = weakref.ref(other)
    embedding = weakref.ref(_tensor(other(16)))
    del other
    gc.collect()
    assert owner() is None
    assert embedding() is None


def test_captured_constant_survives_eviction_clear_and_recapture(rotary, monkeypatch):
    with torch.inference_mode():
        expected = _tensor(rotary(16)) + 1
        constant = weakref.ref(_tensor(rotary(16)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _tensor(rotary(16)) + 1

    for seq_len in range(17, 17 + rotary.CACHE_CAPACITY + 2):
        rotary(seq_len)
    rotary.clear_cache()
    gc.collect()
    torch.cuda.empty_cache()
    assert constant() is not None
    assert len(rotary._embedding_cache) == 1

    def unexpected_recompute(*args, **kwargs):
        pytest.fail("A retained graph constant must be reused without recomputation.")

    monkeypatch.setattr(rotary, "get_emb", unexpected_recompute)
    assert _tensor(rotary(16)) is constant()
    with torch.inference_mode():
        second_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second_graph):
            second_output = _tensor(rotary(16)) + 1
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        second_graph.replay()
        torch.testing.assert_close(second_output, expected, rtol=0, atol=0)

    del graph, second_graph, output, second_output
    rotary.clear_cache(include_cuda_graphs=True)
    gc.collect()
    assert constant() is None
