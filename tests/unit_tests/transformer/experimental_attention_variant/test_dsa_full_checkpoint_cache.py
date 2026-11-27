# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest

from megatron.core.recompute import current_checkpoint_topk_cache, with_checkpoint_topk_cache


def test_overlapping_microbatch_checkpoints_of_the_same_layer():
    layer = object()

    def layer_call(record=None):
        cache = current_checkpoint_topk_cache()
        if record is None:
            return cache[layer]
        cache[layer] = record

    first = with_checkpoint_topk_cache(layer_call)
    second = with_checkpoint_topk_cache(layer_call)
    first(record=1)
    second(record=2)
    assert current_checkpoint_topk_cache() is None
    assert second() == 2
    assert first() == 1
    assert first() == 1


def test_nested_cache_restored_even_on_error():
    def failed():
        assert not current_checkpoint_topk_cache()
        raise RuntimeError("failed checkpoint")

    def outer():
        current_checkpoint_topk_cache()["outer"] = True
        with pytest.raises(RuntimeError, match="failed checkpoint"):
            with_checkpoint_topk_cache(failed)()
        assert current_checkpoint_topk_cache() == {"outer": True}

    with_checkpoint_topk_cache(outer)()
    assert current_checkpoint_topk_cache() is None
