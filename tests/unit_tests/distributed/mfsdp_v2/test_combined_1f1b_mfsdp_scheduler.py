"""Gradient callback totals declared by the combined 1F1B scheduler."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    _register_borrowed_weight_unshard_hooks,
    _unit_grad_accumulation_count,
    _window_extras,
)


class TestUnitGradAccumulationCount:
    """Scheduler totals account for ownership, aliases, and additive consumers."""

    def test_additive_shared_weight_extras(self):
        """Borrowed embedding and projection consumers add to one physical weight."""
        embedding = nn.Parameter(torch.ones(2, 2))
        projection = nn.Parameter(torch.ones(2, 2))
        ordinary = nn.Parameter(torch.ones(2, 2))
        parameters = [
            SimpleNamespace(unsharded=embedding, sharded=object()),
            SimpleNamespace(unsharded=object(), sharded=projection),
            SimpleNamespace(unsharded=ordinary, sharded=object()),
        ]
        unit = SimpleNamespace(_trainable_fsdp_parameters=lambda: iter(parameters))
        extras = Counter({id(embedding): 2, id(projection): 1})
        assert _unit_grad_accumulation_count(unit, extras) == 6
        assert _unit_grad_accumulation_count(unit, Counter()) == 3

    def test_identical_representations_count_once(self):
        weight = nn.Parameter(torch.ones(2, 2))
        parameter = SimpleNamespace(unsharded=weight, sharded=weight)
        unit = SimpleNamespace(_trainable_fsdp_parameters=lambda: iter([parameter]))
        assert _unit_grad_accumulation_count(unit, Counter({id(weight): 2})) == 3

    def test_no_trainable_parameters(self):
        """Parameter-free units retain the module-hook fallback."""
        unit = SimpleNamespace(_trainable_fsdp_parameters=lambda: iter(()))
        assert _unit_grad_accumulation_count(unit, Counter()) == 0


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("pipeline_size", [1, 2])
@pytest.mark.parametrize("mtp_process", [False, True])
def test_stage_local_shared_weight_counts(tied, pipeline_size, mtp_process):
    embedding = nn.Parameter(torch.ones(2, 2))
    projection = nn.Parameter(torch.ones(2, 2))
    final_norm = nn.Parameter(torch.ones(2))
    model = nn.Module()
    model.share_embeddings_and_output_weights = tied
    model.pre_process = True
    model.mtp_process = mtp_process
    model.config = SimpleNamespace(mtp_num_layers=1, pipeline_model_parallel_size=pipeline_size)
    model.embedding = SimpleNamespace(word_embeddings=SimpleNamespace(weight=embedding))
    model.output_layer = SimpleNamespace(weight=projection)
    model.decoder = SimpleNamespace(final_layernorm=SimpleNamespace(weight=final_norm))

    extras = _window_extras(model)

    interleaved_mtp = int(mtp_process and pipeline_size > 1)
    assert extras[id(embedding)] == int(tied) + int(mtp_process) + int(tied) * interleaved_mtp
    assert extras[id(projection)] == int(not tied) * interleaved_mtp
    assert extras[id(final_norm)] == interleaved_mtp


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("owns_weight", [False, True])
def test_borrowed_weight_unshard_hooks(tied, owns_weight):
    output_layer = nn.Linear(2, 2, bias=False) if owns_weight else nn.Identity()
    embedding = nn.Embedding(2, 2)
    events = []
    owner = SimpleNamespace(is_root=lambda: False, unshard=lambda: events.append("unshard"))
    model = SimpleNamespace(
        share_embeddings_and_output_weights=tied,
        output_layer=output_layer,
        embedding=SimpleNamespace(word_embeddings=embedding),
    )
    _register_borrowed_weight_unshard_hooks(model, {embedding: owner})
    output_layer(torch.ones(2, 2, requires_grad=True)).sum().backward()
    assert events == (["unshard", "unshard"] if tied and not owns_weight else [])
