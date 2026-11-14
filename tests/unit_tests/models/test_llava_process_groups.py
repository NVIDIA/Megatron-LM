# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""LLaVA shards its language-model input over the groups of its own collection."""

import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def _forbid_global_tp(patch):
    """Make the global TP accessors raise, including by-name imports of them."""
    for name in dir(parallel_state):
        if not name.startswith("get_tensor_model_parallel"):
            continue
        original = getattr(parallel_state, name)

        def forbid(*args, _name=name, **kwargs):
            raise AssertionError(f"read of the global grid: parallel_state.{_name}")

        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("megatron.") and (
                getattr(module, "__dict__", {}).get(name) is original
            ):
                patch.setattr(module, name, forbid)


def _make_sequence_parallel_model(pg_collection):
    """Build only the state that sequence-parallel input sharding reads."""
    model = object.__new__(LLaVAModel)
    torch.nn.Module.__init__(model)
    model.pre_process = True
    model.post_process = True
    model.context_parallel_lm = 1
    model.sequence_parallel_lm = True
    model.tensor_model_parallel_size_lm = pg_collection.tp.size()
    model.tp_comm_overlap_lm = False
    model.pg_collection = pg_collection
    return model


@pytest.mark.internal
@pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2 != 0, reason="needs an even number of ranks"
)
def test_sequence_parallel_input_is_split_over_model_tp_group(monkeypatch):
    """Embeddings go to the model's TP=2 group even though the global grid has TP=1."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    grid = HyperCommGrid([2, Utils.world_size // 2], ["tp", "dp"])
    try:
        tp_group = grid.create_pg("tp")
        model = _make_sequence_parallel_model(ProcessGroupCollection(tp=tp_group))
        sequence_length, batch_size, hidden_size = 8, 2, 4
        combined_embeddings = (
            torch.arange(sequence_length * batch_size * hidden_size, device="cuda")
            .view(sequence_length, batch_size, hidden_size)
            .float()
            .requires_grad_(True)
        )
        labels = torch.ones((batch_size, sequence_length), dtype=torch.long, device="cuda")
        loss_mask = torch.ones((batch_size, sequence_length), device="cuda")

        with monkeypatch.context() as patch:
            _forbid_global_tp(patch)
            embeddings, labels_out, loss_mask_out, _ = model._process_embedding_token_parallel(
                combined_embeddings, labels, loss_mask, None
            )
            # Weight each rank's chunk by its global rank, so the gathered gradient shows which
            # ranks the backward pass collected from.
            (embeddings * (torch.distributed.get_rank() + 1)).sum().backward()

        chunk = sequence_length // tp_group.size()
        start = tp_group.rank() * chunk
        assert torch.equal(embeddings, combined_embeddings[start : start + chunk])
        # Sequence parallelism shards only the embeddings; labels and loss mask stay whole.
        assert torch.equal(labels_out, labels)
        assert torch.equal(loss_mask_out, loss_mask)
        expected_grad = torch.cat(
            [
                torch.full((chunk, batch_size, hidden_size), rank + 1.0, device="cuda")
                for rank in torch.distributed.get_process_group_ranks(tp_group)
            ]
        )
        torch.testing.assert_close(combined_embeddings.grad, expected_grad)
    finally:
        grid.destroy()
        Utils.destroy_model_parallel()
