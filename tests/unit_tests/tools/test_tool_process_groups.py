# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference and embedding tools take their process groups from the model they run."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _forbid_global_grid(monkeypatch, *names):
    """Make the named parallel_state accessors raise for the rest of the test."""

    def read_global_grid(*args, **kwargs):
        raise AssertionError("the tool must use the process groups of the model it runs")

    for name in names:
        monkeypatch.setattr(parallel_state, name, read_global_grid)


@pytest.fixture
def pipeline_group(request):
    """A pipeline group of `request.param` consecutive ranks on a global grid without PP."""
    pipeline_stages = request.param
    world_size = Utils.world_size
    if world_size < pipeline_stages or world_size % pipeline_stages:
        pytest.skip(f"needs a world size divisible by {pipeline_stages}")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    group, _ = torch.distributed.new_subgroups_by_enumeration(
        [
            list(range(start, start + pipeline_stages))
            for start in range(0, world_size, pipeline_stages)
        ]
    )
    try:
        yield group
    finally:
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("pipeline_group", [1, 2], indirect=True)
def test_static_engine_pipeline_flag_follows_the_wrapper_group(pipeline_group, monkeypatch):
    """The static engine's pipeline flag comes from the wrapper's pipeline group.

    The global grid has no pipeline parallelism, so reading it would report a two-stage
    pipeline group as not pipeline parallel.
    """
    import tools.run_inference_performance_test as performance_test

    class FakeInferenceWrapper:
        def __init__(self, model, inference_context):
            self.pp_group = pipeline_group

    monkeypatch.setattr(performance_test, "get_tokenizer", lambda: None)
    monkeypatch.setattr(performance_test, "StaticInferenceContext", lambda *args: None)
    monkeypatch.setattr(performance_test, "GPTInferenceWrapper", FakeInferenceWrapper)
    monkeypatch.setattr(
        performance_test,
        "TextGenerationController",
        lambda inference_wrapped_model, tokenizer: SimpleNamespace(
            inference_wrapped_model=inference_wrapped_model
        ),
    )
    monkeypatch.setattr(
        performance_test,
        "StaticInferenceEngine",
        lambda text_generation_controller: SimpleNamespace(controller=text_generation_controller),
    )
    _forbid_global_grid(
        monkeypatch,
        "get_pipeline_model_parallel_group",
        "get_pipeline_model_parallel_rank",
        "get_pipeline_model_parallel_world_size",
        "is_pipeline_first_stage",
        "is_pipeline_last_stage",
    )

    engine = performance_test.get_inference_engine(
        SimpleNamespace(
            engine_type="static", inference_max_requests=1, inference_max_sequence_length=8
        ),
        model=None,
    )

    expected = pipeline_group.size() > 1
    assert engine.controller.inference_wrapped_model.model_is_pipeline_parallel is expected


def test_disk_embedder_splits_blocks_over_the_model_data_parallel_group(monkeypatch, tmp_path):
    """The block split uses the embedding model's data-parallel group, not the global one."""
    pytest.importorskip("h5py")
    pytest.importorskip("transformers")
    # The embedding tool imports the BERT example's model provider as a top-level module.
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "examples" / "bert"))
    embed = importlib.import_module("tools.bert_embedding.embed")

    data_parallel_group = object()
    block_split_calls = []

    def get_blocks_by_rank(*args, **kwargs):
        block_split_calls.append(kwargs)
        return SimpleNamespace(missing=[])

    monkeypatch.setattr(embed, "get_blocks_by_rank", get_blocks_by_rank)
    monkeypatch.setattr(embed.torch.distributed, "barrier", lambda *args, **kwargs: None)
    _forbid_global_grid(monkeypatch, "get_data_parallel_group")

    embedder = embed.BertEmbedder.__new__(embed.BertEmbedder)
    embedder.models = [
        SimpleNamespace(pg_collection=ProcessGroupCollection(dp_gtp_remat=data_parallel_group))
    ]
    disk_embedder = embed.DiskDataParallelBertEmbedder(embedder, block_size=4)

    disk_embedder.embed_text_dataset("train", str(tmp_path), embed.TextDataset(["a", "b"]))

    assert len(block_split_calls) == 1
    assert block_split_calls[0].get("process_group") is data_parallel_group
