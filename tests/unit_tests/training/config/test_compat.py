# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The two seams between the run config and the legacy arguments: the data sequence length and the
model topology the data path reads, with their fallbacks for sections and containers that lack the
owning object."""

from types import SimpleNamespace

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training.config import DataConfig
from megatron.training.config.compat import data_sequence_length, model_layout, model_topology

BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)


def _dataset_section(sequence_length=16):
    return DataConfig(
        dataset=GPTDatasetConfig(sequence_length=sequence_length, random_seed=1, **BOOLS)
    )


def _transformer(**overrides):
    values = dict(
        context_parallel_size=2,
        pipeline_model_parallel_size=4,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="dp_balanced",
        pipeline_model_parallel_layout="layout-from-model",
        mtp_num_layers=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_data_sequence_length_comes_from_the_dataset_and_never_reads_args_then():
    """With a dataset the namespace is not consulted: a different, or absent, args.seq_length
    cannot leak in. Without a dataset (a loader-only section) the namespace is the source."""
    cfg = SimpleNamespace(data=_dataset_section(16))
    assert data_sequence_length(cfg, SimpleNamespace(seq_length=32)) == 16
    assert data_sequence_length(cfg, SimpleNamespace()) == 16  # no seq_length attribute at all
    loader_only = SimpleNamespace(data=DataConfig())
    assert data_sequence_length(loader_only, SimpleNamespace(seq_length=32)) == 32
    with pytest.raises(AttributeError):
        data_sequence_length(loader_only, SimpleNamespace())


def test_model_topology_reads_the_model_section_when_there_is_one():
    args = SimpleNamespace(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
    )
    with_model = SimpleNamespace(model=SimpleNamespace(transformer=_transformer()))
    topology = model_topology(with_model, args)
    assert topology.source == "model"
    assert (topology.context_parallel_size, topology.pipeline_model_parallel_size) == (2, 4)
    assert topology.dynamic_context_parallel is True
    assert topology.sequence_packing_scheduler == "dp_balanced"
    assert model_layout(with_model, args) == ("layout-from-model", 3)


@pytest.mark.parametrize("modelless", [SimpleNamespace(model=None), SimpleNamespace()])
def test_model_topology_falls_back_to_the_namespace_without_a_model_section(monkeypatch, modelless):
    """A container built without model_cfg (pretrain_hybrid_flex.py today): the namespace is the
    source, and the layout comes from the transformer config the arguments describe, as the rank
    predicate built it before."""
    args = SimpleNamespace(
        context_parallel_size=3,
        pipeline_model_parallel_size=2,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
    )
    topology = model_topology(modelless, args)
    assert topology.source == "args"
    assert (topology.context_parallel_size, topology.pipeline_model_parallel_size) == (3, 2)
    assert (
        topology.dynamic_context_parallel is False and topology.sequence_packing_scheduler is None
    )
    import megatron.training.arguments as arguments

    monkeypatch.setattr(
        arguments,
        "core_transformer_config_from_args",
        lambda ns: _transformer(
            pipeline_model_parallel_layout="layout-from-args", mtp_num_layers=None
        ),
    )
    assert model_layout(modelless, args) == ("layout-from-args", None)
