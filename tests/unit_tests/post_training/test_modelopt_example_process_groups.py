# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The ModelOpt example scripts take their process groups from the model they run.

The groups here are stand-ins with fixed ranks, so a read of the global grid either raises or
returns a group the assertions reject.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("modelopt")
transformers = pytest.importorskip("transformers")

_MODEL_OPT_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "post_training" / "modelopt"
sys.path.insert(0, str(_MODEL_OPT_EXAMPLE_DIR))

import finetune  # noqa: E402

from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from tests.unit_tests.test_utilities import forbid_global_process_groups  # noqa: E402


class _Group:
    """A stand-in process group with a fixed rank and size."""

    def __init__(self, rank=0, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _import_script(name):
    """Import an example script whose ModelOpt imports need a recent ModelOpt release."""
    try:
        return importlib.import_module(name)
    except (ImportError, AttributeError) as error:
        pytest.skip(f"{name}.py needs a newer ModelOpt: {error}")


def _pg_collection(pp_rank=0, pp_size=1):
    return ProcessGroupCollection(
        tp=_Group(0, 2),
        pp=_Group(pp_rank, pp_size),
        cp=_Group(0, 2),
        ep=_Group(0, 2),
        expt_dp_gtp_remat=_Group(2, 4),
    )


def test_sft_datasets_shard_over_the_collection_expert_data_parallel_group(monkeypatch):
    args = SimpleNamespace(
        micro_batch_size=1,
        export_offline_model=False,
        finetune_hf_dataset="dataset",
        finetune_data_split="train",
        finetune_data_files=None,
        seq_length=8,
    )
    dataset_kwargs = []
    monkeypatch.setattr(finetune, "get_args", lambda: args)
    monkeypatch.setattr(
        finetune, "get_hf_tokenizer", lambda: MagicMock(spec=transformers.PreTrainedTokenizerBase)
    )
    monkeypatch.setattr(
        finetune, "SFTDataset", lambda num_samples, **kwargs: dataset_kwargs.append(kwargs)
    )

    with forbid_global_process_groups():
        finetune.train_valid_test_sft_datasets_provider([4, 2, 1], pg_collection=_pg_collection())

    assert [(kw["num_shards"], kw["shard_index"]) for kw in dataset_kwargs] == [(4, 2)] * 3


def _batch_args():
    return SimpleNamespace(
        export_offline_model=False,
        seq_length=8,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )


def test_get_batch_broadcasts_and_shards_over_the_model_groups(monkeypatch):
    broadcast_groups, cp_groups = [], []
    monkeypatch.setattr(finetune, "get_args", _batch_args)
    monkeypatch.setattr(finetune, "get_eos_token_id", lambda: 0)

    def broadcast_data(keys, data, datatype, tp_group=None):
        broadcast_groups.append(tp_group)
        return {key: torch.ones(1, 9, dtype=datatype) for key in keys}

    def build_lm_batch(input_ids, seq_length, *, cp_group=None, **kwargs):
        cp_groups.append(cp_group)
        return {"tokens": input_ids[:, :seq_length]}

    monkeypatch.setattr(tensor_parallel, "broadcast_data", broadcast_data)
    monkeypatch.setattr(finetune, "build_lm_batch", build_lm_batch)
    pg_collection = _pg_collection()

    with forbid_global_process_groups():
        batch = finetune.get_batch(iter([{}]), pg_collection=pg_collection)

    assert batch["tokens"].shape == (1, 8)
    assert broadcast_groups == [pg_collection.tp]
    assert cp_groups == [pg_collection.cp]


def test_get_batch_skips_the_middle_pipeline_stages_of_the_model(monkeypatch):
    # The stage helpers read the rank of a group only once torch.distributed is up.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(finetune, "get_args", _batch_args)

    with forbid_global_process_groups():
        batch = finetune.get_batch(iter([{}]), pg_collection=_pg_collection(pp_rank=1, pp_size=3))

    assert batch == (None, None, None, None, None)


def test_forward_step_passes_the_model_groups(monkeypatch):
    pg_collection = _pg_collection()
    batch_groups = []

    def get_batch(data_iterator, pg_collection=None):
        batch_groups.append(pg_collection)
        return dict.fromkeys(("tokens", "labels", "loss_mask", "attention_mask", "position_ids"))

    monkeypatch.setattr(finetune, "get_batch", get_batch)
    monkeypatch.setattr(finetune, "get_timers", lambda: MagicMock())
    monkeypatch.setattr(finetune, "get_args", lambda: SimpleNamespace(export_offline_model=False))
    model = MagicMock(spec=["__call__", "pg_collection"])
    model.pg_collection = pg_collection

    with forbid_global_process_groups():
        finetune.forward_step(iter([]), model)

    assert batch_groups == [pg_collection]


def test_prune_counts_parameters_over_the_model_groups(monkeypatch):
    prune = _import_script("prune")

    reduced_over = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, group=None: reduced_over.append(group)
    )
    model = torch.nn.Linear(2, 3)
    model.pg_collection = _pg_collection()

    with forbid_global_process_groups():
        assert prune.get_params(model) == 9

    assert reduced_over == [model.pg_collection.pp, model.pg_collection.tp]


def test_convert_builds_the_model_on_the_stage_of_its_groups(monkeypatch):
    convert_model = _import_script("convert_model")

    pg_collection = _pg_collection(pp_rank=1, pp_size=2)
    built_with = []

    def model_provider(**kwargs):
        built_with.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        convert_model, "get_args", lambda: SimpleNamespace(init_model_with_meta_device=False)
    )
    # The script takes the groups of the grid that initialize_megatron registered.
    monkeypatch.setattr(convert_model, "get_registered_process_groups", lambda: pg_collection)

    with forbid_global_process_groups():
        convert_model.get_model(model_provider)

    assert built_with == [
        {"pre_process": False, "post_process": True, "pg_collection": pg_collection}
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_feature_extraction_splits_samples_over_the_model_expert_data_parallel_group(
    monkeypatch, tmp_path
):
    from examples.post_training.modelopt import offline_feature_extract

    class _Dataset:
        seq_length = 4

        def __getitem__(self, index):
            return {"input_ids": torch.full((4,), index)}

    class _Model:
        pg_collection = _pg_collection()

        def __call__(self, input_ids, return_eagle_inputs):
            return input_ids

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    with forbid_global_process_groups():
        offline_feature_extract.extract_feature(_Dataset(), _Model(), str(tmp_path), 0, 12)

    # Expert data-parallel rank 2 of 4 handles samples 2, 6 and 10.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "00000002.pt",
        "00000006.pt",
        "00000010.pt",
    ]
