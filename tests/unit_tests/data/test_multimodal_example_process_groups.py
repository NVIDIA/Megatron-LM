# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The multimodal example reads its batch and reduces its loss over the model's process groups.

The groups here are stand-ins with fixed ranks, or a test grid built without the global one, so a
read of the global grid either raises or returns a group the assertions reject.
"""

import importlib.util
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core import process_groups_config, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

_MULTIMODAL_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"


class _Group:
    """A stand-in process group with a fixed rank and size."""

    def __init__(self, rank=0, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _pg_collection(tp_rank=0, pp_rank=0, pp_size=1):
    return ProcessGroupCollection(
        tp=_Group(tp_rank, 2),
        cp=_Group(0, 2),
        tp_cp=_Group(tp_rank, 4),
        pp=_Group(pp_rank, pp_size),
    )


@pytest.fixture
def train(monkeypatch):
    """examples/multimodal/train.py, imported with its model provider and arguments stubbed."""
    monkeypatch.syspath_prepend(str(_MULTIMODAL_DIR))
    for name, attrs in (
        ("model", {"model_provider": None}),
        ("multimodal_args", {"add_multimodal_extra_args": None}),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "multimodal_example_train", _MULTIMODAL_DIR / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_timers", lambda: MagicMock())
    monkeypatch.setattr(module, "get_tokenizer", lambda: SimpleNamespace(pad=0))
    monkeypatch.setattr(
        module,
        "get_run_config",
        lambda: SimpleNamespace(dist=SimpleNamespace(use_torch_fsdp2=False)),
    )
    return module


def _batch_args(deduplicate=False, context_parallel_size=1):
    return SimpleNamespace(
        deduplicate_dataloader_across_context_parallel=deduplicate,
        context_parallel_size=context_parallel_size,
        encoder_pipeline_model_parallel_size=0,
        log_packed_sequence_stats=False,
        reset_position_ids_from_packed_metadata=False,
    )


def _record_broadcasts(monkeypatch):
    """Replace broadcast_data with one that records its group and returns an unpacked batch."""
    groups = []

    def broadcast_data(keys, data, datatype, tp_group=None):
        groups.append(tp_group)
        shapes = {"tokens": (1, 8), "labels": (1, 9)}
        return {key: torch.zeros(shapes.get(key, (1, 1)), dtype=datatype) for key in keys}

    monkeypatch.setattr(tensor_parallel, "broadcast_data", broadcast_data)
    return groups


@pytest.mark.parametrize(
    ("deduplicate", "context_parallel_size", "group_name"),
    [(False, 1, "tp"), (False, 2, "tp"), (True, 2, "tp_cp")],
)
def test_get_batch_broadcasts_over_the_model_groups(
    train, monkeypatch, deduplicate, context_parallel_size, group_name
):
    monkeypatch.setattr(train, "get_args", lambda: _batch_args(deduplicate, context_parallel_size))
    groups = _record_broadcasts(monkeypatch)
    pg_collection = _pg_collection()

    with forbid_global_process_groups():
        tokens = train.get_batch(iter([{}]), 0, 1, pg_collection=pg_collection)[0]

    assert tokens.shape == (1, 8)
    assert groups and all(group is getattr(pg_collection, group_name) for group in groups)


def test_get_batch_reads_the_tp_cp_group_only_for_a_batch_deduplicated_across_cp(
    train, monkeypatch
):
    """A model collection without tp_cp serves get_batch unless the batch spans TP x CP."""
    groups = _record_broadcasts(monkeypatch)
    pg_collection = ProcessGroupCollection(tp=_Group(0, 2), pp=_Group(0, 1))

    monkeypatch.setattr(train, "get_args", lambda: _batch_args(False, context_parallel_size=2))
    with forbid_global_process_groups():
        train.get_batch(iter([{}]), 0, 1, pg_collection=pg_collection)

    assert groups and all(group is pg_collection.tp for group in groups)

    monkeypatch.setattr(train, "get_args", lambda: _batch_args(True, context_parallel_size=2))
    with pytest.raises(ValueError, match="tp_cp"):
        train.get_batch(iter([{}]), 0, 1, pg_collection=pg_collection)


def _packed_batch_broadcasts(monkeypatch):
    """Replace broadcast_data with one that returns a batch packing two samples of 4 tokens."""
    packed = {
        "tokens": torch.ones(1, 8),
        "labels": torch.ones(1, 9),
        "cu_lengths": torch.tensor([[0, 4, 8]]),
        "cu_lengths_padded": torch.tensor([[0, 4, 8]]),
        "max_lengths": torch.tensor([[4]]),
        "sample_lengths": torch.tensor([[4, 4]]),
    }

    def broadcast_data(keys, data, datatype, tp_group=None):
        return {key: packed.get(key, torch.zeros(1, 1)).to(datatype) for key in keys}

    monkeypatch.setattr(tensor_parallel, "broadcast_data", broadcast_data)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_get_batch_counts_packed_sequence_stats_on_the_model_groups(train, monkeypatch, cp_rank):
    """Only context-parallel rank 0 of the model collection counts a packed batch."""
    from megatron.training.logging import packed_sequence_stats

    args = _batch_args(context_parallel_size=2)
    args.log_packed_sequence_stats = True
    monkeypatch.setattr(train, "get_args", lambda: args)
    monkeypatch.setattr(train, "update_seqlen_stats_from_cu_seqlens", lambda cu_seqlens: None)
    _packed_batch_broadcasts(monkeypatch)
    stats_groups = []

    def update_packed_sequence_stats(sample_lengths, loss_mask, pg_collection=None):
        stats_groups.append(pg_collection)
        packed_sequence_stats.update_packed_sequence_stats(
            sample_lengths, loss_mask, pg_collection=pg_collection
        )

    monkeypatch.setattr(train, "update_packed_sequence_stats", update_packed_sequence_stats)
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    pg_collection = ProcessGroupCollection(tp=_Group(0, 2), cp=_Group(cp_rank, 2), pp=_Group(0, 1))
    packed_sequence_stats._reset_packed_sequence_stats_in_iteration()

    try:
        with warnings.catch_warnings():
            process_groups_config.enable_strict_process_group_fallbacks()
            with forbid_global_process_groups():
                train.get_batch(iter([{}]), 0, 1, pg_collection=pg_collection)
        counted = [
            lengths.tolist()
            for lengths in packed_sequence_stats._packed_sequence_lengths_in_iteration
        ]
    finally:
        packed_sequence_stats._reset_packed_sequence_stats_in_iteration()

    assert stats_groups == [pg_collection]
    assert counted == ([[4.0, 4.0]] if cp_rank == 0 else [])


def test_get_batch_requires_the_cp_group_for_packed_sequence_stats(train, monkeypatch):
    """The statistics read the collection's cp group, so get_batch asks for it up front."""
    args = _batch_args()
    args.log_packed_sequence_stats = True
    monkeypatch.setattr(train, "get_args", lambda: args)
    _record_broadcasts(monkeypatch)

    with pytest.raises(ValueError, match="requires pg_collection to set cp"):
        train.get_batch(
            iter([{}]), 0, 1, pg_collection=ProcessGroupCollection(tp=_Group(0, 2), pp=_Group())
        )


def test_get_batch_reads_data_only_on_the_first_tensor_parallel_rank(train, monkeypatch):
    monkeypatch.setattr(train, "get_args", lambda: _batch_args())
    _record_broadcasts(monkeypatch)
    data_iterator = iter([{}])

    with forbid_global_process_groups():
        train.get_batch(data_iterator, 0, 1, pg_collection=_pg_collection(tp_rank=1))

    assert next(data_iterator, None) == {}


def test_get_batch_skips_the_middle_pipeline_stages_of_the_model(train, monkeypatch):
    monkeypatch.setattr(train, "get_args", lambda: _batch_args())
    groups = _record_broadcasts(monkeypatch)

    with forbid_global_process_groups():
        batch = train.get_batch(iter([{}]), 0, 1, pg_collection=_pg_collection(1, 1, 3))

    assert all(item is None for item in batch)
    assert not groups


def test_loss_func_sums_over_the_model_context_parallel_group(train, monkeypatch):
    monkeypatch.setattr(
        train,
        "get_args",
        lambda: SimpleNamespace(
            no_calculate_per_token_loss=False, use_loss_scaling=False, context_parallel_size=2
        ),
    )
    reduced_over = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: reduced_over.append(group),
    )
    pg_collection = _pg_collection()

    with forbid_global_process_groups():
        train.loss_func(
            torch.ones(1, 4), torch.ones(1, 4), torch.tensor(1), pg_collection=pg_collection
        )

    assert reduced_over == [pg_collection.cp, pg_collection.cp]


def test_forward_step_passes_the_model_groups(train, monkeypatch):
    monkeypatch.setattr(
        train, "get_args", lambda: SimpleNamespace(use_loss_scaling=False, context_parallel_size=1)
    )
    pg_collection = _pg_collection()
    batch_groups = []

    def get_batch(data_iterator, image_token_index, img_seq_len, pg_collection=None):
        batch_groups.append(pg_collection)
        return (None,) * 13

    monkeypatch.setattr(train, "get_batch", get_batch)

    class _Model:
        """A wrapped model as the training loop passes it, which carries the model's groups."""

        def __init__(self):
            llava = SimpleNamespace(image_token_index=0, img_seq_len=1)
            self.module = SimpleNamespace(module=llava, pg_collection=pg_collection)

        def __call__(self, *args, **kwargs):
            return None, None

    _, loss_function = train.forward_step(iter([]), _Model())

    assert batch_groups == [pg_collection]
    assert loss_function.keywords["pg_collection"] is pg_collection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(Utils.world_size % 2 != 0, reason="needs an even world size")
@pytest.mark.parametrize("grid_pg_collection", [{"tp": 2}], indirect=True)
def test_internvit_attention_builds_its_layers_on_the_attention_tp_group(grid_pg_collection):
    pytest.importorskip("transformer_engine")
    from examples.multimodal.nvlm.internvit import get_internvit_layer_spec

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=8,
        qk_layernorm=True,
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
    )
    attention_spec = get_internvit_layer_spec(use_te=False).submodules.self_attention

    # The global grid is not initialized. The guard also fails a warned fallback to it, which
    # would otherwise return None groups.
    with forbid_global_process_groups():
        attention = build_module(
            attention_spec, config=config, layer_number=1, pg_collection=grid_pg_collection
        )

    assert attention.linear_qkv.tp_group is grid_pg_collection.tp
    assert attention.q_layernorm.tp_group is grid_pg_collection.tp
    assert attention.k_layernorm.tp_group is grid_pg_collection.tp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(Utils.world_size % 2 != 0, reason="needs an even world size")
@pytest.mark.parametrize("grid_pg_collection", [{"tp": 2}], indirect=True)
@pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)
def test_internvit_layer_checkpoints_its_norms_over_the_layer_tp_group(
    grid_pg_collection, monkeypatch
):
    pytest.importorskip("transformer_engine")
    from examples.multimodal.nvlm.internvit import get_internvit_layer_spec

    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=8,
        qk_layernorm=True,
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
    )

    # The global grid is not initialized. The guard also fails a warned fallback to it, which
    # would otherwise return None groups.
    with forbid_global_process_groups():
        layer = build_module(
            get_internvit_layer_spec(use_te=False),
            config=config,
            layer_number=1,
            pg_collection=grid_pg_collection,
        )
        sharded_state_dict = layer.sharded_state_dict(
            prefix="layer.", metadata={"dp_cp_group": grid_pg_collection.dp_cp}
        )

    assert layer.input_layernorm.tp_group is grid_pg_collection.tp
    assert layer.pre_mlp_layernorm.tp_group is grid_pg_collection.tp
    # The norm weights are replicated over the layer's tensor-parallel group.
    tp_rank = grid_pg_collection.tp.rank()
    for name in ("input_layernorm", "pre_mlp_layernorm"):
        weight = sharded_state_dict[f"layer.{name}.weight"]
        assert weight.replica_id[1] == tp_rank
