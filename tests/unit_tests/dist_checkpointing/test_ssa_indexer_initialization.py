# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loading simplified sparse attention checkpoints, including phase 1 from one without indexers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.training.checkpointing import (
    CheckpointType,
    keep_new_ssa_indexers,
    load_args_from_checkpoint,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_BACKBONE = "decoder.layers.0.self_attention.linear_qkv.weight"
_INDEXER = "decoder.layers.0.self_attention.core_attention.indexer.linear_q.weight"


def _tensor(key, value):
    return ShardedTensor.from_rank_offsets(key, value, replica_id=Utils.rank)


def test_only_indexer_entries_missing_from_the_checkpoint_stay_local():
    state = {
        "backbone": _tensor(_BACKBONE, torch.ones(2)),
        "new_indexer": _tensor(_INDEXER, torch.ones(2)),
        "saved_indexer": _tensor(_INDEXER.replace("layers.0", "layers.1"), torch.ones(2)),
    }
    saved = {_BACKBONE, _INDEXER.replace("layers.0", "layers.1")}
    assert keep_new_ssa_indexers(state, saved) == 1
    assert isinstance(state["new_indexer"], LocalNonpersistentObject)
    assert isinstance(state["backbone"], ShardedTensor)
    assert isinstance(state["saved_indexer"], ShardedTensor)


def test_dense_checkpoint_loads_backbone_and_keeps_initialized_indexer(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(1, 1)
    try:
        with TempNamedDir(tmp_path_dist_ckpt / "ssa_indexer_init", sync=True) as ckpt_dir:
            save({"backbone": _tensor(_BACKBONE, torch.full((2, 4), 3.0))}, ckpt_dir)
            torch.distributed.barrier()

            indexer = torch.full((2, 4), 5.0)
            request = {
                "backbone": _tensor(_BACKBONE, torch.zeros(2, 4)),
                "indexer": _tensor(_INDEXER, indexer),
            }
            saved = {entry.key for entry in load_sharded_metadata(ckpt_dir).values()}
            assert keep_new_ssa_indexers(request, saved) == 1
            loaded = load(request, ckpt_dir)
            torch.testing.assert_close(loaded["backbone"], torch.full((2, 4), 3.0))
            torch.testing.assert_close(loaded["indexer"], indexer)
    finally:
        Utils.destroy_model_parallel()


_SSA_SPEC = [
    "megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_layer_specs",
    "ssa_stack_spec",
]


def _load_args(checkpoint_args, **args):
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
        **args,
    )
    with patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(
            {"args": checkpoint_args, "iteration": 2},
            "checkpoint",
            False,
            CheckpointType.LEGACY,
        ),
    ):
        restored_args, _ = load_args_from_checkpoint(args)
    return restored_args


def test_checkpoint_args_restore_the_ssa_variant_and_topk():
    checkpoint_args = SimpleNamespace(
        experimental_attention_variant="ssa", dsa_indexer_topk=1024, spec=_SSA_SPEC
    )
    args = _load_args(
        checkpoint_args, experimental_attention_variant=None, dsa_indexer_topk=None, spec=None
    )
    assert args.experimental_attention_variant == "ssa"
    assert args.dsa_indexer_topk == 1024
    assert args.spec == _SSA_SPEC


@pytest.mark.parametrize("indexer_only", [True, False])
def test_checkpoint_args_keep_the_ssa_spec_only_for_phase_1_from_dense(indexer_only):
    dense_spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    args = _load_args(
        SimpleNamespace(spec=dense_spec),
        experimental_attention_variant="ssa",
        ssa_train_indexer_only=indexer_only,
        dsa_indexer_topk=512,
        spec=_SSA_SPEC,
    )
    assert args.spec == (_SSA_SPEC if indexer_only else dense_spec)
    assert args.experimental_attention_variant == "ssa"
    assert args.dsa_indexer_topk == 512
