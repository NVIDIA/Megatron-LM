# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint layout of the GDN/KDA ``in_proj.weight`` under GTP weight shards.

The tests call ``_GDNBase.sharded_state_dict`` on a stand-in layer, so they need no process
groups and no GPU.
"""

from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.ssm.gated_delta_net import common as gdn_common
from megatron.core.ssm.utils import _split_tensor_factory

# Section names of a KDA in_proj; the TP-local sizes are arbitrary.
_SECTIONS, _NAMES = [8, 8, 16, 3, 5], ["query", "key", "value", "g", "gate"]
_QK_ROWS, _V_ROWS = 1, 2  # rows of the stand-in conv1d weight


def _in_proj_factory(sections, names, weight, tp_rank, tp_size, gtp_rank=None, gtp_size=1):
    """The ``in_proj.weight`` entry ``_GDNBase.sharded_state_dict`` builds for one rank.

    ``weight`` is the rank's local weight: the TP-local rows, or with ``gtp_rank`` one GTP shard
    of them, carrying the attributes the checkpoint code reads from a GTP parameter.
    """
    if gtp_rank is None:
        rank, size = tp_rank, tp_size
    else:
        rank, size = tp_rank * gtp_size + gtp_rank, tp_size * gtp_size
    patches = [
        patch.object(
            gdn_common,
            "make_sharded_tensors_for_checkpoint",
            side_effect=lambda state_dict, prefix, *args, **kwargs: {
                f"{prefix}{name}": ShardedTensor.from_rank_offsets(
                    f"{prefix}{name}", tensor, (0, tp_rank, tp_size)
                )
                for name, tensor in state_dict.items()
            },
        ),
        patch.object(
            gdn_common,
            "sharded_state_dict_default",
            side_effect=lambda module, prefix, *args, **kwargs: {
                f"{prefix}weight": ShardedTensor.from_rank_offsets(
                    f"{prefix}weight", module.weight, (0, rank, size)
                )
            },
        ),
    ]
    if gtp_rank is not None:
        weight.gtp_remat_size = gtp_size
        weight.group = SimpleNamespace(gtp_rank=gtp_rank)
        patches += [
            patch.object(gdn_common, "HAVE_GTP", True, create=True),
            patch.object(gdn_common, "is_gtp_param", lambda w: hasattr(w, "group"), create=True),
            patch.object(gdn_common, "get_pg_rank", lambda group: tp_rank, create=True),
            patch.object(gdn_common, "get_pg_size", lambda group: tp_size),
            patch("torch.distributed.get_rank", lambda group: group.gtp_rank),
        ]
    in_proj = SimpleNamespace(weight=weight)
    conv1d = SimpleNamespace(
        state_dict=lambda *args, **kwargs: {"weight": torch.zeros(2 * _QK_ROWS + _V_ROWS, 1, 1)}
    )
    layer = SimpleNamespace(
        pg_collection=SimpleNamespace(tp=object()),
        in_proj=in_proj,
        in_proj_dim=sum(sections) * tp_size,
        tp_size=tp_size,
        in_proj_split_sections=tuple(sections),
        in_proj_split_names=list(names),
        conv_bias=False,
        conv_dim_local_tp=2 * _QK_ROWS + _V_ROWS,
        qk_dim_local_tp=_QK_ROWS,
        v_dim_local_tp=_V_ROWS,
        _save_to_state_dict=lambda state_dict, *args, **kwargs: None,
        named_children=lambda: (("in_proj", in_proj), ("conv1d", conv1d)),
    )
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        state_dict = gdn_common._GDNBase.sharded_state_dict(
            layer, prefix="mixer.", metadata={"dp_cp_group": object()}
        )
    return state_dict["mixer.in_proj.weight"]


@pytest.mark.parametrize(
    "gtp_size, local_rows",
    [
        (2, 20),  # the shard boundary cuts a section, no padding
        (3, 20),  # padded: the last shard is padding only
    ],
    ids=["tp2_gtp2", "tp2_gtp3_padded"],
)
def test_gdn_in_proj_gtp_shards_save_the_non_gtp_chunks(gtp_size, local_rows):
    """Each GTP rank saves the parts of the sections it holds; together: the non-GTP chunks."""
    sections, names, tp_size = _SECTIONS, _NAMES, 2
    real_rows = sum(sections)
    pad_rows = gtp_size * local_rows - real_rows

    expected_chunks, actual_chunks = defaultdict(list), defaultdict(list)
    for tp_rank in range(tp_size):
        tp_local = (torch.arange(real_rows * 2).reshape(real_rows, 2) + 1000 * tp_rank).float()
        plain = _in_proj_factory(sections, names, tp_local.clone(), tp_rank, tp_size)
        for chunk in plain.build():
            expected_chunks[chunk.key].append(chunk)

        padded = torch.nn.functional.pad(tp_local, (0, 0, 0, pad_rows))
        for gtp_rank in range(gtp_size):
            shard = padded[gtp_rank * local_rows : (gtp_rank + 1) * local_rows].clone()
            factory = _in_proj_factory(
                sections, names, shard.clone(), tp_rank, tp_size, gtp_rank, gtp_size
            )
            chunks = [chunk for chunk in factory.build() if isinstance(chunk, ShardedTensor)]
            if chunks:
                # Loading puts the saved rows back and zeroes the pad rows of the live shard.
                torch.testing.assert_close(
                    factory.merge_fn([chunk.data for chunk in chunks]), shard, rtol=0, atol=0
                )
            for chunk in chunks:
                actual_chunks[chunk.key].append(chunk)

    assert actual_chunks.keys() == expected_chunks.keys()
    for key, references in expected_chunks.items():
        chunks = sorted(actual_chunks[key], key=lambda chunk: chunk.global_offset)
        global_shape, covered_rows = references[0].global_shape, 0
        for chunk in chunks:
            assert chunk.global_shape == global_shape, (key, chunk.global_shape, global_shape)
            assert chunk.global_offset == (covered_rows, 0), (key, chunk.global_offset)
            covered_rows += chunk.local_shape[0]
        assert covered_rows == global_shape[0], (key, covered_rows, global_shape)
        torch.testing.assert_close(
            torch.cat([chunk.data for chunk in chunks]),
            torch.cat([reference.data for reference in references]),
            rtol=0,
            atol=0,
        )


def test_gdn_in_proj_without_gtp_keeps_the_plain_split():
    """A weight that is not GTP-sharded is split into its sections as before."""
    sections, names = _SECTIONS, _NAMES
    weight = torch.arange(sum(sections) * 2).reshape(sum(sections), 2).float()
    entry = ShardedTensor.from_rank_offsets("mixer.in_proj.weight", weight, (0, 0, 1))
    reference = _split_tensor_factory(entry, list(sections), list(names), 0).build()
    chunks = _in_proj_factory(sections, names, weight, 0, 1).build()
    assert [chunk.key for chunk in chunks] == [chunk.key for chunk in reference]
    for chunk, expected in zip(chunks, reference):
        assert chunk.global_shape == expected.global_shape
        assert chunk.global_offset == expected.global_offset
        torch.testing.assert_close(chunk.data, expected.data, rtol=0, atol=0)
