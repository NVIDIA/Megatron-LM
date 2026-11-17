# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The context-parallel layouts of the n-gram memory through the real stack pieces.

The module resolves the layout of its layer from the layer's config, and the batch split and
HybridStack decide which rows and which packed metadata that layer receives. These tests put
real layer configs, real layers, HybridStack's own layout choice and the rows of
``get_batches_on_this_cp_rank`` together, in one process: a stand-in carries the size and rank
of the context-parallel group, and the token all-gather returns the shards the split produced.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.models.engram import engram as engram_module
from megatron.core.models.engram import hashing as hashing_module
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import select_contiguous, select_zigzag
from megatron.core.models.engram.engram import Engram
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from ._test_utils import write_tokenizer_map

_SEQUENCE = 48
# Attention, MoE, MLP, attention: with linear_cp_layout="contiguous" the attention layers stay
# zigzag and the others run contiguous.
_PATTERN = "*E-*"


class _Group:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _patch_groups(monkeypatch):
    def size(group):
        return group.size() if isinstance(group, _Group) else 1

    def rank(group):
        return group.rank() if isinstance(group, _Group) else 0

    for namespace in (engram_module, hashing_module):
        monkeypatch.setattr(namespace, "get_pg_size", size)
        monkeypatch.setattr(namespace, "get_pg_rank", rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: size(group))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank(group))


def _engram_config(tmp_path, layer_ids, variant="deepseek_v41"):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=layer_ids)
    return EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=layer_ids,
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        variant=variant,
    )


def _stack_config(**overrides):
    values = dict(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.local,
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _real_layer(layer_config, engram_spec, layer_number, cp_group):
    pg = ProcessGroupCollection()
    pg.cp = cp_group
    return TransformerLayer(
        layer_config,
        TransformerLayerSubmodules(engram=engram_spec),
        layer_number=layer_number,
        add_layer_offset=False,
        pg_collection=pg,
    )


def test_real_layers_resolve_the_layout_of_the_stack(tmp_path, monkeypatch):
    """The memory of every real layer resolves the layout HybridStack picked for the layer."""
    _patch_groups(monkeypatch)
    config = _stack_config()
    layer_configs = validate_segment_layers(_PATTERN, config)
    engram_spec = ModuleSpec(
        module=Engram, params={"engram_config": _engram_config(tmp_path, (1, 2, 3, 4))}
    )
    cp = _Group(2, 0)
    layers = [
        _real_layer(layer_config, engram_spec, index + 1, cp)
        for index, layer_config in enumerate(layer_configs)
    ]
    with mock.patch("megatron.core.models.hybrid.hybrid_block.build_module", side_effect=layers):
        stack = HybridStack(
            config=config,
            # build_module is replaced, so the specs only have to be configured.
            submodules=HybridStackSubmodules(
                attention_layer=IdentityOp, mlp_layer=IdentityOp, moe_layer=IdentityOp
            ),
            layer_config_list=layer_configs,
            post_process=False,
            post_layer_norm=False,
            pg_collection=SimpleNamespace(pp=_Group(1, 0), tp=_Group(1, 0), cp=cp, tp_cp=None),
        )
    resolved = tuple(layer.engram.cp_layout for layer in stack.layers)
    assert resolved == stack._cp_layout_manager.layer_layouts
    assert resolved == ("zigzag", "contiguous", "contiguous", "zigzag")
    assert all(
        layer.engram.config is layer_config for layer, layer_config in zip(layers, layer_configs)
    )


def _sft_batch(padded):
    generator = torch.Generator().manual_seed(3)
    tokens = torch.randint(1, 32, (1, _SEQUENCE), generator=generator, dtype=torch.int64)
    positions = torch.tensor([0, 8, 24, _SEQUENCE], dtype=torch.int32)
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(1, _SEQUENCE),
        "position_ids": torch.arange(_SEQUENCE).view(1, _SEQUENCE),
        "max_seqlen": torch.tensor([16], dtype=torch.int32),
    }
    if padded:
        pad = torch.tensor([2, 1, 3], dtype=torch.int32)
        counts = torch.cat((positions[:1], (positions.diff() - pad).cumsum(0))).to(torch.int32)
        batch["cu_seqlens"] = counts.unsqueeze(0)
        batch["cu_seqlens_padded"] = positions.unsqueeze(0)
    else:
        batch["cu_seqlens"] = positions.unsqueeze(0)
    return batch, tokens, positions


@pytest.mark.parametrize("variant", ["deepseek", "deepseek_v41"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_rows_of_the_data_path_through_a_contiguous_layer(
    tmp_path, monkeypatch, cp_size, mixed, padded, variant
):
    """Packed rows as the batch split builds them, looked up on a contiguous layer.

    The split of a row that pads its documents puts their token counts in ``cu_seqlens_q`` and
    their positions in ``cu_seqlens_q_padded``. A zigzag layer of the same stack refuses the
    row, which reaches it in the padded per-sequence zigzag.
    """
    _patch_groups(monkeypatch)
    config = _stack_config()
    moe_config = validate_segment_layers("E", config)[0]
    engram_config = _engram_config(tmp_path, (1,), variant)

    views = []
    for cp_rank in range(cp_size):
        batch, tokens, positions = _sft_batch(padded)
        cp_group = _Group(cp_size, cp_rank)
        cp_batch = get_batches_on_this_cp_rank(
            batch,
            boundary_layout="contiguous",
            dynamic_cp=False,
            cp_group=cp_group,
            additional_layouts={"zigzag"} if mixed else set(),
            sequence_parallel=False,
            tp_group=_Group(1, 0),
            tp_cp_group=None,
            tokens_per_sample=_SEQUENCE,
        )
        assert cp_batch.boundary_layout == "contiguous"
        views.append((cp_group, cp_batch))

    # The metadata of the contiguous view: the token counts of the documents, and their
    # positions when the row pads them.
    params = views[0][1].get_packed_seq_params("contiguous")
    # The data path tags the parameters with the partition mode where they have the field.
    assert getattr(params, "cp_partition_mode", "contiguous") == "contiguous"
    assert torch.equal(params.cu_seqlens_q, batch["cu_seqlens"].squeeze(0))
    if padded:
        assert torch.equal(params.cu_seqlens_q_padded, positions)
    else:
        assert params.cu_seqlens_q_padded is None

    local = _SEQUENCE // cp_size
    shards = [view.get_batch()["tokens"] for _, view in views]
    assert torch.equal(torch.cat(shards, dim=1), tokens)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda parts, tensor, group: [part.copy_(shards[r]) for r, part in enumerate(parts)],
    )

    torch.manual_seed(5)
    reference_module = Engram(
        config=moe_config,
        engram_config=engram_config,
        layer_number=1,
        pg_collection=ProcessGroupCollection(),
    )
    reference = reference_module.lookup(
        tokens, PackedSeqParams(qkv_format="thd", cu_seqlens_q=positions)
    )
    holds_token = torch.arange(_SEQUENCE) < positions[-1]
    if padded:
        counts = batch["cu_seqlens"].squeeze(0).diff().tolist()
        for start, end, count in zip(positions[:-1].tolist(), positions[1:].tolist(), counts):
            holds_token[start + count : end] = False

    for cp_rank, (cp_group, cp_batch) in enumerate(views):
        torch.manual_seed(5)
        pg = ProcessGroupCollection()
        pg.cp = cp_group
        module = Engram(
            config=moe_config, engram_config=engram_config, layer_number=1, pg_collection=pg
        )
        assert module.cp_layout == "contiguous"
        rows = slice(cp_rank * local, (cp_rank + 1) * local)
        result = module.lookup(cp_batch.get_batch()["tokens"], cp_batch.get_packed_seq_params())
        assert torch.equal(result.memory, reference.memory[rows])
        if variant == "deepseek_v41":
            assert torch.equal(result.valid.flatten(), holds_token[rows])

        if mixed:
            # A zigzag attention layer of the same stack refuses the packed row.
            attention_config = validate_segment_layers("*", config)[0]
            zigzag_module = Engram(
                config=attention_config,
                engram_config=engram_config,
                layer_number=1,
                pg_collection=pg,
            )
            assert zigzag_module.cp_layout == "zigzag"
            with pytest.raises(NotImplementedError, match="padded per-sequence zigzag"):
                zigzag_module.lookup(
                    cp_batch.get_batch("zigzag")["tokens"], cp_batch.get_packed_seq_params("zigzag")
                )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_middle_stage_row_in_the_contiguous_layout(tmp_path, monkeypatch, cp_size):
    """The tokens-only batch of a middle pipeline stage, split for both layouts of a stack."""
    _patch_groups(monkeypatch)
    config = _stack_config()
    moe_config, attention_config = validate_segment_layers("E*", config)
    engram_config = _engram_config(tmp_path, (1,), "deepseek")
    generator = torch.Generator().manual_seed(9)
    row = torch.randint(1, 32, (2, _SEQUENCE), generator=generator, dtype=torch.int64)
    keys = ("tokens", "labels", "loss_mask", "position_ids", "cu_seqlens", "max_seqlen")

    views = []
    for cp_rank in range(cp_size):
        batch = dict.fromkeys(keys)
        batch["tokens"] = row
        cp_group = _Group(cp_size, cp_rank)
        views.append(
            (
                cp_group,
                get_batches_on_this_cp_rank(
                    batch,
                    boundary_layout="contiguous",
                    dynamic_cp=False,
                    cp_group=cp_group,
                    additional_layouts={"zigzag"},
                    tokens_per_sample=_SEQUENCE,
                ),
            )
        )
    for cp_rank, (_, view) in enumerate(views):
        assert view.get_packed_seq_params() is None
        assert torch.equal(view.get_batch()["tokens"], select_contiguous(row, cp_size, cp_rank, 1))
        assert torch.equal(
            view.get_batch("zigzag")["tokens"], select_zigzag(row, cp_size, cp_rank, 1)
        )

    torch.manual_seed(5)
    reference = Engram(
        config=moe_config,
        engram_config=engram_config,
        layer_number=1,
        pg_collection=ProcessGroupCollection(),
    ).lookup(row)
    for layout, layer_config, select in (
        ("contiguous", moe_config, select_contiguous),
        ("zigzag", attention_config, select_zigzag),
    ):
        shards = [view.get_batch(layout)["tokens"] for _, view in views]
        monkeypatch.setattr(
            torch.distributed,
            "all_gather",
            lambda parts, tensor, group, shards=shards: [
                part.copy_(shards[r]) for r, part in enumerate(parts)
            ],
        )
        for cp_rank, (cp_group, view) in enumerate(views):
            torch.manual_seed(5)
            pg = ProcessGroupCollection()
            pg.cp = cp_group
            module = Engram(
                config=layer_config, engram_config=engram_config, layer_number=1, pg_collection=pg
            )
            assert module.cp_layout == layout
            result = module.lookup(view.get_batch(layout)["tokens"])
            assert torch.equal(result.memory, select(reference.memory, cp_size, cp_rank, 0))
