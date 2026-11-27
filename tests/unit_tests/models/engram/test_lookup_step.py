# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The lookup step of the n-gram memory, called on its own.

``Engram.forward`` hashes the token IDs, looks the rows up and fuses them into the hidden states.
``Engram.lookup`` is the first two of those, so a caller can run it ahead of the fusion and hand
the result to ``forward``. The two calls must then be the one call, bit for bit, in the output
and in every gradient.

The lookup needs no hidden states, which also makes the context-parallel layouts checkable in
one process: each rank's lookup, with the token all-gather replaced by the row it would return,
must hold the rows of that rank's positions of the full row.
"""

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.engram import engram as engram_module
from megatron.core.models.engram import hashing as hashing_module
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import select_contiguous, select_zigzag
from megatron.core.models.engram.engram import Engram, EngramLookupResult
from megatron.core.models.engram.variants import GATED_FUSION
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import random as tensor_parallel_random

from ._test_utils import make_module_config, make_pg_collection
from ._test_utils import packed_seq_params as tagged_packed_seq_params
from ._test_utils import write_tokenizer_map

_HIDDEN_SIZE = 8
_SEQUENCE = 24
# The gated_conv fusion with and without a convolution history, and the gated fusion.
_VARIANT_AND_KERNEL_SIZE = [("deepseek", 4), ("deepseek", 1), ("deepseek_v41", 4)]


def _module(
    tmp_path,
    variant="deepseek_v41",
    kernel_size=4,
    num_streams=1,
    layer_number=1,
    sequence_parallel=False,
    tp_group=None,
    cp_group=None,
    cp_layout=None,
):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=(1, 2))
    engram_config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1, 2),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=kernel_size,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        variant=variant,
    )
    config = make_module_config(num_streams=num_streams, sequence_parallel=sequence_parallel)
    if cp_layout is not None:
        config.linear_cp_layout = cp_layout
    torch.manual_seed(100 + layer_number)
    module = Engram(
        config=config,
        engram_config=engram_config,
        layer_number=layer_number,
        pg_collection=make_pg_collection(tp=tp_group, cp=cp_group),
    )
    with torch.no_grad():
        if engram_config.variant_spec.fusion == GATED_FUSION:
            module.q_weight.normal_(mean=1.0, std=0.2)
            module.k_weight.normal_(mean=1.0, std=0.2)
        else:
            module.short_conv.weight.normal_(mean=0.0, std=0.05)
    return module


def _inputs(num_streams=1, batch_size=2, sequence=_SEQUENCE, seed=7):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 32, (batch_size, sequence), generator=generator, dtype=torch.int64)
    hidden = torch.randn(
        sequence, batch_size, num_streams * _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    )
    return tokens, hidden


def _packed(*document_ends, **fields):
    cu_seqlens = torch.tensor([0, *document_ends], dtype=torch.int32)
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens, **fields)


def _gradients(module, output, hidden):
    """Backpropagate a fixed functional of ``output`` and return every gradient, then reset."""
    weights = torch.linspace(-1.0, 1.0, output.numel(), dtype=output.dtype, device=output.device)
    (output * weights.view_as(output)).sum().backward()
    gradients = [hidden.grad.clone()] + [
        parameter.grad.clone() for parameter in module.parameters()
    ]
    hidden.grad = None
    module.zero_grad()
    return gradients


@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
@pytest.mark.parametrize("num_streams", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("add_to_hidden_states", [False, True])
def test_lookup_then_forward_is_the_single_call(
    tmp_path, variant, kernel_size, num_streams, packed, add_to_hidden_states
):
    module = _module(tmp_path, variant, kernel_size, num_streams)
    batch_size = 1 if packed else 2
    tokens, hidden = _inputs(num_streams, batch_size)
    hidden.requires_grad_(True)
    # The last document ends five positions before the row does.
    packed_seq_params = _packed(7, _SEQUENCE - 5) if packed else None

    single = module(hidden, tokens, packed_seq_params, add_to_hidden_states=add_to_hidden_states)
    single_gradients = _gradients(module, single, hidden)

    result = module.lookup(tokens, packed_seq_params)
    assert isinstance(result, EngramLookupResult)
    assert result.memory.shape == (_SEQUENCE, batch_size, module.engram_config.total_memory_dim)
    assert result.memory.requires_grad
    assert (result.cu_seqlens is None) == (packed_seq_params is None)
    # The result holds what the fusion needs from the row.
    two_step = module(hidden, add_to_hidden_states=add_to_hidden_states, lookup=result)
    assert torch.equal(two_step, single)
    for actual, expected in zip(_gradients(module, two_step, hidden), single_gradients):
        assert torch.equal(actual, expected)


def test_lookup_result_marks_the_positions_that_hold_a_token(tmp_path):
    tokens, _ = _inputs(batch_size=1)
    packed_seq_params = _packed(7, _SEQUENCE - 5)
    holds_token = torch.arange(_SEQUENCE) < _SEQUENCE - 5

    gated = _module(tmp_path, "deepseek_v41")
    assert gated.lookup(tokens).valid is None
    valid = gated.lookup(tokens, packed_seq_params).valid
    assert valid.shape == (1, _SEQUENCE, 1)
    assert torch.equal(valid.flatten(), holds_token)

    # The gated_conv fusion has no gate to close on them.
    gated_conv = _module(tmp_path, "deepseek")
    assert gated_conv.lookup(tokens, packed_seq_params).valid is None


def test_forward_needs_the_tokens_or_a_lookup_that_fits(tmp_path):
    module = _module(tmp_path)
    tokens, hidden = _inputs()
    with pytest.raises(ValueError, match="needs input_ids, or the result of an earlier lookup"):
        module(hidden)
    result = module.lookup(tokens[:, : _SEQUENCE // 2])
    with pytest.raises(ValueError, match="lookup result does not match the hidden states"):
        module(hidden, lookup=result)
    # The hidden states are still checked first.
    with pytest.raises(ValueError, match="expected hidden width"):
        module(hidden[..., :-1], lookup=module.lookup(tokens))


@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
def test_lookups_of_two_layers_can_precede_both_fusions(tmp_path, variant, kernel_size):
    """Both lookups first, then the two layers: what a caller does that looks up ahead."""
    first = _module(tmp_path, variant, kernel_size, layer_number=1)
    second = _module(tmp_path, variant, kernel_size, layer_number=2)
    tokens, hidden = _inputs()
    hidden.requires_grad_(True)

    def gradients(output):
        weights = torch.linspace(-1.0, 1.0, output.numel(), dtype=output.dtype).view_as(output)
        (output * weights).sum().backward()
        parameters = [*first.parameters(), *second.parameters()]
        collected = [hidden.grad.clone()] + [parameter.grad.clone() for parameter in parameters]
        hidden.grad = None
        first.zero_grad()
        second.zero_grad()
        return collected

    layer_by_layer = second(
        first(hidden, tokens, add_to_hidden_states=True), tokens, add_to_hidden_states=True
    )
    expected = gradients(layer_by_layer)

    first_result, second_result = first.lookup(tokens), second.lookup(tokens)
    ahead = second(
        first(hidden, add_to_hidden_states=True, lookup=first_result),
        add_to_hidden_states=True,
        lookup=second_result,
    )
    assert torch.equal(ahead, layer_by_layer)
    for actual, wanted in zip(gradients(ahead), expected):
        assert torch.equal(actual, wanted)


@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
def test_lookup_ahead_of_a_recomputed_region(tmp_path, monkeypatch, variant, kernel_size):
    """The fusion replays inside a recomputed region; the lookup, made ahead of it, does not.

    The looked-up rows enter the region as one of its tensor arguments, so the region returns
    their gradient to the lookup made outside it, once.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # The checkpoint snapshots the CUDA generators; without a device, only the CPU one.
        device = torch.device("cpu")
        monkeypatch.setattr(
            tensor_parallel_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
        )
        monkeypatch.setattr(
            tensor_parallel_random,
            "_set_all_rng_states",
            lambda cpu_rng_state, *others, **kind: torch.set_rng_state(cpu_rng_state),
        )
    module = _module(tmp_path, variant, kernel_size).to(device)
    tokens, hidden = (tensor.to(device) for tensor in _inputs(batch_size=1))
    hidden.requires_grad_(True)
    packed_seq_params = _packed(7, _SEQUENCE - 5)
    packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q.to(device)
    lookups = []
    table_lookup = module._lookup
    monkeypatch.setattr(
        module, "_lookup", lambda hash_ids: lookups.append(1) or table_lookup(hash_ids)
    )

    plain = module(hidden, tokens, packed_seq_params)
    plain_gradients = _gradients(module, plain, hidden)
    assert len(lookups) == 1

    # The whole call inside the region: the lookup runs without gradients, then again in backward.
    replayed = tensor_parallel.checkpoint(
        lambda states: module(states, tokens, packed_seq_params), False, hidden
    )
    replayed_gradients = _gradients(module, replayed, hidden)
    assert len(lookups) == 3

    result = module.lookup(tokens, packed_seq_params)
    ahead = tensor_parallel.checkpoint(
        lambda states, memory: module(
            states, lookup=EngramLookupResult(memory, result.valid, result.cu_seqlens)
        ),
        False,
        hidden,
        result.memory,
    )
    ahead_gradients = _gradients(module, ahead, hidden)
    assert len(lookups) == 4

    # The region recomputes the fusion on the device, so compare to rounding, not bit for bit.
    tight = dict(rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(ahead, plain, **tight)
    torch.testing.assert_close(replayed, plain, **tight)
    for actual, replay, expected in zip(ahead_gradients, replayed_gradients, plain_gradients):
        torch.testing.assert_close(actual, expected, **tight)
        torch.testing.assert_close(replay, expected, **tight)


class _Group:
    """A stand-in process group that only carries a size and this rank's index."""

    def __init__(self, size, rank):
        self.size, self.rank = size, rank


def _patch_groups(monkeypatch):
    """Let the module read the size and rank of :class:`_Group` stand-ins."""

    def size(group):
        return group.size if isinstance(group, _Group) else 1

    def rank(group):
        return group.rank if isinstance(group, _Group) else 0

    for namespace in (engram_module, hashing_module):
        monkeypatch.setattr(namespace, "get_pg_size", size)
        monkeypatch.setattr(namespace, "get_pg_rank", rank)


def test_lookup_length_follows_sequence_parallelism(tmp_path, monkeypatch):
    """Without hidden states the lookup takes its length from the row and the TP size."""
    _patch_groups(monkeypatch)
    tokens, _ = _inputs()
    reference = _module(tmp_path).lookup(tokens).memory
    tp_size = 2
    for tp_rank in range(tp_size):
        module = _module(tmp_path, sequence_parallel=True, tp_group=_Group(tp_size, tp_rank))
        local = _SEQUENCE // tp_size
        memory = module.lookup(tokens).memory
        assert torch.equal(memory, reference[tp_rank * local : (tp_rank + 1) * local])
        assert torch.equal(memory, module.lookup(tokens, local_sequence_length=local).memory)
    # Not sliced without sequence parallelism, whatever the tensor-parallel size.
    module = _module(tmp_path, tp_group=_Group(tp_size, 1))
    assert torch.equal(module.lookup(tokens).memory, reference)


def _document_rows(layout):
    """Positions and token counts of the documents of a packed row of ``_SEQUENCE`` positions."""
    positions = torch.tensor([0, 7, 19, _SEQUENCE - 3], dtype=torch.int32)
    if layout == "packed":
        return positions, None
    # The documents hold 2, 1 and 1 tokens less than the positions they occupy.
    padding = torch.tensor([2, 1, 1], dtype=torch.int32)
    return positions, torch.cat((positions[:1], (positions.diff() - padding).cumsum(0)))


@pytest.mark.parametrize("variant", ["deepseek", "deepseek_v41"])
@pytest.mark.parametrize("layout", ["unpacked", "packed", "packed_padded_documents"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_contiguous_cp_lookup_holds_the_rows_of_each_rank(
    tmp_path, monkeypatch, cp_size, layout, variant
):
    """Each rank's lookup in the contiguous layout against the lookup of the full row.

    A packed row is split like an unpacked one, so its documents straddle the ranks, and a hash
    window reads the tokens of the previous rank. A row that pads its documents carries their
    token counts in ``cu_seqlens_q`` and their positions in ``cu_seqlens_q_padded``; the windows
    follow the positions, and the positions of the padding hold no token.
    """
    _patch_groups(monkeypatch)
    batch_size = 2 if layout == "unpacked" else 1
    tokens, _ = _inputs(batch_size=batch_size)
    local = _SEQUENCE // cp_size
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda parts, tensor, group: [
            part.copy_(tokens[:, rank * local : (rank + 1) * local])
            for rank, part in enumerate(parts)
        ],
    )

    packed_seq_params = reference_packed_seq_params = None
    holds_token = torch.ones(_SEQUENCE, dtype=torch.bool)
    if layout != "unpacked":
        positions, token_counts = _document_rows(layout)
        holds_token[positions[-1] :] = False
        if token_counts is not None:
            for start, count in zip(positions[:-1].tolist(), token_counts.diff().tolist()):
                holds_token[start + count : positions[positions > start][0]] = False
        packed_seq_params = tagged_packed_seq_params(
            qkv_format="thd",
            cu_seqlens_q=positions if token_counts is None else token_counts,
            cu_seqlens_q_padded=None if token_counts is None else positions,
            cp_partition_mode="contiguous",
        )
        reference_packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=positions)
    reference = _module(tmp_path, variant).lookup(tokens, reference_packed_seq_params)

    for cp_rank in range(cp_size):
        module = _module(
            tmp_path, variant, cp_group=_Group(cp_size, cp_rank), cp_layout="contiguous"
        )
        assert module.cp_layout == "contiguous"
        rows = slice(cp_rank * local, (cp_rank + 1) * local)
        result = module.lookup(tokens[:, rows], packed_seq_params)
        assert torch.equal(result.memory, reference.memory[rows])
        if variant == "deepseek_v41" and layout != "unpacked":
            assert torch.equal(result.valid.flatten(), holds_token[rows])
        else:
            assert result.valid is None


@pytest.mark.parametrize("layout", ["unpacked", "packed"])
def test_zigzag_cp_lookup_holds_the_rows_of_each_rank(tmp_path, monkeypatch, layout):
    """The zigzag layout through the same stand-ins, so the two layouts share one check."""
    from megatron.core.models.engram.cp_layout import thd_partition_index

    _patch_groups(monkeypatch)
    cp_size = 2
    batch_size = 2 if layout == "unpacked" else 1
    tokens, _ = _inputs(batch_size=batch_size)
    packed_seq_params = _packed(8, _SEQUENCE) if layout == "packed" else None

    def shard(tensor, cp_rank, sequence_dim):
        if packed_seq_params is None:
            return select_zigzag(tensor, cp_size, cp_rank, sequence_dim)
        index = thd_partition_index(packed_seq_params.cu_seqlens_q, cp_size, cp_rank)
        return tensor.index_select(sequence_dim, index)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda parts, tensor, group: [
            part.copy_(shard(tokens, rank, 1)) for rank, part in enumerate(parts)
        ],
    )
    reference = _module(tmp_path).lookup(tokens, packed_seq_params)
    for cp_rank in range(cp_size):
        module = _module(tmp_path, cp_group=_Group(cp_size, cp_rank))
        assert module.cp_layout == "zigzag"
        result = module.lookup(shard(tokens, cp_rank, 1), packed_seq_params)
        assert torch.equal(result.memory, shard(reference.memory, cp_rank, 0))
        # The contiguous slice of the same rank holds other rows.
        assert not torch.equal(
            result.memory, select_contiguous(reference.memory, cp_size, cp_rank, 0)
        )
