# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TP/SP correctness for Engram.

Runs under the CI unit-test launcher (torch.distributed.run at 8 ranks) as well as local
torchrun with 2 or 4 ranks: each case treats the first ``tp_size`` ranks as one
tensor-parallel group with sequence parallelism enabled (EP=1, expert-DP=1). Each rank
forwards its contiguous sequence slice; outputs must match the full-sequence reference
(including the convolution halo across the slice boundary), and gradients of the replicated
Engram parameters must match after the TP sum that
``_allreduce_non_tensor_model_parallel_grads`` performs during gradient finalization.

The TP x CP cases (four ranks or more) add context parallelism on top: each rank forwards its
sequence-parallel slice of its CP shard, and the gradients must match after the TP sum and the
CP sum that DDP performs over dp_cp. The shard is the zigzag one (per-document zigzag for packed
rows) or the contiguous one, in which a packed row is split like an unpacked row and its
documents straddle ranks. The CP-only cases run the contiguous layout at CP2 and CP4.

The ``deepseek_v41`` cases run the same comparisons for the fusion without the convolution,
which exchanges no halo, and check that the positions after the last document boundary of a
packed row keep a closed gate on the rank that holds them. In the contiguous layout a packed row
also carries the token count of each document, and the gate stays closed on their padding.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import (
    select_contiguous,
    select_zigzag,
    thd_partition_index,
)
from megatron.core.models.engram.engram import Engram
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils

from ._test_utils import make_module_config, make_pg_collection
from ._test_utils import packed_seq_params as tagged_packed_seq_params
from ._test_utils import write_tokenizer_map

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

# Must be at least (kernel_size - 1) * max_ngram_order = 9 so the SP convolution halo is
# a single-hop exchange; the runtime guard in Engram enforces the same bound.
_SEQUENCE_PER_RANK = 12
_BATCH_SIZE = 2
_HIDDEN_SIZE = 8


def _build_module(
    tmp_path,
    tp_group,
    sequence_parallel,
    cp_group=None,
    kernel_size=4,
    variant="deepseek",
    cp_layout="zigzag",
):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=(1,))
    engram_config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=kernel_size,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        variant=variant,
    )
    torch.manual_seed(2026)
    config = make_module_config(sequence_parallel=sequence_parallel)
    # The layout the layer of this module runs in under context parallelism.
    config.linear_cp_layout = cp_layout
    module = Engram(
        config=config,
        engram_config=engram_config,
        layer_number=1,
        pg_collection=make_pg_collection(tp=tp_group, cp=cp_group),
    )
    with torch.no_grad():
        if engram_config.variant_spec.has_short_convolution:
            # A nonzero convolution branch so the halo exchange and its gradients matter.
            module.short_conv.weight.normal_(mean=0.0, std=0.05)
        else:
            # Gate weights away from their initial ones, so their gradients matter.
            module.q_weight.normal_(mean=1.0, std=0.2)
            module.k_weight.normal_(mean=1.0, std=0.2)
    return module.to(torch.cuda.current_device())


# A one-tap kernel has no left context, so no halo may be exchanged; the deepseek_v41 fusion
# has no convolution at all.
_VARIANT_AND_KERNEL_SIZE = [("deepseek", 4), ("deepseek", 1), ("deepseek_v41", 4)]


@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_sp_sliced_gradients_match_full_sequence_reference(
    tmp_path_factory, tp_size, variant, kernel_size
):
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size < tp_size:
        pytest.skip(f"requires at least {tp_size} ranks")
    # Collective on the default group: every rank must participate in group creation.
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))
    if rank >= tp_size:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    sp_module = _build_module(
        tmp_path, tp_group, sequence_parallel=True, kernel_size=kernel_size, variant=variant
    )
    reference_module = _build_module(
        tmp_path, tp_group, sequence_parallel=False, kernel_size=kernel_size, variant=variant
    )
    for sp_param, ref_param in zip(sp_module.parameters(), reference_module.parameters()):
        torch.testing.assert_close(sp_param, ref_param)

    # The constructors route the sparse tables to the dedicated Engram reduction path
    # (expert DDP bucket via allreduce=False) and mark every parameter for the SP TP-sum.
    for name, parameter in sp_module.named_parameters():
        assert getattr(parameter, "sequence_parallel", False), name
        if getattr(parameter, "is_engram_embedding", False):
            assert parameter.allreduce is False, name
        else:
            assert getattr(parameter, "allreduce", True), name

    full_sequence = _SEQUENCE_PER_RANK * tp_size
    generator = torch.Generator(device="cpu").manual_seed(11)
    tokens = torch.randint(
        0, 32, (_BATCH_SIZE, full_sequence), generator=generator, dtype=torch.int64
    ).to(device)
    hidden_full = torch.randn(
        full_sequence, _BATCH_SIZE, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)
    projection = torch.randn(
        full_sequence, _BATCH_SIZE, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)

    sequence_slice = slice(rank * _SEQUENCE_PER_RANK, (rank + 1) * _SEQUENCE_PER_RANK)
    local_output = sp_module(hidden_full[sequence_slice], tokens)
    reference_output = reference_module(hidden_full, tokens)
    # Covers the convolution halo: positions near the slice start depend on the previous
    # rank's values, which only the halo exchange can provide.
    torch.testing.assert_close(local_output, reference_output[sequence_slice])

    (local_output * projection[sequence_slice]).sum().backward()
    (reference_output * projection).sum().backward()

    # Gradient finalization: the TP sum for sequence-parallel parameters, including the
    # dedicated unflattened path for the sparse tables.
    model_chunk = sp_module
    model_chunk.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    finalize_config = SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
    _allreduce_non_tensor_model_parallel_grads([model_chunk], finalize_config, tp_group)

    for (name, sp_param), ref_param in zip(
        sp_module.named_parameters(), reference_module.parameters()
    ):
        assert sp_param.grad is not None, name
        torch.testing.assert_close(sp_param.grad, ref_param.grad, msg=name)


def _packed_row(layout, cp_layout, full_sequence):
    """Return the row length and document ends of a packed row for the given layouts."""
    if cp_layout == "zigzag":
        if layout == "packed_short_documents":
            return 16, [4, 8, 12, 16]
        # Two documents whose lengths are multiples of 2 * cp_size.
        return full_sequence, [full_sequence // 3 // 4 * 4, full_sequence]
    # The contiguous layout splits the row, not the documents: document lengths are free, the
    # documents straddle the rank boundaries, and the row may end after the last document.
    if layout == "packed_short_documents":
        full_sequence //= 2
        return full_sequence, list(range(5, full_sequence - 2, 5))
    return full_sequence, [full_sequence // 3 - 1, 3 * full_sequence // 4 - 2, full_sequence - 6]


# Unpacked rows, packed (THD) rows of long documents, and a packed row of short documents. In
# the zigzag layout the first positions of the last one, whose left context is zeros, also sit
# on TP rank 1, so a halo taken from the restored row would show.
@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
@pytest.mark.parametrize("layout", ["unpacked", "packed", "packed_short_documents"])
@pytest.mark.parametrize("cp_layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("two_step", [False, True])
def test_tp_and_cp_gradients_match_full_sequence_reference(
    tmp_path_factory, two_step, cp_layout, layout, variant, kernel_size
):
    """TP2 x CP2 with sequence parallelism, for unpacked and packed (THD) rows.

    The CP paths convolve the whole CP-local sequence on every TP rank; each rank must keep and
    backpropagate only its own slice, or the TP sum counts the dense gradients TP times. A packed
    zigzag row is restored whole, so its convolution starts from zeros, not from a
    sequence-parallel halo; a contiguous slice takes its left context from the previous CP rank.

    ``two_step`` looks the rows up first and hands the result to the module, which must take
    the length of the sequence-parallel slice and the boundaries of the row from the lookup.
    """
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() < 4:
        pytest.skip("requires at least 4 ranks")
    tp_size = cp_size = 2
    # Ranks 0-3 form a TP-fastest 2 x 2 grid. Group creation is collective on the default
    # group, so every rank creates every group.
    tp_groups = [torch.distributed.new_group(ranks=[0, 1]), torch.distributed.new_group([2, 3])]
    cp_groups = [torch.distributed.new_group(ranks=[0, 2]), torch.distributed.new_group([1, 3])]
    if rank >= tp_size * cp_size:
        return
    tp_rank, cp_rank = rank % tp_size, rank // tp_size
    tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    module = _build_module(
        tmp_path,
        tp_group,
        sequence_parallel=True,
        cp_group=cp_group,
        kernel_size=kernel_size,
        variant=variant,
        cp_layout=cp_layout,
    )
    reference_module = _build_module(
        tmp_path, None, sequence_parallel=False, kernel_size=kernel_size, variant=variant
    )
    for parameter, ref_param in zip(module.parameters(), reference_module.parameters()):
        torch.testing.assert_close(parameter, ref_param)

    packed = layout != "unpacked"
    # Unpacked rows need zigzag chunks at least as long as the convolution history.
    full_sequence = 4 * _SEQUENCE_PER_RANK * tp_size
    if packed:
        full_sequence, document_ends = _packed_row(layout, cp_layout, full_sequence)
    batch_size = 1 if packed else _BATCH_SIZE
    generator = torch.Generator(device="cpu").manual_seed(12)
    tokens = torch.randint(
        0, 32, (batch_size, full_sequence), generator=generator, dtype=torch.int64
    ).to(device)
    hidden_full = torch.randn(
        full_sequence, batch_size, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)
    projection = torch.randn(
        full_sequence, batch_size, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)

    packed_seq_params = reference_packed_seq_params = None
    if packed:
        cu_seqlens = torch.tensor([0, *document_ends], dtype=torch.int32, device=device)
        packed_seq_params = tagged_packed_seq_params(
            qkv_format="thd", cu_seqlens_q=cu_seqlens, cp_partition_mode=cp_layout
        )
        reference_packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens)
    if cp_layout == "contiguous":

        def cp_shard(tensor, sequence_dim):
            return select_contiguous(tensor, cp_size, cp_rank, sequence_dim=sequence_dim)

    elif packed:
        cp_index = thd_partition_index(cu_seqlens, cp_size, cp_rank)

        def cp_shard(tensor, sequence_dim):
            return tensor.index_select(sequence_dim, cp_index)

    else:

        def cp_shard(tensor, sequence_dim):
            return select_zigzag(tensor, cp_size, cp_rank, sequence_dim=sequence_dim)

    local_length = full_sequence // cp_size // tp_size
    sp_slice = slice(tp_rank * local_length, (tp_rank + 1) * local_length)
    if two_step:
        result = module.lookup(cp_shard(tokens, 1), packed_seq_params)
        local_output = module(cp_shard(hidden_full, 0)[sp_slice], lookup=result)
    else:
        local_output = module(
            cp_shard(hidden_full, 0)[sp_slice], cp_shard(tokens, 1), packed_seq_params
        )
    reference_output = reference_module(hidden_full, tokens, reference_packed_seq_params)

    (local_output * cp_shard(projection, 0)[sp_slice]).sum().backward()
    (reference_output * projection).sum().backward()

    # Gradient finalization: the TP sum for sequence-parallel parameters, then the CP sum that
    # DDP's reduction over dp_cp performs.
    module.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    finalize_config = SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
    _allreduce_non_tensor_model_parallel_grads([module], finalize_config, tp_group)
    for parameter in module.parameters():
        torch.distributed.all_reduce(parameter.grad, group=cp_group)

    # Compared after the last collective, so a rank that fails leaves no other rank in one.
    torch.testing.assert_close(local_output, cp_shard(reference_output, 0)[sp_slice])
    for (name, parameter), ref_param in zip(
        module.named_parameters(), reference_module.parameters()
    ):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref_param.grad, msg=name)


# Unpacked rows; a packed row whose documents straddle the ranks and end before the row does; and
# a packed row that pads its documents, which carries their token counts in cu_seqlens_q and
# their positions in cu_seqlens_q_padded.
@pytest.mark.parametrize("variant,kernel_size", _VARIANT_AND_KERNEL_SIZE)
@pytest.mark.parametrize("layout", ["unpacked", "packed", "packed_padded_documents"])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("two_step", [False, True])
def test_contiguous_cp_gradients_match_full_sequence_reference(
    tmp_path_factory, two_step, cp_size, layout, variant, kernel_size
):
    """Context parallelism alone in the contiguous layout, against the full-sequence module.

    A hash window and the convolution reach into the previous rank's slice, and the reference
    sees the positions of the documents only. So it stands for the padded documents too, once
    the gate of the ``gated`` fusion is closed on their padding, which the reference cannot know.
    """
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() < cp_size:
        pytest.skip(f"requires at least {cp_size} ranks")
    cp_group = torch.distributed.new_group(ranks=list(range(cp_size)))
    if rank >= cp_size:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    module = _build_module(
        tmp_path,
        None,
        sequence_parallel=False,
        cp_group=cp_group,
        kernel_size=kernel_size,
        variant=variant,
        cp_layout="contiguous",
    )
    reference_module = _build_module(
        tmp_path, None, sequence_parallel=False, kernel_size=kernel_size, variant=variant
    )
    for parameter, ref_param in zip(module.parameters(), reference_module.parameters()):
        torch.testing.assert_close(parameter, ref_param)

    # Every slice holds the convolution history of 9 positions.
    full_sequence = _SEQUENCE_PER_RANK * cp_size
    batch_size = 1 if layout != "unpacked" else _BATCH_SIZE
    generator = torch.Generator(device="cpu").manual_seed(14)
    tokens = torch.randint(
        0, 32, (batch_size, full_sequence), generator=generator, dtype=torch.int64
    ).to(device)
    hidden_full = torch.randn(
        full_sequence, batch_size, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)
    projection = torch.randn(
        full_sequence, batch_size, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)

    packed_seq_params = reference_packed_seq_params = None
    holds_token = torch.ones(full_sequence, dtype=torch.bool, device=device)
    if layout != "unpacked":
        # Three documents; the second straddles the first rank boundary, and the last ends
        # three positions before the row does.
        document_ends = [_SEQUENCE_PER_RANK - 5, _SEQUENCE_PER_RANK + 7, full_sequence - 3]
        positions = torch.tensor([0, *document_ends], dtype=torch.int32, device=device)
        token_counts = positions
        holds_token[full_sequence - 3 :] = False
        if layout == "packed_padded_documents":
            # The documents hold 2, 1 and 1 tokens less than the positions they occupy.
            padding = torch.tensor([2, 1, 1], dtype=torch.int32, device=device)
            token_counts = torch.cat((positions[:1], (positions.diff() - padding).cumsum(0)))
            for end, pad in zip(document_ends, padding.tolist()):
                holds_token[end - pad : end] = False
        packed_seq_params = tagged_packed_seq_params(
            qkv_format="thd",
            cu_seqlens_q=token_counts,
            cu_seqlens_q_padded=positions if layout == "packed_padded_documents" else None,
            cp_partition_mode="contiguous",
        )
        reference_packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=positions)

    cp_rank = rank
    local_slice = slice(cp_rank * _SEQUENCE_PER_RANK, (cp_rank + 1) * _SEQUENCE_PER_RANK)
    if two_step:
        result = module.lookup(tokens[:, local_slice], packed_seq_params)
        local_output = module(hidden_full[local_slice], lookup=result)
    else:
        local_output = module(hidden_full[local_slice], tokens[:, local_slice], packed_seq_params)
    reference_output = reference_module(hidden_full, tokens, reference_packed_seq_params)
    if variant == "deepseek_v41":
        # The residual of the gated fusion is zero wherever the gate is closed.
        reference_output = reference_output * holds_token.view(-1, 1, 1)

    (local_output * projection[local_slice]).sum().backward()
    (reference_output * projection).sum().backward()
    # The CP sum that DDP's reduction over dp_cp performs.
    for parameter in module.parameters():
        torch.distributed.all_reduce(parameter.grad, group=cp_group)

    # Compared after the last collective, so a rank that fails leaves no other rank in one.
    if variant == "deepseek_v41" and layout != "unpacked":
        closed = ~holds_token[local_slice]
        assert torch.count_nonzero(local_output[closed]) == 0
        assert torch.count_nonzero(local_output[~closed]) > 0
    torch.testing.assert_close(local_output, reference_output[local_slice])
    for (name, parameter), ref_param in zip(
        module.named_parameters(), reference_module.parameters()
    ):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref_param.grad, msg=name)


def test_packed_cp_rows_must_follow_the_layout_of_the_layer(tmp_path_factory):
    """A packed row in another partition than the layer's, or the padded zigzag of a mixed stack."""
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    cp_size = 2
    cp_group = torch.distributed.new_group(ranks=list(range(cp_size)))
    if rank >= cp_size:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    full_sequence = _SEQUENCE_PER_RANK * cp_size
    tokens = torch.zeros((1, _SEQUENCE_PER_RANK), dtype=torch.int64, device=device)
    hidden = torch.zeros((_SEQUENCE_PER_RANK, 1, _HIDDEN_SIZE), dtype=torch.float64, device=device)
    cu_seqlens = torch.tensor([0, full_sequence], dtype=torch.int32, device=device)

    def packed(cp_partition_mode):
        return tagged_packed_seq_params(
            qkv_format="thd", cu_seqlens_q=cu_seqlens, cp_partition_mode=cp_partition_mode
        )

    # Both checks raise before any collective, on every rank alike. A one-tap kernel keeps the
    # short zigzag chunks of the unpacked call below free of a convolution history.
    for cp_layout, other in (("zigzag", "contiguous"), ("contiguous", "zigzag")):
        module = _build_module(
            tmp_path,
            None,
            sequence_parallel=False,
            cp_group=cp_group,
            kernel_size=1,
            cp_layout=cp_layout,
        )
        with pytest.raises(ValueError, match=f"partitioned in the {other} layout"):
            module(hidden, tokens, packed(other))

    # A zigzag layer of a stack that also has contiguous layers gets the padded per-sequence
    # zigzag of a packed row, which the memory does not follow. Unpacked rows are fine.
    module = _build_module(
        tmp_path, None, sequence_parallel=False, cp_group=cp_group, kernel_size=1
    )
    module.config.attention_cp_layout = "contiguous"
    mixed_module = Engram(
        config=module.config,
        engram_config=module.engram_config,
        layer_number=1,
        pg_collection=module.pg_collection,
    ).to(device)
    assert mixed_module.cp_layout == "zigzag"
    with pytest.raises(NotImplementedError, match="padded per-sequence zigzag"):
        mixed_module(hidden, tokens, packed("zigzag"))
    mixed_module(hidden, tokens)


def test_v41_gate_stays_closed_after_the_last_boundary_under_sequence_parallelism(tmp_path_factory):
    """A packed row whose last document ends inside the second sequence-parallel rank's slice.

    Each rank selects its slice of the row's valid positions, so only the rank that holds the
    positions after the last boundary closes gates, and both ranks match the full-sequence
    module.
    """
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    tp_size = 2
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))
    if rank >= tp_size:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    sp_module = _build_module(tmp_path, tp_group, sequence_parallel=True, variant="deepseek_v41")
    reference_module = _build_module(
        tmp_path, None, sequence_parallel=False, variant="deepseek_v41"
    )

    full_sequence = _SEQUENCE_PER_RANK * tp_size
    # The two documents end at position 18 of 24, in the middle of rank 1's slice.
    valid_length = full_sequence - _SEQUENCE_PER_RANK // 2
    cu_seqlens = torch.tensor([0, 8, valid_length], dtype=torch.int32, device=device)
    packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens)
    generator = torch.Generator(device="cpu").manual_seed(13)
    tokens = torch.randint(0, 32, (1, full_sequence), generator=generator, dtype=torch.int64).to(
        device
    )
    hidden_full = torch.randn(
        full_sequence, 1, _HIDDEN_SIZE, generator=generator, dtype=torch.float64
    ).to(device)

    sequence_slice = slice(rank * _SEQUENCE_PER_RANK, (rank + 1) * _SEQUENCE_PER_RANK)
    local_output = sp_module(hidden_full[sequence_slice], tokens, packed_seq_params)
    reference_output = reference_module(hidden_full, tokens, packed_seq_params)
    torch.testing.assert_close(local_output, reference_output[sequence_slice])

    assert torch.count_nonzero(reference_output[valid_length:]) == 0
    assert torch.count_nonzero(reference_output[:valid_length]) > 0
    closed = torch.arange(full_sequence, device=device)[sequence_slice] >= valid_length
    assert torch.count_nonzero(local_output[closed]) == 0
    assert torch.count_nonzero(local_output[~closed]) > 0
