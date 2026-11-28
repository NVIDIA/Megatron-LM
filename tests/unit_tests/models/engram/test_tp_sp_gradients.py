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
sequence-parallel slice of its zigzag (or, for packed rows, per-document zigzag) CP shard, and
the gradients must match after the TP sum and the CP sum that DDP performs over dp_cp.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import select_zigzag, thd_partition_index
from megatron.core.models.engram.engram import Engram
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils

from ._test_utils import make_module_config, make_pg_collection, write_tokenizer_map

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

# Must be at least (kernel_size - 1) * max_ngram_order = 9 so the SP convolution halo is
# a single-hop exchange; the runtime guard in Engram enforces the same bound.
_SEQUENCE_PER_RANK = 12
_BATCH_SIZE = 2
_HIDDEN_SIZE = 8


def _build_module(tmp_path, tp_group, sequence_parallel, cp_group=None, kernel_size=4):
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
    )
    torch.manual_seed(2026)
    module = Engram(
        config=make_module_config(sequence_parallel=sequence_parallel),
        engram_config=engram_config,
        layer_number=1,
        pg_collection=make_pg_collection(tp=tp_group, cp=cp_group),
    )
    with torch.no_grad():
        # A nonzero convolution branch so the halo exchange and its gradients matter.
        module.short_conv.weight.normal_(mean=0.0, std=0.05)
    return module.to(torch.cuda.current_device())


# A one-tap kernel has no left context, so no halo may be exchanged.
@pytest.mark.parametrize("kernel_size", [4, 1])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_sp_sliced_gradients_match_full_sequence_reference(tmp_path_factory, tp_size, kernel_size):
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

    sp_module = _build_module(tmp_path, tp_group, sequence_parallel=True, kernel_size=kernel_size)
    reference_module = _build_module(
        tmp_path, tp_group, sequence_parallel=False, kernel_size=kernel_size
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


# Unpacked rows, packed (THD) rows of two long documents, and a packed row of 4-token documents.
# In the last one the first positions of the row, whose left context is zeros, also sit on TP
# rank 1, so a halo taken from the restored row would show.
@pytest.mark.parametrize("kernel_size", [4, 1])
@pytest.mark.parametrize("layout", ["unpacked", "packed", "packed_short_documents"])
def test_tp_and_cp_gradients_match_full_sequence_reference(tmp_path_factory, layout, kernel_size):
    """TP2 x CP2 with sequence parallelism, for unpacked and packed (THD) rows.

    The CP paths convolve the whole CP-local sequence on every TP rank; each rank must keep and
    backpropagate only its own slice, or the TP sum counts the dense gradients TP times. A packed
    row is restored whole, so its convolution starts from zeros, not from a sequence-parallel halo.
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
        tmp_path, tp_group, sequence_parallel=True, cp_group=cp_group, kernel_size=kernel_size
    )
    reference_module = _build_module(
        tmp_path, None, sequence_parallel=False, kernel_size=kernel_size
    )
    for parameter, ref_param in zip(module.parameters(), reference_module.parameters()):
        torch.testing.assert_close(parameter, ref_param)

    packed = layout != "unpacked"
    if layout == "packed_short_documents":
        full_sequence, document_ends = 16, [4, 8, 12, 16]
    else:
        # Unpacked rows need zigzag chunks at least as long as the convolution history. The two
        # documents of the packed row have lengths that are multiples of 2 * cp_size.
        full_sequence = 4 * _SEQUENCE_PER_RANK * tp_size
        document_ends = [full_sequence // 3 // 4 * 4, full_sequence]
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

    if packed:
        cu_seqlens = torch.tensor([0, *document_ends], dtype=torch.int32, device=device)
        packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens)
        cp_index = thd_partition_index(cu_seqlens, cp_size, cp_rank)

        def cp_shard(tensor, sequence_dim):
            return tensor.index_select(sequence_dim, cp_index)

    else:
        packed_seq_params = None

        def cp_shard(tensor, sequence_dim):
            return select_zigzag(tensor, cp_size, cp_rank, sequence_dim=sequence_dim)

    local_length = full_sequence // cp_size // tp_size
    sp_slice = slice(tp_rank * local_length, (tp_rank + 1) * local_length)
    local_output = module(
        cp_shard(hidden_full, 0)[sp_slice], cp_shard(tokens, 1), packed_seq_params
    )
    reference_output = reference_module(hidden_full, tokens, packed_seq_params)
    torch.testing.assert_close(local_output, cp_shard(reference_output, 0)[sp_slice])

    (local_output * cp_shard(projection, 0)[sp_slice]).sum().backward()
    (reference_output * projection).sum().backward()

    # Gradient finalization: the TP sum for sequence-parallel parameters, then the CP sum that
    # DDP's reduction over dp_cp performs.
    module.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    finalize_config = SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
    _allreduce_non_tensor_model_parallel_grads([module], finalize_config, tp_group)
    for parameter in module.parameters():
        torch.distributed.all_reduce(parameter.grad, group=cp_group)

    for (name, parameter), ref_param in zip(
        module.named_parameters(), reference_module.parameters()
    ):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, ref_param.grad, msg=name)
