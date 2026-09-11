# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TP/SP correctness for Engram.

Runs under the CI unit-test launcher (torch.distributed.run at 8 ranks) as well as local
torchrun with 2 or 4 ranks: each case treats the first ``tp_size`` ranks as one
tensor-parallel group with sequence parallelism enabled (EP=1, expert-DP=1). Each rank
forwards its contiguous sequence slice; outputs must match the full-sequence reference
(including the convolution halo across the slice boundary), and gradients of the replicated
Engram parameters must match after the TP sum that
``_allreduce_non_tensor_model_parallel_grads`` performs during gradient finalization.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
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


def _build_module(tmp_path, tp_group, sequence_parallel):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=(1,))
    engram_config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )
    torch.manual_seed(2026)
    module = Engram(
        config=make_module_config(sequence_parallel=sequence_parallel),
        engram_config=engram_config,
        layer_number=1,
        pg_collection=make_pg_collection(tp=tp_group),
    )
    with torch.no_grad():
        # A nonzero convolution branch so the halo exchange and its gradients matter.
        module.short_conv.weight.normal_(mean=0.0, std=0.05)
    return module.to(torch.cuda.current_device())


@pytest.mark.parametrize("tp_size", [2, 4])
def test_sp_sliced_gradients_match_full_sequence_reference(tmp_path_factory, tp_size):
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

    sp_module = _build_module(tmp_path, tp_group, sequence_parallel=True)
    reference_module = _build_module(tmp_path, tp_group, sequence_parallel=False)
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
