# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""moe_ep_invariant_combine at inference: the ordered pairwise reduce-scatter-v over NVLS.

Kept in the inference test bucket: it allocates CUDA-backend symmetric memory, which fixes
the process-wide symmetric-memory backend and would break later NCCL-backend users (the
ncclep dispatcher tests) in the transformer/moe bucket.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import ep_invariant_combine as eic
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_moe_ep_invariant_combine import (
    _bits,
    _case,
    _rank_inverse_map,
    _reference,
    needs_cuda,
)


@pytest.fixture
def ep_world():
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@needs_cuda
@pytest.mark.skipif(Utils.world_size < 2, reason="needs >= 2 GPUs")
def test_nvls_pairwise_reduce_scatter_equals_ep1(ep_world):
    from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
    from megatron.core.inference.moe import batch_invariant as inf_bi
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    if not is_device_nvls_capable(torch.device("cuda")):
        pytest.skip("needs NVLS")
    ep, rank = Utils.world_size, torch.distributed.get_rank()
    group = parallel_state.get_expert_model_parallel_group()
    experts = 8 * Utils.world_size
    idx, vals = _case(48 * Utils.world_size, experts, 6, 512, seed=11)
    tokens, hidden = idx.shape[0], vals.shape[-1]
    counts = [tokens // ep - 1] * (ep - 1) + [tokens - (tokens // ep - 1) * (ep - 1)]
    buf = SymmetricMemoryManager.get_buffer(
        "test_ep_inv", process_group=group, size_mb=-(-tokens * hidden * 4 // (1 << 20))
    )
    rsv = buf.maybe_get_tensor([tokens, hidden], dtype=torch.float32)
    inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
    valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    y = vals[t, k].to(torch.bfloat16).contiguous()
    inf_bi.unpermute_tokens_in_expert_order(
        y, inv, valid, rsv["tensor"], eic.num_leaf_blocks(experts) // ep
    )
    out = torch.empty(counts[rank], hidden, dtype=torch.float32, device="cuda")
    inf_bi.ordered_reduce_scatter_v(
        out,
        rsv["tensor"],
        rsv["handle"],
        rank_token_offset=torch.tensor([sum(counts[:rank])], dtype=torch.int32, device="cuda"),
        ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
        per_rank_max_tokens=max(counts),
        pairwise_tree=True,
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    ref = _reference(idx, vals, experts)
    off = sum(counts[:rank])
    assert torch.equal(_bits(out), _bits(ref[off : off + counts[rank]]))
    SymmetricMemoryManager.destroy("test_ep_inv")
