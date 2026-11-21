# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_fused_metadata_kernel`` (megatron/core/inference/moe/metadata.py), NVLS.

Launcher: ``fused_metadata_update(local_tokens, local_buf, symm_mem_hdl, step_metadata)``. Its only
caller is ``NVLSAllGatherVDispatcher.update_metadata``, which the first MoE layer of each forward
runs at EP > 1 on the ``tp_ep`` group before the AllGather-V (AGV); ``step_metadata[0]`` is the
``valid`` scalar every inference MoE kernel reads, ``[1]`` and ``[2]`` drive AGV / RSV placement.

Contract (from the module's documented API: it replaces an all-gather of per-rank counts plus
sum / prefix-sum / max; written before reading the body):

* A collective over the group of the symmetric-memory handle (``W`` ranks, this rank ``r``):
  with ``n_q`` the ``local_tokens`` of rank ``q``, it writes the int32 triple::

      step_metadata = [sum_q n_q, sum_{q < r} n_q, max_q n_q]

  on every rank. Every rank of the group must call it (it barriers internally).
* Inputs: ``local_tokens`` a host int (baked into a CUDA graph at capture, so each replay
  publishes the captured value); ``local_buf`` the first ``W`` int32 slots of a symmetric buffer
  (scratch); ``step_metadata`` a ``[3]`` int32 CUDA tensor, only ``[0:3]`` written.
* Preconditions (caller's): ``W`` a power of two; symmetric memory with NVLS multicast; the sum
  fits int32; consecutive calls on one buffer are separated by an inter-rank barrier (in
  production the AGV's end barrier and the RSV provide it; here the NCCL reference gather does).
* Memory: nothing past the ``W`` slots of the symmetric buffer and nothing outside
  ``step_metadata[0:3]`` is written.
* Tolerance class: **exact** (int32).

Oracle: ``torch.distributed.all_gather_into_tensor`` (NCCL) of each rank's ``local_tokens`` on
the same group, combined on the host into ``[sum, prefix, max]``. End to end, the dispatcher
sequence (metadata, AGV, RSV, over several steps with changing per-rank counts and no extra
synchronisation) is checked against NCCL all-gathers of every rank's inputs: the AGV rows
``[offset_q, offset_q + n_q)`` hold rank ``q``'s rows, and the RSV returns ``W`` times this rank's
rows.

Structure (read after the contract): one compiled variant per ``(RANK, WORLD_SIZE)``
(constexpr) and per Triton specialisation of ``local_tokens`` (== 1, divisible by 16, other);
single CTA; ``num_warps = min(max(1, cdiv(W, 32)), 8)``; the barrier needs ``W`` threads.
Inputs therefore cover every group size the world allows (2, 4, 8 on one 8-GPU node), every rank
position, contiguous and strided group membership (EP and ``tp_ep`` groups), counts 0 / 1 / 16 /
mixed / tied maxima / one active rank / a sum of exactly 2^31 - 1, CUDA-graph replays of
several baked buckets in shuffled order, and the dispatcher at expert TP 1 and 2.

Hardware: Hopper or newer with NVLink multicast (8 x H100 SXM in the manifest); needs a
power-of-two world of at least 2 ranks (``torch.distributed.run``). With
``MCORE_ORACLE_TARGET_SM`` set (a runner on the target node), missing GPUs or multicast fail the
test instead of skipping it.

Known finding F1 (audit family 01) concerns which forwards *run* this kernel (MTP forwards reuse
the main forward's metadata), not its arithmetic; #8563 makes every MTP forward run it and does
not change the kernel.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("triton")
metadata = pytest.importorskip("megatron.core.inference.moe.metadata")

SYMM_CANARY_BYTES = 4096
"""Bytes of the symmetric buffer past the ``W`` int32 slots that must stay untouched."""

PER_RANK_MAX = 64
"""Per-rank worst-case token count of the dispatcher test (``allocate_buffers``)."""


def _world_size() -> int:
    return Utils.world_size if torch.distributed.is_available() else 1


def _require_nvls_world() -> None:
    """Skip, or fail on a runner that targets this hardware, without a usable NVLS world."""
    world = _world_size()
    problem = None
    if not torch.cuda.is_available():
        problem = "no CUDA device"
    elif torch.cuda.get_device_capability()[0] < ou.HOPPER:
        problem = f"sm_{torch.cuda.get_device_capability()[0]}x has no NVLS multicast"
    elif world < 2 or world & (world - 1):
        problem = f"needs a power-of-two world of >= 2 ranks (torch.distributed.run), got {world}"
    if problem is None:
        return
    if ou.session_target_sm() is not None:
        pytest.fail(f"fused metadata oracle on the target hardware: {problem}", pytrace=False)
    pytest.skip(problem)


def _family_groups(world: int):
    """Rank lists per group family: every power-of-two size the world allows, contiguous and
    strided (EP groups are contiguous, ``tp_ep`` groups with expert TP interleave)."""
    families = {"world": [list(range(world))]}
    if world >= 4:
        half = world // 2
        families["halves"] = [list(range(0, half)), list(range(half, world))]
        families["halves-strided"] = [list(range(0, world, 2)), list(range(1, world, 2))]
        families["pairs"] = [[i, i + 1] for i in range(0, world, 2)]
        families["pairs-strided"] = [[i, i + half] for i in range(half)]
    unique, seen = {}, set()
    for name, groups in families.items():
        key = tuple(tuple(g) for g in groups)
        if key not in seen:
            seen.add(key)
            unique[name] = groups
    return unique


FAMILIES = list(_family_groups(max(_world_size(), 2)))


class _GroupBuffer:
    """One family's process group (the one holding this rank) and its symmetric buffer."""

    def __init__(self, group):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer

        self.group = group
        self.size = torch.distributed.get_world_size(group)
        self.rank = torch.distributed.get_rank(group)
        buf = SymmetricMemoryBuffer(size_in_mb=1, process_group=group)
        self.problem = buf.init_failure_reason
        if self.problem is None and buf.symm_mem_hdl.multicast_ptr == 0:
            self.problem = "symmetric memory has no multicast pointer (NVLS unavailable)"
        self.hdl = buf.symm_mem_hdl
        self.raw = buf.symm_buffer
        if self.problem is None:
            self.local_buf = self.raw[: 4 * self.size].view(torch.int32)
            self.canary = self.raw[4 * self.size : 4 * self.size + SYMM_CANARY_BYTES]
            self.canary.fill_(0xA5)

    def check_canary(self) -> None:
        changed = int((self.canary != 0xA5).sum())
        assert (
            changed == 0
        ), f"{changed} symmetric-buffer bytes past the {self.size} metadata slots were written"


@pytest.fixture(scope="module")
def nvls_groups():
    _require_nvls_world()
    Utils.initialize_distributed()
    world = torch.distributed.get_world_size()
    me = torch.distributed.get_rank()
    groups, made = {}, []
    for name, rank_lists in _family_groups(world).items():
        for ranks in rank_lists:
            pg = torch.distributed.new_group(ranks)
            made.append(pg)
            if me in ranks:
                groups[name] = pg
    buffers = {name: _GroupBuffer(pg) for name, pg in groups.items()}
    torch.cuda.synchronize()
    torch.distributed.barrier()
    yield buffers
    torch.cuda.synchronize()
    torch.distributed.barrier()
    buffers.clear()
    for pg in made:
        torch.distributed.destroy_process_group(pg)


def _buffer(nvls_groups, family) -> "_GroupBuffer":
    gb = nvls_groups[family]
    if gb.problem is not None:
        if ou.session_target_sm() is not None:
            pytest.fail(f"symmetric memory on {family}: {gb.problem}", pytrace=False)
        pytest.skip(gb.problem)
    return gb


def _gather_counts(n: int, group) -> list:
    """NCCL reference: every rank's count, in group-rank order. Also the inter-rank barrier
    that separates consecutive metadata calls (the kernel's precondition)."""
    size = torch.distributed.get_world_size(group)
    mine = torch.tensor([n], dtype=torch.int64, device="cuda")
    out = torch.empty(size, dtype=torch.int64, device="cuda")
    torch.distributed.all_gather_into_tensor(out, mine, group=group)
    return out.cpu().tolist()


def _expected(counts: list, rank: int) -> torch.Tensor:
    return torch.tensor([sum(counts), sum(counts[:rank]), max(counts)], dtype=torch.int64)


INT32_MAX = 2**31 - 1

PATTERNS = {
    "zeros": lambda r, w: 0,
    "ones": lambda r, w: 1,
    "sixteens": lambda r, w: 16,
    "ascending": lambda r, w: r + 1,
    "descending": lambda r, w: w - r,
    "alternating-0-1": lambda r, w: r % 2,
    "tied-max": lambda r, w: 4096 if r in (0, w - 1) else 17 * r,
    "one-active": lambda r, w: 8192 if r == w // 2 else 0,
    "uneven": lambda r, w: (r * 2654435761 + 12345) % 8192 + 1,
    "sum-int32-max": lambda r, w: INT32_MAX // w + (INT32_MAX % w if r == w - 1 else 0),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("family", FAMILIES)
def test_step_metadata_matches_nccl(nvls_groups, family):
    """Every count pattern on every group of a family, eagerly, one call after another."""
    _require_nvls_world()
    gb = _buffer(nvls_groups, family)
    for name, pattern in PATTERNS.items():
        n = pattern(gb.rank, gb.size)
        step = ou.guarded_empty((3,), torch.int32)
        metadata.fused_metadata_update(n, gb.local_buf, gb.hdl, step.tensor)
        counts = _gather_counts(n, gb.group)
        ou.check_guards(step_metadata=step)
        ou.assert_exact(
            step.tensor,
            _expected(counts, gb.rank),
            what=f"{family} rank {gb.rank}/{gb.size} pattern {name} counts {counts}",
        )
        gb.check_canary()


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_sanitizer_small_case(nvls_groups):
    """Small representative case for the compute-sanitizer memcheck pass."""
    _require_nvls_world()
    gb = _buffer(nvls_groups, "world")
    n = 3 * gb.rank + 1
    step = ou.guarded_empty((3,), torch.int32)
    metadata.fused_metadata_update(n, gb.local_buf, gb.hdl, step.tensor)
    counts = _gather_counts(n, gb.group)
    ou.check_guards(step_metadata=step)
    ou.assert_exact(step.tensor, _expected(counts, gb.rank), what="sanitizer case")


BUCKETS = [
    lambda r, w: 3 * (r + 1),
    lambda r, w: 16,
    lambda r, w: 1,
    lambda r, w: 0 if r == 0 else 100 + r,
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("family", [f for f in ("world", "pairs-strided") if f in FAMILIES])
def test_cuda_graph_replays_publish_baked_counts(nvls_groups, family):
    """One graph per bucket (``local_tokens`` baked at capture), replayed in shuffled order.

    ``step_metadata`` is re-poisoned before each replay, so each replay must recompute it from
    the counts baked into that bucket's graph on every rank.
    """
    _require_nvls_world()
    gb = _buffer(nvls_groups, family)
    step = ou.guarded_empty((3,), torch.int32)
    graphs = []
    for bucket in BUCKETS:
        n = bucket(gb.rank, gb.size)
        metadata.fused_metadata_update(n, gb.local_buf, gb.hdl, step.tensor)  # compile
        _gather_counts(n, gb.group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata.fused_metadata_update(n, gb.local_buf, gb.hdl, step.tensor)
        graphs.append((graph, n))
    torch.cuda.synchronize()
    for b in [0, 2, 1, 3, 3, 0, 2]:
        graph, n = graphs[b]
        ou.poison_(step.tensor)
        graph.replay()
        counts = _gather_counts(n, gb.group)
        ou.check_guards(step_metadata=step)
        ou.assert_exact(
            step.tensor,
            _expected(counts, gb.rank),
            what=f"{family} replay of bucket {b}, counts {counts}",
        )
        gb.check_canary()


STEP_COUNTS = [
    lambda r, w: r + 1,
    lambda r, w: PER_RANK_MAX - r,
    lambda r, w: 1,
    lambda r, w: PER_RANK_MAX,
    lambda r, w: (r * 37 + 11) % PER_RANK_MAX + 1,
    lambda r, w: 16 if r % 2 else 3,
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("expert_tp", [1, 2])
def test_dispatcher_steps_place_agv_and_rsv_rows(expert_tp):
    """The production sequence over several steps: metadata, AGV, RSV, nothing else between.

    Per step the per-rank counts change. After all steps (so no reference collective separates
    the dispatcher's own calls), each step's ``step_metadata`` (and its ``valid`` alias) is
    checked against an NCCL all-gather of the counts, the AGV rows ``[offset_q, offset_q + n_q)``
    against NCCL all-gathers of rank ``q``'s hidden states, probabilities and routing map, and the
    RSV output against ``tp_ep`` times this rank's rows (exact: power-of-two multiples of bf16).
    """
    _require_nvls_world()
    from megatron.core.activations import squared_relu
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import get_pg_rank, get_pg_size

    world = _world_size()
    ep = world // expert_tp
    hidden, topk, num_experts = 128, 6, 8 * ep
    NVLSAllGatherVDispatcher._delete_buffers()
    SymmetricMemoryManager.destroy()
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=ep, expert_tensor_parallel_size=expert_tp
    )
    try:
        pgs = get_default_pg_collection()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=128,
            moe_router_topk=topk,
            moe_router_dtype="fp32",
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=expert_tp,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=True,
            transformer_impl="inference_optimized",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            activation_func=squared_relu,
            normalization="RMSNorm",
        )
        num_local = num_experts // ep
        ep_rank = get_pg_rank(pgs.ep)
        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=PER_RANK_MAX,
            topk=topk,
            hidden_size=hidden,
            tp_ep_group=pgs.tp_ep,
        )
        dispatcher = NVLSAllGatherVDispatcher(
            num_local_experts=num_local,
            local_expert_indices=[ep_rank * num_local + i for i in range(num_local)],
            config=config,
            pg_collection=pgs,
            runs_metadata_sync=True,
        )
        size, rank = get_pg_size(pgs.tp_ep), get_pg_rank(pgs.tp_ep)
        assert size == world

        steps = []
        for s, count in enumerate(STEP_COUNTS):
            n = count(rank, size)
            g = ou.make_generator(seed=ou.DEFAULT_SEED + 1000 * s + rank, device="cuda")
            x = torch.randn(n, hidden, device="cuda", generator=g).to(torch.bfloat16)
            p = torch.rand(n, topk, device="cuda", generator=g)
            rm = torch.randint(0, num_experts, (n, topk), device="cuda", generator=g)
            h_in, p_in = dispatcher.dispatch_preprocess(x, rm, p)
            agv_h, agv_p = dispatcher.token_dispatch(h_in, p_in)
            meta = NVLSAllGatherVDispatcher._step_metadata.clone()
            valid = InferenceAllGatherDispatcherBase._valid_tokens().clone()
            record = (n, x, p, rm, meta, valid, agv_h.clone(), agv_p.clone())
            record += (dispatcher.routing_map.clone(), dispatcher.token_combine(agv_h.clone()))
            steps.append(record)

        for s, (n, x, p, rm, meta, valid, agv_h, agv_p, agv_r, combined) in enumerate(steps):
            counts = _gather_counts(n, pgs.tp_ep)
            what = f"step {s} tp_ep rank {rank}/{size} counts {counts}"
            ou.assert_exact(meta, _expected(counts, rank), what=f"step_metadata, {what}")
            ou.assert_exact(valid, meta[:1], what=f"valid aliases step_metadata[0], {what}")
            offsets = [sum(counts[:q]) for q in range(size)]
            for name, local, gathered in (("hidden", x, agv_h), ("probs", p, agv_p)):
                ref = _gather_padded(local, pgs.tp_ep)
                for q in range(size):
                    ou.assert_exact(
                        gathered[offsets[q] : offsets[q] + counts[q]],
                        ref[q, : counts[q]],
                        what=f"AGV {name} rows of rank {q}, {what}",
                    )
            ref_rm = _gather_padded(rm, pgs.tp_ep)
            for q in range(size):
                ou.assert_exact(
                    agv_r[offsets[q] : offsets[q] + counts[q]],
                    ref_rm[q, : counts[q]],
                    what=f"AGV routing rows of rank {q}, {what}",
                )
            ou.assert_exact(
                combined,
                (x.float() * size).to(torch.bfloat16),
                what=f"RSV output (tp_ep x own rows), {what}",
            )
    finally:
        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def _gather_padded(local: torch.Tensor, group) -> torch.Tensor:
    """NCCL all-gather of ``local`` padded to ``PER_RANK_MAX`` rows: ``[W, PER_RANK_MAX, ...]``."""
    size = torch.distributed.get_world_size(group)
    padded = torch.zeros((PER_RANK_MAX, *local.shape[1:]), dtype=local.dtype, device="cuda")
    padded[: local.shape[0]] = local
    out = torch.empty((size, *padded.shape), dtype=local.dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(out, padded, group=group)
    return out
