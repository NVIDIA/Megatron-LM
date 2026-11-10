# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests of the Engram symmetric lookup runtime.

The ``cpu`` cases spawn their own four-process Gloo world and drive the runtime through the
all-to-all reference transport, so routing, deduplication, the count rows, the collective
registration decisions, gradient delivery and the transport lifecycle run without CUDA. The
GPU cases run under torchrun with the CuTe/NCCL transport: one node with four GPUs exercises
LSA, two nodes in different NVLink domains exercise GIN. They skip unless CuTe DSL >= 4.8.1,
nccl4py >= 0.6.0 with a matching NCCL runtime and device bitcode are installed, and the launch
environment sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. The tests cannot enable
expandable segments themselves: the caching allocator keeps serving the plain blocks it
already holds.

A GPU case closes its transports cleanly only when it passed. After a failure on one rank, a
clean close would wait in a device barrier for peers that are still in the test, so the case
aborts instead and the failure is reported.
"""

import ast
import math
import os
import socket
import subprocess
import sys
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.engram import symmetric_lookup
from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.ops import _backends
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

from ._reference_transport import ReferenceEngramTransport
from ._test_utils import make_module_config

REPO_ROOT = Path(__file__).resolve().parents[4]
WORLD_SIZE = 4
# Table 1 has fewer rows than ranks, so ranks 2 and 3 own none of it.
TABLE_SIZES = (7, 2, 13)
DIM = 4
SYMMETRIC = SimpleNamespace(communication_backend="symmetric", symmetric_workspace_mb=64)
_SYMMETRIC_2MB = SimpleNamespace(communication_backend="symmetric", symmetric_workspace_mb=2)


# --- Spawned Gloo cases ------------------------------------------------------------------------


def _worker(rank, store, case, args):
    # The world runs on CPU: keep CUDA and NVML out of the NVLink-domain lookup, so the host is
    # every rank's domain.
    symmetric_lookup._gpu_fabric_clique = lambda: None
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=2),
    )
    try:
        case(rank, *args)
    except pytest.fail.Exception as exc:
        # pytest outcomes are BaseExceptions, which mp.spawn reports only as an exit code.
        raise AssertionError(f"rank {rank}: {exc}") from None
    finally:
        symmetric_lookup.close_engram_transports()
        dist.destroy_process_group()


def _spawn(tmp_path, case, *args):
    mp.spawn(_worker, args=(str(tmp_path / "store"), case, args), nprocs=WORLD_SIZE, join=True)


def _full_tables(dtype=torch.float64):
    """Global tables with dyadic values, so every sum in these tests is exact."""
    return [
        (torch.arange(size * DIM, dtype=dtype).view(size, DIM) + 1000 * table) / 8
        for table, size in enumerate(TABLE_SIZES)
    ]


def _build(group, dtype=torch.float64):
    """A row-sharded embedding loaded with known rows, served by a reference transport."""
    config = make_module_config(dtype=dtype)
    embedding = EPShardedMultiTableEmbedding(
        config, TABLE_SIZES, DIM, config.init_method, ep_group=group
    )
    full = _full_tables(dtype)
    with torch.no_grad():
        for table, values in zip(embedding.tables, full):
            table.weight.copy_(values[table.row_start : table.row_end])
    transport = ReferenceEngramTransport(group)
    symmetric_lookup._TRANSPORTS[group] = transport
    return embedding, full, transport, symmetric_lookup.SymmetricEngramLookup(embedding)


def _hash_ids(rank, step, *, empty_rank=None):
    """``[S, 2, T]`` ids with many duplicates; ``empty_rank`` requests nothing."""
    tokens = 0 if rank == empty_rank else 2 * (5 + rank)
    position = torch.arange(tokens)
    ids = torch.stack(
        [(position * (table + 1) + step + rank) % size for table, size in enumerate(TABLE_SIZES)],
        dim=-1,
    )
    return ids.view(-1, 2, len(TABLE_SIZES))


def _expected_rows(full, ids):
    return torch.stack([full[table][ids[..., table]] for table in range(len(full))], dim=-2)


def _upstream_grad(ids, dtype=torch.float64):
    count = ids.numel() * DIM
    return (torch.arange(count, dtype=dtype).view(*ids.shape, DIM) % 13 - 6) / 16


def _expected_grads(full, ids, grad, embedding):
    """Each owner's rows of the all-rank sum of ``index_add`` over the global tables."""
    result = []
    for table, module in enumerate(embedding.tables):
        total = torch.zeros_like(full[table])
        total.index_add_(0, ids[..., table].reshape(-1), grad[..., table, :].reshape(-1, DIM))
        dist.all_reduce(total)
        result.append(total[module.row_start : module.row_end])
    return result


def _empty_ranks_case(rank):
    embedding, full, transport, lookup = _build(dist.group.WORLD)
    assert embedding.tables[1].local_num_embeddings == (1 if rank < 2 else 0)
    for step in range(3):
        # Rank 0 requests nothing in the middle step; ranks 2 and 3 never own table 1 rows.
        ids = _hash_ids(rank, step, empty_rank=0 if step == 1 else None)
        output = lookup(ids)
        assert output.shape == (*ids.shape, DIM)
        assert torch.equal(output, _expected_rows(full, ids))
        grad = _upstream_grad(ids)
        output.backward(grad)
        for module, expected in zip(embedding.tables, _expected_grads(full, ids, grad, embedding)):
            assert torch.equal(module.weight.grad, expected)
            module.weight.grad = None
    # Every lookup ran one count exchange and two row exchanges, and every backward one
    # gradient exchange, on every rank, including the empty requester.
    assert transport.exchanges == 3 * 4


def test_cpu_empty_requester_and_empty_owners(tmp_path):
    _spawn(tmp_path, _empty_ranks_case)


def _duplicates_case(rank):
    embedding, full, transport, lookup = _build(dist.group.WORLD)
    ids = _hash_ids(rank, 0)
    unique = {
        (table, int(row))
        for table in range(len(TABLE_SIZES))
        for row in ids[..., table].reshape(-1)
    }
    assert len(unique) < ids.numel()
    assert torch.equal(lookup(ids), _expected_rows(full, ids))
    # Only unique (table, row) pairs travel, grouped by owner in peer order.
    call = transport.lookups[-1]
    owners = torch.repeat_interleave(torch.arange(WORLD_SIZE), torch.tensor(call.plan.send_splits))
    requested = {
        (table, get_contiguous_row_range(TABLE_SIZES[table], owner, WORLD_SIZE)[0] + row)
        for (table, row), owner in zip(call.requests.tolist(), owners.tolist())
    }
    assert len(call.requests) == len(unique) and requested == unique


def test_cpu_duplicate_rows_and_multiple_tables(tmp_path):
    _spawn(tmp_path, _duplicates_case)


def _gradients_case(rank):
    embedding, full, transport, lookup = _build(dist.group.WORLD)
    weights = [table.weight for table in embedding.tables]
    ids = _hash_ids(rank, 1)
    grad = _upstream_grad(ids)
    expected = _expected_grads(full, ids, grad, embedding)

    # Without main_grad the owner returns a dense gradient and clears a stale DDP flag.
    for weight in weights:
        weight.grad_added_to_main_grad = True
    lookup(ids).backward(grad)
    for weight, reference in zip(weights, expected):
        assert torch.equal(weight.grad, reference)
        assert not weight.grad_added_to_main_grad
        weight.grad = None

    # With main_grad the rows are added in place and autograd gets a zero-copy placeholder.
    for weight in weights:
        weight.main_grad = torch.full_like(weight, 0.5)
        weight.grad_added_to_main_grad = False
    lookup(ids).backward(grad)
    for weight, reference in zip(weights, expected):
        assert torch.equal(weight.main_grad, reference + 0.5)
        assert weight.grad_added_to_main_grad
        assert weight.grad.data_ptr() == weight.data_ptr()
        # What DDP's backward hook does with it.
        weight.grad = None

    # create_graph needs a real gradient: dense result, flag cleared, main_grad untouched.
    before = [weight.main_grad.clone() for weight in weights]
    gradients = torch.autograd.grad(lookup(ids), weights, grad, create_graph=True)
    for weight, gradient, reference, previous in zip(weights, gradients, expected, before):
        assert torch.equal(gradient, reference)
        assert not weight.grad_added_to_main_grad
        assert torch.equal(weight.main_grad, previous)

    # The placeholder aliases its table: accumulating onto a .grad that the DDP hook did not
    # release would write into the table. Every rank raises after the gradient exchange.
    lookup(ids).backward(grad)
    tables = [weight.detach().clone() for weight in weights]
    main_grads = [weight.main_grad.clone() for weight in weights]
    exchanges = transport.exchanges
    with pytest.raises(RuntimeError, match="not released by the DDP hook"):
        lookup(ids).backward(grad)
    # One count, two row and one gradient exchange: the same on every rank.
    assert transport.exchanges == exchanges + 4
    for weight, table, main_grad in zip(weights, tables, main_grads):
        assert torch.equal(weight.detach(), table)
        assert torch.equal(weight.main_grad, main_grad)

    # zero_out_wgrad would zero main_grad rows this backward adds to: rejected the same way.
    for weight in weights:
        weight.grad = None
        weight.zero_out_wgrad = True
    with pytest.raises(RuntimeError, match="cannot set zero_out_wgrad"):
        lookup(ids).backward(grad)

    # Every rank raised at the same point, so the row group keeps working.
    for weight in weights:
        del weight.zero_out_wgrad
        weight.main_grad.zero_()
    lookup(ids).backward(grad)
    for weight, reference in zip(weights, expected):
        assert torch.equal(weight.main_grad, reference)


def test_cpu_gradients_match_index_add(tmp_path):
    _spawn(tmp_path, _gradients_case)


def _frozen_case(rank):
    embedding, full, transport, lookup = _build(dist.group.WORLD)
    weights = [table.weight for table in embedding.tables]
    ids = _hash_ids(rank, 2)
    scale = torch.ones((), dtype=torch.float64, requires_grad=True)

    # Frozen tables: the lookup has no autograd edge and backward exchanges nothing.
    for weight in weights:
        weight.requires_grad_(False)
    output = lookup(ids)
    assert not output.requires_grad
    exchanges = transport.exchanges
    (output * scale).sum().backward()
    assert transport.exchanges == exchanges
    assert all(weight.grad is None for weight in weights)

    # One frozen table: one gradient exchange still serves the others.
    weights[0].requires_grad_(True)
    weights[2].requires_grad_(True)
    grad = _upstream_grad(ids)
    output = lookup(ids)
    exchanges = transport.exchanges
    output.backward(grad)
    assert transport.exchanges == exchanges + 1
    expected = _expected_grads(full, ids, grad, embedding)
    assert weights[1].grad is None
    assert torch.equal(weights[0].grad, expected[0])
    assert torch.equal(weights[2].grad, expected[2])


def test_cpu_frozen_tables_skip_the_gradient_exchange(tmp_path):
    _spawn(tmp_path, _frozen_case)


def _flags_case(rank):
    embedding, full, transport, lookup = _build(dist.group.WORLD)
    ids = _hash_ids(rank, 0)
    stride = DIM * 8

    def layout(peer):
        # Rank- and table-specific window offsets, as different suballocations produce.
        return tuple((1000 * peer + 16 * table, stride) for table in range(len(TABLE_SIZES)))

    state = {"changed": False, "error": None}

    def describe(key, tables):
        transport.describes += 1
        return state["changed"], state["error"], layout(rank)

    transport.describe_tables = describe
    expected_metadata = torch.tensor(
        [[layout(peer)[table] for peer in range(WORLD_SIZE)] for table in range(len(TABLE_SIZES))]
    )

    # Nothing changed anywhere: every peer's offsets arrive, nothing re-registers.
    assert torch.equal(lookup(ids), _expected_rows(full, ids))
    assert torch.equal(transport.lookups[-1].metadata, expected_metadata)
    assert transport.registrations == []

    # A change on rank 2 re-registers on every rank, with every peer's offsets.
    state["changed"] = rank == 2
    assert torch.equal(lookup(ids), _expected_rows(full, ids))
    assert len(transport.registrations) == 1
    key, metadata = transport.registrations[0]
    assert key is embedding and torch.equal(metadata, expected_metadata)

    # A failure on rank 1 raises on every rank after the count exchange, before any lookup.
    state["changed"] = False
    state["error"] = "RuntimeError: simulated" if rank == 1 else None
    lookups, exchanges = len(transport.lookups), transport.exchanges
    with pytest.raises(RuntimeError, match=r"row ranks \[1\]"):
        lookup(ids)
    assert len(transport.lookups) == lookups and transport.exchanges == exchanges + 1
    assert len(transport.registrations) == 1

    # Every rank raised at the same point, so the row group keeps working.
    state["error"] = None
    assert torch.equal(lookup(ids), _expected_rows(full, ids))


def test_cpu_count_rows_carry_flags_and_offsets(tmp_path):
    _spawn(tmp_path, _flags_case)


def _stage_chunk(group, *, tables=True, dim=DIM):
    """A model chunk with one Engram embedding, or with none."""
    chunk = torch.nn.Module()
    if tables:
        config = make_module_config()
        chunk.embedding = EPShardedMultiTableEmbedding(
            config, TABLE_SIZES, dim, config.init_method, ep_group=group
        )
    return chunk


def _transport_module(rank, transport=ReferenceEngramTransport):
    """What require_symmetric_transport() returns, with a reference transport."""
    return SimpleNamespace(
        EngramDeviceTransport=transport, create_unique_id=lambda: f"uid-{rank}".encode()
    )


def _lifecycle_case(rank):
    """Row groups {0, 1} with Engram tables and {2, 3} without, as two pipeline stages."""
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    chunk = _stage_chunk(group, tables=rank < 2)
    symmetric_lookup.require_symmetric_transport = lambda: _transport_module(rank)
    symmetric_lookup._engram_row_group = lambda pg_collection, engram_config: group

    symmetric_lookup.initialize_engram_transports([chunk], None, _SYMMETRIC_2MB)
    if rank < 2:
        transport = symmetric_lookup._TRANSPORTS[group]
        # The unique id of row rank 0 reaches its whole group; both ranks share an NVLink
        # domain, here their host.
        assert transport.options == {
            "unique_id": b"uid-0",
            "workspace_bytes": 2 * 2**20,
            "expected_lsa_ranks": 2,
        }
        # The workspace is allocated before the communicator is initialized; one barrier
        # publishes it, then eager registration runs one count exchange per Engram module,
        # before any lookup.
        assert transport.prepared and transport.initialized and transport.barriers == 1
        assert transport.describes == 1 and transport.exchanges == 1
    else:
        # Stages without Engram tables create no transport.
        assert symmetric_lookup._TRANSPORTS == {}
    with pytest.raises(RuntimeError, match="already initialized"):
        symmetric_lookup.initialize_engram_transports([chunk], None, _SYMMETRIC_2MB)
    transports = list(symmetric_lookup._TRANSPORTS.values())
    symmetric_lookup.close_engram_transports(clean=True)
    assert symmetric_lookup._TRANSPORTS == {}
    assert all(transport.closed is True for transport in transports)
    # The nccl backend creates nothing and runs no collective.
    nccl = SimpleNamespace(communication_backend="nccl", symmetric_workspace_mb=2)
    symmetric_lookup.initialize_engram_transports([chunk], None, nccl)
    assert symmetric_lookup._TRANSPORTS == {}


def test_cpu_transport_lifecycle(tmp_path):
    _spawn(tmp_path, _lifecycle_case)


def _nvlink_domain_case(rank):
    """The LSA team a transport expects is its row group's ranks in the rank's NVLink domain.

    All four ranks share a host, but sit in two GPU fabric cliques, as two NVL72 racks would.
    """
    group = dist.group.WORLD
    symmetric_lookup._nvlink_domain = lambda: ("fabric", "ab" * 16, rank % 2)
    symmetric_lookup.require_symmetric_transport = lambda: _transport_module(rank)
    symmetric_lookup._engram_row_group = lambda pg_collection, engram_config: group
    symmetric_lookup.initialize_engram_transports([_stage_chunk(group)], None, _SYMMETRIC_2MB)
    assert symmetric_lookup._TRANSPORTS[group].options["expected_lsa_ranks"] == 2
    symmetric_lookup.close_engram_transports(clean=True)


def test_cpu_expected_lsa_team_follows_the_nvlink_domain(tmp_path):
    _spawn(tmp_path, _nvlink_domain_case)


def test_cpu_nvlink_domain_is_the_fabric_clique_or_the_host(monkeypatch):
    host = ("host", socket.gethostname())
    monkeypatch.delenv("NCCL_MNNVL_ENABLE", raising=False)
    monkeypatch.setattr(symmetric_lookup, "_gpu_fabric_clique", lambda: ("ab" * 16, 7))
    assert symmetric_lookup._nvlink_domain() == ("fabric", "ab" * 16, 7)
    # Without multi-node NVLink, NCCL leaves the host over the network.
    monkeypatch.setenv("NCCL_MNNVL_ENABLE", "0")
    assert symmetric_lookup._nvlink_domain() == host
    monkeypatch.delenv("NCCL_MNNVL_ENABLE")
    monkeypatch.setattr(symmetric_lookup, "_gpu_fabric_clique", lambda: None)
    assert symmetric_lookup._nvlink_domain() == host


def _start_failures_case(rank):
    """A startup failure on rank 1 raises on all four ranks and leaves no state behind.

    The stages are those of the lifecycle case. Each scenario fails in a different step:
    the table check of the first gather, the local preparation, the workspace allocation, the
    transport initialization, and the eager registration. A failed allocation stops every rank
    before any of them initializes its transport, whose NCCL calls would wait for the rank that
    failed.
    """
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    prepared, initialized = [], []

    class Recording(ReferenceEngramTransport):
        def prepare(self):
            prepared.append(self)
            super().prepare()

        def initialize(self):
            initialized.append(self)
            super().initialize()

    class FailingPrepare(Recording):
        def prepare(self):
            if rank == 1:
                prepared.append(self)
                raise RuntimeError("simulated workspace allocation failure")
            super().prepare()

    class FailingInitialize(Recording):
        def initialize(self):
            if rank == 1:
                initialized.append(self)
                raise RuntimeError("simulated communicator failure")
            super().initialize()

    class FailingDescribe(Recording):
        def describe_tables(self, key, tables):
            changed, _, layout = super().describe_tables(key, tables)
            return changed, ("ValueError: simulated host table" if rank == 1 else None), layout

    def available():
        return _transport_module(rank, Recording)

    def missing():
        if rank == 1:
            raise ImportError("simulated missing NCCL device bitcode")
        return available()

    def resolved(pg_collection, engram_config):
        return group

    def unresolvable(pg_collection, engram_config):
        if rank == 1:
            raise ValueError("simulated unresolvable row group")
        return group

    stage = _stage_chunk(group, tables=rank < 2)
    wider = _stage_chunk(group, tables=rank < 2, dim=DIM + 4 if rank == 1 else DIM)
    # DDP put rank 1's buffers in a torch_memory_saver region (a DDP config built in code).
    saver = _stage_chunk(group, tables=rank < 2)
    saver.ddp_config = SimpleNamespace(
        disable_param_buffers_cpu_backup=rank == 1, disable_grad_buffers_cpu_backup=False
    )
    scenarios = [
        # Rank 1 holds no Engram tables, unlike the rest of its row group.
        ("mixed", _stage_chunk(group, tables=rank == 0), available, resolved, "mix ranks"),
        # Rank 1's tables are wider than its peer's.
        ("wider", wider, available, resolved, "mix ranks"),
        # Rank 1 lacks a dependency.
        ("missing", stage, missing, resolved, r"cannot start on ranks \[1\]"),
        # Rank 1 fails with neither an ImportError nor a RuntimeError.
        ("resolve", stage, available, unresolvable, r"cannot start on ranks \[1\]"),
        ("saver", saver, available, resolved, r"cannot start on ranks \[1\]"),
        # Rank 1 cannot allocate its workspace: no rank initializes its transport.
        (
            "prepare",
            stage,
            lambda: _transport_module(rank, FailingPrepare),
            resolved,
            r"cannot allocate their workspace on ranks \[1\]",
        ),
        # Rank 1 cannot initialize its transport after every rank allocated its workspace.
        (
            "initialize",
            stage,
            lambda: _transport_module(rank, FailingInitialize),
            resolved,
            r"cannot initialize on ranks \[1\]",
        ),
        # Rank 1 cannot describe its tables, so its whole row group fails eager registration.
        (
            "register",
            stage,
            lambda: _transport_module(rank, FailingDescribe),
            resolved,
            r"cannot register table windows on ranks \[0, 1\]",
        ),
    ]
    # Ranks with tables reach the allocation once the first gather passed, and the
    # initialization once the allocation passed on every rank.
    reaches_prepare = ("prepare", "initialize", "register")
    reaches_initialize = ("initialize", "register")
    for name, chunk, require, row_group, message in scenarios:
        symmetric_lookup.require_symmetric_transport = require
        symmetric_lookup._engram_row_group = row_group
        prepared.clear()
        initialized.clear()
        # Every rank raises, including the stage without tables, so none is left waiting.
        with pytest.raises(RuntimeError, match=message) as error:
            symmetric_lookup.initialize_engram_transports([chunk], None, _SYMMETRIC_2MB)
        if name == "register" and rank < 2:
            assert "row ranks [1]" in str(error.value), name
        # The failing rank names its own cause.
        if rank == 1 and name == "saver":
            assert "disable_param_buffers_cpu_backup" in str(error.value)
        if rank == 1 and name in ("prepare", "initialize"):
            assert "simulated" in str(error.value), name
        # The failed start aborted what it created and left no state behind.
        assert symmetric_lookup._TRANSPORTS == {} and not symmetric_lookup._INITIALIZED, name
        assert len(prepared) == (1 if name in reaches_prepare and rank < 2 else 0), name
        assert len(initialized) == (1 if name in reaches_initialize and rank < 2 else 0), name
        assert all(transport.closed is False for transport in prepared), name


def test_cpu_transport_start_failures_raise_on_every_rank(tmp_path):
    _spawn(tmp_path, _start_failures_case)


# --- Single-process cases ----------------------------------------------------------------------


_ENGRAM_PACKAGE = "megatron.core.models.engram"
_TRANSPORT_MODULE = "megatron.core.ops.engram"


def _imported_modules(path, package):
    """Every module an ``import`` statement in ``path`` names, guarded or not, with its line."""
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1] if node.level else [])
            module = ".".join(name for name in (base, node.module) if name)
            yield node.lineno, module
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


def test_engram_package_never_imports_cutlass_or_nccl():
    """The transport module is reached only through require(), never at package import."""
    # A guarded import would pass the import check below silently, so scan the sources too.
    package_dir = REPO_ROOT.joinpath(*_ENGRAM_PACKAGE.split("."))
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).parent.parts
        package = ".".join((_ENGRAM_PACKAGE, *relative))
        for line, module in _imported_modules(path, package):
            where = f"{path.relative_to(REPO_ROOT)}:{line} imports {module}"
            assert module.split(".")[0] not in ("cutlass", "nccl"), where
            assert not (module + ".").startswith(_TRANSPORT_MODULE + "."), where
    # torch and other Megatron Core modules may import CuTe DSL themselves (e.g. torch._native),
    # so load them first and then block only what the Engram package must not pull in: nccl4py
    # and the transport module.
    code = (
        "import importlib.abc, sys\n"
        "import torch\n"
        "import megatron.core\n"
        "def nccl_modules():\n"
        "    return {name for name in sys.modules if name.split('.')[0] == 'nccl'}\n"
        "before = nccl_modules()\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'nccl' or name == 'megatron.core.ops.engram':\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import megatron.core.models.engram\n"
        "import megatron.core.models.engram.symmetric_lookup\n"
        "assert 'megatron.core.ops.engram' not in sys.modules\n"
        "assert nccl_modules() == before\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT, env=dict(os.environ))


def test_exchange_plan_device_splits_and_reverse():
    """The ``[N, 4]`` splits the kernels read, built from per-peer counts with zeros."""
    send, recv = [3, 0, 2, 5], [0, 4, 1, 0]
    plan = symmetric_lookup.EngramExchangePlan.from_counts(
        torch.tensor(send), torch.tensor(recv), send, recv
    )
    # (send start, send rows, receive start, receive rows) per peer.
    assert plan.splits.dtype == torch.int64 and plan.splits.is_contiguous()
    assert plan.splits.tolist() == [[0, 3, 0, 0], [3, 0, 0, 4], [3, 2, 4, 1], [5, 5, 5, 0]]
    reverse = plan.reverse()
    assert (reverse.send_splits, reverse.recv_splits) == (tuple(recv), tuple(send))
    assert reverse.splits.is_contiguous()
    assert reverse.splits.tolist() == [[0, 0, 0, 3], [0, 4, 3, 0], [4, 1, 3, 2], [5, 0, 5, 5]]
    twice = reverse.reverse()
    assert (twice.send_splits, twice.recv_splits) == (plan.send_splits, plan.recv_splits)
    assert torch.equal(twice.splits, plan.splits)
    uniform = symmetric_lookup._uniform_plan(3, torch.device("cpu"))
    assert uniform.send_splits == uniform.recv_splits == (1, 1, 1)
    assert uniform.splits.tolist() == [[0, 1, 0, 1], [1, 1, 1, 1], [2, 1, 2, 1]]


def _embedding():
    config = make_module_config()
    return EPShardedMultiTableEmbedding(config, (5,), 2, config.init_method)


def test_selector_keeps_the_module_call_for_nccl():
    make_lookup = symmetric_lookup.select_engram_lookup(
        SimpleNamespace(communication_backend="nccl"), None
    )
    embedding = _embedding()
    # The module itself, hooks included, as the Engram layer called it before.
    assert make_lookup(embedding) == embedding.__call__
    hooked = []
    embedding.register_forward_pre_hook(lambda module, args: hooked.append(module))
    ids = torch.tensor([[1, 4, 0]]).unsqueeze(-1)
    assert torch.equal(make_lookup(embedding)(ids), embedding.forward(ids))
    assert hooked == [embedding]


def test_selector_fails_before_allocation_without_the_transport(monkeypatch):
    requested = []

    def import_module(name):
        requested.append(name)
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(_backends, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="Engram symmetric lookup requires"):
        symmetric_lookup.select_engram_lookup(SYMMETRIC, None)
    # Only the transport module is asked for; no table exists yet to be allocated.
    assert requested == ["megatron.core.ops.engram"]


def test_selector_fails_before_allocation_without_triton(monkeypatch):
    requested = []

    def import_module(name):
        requested.append(name)
        if name == "megatron.core.ops.engram":
            return SimpleNamespace(EngramDeviceTransport=object, check_runtime=lambda: None)
        raise ModuleNotFoundError("No module named 'triton'", name="triton")

    monkeypatch.setattr(_backends, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="Engram symmetric lookup requires") as error:
        symmetric_lookup.select_engram_lookup(SYMMETRIC, None)
    assert error.value.name == "triton"
    # The row writer is checked with the transport, before any table exists.
    assert requested == ["megatron.core.ops.engram", "megatron.core.ops.engram_row_accumulation"]


def test_selector_binds_the_symmetric_lookup(monkeypatch):
    monkeypatch.setattr(symmetric_lookup, "require_symmetric_transport", lambda: None)
    monkeypatch.setattr(
        symmetric_lookup, "get_pg_size", lambda group: 1 if group is None else group.size()
    )
    sharded = SimpleNamespace(size=lambda: 4)
    assert symmetric_lookup.select_engram_lookup(SYMMETRIC, sharded) is (
        symmetric_lookup.SymmetricEngramLookup
    )
    # A single row shard owns every row and keeps the module's local lookup.
    embedding = _embedding()
    assert symmetric_lookup.select_engram_lookup(SYMMETRIC, None)(embedding) == embedding.__call__


def test_row_accumulation_binds_the_triton_writer_on_cuda_hosts(monkeypatch):
    requested = []

    def writer(target, rows, order, gradients):
        return True

    def import_module(name):
        requested.append(name)
        return SimpleNamespace(accumulate_rows=writer)

    monkeypatch.setattr(_backends, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert symmetric_lookup.select_row_accumulation() is writer
    # Bound once, when the lookup is constructed; only the writer's module is asked for.
    assert symmetric_lookup.SymmetricEngramLookup(_embedding()).accumulate_rows is writer
    assert requested == ["megatron.core.ops.engram_row_accumulation"] * 2


def test_row_accumulation_fails_at_construction_without_triton(monkeypatch):
    def import_module(name):
        raise ModuleNotFoundError("No module named 'triton'", name="triton")

    monkeypatch.setattr(_backends, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ModuleNotFoundError, match="Engram symmetric lookup requires") as error:
        symmetric_lookup.SymmetricEngramLookup(_embedding())
    assert error.value.name == "triton"


def test_row_accumulation_uses_pytorch_without_cuda(monkeypatch):
    def import_module(name):
        raise AssertionError(f"{name} must not be imported on a host without CUDA")

    monkeypatch.setattr(_backends, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert symmetric_lookup.select_row_accumulation() is None
    assert symmetric_lookup.SymmetricEngramLookup(_embedding()).accumulate_rows is None
    target = torch.zeros(3, 2)
    rows, gradients = torch.tensor([2, 0, 2]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    symmetric_lookup._accumulate_embedding_rows(target, rows, gradients, None)
    assert target.tolist() == [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]


def test_selector_rejects_an_unresolved_backend():
    with pytest.raises(ValueError, match="must be resolved"):
        symmetric_lookup.select_engram_lookup(SimpleNamespace(communication_backend=None), None)


_CUDA_CONF, _HIP_CONF, _CONF = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
)


@pytest.mark.parametrize(
    "environ,enabled",
    [
        ({}, False),
        ({_CUDA_CONF: "expandable_segments:True"}, True),
        ({_CUDA_CONF: "max_split_size_mb:512, expandable_segments : True"}, True),
        ({_CUDA_CONF: "expandable_segments:False"}, False),
        # The last setting of the key wins, as in PyTorch's parser.
        ({_CUDA_CONF: "expandable_segments:True,expandable_segments:False"}, False),
        ({_CONF: "expandable_segments:True"}, True),
        ({_HIP_CONF: "expandable_segments:True", _CONF: "expandable_segments:False"}, True),
        # PyTorch parses the first of the three that is set, even an empty one, and ignores the
        # others: the CUDA variable hides the device-generic one.
        ({_CUDA_CONF: "", _CONF: "expandable_segments:True"}, False),
        ({_CUDA_CONF: "roundup_power2_divisions:4", _CONF: "expandable_segments:True"}, False),
        ({_CUDA_CONF: "expandable_segments:True", _CONF: "expandable_segments:False"}, True),
    ],
)
def test_expandable_segments_follow_the_variable_pytorch_reads(environ, enabled):
    assert symmetric_lookup.expandable_segments_enabled(environ) is enabled


class _ClosingTransport:
    """Records how it was closed; a failing one raises from a clean close."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def close(self, clean=False):
        self.calls.append(clean)
        if self.fail and clean:
            raise RuntimeError("simulated window close failure")


@pytest.fixture
def isolated_registry(monkeypatch):
    """A fresh transport registry, so these cases never close a transport of another test."""
    monkeypatch.setattr(symmetric_lookup, "_TRANSPORTS", {})
    monkeypatch.setattr(symmetric_lookup, "_INITIALIZED", False)


@pytest.mark.usefixtures("isolated_registry")
def test_close_engram_transports_closes_every_transport_after_a_failure(monkeypatch):
    first, second, third = _ClosingTransport(fail=True), _ClosingTransport(), _ClosingTransport()
    for key, transport in enumerate((first, second, third)):
        symmetric_lookup._TRANSPORTS[key] = transport
    monkeypatch.setattr(symmetric_lookup, "_INITIALIZED", True)
    with pytest.raises(RuntimeError, match="simulated window close failure"):
        symmetric_lookup.close_engram_transports(clean=True)
    # After the failed clean close, the rest are aborted: none stays open or unreachable.
    assert (first.calls, second.calls, third.calls) == ([True], [False], [False])
    assert symmetric_lookup._TRANSPORTS == {} and not symmetric_lookup._INITIALIZED


@pytest.mark.usefixtures("isolated_registry")
def test_failed_start_reports_its_own_error_when_the_abort_fails(monkeypatch):
    class StuckTransport(_ClosingTransport):
        def close(self, clean=False):
            self.calls.append(clean)
            raise RuntimeError("simulated abort failure")

    stuck = StuckTransport()

    def start(model_chunks, pg_collection, engram_config):
        symmetric_lookup._TRANSPORTS["row group"] = stuck
        raise RuntimeError("simulated start failure")

    monkeypatch.setattr(symmetric_lookup, "_start_engram_transports", start)
    with pytest.raises(RuntimeError, match="simulated start failure"):
        symmetric_lookup.initialize_engram_transports([], None, SYMMETRIC)
    assert stuck.calls == [False]
    assert symmetric_lookup._TRANSPORTS == {} and not symmetric_lookup._INITIALIZED


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.internal,
                pytest.mark.launch_on_gb200,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Triton row accumulation needs CUDA"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [3, 32, 256])
def test_row_accumulation_matches_fp64(device, dtype, width):
    """Ported from NVIDIA/Megatron-LM#7224 (hxbai); the CPU case runs the PyTorch path."""
    generator = torch.Generator().manual_seed(171)
    # Include a segment larger than a CTA tile and segments crossing tile boundaries.
    rows = torch.randint(0, 11, (317,), generator=generator)
    gradients = torch.randn(317, width, generator=generator).to(dtype)
    initial = torch.randn(15, width, generator=generator).to(dtype)
    expected = initial.double()
    expected.index_add_(0, rows, gradients.double())
    target = initial.clone().to(device)
    # Test non-contiguous row IDs and gradient row strides, as in a received packet.
    ids = torch.stack((rows, rows), dim=1).to(device)[:, 1]
    packet = torch.zeros(317, width + 4, dtype=dtype, device=device)
    packet[:, :width].copy_(gradients)
    values = packet[:, :width]
    accumulate_rows = symmetric_lookup.select_row_accumulation()
    symmetric_lookup._accumulate_embedding_rows(target, ids, values, accumulate_rows)
    torch.testing.assert_close(target.cpu(), expected.to(dtype), rtol=2e-5, atol=2e-5)
    replay = initial.clone().to(device)
    symmetric_lookup._accumulate_embedding_rows(replay, ids, values, accumulate_rows)
    assert torch.equal(target, replay)
    assert torch.equal(target[11:].cpu(), initial[11:])
    if accumulate_rows is not None:
        # The Triton writer takes CUDA inputs itself and declines CPU ones without writing.
        probe = initial.clone().to(device)
        took = accumulate_rows(probe, ids, torch.argsort(ids, stable=True), values)
        assert took == (device == "cuda")
        assert torch.equal(probe, target if took else initial.to(device))


# --- GPU cases (torchrun) ----------------------------------------------------------------------


def _symmetric_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("The Engram symmetric transport needs CUDA")
    if not symmetric_lookup.expandable_segments_enabled():
        pytest.skip("Engram table windows need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    try:
        symmetric_lookup.require_symmetric_transport()
    except ImportError as exc:
        pytest.skip(f"Engram symmetric transport unavailable: {exc}")
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("Use torchrun with two or more ranks; ranks in two NVLink domains use GIN")


def _start_transport(monkeypatch, chunks, group):
    monkeypatch.setattr(symmetric_lookup, "_engram_row_group", lambda pg, config: group)
    symmetric_lookup.initialize_engram_transports(chunks, None, SYMMETRIC)
    return symmetric_lookup._TRANSPORTS[group]


def _cuda_table_sizes(size):
    # A large table, a table with fewer rows than ranks (empty owners) and a prime-sized one.
    return (1000 * size + 3, max(1, size - 1), 97)


def _cuda_embedding(group, dtype, dim=160, table_sizes=None):
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=dtype,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=False,
    )
    table_sizes = table_sizes or _cuda_table_sizes(dist.get_world_size(group))
    embedding = EPShardedMultiTableEmbedding(
        config, table_sizes, dim, lambda tensor: None, ep_group=group
    )
    with torch.no_grad():
        for table, module in enumerate(embedding.tables):
            rows = torch.arange(module.row_start, module.row_end, device="cuda")
            columns = torch.arange(dim, device="cuda")
            values = (rows[:, None] * 7 + columns + 1000 * table) % 251 - 125
            module.weight.copy_((values / 8).to(dtype))
    return embedding


def _cuda_ids(embedding, rank, step, *, empty_rank=None):
    generator = torch.Generator(device="cuda").manual_seed(1000 * step + rank)
    tokens = 0 if rank == empty_rank else 257 + 31 * rank
    ids = [
        torch.randint(0, size, (tokens, 2), device="cuda", generator=generator)
        for size in embedding.table_sizes
    ]
    return torch.stack(ids, dim=-1)


def _cuda_grad(ids, dim, dtype, step):
    generator = torch.Generator(device="cuda").manual_seed(7 + step)
    return torch.randn((*ids.shape, dim), device="cuda", generator=generator).to(dtype)


def _attach_main_grad(weights):
    for weight in weights:
        weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        weight.grad_added_to_main_grad = False


def _same_bits(a, b):
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _close(symmetric, ok):
    """Put the symmetric transport back and close it: collectively only if the case passed."""
    if symmetric is not None:
        symmetric_lookup._TRANSPORTS[symmetric.group] = symmetric
    symmetric_lookup.close_engram_transports(clean=ok)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_symmetric_lookup_matches_reference_bitwise(monkeypatch, dtype):
    _symmetric_or_skip()
    group, rank = dist.group.WORLD, dist.get_rank()
    embedding = _cuda_embedding(group, dtype)
    weights = [table.weight for table in embedding.tables]
    lookup = symmetric_lookup.SymmetricEngramLookup(embedding)
    symmetric, ok = None, False
    try:
        symmetric = _start_transport(monkeypatch, [embedding], group)
        results = {}
        for name, transport in (("symmetric", symmetric), ("reference", None)):
            symmetric_lookup._TRANSPORTS[group] = transport or ReferenceEngramTransport(group)
            _attach_main_grad(weights)
            outputs = []
            for step in range(3):
                ids = _cuda_ids(embedding, rank, step, empty_rank=0 if step == 1 else None)
                output = lookup(ids)
                output.backward(_cuda_grad(ids, output.shape[-1], dtype, step))
                for weight in weights:
                    # What DDP's backward hook does with the placeholder.
                    assert weight.grad.data_ptr() == weight.data_ptr()
                    weight.grad = None
                outputs.append(output.detach())
            results[name] = outputs, [weight.main_grad.clone() for weight in weights]
        for actual, expected in zip(*(results[name][0] for name in ("symmetric", "reference"))):
            assert _same_bits(actual, expected)
        for actual, expected in zip(*(results[name][1] for name in ("symmetric", "reference"))):
            assert _same_bits(actual, expected)
        ok = True
    finally:
        _close(symmetric, ok)


class _TableModule(MegatronModule):
    def __init__(self, config, pg_collection, table_sizes, dim) -> None:
        super().__init__(config=config)
        self.embedding = EPShardedMultiTableEmbedding(
            config,
            table_sizes,
            dim,
            config.init_method,
            ep_group=pg_collection.ep,
            tp_group=pg_collection.tp,
            expt_dp_group=pg_collection.expt_dp,
        )
        self.lookup = symmetric_lookup.SymmetricEngramLookup(self.embedding)

    def forward(self, hash_ids):
        return self.lookup(hash_ids)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
def test_ddp_placeholder_gradient_allocates_no_dense_table_gradient(
    monkeypatch, overlap_grad_reduce
):
    _symmetric_or_skip()
    size = dist.get_world_size()
    Utils.initialize_model_parallel(expert_model_parallel_size=size)
    symmetric, ok = None, False
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=1,
            params_dtype=torch.bfloat16,
            bf16=True,
        )
        shard_rows, dim = 1 << 20, 64
        module = _TableModule(config, pg, (shard_rows * size, 4099), dim)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=overlap_grad_reduce, grad_reduce_in_fp32=True
            ),
            module,
            pg_collection=pg,
        )
        symmetric = _start_transport(monkeypatch, [ddp], pg.ep)
        weights = [table.weight for table in module.embedding.tables]
        dense_bytes = weights[0].numel() * weights[0].element_size()
        ids = _cuda_ids(module.embedding, dist.get_rank(), 0)
        grad = _cuda_grad(ids, dim, torch.bfloat16, 0)

        def step():
            ddp.zero_grad_buffer()
            ddp(ids).backward(grad)
            ddp.finish_grad_sync()
            torch.cuda.synchronize()

        step()  # compile the kernels before measuring
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        step()
        peak = torch.cuda.max_memory_allocated() - base
        # A dense gradient of the large table alone would take dense_bytes.
        assert peak < dense_bytes // 4, f"step peak {peak} B, dense table gradient {dense_bytes} B"
        assert all(weight.grad is None and weight.grad_added_to_main_grad for weight in weights)
        main_grads = [weight.main_grad.clone() for weight in weights]
        assert all(torch.count_nonzero(main_grad) for main_grad in main_grads)
        # The reference transport produces the same reduced gradients, bit for bit.
        symmetric_lookup._TRANSPORTS[pg.ep] = ReferenceEngramTransport(pg.ep)
        step()
        for weight, expected in zip(weights, main_grads):
            assert _same_bits(weight.main_grad, expected)
        ok = True
    finally:
        _close(symmetric, ok)
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
def test_full_recompute_replays_the_lookup(monkeypatch):
    _symmetric_or_skip()
    Utils.initialize_model_parallel()
    ok = False
    try:
        model_parallel_cuda_manual_seed(1234)
        group, rank = dist.group.WORLD, dist.get_rank()
        embedding = _cuda_embedding(group, torch.bfloat16)
        weights = [table.weight for table in embedding.tables]
        _attach_main_grad(weights)
        lookup = symmetric_lookup.SymmetricEngramLookup(embedding)
        transport = _start_transport(monkeypatch, [embedding], group)
        calls = {"describe_tables": 0, "lookup": 0}
        for name in calls:

            def counted(*args, _method=getattr(transport, name), _name=name, **kwargs):
                calls[_name] += 1
                return _method(*args, **kwargs)

            monkeypatch.setattr(transport, name, counted)
        ids = _cuda_ids(embedding, rank, 0)
        grad = _cuda_grad(ids, 160, torch.bfloat16, 0)
        scale = torch.ones((), device="cuda", requires_grad=True)

        def run(checkpointed):
            for weight in weights:
                weight.main_grad.zero_()
            if checkpointed:
                # The first pass runs without grad: the count exchange still runs, then the
                # whole lookup replays in backward on every rank of the row group.
                output = tensor_parallel.checkpoint(
                    lambda factor, hash_ids: lookup(hash_ids) * factor, False, scale, ids
                )
            else:
                output = lookup(ids) * scale
            output.backward(grad)
            for weight in weights:
                weight.grad = None
            return output.detach(), [weight.main_grad.clone() for weight in weights]

        plain = run(False)
        assert calls == {"describe_tables": 1, "lookup": 1}
        recomputed = run(True)
        assert calls == {"describe_tables": 3, "lookup": 3}
        assert _same_bits(plain[0], recomputed[0])
        for actual, expected in zip(recomputed[1], plain[1]):
            assert _same_bits(actual, expected)
        ok = True
    finally:
        _close(None, ok)
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
def test_dense_fallback_resets_grad_added_to_main_grad(monkeypatch):
    _symmetric_or_skip()
    group, rank = dist.group.WORLD, dist.get_rank()
    embedding = _cuda_embedding(group, torch.bfloat16)
    weights = [table.weight for table in embedding.tables]
    lookup = symmetric_lookup.SymmetricEngramLookup(embedding)
    symmetric, ok = None, False
    try:
        symmetric = _start_transport(monkeypatch, [embedding], group)
        ids = _cuda_ids(embedding, rank, 0)
        grad = _cuda_grad(ids, 160, torch.bfloat16, 0)
        dense = {}
        for name, transport in (("symmetric", symmetric), ("reference", None)):
            symmetric_lookup._TRANSPORTS[group] = transport or ReferenceEngramTransport(group)
            for weight in weights:
                # A stale flag, as left by a captured step, must not hide a dense gradient.
                weight.grad_added_to_main_grad = True
            lookup(ids).backward(grad)
            assert not any(weight.grad_added_to_main_grad for weight in weights)
            dense[name] = [weight.grad for weight in weights]
            for weight in weights:
                weight.grad = None
        for actual, expected in zip(dense["symmetric"], dense["reference"]):
            assert _same_bits(actual, expected)

        # create_graph: a real gradient even with main_grad; main_grad stays untouched.
        symmetric_lookup._TRANSPORTS[group] = symmetric
        _attach_main_grad(weights)
        for weight in weights:
            weight.grad_added_to_main_grad = True
        gradients = torch.autograd.grad(lookup(ids), weights, grad, create_graph=True)
        for weight, gradient, expected in zip(weights, gradients, dense["reference"]):
            assert _same_bits(gradient, expected)
            assert not weight.grad_added_to_main_grad
            assert torch.count_nonzero(weight.main_grad) == 0
        ok = True
    finally:
        _close(symmetric, ok)


class _PlainCudaAllocation:
    """Memory from ``cuMemAlloc``: no VMM allocation handle, so no NCCL window can map it."""

    def __init__(self, nbytes: int) -> None:
        from cuda.bindings import driver

        status, pointer = driver.cuMemAlloc(nbytes)
        assert status == driver.CUresult.CUDA_SUCCESS, status
        # Freed with the last reference, which the tensor viewing this memory holds.
        weakref.finalize(self, driver.cuMemFree, pointer)
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "strides": None,
            "data": (int(pointer), False),
            "version": 3,
        }


def _plain_cuda_copy(tensor):
    """A copy of CUDA ``tensor`` in plain ``cuMemAlloc`` memory, outside the caching allocator."""
    nbytes = math.prod(tensor.shape) * tensor.element_size()
    # The tensor keeps the allocation object, which frees the memory with the last reference.
    plain = torch.as_tensor(_PlainCudaAllocation(nbytes), device=tensor.device)
    plain = plain.view(tensor.dtype).view(tensor.shape)
    plain.copy_(tensor)
    return plain


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "storage,cause",
    [
        # Host memory cannot back an NCCL window: the device check fails first.
        ("host", "Engram table on cpu"),
        # Device memory without expandable segments or an NCCL allocator: the VMM check fails.
        ("plain_cuda", "is not a CUDA VMM allocation"),
    ],
)
def test_unregisterable_storage_fails_at_startup_on_every_rank(monkeypatch, storage, cause):
    _symmetric_or_skip()
    group = dist.group.WORLD
    embedding = _cuda_embedding(group, torch.bfloat16)
    weight = embedding.tables[0].weight
    if dist.get_rank() == 1:
        weight.data = weight.data.cpu() if storage == "host" else _plain_cuda_copy(weight.data)
    try:
        # Eager registration fails on rank 1; every rank raises at startup, before any lookup.
        with pytest.raises(RuntimeError, match=r"row ranks \[1\]") as error:
            _start_transport(monkeypatch, [embedding], group)
        if dist.get_rank() == 1:
            assert cause in str(error.value)
        # The failed start aborted its transport and left no state behind.
        assert symmetric_lookup._TRANSPORTS == {} and not symmetric_lookup._INITIALIZED
    finally:
        symmetric_lookup.close_engram_transports()
