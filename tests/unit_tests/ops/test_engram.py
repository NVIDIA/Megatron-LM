# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compile checks and real device execution checks for the Engram symmetric transport.

Ported from tests/unit_tests/ops/test_engram.py of NVIDIA/Megatron-LM#7224 by Hongxiao Bai
(hxbai), commits 610449cd5 and 906ccd59e, and adapted to the transport's host interface: the
unique id is passed in, plans carry this rank's splits, and table registration follows the
flags of the count exchange instead of a per-lookup object gather. The FP8 scale cases are
gone with the MXFP8 tables. Device cases run under torchrun on GB300: one node with four GPUs
for LSA, two nodes in different NVLink domains for GIN. They close their transport
collectively only when they pass: after a failure on one rank, a clean close would wait in a
device barrier for peers that are still in the test. The compile checks need no GPU and carry
no device markers.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.engram import symmetric_lookup
from megatron.core.models.engram.symmetric_lookup import EngramExchangePlan
from megatron.core.ops._backends import require
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("cutlass", exc_type=ImportError)
pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
try:
    # Any other missing or broken piece of the stack (cuda.core, cuda.bindings, NCCL4Py
    # interop, a CuTe DSL that fails to load) skips the module instead of failing collection.
    engram = require("megatron.core.ops.engram", needed_by="Engram symmetric lookup")
except ImportError as exc:
    pytest.skip(str(exc), allow_module_level=True)


@pytest.fixture
def runtime():
    """Skip unless CuTe DSL, NCCL4Py, the NCCL runtime and its device bitcode match."""
    try:
        engram.check_runtime()
    except ImportError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("rows", [0, 1, 17])
def test_engram_dynamic_dlpack_handles_empty_and_singleton_rows(rows):
    # PyTorch DLPack canonicalizes the first stride to 1 for a 0- or 1-row
    # matrix; flattening must accept those just like a normal request matrix.
    requests = torch.empty(rows, 2, dtype=torch.int64)
    result = engram._rows_tensor(requests)
    assert str(result).endswith("o (?):(1)>")


@pytest.mark.parametrize("storage_offset", [0, 512, 3584, 4608])
@pytest.mark.parametrize("view_offset", [0, 13])
def test_engram_window_range_includes_unaligned_storage(monkeypatch, storage_offset, view_offset):
    """Storage and tensor offsets both contribute to the registered window offset."""
    base, size = 0x10000, 16384
    address = base + storage_offset
    storage = SimpleNamespace(data_ptr=lambda: address, nbytes=lambda: size)
    tensor = SimpleNamespace(
        untyped_storage=lambda: storage,
        data_ptr=lambda: address + view_offset,
        stride=lambda dim: 32,
        element_size=lambda: 1,
        device=torch.device("cuda"),
    )
    # The storage extends beyond its first mapping, as with expandable segments.
    query = Mock(return_value=(engram.cuda.CUresult.CUDA_SUCCESS, base, 8192))
    monkeypatch.setattr(engram.cuda, "cuMemGetAddressRange", query)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    make_buffer = Mock(
        side_effect=lambda ptr, size, owner: SimpleNamespace(handle=ptr, size=size, owner=owner)
    )
    monkeypatch.setattr(engram, "Buffer", SimpleNamespace(from_handle=make_buffer))

    buffer, offset, stride = engram.EngramDeviceTransport._storage_view(tensor)
    query.assert_called_once_with(address)
    # A rounded suballocation would register successfully, but NCCL LSA reads
    # would use this rank's window offset on peers with different suballocations.
    assert buffer.handle == base
    assert buffer.handle + buffer.size == address + size
    assert buffer.handle + offset == tensor.data_ptr()
    assert buffer.owner is storage and stride == 32


def _host_transport(monkeypatch, size=2):
    """A two-rank transport with NCCL replaced by mocks, for the host registration logic."""
    transport = engram.EngramDeviceTransport.__new__(engram.EngramDeviceTransport)
    transport.group, transport.rank, transport.size = object(), 0, size
    transport.device = torch.device("cpu")
    transport.tables, transport.pending = {}, {}
    transport.windows, transport.next_window_id = {}, 0
    transport.lookup_kernels, transport.exchange_kernels = {}, {}
    transport.barrier = Mock()
    transport.uniform_plan = None
    transport.device_view, transport.published, transport.failed = None, True, False
    # Every window the mocked communicator registered, in order.
    transport.registered = []

    def register(view):
        window = SimpleNamespace(is_valid=True, close=Mock(), view=view)
        transport.registered.append(window)
        return window

    transport.comm = SimpleNamespace(register_window=register, abort=Mock(), destroy=Mock())
    monkeypatch.setattr(engram.nc, "Window", lambda window: ("device", window))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=Mock()))
    monkeypatch.setattr(transport, "_check_vmm", Mock())
    return transport


def _peer_exchange(state):
    """Registration rows of a two-rank group whose peer mirrors this rank, unless ``state``
    says the peer's storage is new (it holds no window) or keeps the tables apart. A one-word
    row is a registration outcome: the peer failed if ``state`` says so."""

    def exchange(rows, plan):
        if rows.shape[1] == 1:
            return torch.stack((rows[0], torch.tensor([int(state.get("peer_failed", False))])))
        peer = rows[0].clone()
        if state.get("peer_storage_new"):
            peer[1::2] = -1
        if state.get("peer_separate"):
            peer[0::2] = torch.arange(len(peer) // 2)
            peer[1::2] = -1
        return torch.stack((rows[0], peer))

    return Mock(side_effect=exchange)


@pytest.mark.parametrize("change", ["none", "local", "remote"])
def test_engram_table_window_refresh_is_collective(monkeypatch, change):
    """A change on one shard refreshes all windows and remote view metadata."""
    transport = _host_transport(monkeypatch)
    # Mock only CUDA allocation discovery; retain real storage lifetime/signatures.
    describe = Mock(
        side_effect=lambda tensor: (
            SimpleNamespace(owner=tensor.untyped_storage()),
            tensor.storage_offset() * tensor.element_size(),
            tensor.stride(0) * tensor.element_size(),
        )
    )
    monkeypatch.setattr(transport, "_storage_view", describe)
    state = {}
    transport.exchange = _peer_exchange(state)

    def metadata(remote_offset):
        # [tables, row ranks, (offset, stride)] as received through the count exchange.
        return torch.tensor([[[0, 32], [remote_offset, 32]]])

    data = [torch.zeros(3, 8)]
    key = object()
    assert transport.describe_tables(key, data) == (True, None, ((0, 32),))
    transport.register_tables(key, metadata(0))
    first = transport.tables[key]
    assert first.aligned and len(transport.registered) == 1
    old_pointer = data[0].untyped_storage().data_ptr()
    assert first.description.storages[0].view.owner.data_ptr() == old_pointer
    if change == "local":
        data[0] = data[0].clone()
    changed, error, layout = transport.describe_tables(key, data)
    assert (changed, error, layout) == (change == "local", None, ((0, 32),))
    # The count rows carry every rank's change flag: a change anywhere re-registers here.
    if changed or change == "remote":
        state["peer_storage_new"] = change == "remote"
        # Exercise an unaligned remote view, even when this rank's view is aligned.
        transport.register_tables(key, metadata(2 if change == "remote" else 0))
    second = transport.tables[key]
    assert len(transport.registered) == (1 if change == "none" else 2)
    assert transport.registered[0].close.call_count == (change != "none")
    assert len(transport.windows) == 1
    assert (second is first) == (change == "none")
    assert second.aligned == (change != "remote")
    assert describe.call_count == (2 if change == "local" else 1)


@pytest.mark.parametrize("peer_shares", [True, False])
def test_engram_tables_share_a_window_only_where_every_rank_shares_storage(
    monkeypatch, peer_shares
):
    """Tables of one storage share a window unless some rank keeps them apart."""
    transport = _host_transport(monkeypatch)
    monkeypatch.setattr(
        transport,
        "_storage_view",
        lambda tensor: (SimpleNamespace(owner=tensor.untyped_storage()), 0, 32),
    )
    transport.exchange = _peer_exchange({"peer_separate": not peer_shares})
    buffer = torch.zeros(4, 8)
    tables = [buffer[:2], buffer[2:]]
    key = object()
    changed, error, layout = transport.describe_tables(key, tables)
    # Offsets are relative to the shared storage window.
    assert (changed, error, layout) == (True, None, ((0, 32), (64, 32)))
    transport.register_tables(key, torch.tensor([[[0, 32]] * 2, [[64, 32]] * 2]))
    registered = transport.tables[key]
    assert len(registered.window_ids) == len(transport.registered) == (1 if peer_shares else 2)
    assert (registered.device_windows[0] == registered.device_windows[1]) == peer_shares


def test_engram_lookups_share_the_window_of_one_storage(monkeypatch):
    """Lookups over one storage share its window, which closes only with its last user."""
    transport = _host_transport(monkeypatch)
    monkeypatch.setattr(
        transport,
        "_storage_view",
        lambda tensor: (SimpleNamespace(owner=tensor.untyped_storage()), 0, 32),
    )
    state = {}
    transport.exchange = _peer_exchange(state)
    # Two Engram layers whose tables sit in one DDP parameter buffer.
    buffer = torch.zeros(4, 8)
    first, second = object(), object()
    metadata = torch.tensor([[[0, 32], [0, 32]]])
    for key, rows in ((first, buffer[:2]), (second, buffer[2:])):
        transport.describe_tables(key, [rows])
        transport.register_tables(key, metadata)
    assert len(transport.registered) == 1
    assert transport.tables[first].window_ids == transport.tables[second].window_ids
    # The peer replaced the storage behind both layers; the first layer refreshes first.
    state["peer_storage_new"] = True
    assert transport.describe_tables(first, [buffer[:2]])[0] is False
    transport.register_tables(first, metadata)
    assert len(transport.registered) == 2
    # The second layer still reads the old window, so it stays open until that layer refreshes.
    assert transport.registered[0].close.call_count == 0
    # By then every rank holds both storages in the new window, so the layers share it again.
    state["peer_storage_new"] = False
    transport.register_tables(second, metadata)
    assert len(transport.registered) == 2
    assert transport.registered[0].close.call_count == 1
    assert transport.tables[first].window_ids == transport.tables[second].window_ids == (1,)
    assert sorted(transport.windows) == [1]


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_engram_window_preparation_failure_is_collective(monkeypatch, failed_rank):
    """A bad allocation on one rank must not leave peers entering registration."""
    transport = _host_transport(monkeypatch)
    describe = Mock(return_value=(object(), 0, 32))
    if failed_rank == 0:
        describe.side_effect = RuntimeError("allocation query failed")
    monkeypatch.setattr(transport, "_storage_view", describe)

    def exchange(rows, plan):
        # Count rows of one table: [requests, changed, error, offset, stride].
        peer = torch.tensor([0, 1, int(failed_rank == 1), 0, 32])
        return torch.stack((rows[0], peer))

    transport.exchange = Mock(side_effect=exchange)
    message = r"row ranks \[0\]; on this rank: RuntimeError: allocation query failed"
    with pytest.raises(RuntimeError, match=message if failed_rank == 0 else r"row ranks \[1\]"):
        symmetric_lookup.prepare_table_windows(
            transport, object(), [torch.zeros(3, 8)], torch.zeros(2, 1, dtype=torch.int64)
        )
    assert transport.registered == [] and transport.windows == {}
    transport.barrier.assert_not_called()


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_engram_window_registration_failure_is_collective(monkeypatch, failed_rank):
    """A window one rank cannot register makes every rank raise, after the same calls."""
    transport = _host_transport(monkeypatch)
    monkeypatch.setattr(
        transport,
        "_storage_view",
        lambda tensor: (SimpleNamespace(owner=tensor.untyped_storage()), 0, 32),
    )
    transport.exchange = _peer_exchange({"peer_failed": failed_rank == 1})
    if failed_rank == 0:
        register = transport.comm.register_window

        def register_first_invalid(view):
            window = register(view)
            window.is_valid = len(transport.registered) > 1
            return window

        transport.comm.register_window = register_first_invalid
    # Two tables in two storages need two windows, on both ranks.
    tables = [torch.zeros(2, 8), torch.zeros(2, 8)]
    key = object()
    transport.describe_tables(key, tables)
    message = r"row ranks \[0\]; on this rank: RuntimeError: NCCL returned an invalid table window"
    with pytest.raises(RuntimeError, match=message if failed_rank == 0 else r"row ranks \[1\]\."):
        transport.register_tables(key, torch.tensor([[[0, 32]] * 2] * 2))
    # The failing rank still issued the peer's second registration, and no rank closed a
    # window on its own: the windows may now differ between ranks.
    assert len(transport.registered) == 2
    assert all(window.close.call_count == 0 for window in transport.registered)
    assert key not in transport.tables and transport.failed
    # So even a clean close aborts, without a barrier.
    comm, transport.device_view = transport.comm, object()
    transport.close(clean=True)
    comm.abort.assert_called_once_with()
    comm.destroy.assert_not_called()
    transport.barrier.assert_called_once()


@pytest.mark.parametrize("failing", ["barrier", "window"])
def test_engram_failed_clean_close_aborts(monkeypatch, failing):
    """A clean close that cannot finish aborts the communicator before it re-raises, so no
    communicator, window or GIN resource is left behind for a later abort to miss."""
    transport = _host_transport(monkeypatch)
    comm, transport.device_view = transport.comm, object()
    transport.window = SimpleNamespace(close=Mock())
    if failing == "barrier":
        transport.barrier.side_effect = RuntimeError("simulated barrier failure")
    else:
        broken = SimpleNamespace(close=Mock(side_effect=RuntimeError("simulated window failure")))
        transport.windows[0] = SimpleNamespace(window=broken)
    with pytest.raises(RuntimeError, match=f"simulated {failing} failure"):
        transport.close(clean=True)
    comm.abort.assert_called_once_with()
    comm.destroy.assert_not_called()
    assert transport.comm is None and transport.window is None and transport.windows == {}


def test_engram_close_before_initialize_releases_the_workspace(monkeypatch):
    """After prepare() alone (a peer failed to allocate), close drops the workspace."""
    transport = _host_transport(monkeypatch)
    transport.comm, transport.buffer = None, torch.empty(16, dtype=torch.uint8)
    transport.close(clean=True)
    assert transport.buffer is None and transport.uniform_plan is None


def test_engram_unsharded_transport_does_not_use_world(monkeypatch):
    def unexpected(*args):
        raise AssertionError("None denotes an unsharded table, not the default process group")

    monkeypatch.setattr(dist, "get_rank", unexpected)
    monkeypatch.setattr(dist, "get_world_size", unexpected)
    monkeypatch.setattr(engram, "check_runtime", lambda: None)
    transport = engram.EngramDeviceTransport(None, unique_id=b"", workspace_bytes=2**20)
    assert (transport.rank, transport.size) == (0, 1)


def test_engram_communicator_has_one_cta_and_no_nvls(runtime):
    """The row group's communicator serves the device API alone, which needs no NCCL channel
    beyond one and no NVLS state."""
    config = engram._communicator_config()
    assert (config.min_ctas, config.max_ctas) == (1, 1)
    if engram.nccl.get_version().libnccl.version >= engram.Version("2.32.0"):
        mode = engram.nccl.NcclNvlsHostMode
        assert config.nvls_host_mode == mode.DISABLE_TRANSPORT | mode.DISABLE_SYMMETRIC_MULTIMEM


@pytest.mark.parametrize("has_gin,remote_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("word_bytes", [1, 4])
def test_engram_device_exchange_compiles(runtime, has_gin, remote_only, word_bytes):
    """Compile actual NCCL bitcode for Blackwell, including both signal protocols."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from nccl.core.device.cute.runtime import make_fake_dev_comm, make_fake_window

    dtype = cutlass.Int32 if word_bytes == 4 else cutlass.Uint8
    result = cute.compile(
        engram._exchange,
        make_fake_dev_comm(),
        make_fake_window(),
        make_fake_tensor(dtype, (cute.sym_int64(),), (1,)),
        make_fake_tensor(dtype, (cute.sym_int64(),), (1,)),
        make_fake_tensor(cutlass.Int64, (4, 4), (4, 1)),
        make_fake_tensor(cutlass.Int64, (16,), (1,)),
        cutlass.Int64(4096),
        cutlass.Int64(65536),
        cutlass.Int64(264 // word_bytes),
        4,
        1,
        has_gin,
        remote_only,
        make_fake_stream(),
        options="--gpu-arch sm_100a",
    )
    assert result is not None
    assert (
        cute.compile(
            engram._table_barrier,
            make_fake_dev_comm(),
            has_gin,
            make_fake_stream(),
            options="--gpu-arch sm_100a",
        )
        is not None
    )


@pytest.mark.parametrize("local_requests", [False, True])
@pytest.mark.parametrize("data_bytes", [6, 512])
def test_engram_device_lookup_compiles(runtime, local_requests, data_bytes):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from nccl.core.device.cute.runtime import make_fake_dev_comm, make_fake_window

    word_bytes = 4 if data_bytes % 4 == 0 else 1
    dtype = cutlass.Int32 if word_bytes == 4 else cutlass.Uint8
    result = cute.compile(
        engram._lookup,
        make_fake_dev_comm(),
        (make_fake_window(), make_fake_window()),
        (),
        # [tables, row ranks, (offset, stride)], as the count exchange delivers it.
        make_fake_tensor(cutlass.Int64, (2, 4, 2), (8, 2, 1)),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
        make_fake_tensor(dtype, (cute.sym_int64(),), (1,)),
        cutlass.Int64(16),
        data_bytes // word_bytes,
        0,
        local_requests,
        make_fake_stream(),
        options="--gpu-arch sm_100a",
    )
    assert result is not None


@pytest.fixture
def device_group(runtime):
    if not torch.cuda.is_available():
        pytest.skip("Engram device transport needs CUDA")
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("Use torchrun with at least two ranks; cross-domain ranks exercise GIN")
    yield dist.group.WORLD
    torch.cuda.synchronize()
    dist.barrier()


def _transport(group, workspace_bytes):
    """A published transport; tests broadcast the unique id instead of gathering it."""
    unique_id = [engram.create_unique_id() if dist.get_rank(group) == 0 else None]
    dist.broadcast_object_list(unique_id, src=dist.get_global_rank(group, 0), group=group)
    transport = engram.EngramDeviceTransport(
        group, unique_id=unique_id[0], workspace_bytes=workspace_bytes
    )
    try:
        transport.initialize()
        transport.barrier()
    except BaseException:
        transport.close()
        raise
    return transport


def _plan(counts, rank):
    """This rank's plan from a full ``[N, N]`` matrix of rows sent from row to column."""
    send, recv = counts[rank], counts[:, rank]
    return EngramExchangePlan.from_counts(send.cuda(), recv.cuda(), send.tolist(), recv.tolist())


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("stripes", [1, 8])
def test_engram_device_exchange_reuses_slots(device_group, stripes):
    """Real kernels: skew, empty ranks, changes of dtype/row width, and replay."""
    rank, size = dist.get_rank(device_group), dist.get_world_size(device_group)
    # Keep slots deliberately tiny, forcing >2 chunks even for the largest row.
    workspace = (4096 * 2 * (size + min(4, size)) + size * 32) * stripes
    transport = _transport(device_group, workspace)
    assert transport.stripes == stripes
    ok = False
    try:
        for dtype, width in (
            (torch.int64, 2),
            (torch.uint8, 264),
            (torch.bfloat16, 256),
            (torch.float32, 256),
            (torch.uint8, 7),
            (torch.uint8, 1040),
        ):
            for pattern in ("uneven", "hotspot", "empty_requester", "empty", "uneven"):
                splits = [(19 + (rank + peer) % 7) * stripes for peer in range(size)]
                if pattern == "hotspot":
                    splits = [(37 + rank) * stripes] + [0] * (size - 1)
                if pattern == "empty" or (pattern == "empty_requester" and rank == 0):
                    splits = [0] * size
                gathered = [None] * size
                dist.all_gather_object(gathered, splits, group=device_group)
                counts = torch.tensor(gathered)
                recv = counts[:, rank].tolist()
                plan = _plan(counts, rank)
                reverse = plan.reverse()
                data = (
                    torch.arange(sum(splits) * width * dtype.itemsize, device="cuda") + rank * 17
                ).to(torch.uint8)
                data = data.view(dtype).view(sum(splits), width)
                expected = data.new_empty((sum(recv), width))
                dist.all_to_all_single(expected, data, recv, splits, group=device_group)
                for _ in range(3):
                    actual = transport.exchange(data, plan)
                    torch.testing.assert_close(
                        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                    )
                    restored = transport.exchange(actual, reverse)
                    torch.testing.assert_close(
                        restored.view(torch.uint8), data.view(torch.uint8), rtol=0, atol=0
                    )
        ok = True
    finally:
        transport.close(clean=ok)


def _allocator(name, rank):
    def allocate(shape):
        if name == "nccl":
            from nccl.core.interop import torch as nccl_torch

            return nccl_torch.empty(shape, dtype=torch.uint8, device="cuda")
        # DLPack gives the view its own storage, whose base is deliberately not
        # 4-KiB aligned. This reproduces allocator suballocations deterministically.
        size = shape[0] * shape[1]
        parent = torch.empty(size + 4096 + rank * 8192, dtype=torch.uint8, device="cuda")
        start = (512 * (rank + 1) - parent.data_ptr()) % 4096 + rank * 8192
        storage = torch.from_dlpack(parent[start : start + size]).view(shape)
        assert storage.storage_offset() == 0
        assert storage.untyped_storage().data_ptr() % 4096 == 512 * (rank + 1) % 4096
        return storage

    return allocate


def _skip_torch_allocator_without_vmm(allocator):
    if allocator == "torch" and "expandable_segments:True" not in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ):
        pytest.skip("Torch-allocated windows need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def _lookup_counts(transport, key, tables, requests, send):
    """The count exchange of a lookup whose requests are grouped by owner."""
    size = len(send)
    owners = torch.repeat_interleave(torch.arange(size), torch.tensor(send))
    send_counts = torch.zeros(size, len(tables), dtype=torch.int64)
    send_counts.index_put_(
        (owners, requests[:, 0].cpu()), torch.ones(len(owners), dtype=torch.int64), accumulate=True
    )
    return symmetric_lookup.prepare_table_windows(transport, key, tables, send_counts.cuda())


def _expected_rows(tables, requests, send, group):
    size = len(send)
    all_tables = [None] * size
    dist.all_gather_object(all_tables, [table.cpu() for table in tables], group=group)
    expected = [
        all_tables[peer][table][row]
        for (table, row), peer in zip(
            requests.tolist(), torch.repeat_interleave(torch.arange(size), torch.tensor(send))
        )
    ]
    width = tables[0].shape[1]
    return torch.stack(expected) if expected else torch.empty((0, width), dtype=torch.uint8)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("width", [6, 256])
@pytest.mark.parametrize("allocator", ["nccl", "torch"])
def test_engram_device_lookup_refreshes_one_shard(device_group, width, allocator):
    """Read raw rows exactly, including offsets into shared allocations."""
    _skip_torch_allocator_without_vmm(allocator)
    rank, size = dist.get_rank(device_group), dist.get_world_size(device_group)
    transport = _transport(device_group, 2**20)
    allocate = _allocator(allocator, rank)
    row_count = 19
    tables = []
    for table in range(2):
        # A nonzero storage offset exercises DDP-style parameter views. Different
        # ranks deliberately have different offsets.
        storage = allocate((row_count + rank + 1, width))
        view = storage[rank + 1 :]
        view.copy_(
            (torch.arange(view.numel(), device="cuda").view(view.shape) + rank + table * 23).to(
                torch.uint8
            )
        )
        tables.append(view)
    key = object()
    ok = False
    try:
        for update in range(3):
            if update == 1:
                for table in tables:
                    table.add_(1)
            if update == 2 and rank == 0:
                replacement = allocate(tables[0].shape)
                replacement.copy_(tables[0] + 2)
                tables[0] = replacement
            counts = torch.tensor(
                [[3 + (src + dst + update) % 3 for dst in range(size)] for src in range(size)]
            )
            if update == 1:
                # Reuse the compiled executable with singleton/empty requests.
                counts.zero_()
                counts[0, -1] = 1
            send = counts[rank].tolist()
            ids = torch.arange(sum(send), device="cuda")
            requests = torch.stack((ids % 2, ids % row_count), dim=1)
            lookup = _lookup_counts(transport, key, tables, requests, send)
            assert lookup.plan.send_splits == tuple(send)
            assert lookup.plan.recv_splits == tuple(counts[:, rank].tolist())
            # Every rank registers first; later only the rank that replaced a storage changed.
            assert lookup.changed_ranks == {0: tuple(range(size)), 1: (), 2: (0,)}[update]
            expected = _expected_rows(tables, requests, send, device_group)
            actual = transport.lookup(key, tables, requests, lookup.plan, lookup.metadata)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        ok = True
    finally:
        transport.close(clean=ok)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("layout", ["shared", "mixed"])
def test_engram_device_lookup_shares_windows_of_one_storage(device_group, layout):
    """Tables in one allocation share a window when every rank holds them that way."""
    rank, size = dist.get_rank(device_group), dist.get_world_size(device_group)
    transport = _transport(device_group, 2**20)
    allocate = _allocator("nccl", rank)
    row_count, width = 23, 64
    if layout == "shared" or rank == 0:
        # Like a DDP parameter buffer: both tables are views of one storage.
        storage = allocate((2 * row_count + rank + 1, width))
        tables = [storage[rank + 1 : rank + 1 + row_count], storage[rank + 1 + row_count :]]
    else:
        tables = [allocate((row_count, width)) for _ in range(2)]
    for table, view in enumerate(tables):
        view.copy_(
            (torch.arange(view.numel(), device="cuda").view(view.shape) + 5 * rank + table).to(
                torch.uint8
            )
        )
    ok = False
    try:
        counts = torch.tensor(
            [[5 + (src + 2 * dst) % 4 for dst in range(size)] for src in range(size)]
        )
        send = counts[rank].tolist()
        ids = torch.arange(sum(send), device="cuda")
        requests = torch.stack((ids % 2, (ids * 7) % row_count), dim=1)
        # Two lookups over the same tables, like two Engram layers in one DDP buffer.
        keys = (object(), object())
        for key in keys:
            lookup = _lookup_counts(transport, key, tables, requests, send)
            assert len(transport.tables[key].window_ids) == (1 if layout == "shared" else 2)
            expected = _expected_rows(tables, requests, send, device_group)
            actual = transport.lookup(key, tables, requests, lookup.plan, lookup.metadata)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if layout == "shared":
            # The second lookup reused the first one's window.
            assert len(transport.windows) == 1
        else:
            # Rank 0 holds both tables in one storage, so it names that storage's newest
            # window (1) for both, while its peers name 0 and 1. Only the second block
            # agrees: the second lookup reuses window 1 and registers window 2.
            assert [transport.tables[key].window_ids for key in keys] == [(0, 1), (2, 1)]
            assert len(transport.windows) == 3
        ok = True
    finally:
        transport.close(clean=ok)
