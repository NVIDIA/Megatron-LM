# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compile checks and real device execution checks for Engram communication."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.engram import distributed_embedding as embedding_impl
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("rows", [0, 1, 17])
def test_engram_dynamic_dlpack_handles_empty_and_singleton_rows(rows):
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    from megatron.core.ops.engram import _rows_tensor

    # PyTorch DLPack canonicalizes the first stride to 1 for a 0- or 1-row
    # matrix; flattening must accept those just like a normal request matrix.
    requests = torch.empty(rows, 2, dtype=torch.int64)
    result = _rows_tensor(requests)
    assert str(result).endswith("o (?):(1)>")


@pytest.mark.parametrize("storage_offset", [0, 512, 3584, 4608])
@pytest.mark.parametrize("view_offset", [0, 13])
def test_engram_window_range_includes_unaligned_storage(monkeypatch, storage_offset, view_offset):
    """Storage and tensor offsets both contribute to the registered window offset."""
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    from megatron.core.ops import engram

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


@pytest.mark.parametrize("change", ["none", "local", "remote"])
def test_engram_table_window_refresh_is_collective(monkeypatch, change):
    """A change on one shard refreshes all windows and remote view metadata."""
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    from megatron.core.ops.engram import EngramDeviceTransport

    transport = EngramDeviceTransport.__new__(EngramDeviceTransport)
    transport.group, transport.size, transport.tables = object(), 2, {}
    transport.barrier = Mock()
    windows = []

    def register(view):
        window = SimpleNamespace(is_valid=True, close=Mock())
        windows.append(window)
        return window

    transport.comm = SimpleNamespace(register_window=register)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=Mock()))
    # Mock only CUDA allocation discovery; retain real storage lifetime/signatures.
    describe = Mock(
        side_effect=lambda tensor: (
            SimpleNamespace(owner=tensor.untyped_storage()),
            tensor.storage_offset() * tensor.element_size(),
            tensor.stride(0) * tensor.element_size(),
        )
    )
    monkeypatch.setattr(transport, "_storage_view", describe)
    remote_changed = False

    def gather(result, info, group):
        assert group is transport.group
        result[0] = info
        # Exercise an unaligned remote view, even when this rank's view is aligned.
        result[1] = (remote_changed, [(2 if remote_changed else 0, 32)], None)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    data = [torch.zeros(3, 8)]
    key = object()
    first, _, aligned = transport._register_tables(key, data, [])
    assert aligned
    old_pointer = data[0].untyped_storage().data_ptr()
    assert transport.tables[key][2][0][0].owner.data_ptr() == old_pointer
    if change == "local":
        data[0] = data[0].clone()
    remote_changed = change == "remote"
    second, metadata, aligned = transport._register_tables(key, data, [])
    assert len(windows) == (1 if change == "none" else 2)
    assert first[0].close.call_count == (change != "none")
    assert (second is first) == (change == "none")
    assert aligned == (change != "remote")
    assert metadata[0, 1, 0] == (2 if remote_changed else 0)
    assert describe.call_count == (2 if change == "local" else 1)


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_engram_window_preparation_failure_is_collective(monkeypatch, failed_rank):
    """A bad allocation on one rank must not leave peers entering registration."""
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    from megatron.core.ops.engram import EngramDeviceTransport

    transport = EngramDeviceTransport.__new__(EngramDeviceTransport)
    transport.group, transport.size, transport.tables = object(), 2, {}
    transport.barrier = Mock()
    transport.comm = SimpleNamespace(register_window=Mock())
    describe = Mock(return_value=(object(), 0, 32))
    if failed_rank == 0:
        describe.side_effect = RuntimeError("allocation query failed")
    monkeypatch.setattr(transport, "_storage_view", describe)

    def gather(result, info, group):
        result[0] = info
        result[1] = (True, [(0, 32)], "allocation query failed" if failed_rank == 1 else None)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match=f"row rank {failed_rank}: .*allocation query failed"):
        transport._register_tables(object(), [torch.zeros(3, 8)], [])
    transport.comm.register_window.assert_not_called()
    transport.barrier.assert_not_called()


def test_engram_unsharded_transport_does_not_use_world(monkeypatch):
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    from megatron.core.ops.engram import EngramDeviceTransport

    def unexpected(*args):
        raise AssertionError("None denotes an unsharded table, not the default process group")

    monkeypatch.setattr(dist, "get_rank", unexpected)
    monkeypatch.setattr(dist, "get_world_size", unexpected)
    transport = EngramDeviceTransport(None)
    assert (transport.rank, transport.size) == (0, 1)


@pytest.mark.parametrize("has_gin,remote_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("word_bytes", [1, 4])
def test_engram_device_exchange_compiles(has_gin, remote_only, word_bytes):
    """Compile actual NCCL bitcode for Blackwell, including both signal protocols."""
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from nccl.core.device.cute.runtime import make_fake_dev_comm, make_fake_window

    from megatron.core.ops.engram import _exchange, _table_barrier

    dtype = cutlass.Int32 if word_bytes == 4 else cutlass.Uint8
    result = cute.compile(
        _exchange,
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
            _table_barrier,
            make_fake_dev_comm(),
            has_gin,
            make_fake_stream(),
            options="--gpu-arch sm_100a",
        )
        is not None
    )


@pytest.mark.parametrize("local_requests", [False, True])
@pytest.mark.parametrize("data_bytes,scale_bytes", [(6, 0), (512, 0), (256, 8)])
def test_engram_device_lookup_compiles(local_requests, data_bytes, scale_bytes):
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from nccl.core.device.cute.runtime import make_fake_dev_comm, make_fake_window

    from megatron.core.ops.engram import _lookup

    word_bytes = 4 if data_bytes % 4 == scale_bytes % 4 == 0 else 1
    dtype = cutlass.Int32 if word_bytes == 4 else cutlass.Uint8
    windows = (make_fake_window(), make_fake_window())
    result = cute.compile(
        _lookup,
        make_fake_dev_comm(),
        windows,
        windows if scale_bytes else (),
        make_fake_tensor(cutlass.Int64, (2, 4, 4), (16, 4, 1)),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(),), (1,)),
        make_fake_tensor(dtype, (cute.sym_int64(),), (1,)),
        cutlass.Int64(16),
        data_bytes // word_bytes,
        scale_bytes // word_bytes,
        local_requests,
        make_fake_stream(),
        options="--gpu-arch sm_100a",
    )
    assert result is not None


@pytest.fixture
def device_group():
    if not torch.cuda.is_available():
        pytest.skip("Engram device transport needs CUDA")
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("Use torchrun with at least two ranks; cross-domain ranks exercise GIN")
    yield dist.group.WORLD
    torch.cuda.synchronize()
    dist.barrier()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("stripes", [1, 8])
def test_engram_device_exchange_reuses_slots(device_group, stripes):
    """Real kernels: skew, empty ranks, changes of dtype/row width, and replay."""
    from megatron.core.ops.engram import EngramDeviceTransport

    transport = EngramDeviceTransport(device_group)
    rank, size = dist.get_rank(device_group), dist.get_world_size(device_group)
    # Keep slots deliberately tiny, forcing >2 chunks even for the largest row.
    workspace = (4096 * 2 * (size + transport.channels) + size * 32) * stripes
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
                plan = SimpleNamespace(peer_counts=counts, max_rows=int(counts.sum(1).max()))
                reverse = SimpleNamespace(peer_counts=counts.T, max_rows=int(counts.sum(0).max()))
                data = (
                    torch.arange(sum(splits) * width * dtype.itemsize, device="cuda") + rank * 17
                ).to(torch.uint8)
                data = data.view(dtype).view(sum(splits), width)
                expected = data.new_empty((sum(recv), width))
                dist.all_to_all_single(expected, data, recv, splits, group=device_group)
                for _ in range(3):
                    actual = transport.exchange(data, recv, splits, plan, workspace)
                    torch.testing.assert_close(
                        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                    )
                    restored = transport.exchange(actual, splits, recv, reverse, workspace)
                    torch.testing.assert_close(
                        restored.view(torch.uint8), data.view(torch.uint8), rtol=0, atol=0
                    )
    finally:
        transport.close()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("fp8_bytes", [False, True])
@pytest.mark.parametrize("allocator", ["nccl", "torch"])
def test_engram_device_lookup_refreshes_one_shard(device_group, fp8_bytes, allocator):
    """Read raw data/scales exactly, including offsets into shared allocations."""
    from nccl.core.interop import torch as nccl_torch

    from megatron.core.ops.engram import EngramDeviceTransport

    rank, size = dist.get_rank(device_group), dist.get_world_size(device_group)
    transport = EngramDeviceTransport(device_group)
    width = 256 if fp8_bytes else 6
    row_count = 19
    data, scales = [], []

    def allocate(shape):
        if allocator == "nccl":
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

    for table in range(2):
        # A nonzero storage offset exercises DDP-style parameter views. Different
        # ranks deliberately have different offsets and scale strides.
        storage = allocate((row_count + rank + 1, width))
        view = storage[rank + 1 :]
        view.copy_(
            (torch.arange(view.numel(), device="cuda").view(view.shape) + rank + table * 23).to(
                torch.uint8
            )
        )
        data.append(view)
        if fp8_bytes:
            storage = allocate((row_count + 1, 8 + rank * 4))
            storage.fill_(rank + table + 120)
            scales.append(storage[1:])
    module_key = object()
    try:
        for update in range(3):
            if update == 1:
                for table in data:
                    table.add_(1)
            if update == 2 and rank == 0:
                replacement = allocate(data[0].shape)
                replacement.copy_(data[0] + 2)
                data[0] = replacement
            counts = torch.tensor(
                [[3 + (src + dst + update) % 3 for dst in range(size)] for src in range(size)]
            )
            if update == 1:
                # Reuse the compiled executable with singleton/empty requests.
                counts.zero_()
                counts[0, -1] = 1
            send, recv = counts[rank].tolist(), counts[:, rank].tolist()
            plans = (
                SimpleNamespace(peer_counts=counts, max_rows=int(counts.sum(1).max())),
                SimpleNamespace(peer_counts=counts.T, max_rows=int(counts.sum(0).max())),
            )
            ids = torch.arange(sum(send), device="cuda")
            requests = torch.stack((ids % 2, ids % row_count), dim=1)
            all_data, all_scales = [None] * size, [None] * size
            dist.all_gather_object(all_data, [tensor.cpu() for tensor in data], group=device_group)
            dist.all_gather_object(
                all_scales, [tensor.cpu() for tensor in scales], group=device_group
            )
            expected = []
            for index, peer in enumerate(
                torch.repeat_interleave(torch.arange(size), torch.tensor(send))
            ):
                table, row = requests[index].tolist()
                parts = [all_data[peer][table][row]]
                if scales:
                    parts.append(all_scales[peer][table][row, :8])
                expected.append(torch.cat(parts))
            actual = transport.lookup(module_key, data, scales, requests, send, recv, plans, 2**20)
            reference = (
                torch.stack(expected)
                if expected
                else torch.empty((0, width + (8 if fp8_bytes else 0)), dtype=torch.uint8)
            )
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    finally:
        transport.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton row accumulation needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [3, 32, 256])
def test_engram_row_accumulation_matches_fp64(dtype, width):
    generator = torch.Generator().manual_seed(171)
    # Include a segment larger than a CTA tile and segments crossing tile boundaries.
    rows = torch.randint(0, 11, (317,), generator=generator)
    gradients = torch.randn(317, width, generator=generator).to(dtype)
    initial = torch.randn(15, width, generator=generator).to(dtype)
    expected = initial.double()
    expected.index_add_(0, rows, gradients.double())
    target = initial.cuda()
    # Test non-contiguous row IDs and gradient row strides, as in a received packet.
    ids = torch.stack((rows, rows), dim=1).cuda()[:, 1]
    packet = torch.zeros(317, width + 4, dtype=dtype, device="cuda")
    packet[:, :width].copy_(gradients)
    values = packet[:, :width]
    embedding_impl._accumulate_embedding_rows(target, ids, values)
    torch.testing.assert_close(target.cpu(), expected.to(dtype), rtol=2e-5, atol=2e-5)
    replay = initial.cuda()
    embedding_impl._accumulate_embedding_rows(replay, ids, values)
    assert torch.equal(target, replay)
    assert torch.equal(target[11:].cpu(), initial[11:])
