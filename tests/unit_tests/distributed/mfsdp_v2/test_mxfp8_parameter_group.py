# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``Fp8ParameterGroup``'s bf16-aligned payload storage.

``Fp8ParameterGroup`` mirrors ``FsdpParameterGroup``: the fp8 payload *storage*
rests in the parameter layout and exposes the optimizer layout as a view
(``post_optimizer_rowwise`` / ``post_optimizer_colwise``), the analogue of
``post_optimizer_model_weight``. Two properties are load-bearing:

- every ``view`` / ``redistribute`` move changes at most one mesh axis, so the
  existing single-axis ``DBuffer`` machinery handles HFSDP dense and expert
  ZeRO-1 without any multi-axis support; and
- the TE amax reduce group is derived from the quantization **source**
  (``main_weight.placements``), never from the payload storage. Under HFSDP those
  differ (``[Replicate, Shard]`` payload over ``[Shard, Shard]`` masters), and a
  payload-derived group would silently under-report the amax.

Transformer Engine and a real ``MXFP8Tensor`` are not needed for either: the
placement/bookkeeping behaviour lives on ``DBuffer`` plus the group's placements,
and the quantization call site is exercised with the TE cast stubbed.
"""

import types

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import parameter_group
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer import (
    _compute_weight_local_views,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    Fp8ParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    Flat,
    changed_mesh_axis,
    sharded_reduce_group,
)

TENSOR_SHAPES = ((256, 128),)
HFSDP_DENSE = ((Replicate(), Flat()), (Flat(), Flat()))
EXPERT_ZERO1 = ((Replicate(),), (Flat(),))


def _two_axis_mesh(distributed_setup):
    return init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_shard"),
    )


def _one_axis_mesh(distributed_setup):
    return init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))


def _require_world_size(distributed_setup, minimum):
    if distributed_setup.world_size < minimum:
        pytest.skip(f"This test requires at least {minimum} ranks.")


def _require_two_axis(distributed_setup):
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D placement test requires an even world size of at least 4.")


def _make_payload(mesh, parameter_placements, optimizer_placements, device):
    """Build a payload storage/view pair exactly as the fp8 group does."""
    storage = DBuffer.empty(
        mesh=mesh,
        placements=parameter_placements,
        tensor_shapes=TENSOR_SHAPES,
        dtype=torch.uint8,
        device=device,
    )
    return storage, storage.view(optimizer_placements)


class _UnshardStub(Fp8ParameterGroup):
    """An Fp8ParameterGroup whose __init__ is skipped, for the unshard path."""

    def _switch_to_unsharded_parameters(self) -> None:
        pass


def _make_unshard_stub(mesh, parameter_placements, optimizer_placements, device):
    ro_storage, ro_view = _make_payload(mesh, parameter_placements, optimizer_placements, device)
    co_storage, co_view = _make_payload(mesh, parameter_placements, optimizer_placements, device)
    stub = _UnshardStub.__new__(_UnshardStub)
    stub.mesh = mesh
    stub._symm_mem_pool = None  # _symmetric_memory_context() -> nullcontext()
    stub.fsdp_parameters = tuple(
        types.SimpleNamespace(unsharded=torch.zeros(shape)) for shape in TENSOR_SHAPES
    )
    stub._rowwise_buffer = ro_storage
    stub._colwise_buffer = co_storage
    stub.post_optimizer_rowwise = ro_view
    stub.post_optimizer_colwise = co_view
    stub._rowwise_is_stale = True
    stub._colwise_is_stale = True
    stub._unsharded_rowwise = DBuffer.empty(
        mesh=mesh,
        placements=(Replicate(),) * mesh.ndim,
        tensor_shapes=TENSOR_SHAPES,
        dtype=torch.uint8,
        device=device,
    )
    stub._unsharded_colwise = DBuffer.empty(
        mesh=mesh,
        placements=(Replicate(),) * mesh.ndim,
        tensor_shapes=TENSOR_SHAPES,
        dtype=torch.uint8,
        device=device,
    )
    return stub, ro_storage, ro_view, co_storage, co_view


def test_sharded_reduce_group_spans_masters_not_payload(distributed_setup):
    """HFSDP pins the amax group to the flattened masters, not the payload storage.

    Payload ``[Replicate, Shard]`` over masters ``[Shard, Shard]``: the payload's
    own inner-axis group does not see the outer axis's disjoint master blocks, so
    deriving the group from the payload would silently produce an amax that is too
    small and fp8 scales that are too large.
    """
    _require_two_axis(distributed_setup)
    mesh = _two_axis_mesh(distributed_setup)
    master_placements, payload_placements = HFSDP_DENSE[1], HFSDP_DENSE[0]
    # The MFSDP adapter records the flattened DP group on a hybrid mesh; mirror that
    # here so the assertion does not depend on DeviceMesh._flatten().
    flattened_group = dist.new_group(ranks=sorted(mesh.mesh.flatten().tolist()))
    mesh._mfsdp_flattened_group = flattened_group

    reduce_group = sharded_reduce_group(mesh, master_placements)
    assert reduce_group is flattened_group
    assert sorted(dist.get_process_group_ranks(reduce_group)) == sorted(
        mesh.mesh.flatten().tolist()
    )

    # The exact mistake this guards against: the payload-derived group is a strictly
    # smaller subgroup, so the two derivations must not be interchangeable.
    payload_group = sharded_reduce_group(mesh, payload_placements)
    assert list(dist.get_process_group_ranks(payload_group)) == list(
        dist.get_process_group_ranks(mesh.get_group(1))
    )
    assert len(list(dist.get_process_group_ranks(payload_group))) < len(
        list(dist.get_process_group_ranks(reduce_group))
    )


def test_sharded_reduce_group_fallbacks_and_rejections(distributed_setup):
    """The no-shard fallback and the unsupported-placement error stay explicit."""
    _require_world_size(distributed_setup, 2)
    mesh = _one_axis_mesh(distributed_setup)

    # Fully replicated: every rank holds an identical copy, so the MAX is idempotent
    # and this mesh's own axis 0 is the documented fallback.
    assert list(dist.get_process_group_ranks(sharded_reduce_group(mesh, (Replicate(),)))) == list(
        dist.get_process_group_ranks(mesh.get_group(0))
    )
    # One sharded axis uses that axis's group.
    assert list(dist.get_process_group_ranks(sharded_reduce_group(mesh, (Flat(),)))) == list(
        dist.get_process_group_ranks(mesh.get_group(0))
    )
    # A Partial axis is neither a replicated copy nor a disjoint shard.
    with pytest.raises(NotImplementedError, match="Unsupported placement"):
        sharded_reduce_group(mesh, (Partial("avg"),))


def test_quantize_call_site_uses_masters_derived_reduce_group(distributed_setup, monkeypatch):
    """The quantization call site itself must pass the masters-derived group.

    This runs the real ``_quantize_model_weight_from_main_weight`` body with the
    TE cast stubbed, so it pins the call site and the copy target rather than only
    the ``_amax_reduce_group`` helper. A revert to the payload placements captures
    the inner-axis group and fails; a revert of the copy target to the storage
    fails on a shape mismatch (the storage is ``[Replicate, Shard]`` here).
    """
    _require_two_axis(distributed_setup)
    mesh = _two_axis_mesh(distributed_setup)
    parameter_placements, optimizer_placements = HFSDP_DENSE
    # The adapter records the flattened DP group on a hybrid mesh; mirror that so
    # the masters-derived group resolves to the flattened group here too.
    flattened_group = dist.new_group(ranks=sorted(mesh.mesh.flatten().tolist()))
    mesh._mfsdp_flattened_group = flattened_group

    main_weight = DBuffer.empty(
        mesh=mesh,
        placements=optimizer_placements,
        tensor_shapes=TENSOR_SHAPES,
        dtype=torch.float32,
        device=distributed_setup.device,
    )
    main_weight.local_buffer.fill_(3.0)
    ro_storage, ro_view = _make_payload(
        mesh, parameter_placements, optimizer_placements, distributed_setup.device
    )
    co_storage, co_view = _make_payload(
        mesh, parameter_placements, optimizer_placements, distributed_setup.device
    )

    stub = _UnshardStub.__new__(_UnshardStub)
    stub.mesh = mesh
    stub.main_weight = main_weight
    stub.fsdp_parameters = (types.SimpleNamespace(unsharded=torch.zeros(TENSOR_SHAPES[0])),)
    # Present so a reverted call site reads the payload placements instead of
    # crashing on a missing attribute, and the assertion below can catch it.
    stub._rowwise_buffer = ro_storage
    stub._colwise_buffer = co_storage
    stub.post_optimizer_rowwise = ro_view
    stub.post_optimizer_colwise = co_view

    captured = {}

    def fake_cast(*, model_weights, master_weights, start_offsets, group, fsdp_shard_model_weights):
        captured["group"] = group
        captured["master_numel"] = [
            None if master is None else master.numel() for master in master_weights
        ]
        for master, (ro_slice, co_slice) in zip(master_weights, fsdp_shard_model_weights):
            if master is None:
                continue
            ro_slice.copy_(master.to(torch.uint8))
            co_slice.copy_((master.to(torch.uint8) + 1) % 251)

    class _FakeTemp:
        def __init__(self, height, width, device):
            self._rowwise_data = torch.zeros((height, width), dtype=torch.uint8, device=device)
            self._columnwise_data = torch.zeros((height, width), dtype=torch.uint8, device=device)

        @property
        def shape(self):
            return self._rowwise_data.shape

    monkeypatch.setattr(parameter_group, "te_cast_master_weights_to_fp8", lambda: fake_cast)
    monkeypatch.setattr(
        parameter_group,
        "allocate_quantize_temp",
        lambda tensor, height, width, device: _FakeTemp(height, width, device),
    )

    Fp8ParameterGroup._quantize_model_weight_from_main_weight(stub)

    assert captured["group"] is flattened_group
    assert captured["group"] is sharded_reduce_group(mesh, optimizer_placements)
    assert captured["group"] is not sharded_reduce_group(mesh, parameter_placements)
    assert captured["master_numel"] == [main_weight.get_local_tensor(0).numel()]

    # The quantized shard landed in the optimizer-layout view, whose shape differs
    # from the coarser storage shard, so the copy target cannot be the storage.
    expected = main_weight.get_local_tensor(0).to(torch.uint8)
    assert ro_view.get_local_tensor(0).shape == expected.shape
    assert ro_view.get_local_tensor(0).shape != ro_storage.get_local_tensor(0).shape
    assert torch.equal(ro_view.get_local_tensor(0), expected)
    assert torch.equal(co_view.get_local_tensor(0), (expected + 1) % 251)


@pytest.mark.parametrize(
    "name, placements, expected_axes",
    [
        pytest.param("hfsdp-dense", HFSDP_DENSE, (0, 0, 1, 1), id="hfsdp-dense"),
        pytest.param("expert-zero1", EXPERT_ZERO1, (0, 0), id="expert-zero1"),
    ],
)
def test_unshard_moves_change_at_most_one_axis(
    distributed_setup, monkeypatch, name, placements, expected_axes
):
    """Every view/redistribute on the unshard path changes at most one mesh axis.

    ``changed_mesh_axis`` raises when two axes change, so recording its result on
    each real call proves the alignment removed the multi-axis move: HFSDP dense
    only widens the outer axis into the storage and the inner axis to everything
    Replicate, and expert ZeRO-1 only widens its single optimizer axis.
    """
    parameter_placements, optimizer_placements = placements
    if len(parameter_placements) == 2:
        _require_two_axis(distributed_setup)
        mesh = _two_axis_mesh(distributed_setup)
    else:
        _require_world_size(distributed_setup, 2)
        mesh = _one_axis_mesh(distributed_setup)

    # Build before instrumenting, so construction's own view() call is not recorded.
    stub, *_ = _make_unshard_stub(
        mesh, parameter_placements, optimizer_placements, distributed_setup.device
    )

    recorded_axes = []
    original_view = DBuffer.view
    original_redistribute = DBuffer.redistribute

    def recording_view(self, new_placements):
        recorded_axes.append(changed_mesh_axis(self.placements, new_placements))
        return original_view(self, new_placements)

    def recording_redistribute(self, new_placements, *, out=None):
        recorded_axes.append(changed_mesh_axis(self.placements, new_placements))
        return original_redistribute(self, new_placements, out=out)

    monkeypatch.setattr(DBuffer, "view", recording_view)
    monkeypatch.setattr(DBuffer, "redistribute", recording_redistribute)

    def fake_all_gather_into_tensor(output_tensor, input_tensor, group):
        chunk = input_tensor.numel()
        for index in range(output_tensor.numel() // chunk):
            output_tensor.narrow(0, index * chunk, chunk).copy_(input_tensor)

    # Only the placement bookkeeping is under test, not the communication, so the
    # collective and the TE payload binding are stubbed out.
    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(parameter_group, "set_rowwise_payload", lambda tensor, data: None)
    monkeypatch.setattr(parameter_group, "set_columnwise_payload", lambda tensor, data: None)

    Fp8ParameterGroup.unshard_parameters(stub)

    # changed_mesh_axis would have raised inside the recorders on any two-axis move.
    assert all(axis in (0, 1, None) for axis in recorded_axes), recorded_axes
    assert tuple(recorded_axes) == expected_axes, recorded_axes
    assert stub._rowwise_is_stale is False
    assert stub._colwise_is_stale is False

    # And the exact placement pairs the two FP8 moves use are single-axis.
    for source, target in (
        (optimizer_placements, parameter_placements),
        (parameter_placements, (Replicate(),) * mesh.ndim),
    ):
        changed_axis = changed_mesh_axis(source, target)
        assert changed_axis is None or changed_axis in range(mesh.ndim)


@pytest.mark.parametrize(
    "parameter_placements, optimizer_placements",
    [
        pytest.param(*HFSDP_DENSE, id="hfsdp-dense"),
        pytest.param((Replicate(), Replicate()), (Replicate(), Flat()), id="zero1-inner-shard"),
    ],
)
def test_payload_storage_is_parameter_layout_with_optimizer_view(
    distributed_setup, parameter_placements, optimizer_placements
):
    """The view is a shard-shaped slice of the parameter-layout payload storage."""
    _require_two_axis(distributed_setup)
    mesh = _two_axis_mesh(distributed_setup)
    storage, view = _make_payload(
        mesh, parameter_placements, optimizer_placements, distributed_setup.device
    )

    assert storage.placements == parameter_placements
    assert view.placements == optimizer_placements
    assert view is not storage

    # The view aliases the storage allocation at exactly the optimizer-layout local
    # offset, so quantization writes land inside the payload storage.
    expected_offset, expected_numel = storage.layout.get_local_range(mesh, optimizer_placements)
    assert view.offset == expected_offset
    assert view.local_buffer.numel() == expected_numel
    assert view.local_buffer.data_ptr() == (
        storage.local_buffer.data_ptr()
        + (expected_offset - storage.offset) * view.local_buffer.element_size()
    )
    assert view.local_buffer.numel() < storage.local_buffer.numel()
    # Shard-shaped copy target: full row width, fewer rows than the storage's shard.
    assert view.get_local_tensor(0).shape == (
        expected_numel // TENSOR_SHAPES[0][1],
        TENSOR_SHAPES[0][1],
    )
    assert storage.get_local_tensor(0).shape[1] == view.get_local_tensor(0).shape[1]

    # The staleness predicate is exactly "view placements != storage placements".
    stub = types.SimpleNamespace(
        _rowwise_buffer=storage,
        _colwise_buffer=storage,
        post_optimizer_rowwise=view,
        post_optimizer_colwise=view,
    )
    Fp8ParameterGroup._update_payload_staleness(stub)
    assert stub._rowwise_is_stale is True
    assert stub._colwise_is_stale is True


def test_zero3_view_is_identity_and_never_stale(distributed_setup):
    """ZeRO-3 keeps storage and view on the same placements, so the view aliases it."""
    _require_world_size(distributed_setup, 2)
    mesh = _one_axis_mesh(distributed_setup)
    placements = (Flat(),)
    storage, view = _make_payload(mesh, placements, placements, distributed_setup.device)

    assert view is storage
    stub = types.SimpleNamespace(
        _rowwise_buffer=storage,
        _colwise_buffer=storage,
        post_optimizer_rowwise=view,
        post_optimizer_colwise=view,
    )
    Fp8ParameterGroup._update_payload_staleness(stub)
    assert stub._rowwise_is_stale is False
    assert stub._colwise_is_stale is False


def test_optimizer_checks_payload_views_not_storage(distributed_setup):
    """The Muon shard-shape check must compare the views, not the coarser storage.

    ``_require_matching_local_shards`` compares each compute-weight view's local
    shape against the sharded parameter's. With the payload storage in the
    parameter layout the storage is *not* shard-shaped under HFSDP or expert
    ZeRO-1, so the check must read ``post_optimizer_rowwise`` /
    ``post_optimizer_colwise``.
    """
    _require_two_axis(distributed_setup)
    mesh = _two_axis_mesh(distributed_setup)
    storage, view = _make_payload(mesh, *HFSDP_DENSE, distributed_setup.device)
    group = types.SimpleNamespace(post_optimizer_rowwise=view, post_optimizer_colwise=view)

    assert [name for name, _ in _compute_weight_local_views(group)] == [
        "post_optimizer_rowwise",
        "post_optimizer_colwise",
    ]
    assert all(buffer is view for _, buffer in _compute_weight_local_views(group))
    assert storage.get_local_tensor(0).shape != view.get_local_tensor(0).shape
