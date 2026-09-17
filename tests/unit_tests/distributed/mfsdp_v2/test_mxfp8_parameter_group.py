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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    COLWISE,
    ROWWISE,
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


class _FakeQuantizer:
    """The ``Quantizer`` surface the unshard path touches, without TE."""

    def __init__(self):
        self.rowwise_usage = True
        self.columnwise_usage = True

    def set_usage(self, rowwise=None, columnwise=None):
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise


class _FakeFp8Tensor:
    """The ``MXFP8Tensor`` surface the unshard path touches, without TE.

    ``_rowwise_scale_inv`` / ``_columnwise_scale_inv`` are the grids TE fills at
    quantization and *drops* from any direction ``update_usage`` disables, and
    ``_quantizer`` carries the usage flags TE propagates into ``update_usage``.
    Both are what the orientation-aware unshard has to manage.
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = torch.zeros(4, dtype=torch.float32)
        self._columnwise_scale_inv = torch.zeros(4, dtype=torch.float32)
        self._quantizer = _FakeQuantizer()


def _make_unshard_stub(mesh, parameter_placements, optimizer_placements, device):
    ro_storage, ro_view = _make_payload(mesh, parameter_placements, optimizer_placements, device)
    co_storage, co_view = _make_payload(mesh, parameter_placements, optimizer_placements, device)
    stub = _UnshardStub.__new__(_UnshardStub)
    stub.mesh = mesh
    stub._symm_mem_pool = None  # _symmetric_memory_context() -> nullcontext()
    stub.fsdp_parameters = tuple(
        types.SimpleNamespace(unsharded=_FakeFp8Tensor(shape)) for shape in TENSOR_SHAPES
    )
    stub._rowwise_buffer = ro_storage
    stub._colwise_buffer = co_storage
    stub.post_optimizer_rowwise = ro_view
    stub.post_optimizer_colwise = co_view
    stub._rowwise_is_stale = True
    stub._colwise_is_stale = True
    stub._materialized_directions = frozenset()
    stub._rowwise_scale_invs = None
    stub._colwise_scale_invs = None
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
    "name, placements, per_direction_axes",
    [
        pytest.param("hfsdp-dense", HFSDP_DENSE, (0, 1), id="hfsdp-dense"),
        pytest.param("expert-zero1", EXPERT_ZERO1, (0,), id="expert-zero1"),
    ],
)
@pytest.mark.parametrize("orientation", ["rowwise", "colwise", "both"])
def test_unshard_moves_change_at_most_one_axis(
    distributed_setup, monkeypatch, name, placements, per_direction_axes, orientation
):
    """Every view/redistribute on the unshard path changes at most one mesh axis.

    ``changed_mesh_axis`` raises when two axes change, so recording its result on
    each real call proves the alignment removed the multi-axis move: HFSDP dense
    only widens the outer axis into the storage and the inner axis to everything
    Replicate, and expert ZeRO-1 only widens its single optimizer axis.

    Only the orientations actually requested are gathered, so the recorded move
    sequence repeats once per requested direction, and only those directions'
    staleness is cleared.
    """
    parameter_placements, optimizer_placements = placements
    directions = {"rowwise": (ROWWISE,), "colwise": (COLWISE,), "both": (ROWWISE, COLWISE)}[
        orientation
    ]
    expected_axes = tuple(per_direction_axes) * len(directions)
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
    # collective is stubbed out; the payload binders only record what was bound.
    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(
        parameter_group,
        "set_rowwise_payload",
        lambda tensor, data: setattr(tensor, "_rowwise_data", data),
    )
    monkeypatch.setattr(
        parameter_group,
        "set_columnwise_payload",
        lambda tensor, data: setattr(tensor, "_columnwise_data", data),
    )

    Fp8ParameterGroup.unshard_parameters(stub, orientation)

    # changed_mesh_axis would have raised inside the recorders on any two-axis move.
    assert all(axis in (0, 1, None) for axis in recorded_axes), recorded_axes
    assert tuple(recorded_axes) == expected_axes, recorded_axes
    # Only the requested directions were redistributed out of the optimizer view,
    # so only their staleness is cleared; the other one waits for its own unshard.
    assert stub._rowwise_is_stale is (ROWWISE not in directions)
    assert stub._colwise_is_stale is (COLWISE not in directions)
    # Exactly the requested payloads were bound, and TE's usage flags were kept in
    # step with them so ``update_usage`` never asks for a payload that is absent.
    for fsdp_parameter in stub.fsdp_parameters:
        tensor = fsdp_parameter.unsharded
        assert (tensor._rowwise_data is not None) is (ROWWISE in directions)
        assert (tensor._columnwise_data is not None) is (COLWISE in directions)
        assert tensor._quantizer.rowwise_usage is (ROWWISE in directions)
        assert tensor._quantizer.columnwise_usage is (COLWISE in directions)

    # And the exact placement pairs the two FP8 moves use are single-axis.
    for source, target in (
        (optimizer_placements, parameter_placements),
        (parameter_placements, (Replicate(),) * mesh.ndim),
    ):
        changed_axis = changed_mesh_axis(source, target)
        assert changed_axis is None or changed_axis in range(mesh.ndim)


def test_restore_scale_inverses_reattaches_grids_te_dropped(distributed_setup):
    """A narrowed unshard re-attaches the scale-inverse grids TE dropped.

    ``MXFP8Tensor.update_usage`` sets a disabled direction's grid to ``None``, so
    after a forward-only materialization TE has dropped the column-wise grid. The
    group has to bring it back itself, or ``update_usage(rowwise_usage=True)`` and
    TE's quantization workspaces (which alias the grids) fail on the next step.
    """
    _require_world_size(distributed_setup, 2)
    mesh = _one_axis_mesh(distributed_setup)
    stub, *_ = _make_unshard_stub(mesh, *EXPERT_ZERO1, distributed_setup.device)
    tensor = stub.fsdp_parameters[0].unsharded
    rowwise_scale_inv = tensor._rowwise_scale_inv
    colwise_scale_inv = tensor._columnwise_scale_inv

    # Construction leaves both grids present; the first restore caches them.
    Fp8ParameterGroup._restore_scale_inverses(stub)
    assert stub._rowwise_scale_invs == (rowwise_scale_inv,)
    assert stub._colwise_scale_invs == (colwise_scale_inv,)

    # TE's ``update_usage`` drops every direction it disables.
    tensor._rowwise_scale_inv = None
    tensor._columnwise_scale_inv = None
    Fp8ParameterGroup._restore_scale_inverses(stub)
    assert tensor._rowwise_scale_inv is rowwise_scale_inv
    assert tensor._columnwise_scale_inv is colwise_scale_inv

    # A grid that is still attached is left exactly as it is.
    replacement = torch.ones(4, dtype=torch.float32)
    tensor._rowwise_scale_inv = replacement
    Fp8ParameterGroup._restore_scale_inverses(stub)
    assert tensor._rowwise_scale_inv is replacement
    assert tensor._columnwise_scale_inv is colwise_scale_inv


def test_unshard_widens_resident_rowwise_materialization(distributed_setup, monkeypatch):
    """A second, wider unshard gathers only the direction still missing.

    A module can be materialized row-wise and then asked for the column-wise
    payload with no reshard in between (activation recomputation runs a forward
    between ``pre_backward`` and ``post_backward``; the module widens it to
    ``"both"``). Re-running the row-wise collective for a payload that is already
    bound would give back the volume this change exists to remove, so only the
    missing direction may be gathered.
    """
    _require_world_size(distributed_setup, 2)
    mesh = _one_axis_mesh(distributed_setup)
    stub, *_ = _make_unshard_stub(mesh, *EXPERT_ZERO1, distributed_setup.device)
    # The optimizer-view-to-storage move is not under test here.
    stub._rowwise_is_stale = False
    stub._colwise_is_stale = False

    gathered = []
    monkeypatch.setattr(
        Fp8ParameterGroup,
        "_gather_payload",
        lambda self, source, target: gathered.append(
            ROWWISE if source is self._rowwise_buffer else COLWISE
        ),
    )
    monkeypatch.setattr(parameter_group, "set_rowwise_payload", lambda tensor, data: None)
    monkeypatch.setattr(parameter_group, "set_columnwise_payload", lambda tensor, data: None)

    Fp8ParameterGroup.unshard_parameters(stub, ROWWISE)
    assert gathered == [ROWWISE]
    assert stub._materialized_directions == frozenset((ROWWISE,))

    Fp8ParameterGroup.unshard_parameters(stub, "both")
    # Only the direction that was still missing is gathered the second time.
    assert gathered == [ROWWISE, COLWISE]
    assert stub._materialized_directions == frozenset((ROWWISE, COLWISE))

    # Releasing the storage forgets what was bound, so the next cycle re-gathers.
    stub._unsharded_rowwise.release_storage = lambda: None
    stub._unsharded_colwise.release_storage = lambda: None
    Fp8ParameterGroup.release_unsharded_storage(stub)
    assert stub._materialized_directions == frozenset()


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
