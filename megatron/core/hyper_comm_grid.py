# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch.distributed as dist

try:
    from absl import logging

    HAVE_ABSL = True
except ImportError:
    import logging
    import warnings

    logging = logging.getLogger(__name__)
    warnings.warn(
        "absl.logging is not installed. Using logging.getLogger(__name__) instead. "
        "Please install absl.logging with `pip install absl-py` to use absl.logging."
    )
    HAVE_ABSL = False


def _is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Whether the current rank belongs to ``pg``.

    ``new_subgroups_by_enumeration`` returns the ``GroupMember.NON_GROUP_MEMBER``
    sentinel (not ``None``) for non-member ranks, which must not be destroyed.
    """
    non_member = getattr(getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", None)
    return pg is not None and pg is not non_member


_BASE_VIEW_NAME = "base"


@dataclass
class _RankViewSpec:
    """A named rank factorization over the same rank span as the base grid."""

    name: str
    shape: list[int]
    dim_names: list[str]
    shared_dims: list[str]


class HyperCommGrid:
    r"""N-dimensional communication grid.

    Manages an arbitrary number of parallelisms as a hyperrectangle. Each dimension is given a name
    at initialization time. The order of ``dim_names`` implies the mapping order equivalent to
    the ``order`` argument of MCore's ``initialize_model_parallel``. Internally, it has to be
    reversed to match n-D array.

    For any combination of dimensions, a process group can only be created once.
    Creating process groups for the same combination with different options is not supported.

    The grid's own :meth:`create_pg` / :meth:`get_pg` / :meth:`get_rank_enum` operate on the base
    factorization passed to the constructor by default. A rank span that admits more than one
    factorization (for example dense ``tp/cp/dp/pp`` groups alongside expert
    ``expt_tp/ep/expt_dp/pp`` groups over the same ranks) can register an additional rank view
    with :meth:`register_view`. View-specific groups are still created and retrieved through this
    root grid by passing ``view="..."``; process-group lifecycle is owned in exactly one place.

    Note:
        ``create_pg()`` over specific dims must be explicitly called to create a process group.
        We don't create a process group in the ``get_pg()`` function because there are many options
        (kwargs) that can be passed when creating a process group, which ``get_pg()`` should not
        be exposed to.

    Examples:
        >>> grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])
        >>> dp_group = grid.create_pg("dp")
        >>> # retrieve dp_group from grid after creation
        >>> # dp_group = grid.get_pg("dp")
        >>>
        >>> # It is equivalent to calling the following functions in MCore parallel_state
        >>> # with world size 120.
        >>> parallel_state.initialize_model_parallel(
        >>>     tensor_model_parallel_size=2,
        >>>     context_parallel_size=3,
        >>>     pipeline_model_parallel_size=4,
        >>>     order="tp-cp-pp-dp")
        >>> dp_group_mcore = parallel_state.get_data_parallel_group()
        >>>
        >>> # We can create group from multiple leading dims and also pass more options.
        >>> pg_options = ProcessGroupNCCL.Options()
        >>> pg_options.config.max_ctas = 8
        >>> dp_cp_group = grid.create_pg(
        >>>     ["cp", "dp"], pg_options=pg_options,
        >>>     group_desc="WEIGHT_GRADIENT_COMM_GROUP")


    Args:
        shape: Shape of the communication grid.
        dim_names: Name of each dimension corresponding to shape. Must have the same length as
            shape.
        rank_offset: Starting rank when the grid doesn't span the entire communication world.
            Default 0.
        backend: Backend for creating process group. Default None and will use default backend.
    """

    def __init__(
        self,
        shape: list[int],
        dim_names: list[str],
        rank_offset: int = 0,
        backend: Optional[str] = None,
    ) -> None:
        if len(shape) != len(dim_names):
            raise ValueError(f"len(shape) {shape} != len(dim_names) {dim_names}")

        # Querying environment instead of calling torch.distributed.get_world_size() for mock
        # testing without initializing process group.
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            raise RuntimeError(
                "Cannot determine world size: WORLD_SIZE environment variable not set and "
                "torch.distributed is not initialized. Please either set WORLD_SIZE or "
                "initialize torch.distributed before creating HyperCommGrid."
            )
        self.rank_offset = rank_offset
        self.size = int(np.prod(shape))
        if rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {rank_offset}")
        if self.size > world_size - rank_offset:
            raise RuntimeError(
                f"Grid shape {shape} is over sized with world size {world_size} and rank "
                f"offset {self.rank_offset}"
            )

        # [:] insures a copy
        self.shape = shape[:]
        self.dim_names = dim_names[:]
        self.backend = backend
        self._views: dict[str, _RankViewSpec] = {
            _BASE_VIEW_NAME: _RankViewSpec(
                _BASE_VIEW_NAME, self.shape[:], self.dim_names[:], shared_dims=[]
            )
        }
        # Base-view groups are keyed by their dash-joined dim string (unchanged from the
        # single-view design); view-private groups are keyed by ``(view_name, dims_tuple)``.
        self._pgs: dict[Union[str, tuple[str, tuple[str, ...]]], dist.ProcessGroup] = {}

    def register_view(
        self,
        name: str,
        shape: list[int],
        dim_names: list[str],
        shared_dims: Optional[list[str]] = None,
    ) -> None:
        r"""Register an additional rank factorization over this grid's rank span.

        Shared dims must exist in both the base view and the new view, and must enumerate to the
        same rank groups as the base view.
        """
        if name in self._views:
            raise ValueError(f"View {name!r} is already registered")
        if len(shape) != len(dim_names):
            raise ValueError(f"len(shape) {shape} != len(dim_names) {dim_names}")
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"View {name!r} has duplicate dim_names: {dim_names}")
        if any(not isinstance(s, int) or s <= 0 for s in shape):
            raise ValueError(f"View {name!r} shape must be positive ints, got {shape}")
        if int(np.prod(shape)) != self.size:
            raise ValueError(
                f"View {name!r} shape {shape} has size {int(np.prod(shape))}, but the grid "
                f"size is {self.size}"
            )

        shared_dims = list(shared_dims) if shared_dims is not None else []
        for dim in shared_dims:
            if dim not in self.dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of view {name!r} is not in the base view "
                    f"{self.dim_names}"
                )
            if dim not in dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of view {name!r} is not in the view's dim_names "
                    f"{dim_names}"
                )
            base_dims, _ = self._order_dims_for(self.dim_names, dim)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_dims)
            view_dims, _ = self._order_dims_for(dim_names, dim)
            view_enum = self._gen_rank_enum_for(shape, dim_names, view_dims)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dim {dim!r} has different membership across views: base "
                    f"enumeration {base_enum} != view {name!r} enumeration {view_enum}"
                )

        if len(shared_dims) > 1:
            base_dims, _ = self._order_dims_for(self.dim_names, shared_dims)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_dims)
            view_dims, _ = self._order_dims_for(dim_names, shared_dims)
            view_enum = self._gen_rank_enum_for(shape, dim_names, view_dims)
            if base_enum != view_enum:
                raise ValueError(
                    f"Shared dims {shared_dims!r} have different membership across views: base "
                    f"enumeration {base_enum} != view {name!r} enumeration {view_enum}"
                )

        self._views[name] = _RankViewSpec(name, shape[:], dim_names[:], shared_dims[:])

    def create_pg(
        self, dims: Union[str, list[str]], view: Optional[str] = None, **kwargs: Any
    ) -> dist.ProcessGroup | None:
        r"""Create a process group based on a list of dimension names

        Note: The unique key used to store the process group internally will follow the reversed
        order of the original dim_names. For example, if dim_names=["tp", "cp", "dp"] and you
        create a process group with dims=["dp", "tp"], the unique_group_key will be "dp-tp"
        (ordered according to the reversed dim_names order: ["dp", "cp", "tp"]).

        Args:
            dims: Name of leading dimensions to create process group
            view: Optional registered rank view name. Defaults to the base view.

        Keyword arguments are directly passed into new_subgroups_by_enumeration(). The docstring
        is copied from new_subgroups_by_enumeration().

        Keyword args from `dist.new_subgroups_by_enumeration`:
            timeout (timedelta, optional): see `init_process_group` for details and default value.
            pg_options (ProcessGroupOptions, optional): process group options
                specifying what additional options need to be passed in during
                the construction of specific process groups.
            group_desc (str, optional): A string describing the group. Each subgroup will
                inherit its group_desc.

        Returns:
            dist.ProcessGroup | None: The created process group.

        Raises:
            KeyError: If attempting to recreate a process group with an existing key.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_group_key, enum_view, enum_dims = self._canonical_pg_key_and_enum_view(
            view_spec, ordered_dims
        )

        if unique_group_key in self._pgs:
            if self._is_base_pg_key(unique_group_key):
                raise KeyError(
                    f"Process group {dims} has already been created. Because there is no way "
                    f"to check whether options to create process group matches the first, we "
                    f"error out instead of returning the process group that has already been "
                    f"created before."
                )
            raise KeyError(
                f"Process group {dims} for view {view_spec.name!r} has already been created. "
                f"Because there is no way to check whether options to create process group "
                f"matches the first, we error out instead of returning the process group that "
                f"has already been created before."
            )

        rank_enum = self._gen_rank_enum_for(enum_view.shape, enum_view.dim_names, enum_dims)
        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend, **kwargs)

        if dist.is_initialized() and dist.get_rank() == 0:
            if view_spec.name == _BASE_VIEW_NAME:
                logging.info(
                    f"Generated process group for {unique_group_key} with enumeration {rank_enum}"
                )
            else:
                logging.info(
                    f"Generated process group for view {view_spec.name!r} {ordered_dims} with "
                    f"enumeration {rank_enum}"
                )
        self._pgs[unique_group_key] = pg
        return pg

    def destroy(self) -> None:
        """Destroy all process groups created by this grid that the current rank belongs to.

        This includes base-view groups and view-private groups. A base group reused by a
        view for a shared dim is stored under a single key, so it is torn down exactly once.
        """
        destroyed: set[int] = set()
        for pg in self._pgs.values():
            if _is_process_group_member(pg) and id(pg) not in destroyed:
                dist.destroy_process_group(pg)
                destroyed.add(id(pg))
        self._pgs.clear()

    def get_pg(self, dims: Union[str, list[str]], view: Optional[str] = None) -> dist.ProcessGroup:
        r"""Get a process group based on a list of dimension names

        Args:
            dims: Name of leading dimensions to create process group
            view: Optional registered rank view name. Defaults to the base view.
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        unique_group_key, _, _ = self._canonical_pg_key_and_enum_view(view_spec, ordered_dims)

        if unique_group_key not in self._pgs:
            if self._is_base_pg_key(unique_group_key):
                raise KeyError(
                    f"Process group for {unique_group_key} hasn't been created. Call create_pg "
                    f"first."
                )
            raise KeyError(
                f"Process group {dims} for view {view_spec.name!r} hasn't been created. Call "
                f"create_pg first."
            )

        return self._pgs[unique_group_key]

    def get_rank_enum(
        self, dims: Union[str, list[str]], view: Optional[str] = None
    ) -> list[list[int]]:
        r"""Get the rank enumeration for the requested dimension(s).

        This is the exact enumeration that would be used by create_pg for the same
        dims. It is useful for creating additional groups whose membership is derived from
        the grid (e.g., embedding/position-embedding groups derived from PP groups).

        Args:
            dims: Dimension name or list of dimension names.
            view: Optional registered rank view name. Defaults to the base view.

        Returns:
            List of rank lists (one per subgroup).
        """
        view_spec = self._resolve_view(view)
        ordered_dims, _ = self._order_dims_for_view(view_spec, dims)
        return self._gen_rank_enum_for(view_spec.shape, view_spec.dim_names, ordered_dims)

    def _gen_rank_enum(self, dims: list[str]) -> list[list[int]]:
        r"""Generate rank enumeration before calling new_subgroups_by_enumeration

        This function returns ranks grouped by the specified dimensions, but in REVERSE order
        of the input dimensions. For example, if you request dimensions ["a", "b"],
        the ranks will be grouped by "b-a" order.

        Example:
            For a grid with shape [2, 2, 2] and dim_names ["a", "b", "c"]:
            _gen_rank_enum(["a", "b"]) returns [[0, 2, 1, 3], [4, 6, 5, 7]]

            This groups ranks first by dimension "b", then by dimension "a":
            - Group 0: ranks where c=0, grouped by b-a: [0, 2, 1, 3]
            - Group 1: ranks where c=1, grouped by b-a: [4, 6, 5, 7]

        Args:
            dims: Name of leading dimensions to create process group

        Although the function is lightweight enough to be inlined, a standalone one makes it
        easier to test against MCore's RankGenerator
        """
        return self._gen_rank_enum_for(self.shape, self.dim_names, dims)

    def _gen_rank_enum_for(
        self, shape: list[int], dim_names: list[str], dims: list[str]
    ) -> list[list[int]]:
        r"""Generate rank enumeration for ``dims`` under an explicit ``shape``/``dim_names``."""
        # Need to reverse order of dim_names to match MCore convention.
        dim_names_reverse = dim_names[::-1]
        shape_dict = {d: s for d, s in zip(dim_names, shape)}
        rank_tensor = np.arange(self.rank_offset, self.rank_offset + self.size).reshape(
            [shape_dict[d] for d in dim_names_reverse]
        )

        source_axes = [dim_names_reverse.index(d) for d in dims]
        target_axes = list(range(len(dim_names_reverse) - len(dims), len(dim_names_reverse)))
        logging.debug(
            "Moving axes %s to %s for dim_names=%s dims=%s",
            source_axes,
            target_axes,
            dim_names,
            dims,
        )
        rank_tensor = np.moveaxis(rank_tensor, source_axes, target_axes)

        group_size = int(np.prod([shape_dict[d] for d in dims]))
        return rank_tensor.reshape(-1, group_size).tolist()

    def _order_dims(self, dims: Union[str, list[str]]) -> tuple[list[str], str]:
        r"""Reorder dims based on the order of self.dim_names"""
        ordered_dims, _ = self._order_dims_for_view(self._views[_BASE_VIEW_NAME], dims)
        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def _order_dims_for(
        self, dim_names: list[str], dims: Union[str, list[str]]
    ) -> tuple[list[str], str]:
        r"""Reorder ``dims`` against an explicit ``dim_names``."""
        if not isinstance(dims, list):
            ordered_dims = [dims]
        else:
            dim_names_reverse = dim_names[::-1]
            indices = sorted([dim_names_reverse.index(d) for d in dims])
            ordered_dims = [dim_names_reverse[i] for i in indices]

        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def _resolve_view(self, view: Optional[str]) -> _RankViewSpec:
        r"""Return the requested rank view, defaulting to the base view."""
        view_name = _BASE_VIEW_NAME if view is None else view
        if view_name not in self._views:
            raise KeyError(
                f"View {view_name!r} is not registered. Registered views: {sorted(self._views)}"
            )
        return self._views[view_name]

    def _order_dims_for_view(
        self, view: _RankViewSpec, dims: Union[str, list[str]]
    ) -> tuple[list[str], str]:
        r"""Reorder ``dims`` against a registered view and report missing dims clearly."""
        requested_dims = [dims] if not isinstance(dims, list) else dims
        missing_dims = [d for d in requested_dims if d not in view.dim_names]
        if missing_dims:
            raise ValueError(
                f"{missing_dims[0]!r} is not in view {view.name!r} with dim_names "
                f"{view.dim_names}"
            )
        return self._order_dims_for(view.dim_names, dims)

    def _canonical_pg_key_and_enum_view(
        self, view: _RankViewSpec, ordered_dims: list[str]
    ) -> tuple[Union[str, tuple[str, tuple[str, ...]]], _RankViewSpec, list[str]]:
        r"""Return the storage key and rank view used to enumerate a process group."""
        if view.name == _BASE_VIEW_NAME:
            return "-".join(ordered_dims), view, ordered_dims

        if all(d in view.shared_dims for d in ordered_dims):
            base_view = self._views[_BASE_VIEW_NAME]
            base_ordered_dims, base_key = self._order_dims_for_view(base_view, ordered_dims)
            return base_key, base_view, base_ordered_dims

        return (view.name, tuple(ordered_dims)), view, ordered_dims

    def _is_base_pg_key(self, key: Union[str, tuple[str, tuple[str, ...]]]) -> bool:
        r"""Whether a process-group key belongs to the base view namespace."""
        return isinstance(key, str)

    def is_current_rank_in_grid(self) -> bool:
        """Check if the current rank belongs to this grid.

        Returns:
            True if the current rank is within [rank_offset, rank_offset + size).
        """
        rank = dist.get_rank()
        return bool(self.rank_offset <= rank < self.rank_offset + self.size)
