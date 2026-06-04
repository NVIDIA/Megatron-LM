# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from operator import itemgetter
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

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


class GridLayout:
    r"""An additional named factorization of a :class:`HyperCommGrid`'s ranks.

    Lets one rank span carry a second factorization (e.g. an expert
    ``[expt_tp, ep, expt_dp, pp]`` alongside the dense base ``[tp, cp, dp, pp]``). Obtained from
    :meth:`HyperCommGrid.register_layout`, retrieved with :meth:`HyperCommGrid.get_layout`; its
    groups are reached only through this handle (the grid's own methods use the base layout). Dims
    in ``shared_dims`` (e.g. ``pp``, which must match between dense and expert) reuse the base
    grid's group rather than creating a duplicate.
    """

    def __init__(
        self,
        grid: "HyperCommGrid",
        name: str,
        shape: list[int],
        dim_names: list[str],
        shared_dims: list[str],
    ) -> None:
        # [:] insures a copy.
        self._grid = grid
        self.name = name
        self.shape = shape[:]
        self.dim_names = dim_names[:]
        self.shared_dims = shared_dims[:]

    def _is_shared(self, ordered_dims: list[str]) -> bool:
        """Whether ``ordered_dims`` spans only shared dims (so the group reuses the base grid's)."""
        return all(d in self.shared_dims for d in ordered_dims)

    def create_pg(self, dims: Union[str, list[str]], **kwargs: Any) -> dist.ProcessGroup | None:
        """Create this layout's process group for ``dims`` (collective -- call on all ranks).

        A group spanning only shared dims reuses the base grid's group; otherwise it is
        layout-private. Not idempotent (re-creating raises ``KeyError``); retrieve with
        :meth:`get_pg`. ``kwargs`` forward to ``dist.new_subgroups_by_enumeration``.
        """
        ordered_dims, _ = self._grid._order_dims_for(self.dim_names, dims)
        if self._is_shared(ordered_dims):
            # Shared dims must resolve to the *same* group as the base layout. Delegate to the
            # base grid so the object is shared rather than duplicated.
            return self._grid.create_pg(dims, **kwargs)

        key = (self.name, tuple(ordered_dims))
        if key in self._grid._pgs:
            raise KeyError(
                f"Process group {dims} for layout {self.name!r} has already been created. Because "
                f"there is no way to check whether options to create process group matches the "
                f"first, we error out instead of returning the process group that has already "
                f"been created before."
            )

        rank_enum = self._grid._gen_rank_enum_for(self.shape, self.dim_names, ordered_dims)
        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self._grid.backend, **kwargs)

        if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(
                f"Generated process group for layout {self.name!r} {ordered_dims} with "
                f"enumeration {rank_enum}"
            )
        self._grid._pgs[key] = pg
        return pg

    def get_pg(self, dims: Union[str, list[str]]) -> dist.ProcessGroup:
        """Get this layout's previously-created group for ``dims`` (shared dims return the base group).

        Raises ``KeyError`` if it has not been created yet.
        """
        ordered_dims, _ = self._grid._order_dims_for(self.dim_names, dims)
        if self._is_shared(ordered_dims):
            return self._grid.get_pg(dims)

        key = (self.name, tuple(ordered_dims))
        if key not in self._grid._pgs:
            raise KeyError(
                f"Process group {dims} for layout {self.name!r} hasn't been created. Call "
                f"create_pg first."
            )
        return self._grid._pgs[key]

    def get_rank_enum(self, dims: Union[str, list[str]]) -> list[list[int]]:
        """Rank enumeration for ``dims`` under this layout (matches the base grid's for shared dims)."""
        ordered_dims, _ = self._grid._order_dims_for(self.dim_names, dims)
        return self._grid._gen_rank_enum_for(self.shape, self.dim_names, ordered_dims)


class HyperCommGrid:
    r"""N-dimensional communication grid.

    Manages an arbitrary number of parallelisms as a hyperrectangle. Each dimension is given a name
    at initialization time. The order of ``dim_names`` implies the mapping order equivalent to
    the ``order`` argument of MCore's ``initialize_model_parallel``. Internally, it has to be
    reversed to match n-D array.

    For any combination of dimensions, a process group can only be created once.
    Creating process groups for the same combination with different options is not supported.

    The grid's own :meth:`create_pg` / :meth:`get_pg` / :meth:`get_rank_enum` always operate on
    the base factorization passed to the constructor. A rank span that admits more than one
    factorization (for example dense ``tp/cp/dp/pp`` groups alongside expert
    ``expt_tp/ep/expt_dp/pp`` groups over the same ranks) can register an additional layout with
    :meth:`register_layout`, which returns a :class:`GridLayout` handle. Layout-specific groups
    are reached only through that handle; the base methods do not infer or route across layouts.

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
        self.size = np.prod(shape)
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
        # Base-layout groups are keyed by their dash-joined dim string (unchanged from the
        # single-layout design); layout-private groups are keyed by ``(layout_name, dims_tuple)``.
        self._pgs: dict[Union[str, Tuple[str, Tuple[str, ...]]], dist.ProcessGroup] = {}
        self._layouts: dict[str, GridLayout] = {}

    def register_layout(
        self,
        name: str,
        shape: list[int],
        dim_names: list[str],
        shared_dims: Optional[list[str]] = None,
    ) -> GridLayout:
        r"""Register an additional factorization over this grid's rank span.

        Returns a :class:`GridLayout` handle through which the layout's process groups are
        created and retrieved. The base layout (the constructor's ``shape``/``dim_names``) is
        unaffected.

        ``shared_dims`` names dims that are common to the base layout and must map to the *same*
        process group across both. For example MCore requires the dense and expert pipeline
        groups to be identical ranks, so ``"pp"`` is declared shared and a request for the
        ``"pp"`` group through the handle reuses the base grid's group. Every shared dim must
        exist in the base ``dim_names`` and must enumerate to the same ranks under both layouts,
        otherwise registration raises.

        Args:
            name: Unique name for the layout.
            shape: Shape of the layout. Its product must equal the grid size.
            dim_names: Name of each dimension corresponding to ``shape``. Must have the same
                length as ``shape``.
            shared_dims: Dims shared with (and reused from) the base layout. Default ``None``
                (no shared dims).

        Returns:
            GridLayout: A handle for creating/retrieving this layout's process groups.

        Raises:
            ValueError: If ``name`` is already registered, if ``shape`` and ``dim_names`` lengths
                differ, if ``dim_names`` are not unique, if any ``shape`` entry is not a positive
                int, if the layout size does not match the grid size, or if a shared dim is
                missing from the base layout or enumerates to different ranks across layouts.
        """
        if name in self._layouts:
            raise ValueError(f"Layout {name!r} is already registered")
        if len(shape) != len(dim_names):
            raise ValueError(f"len(shape) {shape} != len(dim_names) {dim_names}")
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"Layout {name!r} has duplicate dim_names: {dim_names}")
        if any(not isinstance(s, int) or s <= 0 for s in shape):
            raise ValueError(f"Layout {name!r} shape must be positive ints, got {shape}")
        if np.prod(shape) != self.size:
            raise ValueError(
                f"Layout {name!r} shape {shape} has size {int(np.prod(shape))}, but the grid "
                f"size is {self.size}"
            )

        shared_dims = list(shared_dims) if shared_dims is not None else []
        for dim in shared_dims:
            if dim not in self.dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of layout {name!r} is not in the base layout "
                    f"{self.dim_names}"
                )
            if dim not in dim_names:
                raise ValueError(
                    f"Shared dim {dim!r} of layout {name!r} is not in the layout's dim_names "
                    f"{dim_names}"
                )
            base_dims, _ = self._order_dims_for(self.dim_names, dim)
            base_enum = self._gen_rank_enum_for(self.shape, self.dim_names, base_dims)
            layout_dims, _ = self._order_dims_for(dim_names, dim)
            layout_enum = self._gen_rank_enum_for(shape, dim_names, layout_dims)
            if base_enum != layout_enum:
                raise ValueError(
                    f"Shared dim {dim!r} has different membership across layouts: base "
                    f"enumeration {base_enum} != layout {name!r} enumeration {layout_enum}"
                )

        layout = GridLayout(self, name, shape, dim_names, shared_dims)
        self._layouts[name] = layout
        return layout

    def get_layout(self, name: str) -> GridLayout:
        r"""Return the registered :class:`GridLayout` for ``name``.

        Args:
            name: Name a layout was registered under via :meth:`register_layout`.

        Raises:
            KeyError: If no layout with that name is registered.
        """
        if name not in self._layouts:
            raise KeyError(
                f"Layout {name!r} is not registered. Registered layouts: "
                f"{sorted(self._layouts)}"
            )
        return self._layouts[name]

    def create_pg(self, dims: Union[str, list[str]], **kwargs: Any) -> dist.ProcessGroup | None:
        r"""Create a process group based on a list of dimension names

        Note: The unique key used to store the process group internally will follow the reversed
        order of the original dim_names. For example, if dim_names=["tp", "cp", "dp"] and you
        create a process group with dims=["dp", "tp"], the unique_group_key will be "dp-tp"
        (ordered according to the reversed dim_names order: ["dp", "cp", "tp"]).

        Args:
            dims: Name of leading dimensions to create process group

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
        # ordered_dims and unique_group_key will follow the reversed order of self.dim_names
        ordered_dims, unique_group_key = self._order_dims(dims)

        if unique_group_key in self._pgs:
            raise KeyError(
                f"Process group {dims} has already been created. Because there is no way to check "
                f"whether options to create process group matches the first, we error out instead "
                f"of returning the process group that has already been created before."
            )

        rank_enum = self._gen_rank_enum(ordered_dims)
        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend, **kwargs)

        if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(
                f"Generated process group for {unique_group_key} with enumeration {rank_enum}"
            )
        self._pgs[unique_group_key] = pg
        return pg

    def destroy(self) -> None:
        """Destroy all process groups created by this grid that the current rank belongs to.

        This includes base-layout groups and layout-private groups. A base group reused by a
        layout for a shared dim is stored under a single key, so it is torn down exactly once.
        """
        destroyed: set[int] = set()
        for pg in self._pgs.values():
            if _is_process_group_member(pg) and id(pg) not in destroyed:
                dist.destroy_process_group(pg)
                destroyed.add(id(pg))
        self._pgs.clear()

    def get_pg(self, dims: Union[str, list[str]]) -> dist.ProcessGroup:
        r"""Get a process group based on a list of dimension names

        Args:
            dims: Name of leading dimensions to create process group
        """
        _, unique_group_key = self._order_dims(dims)

        if unique_group_key not in self._pgs:
            raise KeyError(
                f"Process group for {unique_group_key} hasn't been created. Call create_pg first."
            )

        return self._pgs[unique_group_key]

    def get_rank_enum(self, dims: Union[str, list[str]]) -> list[list[int]]:
        r"""Get the rank enumeration for the requested dimension(s).

        This is the exact enumeration that would be used by create_pg for the same
        dims. It is useful for creating additional groups whose membership is derived from
        the grid (e.g., embedding/position-embedding groups derived from PP groups).

        Args:
            dims: Dimension name or list of dimension names.

        Returns:
            List of rank lists (one per subgroup).
        """
        ordered_dims, _ = self._order_dims(dims)
        return self._gen_rank_enum(ordered_dims)

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
        r"""Generate rank enumeration for ``dims`` under an explicit ``shape``/``dim_names``.

        Identical logic to :meth:`_gen_rank_enum` but parameterized by the factorization, so it
        can serve both the base grid and a registered :class:`GridLayout` without any layout
        inference. ``dims`` is assumed already ordered against the reversed ``dim_names``.
        """
        if not HAVE_EINOPS:
            raise RuntimeError(
                "einops is not installed. Please install it with `pip install einops`."
            )

        # Need to reverse order of dim_names to match MCore convention
        dim_names_reverse = dim_names[::-1]

        remaining_dims = []
        for v in dim_names_reverse:
            if v not in dims:
                remaining_dims.append(v)

        rearrange_str = (
            f"({' '.join(dim_names_reverse)}) -> ({' '.join(remaining_dims)}) ({' '.join(dims)})"
        )
        logging.debug(rearrange_str)

        shape_dict = {d: s for d, s in zip(dim_names, shape)}
        return einops.rearrange(
            np.arange(self.rank_offset, self.rank_offset + self.size), rearrange_str, **shape_dict
        ).tolist()

    def _order_dims(self, dims: Union[str, list[str]]) -> Tuple[list[str], str]:
        r"""Reorder dims based on the order of self.dim_names"""
        ordered_dims, _ = self._order_dims_for(self.dim_names, dims)
        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def _order_dims_for(
        self, dim_names: list[str], dims: Union[str, list[str]]
    ) -> Tuple[list[str], str]:
        r"""Reorder ``dims`` against an explicit ``dim_names``.

        Identical ordering logic to :meth:`_order_dims` but parameterized by the factorization's
        ``dim_names``, so it serves both the base grid and a registered :class:`GridLayout`. The
        returned dash-joined key is informational; callers build their own storage key.
        """
        if not isinstance(dims, list):
            ordered_dims = [dims]
        else:
            dim_names_reverse = dim_names[::-1]
            indices = sorted([dim_names_reverse.index(d) for d in dims])
            if len(indices) == 1:
                ordered_dims = [dim_names_reverse[indices[0]]]
            else:
                ordered_dims = list(itemgetter(*indices)(dim_names_reverse))

        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def is_current_rank_in_grid(self) -> bool:
        """Check if the current rank belongs to this grid.

        Returns:
            True if the current rank is within [rank_offset, rank_offset + size).
        """
        rank = dist.get_rank()
        return bool(self.rank_offset <= rank < self.rank_offset + self.size)
