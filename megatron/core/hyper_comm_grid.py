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


class HyperCommGrid:
    r"""N-dimensional communication grid.

    Manages an arbitrary number of parallelisms as a hyperrectangle. Each dimension is given a name
    at initialization time. The order of ``dim_names`` implies the mapping order equivalent to
    the ``order`` argument of MCore's ``initialize_model_parallel``. Internally, it has to be
    reversed to match n-D array.

    For any combination of dimensions, a process group can only be created once.
    Creating process groups for the same combination with different options is not supported.

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
        self._pgs: dict[str, dist.ProcessGroup] = {}

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

        logging.info(f"Generated process group for {unique_group_key} with enumeration {rank_enum}")
        self._pgs[unique_group_key] = pg
        return pg

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

        if not HAVE_EINOPS:
            raise RuntimeError(
                "einops is not installed. Please install it with `pip install einops`."
            )

        # Need to reverse order of dim_names to match MCore convention
        dim_names_reverse = self.dim_names[::-1]

        remaining_dims = []
        for v in dim_names_reverse:
            if v not in dims:
                remaining_dims.append(v)

        rearrange_str = (
            f"({' '.join(dim_names_reverse)}) -> ({' '.join(remaining_dims)}) ({' '.join(dims)})"
        )
        logging.debug(rearrange_str)

        shape_dict = {d: s for d, s in zip(self.dim_names, self.shape)}
        return einops.rearrange(
            np.arange(self.rank_offset, self.rank_offset + self.size), rearrange_str, **shape_dict
        ).tolist()

    def _order_dims(self, dims: Union[str, list[str]]) -> Tuple[list[str], str]:
        r"""Reorder dims based on the order of self.dim_names"""
        if not isinstance(dims, list):
            ordered_dims = [dims]
        else:
            dim_names_reverse = self.dim_names[::-1]
            indices = sorted([dim_names_reverse.index(d) for d in dims])
            if len(indices) == 1:
                ordered_dims = [dim_names_reverse[indices[0]]]
            else:
                ordered_dims = list(itemgetter(*indices)(dim_names_reverse))

        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key
