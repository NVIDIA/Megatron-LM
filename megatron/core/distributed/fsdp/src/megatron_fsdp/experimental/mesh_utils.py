# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-parallel submesh derivation for Megatron-FSDP.

``DBuffer`` owns only the data-parallel (DP) submesh, while
``FsdpParameterGroup`` owns the full user-provided mesh. This module derives the
DP submesh from the full mesh so the parameter group can hand it to its
DBuffers; the reverse mapping (lifting a DP DTensor back onto the full mesh) is
recovered by name inside ``FsdpParameterGroup.get_dtensor``.
"""

from torch.distributed import DeviceMesh

from .placement import MeshAxis


def axis_index(mesh: DeviceMesh, axis: MeshAxis) -> int:
    """Resolve a mesh axis (index or name) to a non-negative axis index."""
    if isinstance(axis, int):
        axis_number = axis
        if axis_number < 0:
            axis_number += mesh.ndim
        if axis_number < 0 or axis_number >= mesh.ndim:
            raise ValueError(f"Mesh axis {axis} is out of bounds for mesh ndim {mesh.ndim}.")
        return axis_number

    dim_names = mesh.mesh_dim_names
    if dim_names is None or axis not in dim_names:
        raise ValueError(f"Mesh axis {axis!r} is not present in mesh dim names {dim_names}.")
    return dim_names.index(axis)


def build_dp_mesh(full_mesh: DeviceMesh, dp_axes: list[MeshAxis]) -> DeviceMesh:
    """Select the data-parallel submesh of ``full_mesh``.

    The submesh axes follow ``dp_axes`` order. When the DP axes cover the whole
    mesh in order, the full mesh is returned unchanged (this also preserves
    behavior for meshes without dim names). Otherwise the submesh is selected by
    name, so a mesh with non-DP (tensor-parallel) axes must have dim names.
    """
    if not dp_axes:
        raise ValueError("FSDP requires at least one data-parallel mesh axis.")

    dp_axis_indices = tuple(axis_index(full_mesh, axis) for axis in dp_axes)
    seen_axis_indices: set[int] = set()
    for dp_axis_index in dp_axis_indices:
        if dp_axis_index in seen_axis_indices:
            raise ValueError(
                f"Duplicate data-parallel mesh axis {dp_axis_index} in dp_axes {dp_axes!r}."
            )
        seen_axis_indices.add(dp_axis_index)

    if dp_axis_indices == tuple(range(full_mesh.ndim)):
        return full_mesh

    dim_names = full_mesh.mesh_dim_names
    if dim_names is None:
        raise ValueError(
            "A mesh with non-data-parallel axes must have dim names so the "
            "data-parallel submesh can be selected by name."
        )
    dp_axis_names = tuple(dim_names[axis] for axis in dp_axis_indices)
    return full_mesh[dp_axis_names[0]] if len(dp_axis_names) == 1 else full_mesh[dp_axis_names]
