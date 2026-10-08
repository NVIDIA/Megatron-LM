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

"""Immutable, backend-neutral token-span trees for shared-prefix packing.

Each node stores one contiguous physical token span and its parent node index.
Parents precede children, so a bin can contain a forest of arbitrary-depth
trees. Logical lengths exclude each node's padding tail: children continue from
the parent's last real token, not from its physical padding. This descriptor
does not imply that a particular fused execution backend supports deeper trees.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PackedTreeLayout:
    """A topologically ordered forest whose spans cover the packed token buffer.

    ``node_start`` and ``node_len`` describe physical storage. ``node_parent``
    contains ``-1`` for roots or an earlier node index for continuations.
    ``logical_node_len`` defaults to the physical lengths. Input sequences are
    snapshotted as tuples, preventing mutations after validation.
    """

    node_start: tuple[int, ...]
    node_len: tuple[int, ...]
    node_parent: tuple[int, ...]
    logical_node_len: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("node_start", "node_len", "node_parent", "logical_node_len"):
            values = tuple(getattr(self, name))
            if any(type(value) is not int for value in values):
                raise ValueError(f"{name} must contain integers, excluding booleans")
            object.__setattr__(self, name, values)
        logical = self.logical_node_len or self.node_len
        object.__setattr__(self, "logical_node_len", logical)
        if not self.node_start or not (
            len(self.node_start) == len(self.node_len) == len(self.node_parent) == len(logical)
        ):
            raise ValueError("tree node arrays must have equal, nonzero length")
        cursor = 0
        for node, (start, length, parent, real_length) in enumerate(
            zip(self.node_start, self.node_len, self.node_parent, logical, strict=True)
        ):
            if start != cursor or length < 1:
                raise ValueError("tree node spans must be positive and contiguous")
            if real_length < 1 or real_length > length:
                raise ValueError("logical node lengths must be positive and no longer than storage")
            if parent < -1 or parent >= node:
                raise ValueError("tree parents must be -1 or an earlier node index")
            cursor += length

    @classmethod
    def from_shared_prefix(
        cls,
        prefix_len: int,
        completion_lens: Sequence[int],
        *,
        logical_completion_lens: Sequence[int] | None = None,
    ) -> PackedTreeLayout:
        """Describe a prompt root and its direct completion children."""
        lengths = (prefix_len, *completion_lens)
        starts = []
        cursor = 0
        for length in lengths:
            if type(length) is not int:
                raise ValueError("node lengths must be integers, excluding booleans")
            starts.append(cursor)
            cursor += length
        logical = (
            lengths if logical_completion_lens is None else (prefix_len, *logical_completion_lens)
        )
        return cls(tuple(starts), lengths, (-1, *((0,) * (len(lengths) - 1))), logical)

    @classmethod
    def concat(cls, layouts: Sequence[PackedTreeLayout]) -> PackedTreeLayout:
        """Combine forests, rebasing physical offsets and parent indices."""
        starts, lengths, parents, logical = [], [], [], []
        token_offset = 0
        for layout in layouts:
            node_offset = len(starts)
            starts.extend(token_offset + value for value in layout.node_start)
            lengths.extend(layout.node_len)
            parents.extend(
                -1 if parent == -1 else node_offset + parent for parent in layout.node_parent
            )
            logical.extend(layout.logical_node_len)
            token_offset += layout.total_len
        return cls(tuple(starts), tuple(lengths), tuple(parents), tuple(logical))

    @property
    def num_nodes(self) -> int:
        return len(self.node_start)

    @property
    def total_len(self) -> int:
        """Physical token count including node padding tails."""
        return sum(self.node_len)

    @property
    def logical_total_len(self) -> int:
        return sum(self.logical_node_len)

    def roots(self) -> tuple[int, ...]:
        return tuple(node for node, parent in enumerate(self.node_parent) if parent == -1)

    def _check_node(self, node: int) -> None:
        if type(node) is not int or not 0 <= node < self.num_nodes:
            raise ValueError("node index must be an integer within the tree")

    def ancestors(self, node: int) -> tuple[int, ...]:
        """Return strict ancestors in root-to-parent order."""
        self._check_node(node)
        result = []
        parent = self.node_parent[node]
        while parent != -1:
            result.append(parent)
            parent = self.node_parent[parent]
        return tuple(reversed(result))

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        """Test strict ancestry; a node is not its own ancestor."""
        self._check_node(ancestor)
        return ancestor in self.ancestors(node)

    def depth(self) -> int:
        """Maximum edge depth: roots have depth zero, a GRPO star has one."""
        depths = []
        for parent in self.node_parent:
            depths.append(0 if parent == -1 else depths[parent] + 1)
        return max(depths)

    def leaf_nodes(self) -> tuple[int, ...]:
        parents = set(self.node_parent)
        return tuple(node for node in range(self.num_nodes) if node not in parents)

    def iter_star_roots(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Lower only a forest of stars; reject deeper trees before execution.

        A backend that implements only prompt-to-completion forks must use this
        guard rather than silently treating all descendants as sibling branches.
        Root-only nodes are permitted for conventional trajectories. The
        current backend also requires each star's nodes to be consecutive and
        roots to have no padding; reject unsupported storage orders explicitly.
        """
        if self.depth() > 1:
            raise NotImplementedError("this execution backend supports only a forest of stars")
        stars = tuple(
            (root, tuple(node for node, parent in enumerate(self.node_parent) if parent == root))
            for root in self.roots()
        )
        if tuple(node for root, children in stars for node in (root, *children)) != tuple(
            range(self.num_nodes)
        ):
            raise NotImplementedError("this execution backend requires contiguous star storage")
        if any(self.node_len[root] != self.logical_node_len[root] for root, _ in stars):
            raise NotImplementedError("this execution backend requires unpadded prompt roots")
        yield from stars

    def segment_ids(self) -> tuple[int, ...]:
        """Physical token-to-node map."""
        return tuple(node for node, length in enumerate(self.node_len) for _ in range(length))

    def node_pos_offset(self) -> tuple[int, ...]:
        """Logical depth along each node's root-to-parent token path."""
        offsets = []
        for parent in self.node_parent:
            offsets.append(0 if parent == -1 else offsets[parent] + self.logical_node_len[parent])
        return tuple(offsets)

    def position_ids(self) -> tuple[int, ...]:
        """Path positions, with each node's physical tail continuing locally."""
        return tuple(
            offset + index
            for offset, length in zip(self.node_pos_offset(), self.node_len, strict=True)
            for index in range(length)
        )

    def first_predecessors(self) -> tuple[int, ...]:
        """Packed predecessor for each node's first token, or -1 for a root."""
        return tuple(
            -1 if parent == -1 else self.node_start[parent] + self.logical_node_len[parent] - 1
            for parent in self.node_parent
        )

    def prev_token_index(self) -> tuple[int, ...]:
        """Next-token prediction predecessors, including local padding positions."""
        return tuple(
            first if index == 0 else start + index - 1
            for start, length, first in zip(
                self.node_start, self.node_len, self.first_predecessors(), strict=True
            )
            for index in range(length)
        )

    def path_token_indices(self, node: int) -> tuple[int, ...]:
        """Reconstruct a dense logical trajectory through a node, skipping padding."""
        return tuple(
            index
            for ancestor in (*self.ancestors(node), node)
            for index in range(
                self.node_start[ancestor],
                self.node_start[ancestor] + self.logical_node_len[ancestor],
            )
        )

    def padding_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for start, physical, logical in zip(
                self.node_start, self.node_len, self.logical_node_len, strict=True
            )
            for index in range(start + logical, start + physical)
        )
