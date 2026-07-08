# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical logical names and deterministic indices for model parameters."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import torch

from megatron.core.utils import unwrap_model

_GROUPED_EXPERT_PATTERN = re.compile(
    r"^((?:.*\.)?mlp\.experts\.linear_fc\d\.(?:weight|bias))(\d+)(.*)$"
)
_SEQUENTIAL_EXPERT_PATTERN = re.compile(r"^((?:.*\.)?mlp\.experts\.local_experts\.)(\d+)(\..*)$")


@dataclass(frozen=True, init=False)
class CanonicalParameterNameIndex(Mapping[str, int]):
    """Deterministic indices for a set of canonical parameter names.

    Duplicate names are collapsed and the remaining names are ordered
    lexicographically. The resulting mapping is immutable after construction.

    Args:
        names: Canonical parameter names to index.
    """

    names: tuple[str, ...]
    _name_to_index: dict[str, int] = field(repr=False, compare=False, hash=False)

    def __init__(self, names: Iterable[str]) -> None:
        ordered_names = tuple(sorted(set(names)))
        object.__setattr__(self, "names", ordered_names)
        object.__setattr__(
            self, "_name_to_index", {name: index for index, name in enumerate(ordered_names)}
        )

    def __getitem__(self, name: str) -> int:
        return self._name_to_index[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


class CanonicalParameterNameMap(Mapping[torch.nn.Parameter, str]):
    """Map model parameters to topology-independent logical names.

    Pipeline-local layer indices are replaced with the global ``layer_number``
    assigned to their owning layer module. Expert-local indices are replaced
    with global expert indices when an expert-parallel rank and size are
    supplied. The map only contains original model parameters; consumers are
    responsible for mapping optimizer copies or shards back to those parameters.

    Construction performs no distributed collectives and does not read global
    process-group state. Call :meth:`all_gather_index` explicitly when every
    rank in a process group needs the same global name index.

    Args:
        model_chunks: A model module or iterable of model chunks.
        expert_parallel_rank: Rank within the expert-model-parallel group.
        expert_parallel_size: Size of the expert-model-parallel group.

    Raises:
        ValueError: If the model list or expert topology is invalid, or if two
            distinct local parameters resolve to the same canonical name.
    """

    def __init__(
        self,
        model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
        *,
        expert_parallel_rank: int = 0,
        expert_parallel_size: int = 1,
    ) -> None:
        if expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be at least 1")
        if not 0 <= expert_parallel_rank < expert_parallel_size:
            raise ValueError(
                f"expert_parallel_rank must be in [0, {expert_parallel_size}), "
                f"got {expert_parallel_rank}"
            )

        normalized_chunks = _normalize_model_chunks(model_chunks)
        if not normalized_chunks:
            raise ValueError("Cannot build canonical parameter names for an empty model list.")

        self.model_chunks = tuple(unwrap_model(normalized_chunks))
        self.expert_parallel_rank = expert_parallel_rank
        self.expert_parallel_size = expert_parallel_size
        self._param_to_name = self._build_param_to_name()
        self._local_index = CanonicalParameterNameIndex(self._param_to_name.values())

    def __getitem__(self, param: torch.nn.Parameter) -> str:
        return self._param_to_name[param]

    def __iter__(self) -> Iterator[torch.nn.Parameter]:
        return iter(self._param_to_name)

    def __len__(self) -> int:
        return len(self._param_to_name)

    @property
    def local_index(self) -> CanonicalParameterNameIndex:
        """Return deterministic indices for locally present parameter names."""
        return self._local_index

    def name_for_param(self, param: torch.nn.Parameter) -> str:
        """Return the canonical logical name for a model parameter."""
        return self[param]

    def all_gather_index(
        self, group: torch.distributed.ProcessGroup | None = None
    ) -> CanonicalParameterNameIndex:
        """Collect and deterministically index canonical names from a process group.

        This is a collective operation. Every rank in ``group`` must call it in
        the same collective order. If distributed communication is unavailable
        or uninitialized, the local index is returned without communication.

        Args:
            group: Process group whose canonical names should be collected. If
                ``None``, use the default world process group.

        Returns:
            A global, deterministic parameter-name index shared by the group.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return self.local_index

        world_size = torch.distributed.get_world_size(group=group)
        if world_size == 1:
            return self.local_index

        gathered_names: list[tuple[str, ...] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_names, self.local_index.names, group=group)
        return CanonicalParameterNameIndex(
            name for rank_names in gathered_names if rank_names is not None for name in rank_names
        )

    def _build_param_to_name(self) -> dict[torch.nn.Parameter, str]:
        param_to_name: dict[torch.nn.Parameter, str] = {}
        name_to_param: dict[str, torch.nn.Parameter] = {}

        for model_chunk in self.model_chunks:
            layer_prefixes = _build_global_layer_prefixes(model_chunk)
            num_experts = _get_num_moe_experts(model_chunk)
            expert_offset = _get_local_expert_offset(
                num_experts, self.expert_parallel_rank, self.expert_parallel_size
            )

            for local_name, param in model_chunk.named_parameters():
                canonical_name = _canonical_parameter_name(
                    local_name, layer_prefixes, num_experts, expert_offset
                )

                previous_name = param_to_name.get(param)
                if previous_name is not None and previous_name != canonical_name:
                    raise ValueError(
                        "A shared parameter resolved to multiple canonical names: "
                        f"{previous_name!r} and {canonical_name!r}."
                    )

                previous_param = name_to_param.get(canonical_name)
                if previous_param is not None and previous_param is not param:
                    raise ValueError(
                        f"Canonical parameter name {canonical_name!r} refers to multiple "
                        "distinct local parameters."
                    )

                param_to_name[param] = canonical_name
                name_to_param[canonical_name] = param

        return param_to_name


def _normalize_model_chunks(
    model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
) -> list[torch.nn.Module]:
    if isinstance(model_chunks, torch.nn.Module):
        return [model_chunks]
    return list(model_chunks)


def _build_global_layer_prefixes(model_chunk: torch.nn.Module) -> dict[str, str]:
    """Build local-to-global prefixes for numbered layer modules."""
    prefixes = {}
    for module_name, module in model_chunk.named_modules():
        if not module_name or "mtp" in module_name:
            continue
        layer_number = getattr(module, "layer_number", None)
        if not isinstance(layer_number, int):
            continue
        parts = module_name.split(".")
        if not parts[-1].isdigit():
            continue
        parts[-1] = str(layer_number - 1)
        prefixes[module_name] = ".".join(parts)
    return prefixes


def _canonical_parameter_name(
    local_name: str, layer_prefixes: Mapping[str, str], num_experts: int | None, expert_offset: int
) -> str:
    name = _replace_longest_prefix(local_name, layer_prefixes)
    return _global_expert_parameter_name(name, num_experts, expert_offset)


def _replace_longest_prefix(name: str, replacements: Mapping[str, str]) -> str:
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        prefix = ".".join(parts[:index])
        replacement = replacements.get(prefix)
        if replacement is None:
            continue
        suffix = ".".join(parts[index:])
        return replacement if not suffix else f"{replacement}.{suffix}"
    return name


def _global_expert_parameter_name(
    local_name: str, num_experts: int | None, expert_offset: int
) -> str:
    if not num_experts or expert_offset == 0:
        return local_name

    for pattern in (_GROUPED_EXPERT_PATTERN, _SEQUENTIAL_EXPERT_PATTERN):
        match = pattern.match(local_name)
        if match is not None:
            prefix, local_expert_index, suffix = match.groups()
            return f"{prefix}{int(local_expert_index) + expert_offset}{suffix}"

    return local_name


def _get_local_expert_offset(
    num_experts: int | None, expert_parallel_rank: int, expert_parallel_size: int
) -> int:
    if not num_experts or expert_parallel_size == 1:
        return 0
    if num_experts % expert_parallel_size != 0:
        raise ValueError(
            f"num_moe_experts ({num_experts}) must be divisible by "
            f"expert_parallel_size ({expert_parallel_size})"
        )
    return expert_parallel_rank * (num_experts // expert_parallel_size)


def _get_num_moe_experts(model_chunk: torch.nn.Module) -> int | None:
    config = getattr(model_chunk, "config", None)
    return getattr(config, "num_moe_experts", None)
