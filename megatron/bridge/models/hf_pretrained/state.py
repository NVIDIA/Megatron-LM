# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/state.py
# Source license: Apache-2.0.

import fnmatch
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Pattern, Set, Tuple, Union, overload

import torch

logger = logging.getLogger(__name__)


def _validate_safetensors_shard_filename(
    filename: object, *, tensor_key: str, index_file: Path
) -> str:
    """Validate a shard filename loaded from a safetensors index."""
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: expected a non-empty string."
        )

    path = Path(filename)
    windows_path = PureWindowsPath(filename)
    if (
        "\x00" in filename
        or "\\" in filename
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
    ):
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: {filename!r} must be a relative "
            "path within the checkpoint directory."
        )

    if path.suffix != ".safetensors":
        raise ValueError(
            f"Invalid shard filename for tensor '{tensor_key}' in {index_file}: {filename!r} must end with "
            "'.safetensors'."
        )

    return filename


def _resolve_output_shard_path(output_path: Path, filename: str) -> Path:
    """Resolve a shard output path and ensure it remains inside output_path."""
    output_root = output_path.resolve()
    output_file_path = (output_root / filename).resolve()
    try:
        output_file_path.relative_to(output_root)
    except ValueError:
        raise ValueError(
            f"Shard filename {filename!r} escapes output directory {output_root}."
        ) from None
    return output_file_path


def _existing_safetensors_files(path: Path) -> set[str]:
    """Find previous checkpoint files without taking ownership of symlink targets."""
    if not path.is_dir():
        return set()
    files = {
        entry.name
        for entry in path.iterdir()
        if re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", entry.name)
        and (entry.is_file() or entry.is_symlink())
    }
    index_file = path / "model.safetensors.index.json"
    if index_file.is_file():
        files.add(index_file.name)
        try:
            with index_file.open() as f:
                index = json.load(f)
        except json.JSONDecodeError:
            return files
        for key, filename in index["weight_map"].items():
            filename = _validate_safetensors_shard_filename(
                filename, tensor_key=key, index_file=index_file
            )
            # Unlink the named file, never a target outside the output directory.
            if (path / filename).parent.resolve().is_relative_to(path):
                files.add(filename)
    return files


def _contiguous_safetensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return tensors in the contiguous layout required by safetensors."""
    return {
        name: tensor if tensor.is_contiguous() else tensor.contiguous()
        for name, tensor in tensors.items()
    }


class StateDict(Mapping[str, torch.Tensor]):
    """
    A state dict accessor that provides a unified interface for querying model
    checkpoints.

    `StateDict` allows for efficient and flexible access to tensor data from
    various sources, such as in-memory dictionaries or directories of
    `.safetensors` files. A key feature is its ability to query and load only
    the required tensors without loading the entire checkpoint into memory,
    making it highly memory-efficient for large models.

    It supports a flexible, pandas-like querying interface that allows for
    accessing tensors by exact name, a list of names, glob patterns, or regular
    expressions. This makes it easy to inspect and manipulate model
    checkpoints.

    Examples:
        >>> # Setup an example StateDict from an in-memory dictionary
        >>> import torch
        >>> import re
        >>> d = {
        ...     "model.layer.0.weight": torch.randn(10, 10),
        ...     "model.layer.0.bias": torch.randn(10),
        ...     "model.layer.1.weight": torch.randn(10, 10),
        ...     "model.layer.1.bias": torch.randn(10),
        ... }
        >>> state = StateDict(d)
        >>>
        >>> # 1. Access a single tensor by exact key
        >>> state["model.layer.0.weight"].shape
        torch.Size([10, 10])
        >>>
        >>> # 2. Access multiple tensors with a list of strings
        >>> list(state[["model.layer.0.weight", "model.layer.1.weight"]].keys())
        ['model.layer.0.weight', 'model.layer.1.weight']
        >>>
        >>> # 3. Access with a glob pattern
        >>> sorted(list(state["model.layer.*.bias"].keys()))
        ['model.layer.0.bias', 'model.layer.1.bias']
        >>>
        >>> # 4. Access with a compiled regex pattern
        >>> regex = re.compile(r"model\\\\.layer\\\\.0\\\\..*")
        >>> sorted(list(state[regex].keys()))
        ['model.layer.0.bias', 'model.layer.0.weight']

    The same querying flexibility applies to checkpoints on disk. The following
    is a conceptual example of using `StateDict` with a `SafetensorsStateSource`
    to query a sharded checkpoint without loading all of it into memory.

    .. code-block:: python

        # Assume SafetensorsStateSource is available
        # from megatron.bridge.models.state import SafetensorsStateSource

        # Imagine a directory 'my_model_checkpoint/' with sharded weights.
        state_from_disk = StateDict(SafetensorsStateSource('my_model_checkpoint/'))

        # You can query it just like the in-memory dictionary. Only the required
        # tensors (e.g., all weight tensors) will be loaded from disk.
        weights = state_from_disk["model.layer.*.weight"]
    """

    source: "StateSource"

    def __init__(self, source: Dict[str, torch.Tensor] | "StateSource"):
        """
        Initializes the StateDict query accessor.

        Args:
            source: The source of the tensor data. This can be a standard
                Python dictionary mapping tensor names to `torch.Tensor` objects,
                or an instance of a `StateSource` subclass (e.g.,
                `SafetensorsStateSource`) for more advanced, out-of-memory
                access.
        """
        if isinstance(source, dict):
            source = DictStateSource(source)

        if not isinstance(source, StateSource):
            raise TypeError(f"StateDict source must be a dict or a StateSource, got {type(source)}")

        self.source = source
        self._key_set: set[str] | None = None
        self._key_set_source: List[str] | None = None

    def _get_all_keys(self) -> List[str]:
        """
        Get all available tensor keys from the underlying source.
        """
        return self.source.get_all_keys()

    def _get_key_set(self) -> set[str]:
        # Membership checks run once per tensor during conversion; a list scan made
        # them quadratic in the number of checkpoint keys. Sources return a cached
        # list, so the set is rebuilt only if they hand back a different one (e.g.
        # while a bridge temporarily wraps ``get_all_keys`` with virtual keys).
        keys = self._get_all_keys()
        if self._key_set_source is not keys:
            self._key_set = set(keys)
            self._key_set_source = keys
        return self._key_set

    def _load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load specified tensors from the underlying source.
        """
        return self.source.load_tensors(keys_to_load)

    def _match_keys(self, pattern: Union[str, Pattern]) -> List[str]:
        """Match keys against a glob pattern or regex."""
        all_keys = self._get_all_keys()

        if isinstance(pattern, Pattern):
            # Regex pattern
            return [k for k in all_keys if pattern.search(k)]
        elif "*" in pattern or "?" in pattern or "[" in pattern:
            # Glob pattern
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
        else:
            # Exact match
            return [pattern] if pattern in self._get_key_set() else []

    @overload
    def __getitem__(self, key: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

    @overload
    def __getitem__(self, key: List[str]) -> Dict[str, torch.Tensor]: ...

    @overload
    def __getitem__(self, key: Pattern) -> Dict[str, torch.Tensor]: ...

    def __getitem__(
        self, key: Union[str, List[str], Pattern]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Accesses state dict entries using various key types.

        This method allows for retrieving tensors using:
        - A single string for an exact key match.
        - A list of strings for multiple exact key matches.
        - A string with glob-style wildcards (`*`, `?`, `[]`).
        - A compiled regular expression object.

        Args:
            key: A single key string, a list of keys, a glob pattern string, or a
                compiled regular expression.

        Returns:
            - A single `torch.Tensor` if `key` is a string that matches exactly one key
              and does not contain wildcards.
            - A `Dict[str, torch.Tensor]` for all other cases (list of keys, glob
              pattern, or regex), mapping the matched keys to their corresponding
              tensors.

        Raises:
            KeyError: If the key (or any key in a list) is not found, or if a
                pattern matches no keys.

        Examples:
            >>> d = {
            ...     "model.embed_tokens.weight": torch.randn(10, 1),
            ...     "model.layers.0.mlp.weight": torch.randn(10, 1),
            ...     "model.layers.0.self_attn.q_proj.weight": torch.randn(10, 1),
            ...     "lm_head.weight": torch.randn(10, 1),
            ... }
            >>> state = StateDict(d)
            >>>
            >>> # Exact match (returns a single tensor)
            >>> tensor = state["model.embed_tokens.weight"]
            >>> isinstance(tensor, torch.Tensor)
            True
            >>>
            >>> # List of keys (returns a dict of tensors)
            >>> tensors = state[["model.embed_tokens.weight", "lm_head.weight"]]
            >>> sorted(tensors.keys())
            ['lm_head.weight', 'model.embed_tokens.weight']
            >>>
            >>> # Glob pattern (returns a dict of tensors)
            >>> layer_0_weights = state["model.layers.0.*.weight"]
            >>> sorted(layer_0_weights.keys())
            ['model.layers.0.mlp.weight', 'model.layers.0.self_attn.q_proj.weight']
            >>>
            >>> # Regex pattern (returns a dict of tensors)
            >>> import re
            >>> attn_weights = state[re.compile(r".*self_attn.*")]
            >>> list(attn_weights.keys())
            ['model.layers.0.self_attn.q_proj.weight']
        """
        if isinstance(key, Pattern):
            matched_keys = self._match_keys(key)
            if not matched_keys:
                raise KeyError(f"No keys match regex pattern: {key.pattern}")
            return self._load_tensors(matched_keys)
        elif isinstance(key, str):
            if "*" in key or "?" in key or "[" in key:
                matched_keys = self._match_keys(key)
                if not matched_keys:
                    raise KeyError(f"No keys match pattern: {key}")
                return self._load_tensors(matched_keys)
            else:
                if key not in self._get_key_set():
                    raise KeyError(f"Key not found: {key}")
                return self._load_tensors([key])[key]
        elif isinstance(key, list):
            all_keys_set = self._get_key_set()
            missing_keys = [k for k in key if k not in all_keys_set]
            if missing_keys:
                raise KeyError(f"Keys not found: {missing_keys}")
            return self._load_tensors(key)
        else:
            raise TypeError(f"Key must be str, list of str, or compiled regex, got {type(key)}")

    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Loads and returns the entire state dict as a dictionary.

        Note:
            This method loads all tensors from the source into memory. For large
            models, this can be memory-intensive. Prefer using pattern-based
            or single-key lookups for more efficient access if you only need a
            subset of the state dict.

        Returns:
            A dictionary containing all tensor names and their corresponding
            `torch.Tensor` objects.
        """
        all_keys = self._get_all_keys()
        return self._load_tensors(all_keys)

    def keys(self) -> List[str]:
        """Get all state dict keys."""
        return self._get_all_keys()

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state dict."""
        return key in self._get_key_set()

    def __repr__(self) -> str:
        """String representation."""
        try:
            num_params = len(self)
            return f"<StateDict with {num_params} entries>"
        except Exception:
            return "<StateDict (not accessible)>"

    def get(self, key: str, default=None) -> Optional[torch.Tensor]:
        """
        Gets a tensor from the state dict.
        Returns `default` if the key is not found.
        Note: This method is for single key lookup and does not support patterns.
        """
        if key in self._get_key_set():
            return self._load_tensors([key])[key]
        return default

    def __iter__(self) -> Iterable[str]:
        """Iterate over state dict keys."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Get number of entries in the state dict."""
        return len(self.keys())

    def has_glob(self, pattern: str) -> bool:
        """
        Efficiently checks if any tensor key matches the given glob pattern.
        This is forwarded to the underlying StateSource which may have an
        optimized implementation that avoids iterating over all keys.

        Args:
            pattern: The glob pattern to match against tensor keys.

        Returns:
            True if a matching key is found, False otherwise.
        """
        return self.source.has_glob(pattern)


class StateSource(ABC, Mapping[str, torch.Tensor]):
    """
    Abstract base class for a source of model state.

    This class defines a standard interface for `StateDict` to access tensor
    data, abstracting away the details of how and where the data is stored.
    Subclasses can implement loading from different storage backends, such as
    in-memory dictionaries or files on disk. This allows `StateDict` to handle
    various checkpoint formats in a uniform way.
    """

    @abstractmethod
    def get_all_keys(self) -> List[str]:
        """Returns a list of all available tensor keys in the source."""
        pass

    @abstractmethod
    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Loads the specified tensors from the source."""
        pass

    def __getitem__(self, key: str) -> torch.Tensor:
        """Loads a single tensor by key."""
        tensors = self.load_tensors([key])
        if key not in tensors:
            raise KeyError(f"Key not found in source: {key}")
        return tensors[key]

    def __iter__(self) -> Iterable[str]:
        """Iterates over all tensor keys."""
        return iter(self.get_all_keys())

    def __len__(self) -> int:
        """Returns the total number of tensors in the source."""
        return len(self.get_all_keys())

    def has_glob(self, pattern: str) -> bool:
        """
        Checks if any tensor key matches the given glob pattern.
        This default implementation is not efficient for all sources, as it may
        load all keys. Subclasses should override this method if a more
        performant implementation is available.
        """
        import fnmatch

        for key in self.get_all_keys():
            if fnmatch.fnmatch(key, pattern):
                return True
        return False


class DictStateSource(StateSource):
    """
    A state source backed by an in-memory Python dictionary.

    This is the simplest `StateSource` implementation. It's used when the entire
    model state dict is already loaded into a dictionary in memory.

    Args:
        state_dict: A dictionary mapping tensor names (str) to `torch.Tensor` objects.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        self._dict = state_dict
        self._keys_cache: Optional[List[str]] = None

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is None:
            self._keys_cache = sorted(list(self._dict.keys()))
        return self._keys_cache

    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        return {key: self._dict[key] for key in keys if key in self._dict}


class SafeTensorsStateSource(StateSource):
    """
    A state source backed by a directory of .safetensors files.

    This source is designed for efficiently loading tensors from checkpoints saved
    in the Safetensors format, which is common for large models that are often
    "sharded" into multiple files.

    It can handle two common scenarios:
    1. A directory containing multiple `.safetensors` files.
    2. A directory containing a `model.safetensors.index.json` file, which maps
       tensor names to the specific `.safetensors` file they reside in. This is
       the standard format used by Hugging Face Transformers.

    Using this source allows `StateDict` to query for tensor keys and load only
    the necessary files and tensors from disk, avoiding high memory usage.

    Args:
        path: The path to the directory containing the `.safetensors` files
              and/or the index file. Can also be a Hugging Face Hub model ID.
        key_prefix: Select this subtree and remove its prefix from exposed keys
            and exported weights. None preserves the original names.
        revision: Optional Hub revision used for both index and shard downloads.
        hub_kwargs: Optional Hugging Face download arguments. Explicit Hub
            options or prefix selection enable index-first, lazy shard downloads.
    """

    def __init__(
        self,
        path: str | Path,
        revision: str | None = None,
        *,
        key_prefix: str | None = None,
        hub_kwargs: dict[str, object] | None = None,
    ) -> None:
        if key_prefix is not None and (not key_prefix or not key_prefix.endswith(".")):
            raise ValueError("key_prefix must be nonempty and end with '.'.")
        self.model_name_or_path = path
        self.key_prefix = key_prefix
        self.revision = revision
        self.hub_kwargs = dict(hub_kwargs or {})
        # Small tensors (block/weight scales) are cached per source instance and
        # safe_open handles are reused: export_hf_weights() re-quantizes every weight
        # against its original HF scale and asks for the scales one key at a time, which
        # otherwise re-opens the shard file per key on every refit (~150k opens for
        # DeepSeek-V4-Pro). Both can be disabled through environment variables.
        self._small_tensor_cache: Dict[str, torch.Tensor] = {}
        self._open_handles: Dict[str, Any] = {}
        self._open_handle_keys: dict[str, set[str]] = {}
        self._cache_small = os.environ.get("MBRIDGE_STATE_CACHE_SMALL_TENSORS", "1") == "1"
        self._reuse_handles = os.environ.get("MBRIDGE_STATE_REUSE_HANDLES", "1") == "1"
        self._small_numel = int(
            os.environ.get("MBRIDGE_STATE_SMALL_TENSOR_NUMEL", str(16 * 1024 * 1024))
        )
        self._lazy_hub = key_prefix is not None or hub_kwargs is not None
        self._resolved_path_cache: Optional[Path] = None
        self._keys_cache: Optional[List[str]] = None
        self._key_to_filename_map_cache: Optional[Dict[str, str]] = None

    @staticmethod
    def _filter_source_keys(
        key_to_filename_map: Mapping[str, str] | None,
        ignored_source_key_prefixes: Iterable[str] | None,
        ignored_source_key_suffixes: Iterable[str] | None,
    ) -> Dict[str, str]:
        if not key_to_filename_map:
            return {}
        if not ignored_source_key_prefixes and not ignored_source_key_suffixes:
            return dict(key_to_filename_map)

        prefixes = tuple(ignored_source_key_prefixes or ())
        suffixes = tuple(ignored_source_key_suffixes or ())
        return {
            key: filename
            for key, filename in key_to_filename_map.items()
            if not key.startswith(prefixes) and not key.endswith(suffixes)
        }

    @property
    def path(self) -> Path:
        """
        The local path to the checkpoint files.
        If the initial path is a Hugging Face Hub model ID, this property
        will handle downloading the necessary files and return the local
        cache path.
        """
        if self._resolved_path_cache is None:
            if self._lazy_hub and not Path(self.model_name_or_path).is_dir():
                from huggingface_hub.errors import EntryNotFoundError

                try:
                    index = self._download_file("model.safetensors.index.json")
                except EntryNotFoundError:
                    index = self._download_file("model.safetensors")
                self._resolved_path_cache = index.parent
            else:
                self._resolved_path_cache = self._resolve_path(
                    self.model_name_or_path, revision=self.revision
                )
        return self._resolved_path_cache

    def _download_file(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                str(self.model_name_or_path), filename, revision=self.revision, **self.hub_kwargs
            )
        )

    def _select_keys(self, key_map: Dict[str, str]) -> Dict[str, str]:
        if self.key_prefix is None:
            return key_map
        selected = {
            key.removeprefix(self.key_prefix): filename
            for key, filename in key_map.items()
            if key.startswith(self.key_prefix)
        }
        if not selected:
            raise ValueError(f"Checkpoint contains no weights under {self.key_prefix!r}.")
        return selected

    @property
    def key_to_filename_map(self) -> Dict[str, str]:
        """
        Provides a mapping from tensor keys to the safetensor filename they
        are stored in.

        This map is constructed either from `model.safetensors.index.json` if
        it exists, or by scanning all `.safetensors` files in the directory.
        The result is cached for efficiency.
        """
        if self._key_to_filename_map_cache is not None:
            return self._key_to_filename_map_cache

        # First, try to load from the index file.
        key_map = self._cached_get_key_to_filename_map(self.path)
        if key_map:
            self._key_to_filename_map_cache = self._select_keys(key_map)
            return self._key_to_filename_map_cache

        # If no index, scan the directory.
        import os
        from glob import glob as file_glob

        from safetensors import safe_open

        key_map = {}
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        for file_path in safetensor_files:
            filename = os.path.basename(file_path)
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in key_map:
                            # This is an issue. Same key in multiple files, and no index.
                            # How to resolve ambiguity? Let's just warn and overwrite. Last one wins.
                            print(
                                f"Warning: duplicate key '{key}' found in '{filename}' and '{key_map[key]}'. Using '{filename}'."
                            )
                        key_map[key] = filename
            except Exception as e:
                # Can be not a safetensor file, etc.
                print(f"Warning: could not open {filename} as a safetensors file: {e}")

        self._key_to_filename_map_cache = self._select_keys(key_map)
        return self._key_to_filename_map_cache

    @staticmethod
    def _resolve_path(model_name_or_path: Union[str, Path], revision: Optional[str] = None) -> Path:
        """
        Resolves a model name or path to a local directory.
        If the path is not a local directory, it is treated as a Hugging
        Face Hub model ID, and the corresponding files are downloaded at
        `revision` (defaulting to the repository's main branch).
        """
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            return local_path

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.constants import HF_HUB_OFFLINE
            from huggingface_hub.utils import HfHubHTTPError

            # Not a local directory, so we assume it's a model ID
            # on the Hugging Face Hub.
            return Path(
                snapshot_download(
                    repo_id=str(model_name_or_path),
                    revision=revision,
                    local_files_only=HF_HUB_OFFLINE,
                    allow_patterns=["*.safetensors", "model.safetensors.index.json"],
                    # Ignore other large files.
                    ignore_patterns=["*.bin", "*.pt", "*.pth"],
                )
            )
        except (ImportError, HfHubHTTPError, ValueError) as e:
            logger.warning(
                f"Failed to download '{model_name_or_path}' from HuggingFace Hub: {e}. "
                f"Falling back to treating it as a local path."
            )
            return local_path

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is not None:
            return self._keys_cache

        from glob import glob as file_glob

        from safetensors import safe_open

        all_keys = set()
        key_to_filename_map = self.key_to_filename_map
        if key_to_filename_map:
            all_keys.update(key_to_filename_map.keys())

        if not all_keys:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map:
                raise FileNotFoundError(
                    f"No .safetensors files or index found in {self.model_name_or_path}"
                )
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    all_keys.update(f.keys())

        self._keys_cache = sorted(list(all_keys))
        return self._keys_cache

    def _get_handle(self, file_path: Path):
        """Return a safe_open handle for ``file_path``, reused across calls when enabled."""
        from safetensors import safe_open

        if not self._reuse_handles:
            return safe_open(file_path, framework="pt", device="cpu")
        key = str(file_path)
        handle = self._open_handles.get(key)
        if handle is None:
            handle = safe_open(file_path, framework="pt", device="cpu")
            self._open_handles[key] = handle
        return handle

    def _get_handle_keys(self, file_path: Path, handle: Any) -> set[str]:
        """Return the key set of ``handle``, cached alongside reused handles."""
        if not self._reuse_handles:
            return set(handle.keys())
        key = str(file_path)
        keys = self._open_handle_keys.get(key)
        if keys is None:
            keys = set(handle.keys())
            self._open_handle_keys[key] = keys
        return keys

    def _maybe_cache(self, key: str, tensor: torch.Tensor) -> None:
        # Keep a private copy: callers may modify the tensors they get back in place.
        if self._cache_small and tensor.numel() <= self._small_numel:
            self._small_tensor_cache[key] = tensor.clone()

    def _invalidate_handle(self, file_path: Path) -> None:
        """Drop a reused handle so the next access re-reads the shard header (e.g. after a retry)."""
        self._open_handles.pop(str(file_path), None)
        self._open_handle_keys.pop(str(file_path), None)

    def load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
        if not keys_to_load:
            return {}

        import time
        from glob import glob as file_glob

        from safetensors import safe_open

        loaded_tensors = {}
        remaining_keys = set(keys_to_load)
        key_to_filename_map = self.key_to_filename_map

        if self._cache_small:
            for key in list(remaining_keys):
                cached = self._small_tensor_cache.get(key)
                if cached is not None:
                    loaded_tensors[key] = cached.clone()
                    remaining_keys.discard(key)
            if not remaining_keys:
                return loaded_tensors

        max_retries = 3

        # A selected subtree is authoritative: do not fall back to unrelated
        # shards for absent keys, including when a selected shard is incomplete.
        if self.key_prefix is not None:
            for key in keys_to_load:
                if key not in key_to_filename_map:
                    raise KeyError(key)

        if key_to_filename_map:
            file_to_keys_map = defaultdict(list)
            for key in list(remaining_keys):
                if key in key_to_filename_map:
                    filename = key_to_filename_map[key]
                    file_to_keys_map[filename].append(key)

            for filename, keys_in_file in file_to_keys_map.items():
                file_path = (
                    self._download_file(filename)
                    if self._lazy_hub and not Path(self.model_name_or_path).is_dir()
                    else self.path / filename
                )
                if file_path.exists():
                    for attempt in range(max_retries):
                        f = self._get_handle(file_path)
                        if True:  # the handle may be shared across calls; do not close it here
                            file_keys = self._get_handle_keys(file_path, f)
                            for key in keys_in_file:
                                source_key = (self.key_prefix or "") + key
                                if source_key in file_keys and key not in loaded_tensors:
                                    loaded_tensors[key] = f.get_tensor(source_key)
                                    self._maybe_cache(key, loaded_tensors[key])
                                    remaining_keys.discard(key)
                        still_missing = [k for k in keys_in_file if k in remaining_keys]
                        if not still_missing:
                            break
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries}: {len(still_missing)} keys "
                                f"not found in {filename}, retrying after delay..."
                            )
                            # the shard may be replaced meanwhile; do not retry on a stale header
                            self._invalidate_handle(file_path)
                            time.sleep(1.0 * (attempt + 1))

        if remaining_keys and self.key_prefix is None:
            safetensor_files = file_glob(str(self.path / "*.safetensors"))
            if not safetensor_files and not key_to_filename_map and not loaded_tensors:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.model_name_or_path} to load keys: {remaining_keys}"
                )
            for safetensor_file_path in safetensor_files:
                if not remaining_keys:
                    break
                with safe_open(safetensor_file_path, framework="pt", device="cpu") as f:
                    current_file_keys = f.keys()
                    for key in list(remaining_keys):
                        if key in current_file_keys:
                            loaded_tensors[key] = f.get_tensor(key)
                            remaining_keys.remove(key)

        if remaining_keys:
            raise KeyError(
                f"Keys not found in safetensors from {self.model_name_or_path}: {remaining_keys}"
            )

        return loaded_tensors

    def has_glob(self, pattern: str) -> bool:
        """
        Efficiently checks if any tensor key matches the given glob pattern.

        This method avoids loading all tensor keys into memory at once. It scans
        the checkpoint index or file headers and returns as soon as a match is
        found.

        Args:
            pattern: The glob pattern to match against tensor keys.

        Returns:
            True if a matching key is found, False otherwise.
        """
        import fnmatch
        from glob import glob as file_glob

        from safetensors import safe_open

        key_to_filename_map = self.key_to_filename_map
        if key_to_filename_map:
            for key in key_to_filename_map.keys():
                if fnmatch.fnmatch(key, pattern):
                    return True
            return False

        # If no index map, scan the files directly.
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        if not safetensor_files:
            return False

        for safetensor_file in safetensor_files:
            try:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if fnmatch.fnmatch(key, pattern):
                            return True
            except Exception:
                # Ignore files that are not valid safetensors
                continue

        return False

    def save_generator(
        self,
        generator: Iterable[Tuple[str, torch.Tensor]],
        output_path: Union[str, Path],
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        ignored_source_key_prefixes: Iterable[str] | None = None,
        ignored_source_key_suffixes: Iterable[str] | None = None,
    ) -> Set[str]:
        """
        Saves tensors from a generator to `.safetensors` files, preserving the
        original sharding structure in a memory-efficient, streaming fashion.

        This method reads the sharding information (which tensor belongs to which
        file) from the source checkpoint. It then consumes a generator of tensors,
        buffering them in memory only until a complete file shard can be written to
        disk. This approach minimizes peak memory usage compared to collecting all
        tensors first.

        If the original checkpoint had a `model.safetensors.index.json` file, a new
        one will be created for the saved tensors.

        Args:
            generator: An iterable of (tensor_name, tensor) tuples.
            output_path: The directory where the new safetensor files and index
                         will be saved.
            strict: If True (default), raises a KeyError if the generator
                    yields a tensor name not found in the original model's
                    sharding structure. If False, it prints a warning and
                    skips the tensor.
            distributed_save: Whether to enable distributed saving mode where each rank saves
                part of weights independently.
            save_every_n_ranks: Interval for saving weights across ranks in distributed mode.
                For example, if set to 2, only ranks 0, 2, 4, ... will save weights.
            ignored_source_key_prefixes: Source tensor key prefixes to omit from the expected
                source sharding map when saving.
            ignored_source_key_suffixes: Source tensor key suffixes to omit from the expected
                source sharding map when saving.

        Returns:
            Relative paths of safetensors artifacts written by this call.

        """
        if distributed_save:
            return self._save_generator_distributed(
                generator,
                output_path,
                strict,
                save_every_n_ranks=save_every_n_ranks,
                ignored_source_key_prefixes=ignored_source_key_prefixes,
                ignored_source_key_suffixes=ignored_source_key_suffixes,
            )

        # In a distributed environment, only rank 0 should write to disk.
        # Other ranks must still exhaust the generator to participate in collectives.
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0

        if rank != 0:
            # Other ranks must exhaust the generator to avoid hangs in collectives.
            for _ in generator:
                pass
            return set()

        # Rank 0 proceeds with saving.
        from safetensors.torch import save_file

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        key_to_filename_map = self._filter_source_keys(
            self.key_to_filename_map, ignored_source_key_prefixes, ignored_source_key_suffixes
        )
        all_expected_keys = set(key_to_filename_map.keys())

        if not key_to_filename_map:
            buffered_tensors = dict(generator)
            if buffered_tensors:
                save_file(
                    _contiguous_safetensors(buffered_tensors), output_path / "model.safetensors"
                )
                return {"model.safetensors"}
            return set()

        filename_to_keys_map = defaultdict(set)
        for key, filename in key_to_filename_map.items():
            filename_to_keys_map[filename].add(key)

        files_to_save = dict(filename_to_keys_map)
        remaining_keys_by_file = {filename: set(keys) for filename, keys in files_to_save.items()}
        buffered_tensors = {}
        all_yielded_keys = set()
        all_saved_keys = set()
        saved_filenames = set()
        total_saved_tensor_bytes = 0

        for name, tensor in generator:
            all_yielded_keys.add(name)
            if name not in all_expected_keys:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "Re-run with strict=False to save the partial checkpoint instead of failing."
                    )
                else:
                    print(
                        f"Warning: tensor '{name}' from generator not found in original model structure. Skipping."
                    )
                    continue

            buffered_tensors[name] = tensor

            # Update only the shard containing this tensor. Scanning every shard
            # for every yielded tensor becomes prohibitively expensive for models
            # with tens of thousands of tensors.
            filename = key_to_filename_map[name]
            remaining_keys = remaining_keys_by_file.get(filename)
            if remaining_keys is None:
                # The shard was already saved. Preserve the previous duplicate-key
                # behavior by leaving the tensor buffered for final reporting.
                continue

            remaining_keys.discard(name)
            if not remaining_keys:
                keys_for_file = files_to_save[filename]
                tensors_to_save = {key: buffered_tensors[key] for key in keys_for_file}

                output_file_path = _resolve_output_shard_path(output_path, filename)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(_contiguous_safetensors(tensors_to_save), output_file_path)
                saved_filenames.add(filename)
                total_saved_tensor_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors_to_save.values()
                )

                # Free memory by removing saved tensors from the buffer.
                for key in keys_for_file:
                    del buffered_tensors[key]

                all_saved_keys.update(keys_for_file)
                del files_to_save[filename]
                del remaining_keys_by_file[filename]

        # --- Final Reporting ---
        if files_to_save:
            if strict:
                print(
                    "Warning: The following files could not be saved because the generator did not yield all of their tensors:"
                )
            else:
                print(
                    "Warning: The following files are different from the source because the generator did not yield all "
                    "of their tensors. However they are still saved because strict=False."
                )
            for filename in files_to_save:
                missing_for_file = remaining_keys_by_file[filename]
                if missing_for_file:
                    print(f"  - {filename}: missing {len(missing_for_file)} tensors:")
                    for key in sorted(list(missing_for_file)):
                        print(f"    - {key}")
            if not strict:
                for filename in list(files_to_save.keys()):
                    keys_for_file = files_to_save[filename]
                    tensors_to_save = {
                        key: buffered_tensors[key]
                        for key in keys_for_file
                        if key in buffered_tensors
                    }
                    output_file_path = _resolve_output_shard_path(output_path, filename)
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(_contiguous_safetensors(tensors_to_save), output_file_path)
                    saved_filenames.add(filename)
                    total_saved_tensor_bytes += sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors_to_save.values()
                    )

                    # Free memory by removing saved tensors from the buffer.
                    for key in tensors_to_save.keys():
                        del buffered_tensors[key]

                    # Only the keys we actually wrote belong in the
                    # post-save index. Using ``keys_for_file`` here would
                    # claim unsaved keys as written and produce a
                    # ``model.safetensors.index.json`` that maps keys to a
                    # file which doesn't contain them — HF loading then
                    # crashes when it tries to read the missing tensor.
                    all_saved_keys.update(tensors_to_save.keys())
                    del files_to_save[filename]

        if buffered_tensors:
            print(
                f"Warning: {len(buffered_tensors)} tensors were yielded but not saved because their corresponding file shards were incomplete."
            )

        # Final check on whether all original tensors were written.
        unsaved_keys = all_expected_keys - all_saved_keys
        if unsaved_keys and strict:
            print(
                f"\nError: {len(unsaved_keys)} tensors from the original checkpoint were not written:"
            )
            for key in sorted(unsaved_keys):
                print(f"  - {key}")
            raise RuntimeError(
                f"{len(unsaved_keys)} tensors from the original checkpoint were not written. "
                "Re-run with strict=False to save the partial checkpoint instead of failing."
            )
        if not unsaved_keys:
            extra_keys = all_yielded_keys - all_expected_keys
            if extra_keys:
                print(
                    f"\nSuccess: All tensors from the original checkpoint were written. "
                    f"({len(extra_keys)} extra tensors from generator were ignored as per strict=False)."
                )
            else:
                print("\nSuccess: All tensors from the original checkpoint were written.")
        else:
            print(
                f"\nError: {len(unsaved_keys)} tensors from the original checkpoint were not written. See warnings above for details."
            )

        # Create index file for the saved shards.
        original_index_file = self.path / "model.safetensors.index.json"
        if original_index_file.exists():
            with open(original_index_file, "r") as f:
                original_index_data = json.load(f)

            new_weight_map = {key: key_to_filename_map[key] for key in all_saved_keys}

            metadata = dict(original_index_data.get("metadata", {}))
            metadata["total_size"] = total_saved_tensor_bytes
            new_index_data = {"metadata": metadata, "weight_map": new_weight_map}

            output_index_file = output_path / "model.safetensors.index.json"
            if new_weight_map:
                with open(output_index_file, "w") as f:
                    json.dump(new_index_data, f, indent=4)
                saved_filenames.add("model.safetensors.index.json")

        return saved_filenames

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_get_key_to_filename_map(
        model_name_or_path: Union[str, Path],
    ) -> Optional[Dict[str, str]]:
        """Static, cached method to get the key-to-filename map."""
        index_file = Path(model_name_or_path) / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                try:
                    index_data = json.load(f)
                    if "weight_map" in index_data and isinstance(index_data["weight_map"], dict):
                        return {
                            key: _validate_safetensors_shard_filename(
                                filename, tensor_key=key, index_file=index_file
                            )
                            for key, filename in index_data["weight_map"].items()
                        }
                except json.JSONDecodeError:
                    return None
        return None

    def _save_generator_distributed(
        self,
        generator: Iterable[Tuple[str, torch.Tensor]],
        output_path: Union[str, Path],
        strict: bool = True,
        save_every_n_ranks: int = 1,
        ignored_source_key_prefixes: Iterable[str] | None = None,
        ignored_source_key_suffixes: Iterable[str] | None = None,
    ) -> Set[str]:
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        from safetensors.torch import save_file

        output_path = Path(output_path)

        # Calculate which ranks should participate in saving
        # Only rank % save_every_n_ranks == 0 will save
        num_nodes = (world_size + save_every_n_ranks - 1) // save_every_n_ranks
        is_saver_rank = rank % save_every_n_ranks == 0
        saver_ranks = [
            i * save_every_n_ranks for i in range(num_nodes) if i * save_every_n_ranks < world_size
        ]
        num_savers = len(saver_ranks)
        saver_index = rank // save_every_n_ranks if is_saver_rank else -1

        if rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            torch.distributed.barrier()

        key_to_filename_map = self._filter_source_keys(
            self.key_to_filename_map, ignored_source_key_prefixes, ignored_source_key_suffixes
        )

        # Fallback: no sharding map, single-file save
        if not key_to_filename_map:
            saved_filenames = set()
            if is_saver_rank and saver_index == 0:
                buffered_tensors = dict(generator)
                if buffered_tensors:
                    save_file(
                        _contiguous_safetensors(buffered_tensors), output_path / "model.safetensors"
                    )
                    saved_filenames.add("model.safetensors")
            else:
                for _ in generator:
                    pass
            if is_distributed:
                torch.distributed.barrier()
            return saved_filenames

        if is_saver_rank:
            all_filenames = sorted(set(key_to_filename_map.values()))
        else:
            all_filenames = []

        # Distribute files among saver ranks (one per node)
        if is_saver_rank:
            assigned_filenames = [
                fname for idx, fname in enumerate(all_filenames) if idx % num_savers == saver_index
            ]
            assigned_filenames_set = set(assigned_filenames)
        else:
            assigned_filenames = []
            assigned_filenames_set = set()

        filename_to_keys_map: Dict[str, Set[str]] = defaultdict(set)
        if is_saver_rank:
            for key, fname in key_to_filename_map.items():
                if fname in assigned_filenames_set:
                    filename_to_keys_map[fname].add(key)

        files_to_save = {fname: filename_to_keys_map[fname] for fname in assigned_filenames}
        remaining_keys_by_file = {fname: set(keys) for fname, keys in files_to_save.items()}
        buffered_tensors: Dict[str, torch.Tensor] = {}
        local_saved_tensor_bytes = 0
        local_saved_filenames: Set[str] = set()
        local_export_error: str | None = None

        def write_shard(fname: str, tensors_to_save: Dict[str, torch.Tensor]) -> bool:
            """Write one shard while deferring saver-local failures until the generator is drained."""
            nonlocal local_export_error, local_saved_tensor_bytes

            output_file_path: Path | None = None
            try:
                output_file_path = _resolve_output_shard_path(output_path, fname)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(_contiguous_safetensors(tensors_to_save), output_file_path)
            except Exception as error:
                local_export_error = (
                    f"Rank {rank} failed to write shard '{fname}': {type(error).__name__}: {error}"
                )
                if output_file_path is not None:
                    try:
                        output_file_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                buffered_tensors.clear()
                return False

            local_saved_filenames.add(fname)
            local_saved_tensor_bytes += sum(
                saved_tensor.numel() * saved_tensor.element_size()
                for saved_tensor in tensors_to_save.values()
            )
            return True

        for name, tensor in generator:
            if name not in key_to_filename_map:
                if strict:
                    raise KeyError(
                        f"Tensor '{name}' from generator not found in the original model structure. "
                        "Re-run with strict=False to save the partial checkpoint instead of failing."
                    )
                else:
                    print(
                        f"Warning: tensor '{name}' from generator not found in original model structure. Skipping."
                    )
                    continue

            if is_saver_rank:
                fname = key_to_filename_map[name]
                if fname not in assigned_filenames_set:
                    continue

                remaining_keys = remaining_keys_by_file.get(fname)
                if local_export_error is not None:
                    continue
                if remaining_keys is None or name not in remaining_keys:
                    duplicate_message = f"Duplicate tensor '{name}' from generator."
                    if strict:
                        local_export_error = f"Rank {rank}: {duplicate_message}"
                        buffered_tensors.clear()
                    else:
                        print(f"{duplicate_message} Skipping.", flush=True)
                    continue

                buffered_tensors[name] = tensor
                remaining_keys.discard(name)
                if not remaining_keys:
                    keys_for_file = files_to_save[fname]
                    tensors_to_save = {key: buffered_tensors[key] for key in keys_for_file}
                    if write_shard(fname, tensors_to_save):
                        for key in keys_for_file:
                            del buffered_tensors[key]
                        del files_to_save[fname]
                        del remaining_keys_by_file[fname]
                    del tensors_to_save

        missing_keys = (
            set().union(*remaining_keys_by_file.values()) if remaining_keys_by_file else set()
        )
        if is_saver_rank and missing_keys:
            missing_keys_sorted = sorted(missing_keys)
            missing_preview = ", ".join(missing_keys_sorted[:20])
            missing_suffix = ""
            if len(missing_keys_sorted) > 20:
                missing_suffix = f", ... (+{len(missing_keys_sorted) - 20} more)"
            print(
                f"Rank {rank}: Missing {len(missing_keys_sorted)} tensors for keys: {missing_preview}{missing_suffix}",
                flush=True,
            )

        if is_saver_rank and not strict and local_export_error is None:
            for fname in list(files_to_save):
                keys_for_file = files_to_save[fname]
                tensors_to_save = {
                    key: buffered_tensors[key] for key in keys_for_file if key in buffered_tensors
                }
                if tensors_to_save:
                    if write_shard(fname, tensors_to_save):
                        for key in tensors_to_save:
                            del buffered_tensors[key]
                    del tensors_to_save
                    if local_export_error is not None:
                        break

        if buffered_tensors:
            print(
                f"Rank {rank}: Warning: {len(buffered_tensors)} tensors were yielded but not saved because their "
                "corresponding file shards were incomplete.",
                flush=True,
            )

        # In strict mode an incomplete shard is not written, so every key in
        # that shard is unsaved. In non-strict mode partial shards are written
        # and only keys not yielded by the generator remain unsaved.
        if strict:
            local_unsaved_keys = set().union(*files_to_save.values()) if files_to_save else set()
        else:
            local_unsaved_keys = missing_keys

        if is_saver_rank and local_unsaved_keys and strict:
            keys_sorted = sorted(local_unsaved_keys)
            preview = ", ".join(keys_sorted[:20])
            suffix = f", ... (+{len(keys_sorted) - 20} more)" if len(keys_sorted) > 20 else ""
            print(
                f"Rank {rank}: Error: {len(keys_sorted)} tensors not written: {preview}{suffix}",
                flush=True,
            )

        # Strict-mode check: ensure all expected tensors were written. Aggregate
        # per-rank missing counts so all ranks raise consistently (avoids hangs
        # on the trailing barriers).
        if is_distributed:
            gathered_save_status: list[tuple[int, int, str | None, tuple[str, ...]] | None] = [
                None
            ] * world_size
            torch.distributed.all_gather_object(
                gathered_save_status,
                (
                    len(local_unsaved_keys),
                    local_saved_tensor_bytes,
                    local_export_error,
                    tuple(sorted(local_saved_filenames)),
                ),
            )
            total_unsaved_count = sum(
                status[0] for status in gathered_save_status if status is not None
            )
            total_saved_tensor_bytes = sum(
                status[1] for status in gathered_save_status if status is not None
            )
            export_errors = [
                status[2] for status in gathered_save_status if status is not None and status[2]
            ]
            saved_filenames_aggregated = {
                fname
                for status in gathered_save_status
                if status is not None
                for fname in status[3]
            }
        else:
            total_unsaved_count = len(local_unsaved_keys)
            total_saved_tensor_bytes = local_saved_tensor_bytes
            export_errors = [local_export_error] if local_export_error else []
            saved_filenames_aggregated = local_saved_filenames

        if export_errors:
            raise RuntimeError(
                "Checkpoint export failed after all ranks drained the tensor generator: "
                + "; ".join(export_errors)
            )

        if total_unsaved_count and strict:
            raise RuntimeError(
                f"{total_unsaved_count} tensors from the original checkpoint were not written. "
                "Re-run with strict=False to save the partial checkpoint instead of failing."
            )

        # Rank 0 builds the index from the files that were actually written.
        # This avoids all_gather_object on very large key lists, which can
        # allocate excessive CUDA memory in distributed runs.
        if is_distributed:
            torch.distributed.barrier()

            if rank == 0:
                from safetensors import safe_open

                all_saved_keys_aggregated = set()
                for fname in sorted(saved_filenames_aggregated):
                    file_path = _resolve_output_shard_path(output_path, fname)
                    if not file_path.exists():
                        continue
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        all_saved_keys_aggregated.update(f.keys())
            else:
                all_saved_keys_aggregated = set()
        else:
            from safetensors import safe_open

            all_saved_keys_aggregated = set()
            for fname in sorted(saved_filenames_aggregated):
                file_path = _resolve_output_shard_path(output_path, fname)
                if not file_path.exists():
                    continue
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    all_saved_keys_aggregated.update(f.keys())

        if rank == 0:
            original_index_file = self.path / "model.safetensors.index.json"
            if original_index_file.exists():
                with open(original_index_file, "r") as f:
                    original_index_data = json.load(f)

                # Build weight_map only from actually saved keys, like the non-distributed path
                new_weight_map = {
                    key: key_to_filename_map[key]
                    for key in key_to_filename_map
                    if key in all_saved_keys_aggregated
                }

                metadata = dict(original_index_data.get("metadata", {}))
                metadata["total_size"] = total_saved_tensor_bytes
                new_index_data = {"metadata": metadata, "weight_map": new_weight_map}
                output_index_file = output_path / "model.safetensors.index.json"
                with open(output_index_file, "w") as f:
                    json.dump(new_index_data, f, indent=4)
                saved_filenames_aggregated.add("model.safetensors.index.json")

        if is_distributed:
            torch.distributed.barrier()
        return saved_filenames_aggregated if rank == 0 else set()
