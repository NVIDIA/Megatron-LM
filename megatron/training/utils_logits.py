# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Storage helpers for cached-logit tar shards.

This module keeps object-storage and WebDataset plumbing out of the cached
logits writer/reader code paths.  The helpers are intentionally small and
focused on the current batched tar layout.
"""

import concurrent.futures
import fnmatch
import glob
import importlib
import logging
import os
import time
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from megatron.core.msc_utils import MultiStorageClientFeature

logger = logging.getLogger(__name__)

MSC_PREFIX = "msc://"

_MSC_OPEN_KWARGS = {
    "attributes",
    "check_source_version",
    "disable_read_cache",
    "memory_load_limit",
    "prefetch_file",
}


def is_msc_path(path: str) -> bool:
    """Return whether *path* is an MSC URL."""
    return str(path).startswith(MSC_PREFIX)


def is_remote_storage_path(path: str) -> bool:
    """Return whether *path* needs object-storage handling."""
    return is_msc_path(path)


def _require_msc():
    """Return the MSC package or raise a Megatron-style feature error."""
    return MultiStorageClientFeature.import_package()


def _msc_if_needed(path: str):
    if is_msc_path(path):
        return _require_msc()
    if MultiStorageClientFeature.is_enabled():
        return MultiStorageClientFeature.import_package()
    return None


def storage_basename(path: str) -> str:
    """Return the final path component for local paths and MSC URLs."""
    return os.path.basename(str(path).rstrip("/"))


def storage_dirname(path: str) -> str:
    """Return the parent path while preserving URL schemes."""
    path = str(path).rstrip("/")
    if "://" not in path:
        return os.path.dirname(path)
    head, sep, _ = path.rpartition("/")
    return head if sep else ""


def storage_join(root: str, *parts: str) -> str:
    """Join path components without letting URL schemes confuse ``os.path``."""
    if not parts:
        return root
    if "://" not in str(root):
        return os.path.join(root, *parts)
    path = str(root).rstrip("/")
    for part in parts:
        path += "/" + str(part).strip("/")
    return path


def storage_makedirs(path: str, exist_ok: bool = True) -> None:
    """Create a local or MSC directory/prefix."""
    if not path:
        return
    msc = _msc_if_needed(path)
    if msc is not None:
        msc.os.makedirs(path, exist_ok=exist_ok)
    else:
        os.makedirs(path, exist_ok=exist_ok)


def open_logit_file(path: str, mode: str = "rb", **kwargs):
    """Open a local or MSC file, filtering MSC-only kwargs for builtin open."""
    msc = _msc_if_needed(path)
    if msc is not None:
        return msc.open(path, mode, **kwargs)

    local_kwargs = {k: v for k, v in kwargs.items() if k not in _MSC_OPEN_KWARGS}
    return open(path, mode, **local_kwargs)


def _split_simple_glob(pattern: str) -> Tuple[str, str]:
    parent, sep, name_pattern = str(pattern).rpartition("/")
    if not sep:
        return ".", name_pattern
    return parent, name_pattern


def storage_glob(pattern: str) -> List[str]:
    """Return paths matching *pattern* for local files or MSC URLs."""
    if is_msc_path(pattern):
        msc = _require_msc()
        if hasattr(msc, "glob"):
            return list(msc.glob(pattern))

        parent, name_pattern = _split_simple_glob(pattern)
        return [
            str(entry)
            for entry in msc.Path(parent).iterdir()
            if fnmatch.fnmatch(storage_basename(str(entry)), name_pattern)
        ]

    return glob.glob(pattern)


def storage_move(src: str, dst: str) -> None:
    """Atomically publish a local temporary file."""
    if is_msc_path(src) or is_msc_path(dst):
        raise ValueError("storage_move is local-only; write MSC objects directly")

    os.replace(src, dst)


def register_msc_webdataset_handler(wds_module) -> None:
    """Register a WebDataset opener for ``msc://`` shard URLs if possible."""
    if wds_module is None:
        return

    schemes = getattr(wds_module, "gopen_schemes", None)
    if schemes is None:
        try:
            gopen_module = importlib.import_module("webdataset.gopen")

            schemes = getattr(gopen_module, "gopen_schemes", None)
        except Exception:
            schemes = None
    if schemes is None or schemes.get("msc") is _gopen_msc:
        return

    schemes["msc"] = _gopen_msc


def _gopen_msc(url: str, mode: str = "rb", bufsize: int = 8192, **kwargs):
    if "r" not in mode:
        raise ValueError("WebDataset MSC handler only supports reading")
    # ``bufsize`` is part of WebDataset's opener signature; MSC manages buffering.
    kwargs.setdefault("prefetch_file", True)
    return open_logit_file(url, "rb", **kwargs)


class TarShardPrefetcher:
    """Asynchronously materialize whole tar shards into the MSC cache."""

    def __init__(
        self,
        *,
        enabled: bool,
        depth: int = 2,
        max_workers: Optional[int] = None,
    ):
        self.enabled = bool(enabled and depth > 0)
        self.depth = depth

        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._futures: dict[str, concurrent.futures.Future] = {}

        if self.enabled:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, max_workers or self.depth),
                thread_name_prefix="logits-tar-prefetch",
            )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._futures.clear()

    def __enter__(self) -> "TarShardPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def schedule(self, url: str) -> None:
        if not self.enabled or url in self._futures:
            return
        assert self._executor is not None
        self._futures[url] = self._executor.submit(self._prefetch_url, url)

    def schedule_group(self, urls: Sequence[str]) -> None:
        for url in urls:
            self.schedule(url)

    def wait(self, url: str) -> None:
        future = self._futures.pop(url, None)
        if future is None:
            return
        start = time.monotonic()
        future.result()
        waited = time.monotonic() - start
        if waited > 0.5:
            logger.warning(
                "Waited %.3fs for cached-logit tar shard prefetch: %s", waited, url
            )

    def wait_group(self, urls: Sequence[str]) -> None:
        for url in urls:
            self.wait(url)

    def iter_prefetched(
        self, groups: Iterable[Sequence[str]]
    ) -> Iterator[Tuple[str, ...]]:
        """Yield URL groups, waiting only when a prefetched group is not ready."""
        group_list = [tuple(group) for group in groups]
        if not self.enabled:
            yield from group_list
            return

        for idx in range(min(self.depth, len(group_list))):
            self.schedule_group(group_list[idx])

        for idx, group in enumerate(group_list):
            next_idx = idx + self.depth
            if next_idx < len(group_list):
                self.schedule_group(group_list[next_idx])
            self.wait_group(group)
            yield group

    def _prefetch_url(self, url: str) -> None:
        # Whole-object caching avoids tar-member range bookkeeping while still
        # keeping the object download ahead of WebDataset's sequential reader.
        with open_logit_file(url, "rb", prefetch_file=True) as stream:
            stream.read()
