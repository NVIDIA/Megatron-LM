# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Storage helpers for cached-logit tar shards.

This module keeps object-storage and tar-format plumbing out of the cached
logits writer/reader code paths.  The helpers are intentionally small and
focused on the current batched tar layout.
"""

import concurrent.futures
import fnmatch
import glob
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import zstandard

from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.training import get_args
from megatron.training.utils import get_blend_and_blend_per_split

logger = logging.getLogger(__name__)

MSC_PREFIX = "msc://"

# Matches batched tar filenames.  New shards omit the CP prefix
# (``dp{D}__{I}.tar``); legacy compatible shards may use
# ``cp0_dp{D}__{I}.tar``.  Named groups:
#   cp   – optional CP rank
#   dp   – DP rank
#   iter – trailing iteration number *I*
BATCHED_TAR_RE = re.compile(
    r"^(?:cp(?P<cp>\d+)_)?dp(?P<dp>\d+)__(?P<iter>\d+)\.tar$"
)

# Name of the metadata member written as the first entry of every batched
# tar. Contains the dataset-identity hash and the fields that produced it.
META_TAR_MEMBER = "_meta.json"

LOGPROBS_TAR_MEMBER_SUFFIX = ".pt.zst"

# Matches compressed iteration payload members inside a batched tar archive.
LOGPROBS_TAR_MEMBER_RE = re.compile(
    rf"^(?P<iter>\d+){re.escape(LOGPROBS_TAR_MEMBER_SUFFIX)}$"
)

CACHED_LOGITS_LOGPROB_SENTINEL = -1e3
CACHED_LOGITS_INDEX_SENTINEL = -1

# Cache to not have to re-run the glob operation for the same pattern, which is expensive for MSC.
_STORAGE_GLOB_CACHE: dict[str, List[str]] = {}


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


def storage_makedirs(path: str, exist_ok: bool = True) -> None:
    """Create a local or MSC directory/prefix."""
    if not path:
        return
    msc = _msc_if_needed(path)
    if msc is not None:
        msc.os.makedirs(path, exist_ok=exist_ok)
    else:
        os.makedirs(path, exist_ok=exist_ok)


def storage_move(src: str, dst: str) -> None:
    """Atomically publish a local temporary file."""
    if is_msc_path(src) or is_msc_path(dst):
        raise ValueError("storage_move is local-only; write MSC objects directly")

    os.replace(src, dst)


def storage_glob(pattern: str) -> List[str]:
    """Return paths matching *pattern* for local files or MSC URLs."""
    if is_msc_path(pattern):
        msc = _require_msc()
        return list(msc.glob(pattern))

    return glob.glob(pattern)


def _storage_glob_rank0(pattern: str, cached: bool = False) -> List[str]:
    """Run ``storage_glob`` on rank 0 and broadcast the result."""
    if cached:
        if (paths := _STORAGE_GLOB_CACHE.get(pattern)) is not None:
            return paths

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        payload = [storage_glob(pattern) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        paths = payload[0]
    else:
        paths = storage_glob(pattern)

    paths = list(paths or [])
    _STORAGE_GLOB_CACHE[pattern] = paths
    return paths


def storage_glob_with_caching(root: str, name_pattern: str, cached: bool = True) -> List[str]:
    """Glob under *root* and filter the result by basename."""
    if not is_remote_storage_path(root):
        return storage_glob(os.path.join(root, name_pattern))
    listing = _storage_glob_rank0(os.path.join(root, "*.tar"), cached=cached)

    return [
        str(path)
        for path in listing
        if fnmatch.fnmatch(storage_basename(str(path)), name_pattern)
    ]


def get_current_iteration() -> int:
    """Return the current training iteration from ``get_args()``."""
    args = get_args()
    iteration = getattr(args, 'curr_iteration', None)
    if iteration is None:
        iteration = getattr(args, 'iteration')
    return iteration


def _blend_identifiers(args: Any) -> Dict[str, Any]:
    """Build a path-agnostic representation of the training data blend."""
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    def _normalise(blend_tuple) -> Optional[List[List[Any]]]:
        if blend_tuple is None:
            return None
        prefixes, weights = blend_tuple
        if weights is None:
            weights = [1.0] * len(prefixes)
        return [
            [float(w), os.path.splitext(os.path.basename(str(p)))[0]]
            for w, p in zip(weights, prefixes)
        ]

    if blend is not None:
        return {"kind": "blend", "blend": _normalise(blend)}
    if blend_per_split is not None:
        # Index 0 is the train split (see ``megatron.core.datasets.utils.Split``).
        return {"kind": "blend_per_split", "train": _normalise(blend_per_split[0])}
    return {"kind": "mock", "mock": True}


def compute_dataset_hash() -> Tuple[str, Dict[str, Any]]:
    """Compute the dataset-identity hash for the current training run.

    The fields included are exactly those that determine the global sample
    stream itself: ``seed``, ``train_samples`` (with a fall-back to
    ``train_iters * global_batch_size``), and the data ``blend``.
    """
    args = get_args()
    train_samples = getattr(args, 'train_samples', None)
    if train_samples is None:
        train_iters = getattr(args, 'train_iters', None)
        global_batch_size = getattr(args, 'global_batch_size', None)
        if train_iters is not None and global_batch_size is not None:
            train_samples = int(train_iters) * int(global_batch_size)

    identifiers = OrderedDict()
    identifiers["seed"] = getattr(args, 'seed', None)
    identifiers["train_samples"] = train_samples
    identifiers["blend"] = _blend_identifiers(args)

    description = json.dumps(identifiers, sort_keys=False, separators=(',', ':'))
    md5_hex = hashlib.md5(
        description.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return md5_hex, dict(identifiers)


def batched_tar_filename(dp_rank: int, last_iter: int) -> str:
    """Return the canonical filename for a CP-agnostic DP batched tar shard."""
    return f"dp{dp_rank}__{last_iter}.tar"


def batched_tar_prefix(dp_rank: int) -> str:
    """Return the canonical glob prefix for CP-agnostic DP batched tar shards."""
    return f"dp{dp_rank}__"


def sorted_batched_tars(paths: List[str]) -> List[str]:
    """Sort batched tar paths by their iteration number (numeric, ascending)."""
    keyed = []
    for path in paths:
        if match := BATCHED_TAR_RE.match(storage_basename(path)):
            keyed.append((int(match.group("iter")), path))
    keyed.sort()
    return [path for _, path in keyed]


def pad_topk_dim(tensor: torch.Tensor, target_k: int, pad_value: float) -> torch.Tensor:
    """Pad the top-K dimension so CP ranks can gather variable top-P widths."""
    current_k = tensor.size(-1)
    if current_k == target_k:
        return tensor
    pad_shape = (*tensor.shape[:-1], target_k - current_k)
    padding = torch.full(
        pad_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=-1)


def slice_tensor_for_cp_rank(tensor: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Return this CP rank's zigzag sequence slice from a full sequence tensor."""
    if cp_size <= 1:
        return tensor

    num_cp_chunks = 2 * cp_size
    if tensor.size(0) % num_cp_chunks != 0:
        raise ValueError(
            f"Sequence length ({tensor.size(0)}) must be divisible by "
            f"2 * CP size ({num_cp_chunks}) for CP zigzag slicing."
        )
    chunk_size = tensor.size(0) // num_cp_chunks
    view_shape = (2 * cp_size, chunk_size, *tensor.shape[1:])
    chunks = tensor.view(*view_shape)
    index = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1],
        dtype=torch.long,
        device=tensor.device,
    )
    local = chunks.index_select(0, index)
    return local.reshape(2 * chunk_size, *tensor.shape[1:]).contiguous()


def reassemble_cp_sequence(local_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Reassemble CP-rank zigzag sequence shards into full sequence order."""
    cp_size = len(local_tensors)
    if cp_size == 1:
        return local_tensors[0]

    first = local_tensors[0]
    if first.size(0) % 2 != 0:
        raise ValueError(
            f"Local CP sequence length ({first.size(0)}) must be divisible by 2 "
            "to reassemble zigzag CP shards."
        )
    chunk_size = first.size(0) // 2
    chunks: List[Optional[torch.Tensor]] = [None] * (2 * cp_size)
    for cp_rank, tensor in enumerate(local_tensors):
        if tensor.shape != first.shape:
            raise ValueError(
                f"All CP shards must have matching shapes; got {tensor.shape} "
                f"for rank {cp_rank}, expected {first.shape}."
            )
        local_chunks = tensor.view(2, chunk_size, *tensor.shape[1:])
        chunks[cp_rank] = local_chunks[0]
        chunks[2 * cp_size - cp_rank - 1] = local_chunks[1]

    return torch.cat([chunk for chunk in chunks if chunk is not None], dim=0).contiguous()


def pack_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split 17-bit global indices into uint16 lower bits + bool 17th bit."""
    low_bits = (indices & 0xFFFF).to(torch.uint16)
    bit_17 = (indices >> 16).to(torch.bool)
    return low_bits, bit_17


def unpack_indices(low_bits: torch.Tensor, bit_17: torch.Tensor) -> torch.Tensor:
    """Reconstruct indices from uint16 lower bits + bool 17th bit."""
    return (bit_17.long() << 16) | low_bits.long()


class LogprobsTarEntry(NamedTuple):
    """Raw cached-logits payload read from one batched tar member."""

    iteration: int
    data: bytes


def open_logit_file(path: str, mode: str = "rb", **kwargs):
    """Open a local or MSC file, filtering MSC-only kwargs for builtin open."""
    msc = _msc_if_needed(path)
    if msc is not None:
        return msc.open(path, mode, **kwargs)

    msc_open_kwargs = {
        "attributes",
        "check_source_version",
        "disable_read_cache",
        "memory_load_limit",
        "prefetch_file",
    }
    local_kwargs = {k: v for k, v in kwargs.items() if k not in msc_open_kwargs}
    return open(path, mode, **local_kwargs)


def _verify_logprobs_metadata(
    data: bytes,
    *,
    tar_path: str,
    expected_hash: Optional[str],
) -> None:
    """Validate the per-tar dataset hash stored in ``_meta.json``."""
    if expected_hash is None:
        return
    saved_hash = json.loads(data).get("hash")
    if saved_hash != expected_hash:
        raise RuntimeError(
            f"Teacher tar {tar_path} was saved with hash {saved_hash} "
            f"but the current student run has hash {expected_hash}. "
            "Data does not align!"
        )


def iter_logprobs_tar_entries(
    tar_path: str,
    *,
    start_iteration: int = 0,
    expected_hash: Optional[str] = None,
) -> Iterator[LogprobsTarEntry]:
    """Stream cached-logits payload members from a batched tar archive.

    The tar format is written by :class:`LogitsSaverHooks`: a leading
    ``_meta.json`` member followed by ``{iteration}.pt.zst`` payloads.
    Members before *start_iteration* are skipped before their payload bytes
    are materialized in Python.
    """
    metadata_seen = False

    with open_logit_file(tar_path, "rb", prefetch_file=True) as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                if not member.isreg():
                    continue

                name = member.name
                if name == META_TAR_MEMBER:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Could not read metadata member in '{tar_path}'")
                    _verify_logprobs_metadata(
                        extracted.read(),
                        tar_path=tar_path,
                        expected_hash=expected_hash,
                    )
                    metadata_seen = True
                    continue

                match = LOGPROBS_TAR_MEMBER_RE.match(name)
                if match is None:
                    continue

                if expected_hash is not None and not metadata_seen:
                    raise RuntimeError(
                        f"Teacher tar {tar_path} does not contain leading "
                        f"{META_TAR_MEMBER}; cannot verify data alignment."
                    )

                iteration = int(match.group("iter"))
                if iteration < start_iteration:
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(
                        f"Could not read log-probs member '{name}' in '{tar_path}'"
                    )
                yield LogprobsTarEntry(
                    iteration=iteration,
                    data=extracted.read(),
                )

    if expected_hash is not None and not metadata_seen:
        raise RuntimeError(
            f"Teacher tar {tar_path} does not contain {META_TAR_MEMBER}; "
            "cannot verify data alignment."
        )


def decode_logprobs_payload(data: bytes) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Decode one zstd-compressed cached-logits payload."""
    data = zstandard.ZstdDecompressor().decompress(data)
    tensors = torch.load(io.BytesIO(data), weights_only=True)
    indices_list = [
        unpack_indices(low, bit17)
        for low, bit17 in zip(tensors["indices_low"], tensors["bit_17"])
    ]
    return tensors["values"], indices_list


def detect_saved_dp_size(logprobs_dir: str) -> Optional[int]:
    """Scan *logprobs_dir* for batched tars and return the saved DP world size.

    Legacy prefixed shards are only considered compatible when the saved CP
    rank is 0.  Higher CP ranks imply old per-CP-rank payloads that cannot be
    safely re-sliced under a different CP size.
    """
    dp_ranks_found: set[int] = set()
    for path in storage_glob_with_caching(logprobs_dir, "*.tar"):
        fname = storage_basename(path)
        if match := BATCHED_TAR_RE.match(fname):
            cp_group = match.group("cp")
            if cp_group is not None and int(cp_group) > 0:
                raise ValueError(
                    f"Found cached-logits tar shards with saved CP rank {cp_group}. "
                    "Only legacy shards with cp0_ prefix are compatible."
                )
            dp_ranks_found.add(int(match.group("dp")))
    if not dp_ranks_found:
        return None
    return max(dp_ranks_found) + 1


# NOTE: This function is for interactive debugging purposes
def load_log_probs_from_tar(
    tar_path: str,
    iteration: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load one iteration from a specific batched tar shard."""
    for entry in iter_logprobs_tar_entries(tar_path, start_iteration=iteration):
        if entry.iteration == iteration:
            return decode_logprobs_payload(entry.data)
        if entry.iteration > iteration:
            break

    raise FileNotFoundError(
        f"No log-probs member found for iteration {iteration} in '{tar_path}'"
    )


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

    def schedule_group(self, urls: Sequence[str]) -> None:
        for url in urls:
            if not self.enabled or url in self._futures:
                return
            assert self._executor is not None
            self._futures[url] = self._executor.submit(self._prefetch_url, url)

    def wait_group(self, urls: Sequence[str]) -> None:
        for url in urls:
            future = self._futures.pop(url, None)
            if future is None:
                return
            elapsed = future.result()
            logger.debug("Prefetch fetch time for %s: %.3fs", url, elapsed)

    def iter_prefetched(self, groups: Iterable[Sequence[str]]) -> Iterator[Tuple[str, ...]]:
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

    def _prefetch_url(self, url: str) -> float:
        # Whole-object caching avoids tar-member range bookkeeping while still
        # keeping the object download ahead of the sequential tar reader.
        start = time.monotonic()
        with open_logit_file(url, "rb", prefetch_file=True) as stream:
            stream.read()
        return time.monotonic() - start
