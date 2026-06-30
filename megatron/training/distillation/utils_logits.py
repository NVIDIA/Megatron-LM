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
from dataclasses import dataclass
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

# On-disk payload format version.  ``1`` is the legacy list-of-microbatches
# layout (``values: List[Tensor]``); ``2`` is the monolithic per-iter layout
# (``values: Tensor``) that supports flexible MBS / DP / GBS resharding.
LOGPROBS_FORMAT_VERSION = 2

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


def parse_logprobs_metadata(data: bytes) -> Dict[str, Any]:
    """Parse the JSON metadata stored in the ``_meta.json`` member."""
    return json.loads(data)


def _verify_logprobs_metadata(
    data: bytes,
    *,
    tar_path: str,
    expected_hash: Optional[str],
) -> Dict[str, Any]:
    """Parse the per-tar ``_meta.json`` and validate its dataset hash.

    Returns the parsed metadata dict regardless of whether hash validation
    runs; the caller may consume fields like ``format_version`` or the
    saver-side ``mbs_save`` / ``dp_size_save`` / ``gbs_save`` records.
    """
    meta = parse_logprobs_metadata(data)
    if expected_hash is not None:
        saved_hash = meta.get("hash")
        if saved_hash != expected_hash:
            raise RuntimeError(
                f"Teacher tar {tar_path} was saved with hash {saved_hash} "
                f"but the current student run has hash {expected_hash}. "
                "Data does not align!"
            )
    return meta


def iter_logprobs_tar_entries(
    tar_path: str,
    *,
    start_iteration: int = 0,
    expected_hash: Optional[str] = None,
    metadata_out: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[LogprobsTarEntry]:
    """Stream cached-logits payload members from a batched tar archive.

    The tar format is written by :class:`LogitsSaverHooks`: a leading
    ``_meta.json`` member followed by ``{iteration}.pt.zst`` payloads.
    Members before *start_iteration* are skipped before their payload bytes
    are materialized in Python.

    If *metadata_out* is provided, the parsed ``_meta.json`` dict is appended
    to it once during the scan so callers can read fields such as
    ``format_version`` without a second pass through the tar.
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
                    meta = _verify_logprobs_metadata(
                        extracted.read(),
                        tar_path=tar_path,
                        expected_hash=expected_hash,
                    )
                    if metadata_out is not None:
                        metadata_out.append(meta)
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


def peek_logprobs_metadata(tar_path: str) -> Optional[Dict[str, Any]]:
    """Read the ``_meta.json`` member of *tar_path* without decoding payloads.

    Returns ``None`` when the tar exists but contains no metadata member
    (e.g. a malformed archive).  Useful for interactive inspection; the
    loader factory uses :func:`peek_first_logprobs_metadata` instead.

    Deliberately does **not** pass ``prefetch_file=True`` to
    :func:`open_logit_file`: the metadata member is at the head of the
    tar (written as the first member by the saver), and the streaming
    tarfile reader only needs enough bytes to reach it.  On MSC backends
    that support byte-range reads this turns a full-object download
    (potentially hundreds of MB) into a small header fetch, avoiding a
    multi-second stall on the loader's first ``__call__``.
    """
    with open_logit_file(tar_path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                if not member.isreg():
                    continue
                if member.name == META_TAR_MEMBER:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        return None
                    return parse_logprobs_metadata(extracted.read())
                break
    return None


def peek_first_logprobs_metadata(logprobs_dir: str) -> Optional[Dict[str, Any]]:
    """Read ``_meta.json`` from any available tar in *logprobs_dir*.

    Rank-0 does the I/O and broadcasts the parsed dict to the rest of the
    world to keep object-store list/get calls down to one.  Returns
    ``None`` if no tars are present yet.  Used by the loader factory to
    discover the on-disk format version before instantiating a dataset.
    """
    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if distributed and dist.get_rank() != 0:
        meta: Optional[Dict[str, Any]] = None
    else:
        tars = storage_glob(os.path.join(logprobs_dir, "*.tar"))
        meta = peek_logprobs_metadata(tars[0]) if tars else None
    if distributed:
        payload: List[Optional[Dict[str, Any]]] = [meta]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]
    return meta


def decode_logprobs_payload(data: bytes) -> Tuple[Any, Any]:
    """Decode one zstd-compressed cached-logits payload.

    Dispatches on the payload's ``format_version`` field:

    * v2 (``format_version >= 2``): returns ``(values, indices)`` as
      monolithic per-iteration tensors of shape
      ``(seq, samples_per_dp_per_iter, K)``.  Slicing along the sample
      dim is a zero-copy view, which keeps pinned-memory transfers
      efficient.
    * v1 (legacy, no ``format_version``): returns
      ``(values_list, indices_list)`` mirroring the original saved
      list-of-microbatches structure.  The v1 branch is slated for
      removal alongside :class:`LegacyTeacherTarDataset`.

    Each caller knows which format its tars are in (the loader factory
    inspects ``_meta.json`` up-front) and can treat the return type as
    fixed.
    """
    data = zstandard.ZstdDecompressor().decompress(data)
    tensors = torch.load(io.BytesIO(data), weights_only=True)
    if tensors.get("format_version", 1) >= 2:
        values = tensors["values"]
        indices = unpack_indices(tensors["indices_low"], tensors["bit_17"])
        return values, indices
    # ---- v1 LEGACY (remove with LegacyTeacherTarDataset) ----
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
def load_log_probs_from_tar(tar_path: str, iteration: int):
    """Load one iteration from a specific batched tar shard.

    Returns ``(values, indices)`` as tensors for v2 tars or as lists of
    per-microbatch tensors for v1 tars (``decode_logprobs_payload``
    dispatches internally on ``format_version``).
    """
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


# ---------------------------------------------------------------------------
#  Resharding plan (MBS / DP / GBS change between save and load)
# ---------------------------------------------------------------------------

# Yielded by ``LogprobsReshardPlan.sources_for_microbatch``.
class _ReshardSource(NamedTuple):
    iter_save: int
    d_save: int
    row_start: int
    row_end: int


@dataclass(frozen=True)
class LogprobsReshardPlan:
    """Pure-arithmetic mapping from a load (iter, microbatch, DP rank) tuple to
    the saved (iter, DP rank, row range) tuples that supply its samples.

    Built once at loader init from the saved ``mbs_save / dp_save / gbs_save``
    metadata and the current ``mbs_load / dp_load / gbs_load`` settings.  Holds
    no tensors and no I/O state, so it is cheap to construct and trivial to
    unit-test.

    Construction validates the "integer multiples in either direction"
    restriction on each of {MBS, DP, GBS} plus the Megatron
    ``gbs % (mbs * dp) == 0`` invariant on both sides, raising
    :class:`ValueError` with a precise diagnostic on failure.

    The math relies on ``MegatronPretrainingSampler``'s mapping: within a
    saved iteration, the monolith's row ``r`` at saved DP rank ``d_save``
    corresponds to within-iter sample index

    ``x = ((r // M_save) * D_save + d_save) * M_save + (r % M_save)``.

    Inverting that gives the ``d_save`` / ``row_offset`` for any required
    within-iter sample index.  Crossing an ``M_save`` boundary changes
    ``d_save``; crossing a ``gbs_save`` boundary changes ``iter_save``.
    """

    mbs_save: int
    dp_save: int
    gbs_save: int
    mbs_load: int
    dp_load: int
    gbs_load: int

    def __post_init__(self) -> None:
        if self.gbs_save % self.gbs_load != 0 and self.gbs_load % self.gbs_save != 0:
            raise ValueError(
                f"gbs_save ({self.gbs_save}) and gbs_load ({self.gbs_load}) must "
                "be integer multiples of one another for cached-logits resharding."
            )
        if self.mbs_save % self.mbs_load != 0 and self.mbs_load % self.mbs_save != 0:
            raise ValueError(
                f"mbs_save ({self.mbs_save}) and mbs_load ({self.mbs_load}) must "
                "be integer multiples of one another for cached-logits resharding."
            )
        if self.dp_save % self.dp_load != 0 and self.dp_load % self.dp_save != 0:
            raise ValueError(
                f"dp_save ({self.dp_save}) and dp_load ({self.dp_load}) must "
                "be integer multiples of one another for cached-logits resharding."
            )
        if self.gbs_load % (self.mbs_load * self.dp_load) != 0:
            raise ValueError(
                f"gbs_load ({self.gbs_load}) must be divisible by mbs_load * "
                f"dp_load ({self.mbs_load} * {self.dp_load})."
            )
        if self.gbs_save % (self.mbs_save * self.dp_save) != 0:
            raise ValueError(
                f"gbs_save ({self.gbs_save}) must be divisible by mbs_save * "
                f"dp_save ({self.mbs_save} * {self.dp_save})."
            )

    @property
    def num_mb_load(self) -> int:
        """Number of microbatch steps per load iteration."""
        return self.gbs_load // (self.mbs_load * self.dp_load)

    @property
    def num_mb_save(self) -> int:
        """Number of microbatch steps per saved iteration."""
        return self.gbs_save // (self.mbs_save * self.dp_save)

    @property
    def samples_per_save_shard(self) -> int:
        """Sample count along the monolith's batch dim per saved (iter, DP)."""
        return self.gbs_save // self.dp_save

    def needed_d_saves(self, d_load: int) -> List[int]:
        """Sorted list of saved DP ranks that this load DP rank ever reads from.

        The pattern is ``gbs_save``-periodic, so a single load iteration
        (i_load = 0) is enough to enumerate the full set.
        """
        needed: set[int] = set()
        # Step by mbs_save so we observe every d_save value the microbatches
        # span (only relevant when mbs_load > mbs_save).
        for m_load in range(self.num_mb_load):
            base = (m_load * self.dp_load + d_load) * self.mbs_load
            for o in range(0, self.mbs_load, self.mbs_save):
                x_save = (base + o) % self.gbs_save
                needed.add((x_save // self.mbs_save) % self.dp_save)
            # The last partial M_save block (if any) is captured above because
            # the step is M_save and o stops at mbs_load - mbs_load % mbs_save.
            # If mbs_load < mbs_save the loop runs once for o=0 only, which is
            # correct: the whole microbatch lives in one M_save block.
        return sorted(needed)

    def sources_for_microbatch(
        self, i_load: int, m_load: int, d_load: int
    ) -> Iterator[_ReshardSource]:
        """Yield the saved (iter, DP rank, row range) tuples that cover one
        load microbatch's ``mbs_load`` contiguous global samples, in order.

        With the integer-multiples restriction enforced by
        :meth:`__post_init__` each yielded fragment maps to a contiguous
        slice of one saved shard's monolith (``tensor[:, a:b, :]``), so the
        loader's per-step assembly is at worst a small ``torch.cat``.
        """
        b = (m_load * self.dp_load + d_load) * self.mbs_load
        g = i_load * self.gbs_load + b
        remaining = self.mbs_load
        while remaining > 0:
            i_save = g // self.gbs_save
            x_save = g % self.gbs_save
            d_save = (x_save // self.mbs_save) % self.dp_save
            row_offset = (
                (x_save // self.mbs_save) // self.dp_save * self.mbs_save
                + (x_save % self.mbs_save)
            )
            # Stop the fragment at the nearest M_save boundary (d_save changes
            # past it) or at the saved-iter boundary (i_save changes past it).
            chunk = min(
                remaining,
                self.mbs_save - (x_save % self.mbs_save),
                self.gbs_save - x_save,
            )
            yield _ReshardSource(i_save, d_save, row_offset, row_offset + chunk)
            g += chunk
            remaining -= chunk
