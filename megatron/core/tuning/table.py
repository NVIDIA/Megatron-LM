# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tuned Triton configs, stored as data rather than as source.

A tuned table lets a pinned run use the *fastest* config instead of the cheapest
one while staying a pure function of its inputs. Tables live as JSON so adding
an architecture is a file drop, not a source edit and a rebuild:

    torchrun ... pretrain_gpt.py ... --triton-autotune-record-path /tmp/rec
    python -m megatron.core.tuning merge /tmp/rec.rank*.json -o ~/.mcore/tuning/sm103.json
    torchrun ... --triton-autotune-mode pinned --triton-autotune-table-path ~/.mcore/tuning

Each file records one architecture plus the provenance needed to notice when it
has gone stale::

    {"arch": "sm103",
     "triton": "3.6.0",
     "packages": {"mamba_ssm": "2.3.1"},
     "source": "nemotron_3_ultra/96gpu",
     "kernels": {"<kernel>": {"<shape key>": {"kwargs": {...},
                                              "num_warps": 4, "num_stages": 3}}}}

A miss is never an error by default: the caller falls back to a deterministic
choice. It never falls back to a timed one.
"""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path

from megatron.core.tuning.selection import config_data

logger = logging.getLogger(__name__)

_PACKAGED = Path(__file__).parent / "tables"


def _triton_version() -> str:
    try:
        import triton

        return getattr(triton, "__version__", "") or ""
    except ImportError:
        return ""


def _package_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in ("mamba-ssm", "transformer-engine", "triton"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


class TunedTable:
    """Tuned configs for one architecture, with membership validation."""

    def __init__(self, arch: str, kernels: dict, provenance: dict | None = None):
        self.arch = arch
        self.kernels = kernels
        self.provenance = provenance or {}

    def __bool__(self) -> bool:
        return bool(self.kernels)

    def lookup(self, kernel: str, key: str, candidates):
        """Return the tuned config for this kernel and shape, or ``None``.

        The stored entry is matched back against the kernel's own candidate list
        rather than rebuilt, so its ``pre_hook`` survives, and a stale entry that no longer
        names a real candidate degrades to a miss instead of an invalid launch.
        """
        entries = self.kernels.get(kernel)
        if not entries:
            return None
        wanted = entries.get(key) or entries.get("*")
        if not wanted:
            return None
        # Older tables omitted these launch options; they used Triton's defaults.
        wanted = {"num_ctas": 1, "maxnreg": None, "ir_override": None, **wanted}
        for config in candidates:
            if config_data(config) == wanted:
                return config
        return None


def _search_dirs(extra) -> list:
    dirs = []
    for entry in extra or ():
        directory = Path(entry).expanduser()
        if not directory.is_dir():
            logger.warning("Tuned-table directory %s does not exist; skipping it", directory)
            continue
        dirs.append(directory)
    dirs.append(_PACKAGED)
    return dirs


def load(arch: str, table_path=()) -> TunedTable:
    """Load the first table for ``arch`` found on the search path.

    User directories are searched before the packaged defaults, so a locally
    recorded table wins without touching the tree. A table recorded against a
    different Triton warns rather than being dropped: entries are validated
    against the live candidate list anyway, so the worst case is a miss.
    """
    for directory in _search_dirs(table_path):
        path = directory / f"{arch}.json"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable tuned table %s: %s", path, exc)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("kernels"), dict):
            logger.warning("Ignoring %s: not a tuned table (no 'kernels' mapping)", path)
            continue
        if data.get("arch", arch) != arch:
            logger.warning(
                "Ignoring %s: it holds a table for %s, not %s", path, data.get("arch"), arch
            )
            continue
        recorded = data.get("triton", "")
        current = _triton_version()
        if recorded and current and recorded != current:
            logger.warning(
                "Tuned table %s was recorded against triton %s but this process has %s; "
                "entries that no longer match a live config will fall back to the "
                "deterministic default.",
                path,
                recorded,
                current,
            )
        return TunedTable(arch, data["kernels"], data)
    return TunedTable(arch, {})


def _read_recording(path) -> dict:
    """Load one per-rank recording, rejecting files of any other shape by name."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc}); the capture may be truncated") from exc
    if isinstance(data, dict) and "kernels" in data and isinstance(data.get("arch"), str):
        raise ValueError(f"{path}: this is a tuned table, not a per-rank recording")
    if not isinstance(data, dict) or not all(
        isinstance(kernels, dict)
        and all(
            isinstance(entries, dict) and all(isinstance(c, dict) for c in entries.values())
            for entries in kernels.values()
        )
        for kernels in data.values()
    ):
        raise ValueError(f"{path}: expected {{arch: {{kernel: {{shape key: config}}}}}}")
    return data


def count_votes(paths) -> dict:
    """Count, per (arch, kernel, key), how many ranks recorded each config.

    Raises ``OSError`` for unreadable files and ``ValueError`` naming the file for
    anything that is not a per-rank recording.
    """
    votes: dict = collections.defaultdict(collections.Counter)
    for path in sorted(paths):
        for arch, kernels in _read_recording(path).items():
            for kernel, entries in kernels.items():
                for key, config in entries.items():
                    votes[(arch, kernel, key)][json.dumps(config, sort_keys=True)] += 1
    return votes


def merge_votes(votes: dict) -> dict:
    """Majority winner per (arch, kernel, key); ties break on the serialized config."""
    merged: dict = {}
    for arch, kernel, key in sorted(votes):
        counter = votes[(arch, kernel, key)]
        winner = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merged.setdefault(arch, {}).setdefault(kernel, {})[key] = json.loads(winner)
    return merged


def disagreements(votes: dict) -> dict:
    """The (arch, kernel, key) entries for which ranks recorded more than one config."""
    return {k: dict(v) for k, v in votes.items() if len(v) > 1}


def merge_records(paths) -> dict:
    """Merge per-rank recordings by majority vote, keyed by architecture.

    Ranks routinely disagree about the winner; that disagreement is the very
    variance a table removes, so the merge counts votes rather than letting the
    last file win. Ties break on the serialized config, so a given set of
    recordings always produces the same table.
    """
    return merge_votes(count_votes(paths))


def disagreement_report(paths) -> dict:
    """Count, per (arch, kernel, key), how many distinct winners ranks chose.

    Anything above one is timing variance the table is about to remove, and is
    worth seeing before trusting a recording.
    """
    return disagreements(count_votes(paths))


def write(arch: str, kernels: dict, path, source: str = "") -> None:
    """Write one architecture's table, with the provenance to date it."""
    payload = {
        "arch": arch,
        "triton": _triton_version(),
        "packages": _package_versions(),
        "source": source,
        "kernels": kernels,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
