# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tuned Triton configs, stored as data rather than as source.

A tuned table lets a pinned run use the *fastest* config instead of the cheapest
one while staying a pure function of its inputs. Tables live as JSON so adding
an architecture is a file drop, not a source edit and a rebuild:

    MCORE_AUTOTUNE_RECORD=/tmp/rec  torchrun ... pretrain.py ...
    python -m megatron.core.tuning merge /tmp/rec/*.json -o ~/.mcore/tuning/sm103.json
    MCORE_AUTOTUNE_TABLE_PATH=~/.mcore/tuning  torchrun ... pretrain.py ...

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
import os
import warnings
from pathlib import Path

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
        rather than rebuilt, so fields the table does not carry (``pre_hook``,
        ``num_ctas``, ``maxnreg``) survive, and a stale entry that no longer
        names a real candidate degrades to a miss instead of an invalid launch.
        """
        entries = self.kernels.get(kernel)
        if not entries:
            return None
        wanted = entries.get(key) or entries.get("*")
        if not wanted:
            return None
        for config in candidates:
            if (
                config.kwargs == wanted.get("kwargs")
                and getattr(config, "num_warps", None) == wanted.get("num_warps")
                and getattr(config, "num_stages", None) == wanted.get("num_stages")
            ):
                return config
        return None


def _search_dirs(extra) -> list:
    dirs = [Path(p) for p in (extra or ())]
    env = os.environ.get("MCORE_AUTOTUNE_TABLE_PATH", "")
    dirs += [Path(p) for p in env.split(os.pathsep) if p]
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
            warnings.warn(f"Ignoring unreadable tuned table {path}: {exc}")
            continue
        recorded = data.get("triton", "")
        current = _triton_version()
        if recorded and current and recorded != current:
            warnings.warn(
                f"Tuned table {path} was recorded against triton {recorded} but this "
                f"process has {current}; entries that no longer match a live config "
                "will fall back to the deterministic default."
            )
        return TunedTable(arch, data.get("kernels", {}), data)
    return TunedTable(arch, {})


def merge_records(paths) -> dict:
    """Merge per-rank recordings by majority vote, keyed by architecture.

    Ranks routinely disagree about the winner; that disagreement is the very
    variance a table removes, so the merge counts votes rather than letting the
    last file win. Ties break on the serialized config, so a given set of
    recordings always produces the same table.
    """
    votes: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    )
    for path in sorted(paths):
        with open(path, encoding="utf-8") as handle:
            for arch, kernels in json.load(handle).items():
                for kernel, entries in kernels.items():
                    for key, config in entries.items():
                        votes[arch][kernel][key][json.dumps(config, sort_keys=True)] += 1

    merged: dict = {}
    for arch in sorted(votes):
        for kernel in sorted(votes[arch]):
            for key in sorted(votes[arch][kernel]):
                counter = votes[arch][kernel][key]
                winner = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                merged.setdefault(arch, {}).setdefault(kernel, {})[key] = json.loads(winner)
    return merged


def disagreement_report(paths) -> dict:
    """Count, per (arch, kernel, key), how many distinct winners ranks chose.

    Anything above one is timing variance the table is about to remove, and is
    worth seeing before trusting a recording.
    """
    votes: dict = collections.defaultdict(collections.Counter)
    for path in sorted(paths):
        with open(path, encoding="utf-8") as handle:
            for arch, kernels in json.load(handle).items():
                for kernel, entries in kernels.items():
                    for key, config in entries.items():
                        votes[(arch, kernel, key)][json.dumps(config, sort_keys=True)] += 1
    return {k: dict(v) for k, v in votes.items() if len(v) > 1}


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
