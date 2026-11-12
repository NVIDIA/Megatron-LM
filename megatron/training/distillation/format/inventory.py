# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A scratch, disk-backed mapping for producer startup metadata."""

import json
import sqlite3
import tempfile
import zlib
from collections.abc import Iterator, MutableMapping
from typing import Any


class ChunkInventory(MutableMapping[str, dict[str, Any]]):
    """Cache remote indexes once without retaining every Python block dictionary.

    This database is reconstructible scratch data, not a publication artifact.
    Only one chunk is decoded at a time; SQLite's pager is capped at 8 MiB.
    """

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="offline-kd-inventory-")
        self._db = sqlite3.connect(f"{self._directory.name}/chunks.sqlite")
        self._db.execute("PRAGMA cache_size=-8192")
        self._db.execute("PRAGMA journal_mode=OFF")
        self._db.execute("PRAGMA synchronous=OFF")
        self._db.execute("CREATE TABLE chunks (name TEXT PRIMARY KEY, payload BLOB NOT NULL)")

    def __getitem__(self, key: str) -> dict[str, Any]:
        row = self._db.execute("SELECT payload FROM chunks WHERE name=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(zlib.decompress(row[0]))

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        payload = zlib.compress(json.dumps(value, separators=(",", ":")).encode(), level=1)
        self._db.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?)", (key, payload))
        self._db.commit()

    def __delitem__(self, key: str) -> None:
        cursor = self._db.execute("DELETE FROM chunks WHERE name=?", (key,))
        self._db.commit()
        if not cursor.rowcount:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (row[0] for row in self._db.execute("SELECT name FROM chunks ORDER BY name"))

    def __len__(self) -> int:
        return self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def __contains__(self, key: object) -> bool:
        return self._db.execute("SELECT 1 FROM chunks WHERE name=?", (key,)).fetchone() is not None

    def close(self) -> None:
        """Release scratch storage when the producer is reset."""
        self._db.close()
        self._directory.cleanup()
