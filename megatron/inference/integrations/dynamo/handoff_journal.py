# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable decode handoff ownership and an external-supervisor termination hook.

The database must live on persistent storage with working SQLite file locking
and fsync. Use rollback journaling, not WAL, on a shared filesystem.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass


@dataclass(frozen=True)
class HandoffRelease:
    """Source identity retained until its release has been acknowledged."""

    coordinator_addr: str
    coordinator_instance_id: str
    request_id: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> HandoffRelease:
        """Reject incomplete metadata before a decode import can be submitted."""
        address = metadata.get("coordinator_addr")
        instance = metadata.get("coordinator_instance_id")
        request_id = metadata.get("request_id")
        if not isinstance(address, str) or not address:
            raise ValueError("Handoff release requires a coordinator address")
        if not isinstance(instance, str) or not instance:
            raise ValueError("Handoff release requires a coordinator instance ID")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise ValueError("Handoff release requires an integer request ID")
        return cls(address, instance, request_id)


class HandoffJournal:
    """Persist ownership before submitting transfers; never expire unsafe records."""

    def __init__(self, path: str) -> None:
        self.path = path
        with closing(self._connect()) as db, db:
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("Handoff journal requires SQLite DELETE journal mode")
            db.execute(
                "CREATE TABLE IF NOT EXISTS owners "
                "(owner TEXT PRIMARY KEY, terminated INTEGER NOT NULL DEFAULT 0)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS handoffs "
                "(receipt TEXT PRIMARY KEY, owner TEXT NOT NULL, address TEXT NOT NULL, "
                "instance TEXT NOT NULL, request_id INTEGER NOT NULL, "
                "source_safe INTEGER NOT NULL DEFAULT 0)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def start_owner(self, owner: str) -> None:
        """Claim a deployment attempt exactly once, including across parent restarts."""
        if not owner:
            raise ValueError("A unique handoff owner is required for each deployment attempt")
        with closing(self._connect()) as db, db:
            try:
                db.execute("INSERT INTO owners(owner) VALUES (?)", (owner,))
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "Handoff owner already used; launch with a new owner ID"
                ) from error

    def record(self, owner: str, release: HandoffRelease) -> str:
        """Commit cleanup ownership before the caller submits its decode request."""
        receipt = uuid.uuid4().hex
        with closing(self._connect()) as db, db:
            inserted = db.execute(
                "INSERT INTO handoffs(receipt, owner, address, instance, request_id) "
                "SELECT ?, owner, ?, ?, ? FROM owners WHERE owner = ? AND terminated = 0",
                (
                    receipt,
                    release.coordinator_addr,
                    release.coordinator_instance_id,
                    release.request_id,
                    owner,
                ),
            ).rowcount
            if not inserted:
                raise RuntimeError("Cannot submit handoff for an unknown or terminated owner")
        return receipt

    def mark_source_safe(self, receipt: str) -> None:
        """Record Core's import completion or storage-safe abort acknowledgement."""
        with closing(self._connect()) as db, db:
            db.execute("UPDATE handoffs SET source_safe = 1 WHERE receipt = ?", (receipt,))

    def confirm_terminated(self, owner: str) -> None:
        """Attest that the old parent AND every decode rank have exited.

        Only the deployment supervisor can supply this proof. Discovery removal,
        a missed heartbeat, rank-zero death, or a replacement becoming healthy is
        insufficient. The supervisor must also prevent the old attempt restarting.
        """
        with closing(self._connect()) as db, db:
            if not db.execute(
                "UPDATE owners SET terminated = 1 WHERE owner = ?", (owner,)
            ).rowcount:
                raise ValueError(f"Unknown handoff owner: {owner}")

    def releasable(self) -> list[tuple[str, HandoffRelease]]:
        """Return only completed imports or transfers from a confirmed dead attempt."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT h.receipt, h.address, h.instance, h.request_id FROM handoffs h "
                "JOIN owners o ON h.owner = o.owner "
                "WHERE h.source_safe = 1 OR o.terminated = 1"
            ).fetchall()
        return [
            (receipt, HandoffRelease(address, instance, request_id))
            for receipt, address, instance, request_id in rows
        ]

    def acknowledge_release(self, receipt: str) -> None:
        """Forget an entry only after the source has accepted its fenced release."""
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM handoffs WHERE receipt = ?", (receipt,))


def main() -> None:
    """Expose the termination attestation to Slurm, Ray, and Kubernetes controllers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    parser.add_argument(
        "--confirm-terminated-owner",
        required=True,
        help="Attest that this attempt's parent and ALL decode ranks exited and cannot restart.",
    )
    args = parser.parse_args()
    journal = HandoffJournal(args.journal)
    journal.confirm_terminated(args.confirm_terminated_owner)
    sys.stdout.write(
        "Termination recorded. A decode worker using this journal will retry source releases.\n"
    )


if __name__ == "__main__":
    main()
