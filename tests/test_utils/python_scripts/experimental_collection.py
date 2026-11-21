# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Report whether the production collection contains experimental tests."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

REPORT_DIR_ENV = "UNIT_TEST_EXPERIMENTAL_REPORT_DIR"
_COLLECTED_ITEMS = pytest.StashKey[tuple[pytest.Item, ...]]()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(session, items):
    """Retain original items across marker and deselection hooks."""
    if not os.environ.get(REPORT_DIR_ENV):
        yield
        return
    original_items = tuple(items)
    outcome = yield
    if outcome.excinfo is None:
        session.stash[_COLLECTED_ITEMS] = original_items


def pytest_collection_finish(session):
    """Write reports after every collection-modification wrapper has returned."""
    report_dir = os.environ.get(REPORT_DIR_ENV)
    original_items = session.stash.get(_COLLECTED_ITEMS, None)
    if not report_dir or original_items is None or session.testsfailed:
        return

    try:
        rank = int(os.environ["RANK"])
        report = {
            "rank": rank,
            "has_experimental": any(
                item.get_closest_marker("experimental") is not None for item in original_items
            ),
        }
        (Path(report_dir) / f"rank-{rank}.json").write_text(json.dumps(report))
    except (KeyError, OSError, ValueError) as error:
        # Reporting must never change the production test result. An absent or
        # incomplete report makes the runner keep the experimental launch.
        print(f"Experimental collection report unavailable: {error}", file=sys.stderr)


def can_skip_experimental(report_dir: Path, expected_ranks: int) -> bool:
    """Require an explicit negative report from every distributed rank."""
    if expected_ranks < 1:
        return False
    for rank in range(expected_ranks):
        try:
            report = json.loads((report_dir / f"rank-{rank}.json").read_text())
        except (OSError, ValueError):
            return False
        if (
            not isinstance(report, dict)
            or type(report.get("rank")) is not int
            or report["rank"] != rank
            or report.get("has_experimental") is not False
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("expected_ranks", type=int)
    args = parser.parse_args()
    return 0 if can_skip_experimental(args.report_dir, args.expected_ranks) else 1


if __name__ == "__main__":
    sys.exit(main())
