# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check golden-value JSON files for NaN and infinity values."""

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOT_ACCEPTED_VALUES = [
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
]


def _find_non_finite_values(value: Any, location: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _find_non_finite_values(child, f"{location}[{key!r}]")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_non_finite_values(child, f"{location}[{index}]")
    elif str(value).strip().lower() in NOT_ACCEPTED_VALUES:
        yield location, value


def _format_failures(failures: list[tuple[str, Any]], limit: int = 20) -> str:
    lines = [f"  {location} = {value!r}" for location, value in failures[:limit]]
    if len(failures) > limit:
        lines.append(f"  ... and {len(failures) - limit} more")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if any golden-value JSON file contains NaN or infinity."
    )
    parser.add_argument("files", nargs="+", type=Path, help="Golden-value JSON files to check.")
    return parser.parse_args()


def main() -> int:
    """Check the requested golden-value files and return a process exit code."""
    failed = False
    files = _parse_args().files

    for golden_value_file in files:
        try:
            with golden_value_file.open() as file:
                golden_values = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Could not read %s: %s", golden_value_file, error)
            failed = True
            continue

        failures = list(_find_non_finite_values(golden_values))
        if failures:
            logger.error(
                "Found non-finite values in %s:\n%s", golden_value_file, _format_failures(failures)
            )
            failed = True

    if not failed:
        logger.info("Checked %d golden-value file(s); all values are finite.", len(files))

    return int(failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
