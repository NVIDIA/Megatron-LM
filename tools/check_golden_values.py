# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check golden-value JSON files.

Two checks run on every file:

* No NaN or infinity values.
* For test cases whose results are compared with ``DeterministicTest`` (every
  pretraining case that does not opt out via ``NON_DETERMINSTIC_RESULTS: 1``,
  ``NVTE_ALLOW_NONDETERMINISTIC_ALGO: 1`` or ``SKIP_PYTEST: 1`` in its
  ``model_config.yaml``), every metric must be marked ``"value_precision": "full"``.
  ``read_tb_logs_as_list`` writes that marker for every golden value it produces;
  a metric without it (or marked ``rounded_5_decimal_places``) is a legacy golden
  that the comparison pipeline rounds to five decimals, which silently reduces
  the bit-exact check to a 5e-6 tolerance. Regenerate such files from a CI run
  (``tests/test_utils/python_scripts/download_golden_values.py``) rather than
  hand-editing or copying legacy values.

Invoked by the ``linting`` job in ``.github/workflows/cicd-main.yml`` on the
golden-value files a PR adds or modifies. Only the standard library is used, so
it runs without the test dependencies installed.
"""

import argparse
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirror of ``ValuePrecision`` in tests/functional_tests/python_test_utils/common.py
# (not imported: that module needs pydantic and tensorboard).
FULL_PRECISION = "full"
LEGACY_PRECISION = "rounded_5_decimal_places"

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

# ``run_ci_test.sh`` passes ``--allow-nondeterministic-algo`` (approximate-only
# comparison) to pytest iff NON_DETERMINSTIC_RESULTS or
# NVTE_ALLOW_NONDETERMINISTIC_ALGO is 1, and skips the comparison entirely when
# SKIP_PYTEST is 1. Everything else is compared with ``DeterministicTest``.
# Matched on the raw YAML text so this script does not depend on PyYAML.
_NONDETERMINISTIC_ENV_VAR = re.compile(
    r"^\s*(NON_DETERMINSTIC_RESULTS|NVTE_ALLOW_NONDETERMINISTIC_ALGO|SKIP_PYTEST)"
    r"\s*:\s*['\"]?1['\"]?\s*(#.*)?$",
    re.MULTILINE,
)


def _find_non_finite_values(value: Any, location: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _find_non_finite_values(child, f"{location}[{key!r}]")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_non_finite_values(child, f"{location}[{index}]")
    elif str(value).strip().lower() in NOT_ACCEPTED_VALUES:
        yield location, value


def compares_deterministically(model_config_text: str) -> bool:
    """Return True if the functional-test pipeline applies ``DeterministicTest`` to this case."""
    return _NONDETERMINISTIC_ENV_VAR.search(model_config_text) is None


def _is_metric_block(block: Any) -> bool:
    return isinstance(block, dict) and isinstance(block.get("values"), dict)


def find_legacy_precision_metrics(golden_values: Any) -> list[str]:
    """Return the metrics of ``golden_values`` that are not marked ``value_precision: full``.

    Only the explicit marker is consulted; the spelling of the values is not
    evidence either way (a full-precision ``0.5`` is legitimately short). Files
    with a different layout (for example inference generations) contain no metric
    blocks and yield an empty list.
    """
    if not isinstance(golden_values, dict):
        return []
    return [
        metric
        for metric, block in golden_values.items()
        if _is_metric_block(block) and block.get("value_precision") != FULL_PRECISION
    ]


def check_precision(golden_value_file: Path, golden_values: Any) -> list[str]:
    """Return the legacy-precision metrics of a deterministically compared case."""
    model_config = golden_value_file.parent / "model_config.yaml"
    if not model_config.is_file():
        logger.info("No model_config.yaml next to %s; skipping precision check.", golden_value_file)
        return []
    if not compares_deterministically(model_config.read_text()):
        return []
    return find_legacy_precision_metrics(golden_values)


def _format_failures(failures: list[tuple[str, Any]], limit: int = 20) -> str:
    lines = [f"  {location} = {value!r}" for location, value in failures[:limit]]
    if len(failures) > limit:
        lines.append(f"  ... and {len(failures) - limit} more")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if any golden-value JSON file contains NaN or infinity, or if a "
            "deterministically compared test case carries golden values without the "
            f"'value_precision: {FULL_PRECISION}' marker."
        )
    )
    parser.add_argument("files", nargs="+", type=Path, help="Golden-value JSON files to check.")
    parser.add_argument(
        "--allow-legacy-precision",
        action="store_true",
        help=(
            "Warn instead of failing when a deterministically compared test case has "
            f"metrics without 'value_precision: {FULL_PRECISION}' (legacy goldens are "
            f"compared at {LEGACY_PRECISION.split('_')[1]} decimals)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the requested golden-value files and return a process exit code."""
    args = _parse_args(argv)
    failed = False

    for golden_value_file in args.files:
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

        legacy = check_precision(golden_value_file, golden_values)
        if legacy:
            log = logger.warning if args.allow_legacy_precision else logger.error
            log(
                "%s: %s not marked 'value_precision: %s', but this test case is compared with "
                "DeterministicTest, which needs full-precision golden values to be bit-exact "
                "(legacy goldens are rounded to five decimals before comparison). Regenerate "
                "the file from a CI run with "
                "tests/test_utils/python_scripts/download_golden_values.py instead of editing "
                "or copying legacy values.%s",
                golden_value_file,
                ", ".join(repr(metric) for metric in legacy),
                FULL_PRECISION,
                (
                    ""
                    if args.allow_legacy_precision
                    else " Pass --allow-legacy-precision to override."
                ),
            )
            failed = failed or not args.allow_legacy_precision

    if not failed:
        logger.info(
            "Checked %d golden-value file(s); all values are finite and deterministic "
            "cases carry full-precision golden values.",
            len(args.files),
        )

    return int(failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
