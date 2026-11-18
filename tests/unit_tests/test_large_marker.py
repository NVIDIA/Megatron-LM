# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""How ``tests/unit_tests/run_ci_test.sh`` selects the unit tests marked ``large``.

Every default launch deselects them. ``--large`` runs only them, each case in its own
process, from the buckets of ``tests/test_utils/recipes/h100/unit-tests-large.yaml``.
These checks run on the CPU.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from _pytest.mark.expression import Expression

ROOT = Path(__file__).resolve().parents[2]


def _marker_terms(tag, environment, platform, large):
    """The marker terms (``MARKER_ARG``) that the runner builds for one launch."""
    runner = (ROOT / "tests/unit_tests/run_ci_test.sh").read_text()
    block = runner[runner.index("\nMARKER=()\n") : runner.index("\nexport BUCKET\n")]
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input=block + '\nprintf "%s\\n" "$MARKER_ARG"\n',
        env={
            "PATH": os.environ["PATH"],
            "TAG": tag,
            "ENVIRONMENT": environment,
            "PLATFORM": platform,
            "LARGE": large,
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _selects(expression, *marks):
    """Whether pytest's ``-m`` evaluation selects a test with exactly these marks."""
    return Expression.compile(expression).evaluate(lambda name, **kwargs: name in marks)


@pytest.mark.parametrize(
    ("tag", "environment", "platform"), [("latest", "dev", "h100"), ("legacy", "lts", "gb200")]
)
def test_default_launches_deselect_large_tests(tag, environment, platform):
    """Both phases of every default launch (full, baseline, select and selected) put
    ``not experimental`` or ``experimental`` before the runner's terms."""
    terms = _marker_terms(tag, environment, platform, "false")
    gb200 = ("launch_on_gb200",) if platform == "gb200" else ()
    for phase, marks in (("not experimental", gb200), ("experimental", ("experimental", *gb200))):
        expression = f"{phase} and {terms}"
        assert _selects(expression, *marks), expression
        assert not _selects(expression, "large", *marks), expression


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_large_launches_select_only_large_tests(platform):
    """``--large`` uses the terms alone, and a large test that needs GB200 memory
    (``launch_on_gb200``) runs only on GB200."""
    expression = _marker_terms("latest", "dev", platform, "true")
    gb200 = ("launch_on_gb200",) if platform == "gb200" else ()
    assert _selects(expression, "large", *gb200)
    assert not _selects(expression, *gb200)
    assert not _selects(expression, "large", "experimental", *gb200)
    assert not _selects(expression, "large", "flaky_in_dev", *gb200)
    assert _selects(expression, "large", "launch_on_gb200") == (platform == "gb200")


def test_every_large_test_file_is_in_a_large_bucket():
    """``--large`` searches only the recipe's buckets: a file outside them would never run."""
    recipe = yaml.safe_load(
        (ROOT / "tests/test_utils/recipes/h100/unit-tests-large.yaml").read_text()
    )
    targets = [
        case.removesuffix("/**/*.py")
        for product in recipe["products"]
        for case in product["test_case"]
    ]
    unowned = []
    for path in sorted((ROOT / "tests/unit_tests").rglob("test_*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if re.search(r"mark\.large\b", path.read_text(errors="replace")) and not any(
            relative == target or relative.startswith(target + "/") for target in targets
        ):
            unowned.append(relative)
    assert not unowned, f"Add a bucket to unit-tests-large.yaml for {unowned}"
