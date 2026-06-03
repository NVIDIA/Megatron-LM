#!/usr/bin/env python3
"""Enumerate inference performance-test cases that run in CI.

Parses every `tests/test_utils/recipes/{h100,gb200}/*perf*.yaml` recipe
that drives `tests/performance_tests/shell_test_utils/run_perf_test.sh`
(i.e. the OpenAI-server-style inference perf harness), keeps every
product row whose scope contains `mr` or `mr-github`, environment is
`dev`, and platform is one we recognize, and prints one TSV row per
(test_case, platform):

    <family>/<test_case>\t<gpus>\t<platform>

`family` is the spec.model field of the recipe (e.g. `gpt`, `hybrid`),
which also corresponds to the subdirectory of
`tests/performance_tests/test_cases/`. `platform` matches the recipe's
`platforms` field (e.g. `dgx_h100`, `dgx_gb200`) so callers can route
the test to the right cluster.

Recipes that don't run `run_perf_test.sh` (e.g. `module_performance.yaml`,
which runs a one-off `tests.functional_tests.test_cases.common.<module>`
script) are skipped — the inference-perf skill machinery does not apply.
"""
import pathlib
import sys

import yaml

RECIPES_ROOT = pathlib.Path("tests/test_utils/recipes")
PLATFORM_DIRS = ("h100", "gb200")


def main() -> int:
    rows = []
    for platform_dir in PLATFORM_DIRS:
        recipes_dir = RECIPES_ROOT / platform_dir
        if not recipes_dir.is_dir():
            continue
        for recipe_path in sorted(recipes_dir.glob("*perf*.yaml")):
            with recipe_path.open() as fh:
                data = yaml.safe_load(fh)
            spec = data.get("spec") or {}
            script_block = spec.get("script") or ""
            if "run_perf_test.sh" not in script_block:
                print(
                    f"# skip {recipe_path}: not an inference-perf recipe "
                    f"(no run_perf_test.sh)",
                    file=sys.stderr,
                )
                continue
            family = spec.get("model", "")
            gpus = spec.get("gpus", 1)
            for product in data.get("products") or []:
                for test_case in product.get("test_case", []):
                    for sub in product.get("products") or []:
                        scopes = sub.get("scope") or []
                        envs = sub.get("environment") or []
                        plats = sub.get("platforms") or []
                        if not (
                            any(s in ("mr", "mr-github") for s in scopes)
                            and "dev" in envs
                        ):
                            continue
                        for plat in plats:
                            if plat in ("dgx_h100", "dgx_gb200"):
                                rows.append((f"{family}/{test_case}", gpus, plat))

    seen = set()
    for row in rows:
        key = (row[0], row[2])
        if key in seen:
            continue
        seen.add(key)
        print("\t".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
