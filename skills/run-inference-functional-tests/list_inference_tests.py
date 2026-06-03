#!/usr/bin/env python3
"""Enumerate inference functional-test cases that run in CI.

Parses every `tests/test_utils/recipes/h100/*inference*.yaml` recipe,
keeps every product row whose scope contains `mr` or `mr-github` and whose
environment is `dev` and platform is `dgx_h100`, and prints one TSV row per
test case:

    <test_case>\t<model>\t<training_script_path>\t<n_repeat>

Used by the run-inference-functional-tests skill to drive `cog submit` calls.
"""
import pathlib
import re
import sys

import yaml

RECIPES_DIR = pathlib.Path("tests/test_utils/recipes/h100")


def main() -> int:
    rows = []
    for recipe_path in sorted(RECIPES_DIR.glob("*inference*.yaml")):
        with recipe_path.open() as fh:
            data = yaml.safe_load(fh)
        spec = data.get("spec") or {}
        model = spec.get("model", "")
        script_block = spec.get("script") or ""
        m = re.search(r'TRAINING_SCRIPT_PATH=([^"\s]+)', script_block)
        if not m:
            print(f"# skip {recipe_path.name}: no TRAINING_SCRIPT_PATH", file=sys.stderr)
            continue
        training_script = m.group(1)
        default_n_repeat = spec.get("n_repeat", 1)
        for product in data.get("products") or []:
            for test_case in product.get("test_case", []):
                for sub in product.get("products") or []:
                    scopes = sub.get("scope") or []
                    envs = sub.get("environment") or []
                    plats = sub.get("platforms") or []
                    if (
                        any(s in ("mr", "mr-github") for s in scopes)
                        and "dev" in envs
                        and "dgx_h100" in plats
                    ):
                        rows.append((test_case, model, training_script, default_n_repeat))

    seen = set()
    for row in rows:
        key = (row[0], row[1])
        if key in seen:
            continue
        seen.add(key)
        print("\t".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
