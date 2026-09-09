# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve stable functional-test model names to their source-package directories."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FUNCTIONAL_TEST_ROOT = Path("tests/functional_tests")

MODEL_DIRECTORIES = {
    "bert": "core/models/bert",
    "gpt": "core/models/gpt",
    "mixtral": "core/models/gpt",
    "hybrid": "core/models/hybrid",
    "nemotron": "core/models/hybrid",
    "mimo": "core/models/mimo",
    "multimodal-llava": "core/models/multimodal",
    "t5": "core/models/T5",
    "moe": "core/transformer/moe",
    "gpt-nemo": "",
}
COMMON_DIRECTORIES = {
    "ckpt_converter": "ckpt_converter",
    "moe_perf": "core/transformer/moe/moe_perf",
}


def functional_test_case_dir(model: str, test_case: str, repo_root: Path | None = None) -> Path:
    """Return a case directory relative to the repository root.

    Model and case names remain the identifiers used by recipes and CI artifacts.
    Prefer the package-based layout, while supporting checkouts that still have
    the old layout (for example, when bisecting older commits).

    Args:
        model: Logical model family from the recipe or CI job.
        test_case: Scenario name from the recipe or CI job.
        repo_root: Checkout to inspect; defaults to this module's checkout.
    """
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    legacy_path = FUNCTIONAL_TEST_ROOT / "test_cases" / model / test_case
    if model == "common" and test_case in COMMON_DIRECTORIES:
        path = FUNCTIONAL_TEST_ROOT / COMMON_DIRECTORIES[test_case]
    elif model in MODEL_DIRECTORIES:
        path = FUNCTIONAL_TEST_ROOT / MODEL_DIRECTORIES[model] / test_case
    else:
        return legacy_path

    if not (repo_root / path).is_dir() and (repo_root / legacy_path).is_dir():
        return legacy_path
    return path
