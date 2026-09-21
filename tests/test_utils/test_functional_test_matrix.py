# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.test_utils.python_scripts import generate_functional_test_matrix, recipe_parser


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "Test fixture")
    return _git(repo, "rev-parse", "HEAD")


def _case(repo: Path, name: str, content: str = "MODEL_ARGS: {}\n") -> Path:
    directory = repo / "tests/functional_tests/test_cases/gpt" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_config.yaml").write_text(content)
    return directory


def _product(name: str, scope: str, **dimensions) -> dict:
    return {
        "test_case": [name],
        "products": [
            {"scope": [scope], "environment": ["dev"], "platforms": ["dgx_h100"], **dimensions}
        ],
    }


def _recipe(repo: Path, products: list, name: str = "recipes/h100/gpt.yaml", **spec) -> Path:
    path = repo / "tests/test_utils" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "type": "basic",
                "spec": {
                    "name": "{test_case}_{environment}_{platforms}",
                    "model": "gpt",
                    "build": "mcore-pyt-{environment}",
                    "script": "bash tests/functional_tests/shell_test_utils/run_ci_test.sh",
                    **spec,
                },
                "products": products,
            },
            sort_keys=False,
        )
    )
    return path


def _matrix(repo: Path, base_ref=None, scope="L0", cadence="pr") -> list:
    return generate_functional_test_matrix.generate_matrix(
        scope=scope, platform="dgx_h100", cadence=cadence, base_ref=base_ref, repo_root=repo
    )


def _entry(name: str, scope: str = "L0", cadence: str = "pr") -> dict:
    return {"model": "gpt", "test_case": name, "scope": scope, "cadence": cadence}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Functional matrix test")
    _git(tmp_path, "config", "user.email", "matrix@example.test")
    scripts = tmp_path / "tests/test_utils/python_scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(recipe_parser, "BASE_PATH", scripts)
    return tmp_path


def test_pr_adds_changed_cases_across_all_commits_from_merge_base(repo: Path):
    products = [
        _product("baseline", "L0"),
        _product("full", "L1"),
        _product("nightly", "L2", cadence=["nightly"]),
        _product("untouched", "L3", cadence=["weekly"]),
        _product("main_only", "L2", cadence=["nightly"]),
        _product("wrong_cadence", "L0", cadence=["nightly"]),
    ]
    for product in products:
        _case(repo, product["test_case"][0])
    _recipe(repo, products)
    _commit(repo)
    _git(repo, "checkout", "-b", "feature")

    _case(repo, "full", "MODEL_ARGS: {num_layers: 2}\n")
    _commit(repo)
    (_case(repo, "nightly") / "golden_values_dev_dgx_h100.json").write_text("{}\n")
    _case(repo, "added")
    _recipe(repo, products + [_product("added", "L3", cadence=["weekly"])])
    _commit(repo)
    (repo / "README.md").write_text("An unrelated final commit\n")
    _commit(repo)

    # Advancing main must not make its changes look like changes made by the PR.
    _git(repo, "checkout", "main")
    _case(repo, "main_only", "MODEL_ARGS: {num_layers: 4}\n")
    _commit(repo)
    _git(repo, "checkout", "feature")

    matrix = _matrix(repo, "main")
    assert sorted(matrix, key=lambda row: row["test_case"]) == [
        _entry("added", "L3", ""),
        _entry("baseline"),
        _entry("full", "L1", ""),
        _entry("nightly", "L2", ""),
    ]
    # The runtime must resolve each emitted row, including tests whose normal
    # cadence excludes PRs, to exactly one executable workload.
    for entry in matrix:
        workloads = recipe_parser.load_workloads(
            container_tag="latest", environment="dev", platform="dgx_h100", **entry
        )
        workloads = [workload for workload in workloads if workload.type != "build"]
        assert len(workloads) == 1
        assert workloads[0].spec["test_case"] == entry["test_case"]


def test_product_edit_adds_only_affected_case(repo: Path):
    products = [_product("edited", "L2"), _product("untouched", "L2")]
    for name in ("edited", "untouched"):
        _case(repo, name)
    _recipe(repo, products)
    base = _commit(repo)

    products[0]["products"][0]["time_limit"] = [3600]
    _recipe(repo, products)
    _commit(repo)

    assert _matrix(repo, base) == [_entry("edited", "L2", "")]


def test_shared_recipe_edit_adds_all_affected_cases(repo: Path):
    products = [_product("nightly", "L2"), _product("weekly", "L3")]
    for name in ("nightly", "weekly", "other_recipe"):
        _case(repo, name)
    _recipe(repo, products)
    _recipe(repo, [_product("other_recipe", "L2")], "recipes/h100/other.yaml")
    base = _commit(repo)

    _recipe(repo, products, script="bash updated_runner.sh")
    _commit(repo)

    assert sorted(_matrix(repo, base), key=lambda row: row["test_case"]) == [
        _entry("nightly", "L2", ""),
        _entry("weekly", "L3", ""),
    ]


def test_recipe_comment_does_not_add_unchanged_workloads(repo: Path):
    _case(repo, "nightly")
    recipe = _recipe(repo, [_product("nightly", "L2")])
    base = _commit(repo)
    recipe.write_text("# Updated documentation\n" + recipe.read_text())
    _commit(repo)

    assert _matrix(repo, base) == []


def test_deleted_cases_are_omitted_and_renamed_cases_use_current_names(repo: Path):
    names = ("deleted", "old_name", "surviving", "deleted_config")
    products = [_product(name, "L2") for name in names]
    for name in names:
        directory = _case(repo, name)
        (directory / "golden_values_dev_dgx_h100.json").write_text("{}\n")
    _recipe(repo, products)
    base = _commit(repo)

    root = repo / "tests/functional_tests/test_cases/gpt"
    shutil.rmtree(root / "deleted")
    (root / "old_name").rename(root / "new_name")
    (root / "surviving/golden_values_dev_dgx_h100.json").unlink()
    (root / "deleted_config/model_config.yaml").unlink()
    _recipe(repo, [_product(name, "L2") for name in ("new_name", "surviving", "deleted_config")])
    _commit(repo)

    assert sorted(_matrix(repo, base), key=lambda row: row["test_case"]) == [
        _entry("new_name", "L2", ""),
        _entry("surviving", "L2", ""),
    ]


def test_changed_cases_keep_platform_environment_and_functional_scope_filters(repo: Path):
    products = [
        _product("eligible", "L2"),
        _product("broken", "L2-broken"),
        _product("gitlab", "mr"),
        _product("gitlab_slim", "mr-slim"),
        _product("smoke", "L0-smoke"),
        _product("unit_scope", "unit-tests"),
        _product("lts", "L2", environment=["lts"]),
        _product("gb200", "L2", platforms=["dgx_gb200"]),
    ]
    _recipe(repo, products)
    _recipe(
        repo, [_product("unit_bucket", "L2")], "recipes/h100/unit-tests.yaml", model="unit-tests"
    )
    for product in products:
        _case(repo, product["test_case"][0])
    unit_file = repo / "tests/unit_tests/test_example.py"
    unit_file.parent.mkdir(parents=True)
    unit_file.write_text("def test_example(): pass\n")
    base = _commit(repo)

    for product in products:
        _case(repo, product["test_case"][0], "MODEL_ARGS: {num_layers: 2}\n")
    unit_file.write_text("def test_example(): assert True\n")
    _recipe(
        repo,
        [_product("unit_bucket", "L2", nodes=[2])],
        "recipes/h100/unit-tests.yaml",
        model="unit-tests",
    )
    _commit(repo)

    assert _matrix(repo, base) == [_entry("eligible", "L2", "")]


@pytest.mark.parametrize(("platform", "gpu_capacity"), [("dgx_h100", 8), ("dgx_gb200", 4)])
def test_extra_cases_fit_runner_capacity_without_changing_baseline(
    repo: Path, platform: str, gpu_capacity: int
):
    products = [
        # Capacity checks apply only to additions; the existing suite is preserved.
        _product("baseline", "L0", nodes=[2], gpus=[gpu_capacity + 1]),
        _product("eligible", "L2", nodes=[1], gpus=[gpu_capacity]),
        _product("multi_node", "L2", nodes=[2], gpus=[gpu_capacity]),
        _product("too_many_gpus", "L2", nodes=[1], gpus=[gpu_capacity + 1]),
    ]
    for product in products:
        product["products"][0]["platforms"] = [platform]
        _case(repo, product["test_case"][0])
    _recipe(repo, products)
    base = _commit(repo)

    for product in products:
        _case(repo, product["test_case"][0], "MODEL_ARGS: {num_layers: 2}\n")
    _commit(repo)

    matrix = generate_functional_test_matrix.generate_matrix(
        scope="L0", platform=platform, cadence="pr", base_ref=base, repo_root=repo
    )
    assert matrix == [_entry("baseline"), _entry("eligible", "L2", "")]


@pytest.mark.parametrize(
    ("legacy_scope", "resolved_scope"),
    [("mr-github-slim", "L0"), ("mr-github", "L1"), ("nightly", "L2"), ("weekly", "L3")],
)
def test_changed_cases_support_legacy_scope_aliases(repo: Path, legacy_scope, resolved_scope):
    _case(repo, "changed")
    _recipe(repo, [_product("changed", legacy_scope)])
    base = _commit(repo)
    _case(repo, "changed", "MODEL_ARGS: {num_layers: 2}\n")
    _commit(repo)

    assert _matrix(repo, base) == [
        _entry("changed", resolved_scope, "pr" if resolved_scope == "L0" else "")
    ]


@pytest.mark.parametrize("cadence", ["pr", None])
def test_changed_baseline_case_runs_once_with_baseline_filters(repo: Path, cadence):
    _case(repo, "baseline")
    _recipe(repo, [_product("baseline", "L0"), _product("baseline", "L2")])
    base = _commit(repo)
    _case(repo, "baseline", "MODEL_ARGS: {num_layers: 2}\n")
    _commit(repo)

    assert _matrix(repo, base, cadence=cadence) == [_entry("baseline", cadence=cadence or "")]


@pytest.mark.parametrize(("scope", "cadence"), [("L0", "pr"), ("L1", None)])
def test_changed_cases_are_added_with_or_without_full_suite_label(repo: Path, scope, cadence):
    _recipe(repo, [_product("baseline", scope), _product("changed", "L2", cadence=["nightly"])])
    _case(repo, "baseline")
    _case(repo, "changed")
    base = _commit(repo)
    _case(repo, "changed", "MODEL_ARGS: {num_layers: 2}\n")
    _commit(repo)

    assert _matrix(repo, base, scope=scope, cadence=cadence) == [
        _entry("baseline", scope, cadence or ""),
        _entry("changed", "L2", ""),
    ]


@pytest.mark.parametrize(
    ("scope", "cadence", "expected"),
    [
        ("L0", "pr", [_entry("baseline")]),
        ("L0", None, [_entry("baseline", cadence=""), _entry("nightly_only", cadence="")]),
        ("L1", "pr", [_entry("full", "L1")]),
    ],
)
def test_without_pr_base_ref_keeps_existing_selection(repo: Path, scope, cadence, expected):
    _recipe(
        repo,
        [
            _product("baseline", "L0"),
            _product("nightly_only", "L0", cadence=["nightly"]),
            _product("full", "L1"),
        ],
    )
    for name in ("baseline", "nightly_only", "full"):
        _case(repo, name)
    _commit(repo)

    assert _matrix(repo, scope=scope, cadence=cadence) == expected
