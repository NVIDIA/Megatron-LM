# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import pathlib
import re
import subprocess

import pytest
import yaml
from click.testing import CliRunner

from tests.test_utils.python_scripts import generate_jet_trigger_job, recipe_parser


@pytest.fixture(scope="module")
def workloads():
    return recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", platform="dgx_gb300"
    )


@pytest.mark.parametrize("scope", ["nightly", "L2"])
@pytest.mark.parametrize("cadence", [None, "nightly"])
def test_gb300_platform_selects_every_gb200_nightly_case(scope, cadence):
    cases = {}
    for platform in ("dgx_gb200", "dgx_gb300"):
        specs = [
            workload["spec"]
            for workload in recipe_parser.load_workloads(
                container_tag="validation",
                scope=scope,
                cadence=cadence,
                environment="dev",
                platform=platform,
            )
            if workload["type"] == "basic"
        ]
        cases[platform] = {(spec["model"], spec["test_case"]): spec for spec in specs}
        assert len(cases[platform]) == len(specs), "Duplicate workloads would overwrite GitLab jobs"

    assert cases["dgx_gb200"]
    assert cases["dgx_gb300"].keys() == cases["dgx_gb200"].keys()
    for key, spec in cases["dgx_gb300"].items():
        assert spec["platforms"] == "dgx_gb300"
        source = cases["dgx_gb200"][key]
        for field in ("nodes", "gpus", "segment", "build", "n_repeat", "scope", "cadence"):
            assert spec.get(field) == source.get(field), (key, field)
        for argument in ("TRAINING_SCRIPT_PATH", "TRAINING_PARAMS_PATH"):
            pattern = rf'"{argument}=([^"]+)"'
            assert re.search(pattern, spec["script"]).group(1) == re.search(
                pattern, source["script"]
            ).group(1)


@pytest.mark.parametrize(("cadence", "expected_count"), [(None, 1), ("nightly", 0)])
def test_gb300_mirror_uses_the_callers_cadence_filter(monkeypatch, cadence, expected_count):
    test_case = "gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"
    load_and_flatten = recipe_parser.load_and_flatten

    def load_with_explicit_pr_cadence(config_path):
        workloads = load_and_flatten(config_path)
        for workload in workloads:
            if workload.spec["test_case"] == test_case:
                workload.spec["cadence"] = ["pr"]
        return workloads

    monkeypatch.setattr(recipe_parser, "load_and_flatten", load_with_explicit_pr_cadence)
    for platform in ("dgx_gb200", "dgx_gb300"):
        workloads = recipe_parser.load_workloads(
            container_tag="validation",
            scope="nightly",
            cadence=cadence,
            environment="dev",
            platform=platform,
            test_case=test_case,
        )
        specs = [workload["spec"] for workload in workloads if workload["type"] == "basic"]
        assert len(specs) == expected_count
        for spec in specs:
            assert spec["cadence"] == ["pr"]


@pytest.mark.parametrize(
    ("scope", "environment"),
    [
        ("mr", "dev"),
        ("L0-smoke", "dev"),
        ("unit-tests", "dev"),
        ("weekly", "dev"),
        ("nightly", "lts"),
    ],
)
def test_gb300_platform_only_selects_dev_nightly(scope, environment):
    assert not recipe_parser.load_workloads(
        container_tag="validation", scope=scope, environment=environment, platform="dgx_gb300"
    )


def test_gb300_manifests_mount_required_data_only_on_jhb(workloads):
    for workload in workloads:
        if workload["type"] == "build":
            assert not workload.get("launchers")
        else:
            assert workload["launchers"] == {
                "name:dgxgb300_oci-jhb": {
                    "mounts": {
                        "/lustre/fsw/coreai_dlalgo_mcore": (
                            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/"
                        ),
                        "/mnt/artifacts": (
                            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci"
                        ),
                    }
                }
            }


def test_gb300_manifests_are_valid_for_jet(workloads):
    jetclient = pytest.importorskip("jetclient")
    for workload in workloads:
        jetclient.JETWorkloadManifest(**workload)


def test_gb300_recipes_reference_existing_goldens(workloads):
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for workload in workloads:
        if workload["type"] != "basic":
            continue
        spec = workload["spec"]
        reference = re.search(r'"GOLDEN_VALUES_PATH=([^"]+)"', spec["script"])
        assert reference is not None
        path = repo_root / reference.group(1).format(**spec)
        config = yaml.safe_load((path.parent / "model_config.yaml").read_text())
        assert path.is_file() or config.get("ENV_VARS", {}).get("SKIP_PYTEST") in (1, "1"), path
        if spec["test_case"] in {
            "deepseek_proxy_mfsdp_v1_ep2",
            "gpt3_7b_tp1_pp4_memory_speed",
            "gpt3_7b_tp4_pp1_memory_speed",
            "gpt3_mcore_te_tp2_pp1_te_a2a_ovlp_8experts_etp1_ep4",
            "gpt3_mcore_te_tp2_pp2_resume_torch_dist_defer_embedding_wgrad_compute",
            "gpt3_moe_mcore_te_ep8_resume_torch_dist_dist_optimizer",
            "gpt3_moe_mcore_te_tp4_ep2_etp2_pp2_scoped_cudagraph",
            "nemotron3_5_lightning_nightly_tp1_pp1_cp1_ep8_dgx_gb200",
        }:
            assert path.name == "golden_values_dev_dgx_gb300.json"
        else:
            assert path.name == "golden_values_dev_dgx_gb200.json"
        actual = re.search(r'ACTUAL_VALUES_PATH="?([^"\s]+)', spec["script"])
        assert actual is not None
        assert actual.group(1) == "{assets_dir}/golden_values_{environment}_{platforms}.json"


def test_gb300_workload_allocations_fit_within_24_hours(workloads):
    for workload in workloads:
        if workload["type"] == "basic":
            assert 0 < workload["spec"]["time_limit"] <= 24 * 60 * 60


def test_gb200_recipes_keep_cluster_default_mounts():
    workloads = recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", platform="dgx_gb200"
    )
    assert workloads
    assert all("launchers" not in workload for workload in workloads)


@pytest.mark.parametrize(
    "test_case", [None, "gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"]
)
def test_platformless_nightly_lookups_keep_source_workloads(test_case):
    workloads = recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", test_case=test_case
    )
    specs = [workload["spec"] for workload in workloads if workload["type"] == "basic"]
    assert specs
    assert all(spec["platforms"] != "dgx_gb300" for spec in specs)
    if test_case is not None:
        assert len(specs) == 1
        assert specs[0]["platforms"] == "dgx_gb200"


def generate_pipeline(tmp_path, platform, cluster, options=()):
    output_path = tmp_path / "pipeline.yaml"
    result = CliRunner().invoke(
        generate_jet_trigger_job.main,
        [
            "--scope",
            "nightly",
            "--environment",
            "dev",
            "--time-limit",
            "1800",
            "--test-cases",
            "all",
            "--platform",
            platform,
            "--cluster",
            cluster,
            "--output-path",
            str(output_path),
            "--container-image",
            "utility",
            "--container-tag",
            "validation",
            "--dependent-job",
            "functional:configure",
            "--slurm-account",
            "mcore",
            "--no-enable-warmup",
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    return yaml.safe_load(output_path.read_text())


def test_gb300_generated_jobs_have_24_hour_timeout_and_preserve_failure_status(tmp_path, workloads):
    pipeline = generate_pipeline(
        tmp_path, "dgx_gb300", "dgxgb300_oci-jhb", ["--job-timeout", "24 hours"]
    )
    specs = {
        workload["spec"]["test_case"]: workload["spec"]
        for workload in workloads
        if workload["type"] == "basic"
    }
    jobs = {
        name: job for name, job in pipeline.items() if isinstance(job, dict) and "script" in job
    }
    assert jobs.keys() == specs.keys()
    for name, job in jobs.items():
        assert job["timeout"] == "24 hours"
        assert job["allow_failure"] == (
            specs[name].get("allow_failure", False) or specs[name]["model"] == "gpt-nemo"
        )
        assert not any(tag.startswith("cluster/") for tag in job["tags"])
        assert job["needs"] == [{"pipeline": "$PARENT_PIPELINE_ID", "job": "functional:configure"}]


def test_gb200_generated_jobs_keep_default_timeout_and_failure_behavior(tmp_path):
    pipeline = generate_pipeline(tmp_path, "dgx_gb200", "dgxgb200_oci-hsg")
    workloads = recipe_parser.load_workloads(
        container_tag="validation", scope="nightly", environment="dev", platform="dgx_gb200"
    )
    for workload in workloads:
        if workload["type"] != "basic":
            continue
        spec = workload["spec"]
        job = pipeline[spec["test_case"]]
        assert job["timeout"] == "7 days"
        assert job["allow_failure"] == (
            spec.get("allow_failure", False) or spec["model"] == "gpt-nemo"
        )
        assert "cluster/oci-hsg" in job["tags"]


@pytest.mark.parametrize("timeout", [None, "24 hours"])
def test_empty_generated_pipeline_respects_timeout_option(monkeypatch, tmp_path, timeout):
    monkeypatch.setattr(recipe_parser, "load_workloads", lambda **kwargs: [])
    options = ["--job-timeout", timeout] if timeout else []
    pipeline = generate_pipeline(tmp_path, "dgx_gb300", "dgxgb300_oci-jhb", options)
    job = pipeline["empty-pipeline-placeholder-job"]
    assert job["timeout"] == (timeout or "7 days")
    assert job.get("allow_failure", False) is False


@pytest.mark.parametrize(
    ("scope", "cluster", "enabled"),
    [
        ("nightly", "dgxgb300_oci-jhb", True),
        ("mr", "dgxgb300_oci-jhb", False),
        ("weekly", "dgxgb300_oci-jhb", False),
        ("nightly", "", False),
    ],
)
def test_gitlab_only_configures_gb300_nightly_jobs(scope, cluster, enabled):
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    config = yaml.safe_load((repo_root / ".gitlab/stages/04.functional-tests.yml").read_text())
    script = next(
        script
        for script in config["functional:configure"]["script"]
        if "--platform dgx_gb300" in script
    )
    result = subprocess.run(
        ["bash", "-c", 'python() { printf "%s\\n" "$@"; }\n' + script],
        env={"PATH": os.environ["PATH"], "FUNCTIONAL_TEST_SCOPE": scope, "CLUSTER_GB300": cluster},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    arguments = result.stdout.splitlines()
    if enabled:
        assert arguments[arguments.index("--job-timeout") + 1] == "24 hours"
        assert "--allow-failure" not in arguments
        assert arguments[arguments.index("--cluster") + 1] == cluster
    else:
        assert not arguments


def test_gitlab_gb300_nightly_bridge_does_not_block_parent():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pipeline = yaml.safe_load((repo_root / ".gitlab-ci.yml").read_text())
    assert pipeline["variables"]["CLUSTER_GB300"]["value"] == "dgxgb300_oci-jhb"
    config = yaml.safe_load((repo_root / ".gitlab/stages/04.functional-tests.yml").read_text())
    bridge = config["functional:run_dev_dgx_gb300"]
    assert bridge["allow_failure"] is True
    assert bridge["trigger"]["strategy"] is None
    enabled_rules = [rule["if"] for rule in bridge["rules"] if rule.get("when") == "on_success"]
    assert len(enabled_rules) == 1
    assert '$FUNCTIONAL_TEST_SCOPE == "nightly"' in enabled_rules[0]
    assert '$CLUSTER_GB300 != ""' in enabled_rules[0]
