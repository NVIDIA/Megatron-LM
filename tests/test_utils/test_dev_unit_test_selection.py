# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Guard bucket ownership while main's selective runner includes dev additions."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dev_bucket_finder", ROOT / "tests/unit_tests/find_test_cases.py"
)
finder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finder)


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_ci_parser_preserves_every_declared_unit_bucket(platform):
    recipe_path = ROOT / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    expected = Counter(bucket for entry in recipe["products"] for bucket in entry["test_case"])
    parser_spec = importlib.util.spec_from_file_location(
        "unit_bucket_recipe_parser", ROOT / "tests/test_utils/python_scripts/recipe_parser.py"
    )
    recipe_parser = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(recipe_parser)

    workloads = recipe_parser.load_and_flatten(str(recipe_path))
    actual = Counter(
        workload.spec["test_case"]
        for workload in workloads
        if workload.spec["environment"] == "dev"
        and workload.spec["tag"] == "latest"
        and workload.spec["scope"] == "unit-tests"
    )
    assert actual == expected


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_every_unit_file_has_exactly_one_platform_bucket(monkeypatch, capsys, platform):
    monkeypatch.chdir(ROOT)
    recipe = yaml.safe_load(
        (ROOT / f"tests/test_utils/recipes/{platform}/unit-tests.yaml").read_text()
    )
    buckets = [bucket for entry in recipe["products"] for bucket in entry["test_case"]]
    monkeypatch.setattr(finder, "get_test_cases", lambda _path: buckets)
    owners = Counter()
    owner_buckets = {}
    for bucket in buckets:
        monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, platform])
        finder.main()
        ignored = {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines()}
        owners.update(
            path
            for path in finder.expand_pattern(bucket)
            if Path(path).name.startswith("test_") and path not in ignored
        )
        for path in finder.expand_pattern(bucket):
            if Path(path).name.startswith("test_") and path not in ignored:
                owner_buckets.setdefault(path, []).append(bucket)
    expected = {
        str(path)
        for root in (Path("tests/unit_tests"), Path("examples/multimodal_dev/tests"))
        for path in root.rglob("test_*.py")
    }
    expected -= finder.PLATFORM_EXCLUSIONS.get(platform, set())
    marker = finder.PLATFORM_MARKERS.get(platform)
    if marker:
        expected = {path for path in expected if finder.file_has_marker(path, marker)}
    assert set(owners) == expected
    assert all(count == 1 for count in owners.values())
    scheduler_dir = "tests/unit_tests/transformer/moe/"
    replay = scheduler_dir + "test_scheduler_determinism.py"
    for filename in (
        "test_scheduler_bf16_layer_parity.py",
        "test_moonep_dispatch_capacity.py",
        "test_moonep_pr7450_training.py",
        "test_scheduler_determinism.py",
    ):
        path = scheduler_dir + filename
        if platform == "h100":
            assert path not in owners, "Four-rank GB200 recipes must not run on eight H100s"
        else:
            expected_bucket = replay if path == replay else "tests/unit_tests/**/*.py"
            assert owner_buckets[path] == [expected_bucket]
            assert recipe["spec"]["nodes"] == 1
            assert recipe["spec"]["gpus"] == 4
    # The GB200 opt-in marker must not remove portable scheduler coverage on H100.
    for filename in (
        "test_echo_moe_scheduler.py",
        "test_eplb_moe_scheduler.py",
        "test_replica_nccl_transport.py",
        "test_home_expert_exchange.py",
        "test_home_only_expert_dispatch.py",
    ):
        assert owners[scheduler_dir + filename] == 1
    if platform == "h100":
        # The upstream GDN suite moved to GB200; dev GDN and DSv4 still run here.
        assert "tests/unit_tests/ssm/test_gated_delta_net.py" not in owners
        for path in (
            "tests/unit_tests/ssm/gated_delta_net/test_gdn.py",
            "tests/unit_tests/ssm/gated_delta_net/test_gdn_fusion.py",
            "tests/unit_tests/ssm/gated_delta_net/test_kda.py",
            "tests/unit_tests/ssm/test_gated_delta_net2.py",
            "tests/unit_tests/ssm/test_gdn_gated_output_norm_fusion.py",
            "tests/unit_tests/ssm/test_gdn_gated_output_norm_distributed.py",
            "tests/unit_tests/fusions/test_fused_mhc_kernels.py",
            "tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp_cuda_graph.py",
            "tests/unit_tests/optimizer/test_muon_sync_qkv_cpu.py",
        ):
            assert owners[path] == 1
