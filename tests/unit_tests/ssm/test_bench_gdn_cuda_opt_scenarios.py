import ast
from pathlib import Path


BENCH = Path(__file__).with_name("bench_gdn_cuda_opt.py")


def _literal_assignment(name):
    tree = ast.parse(BENCH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def test_optimized_scenarios_route_through_mcore_wrapper():
    scenarios = _literal_assignment("SCENARIOS")
    optimized = [
        "wy",
        "dv_dhu",
        "dhu",
        "dqkwg",
        "fused",
        "separate",
        "dv_dhu_dqkwg",
        "all_four",
        "all_four_dv_dhu",
    ]

    for key in optimized:
        env = scenarios[key][1]
        assert env["MCORE_GDN_USE_OPT_WRAPPER"] == "1", key
        assert env["MCORE_GDN_OPT_BACKEND"] == "cuda", key
        assert not any(flag.startswith("FLA_CUTE_") for flag in env), key


def test_benchmark_does_not_require_patched_fla_sources():
    text = BENCH.read_text()

    assert "patched flash-linear-attention" not in text
    assert "FLA_DISPATCH_SOURCE" not in text
