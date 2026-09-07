from __future__ import annotations

import ast
from pathlib import Path


ATTENTION_ROOT = (
    Path(__file__).resolve().parents[5]
    / "megatron/lite/model/deepseek_v4/vllm/primitive/attention"
)


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"{function_name} not found in {path}")


def test_attention_gathers_workspace_inside_request_loop() -> None:
    source = _function_source(
        ATTENTION_ROOT / "module.py", "_forward_training_attention"
    )
    assert "request_workspace = workspace.index_select" not in source
    assert source.index("for seq_idx in range") < source.index(
        "local_workspace = workspace.index_select"
    )
    assert "torch.unique" not in source
    assert "torch.searchsorted" not in source
    assert ".cpu()" not in source
    assert ".tolist()" not in source
    assert ".item()" not in source


def test_attention_skips_requests_outside_the_local_cp_rank() -> None:
    source = _function_source(
        ATTENTION_ROOT / "module.py", "_forward_training_attention"
    )
    guard = "if owned_start >= owned_end:"
    assert source.index("owned_end = min(local_end, seq_end)") < source.index(guard)
    assert source.index(guard) < source.index("first_group = max(")
    guard_body = source[source.index(guard) : source.index("first_group = max(")]
    assert "local_boundaries.append(local_boundaries[-1])" in guard_body
    assert "continue" in guard_body


def test_official_request_loops_consume_host_boundaries() -> None:
    for function_name in (
        "official_compact_compressed_visible",
        "official_indexer_topk",
    ):
        source = _function_source(ATTENTION_ROOT / "runtime.py", function_name)
        assert ".cpu()" not in source
        assert ".tolist()" not in source
        assert ".item()" not in source


def test_request_local_layout_capacity_is_host_owned() -> None:
    source = _function_source(
        ATTENTION_ROOT / "request_local_layout.py",
        "build_request_local_layout",
    )
    assert ".item()" not in source
    assert "total_capacity: int" in source


def test_indexer_uses_loaded_deterministic_topk_in_bi_mode() -> None:
    runtime = ATTENTION_ROOT / "runtime.py"
    wrapper = _function_source(runtime, "_top_k_per_row_prefill")
    indexer = _function_source(runtime, "official_indexer_topk")
    assert "envs.VLLM_BATCH_INVARIANT" in wrapper
    assert 'os.environ.get("DS4_BI_TOPK_LIB")' in wrapper
    assert "torch.ops.load_library(library)" in wrapper
    assert "torch.ops.ds4_bi.top_k_per_row_prefill" in wrapper
    assert "requires the loaded DS4 deterministic" in wrapper
    assert "_top_k_per_row_prefill(" in indexer
    assert "ops.top_k_per_row_prefill(" not in indexer
