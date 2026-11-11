# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lowest-index exact-zero tie selection used by the deterministic DSA top-k."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_topk_zero_ties

_SENTINEL = torch.iinfo(torch.int32).max
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _reference(scores, row_limits, quotas, topk_k):
    out = torch.full((scores.size(0), topk_k), _SENTINEL, dtype=torch.int32)
    for row in range(scores.size(0)):
        limit = int(row_limits[row])
        zeros = [col for col in range(limit) if float(scores[row, col]) == 0.0]
        picked = zeros[: int(quotas[row])]
        out[row, : len(picked)] = torch.tensor(picked, dtype=torch.int32)
    return out


def _random_case(seed, rows=24, sk=2500, topk_k=64, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    choices = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("nan"), float("-inf"), float("inf"), 0.5], dtype=dtype
    )
    weights = torch.tensor([6.0, 6.0, 2.0, 2.0, 0.5, 0.5, 0.2, 1.0])
    picks = torch.multinomial(weights, rows * sk, replacement=True, generator=generator)
    scores = choices[picks].view(rows, sk).contiguous()
    # Sparse zeros on some rows force scans across many blocks.
    sparse = torch.full((sk,), 1.0, dtype=dtype)
    sparse[[3, 1100, 2047, 2048, sk - 1]] = torch.tensor([-0.0, 0.0, -0.0, 0.0, -0.0], dtype=dtype)
    scores[0] = sparse
    row_limits = torch.randint(0, sk + 1, (rows,), generator=generator, dtype=torch.int32)
    row_limits[0] = sk
    row_limits[1] = 0
    row_limits[2] = sk
    # Beyond each row limit the buffer holds garbage (zeros, NaN) that must not be read.
    in_row = torch.arange(sk).view(1, sk) < row_limits.view(rows, 1)
    zero_counts = ((scores == 0) & in_row).sum(dim=1, dtype=torch.int32)
    quotas = torch.minimum(
        zero_counts, torch.randint(0, topk_k + 1, (rows,), generator=generator, dtype=torch.int32)
    )
    quotas[0] = min(5, topk_k)
    quotas[2] = 0
    return scores, row_limits, quotas


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torch_zero_tie_selection_matches_reference(seed, dtype, monkeypatch):
    scores, row_limits, quotas = _random_case(seed, dtype=dtype)
    # Several row chunks.
    monkeypatch.setattr(dsa_topk_zero_ties, "_TORCH_MAX_BYTES", 16 * 2500 * 5)

    result = dsa_topk_zero_ties.select_lowest_index_zero_ties(scores, row_limits, quotas, 64)

    torch.testing.assert_close(result, _reference(scores, row_limits, quotas, 64), rtol=0, atol=0)


def test_zero_tie_selection_treats_both_zero_signs_as_ties_and_skips_nan():
    scores = torch.tensor(
        [
            [1.0, -0.0, float("nan"), 0.0, -0.0, 0.0],
            [-0.0, -0.0, -0.0, -0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    row_limits = torch.tensor([6, 4, 6], dtype=torch.int32)
    quotas = torch.tensor([3, 4, 0], dtype=torch.int32)

    result = dsa_topk_zero_ties.select_lowest_index_zero_ties(scores, row_limits, quotas, 4)

    expected = torch.tensor(
        [[1, 3, 4, _SENTINEL], [0, 1, 2, 3], [_SENTINEL] * 4], dtype=torch.int32
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_zero_tie_selection_rejects_mismatched_metadata():
    scores = torch.zeros((2, 4))
    with pytest.raises(RuntimeError, match="int32"):
        dsa_topk_zero_ties.select_lowest_index_zero_ties(
            scores, torch.tensor([4, 4]), torch.tensor([1, 1], dtype=torch.int32), 2
        )
    with pytest.raises(RuntimeError, match="shape"):
        dsa_topk_zero_ties.select_lowest_index_zero_ties(
            scores, torch.tensor([4], dtype=torch.int32), torch.tensor([1], dtype=torch.int32), 2
        )


@pytest.mark.skipif(not dsa_topk_zero_ties.HAVE_TRITON, reason="Triton is not installed")
def test_triton_zero_tie_kernel_matches_reference_in_the_interpreter(tmp_path):
    """Run the Triton kernel through ``TRITON_INTERPRET=1`` on CPU tensors."""
    script = tmp_path / "zero_ties_interpreter.py"
    script.write_text(textwrap.dedent("""
            try:
                import triton.runtime.interpreter  # noqa: F401
            except ImportError:
                print("TRITON_INTERPRETER_UNAVAILABLE")
                raise SystemExit(0)
            import torch
            from megatron.core.transformer.experimental_attention_variant import (
                dsa_topk_zero_ties as zt,
            )
            from tests.unit_tests.transformer.experimental_attention_variant import (
                test_dsa_topk_zero_ties as cases,
            )

            for seed in range(3):
                for dtype in (torch.float32, torch.bfloat16):
                    scores, limits, quotas = cases._random_case(seed, dtype=dtype)
                    got = zt._select_lowest_index_zero_ties_triton(scores, limits, quotas, 64)
                    want = cases._reference(scores, limits, quotas, 64)
                    assert torch.equal(got, want), (seed, dtype)
            print("TRITON_INTERPRETER_OK")
            """))
    python_path = os.pathsep.join(filter(None, (str(_REPO_ROOT), os.environ.get("PYTHONPATH"))))
    environment = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=python_path)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "TRITON_INTERPRETER_UNAVAILABLE" in completed.stdout:
        pytest.skip("this Triton build has no interpreter")
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "TRITON_INTERPRETER_OK" in completed.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seed", range(3))
def test_cuda_zero_tie_selection_matches_reference(seed):
    scores, row_limits, quotas = _random_case(seed)
    expected = _reference(scores, row_limits, quotas, 64)

    result = dsa_topk_zero_ties.select_lowest_index_zero_ties(
        scores.cuda(), row_limits.cuda(), quotas.cuda(), 64
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
