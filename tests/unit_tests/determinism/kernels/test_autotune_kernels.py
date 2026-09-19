# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay through the Triton autotuner policy and architecture tables."""

import inspect
import os

import pytest
import torch

from megatron.core.tuning import interception, selection, table
from megatron.core.tuning.policy import AutotunePolicy
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.runtime.autotuner import Autotuner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and Triton")


@pytest.fixture
def isolated_policy(monkeypatch):
    """Restore the prior process-wide adapter, policy, and diagnostics after each replay."""
    monkeypatch.setattr(Autotuner, "run", inspect.unwrap(Autotuner.run))
    monkeypatch.setattr(interception, "_installed", False)
    monkeypatch.setattr(interception, "_policy", None)
    monkeypatch.setattr(interception, "_explicit_policy", False)
    for name in ("_tables", "_choice_log", "_tune_records"):
        monkeypatch.setattr(interception, name, {})
    monkeypatch.setattr(interception, "_enumerated", set())
    monkeypatch.setattr(selection, "_untuned_kernels_warned", set())
    monkeypatch.setattr(interception.atexit, "register", lambda *_: None)
    monkeypatch.delenv("MCORE_AUTOTUNE_TABLE_PATH", raising=False)
    for name in os.environ:
        if name.startswith("TRITON_AUTOTUNE_BLOCK"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize("use_table", [False, True], ids=["min_cost", "tuned_table"])
def test_pinned_reduction_replays(isolated_policy, monkeypatch, tmp_path, use_table):
    """Both selection paths replay without timing and retain the live candidate list."""

    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE": size}) for size in (128, 256)], key=["columns"]
    )
    @triton.jit
    def row_sum(x, out, columns: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        accumulator = tl.full((BLOCK_SIZE,), 0, tl.float32)
        for start in range(triton.cdiv(columns, BLOCK_SIZE)):
            column = start * BLOCK_SIZE + offsets
            accumulator += tl.load(x + row * columns + column, mask=column < columns, other=0)
        tl.store(out + row, tl.sum(accumulator, axis=0))

    def forbid_benchmark(*args, **kwargs):
        pytest.fail("pinned execution benchmarked a config")

    monkeypatch.setattr(row_sum, "_bench", forbid_benchmark)
    candidates = row_sum.configs
    chosen = candidates[1 if use_table else 0]
    arch = selection.arch_tag()
    if use_table:
        table.write(
            arch, {"row_sum": {"*": selection.config_data(chosen)}}, tmp_path / f"{arch}.json"
        )
    interception.install(
        AutotunePolicy(
            mode="pinned",
            modules=(__name__,),
            table_path=(tmp_path,),
            on_miss="error" if use_table else "min_cost",
        )
    )

    seeded()
    for columns in (257, 1021):
        values = torch.randn(256, columns, device="cuda", dtype=torch.float32)

        def run(values):
            output = torch.empty(values.shape[0], device=values.device, dtype=values.dtype)
            # Exercise cold selection on every replay as well as changing shapes.
            row_sum.cache.clear()
            row_sum[(values.shape[0],)](values, output, values.shape[1])
            assert row_sum.best_config is chosen
            assert row_sum.configs is candidates
            return output

        outputs, _ = assert_replays_bit_exact(
            run, (values,), replays=4, backward=False, what="pinned Triton row reduction"
        )
        torch.testing.assert_close(outputs["out"], values.sum(dim=1), rtol=1e-4, atol=1e-4)

    choices = interception.choice_log()
    assert len(choices) == 2
    assert set(choices.values()) == {selection.config_signature(chosen) + ";pinned"}
