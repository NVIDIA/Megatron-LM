# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay through the Triton autotuner policy and architecture tables."""

import inspect
import sys
from weakref import WeakKeyDictionary, WeakSet

import pytest
import torch

from megatron.core.tuning import interception, selection, table
from megatron.core.tuning.policy import DEFAULT_CONFIG_INVARIANT, AutotunePolicy
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.runtime.autotuner import Autotuner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and Triton")


@pytest.fixture
def isolated_policy(monkeypatch):
    """Restore the prior process-wide adapter, policy, and diagnostics after each replay."""
    monkeypatch.setattr(Autotuner, "run", inspect.unwrap(Autotuner.run))
    for name, value in {
        "_installed": False,
        "_policy": None,
        "_tables": {},
        "_selected_configs": WeakKeyDictionary(),
        "_scopes": WeakKeyDictionary(),
        "_logged_singletons": WeakSet(),
        "_choice_log": {},
        "_unverified": {},
        "_verified": {},
        "_tune_records": {},
        "_record_rank": None,
        "_enumerated": set(),
    }.items():
        monkeypatch.setattr(interception, name, value)
    monkeypatch.setattr(selection, "_untuned_kernels_warned", set())
    monkeypatch.setattr(interception.atexit, "register", lambda *_: None)


@pytest.mark.parametrize("use_table", [False, True], ids=["min_cost", "tuned_table"])
@pytest.mark.parametrize("cold_cache", [False, True], ids=["warm_cache", "cold_cache"])
def test_pinned_reduction_replays(isolated_policy, monkeypatch, tmp_path, use_table, cold_cache):
    """Both selection paths replay bit-exactly with cold and warm config caches."""

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

    assert interception.active_policy().table_path == (str(tmp_path),)
    seeded()
    for columns in (257, 1021):
        values = torch.randn(256, columns, device="cuda", dtype=torch.float32)

        def run(values):
            output = torch.empty(values.shape[0], device=values.device, dtype=values.dtype)
            row_sum.cache.clear()
            if cold_cache:
                interception._selected_configs.clear()
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


def _loaded_autotuners(qualified_names):
    """Autotuners, among already imported modules, whose kernels carry these names."""
    found = {}
    for module in list(sys.modules.values()):
        for value in list(vars(module).values()) if module is not None else ():
            while value is not None and not isinstance(value, Autotuner):
                value = getattr(value, "fn", None) if type(value).__name__ == "Heuristics" else None
            if value is None:
                continue
            name = f"{selection.kernel_module(value)}.{selection.kernel_name(value)}"
            if name in qualified_names:
                found[name] = value
    return found


def _te_permutation_cases():
    from megatron.core.transformer.moe import moe_utils

    if moe_utils.fused_permute_with_probs is None or moe_utils.fused_unpermute is None:
        return []
    cases = []
    for hidden in (1000, 4096):
        tokens, experts, topk = 256, 16, 4
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        top = torch.rand(tokens, experts, device="cuda").topk(topk, dim=1).indices
        routing_map = torch.zeros(tokens, experts, dtype=torch.bool, device="cuda")
        routing_map.scatter_(1, top, True)
        probs = torch.zeros(tokens, experts, device="cuda")
        probs.scatter_(1, top, torch.rand(tokens, topk, device="cuda"))
        grad_out = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

        def permute_round_trip(x=x, routing_map=routing_map, probs=probs, grad_out=grad_out):
            # Mirrors the alltoall dispatcher: probs travel with the tokens, and the
            # unpermute gets no merging probs.
            leaf, p = x.detach().requires_grad_(), probs.detach().requires_grad_()
            out, permuted_probs, row_id_map, _, _ = moe_utils.permute(
                leaf, routing_map, probs=p, num_out_tokens=leaf.shape[0] * topk, fused=True
            )
            back = moe_utils.unpermute(
                out * permuted_probs.unsqueeze(-1).to(out.dtype),
                row_id_map,
                restore_shape=leaf.shape,
                routing_map=routing_map,
                fused=True,
            )
            return (out, permuted_probs, back) + torch.autograd.grad(
                back, [leaf, p], grad_out.clone()
            )

        cases.append(permute_round_trip)
    return cases


def test_config_invariant_kernels_are_invariant(monkeypatch):
    """Every candidate of an exempted kernel must produce bit-identical results.

    The default ``config_invariant`` list leaves these kernels on Triton's timed choice
    under pinning. That is only safe while their outputs do not depend on the launch
    configuration, so force each candidate in turn and compare.
    """
    torch.manual_seed(0)
    cases = _te_permutation_cases()
    if not cases:
        pytest.skip("Transformer Engine's Triton permutation kernels are not available")

    def run_all():
        return [[tensor.clone() for tensor in case()] for case in cases]

    def same(first, second):
        return all(torch.equal(a, b) for x, y in zip(first, second) for a, b in zip(x, y))

    run_all()  # import and compile everything the cases reach
    # Without a stable baseline, any difference below would be noise, not the config.
    assert same(run_all(), run_all()), "an invariance case does not replay bit-exactly"
    tuners = _loaded_autotuners(set(DEFAULT_CONFIG_INVARIANT))
    assert tuners, "the permutation round trip reached no config-invariant kernel"
    varying = {}
    for name, tuner in sorted(tuners.items()):
        candidates = list(tuner.configs)
        reference = None
        for config in candidates:
            monkeypatch.setattr(tuner, "configs", [config])
            tuner.cache.clear()
            outputs = run_all()
            if reference is None:
                reference = outputs
            elif not same(outputs, reference):
                varying[name] = f"{candidates[0]} vs {config}"
                break
        monkeypatch.setattr(tuner, "configs", candidates)
    assert not varying, "config-invariant kernels whose output depends on the config:\n" + (
        "\n".join(f"  {name}: {detail}" for name, detail in sorted(varying.items()))
    )
