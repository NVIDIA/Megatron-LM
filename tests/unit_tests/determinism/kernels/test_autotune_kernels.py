# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay through the Triton autotuner policy and architecture tables."""

import sys

import pytest
import torch

from megatron.core.tuning import cutile, interception, selection, table
from megatron.core.tuning.policy import DEFAULT_CONFIG_INVARIANT, AutotunePolicy
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.tuning.isolation import reset_tuning_state

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.runtime.autotuner import Autotuner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and Triton")


@pytest.fixture
def isolated_policy(monkeypatch):
    """Restore the prior process-wide adapter, policy, and diagnostics after each replay."""
    reset_tuning_state(monkeypatch)


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


def _cutile_or_skip(monkeypatch):
    """cudnn-frontend's linear-attention ops and cuTile launch module with the hooks and launch cache
    reset for the test. Skips without the hooks, and on a CUDA runtime older than the cuTile engine
    accepts (the same check the engine makes at plan selection, where it would fail instead)."""
    la_ops = pytest.importorskip("cudnn.linear_attention.ops")
    engine = pytest.importorskip("cudnn.linear_attention.cutile.engine")
    common = cutile._common()
    if common is None:
        pytest.skip("cudnn-frontend without the cuTile autotune hooks (NVIDIA/cudnn-frontend#1735)")
    runtime = pytest.importorskip("cuda.bindings.runtime")
    err, cudart = runtime.cudaRuntimeGetVersion()
    if int(err) != 0 or cudart < engine.CUTILE_MIN_CUDART:
        pytest.skip(
            f"the cuTile engine needs CUDA runtime {engine.CUTILE_MIN_CUDART}, found {cudart}"
        )
    for name in cutile.HOOKS:
        monkeypatch.setattr(common, name, None)
    common.launch_hint_cache.clear()
    return la_ops, common


def test_cutile_gdn_pinned_pick_replays(isolated_policy, monkeypatch, tmp_path):
    """cudnn-frontend's cuTile GDN engine under the policy: a record run captures the timed
    launch picks, the merged table pins them, and the pinned op replays bit-exactly. The launch
    cache stays warm from the record run: installing the pinned policy must drop the timed picks,
    or the pinned run would silently keep them. The cuTile plan is forced through ``plan_name``;
    on SM100 the default engine is FROST, which has no launch autotune."""
    la_ops, common = _cutile_or_skip(monkeypatch)
    from megatron.core._rank_utils import safe_get_rank

    seeded()
    T, H, K, V = 2048, 16, 128, 128
    q, k, v = (
        torch.randn(T, H, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for d in (K, K, V)
    )
    g = (-torch.nn.functional.softplus(torch.randn(T, H, device="cuda"))).requires_grad_(True)
    beta = torch.rand(T, H, device="cuda").requires_grad_(True)
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)

    def gdn(q, k, v, g, beta):
        return la_ops.gated_delta_net(
            q, k, v, g, beta, cu_seqlens, output_final_state=True, plan_name="gdn_cutile"
        )

    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    o, state = gdn(q, k, v, g, beta)
    (o.float().sum() + state.float().sum()).backward()
    torch.cuda.synchronize()
    recorded = {
        kernel
        for kernels in interception._tune_records.values()
        for kernel in kernels
        if kernel.startswith("cudnn.linear_attention.cutile.")
    }
    assert recorded, "the record run captured no cuTile launch pick"
    interception._dump_records()
    merged = table.merge_records(
        [interception._record_file(str(tmp_path / "rec"), safe_get_rank())]
    )
    (arch,) = merged
    table.write(arch, merged[arch], tmp_path / "tables" / f"{arch}.json")

    assert common.launch_hint_cache, "the record run left no warm launch picks to invalidate"
    interception.install(
        AutotunePolicy(mode="pinned", table_path=(str(tmp_path / "tables"),), on_miss="error")
    )
    assert (
        common.launch_hint_cache == {}
    ), "installing a new policy must drop the cached launch picks"
    assert_replays_bit_exact(
        gdn, (q, k, v, g, beta), replays=3, what="cuTile GDN pinned from table"
    )
    pinned = {
        entry
        for entry, value in interception.choice_log().items()
        if "cudnn.linear_attention.cutile." in entry and value.endswith(";pinned")
    }
    assert {entry.split("|")[1] for entry in pinned} == recorded


def test_cutile_packaged_table_pins_gdn_and_kda(isolated_policy, monkeypatch):
    """The packaged table pins every cuTile GDN and KDA launch on this architecture with no user table,
    in-kernel Q/K normalization included: ``on_miss="error"`` would raise on the first miss."""
    la_ops, common = _cutile_or_skip(monkeypatch)
    if selection.arch_tag() != "sm103":
        pytest.skip("packaged cuTile entries exist for sm103 only")
    interception.install(AutotunePolicy(mode="pinned", on_miss="error"))
    seeded()
    T, H, K, V = 2048, 16, 128, 128
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)

    def leaves(gate_shape, beta_shape):
        q, k, v = (
            torch.randn(T, H, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for d in (K, K, V)
        )
        g = (-torch.nn.functional.softplus(torch.randn(*gate_shape, device="cuda"))).requires_grad_(
            True
        )
        beta = torch.rand(*beta_shape, device="cuda").requires_grad_(True)
        return q, k, v, g, beta

    for name, op, shapes in (
        ("gdn_cutile", la_ops.gated_delta_net, ((T, H), (T, H))),
        ("kda_cutile", la_ops.kimi_delta_attention, ((T, H, K), (T, H))),
    ):
        q, k, v, g, beta = leaves(*shapes)
        # In-kernel Q/K normalization launches every kernel of the plain path plus the two l2norm ones.
        o, state = op(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            plan_name=name,
        )
        (o.float().sum() + state.float().sum()).backward()
        torch.cuda.synchronize()
    pinned = {
        entry.split("|")[1]
        for entry, value in interception.choice_log().items()
        if "cudnn.linear_attention.cutile." in entry and value.endswith(";pinned")
    }
    packaged = {
        kernel
        for kernel in table.load("sm103", ()).kernels
        if kernel.startswith("cudnn.linear_attention.cutile.")
    }
    # Every packaged cuTile entry was used, and nothing was pinned that the table does not carry.
    assert pinned == packaged, pinned ^ packaged
