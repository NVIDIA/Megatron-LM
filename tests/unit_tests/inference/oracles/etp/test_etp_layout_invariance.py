# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Metamorphic oracle: the inference MoE output does not depend on the parallel layout.

Contract (``etp_moe.py``): a token's MoE output is ``sum_k p_k FC2_e(act(FC1_e x))`` with the
unsharded expert weights, whatever the attention TP, expert TP (ETP), EP and token placement.

Each test builds one global problem and one pool of tokens, then runs the real inference MoE
layer (NVLS dispatcher over ``tp_ep``, FFN-sharded ``InferenceGroupedMLP``) once per layout that
fits the world, re-creating the process groups in between: ETP 1 (EP 2), ETP 2 with TP 1 (two
replicas share the experts), ETP 2 = TP 2, ETP 1 < TP 2, and on four ranks ETP 2 x EP 2,
TP 2 x ETP 2 x EP 2 and ETP 4. The pool is spread over each layout's replicas (as evenly as
possible, sequence-parallel shards for TP 2, padding to a multiple of TP).

- ``test_integer_experts_give_the_same_bits_on_every_layout``: integer experts and tokens make
  every rounding point exact, so every layout must return the fp64 reference rounded once to bf16
  for every token, hence the same bits as every other layout (ETP 2 vs ETP 1, and the rest).
- ``test_random_experts_match_the_reference_on_every_layout``: Gaussian experts; every layout
  within the harness bound of the fp64 reference (trusted: torch at the design's rounding points
  on that layout's ETP shards), and every ETP > 1 layout's max-abs and rms error over all tokens
  no larger than twice the ETP 1 layout's on the same tokens, plus the one-ulp floor.

Hardware: H100, 2 or 4 GPUs (NVLS); the layouts that need 4 ranks run when the world allows.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.etp import etp_moe as etp
from tests.unit_tests.test_utilities import Utils

pytestmark = ou.requires_sm(ou.HOPPER)

E, H, TOPK = 8, 128, 2
FFN = 512  # 128 columns per rank at ETP 4: DeepGEMM's grouped GEMM shapes in batch-invariant mode
M = 64
TOKENS = 61  # odd: every layout's split is uneven somewhere


def _layouts():
    if not torch.cuda.is_available() or Utils.world_size < 2:
        pytest.skip("needs at least 2 GPUs")
    fits = [layout for layout in etp.H100_LAYOUTS if Utils.world_size % layout.world == 0]
    assert any(layout.etp == 1 for layout in fits) and any(layout.etp == 2 for layout in fits)
    return fits


def _counts(layout: etp.Layout):
    reps, tp = layout.replicas, layout.tp
    reals = [TOKENS // reps + (j < TOKENS % reps) for j in range(reps)]
    return [(n, max(tp, -(-n // tp) * tp)) for n in reals]


def _run(layout: etp.Layout, problem: etp.Problem, tokens: etp.Tokens, backend: str, bi: bool):
    """Per-token outputs of the layout (every rank gets its tp_ep group's), and its rank info."""
    with etp.parallel_layout(layout) as env:
        etp.allocate_nvls(env, M, problem.topk, problem.hidden)
        layer = etp.build_layer(
            etp.make_config(problem, layout, backend, batch_invariant=bi), problem, env
        )
        batch = etp.rank_batch(tokens, _counts(layout), env)
        with etp.batch_invariant_mode(bi):
            out = etp.run_step(env, layer, problem, batch)
        by_token = etp.gather_by_token(env, out, batch, TOKENS, M)
        del layer
        return by_token, out, batch, env.rank


@pytest.mark.parametrize("bi", [False, True], ids=["nobi", "bi"])
@pytest.mark.parametrize("backend", ["vllm", "torch"])
def test_integer_experts_give_the_same_bits_on_every_layout(backend, bi):
    layouts = _layouts()
    if bi:
        etp.require_batch_invariant_backend(backend)
    problem = etp.integer_problem(E, H, FFN, TOPK, seed=101)
    tokens = etp.make_tokens(problem, TOKENS, seed=102, routing="random", ep=2)
    expected = etp.reference(problem, tokens.hidden, tokens.routing, tokens.probs)
    expected = expected.to(torch.bfloat16)
    for layout in layouts:
        by_token, *_ = _run(layout, problem, tokens, backend, bi)
        ou.assert_exact(by_token, expected, what=f"{layout.name} {backend} vs the exact reference")


@pytest.mark.parametrize("bi", [False, True], ids=["nobi", "bi"])
@pytest.mark.parametrize("activation", ["swiglu", "squared_relu"])
@pytest.mark.parametrize("backend", ["vllm", "torch"])
def test_random_experts_match_the_reference_on_every_layout(backend, activation, bi):
    layouts = _layouts()
    if bi:
        etp.require_batch_invariant_backend(backend)
    problem = etp.random_problem(E, H, FFN, TOPK, activation, seed=111)
    tokens = etp.make_tokens(problem, TOKENS, seed=112, routing="random", ep=2)
    ref = etp.reference(problem, tokens.hidden, tokens.routing, tokens.probs)
    errors = {}
    for layout in layouts:
        by_token, out, batch, rank = _run(layout, problem, tokens, backend, bi)
        etp.check_output(
            out,
            batch,
            tokens,
            problem,
            layout.etp,
            probs_at_activation=bi,
            what=f"{layout.name} rank {rank} {backend}",
        )
        errors[layout] = (by_token.double() - ref).abs()
    # ETP 2 against ETP 1 (EP 2) on the same tokens: the expert-TP split may cost at most a factor
    # of two in the max-abs and rms error over all tokens, plus the harness's one-ulp floor.
    control = errors[etp.Layout(1, 1, 2)]
    floor = float(ou.ulp(ref.abs().amax(), torch.bfloat16))
    for layout, err in errors.items():
        if layout.etp == 1:
            continue
        for metric, value, bound in (
            ("max-abs", err.max(), 2 * control.max() + floor),
            (
                "rms",
                err.square().mean().sqrt(),
                2 * control.square().mean().sqrt() + floor / TOKENS**0.5,
            ),
        ):
            assert value <= bound, (
                f"{layout.name}: {metric} error {float(value):.3e} over all tokens exceeds 2x the"
                f" ETP 1 layout's ({float(bound):.3e} with the floor)"
            )
