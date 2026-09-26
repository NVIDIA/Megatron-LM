# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""A3 part 2: the local token count in a REAL tensor-parallel domain, and its effect on the
auxiliary-loss scaling that reaches the router gradient.

Part 1 (``test_token_count_preservation.py``) proves the helper's return contract on a raw
NCCL world group. This file closes the gap that matters for training:

  * the helper is called with ``Router.tp_cp_group``, so the domain must be a real TP group;
  * its second return value pre-scales the aux loss in
    ``Router.attach_and_log_load_balancing_loss``, so a wrong value changes the gradient the
    router receives.

Run with a real TP group:

    torchrun --standalone --nproc_per_node=2 -m pytest -q \
        tests/unit_tests/transformer/moe/test_local_count_router_gradient.py

Scope note: this validates the SCALING TERM (``local_valid_tokens * tp_cp_size``) that the
router applies to the aux loss, and the ordering of the router gradient it produces. It is
not a full single-process loss-equivalence proof, which would require reproducing
``finalize_model_grads`` normalization; that is reported as NOT_RUN.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.internal

H = 128
FFN = 256
NUM_EXPERTS = 8
TOPK = 2
PHYSICAL_ROWS = 8  # identical on every rank, so only the valid/padding split differs


def _plan(world: int) -> list[int]:
    """Valid token count per rank, all different, all <= PHYSICAL_ROWS.

    Identical physical rows on every rank is deliberate: if the ranks also agreed on the
    valid count, a global-for-local substitution would be invisible.
    """
    plans = {2: [7, 3], 4: [7, 3, 5, 2], 8: [7, 3, 5, 2, 6, 1, 4, 8]}
    if world not in plans:
        pytest.skip(f"no rank plan for world={world}")
    return plans[world]


@pytest.fixture(scope="module")
def tp_domain(nccl_session):
    """A real TP group sized to the world, so Router.tp_cp_group has size > 1.

    Reuses the session-wide process group when one is already up: re-initialising the model
    parallel state per module and tearing it down again races the previous destroy and kills
    the session (see ``_a3_pg.py``).
    """
    world = nccl_session
    if world < 2:
        pytest.skip("needs >= 2 ranks for a non-trivial tp_cp_group")
    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    if not parallel_state.model_parallel_is_initialized():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=world,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
    # No teardown here on purpose: see the module docstring / _a3_pg rationale. The module
    # conftest's session-scoped cleanup destroys the process group once, at the end.
    yield world


def _build_layer(world: int, **over) -> MoELayer:
    cfg = TransformerConfig(
        tensor_model_parallel_size=world,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        num_layers=1,
        hidden_size=H,
        ffn_hidden_size=FFN,
        num_attention_heads=8,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.1,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        moe_router_dtype="fp32",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        use_cpu_initialization=True,
        bf16=True,
        # MoELayer rejects TP>1 without sequence parallelism during training
        sequence_parallel=world > 1,
        **over,
    )
    mlp_spec = get_gpt_layer_local_submodules(
        num_experts=NUM_EXPERTS, moe_grouped_gemm=False
    ).mlp
    submodules = get_submodules(mlp_spec)
    assert isinstance(submodules, MoESubmodules)
    layer = MoELayer(cfg, submodules).cuda().to(dtype=torch.bfloat16)
    layer.set_layer_number(0)
    layer.train()
    return layer


def _padding_mask(valid: int) -> torch.Tensor:
    """[seq, 1] bool with True marking PADDING, on the device, per Router.forward's contract."""
    m = torch.ones(PHYSICAL_ROWS, 1, dtype=torch.bool, device="cuda")
    m[:valid] = False
    return m


def test_helper_local_count_uses_the_real_tp_cp_group(tp_domain):
    """The helper, called the way the router calls it, returns rank-local counts."""
    from megatron.core.transformer.moe.moe_utils import get_tokens_per_expert_and_token_count

    world = tp_domain
    plan = _plan(world)
    rank = torch.distributed.get_rank()
    valid = plan[rank]

    g = torch.Generator(device="cpu").manual_seed(100 + rank)
    rmap = torch.zeros((PHYSICAL_ROWS, NUM_EXPERTS), dtype=torch.bool)
    for i in range(valid):
        rmap[i, torch.randperm(NUM_EXPERTS, generator=g)[:TOPK]] = True
    rmap = rmap.cuda()

    tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
    assert tp_cp_group.size() == world

    _hist, local_num, total_num = get_tokens_per_expert_and_token_count(
        routing_map=rmap, reduce_group=tp_cp_group, topk=TOPK, with_padding_mask=True
    )
    assert float(local_num) == valid, (
        f"rank {rank}: helper returned {float(local_num)} for tp_cp_group, expected the "
        f"rank-local valid count {valid}"
    )
    assert float(total_num) == sum(plan), (
        f"rank {rank}: group total {float(total_num)} != {sum(plan)}"
    )


def _aux_grad_for_reported_count(router, scores, rmap, reported, monkeypatch):
    """Run the real aux-loss chain and return (score-grad abs sum, aux value).

    ``reported`` is what the helper claims the rank-local valid token count is. The routing
    map, scores, aux-loss formula and process group are identical between calls, so any
    difference in the returned gradient comes only from that reported count. That is what
    makes this able to detect the group-global substitution directly.
    """
    import megatron.core.transformer.moe.router as router_mod
    from megatron.core.transformer.moe.moe_utils import (
        get_tokens_per_expert_and_token_count as real_helper,
        switch_load_balancing_loss_func,
    )

    def fake_helper(routing_map, reduce_group, topk=None, with_padding_mask=False):
        hist, _local, total = real_helper(
            routing_map=routing_map, reduce_group=reduce_group, topk=topk,
            with_padding_mask=with_padding_mask,
        )
        return hist, reported, total

    monkeypatch.setattr(router_mod, "get_tokens_per_expert_and_token_count", fake_helper)

    s = scores.detach().clone().requires_grad_(True)
    ghist, local_num, total_num = fake_helper(
        routing_map=rmap, reduce_group=router.tp_cp_group, topk=TOPK, with_padding_mask=True
    )
    aux = switch_load_balancing_loss_func(
        probs=s,
        tokens_per_expert=ghist,
        total_num_tokens=total_num,
        topk=TOPK,
        num_experts=NUM_EXPERTS,
        moe_aux_loss_coeff=router.get_aux_loss_coeff("aux_loss"),
    )
    acting = torch.randn(PHYSICAL_ROWS, 1, H, device="cuda", dtype=torch.bfloat16)
    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    attached = router.attach_and_log_load_balancing_loss(
        acting,
        router.get_aux_loss_coeff("aux_loss"),
        aux,
        "load_balancing_loss",
        router.tp_cp_group,
        valid_token_count=local_num,
    )
    attached.float().sum().backward()
    assert s.grad is not None, "aux loss produced no gradient for the router scores"
    return float(s.grad.abs().sum()), float(aux.detach())


def test_reported_local_count_scales_the_aux_gradient(tp_domain, monkeypatch):
    """The decisive regression: telling the router the group-global count instead of the
    rank-local count changes the aux-loss gradient, by exactly global/local.

    With the A3 bug, ``local_num_tokens`` IS the group-global value, so the router scales its
    aux gradient by that factor on every rank whose local count differs from the group total.
    """
    world = tp_domain
    plan = _plan(world)
    rank = torch.distributed.get_rank()
    layer = _build_layer(world, calculate_per_token_loss=True)
    router = layer.router
    assert router is not None

    torch.manual_seed(4321)  # identical scores on every rank
    scores = torch.rand(PHYSICAL_ROWS, NUM_EXPERTS, device="cuda", dtype=torch.float32)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    rmap = _routing_map(plan[rank], seed=600 + rank)

    valid_local = float(plan[rank])
    valid_total = float(sum(plan))
    assert valid_local != valid_total, "rank plan must make local and global differ"

    g_local, aux_local = _aux_grad_for_reported_count(
        router, scores, rmap, valid_local, monkeypatch
    )
    g_total, aux_total = _aux_grad_for_reported_count(
        router, scores, rmap, valid_total, monkeypatch
    )
    MoEAuxLossAutoScaler.main_loss_backward_scale = None

    # the aux loss value itself is unchanged: only the scaling term differs
    assert abs(aux_local - aux_total) < 1e-5, (
        f"the aux loss value changed ({aux_local} vs {aux_total}); the comparison is not "
        f"isolating the scaling term"
    )
    assert g_local > 0 and g_total > 0, (g_local, g_total)

    ratio = g_total / g_local
    expected = valid_total / valid_local
    assert abs(ratio - expected) < 1e-4, (
        f"rank {rank}: reporting the group-global count ({valid_total}) instead of the local "
        f"count ({valid_local}) scaled the aux gradient by {ratio:.6f}x, expected "
        f"{expected:.6f}x"
    )
    assert abs(ratio - 1.0) > 1e-3, (
        f"rank {rank}: local and global counts produced the same gradient; the fixture cannot "
        f"detect the substitution"
    )


def test_router_uses_the_local_count_end_to_end(tp_domain, monkeypatch):
    """End-to-end through Router.forward: the count the router actually uses in a real masked
    forward must be this rank's valid token count, never the group total."""
    world = tp_domain
    plan = _plan(world)
    rank = torch.distributed.get_rank()
    layer = _build_layer(world, calculate_per_token_loss=True)
    router = layer.router
    assert router is not None

    seen: list[float] = []
    real_attach = router.attach_and_log_load_balancing_loss

    def spy(activation, coeff, aux_loss, name, reduce_group, *a, **kw):
        vt = kw.get("valid_token_count")
        if vt is not None:
            seen.append(float(vt))
        return real_attach(activation, coeff, aux_loss, name, reduce_group, *a, **kw)

    monkeypatch.setattr(router, "attach_and_log_load_balancing_loss", spy)

    torch.manual_seed(4242)
    x = torch.randn(PHYSICAL_ROWS, 1, H, device="cuda", dtype=torch.bfloat16)
    router(x, padding_mask=_padding_mask(plan[rank]))
    torch.cuda.synchronize()
    assert seen, "attach_and_log_load_balancing_loss was never called during Router.forward"
    assert any(abs(v - float(plan[rank])) < 1e-5 for v in seen), (
        f"rank {rank}: router used valid_token_count={seen}, none of which is this rank's "
        f"valid count {plan[rank]} (group total is {sum(plan)})"
    )
    assert not any(abs(v - float(sum(plan))) < 1e-5 for v in seen), (
        f"rank {rank}: router used the group-global count {sum(plan)} (saw {seen})"
    )


def _routing_map(valid_rows: int, seed: int) -> torch.Tensor:
    """[PHYSICAL_ROWS, E] bool, exactly `valid_rows` rows carrying TOPK distinct experts."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rmap = torch.zeros((PHYSICAL_ROWS, NUM_EXPERTS), dtype=torch.bool)
    for i in range(valid_rows):
        rmap[i, torch.randperm(NUM_EXPERTS, generator=g)[:TOPK]] = True
    return rmap.cuda()


def test_moe_layer_training_step_is_finite(tp_domain):
    """A real MoELayer forward+backward+optimizer step must stay finite with the masked
    routing path active. This is the model-level smoke the first pass was missing."""
    world = tp_domain
    plan = _plan(world)
    rank = torch.distributed.get_rank()
    layer = _build_layer(world, calculate_per_token_loss=True)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    torch.manual_seed(9001)
    x = torch.randn(PHYSICAL_ROWS, 1, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        out = layer(x)
        hidden = out[0] if isinstance(out, tuple) else out
        loss = hidden.float().pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))

    assert all(v == v and abs(v) < 1e6 for v in losses), f"non-finite loss: {losses}"
    gnorm = sum(
        float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None
    )
    assert gnorm > 0, "no gradients reached the layer parameters"
    if rank == 0:
        print(f"[a3] training smoke losses={losses} grad_abs_sum={gnorm:.4f}")
