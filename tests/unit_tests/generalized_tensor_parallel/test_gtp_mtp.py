# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP weight-remat correctness with Multi-Token Prediction (MTP).

GTP's chain assumes one consume per weight per pass: one all-gather from its neighbour, one
backward in reverse chain order. MTP consumes the shared embedding/output_layer once per
prediction head, and under ``mtp_use_repeated_layer`` replays its own layer once per depth.

Three ways that breaks:

  * Forward -- consumes past the first get no all-gather of their own, so the GEMM reads
    whatever the shared buffer last held.
  * Backward -- a weight's reduce-scatters overlap. They share one ticket and one tracked
    handle, so a later one overwrites an earlier result and drops a head's gradient.
  * Backward -- a shared weight is reached far from its chain position, so the deferred
    finalize cannot assume its successor has a reduce-scatter pending (``KeyError: None``).

Only the third ever raised. The other two just train on wrong numbers, hence the numeric and
accounting guards below rather than smoke tests.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

HIDDEN = 128
NUM_HEADS = 8
FFN_HIDDEN = 256
NUM_LAYERS = 2
MTP_NUM_LAYERS = 2
VOCAB = 128
SEQ = 16
BATCH = 2
NUM_EXPERTS = 4
dtype = torch.bfloat16


def _expert_parallel_kwargs(moe, world_size):
    """MoE runs EP x EGTP over the world (mirrors the a55b shape: EP2 x EGTP2 on 4 ranks)."""
    if not moe:
        return {}
    return dict(expert_model_parallel_size=2, expert_gtp_remat_size=world_size // 2)


def _build_mtp_gpt_model(repeated_layer=False, moe=False):
    """Full GPTModel (embedding + decoder + output_layer) with an MTP block attached.

    ``moe=True`` puts grouped experts in both the decoder and the MTP layer, which is what puts
    weights on the ``GTP_remat_grouped_fc1/fc2`` chains -- a separate code path from the dense
    chain, with its own one-block-ahead prefetch and double buffering.
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    moe_kwargs = (
        dict(
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=2,
            moe_ffn_hidden_size=FFN_HIDDEN,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
        )
        if moe
        else {}
    )
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        kv_channels=HIDDEN // NUM_HEADS,
        ffn_hidden_size=FFN_HIDDEN,
        use_cpu_initialization=False,
        params_dtype=dtype,
        bf16=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # The unit-test conftest pins NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=0; AttnBackend.auto
        # asserts they are unset or 1, so select the backend that matches that env.
        attention_backend=AttnBackend.unfused,
        mtp_num_layers=MTP_NUM_LAYERS,
        mtp_use_repeated_layer=repeated_layer,
        mtp_loss_scaling_factor=0.1,
        **moe_kwargs,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=NUM_EXPERTS if moe else None, moe_grouped_gemm=moe
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=True,
        post_process=True,
        mtp_block_spec=get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True),
    ).cuda()


def _forward_backward(model):
    """One fwd+bwd on a fixed batch, then drain in-flight GTP comms the way production does."""
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import wait_async_comms

    for p in model.parameters():
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device='cuda')

    gen = torch.Generator(device='cuda').manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)
    position_ids = torch.arange(SEQ, device='cuda').unsqueeze(0).expand(BATCH, SEQ)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)

    # TE rejects fp32 activations against bf16 params outside an autocast region.
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(input_ids, position_ids, attention_mask=None, labels=labels).mean()
    loss.backward()
    # Match the eager production path: finalize_model_grads reaches
    # wait_for_gtp_grad_reduction_on_current_stream, which calls wait_async_comms() WITHOUT
    # finalize_after_drain. So a reduce-scatter still pending here is waited on but never
    # accumulated, and its gradient is lost. Draining with finalize_after_drain=True would
    # rescue exactly that case and hide it from the comparison below.
    wait_async_comms()
    torch.cuda.synchronize()
    return float(loss.item())


def _gathered_main_grads(model):
    """Full (unsharded) main_grad per param name; GTP shards all-gathered over their own axis.

    Expert weights shard over expert_gtp_remat, dense weights over gtp_remat.
    """
    from megatron.core import parallel_state as ps

    out = {}
    for name, p in model.named_parameters():
        mg = p.main_grad
        if isinstance(p, GTPShardedParam):
            group = (
                ps.get_expert_gtp_weight_remat_group()
                if ('experts' in name or not getattr(p, 'allreduce', True))
                else ps.get_gtp_weight_remat_group()
            )
            shards = [torch.empty_like(mg) for _ in range(group.size())]
            dist.all_gather(shards, mg.contiguous(), group=group)
            out[name] = torch.cat(shards, dim=0).float().cpu()
        else:
            out[name] = mg.detach().float().cpu()
    return out


def _assert_grouped_chains_used(model, moe):
    """With MoE the MTP layer's experts must land on the grouped fc1/fc2 chains.

    Those chains use one-block-ahead prefetch with a shape-keyed double buffer, which is a
    different hazard from the dense chain -- so a MoE-flavoured run that quietly produced no
    grouped-chain params would be testing nothing new.
    """
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import _chain_is_grouped

    grouped_mtp = [
        n
        for n, p in model.named_parameters()
        if isinstance(p, GTPShardedParam)
        and 'mtp.' in n
        and _chain_is_grouped(getattr(p, 'chain_id', ''))
    ]
    if moe:
        assert grouped_mtp, "MoE variant produced no MTP params on a grouped fc1/fc2 chain"
    else:
        assert not grouped_mtp, f"dense variant unexpectedly used grouped chains: {grouped_mtp}"


def _worker_shared_weight_grads(rank, world_size, port, repeated_layer=False, moe=False):
    """Async reduce-scatter path must produce the same gradients as the sync path, with MTP.

    Both phases run the SAME model, sharding (gtp_remat=world), init weights and batch -- only
    ``GTP_CONFIG.async_reduction`` differs. The sync path reduce-scatters and accumulates inline
    on every wgrad, so it has no deferred cascade and no repeated-backward hazard; it is a
    trusted reference for exactly the code the async path adds. Holding the sharding identical
    on both sides also keeps every DP/GTP grad-scaling subtlety out of the comparison.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    saved_async = GTP_CONFIG.async_reduction
    saved_pad = GTP_CONFIG.pad_for_alignment
    grads, losses = {}, {}
    try:
        # pad_for_alignment=0 keeps each shard exactly 1/gtp of the full weight, so the
        # all-gather in _gathered_main_grads reconstructs the unsharded gradient directly.
        GTP_CONFIG.pad_for_alignment = 0
        for phase, use_async in (("sync_ref", False), ("async", True)):
            ps.destroy_model_parallel()
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                gtp_remat_size=world_size,
                **_expert_parallel_kwargs(moe, world_size),
            )
            model_parallel_cuda_manual_seed(42)
            torch.manual_seed(42)
            GTP_CONFIG.async_reduction = use_async

            model = _build_mtp_gpt_model(repeated_layer, moe)
            classify_gtp_remat_chains([model])

            shared = [
                n
                for n, p in model.named_parameters()
                if isinstance(p, GTPShardedParam)
                and ("output_layer" in n or "word_embeddings" in n)
            ]
            assert len(shared) == 2, (
                f"expected embedding + output_layer to be GTP-sharded (the MTP-shared weights), "
                f"got {shared} -- test would not exercise the shared-weight path"
            )
            _assert_grouped_chains_used(model, moe)

            losses[phase] = _forward_backward(model)
            grads[phase] = _gathered_main_grads(model)

            del model
            ps.destroy_model_parallel()
            GTPShardedParam._chain_state = {}
            GTPShardedParam._recompute_chain_state = {}
            GTPShardedParam._link_tables_flushed = False
    finally:
        GTP_CONFIG.async_reduction = saved_async
        GTP_CONFIG.pad_for_alignment = saved_pad

    if rank != 0:
        return

    ref, test = grads["sync_ref"], grads["async"]
    assert set(ref) == set(test), "param sets differ between phases"
    torch.testing.assert_close(
        torch.tensor(losses["async"]), torch.tensor(losses["sync_ref"]), atol=1e-2, rtol=1e-2
    )

    max_err, worst = 0.0, None
    for name in ref:
        rg, tg = ref[name], test[name]
        assert rg.shape == tg.shape, f"{name}: {rg.shape} vs {tg.shape}"
        rel = (rg - tg).abs().max().item() / (rg.abs().max().item() + 1e-8)
        if "output_layer" in name or "word_embeddings" in name:
            ratio = (tg.norm() / (rg.norm() + 1e-12)).item()
            print(
                f"[mtp-shared] {name:48s} rel_max_err={rel:.3e} "
                f"norm_ratio(async/sync)={ratio:.4f}",
                flush=True,
            )
        if rel > max_err:
            max_err, worst = rel, name
    print(f"[mtp-shared] max relative grad error async-vs-sync = {max_err:.3e} (worst: {worst})")

    assert max_err < 2e-2, (
        f"MTP shared-weight gradient mismatch between the async and sync GTP reduce-scatter "
        f"paths (max rel err {max_err:.3e} on {worst}). A weight consumed more than once per "
        f"forward had one of its wgrad reduce-scatters dropped or overwritten."
    )


def _worker_runs_end_to_end(rank, world_size, port, repeated_layer=False, moe=False):
    """GTP + MTP completes a fwd+bwd at all (regression guard for the KeyError: None crash)."""
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    saved_pad = GTP_CONFIG.pad_for_alignment
    try:
        GTP_CONFIG.pad_for_alignment = 0
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_remat_size=world_size,
            **_expert_parallel_kwargs(moe, world_size),
        )
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)

        model = _build_mtp_gpt_model(repeated_layer, moe)
        classify_gtp_remat_chains([model])
        mtp_gtp = [
            n
            for n, p in model.named_parameters()
            if isinstance(p, GTPShardedParam) and ".mtp." in f".{n}"
        ]
        assert mtp_gtp, "no MTP parameter was GTP-sharded; test would be vacuous"
        _assert_grouped_chains_used(model, moe)

        loss = _forward_backward(model)
        assert torch.isfinite(torch.tensor(loss)), f"non-finite loss {loss}"
        for name, p in model.named_parameters():
            assert torch.isfinite(p.main_grad).all(), f"non-finite grad in {name}"

        del model
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state = {}
        GTPShardedParam._recompute_chain_state = {}
        GTPShardedParam._link_tables_flushed = False
    finally:
        GTP_CONFIG.pad_for_alignment = saved_pad


def _worker_repeated_consume_all_gathers(rank, world_size, port, repeated_layer=False, moe=False):
    """N consumes of a weight need N all-gathers, not the one its chain neighbour issues.

    The neighbour runs once per pass; MTP consumes shared weights once per prediction head and
    replays the MTP block once per depth. Extra consumes would ``cache.get()`` a stale buffer --
    silently, since that keeps the loss finite and merely wrong.

    So: tally all-gathers issued against consumes, and fail if a consume outruns its issues.
    Behavioural, not structural, so it survives a redesign of how the prefetch is armed.

    Does not cover the recompute chain (separate ``_recompute_*`` slots).
    """
    from collections import defaultdict

    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    P = GTPShardedParam
    o_ag = P._all_gather_weight
    o_get = P._get_prefetched_weight
    o_ondemand = P._all_gather_weight_on_demand

    # Keyed by (name, direction) rather than id(), which can be recycled onto another object.
    issued, consumed, violations = defaultdict(int), defaultdict(int), []

    # Signatures spelled out, not *args: a signature change should fail loudly, not mis-key.
    def ag(self, async_op, fwd, nvtx_label=None):
        issued[(self._debug_name, bool(fwd))] += 1
        return o_ag(self, async_op, fwd, nvtx_label=nvtx_label)

    def get_prefetched(self, fwd):
        key = (self._debug_name, bool(fwd))
        if consumed[key] >= issued[key]:
            violations.append(
                f"{self._debug_name} ({'fwd' if fwd else 'bwd'}): consume "
                f"#{consumed[key] + 1} but only {issued[key]} all-gather(s) issued"
            )
        consumed[key] += 1
        return o_get(self, fwd)

    def on_demand(self, fwd):
        # Issues its own AG then consumes it, so both tallies stay balanced on this path.
        out = o_ondemand(self, fwd)
        consumed[(self._debug_name, bool(fwd))] += 1
        return out

    saved_pad = GTP_CONFIG.pad_for_alignment
    try:
        GTP_CONFIG.pad_for_alignment = 0
        P._all_gather_weight, P._get_prefetched_weight = ag, get_prefetched
        P._all_gather_weight_on_demand = on_demand

        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_remat_size=world_size,
            **_expert_parallel_kwargs(moe, world_size),
        )
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)

        model = _build_mtp_gpt_model(repeated_layer, moe)
        classify_gtp_remat_chains([model])
        _forward_backward(model)
        del model
    finally:
        # _run_distributed shares ONE process across tests: undo everything even on failure.
        P._all_gather_weight = o_ag
        P._get_prefetched_weight = o_get
        P._all_gather_weight_on_demand = o_ondemand
        GTP_CONFIG.pad_for_alignment = saved_pad
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state = {}
        GTPShardedParam._recompute_chain_state = {}
        GTPShardedParam._link_tables_flushed = False

    # Every rank asserts: the tallies are rank-local, so rank 0 alone could miss a violation.
    repeats = {name: n for (name, _), n in consumed.items() if n > 1}
    assert repeats, (
        "no GTP weight was consumed more than once per pass, so this test would pass even with "
        "the prefetch bug present -- check that MTP is still attached and GTP-sharded"
    )
    assert not violations, (
        "a GTP weight was consumed without an all-gather issued for that consume, so the GEMM "
        "read a stale shared buffer (silently wrong weights):\n  " + "\n  ".join(violations)
    )


def _worker_ddp_grad_ready_counts(rank, world_size, port, repeated_layer=False):
    """A weight consumed N times per forward fires DDP grad-ready N times, not once.

    Each consume finalizes the previous reduce-scatter, and every finalize calls the param's
    grad-ready hook. DDP absorbs that only because bucket completion compares the per-param
    count against a golden snapshot taken at the end of the first batch -- so the count has to
    be identical on every later iteration.

    Two requirements keep this separate from the other cases in this file: register_grad_ready
    asserts overlap_grad_reduce, and the golden gate is first evaluated on batch 2, hence three
    iterations. A desynchronized count makes finish_grad_sync raise, so completing the loop is
    itself an assertion.
    """
    import collections

    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed import param_and_grad_buffer as pgb
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    saved_pad = GTP_CONFIG.pad_for_alignment
    orig_register = pgb._ParamAndGradBucketGroup.register_grad_ready
    try:
        GTP_CONFIG.pad_for_alignment = 0
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
        )
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)

        model = _build_mtp_gpt_model(repeated_layer=repeated_layer, moe=False)
        classify_gtp_remat_chains([model])
        name_of = {p: n for n, p in model.named_parameters()}

        counts = collections.Counter()

        def counting_register(self, param, *a, **k):
            counts[name_of.get(param, "?")] += 1
            return orig_register(self, param, *a, **k)

        pgb._ParamAndGradBucketGroup.register_grad_ready = counting_register

        ddp = DistributedDataParallel(
            model.config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=False, overlap_grad_reduce=True
            ),
            model,
        )

        gen = torch.Generator(device='cuda').manual_seed(7)
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)
        position_ids = torch.arange(SEQ, device='cuda').unsqueeze(0).expand(BATCH, SEQ)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)

        # 3 iterations: batch 1 records golden, batches 2 and 3 are compared against it.
        per_iter = []
        for _ in range(3):
            counts.clear()
            ddp.zero_grad_buffer()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = ddp(input_ids, position_ids, attention_mask=None, labels=labels).mean()
            loss.backward()
            ddp.finish_grad_sync()  # raises if a bucket never reached its golden count
            torch.cuda.synchronize()
            per_iter.append(dict(counts))

        del model, ddp
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state = {}
        GTPShardedParam._recompute_chain_state = {}
        GTPShardedParam._link_tables_flushed = False
    finally:
        pgb._ParamAndGradBucketGroup.register_grad_ready = orig_register
        GTP_CONFIG.pad_for_alignment = saved_pad

    if rank != 0:
        return

    expected = 1 + MTP_NUM_LAYERS  # main head + one per MTP depth
    for name in ("embedding.word_embeddings.weight", "output_layer.weight"):
        got = [it.get(name, 0) for it in per_iter]
        print(f"[ddp-grad-ready] {name:38s} fires per iteration={got}", flush=True)
        assert got[0] == expected, f"{name}: {got[0]} grad-ready fires, expected {expected}"

    assert per_iter[1] == per_iter[0] and per_iter[2] == per_iter[0], (
        f"grad-ready counts vary across iterations {per_iter}; DDP's golden gate would never "
        f"match and the bucket would go unreduced"
    )


# Value written into the shared dummy-wgrad buffer to make the leak observable. Any non-zero
# value works; a large one keeps it far outside the range of a real gradient.
DUMMY_POISON = 12345.0


def _worker_dummy_wgrad_not_leaked(rank, world_size, port, repeated_layer=False):
    """A GTP weight's gradient must not depend on the shared dummy-wgrad buffer's contents.

    GTP returns a PLACEHOLDER from backward, not a gradient -- the real wgrad already went into
    main_grad via reduce-scatter. But it becomes ``param.grad``, and DDP adds ``param.grad`` into
    main_grad when ``zero_out_wgrad`` is set (``_make_backward_post_hook``), which MTP does to the
    shared embedding weight. ``get_dummy_wgrad`` hands back ONE reused buffer, and the shared
    weight gets N = (1 main pass + mtp_num_layers) x gradient-accumulation steps backward passes
    per iteration, so::

        want:  S1 + S2 + ... + SN            Si = call i's reduce-scattered gradient
        got:   S1 + S2 + ... + SN  +  N*D    D  = whatever that shared buffer last held

    D is near-zero while the buffer is fresh and real data once traffic has passed through it --
    hence the original failure needing both MTP and gradient accumulation, and vanishing whenever
    an extra allocation shifted the buffer. This test POISONS it with a known value instead, and
    requires the gradients to come out bit-identical. It runs a single step, so N = 1 + 2 = 3 and
    reverting the fix shifts the gradient by ~3x the poison.
    """
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        reset_gtp_state,
    )
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    def _train_one_step_with_poison(poison):
        """Build a fresh MTP model, poison the shared dummy, run fwd+bwd, return {name: grad}."""
        reset_gtp_state()  # reset_gtp_globals is an autouse FIXTURE; this is what it defers to
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=world_size
        )
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)

        model = _build_mtp_gpt_model(repeated_layer, moe=False)
        classify_gtp_remat_chains([model])
        for p in model.parameters():
            p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device='cuda')

        tagged = [n for n, p in model.named_parameters() if getattr(p, "zero_out_wgrad", False)]
        # Fail loudly rather than pass vacuously: with nothing tagged, DDP never reads the
        # placeholder and this test would be green no matter what GTP returns.
        assert tagged, (
            "no parameter has zero_out_wgrad set -- MTP should tag the shared embedding weight. "
            "Without it this test cannot observe the placeholder at all."
        )
        # get_dummy_wgrad keys on main_grad's shape, so this dirties the exact buffer GTP will
        # hand back from _handle_megatron_grad_accum.
        for _, p in model.named_parameters():
            if getattr(p, "zero_out_wgrad", False):
                get_dummy_wgrad(list(p.main_grad.shape), p.dtype).fill_(poison)

        _forward_backward(model)

        # The model here is unwrapped, so replay DDP's accumulation rule explicitly -- see
        # DistributedDataParallel._make_backward_post_hook. This is what makes the test able to
        # observe the placeholder at all: drop it and the test passes even with the bug present.
        for p in model.parameters():
            if p.grad is not None and (
                not getattr(p, "grad_added_to_main_grad", False)
                or getattr(p, "zero_out_wgrad", False)
            ):
                p.main_grad.add_(p.grad.to(p.main_grad.dtype))

        return _gathered_main_grads(model), tagged

    saved_pad = GTP_CONFIG.pad_for_alignment
    try:
        # Shards stay exactly 1/gtp of the weight so _gathered_main_grads reconstructs it.
        GTP_CONFIG.pad_for_alignment = 0
        clean, tagged = _train_one_step_with_poison(0.0)
        dirty, _ = _train_one_step_with_poison(DUMMY_POISON)
    finally:
        GTP_CONFIG.pad_for_alignment = saved_pad

    for name in clean:
        delta = (dirty[name] - clean[name]).abs().max().item()
        assert delta == 0.0, (
            f"{name}: gradient changed by {delta} when the shared dummy-wgrad buffer was "
            f"poisoned with {DUMMY_POISON}. GTP's placeholder grad leaked into the gradient; it "
            f"must be zeroed whenever zero_out_wgrad is set (tagged here: {tagged})."
        )


class TestGTPMTP:
    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_gtp_mtp_runs_end_to_end(self, repeated_layer, moe):
        """GTP + MTP fwd+bwd must complete with finite loss and gradients.

        Before the shared-weight fix this raised ``KeyError: None`` from the deferred
        reduce-scatter finalize, because the embedding's backward runs out of chain order.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_runs_end_to_end, 4, repeated_layer, moe)

    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_mtp_shared_weight_grads_match_sync_reduce_scatter(self, repeated_layer, moe):
        """MTP re-uses embedding/output_layer, so those weights get several backward passes.

        Guards a SILENT failure: a dropped or overwritten wgrad reduce-scatter trains on wrong
        gradients without raising, so only a numeric comparison catches it. The ``moe`` variant
        additionally puts the MTP layer's experts on the grouped fc1/fc2 chains, which prefetch
        one block ahead into a shape-keyed double buffer.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_shared_weight_grads, 4, repeated_layer, moe)

    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_repeated_consume_gets_its_own_all_gather(self, repeated_layer, moe):
        """A weight consumed N times per pass needs N all-gathers, not one.

        The chain prefetches from a weight's neighbour, which runs once, so every consume past
        the first used to read whatever the shared buffer last held. Guards a SILENT failure:
        the stale buffer keeps the loss finite and merely wrong, and the in-tree state guard
        that would catch it is disabled outside debug builds.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_repeated_consume_all_gathers, 4, repeated_layer, moe)

    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_mtp_shared_weight_ddp_grad_ready_counts(self, repeated_layer):
        """A re-used weight fires DDP grad-ready once per consume, not once per iteration.

        Guards the DDP side of the repeated-backward finalize: bucket completion is gated on a
        golden per-param count, so that count must be stable across iterations.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_ddp_grad_ready_counts, 4, repeated_layer)

    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_mtp_dummy_wgrad_does_not_leak_into_grad(self, repeated_layer):
        """MTP sets zero_out_wgrad on the shared embedding, which makes DDP ADD param.grad.

        GTP's placeholder grad must therefore be zeroed, or the shared dummy buffer's stale
        contents land in main_grad -- silently at first, then as a NaN once enough backward
        passes have dirtied it.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_dummy_wgrad_not_leaked, 4, repeated_layer)
