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


def _build_mtp_gpt_model(repeated_layer=False, moe=False, early_backward=False):
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
        mtp_loss_early_backward=early_backward,
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


def _forward_backward(model, main_grad_dtype=torch.float32, microbatches=1):
    """``microbatches`` fwd+bwd on fixed batches, then drain GTP comms the way production does.

    ``main_grad_dtype`` picks the wgrad branch: fp32 against bf16 activations widens the GEMM
    epilogue, bf16 makes the dtypes match and selects the zero-copy write into the reduce-scatter
    send buffer. Production spells these --accumulate-allreduce-grads-in-fp32 and
    --grad-reduce-in-bf16 respectively.

    main_grad is zeroed ONCE and the drain runs ONCE after the loop, mirroring
    finalize_model_grads_func being called outside the microbatch loop in
    pipeline_parallel/schedules.py. That is what makes an accumulation window span the whole
    iteration rather than a single microbatch.
    """
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        wait_for_gtp_grad_reduction_on_current_stream,
    )

    for p in model.parameters():
        p.main_grad = torch.zeros(p.shape, dtype=main_grad_dtype, device='cuda')

    gen = torch.Generator(device='cuda').manual_seed(7)
    position_ids = torch.arange(SEQ, device='cuda').unsqueeze(0).expand(BATCH, SEQ)
    losses = []
    for _ in range(microbatches):
        # Fresh data per microbatch: identical batches would hide an ordering bug.
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ), device='cuda', generator=gen)
        # TE rejects fp32 activations against bf16 params outside an autocast region.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(input_ids, position_ids, attention_mask=None, labels=labels).mean()
        loss.backward()
        losses.append(float(loss.item()))
    # The production fence. Without it an accumulated wgrad never flushes and main_grad stays 0.
    wait_for_gtp_grad_reduction_on_current_stream()
    torch.cuda.synchronize()
    return sum(losses) / len(losses)


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


def _worker_shared_weight_grads(
    rank, world_size, port, repeated_layer=False, moe=False, early_backward=False, microbatches=1
):
    """Overlapped GTP must produce the same loss and gradients as all-synchronous GTP, with MTP.

    Both phases run the SAME model, sharding (gtp_remat=world), init weights and batches -- only
    ``GTP_CONFIG.weight_prefetch`` and ``GTP_CONFIG.async_reduction`` differ. The synchronous
    phase gathers on demand and reduce-scatters and accumulates inline on every wgrad, so it has
    no prefetch, buffer reuse, deferred drain or repeated-backward hazard; it is a trusted
    reference for exactly the code the overlapped path adds. Each phase runs three iterations and
    compares the last: the second records every chain's consumes and its fence builds the
    prefetch plans, so the third runs planned. Holding the sharding identical on both sides also
    keeps every DP/GTP grad-scaling subtlety out of the comparison.

    The planned iteration must also leave no on-demand all-gather but the embedding's, whose
    shard is only final right before the iteration.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    P = GTPShardedParam
    o_ondemand = P._all_gather_weight_on_demand
    on_demand = set()

    def ondemand(self, fwd, recompute=False, recompute_prev=None):
        if not recompute:
            on_demand.add(f"{self._debug_name}.{'fwd' if fwd else 'bwd'}")
        return o_ondemand(self, fwd, recompute, recompute_prev)

    saved = (GTP_CONFIG.pad_for_alignment, GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction)
    grads, losses = {}, {}
    try:
        # pad_for_alignment=0 keeps each shard exactly 1/gtp of the full weight, so the
        # all-gather in _gathered_main_grads reconstructs the unsharded gradient directly.
        GTP_CONFIG.pad_for_alignment = 0
        P._all_gather_weight_on_demand = ondemand
        for phase, overlap in (("sync_ref", False), ("overlap", True)):
            GTP_CONFIG.weight_prefetch = GTP_CONFIG.async_reduction = overlap
            ps.destroy_model_parallel()
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                gtp_remat_size=world_size,
                **_expert_parallel_kwargs(moe, world_size),
            )
            model_parallel_cuda_manual_seed(42)
            torch.manual_seed(42)

            model = _build_mtp_gpt_model(repeated_layer, moe, early_backward)
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

            for _ in range(2):
                _forward_backward(model, microbatches=microbatches)
            on_demand.clear()
            losses[phase] = _forward_backward(model, microbatches=microbatches)
            grads[phase] = _gathered_main_grads(model)

            del model
            ps.destroy_model_parallel()
            GTPShardedParam._chain_state.clear()
            GTPShardedParam._recompute_chain_state.clear()
            GTPShardedParam._link_tables_flushed = False
    finally:
        P._all_gather_weight_on_demand = o_ondemand
        GTP_CONFIG.pad_for_alignment, GTP_CONFIG.weight_prefetch, GTP_CONFIG.async_reduction = saved

    ref, test = grads["sync_ref"], grads["overlap"]
    assert set(ref) == set(test), "param sets differ between phases"
    assert losses["overlap"] == losses["sync_ref"], losses
    differ = [n for n in ref if not torch.equal(ref[n], test[n])]
    assert not differ, (
        f"gradients differ between overlapped and synchronous GTP: {differ}. A weight consumed "
        f"more than once per forward had a gather read stale, or one of its wgrad "
        f"reduce-scatters dropped or overwritten."
    )
    # The cold start fires once per iteration: a later microbatch's first expert gather per list
    # finds its buffer busy until the previous use on that list, so it is gathered on demand.
    later_microbatch = {
        n for n in on_demand if microbatches > 1 and "decoder.layers.0.mlp.experts" in n
    }
    assert on_demand - later_microbatch == {"embedding.word_embeddings.weight.fwd"}, sorted(
        on_demand
    )


def _worker_wgrad_accum_matches_per_backward_rs(
    rank,
    world_size,
    port,
    repeated_layer=False,
    moe=False,
    symm=False,
    microbatches=1,
    iterations=1,
):
    """Summing a weight's repeated backwards before ONE reduce-scatter must not change grads.

    output_layer takes one backward per MTP depth plus one for the main head. Today each issues
    its own reduce-scatter and adds the resulting shard into main_grad. With
    accumulation on, the GEMM epilogue sums them in the wgrad buffer instead and a
    single RS runs at the end-of-backward flush. Same total, different order of summation, so the
    reduced gradients must agree -- and output_layer's RS count must actually drop, or the test
    is passing on a feature that never ran.

    ``symm=True`` covers the OTHER wgrad path. Which branch of the backward computes the wgrad is
    chosen by dtype: fp32 main_grad against bf16 activations widens the GEMM epilogue, while a
    main_grad matching the activation dtype writes straight into the registered reduce-scatter
    send buffer (use_zero_copy_wgrad). Both consume the accumulate flag, and only the first is
    reachable with this file's default fp32 main_grad -- so without this variant the zero-copy
    path, which is what --grad-reduce-in-bf16 plus --gtp-remat-nccl-ub actually runs, would have
    no coverage at all. The path taken is asserted below, not assumed.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        wait_for_gtp_grad_reduction_on_current_stream,
    )
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.gtp_symmetric_memory import (
        deregister_and_clear_gtp_symm_pools,
        register_gtp_symm_pool,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    saved_pad = GTP_CONFIG.pad_for_alignment
    orig_rs = GTPShardedParam._reduce_scatter
    grads, losses, rs_counts = {}, {}, {}
    zero_copy_seen = {}
    try:
        GTP_CONFIG.pad_for_alignment = 0
        for phase, accum in (("per_backward_rs", False), ("accumulated", True)):
            counts = {}

            def _counting_rs(self, wgrads, *a, _counts=counts, **kw):
                _counts[self._debug_name] = _counts.get(self._debug_name, 0) + 1
                return orig_rs(self, wgrads, *a, **kw)

            GTPShardedParam._reduce_scatter = _counting_rs

            ps.destroy_model_parallel()
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                gtp_remat_size=world_size,
                **_expert_parallel_kwargs(moe, world_size),
            )
            model_parallel_cuda_manual_seed(42)
            torch.manual_seed(42)
            if symm:
                # use_zero_copy_wgrad needs the symmetric pool (--gtp-remat-nccl-ub).
                register_gtp_symm_pool(ps.get_gtp_weight_remat_group())

            model = _build_mtp_gpt_model(repeated_layer, moe)
            classify_gtp_remat_chains([model])
            # _wgrad_accum_enabled is the only switch; clear it for the baseline phase.
            if not accum:
                for p in model.parameters():
                    if isinstance(p, GTPShardedParam):
                        p._wgrad_accum_enabled = False
            # The count is learned at the end of iteration 1, so the early flush is unreachable
            # in a single-iteration test. Warm up, then measure the LAST iteration.
            for _ in range(iterations - 1):
                _forward_backward(
                    model,
                    main_grad_dtype=torch.bfloat16 if symm else torch.float32,
                    microbatches=microbatches,
                )
            counts.clear()
            losses[phase] = _forward_backward(
                model,
                main_grad_dtype=torch.bfloat16 if symm else torch.float32,
                microbatches=microbatches,
            )
            zero_copy_seen[phase] = {
                n: p.use_zero_copy_wgrad(torch.bfloat16)
                for n, p in model.named_parameters()
                if n.endswith("output_layer.weight") and hasattr(p, "use_zero_copy_wgrad")
            }
            wait_for_gtp_grad_reduction_on_current_stream()
            grads[phase] = _gathered_main_grads(model)
            rs_counts[phase] = dict(counts)

            del model
            ps.destroy_model_parallel()
            if symm:
                # The pool outlives the process group; the next phase reuses the name.
                deregister_and_clear_gtp_symm_pools()
            GTPShardedParam._chain_state = {}
            GTPShardedParam._recompute_chain_state = {}
            GTPShardedParam._link_tables_flushed = False
    finally:
        GTPShardedParam._reduce_scatter = orig_rs
        GTP_CONFIG.pad_for_alignment = saved_pad
        if symm:
            deregister_and_clear_gtp_symm_pools()

    if rank != 0:
        return

    ref_counts, acc_counts = rs_counts["per_backward_rs"], rs_counts["accumulated"]
    ol = [n for n in ref_counts if n.endswith("output_layer.weight")]
    assert len(ol) == 1, f"expected one output_layer weight, got {ol}"
    name = ol[0]

    # The wgrad path is selected by dtype + pool registration, so assert the intended branch was
    # reached -- otherwise the variant silently re-tests the default one.
    for phase, seen in zero_copy_seen.items():
        assert seen, f"{phase}: no output_layer GTPShardedParam found to check the wgrad path"
        assert all(v is symm for v in seen.values()), (
            f"{phase}: expected use_zero_copy_wgrad=={symm} for output_layer, got {seen}. "
            f"symm={symm} must select the {'zero-copy' if symm else 'epilogue-widening'} "
            "wgrad branch, or this variant covers the wrong code path."
        )
    assert ref_counts[name] > 1, (
        f"baseline issued only {ref_counts[name]} reduce-scatter(s) for {name}; MTP should give "
        "it several backwards, so there is nothing to collapse and the test is vacuous"
    )
    # One RS for the whole iteration, across ALL microbatches.
    assert acc_counts.get(name, 0) == 1, (
        f"accumulated phase issued {acc_counts.get(name, 0)} reduce-scatters for {name}, "
        f"expected exactly 1 (baseline had {ref_counts[name]})"
    )
    assert ref_counts[name] >= microbatches, (
        f"baseline issued {ref_counts[name]} reduce-scatters for {name} across {microbatches} "
        "microbatches; expected at least one per microbatch, so the loop is not running"
    )

    ref, test = grads["per_backward_rs"], grads["accumulated"]
    assert set(ref) == set(test), "param sets differ between phases"
    torch.testing.assert_close(
        torch.tensor(losses["accumulated"]),
        torch.tensor(losses["per_backward_rs"]),
        atol=1e-2,
        rtol=1e-2,
    )
    max_err, worst = 0.0, None
    for n, g in ref.items():
        denom = g.abs().max().item()
        if denom < 1e-30:
            continue
        rel = (g - test[n]).abs().max().item() / denom
        if rel > max_err:
            max_err, worst = rel, n
    print(
        f"[wgrad-accum repeated={repeated_layer} moe={moe}] "
        f"RS {ref_counts[name]} -> {acc_counts[name]} for output_layer; "
        f"max rel grad diff={max_err:.3e} ({worst})",
        flush=True,
    )
    # Tolerance, not equality: summation order differs. A dropped contribution lands near 1.0.
    assert (
        max_err < 2e-2
    ), f"wgrad accumulation changed gradients (max rel {max_err:.3e} on {worst})"


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
        GTPShardedParam._chain_state.clear()
        GTPShardedParam._recompute_chain_state.clear()
        GTPShardedParam._link_tables_flushed = False
    finally:
        GTP_CONFIG.pad_for_alignment = saved_pad


def _worker_repeated_consume_reads_current_weight(
    rank, world_size, port, repeated_layer=False, moe=False
):
    """Every consume reads its weight's current gather, however often it is consumed per pass.

    MTP consumes the shared weights once per prediction head and replays the MTP block once per
    depth, while the chain neighbours run once. A consume that read whatever the shared buffer
    last held would keep the loss finite and merely wrong -- silently. So every consume served
    without a gather of its own (a prefetch, or a gather still resident in its buffer) is
    compared with a fresh all-gather of the weight's current shard, over the default-plan, the
    recording and the first planned iteration, with the shards changed in between like a step.

    Does not cover the recompute chain (separate ``_recompute_*`` slots).
    """
    from collections import defaultdict

    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTP_CONFIG
    from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    P = GTPShardedParam
    o_get = P._get_prefetched_weight
    served, violations = defaultdict(int), []

    def get_prefetched(self, fwd):
        out = o_get(self, fwd)
        for w, got in zip(self._weights, out if self.is_routed_expert else [out]):
            shard = w.get_padded_shard().detach().contiguous()
            full = shard.new_empty((shard.shape[0] * w.group.size(), *shard.shape[1:]))
            dist.all_gather_into_tensor(full, shard, group=w.group)
            if not torch.equal(got, full[: got.shape[0]]):
                violations.append(f"{w._debug_name} ({'fwd' if fwd else 'bwd'})")
        served[(self._debug_name, bool(fwd))] += 1
        return out

    saved_pad = GTP_CONFIG.pad_for_alignment
    try:
        GTP_CONFIG.pad_for_alignment = 0
        P._get_prefetched_weight = get_prefetched

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
        for _ in range(3):
            _forward_backward(model)  # ends at the fence, like finalize_model_grads
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.01)  # the optimizer step
        del model
    finally:
        # _run_distributed shares ONE process across tests: undo everything even on failure.
        P._get_prefetched_weight = o_get
        GTP_CONFIG.pad_for_alignment = saved_pad
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state.clear()
        GTPShardedParam._recompute_chain_state.clear()
        GTPShardedParam._link_tables_flushed = False

    # Every rank asserts: the checks are rank-local, so rank 0 alone could miss a violation.
    assert any(n > 1 for n in served.values()), (
        "no GTP weight was served more than once without its own gather, so this test would "
        "pass even with stale reads -- check that MTP is still attached and GTP-sharded"
    )
    assert not violations, (
        "a GTP weight was consumed from a buffer that did not hold its current gather "
        "(silently wrong weights):\n  " + "\n  ".join(violations)
    )


def _worker_ddp_grad_ready_counts(rank, world_size, port, repeated_layer=False, accumulate=False):
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
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        wait_for_gtp_grad_reduction_on_current_stream,
    )
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
        if not accumulate:
            for p in model.parameters():
                if isinstance(p, GTPShardedParam):
                    p._wgrad_accum_enabled = False
        name_of = {p: n for n, p in model.named_parameters()}

        counts = collections.Counter()

        def counting_register(self, param, *a, **k):
            counts[name_of.get(param, "?")] += 1
            return orig_register(self, param, *a, **k)

        pgb._ParamAndGradBucketGroup.register_grad_ready = counting_register

        ddp = DistributedDataParallel(
            model.config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=False,
                overlap_grad_reduce=True,
                # fp32 main_grad vs bf16 activations selects the widening wgrad path; the bf16
                # default would exercise a path where accumulation never engages.
                grad_reduce_in_fp32=accumulate,
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
            # Production order: fence, then finish_grad_sync.
            wait_for_gtp_grad_reduction_on_current_stream()
            ddp.finish_grad_sync()  # raises if a bucket never reached its golden count
            torch.cuda.synchronize()
            per_iter.append(dict(counts))

        del model, ddp
        ps.destroy_model_parallel()
        GTPShardedParam._chain_state.clear()
        GTPShardedParam._recompute_chain_state.clear()
        GTPShardedParam._link_tables_flushed = False
    finally:
        pgb._ParamAndGradBucketGroup.register_grad_ready = orig_register
        GTP_CONFIG.pad_for_alignment = saved_pad

    if rank != 0:
        return

    # DDP's golden gate needs the count IDENTICAL every iteration, not equal to 1.
    # Scoped to output_layer, so only it collapses; the embedding keeps one fire per backward.
    expected = {
        "embedding.word_embeddings.weight": 1 + MTP_NUM_LAYERS,
        "output_layer.weight": 1 if accumulate else 1 + MTP_NUM_LAYERS,
    }
    for name, exp in expected.items():
        got = [it.get(name, 0) for it in per_iter]
        print(f"[ddp-grad-ready] {name:38s} fires per iteration={got}", flush=True)
        assert got[0] == exp, f"{name}: {got[0]} grad-ready fires, expected {exp}"

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
    def test_wgrad_accum_steady_state(self):
        # Iterations 2+ take the early flush; a single-iteration test never reaches it.
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_wgrad_accum_matches_per_backward_rs, 4, False, False, False, 1, 3)

    def test_wgrad_accum_spans_microbatches(self):
        # finalize_model_grads runs once per ITERATION, so the accumulation window covers every
        # microbatch, not one.
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_wgrad_accum_matches_per_backward_rs, 4, False, False, False, 4)

    # symm = the zero-copy branch (--grad-reduce-in-bf16 + --gtp-remat-nccl-ub); default = the
    # epilogue-widening branch. Both must collapse the RS.
    @pytest.mark.parametrize("symm", [False, True], ids=["widen", "zerocopy"])
    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_wgrad_accum_matches_per_backward_rs(self, repeated_layer, moe, symm):
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_wgrad_accum_matches_per_backward_rs, 4, repeated_layer, moe, symm)

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

    @pytest.mark.parametrize("microbatches", [1, 3])
    @pytest.mark.parametrize("early_backward", [False, True], ids=["late_bwd", "early_bwd"])
    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_mtp_shared_weight_grads_match_sync_reduce_scatter(
        self, repeated_layer, moe, early_backward, microbatches
    ):
        """MTP re-uses embedding/output_layer, so those weights get several backward passes.

        Guards a SILENT failure: a stale gather or a dropped or overwritten wgrad reduce-scatter
        trains on wrong gradients without raising, so only a numeric comparison catches it. The
        ``moe`` variant additionally puts the MTP layer's experts on the grouped fc1/fc2 chains,
        which prefetch one block ahead into a shape-keyed double buffer. Compared bit for bit
        against all-synchronous GTP on the planned iteration.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(
            _worker_shared_weight_grads, 4, repeated_layer, moe, early_backward, microbatches
        )

    @pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_repeated_consume_reads_current_weight(self, repeated_layer, moe):
        """Every consume reads its weight's current gather across iterations: default plan,
        recording and planned, with the shards changed between them like an optimizer step.

        A consume may read a gather issued by an earlier consume and still resident in its buffer,
        so counting one all-gather per consume no longer holds after the first fence.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_repeated_consume_reads_current_weight, 4, repeated_layer, moe)

    @pytest.mark.parametrize("repeated_layer", [False, True])
    def test_ddp_grad_ready_counts_with_wgrad_accum(self, repeated_layer):
        """Collapsing 3 reduce-scatters into 1 also collapses grad-ready 3 -> 1; DDP must cope."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 CUDA devices")
        _run_distributed(_worker_ddp_grad_ready_counts, 4, repeated_layer, True)

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
