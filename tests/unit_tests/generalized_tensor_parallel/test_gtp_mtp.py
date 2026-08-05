# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP weight-remat correctness with Multi-Token Prediction (MTP).

MTP breaks an assumption the GTP prefetch/reduce-scatter machinery was built on: that every
weight is consumed exactly once per forward, so backward visits the chain in exact reverse
order. MTP re-uses the main model's embedding and output_layer (and, under
``mtp_use_repeated_layer``, replays its own layer), which violates that in two ways:

  * ``output_layer`` runs once per prediction head (main + one per MTP depth), so its wgrad
    reduce-scatter can still be in flight when the next wgrad arrives. ``_reduce_scatter``
    re-uses the weight's ticket -- the SAME output buffer -- and only one handle is tracked,
    so a later RS can overwrite an earlier result and silently drop a head's gradient.
  * ``embedding`` is the chain head, but MTP re-embeds near the END of forward, so its backward
    runs long before its chain successor (decoder layer 0) has issued any RS. The deferred
    finalize must not assume the successor has one pending -- doing so read a never-reserved
    ticket and raised ``KeyError: None``.

The first failure mode is SILENT (wrong gradients, no exception), which is why it needs a
numeric guard rather than a smoke test.
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
    """One fwd+bwd on a fixed batch, then drain + finalize any in-flight GTP reduce-scatter."""
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
    # Mirror finalize_model_grads: drain in-flight RS and accumulate anything the chain cascade
    # left pending, so the comparison never depends on drain timing.
    wait_async_comms(finalize_after_drain=True)
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
