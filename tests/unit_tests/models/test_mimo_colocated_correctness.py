# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Gradient-scaling correctness for colocated MimoModel under heterogeneous DP.

Verifies that a heterogeneous-DP MimoModel produces the same post-step
encoder weights as an **equal-DP** reference built on the SAME encoder
TP/DP layout as the dist model (so the bridge is the identity
passthrough — ``BridgeDirection.EQUAL`` in
``ColocatedBridgeCommunicator``). Under correct grad scaling, both
configs yield the DP=1 gradient on every encoder shard, so the Adam
update lands on identical values and the sharded post-step weights
compare directly.

Why an equal-DP reference is the right oracle:
  * Encoder sharding matches exactly — ref and dist both use
    ``enc_tp=dist_enc_tp, enc_dp=dist_enc_dp``. Shards line up 1:1,
    so there is no gather-and-slice in the weight comparison and no
    TP=1-vs-TP>1 accumulation-order drift to contend with.
  * ``enc_dp == llm_dp`` on the ref side → the bridge is identity and
    every encoder rank feeds its colocated LLM rank with no
    redistribution collective.
  * Both sides set ``calculate_per_token_loss=True`` on their
    TransformerConfigs, which pins DDP's ``gradient_scaling_factor=1.0``
    — pure SUM across DP. The custom
    ``finalize_grads_func`` in ``_wire_training_hooks`` all-reduces
    ``total_num_tokens`` over the LLM DP group, then calls
    ``scale_gradients(1/N_global)`` on both encoder and LLM. This lands
    the true global per-token mean on every shard without touching
    ``DistributedDataParallel``.

LLM TP differs between ref (``llm_tp=dist_enc_tp``) and dist
(``llm_tp=dist_llm_tp``), so ref's LLM weights are copied into dist via
all-gather-across-ref-TP + slice-for-dist-TP. The LLM forward then
diverges numerically by fp32 TP accumulation order, but the aggregate
gradient that flows back into the encoder remains the DP=1 gradient in
both models, which is what the post-step encoder weight oracle checks.
The test runs in fp32 with ``add_bias_linear=False`` and dropout
disabled to minimize non-bridge numerical noise — this keeps the
post-bridge hidden states bit-exact and surfaces only TP-shape drift
in the logits oracle.

If the heterogeneous-DP scaling is wrong (e.g. dividing by encoder_dp
when it should be 1, or letting either DDP apply its default ``1/dp_size``
on top of the per-token mean already delivered by the finalize hook),
the dist encoder's post-step weights diverge from the ref encoder's
weights — a single Adam step is enough to detect.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_colocated_correctness.py -v -s
"""

import os
from contextlib import ExitStack, contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.utils import unwrap_model
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.test_utilities import Utils


def loss_func(loss_mask, output_tensor):
    """Per-token-loss 3-tuple: raw local sum + local valid-token count.

    Returns ``(local_sum, local_num_tokens, log_dict)`` — the contract the
    schedule expects when ``calculate_per_token_loss=True`` is set on the
    TransformerConfig. No ``1/num_tokens`` or ``1/num_microbatches``
    division is applied here; the schedule skips the per-microbatch
    division (see ``schedules.py:270-274``) and aggregates ``num_tokens``
    across microbatches for the finalize step.

    Paired with ``mimo_finalize_grads_func`` below, which all-reduces
    ``total_num_tokens`` over the LLM DP group to obtain ``N_global`` and
    then divides both encoder and LLM grads by ``1/N_global`` directly via
    ``scale_gradients`` — landing the true global per-token mean on every
    shard without touching DDP.

    ``output_tensor`` is per-token CE from
    ``GPTModel.compute_language_model_loss`` with shape ``[b, s]``.
    """
    if output_tensor is None:
        zero_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        zero_count = torch.tensor(0, device='cuda', dtype=torch.int)
        return zero_loss, zero_count, {'loss_reduced': 0.0}

    masked = output_tensor.float() * loss_mask.float()
    local_sum = masked.sum()
    local_num_tokens = loss_mask.float().sum().to(torch.int)
    return local_sum, local_num_tokens, {'loss_reduced': local_sum.detach().item()}


def forward_step(data_iterator, model, encoder_grid, llm_grid, encoder_name):
    """Forward step with per-rank data slicing for heterogeneous DP."""
    batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}

    if batch.get('input_ids') is None:
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask)

    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if encoder_dp > llm_dp:
        # Fan-in: input was pre-sliced to LLM-DP (larger per-rank batch).
        # Narrow modality_inputs to the encoder's smaller per-rank slice.
        scale = encoder_dp // llm_dp
        encoder_dp_idx = encoder_grid.get_pg("dp").rank()
        slot = encoder_dp_idx % scale

        if 'modality_inputs' in batch and batch['modality_inputs'] is not None:
            for mod_name, mod_data in batch['modality_inputs'].items():
                for enc_name, enc_data in mod_data.items():
                    for key, tensor in enc_data.items():
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            batch_size = tensor.shape[1]  # [seq, batch, hidden]
                            slice_size = batch_size // scale
                            start = slot * slice_size
                            enc_data[key] = tensor[:, start : start + slice_size, :].contiguous()

    elif llm_dp > encoder_dp:
        # Fan-out: input was pre-sliced to encoder-DP (larger per-rank batch).
        # Narrow the LLM-side tensors to this LLM-DP rank's slice.
        scale = llm_dp // encoder_dp
        llm_dp_idx = llm_grid.get_pg("dp").rank()
        slot = llm_dp_idx % scale

        batch_size = batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size

        for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][start : start + slice_size].contiguous()

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask)


def _set_deterministic_env():
    for k, v in {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ[k] = v
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)


def _wire_training_hooks(mimo_model, language_pg, vision_pg):
    """Attach no_sync / finalize_grads / grad_scale hooks to a MimoModel.

    The finalize hook implements the heterogeneous-DP grad-scaling story
    without touching ``DistributedDataParallel``. Both sub-model configs
    set ``calculate_per_token_loss=True``, so both DDPs pure-SUM across
    their own DP group (``gradient_scaling_factor=1.0``). After backward
    and DDP reduce, every rank's ``main_grad`` holds the un-normalized
    full-batch sum of per-token gradients.

    This hook then:
      1. all-reduces the schedule's ``total_num_tokens`` across the LLM
         DP group to obtain ``N_global`` (total valid tokens in the global
         batch). Since ranks are colocated, every rank now knows
         ``N_global``.
      2. Calls ``finalize_model_grads(num_tokens=None)`` per side — runs
         the usual DDP grad finish + layernorm/embedding AR work without
         letting the built-in divisor path fire.
      3. Calls ``scale_gradients(1/N_global)`` on each side — lands the
         true global per-token mean uniformly on encoder and LLM grads.

    Note: encoder has no loss_func (so nothing emits a per-encoder-DP
    ``num_tokens`` to feed ``finalize_model_grads``' internal all-reduce).
    Doing the all-reduce once ourselves and calling ``scale_gradients``
    directly avoids engineering a fictitious per-encoder-rank count whose
    sum happens to equal ``N_global``.
    """

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    def finalize_grads_func(model_list, num_tokens, force_all_reduce=False, **kwargs):
        # Schedule passes the per-rank sum-across-microbatches of what the
        # loss_func returned. Because loss_func runs only on the LLM side,
        # this is the LLM-local token count.
        assert num_tokens is not None, (
            "finalize_grads_func expects calculate_per_token_loss=True on the "
            "TransformerConfig so the schedule forwards total_num_tokens; got None."
        )

        # Phase 1: lift the all-reduce. After this, every rank (including
        # encoder-only replicas) has N_global = total non-padded tokens in
        # the global batch.
        llm_dp_pg = language_pg.dp_cp if language_pg.dp_cp is not None else language_pg.dp
        dist.all_reduce(num_tokens, group=llm_dp_pg, op=dist.ReduceOp.SUM)
        n_global = num_tokens.item()

        # Phase 2: per-side DDP finish without built-in num_tokens scaling.
        # Forward ``force_all_reduce`` so PP grad-sync semantics (if ever
        # exercised here) aren't silently dropped.
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=vision_pg,
                    force_all_reduce=force_all_reduce,
                )

        # Phase 3: uniform divide by N_global. Guard div-by-zero for the
        # degenerate fully-masked batch.
        if n_global > 0:
            inv = 1.0 / n_global
            if mimo_model.language_model is not None:
                mimo_model.language_model.scale_gradients(inv)
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    submodule.scale_gradients(inv)

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )


def _generate_and_broadcast_global_batches(
    global_mbs,
    seq_length,
    hidden_size,
    vocab_size,
    encoder_name,
    num_batches,
    image_token_id=50257,
    mask_pattern="uniform",
):
    """Generate global batches on rank 0 and broadcast so every rank sees
    identical data. Dist pre-slices per rank; ref consumes the full batch.

    ``mask_pattern``:
      * ``"uniform"`` — every sample has the same valid-token count (image
        tokens masked, text tokens all valid). Local/global denominators
        coincide up to DP-rank partitioning.
      * ``"asymmetric"`` — each sample zeros out an additional sample-
        dependent number of trailing text tokens, so different samples
        (and therefore different DP-rank slices) carry different valid-
        token counts. This exercises the num+den global-mean CE path
        where the old local-mean recipe would be only approximately
        correct.
    """
    if mask_pattern not in ("uniform", "asymmetric"):
        raise ValueError(f"Unknown mask_pattern: {mask_pattern!r}")

    rank = dist.get_rank()
    image_seq_length = seq_length // 2
    batches = []

    for batch_idx in range(num_batches):
        if rank == 0:
            encoder_hidden_states = torch.randn(
                image_seq_length,
                global_mbs,
                hidden_size,
                device='cuda',
                dtype=torch.float32,
            )
            image_tokens = torch.full(
                (global_mbs, image_seq_length),
                image_token_id,
                dtype=torch.long,
                device='cuda',
            )
            text_tokens = torch.randint(
                1,
                vocab_size,
                (global_mbs, seq_length - image_seq_length),
                device='cuda',
            )
            input_ids = torch.cat([image_tokens, text_tokens], dim=1)
        else:
            encoder_hidden_states = torch.empty(
                image_seq_length,
                global_mbs,
                hidden_size,
                device='cuda',
                dtype=torch.float32,
            )
            input_ids = torch.empty(global_mbs, seq_length, dtype=torch.long, device='cuda')

        dist.broadcast(encoder_hidden_states, src=0)
        dist.broadcast(input_ids, src=0)

        labels = input_ids.clone()
        labels[input_ids == image_token_id] = -100
        loss_mask = torch.ones(global_mbs, seq_length, device='cuda', dtype=torch.float32)
        loss_mask[input_ids == image_token_id] = 0.0

        if mask_pattern == "asymmetric":
            # Zero out a sample-dependent trailing run of text tokens so
            # each sample ends up with a different valid-token count.
            # Counts are deterministic given (batch_idx, sample_idx) so the
            # broadcast-on-rank-0 pattern is reproducible on every rank.
            text_len = seq_length - image_seq_length
            for sample_idx in range(global_mbs):
                n_drop = ((batch_idx * 7 + sample_idx * 3) % (text_len - 1)) + 1
                loss_mask[sample_idx, seq_length - n_drop :] = 0.0
                labels[sample_idx, seq_length - n_drop :] = -100
        position_ids = (
            torch.arange(seq_length, device='cuda')
            .unsqueeze(0)
            .expand(global_mbs, -1)
            .clone()
        )

        batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "modality_inputs": {
                    encoder_name: {
                        "clip_encoder": {
                            'hidden_states': encoder_hidden_states,
                            'attention_mask': None,
                        }
                    }
                },
            }
        )

    return batches


def _slice_batch(global_batch, split_dp, split_rank):
    """Return the ``split_rank``-th of ``split_dp`` slices along the batch dim."""
    batch_dim = global_batch['input_ids'].shape[0]
    slice_size = batch_dim // split_dp
    start = split_rank * slice_size
    end = start + slice_size

    per_rank = {}
    for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
        per_rank[key] = global_batch[key][start:end].contiguous()

    mod_inputs_new = {}
    for mod_name, mod_data in global_batch['modality_inputs'].items():
        mod_inputs_new[mod_name] = {}
        for enc_name, enc_data in mod_data.items():
            mod_inputs_new[mod_name][enc_name] = {}
            for key, tensor in enc_data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # modality hidden_states is [seq, batch, hidden] — slice dim 1
                    mod_inputs_new[mod_name][enc_name][key] = tensor[
                        :, start:end, :
                    ].contiguous()
                else:
                    mod_inputs_new[mod_name][enc_name][key] = tensor
    per_rank['modality_inputs'] = mod_inputs_new
    return per_rank


def _slice_global_batch_for_dist(global_batch, encoder_grid, llm_grid):
    """Pre-slice a global batch to the per-rank batch that ``forward_step`` expects.

    ``forward_step`` assumes each rank already has its LLM-DP slice
    (fan-in) or encoder-DP slice (fan-out); this helper performs that
    slicing so both models can consume the same underlying global batch.
    When ``enc_dp == llm_dp`` there is no fan-in/fan-out to pre-slice for
    (``forward_step`` also skips slicing), and the full batch is returned.
    """
    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if enc_dp > llm_dp:
        return _slice_batch(global_batch, llm_dp, llm_grid.get_pg("dp").rank())
    if llm_dp > enc_dp:
        return _slice_batch(global_batch, enc_dp, encoder_grid.get_pg("dp").rank())
    return global_batch


def _slice_global_batch_by_dp(global_batch, dp_pg):
    """Slice a global batch along the batch dim by ``dp_pg`` rank.

    For the equal-DP reference (``enc_dp == llm_dp``, bridge is identity),
    each rank consumes 1/``dp_size`` of the global batch directly.
    ``_slice_global_batch_for_dist`` returns the full batch in that case,
    so this helper does the DP-rank split explicitly.
    """
    dp_size = dist.get_world_size(dp_pg)
    if dp_size <= 1:
        return global_batch
    return _slice_batch(global_batch, dp_size, dist.get_rank(dp_pg))


def _copy_ref_params_to_dist(ref_module, dist_module, ref_tp_group, dist_tp_group):
    """Copy ref params into dist, handling differing TP shardings.

    When ref and dist params have the same shape (same TP size and layout
    at offset=0), shards align 1:1 and we copy directly. When shapes differ
    (different TP sizes), we all-gather ref's shards across ``ref_tp_group``
    to reconstruct the full weight, then slice by the dist ``partition_dim``
    for this rank's dist TP shard.

    Must be called **before** constructing the distributed optimizer, which
    clones current param data into fp32 master weights at __init__.
    """
    ref_tp_size = dist.get_world_size(ref_tp_group)
    dist_tp_rank = dist.get_rank(dist_tp_group)
    dist_tp_size = dist.get_world_size(dist_tp_group)
    ref_params = dict(ref_module.named_parameters())

    with torch.no_grad():
        for name, dist_param in dist_module.named_parameters():
            assert name in ref_params, f"Param '{name}' in dist but not in ref"
            ref_param = ref_params[name]
            partition_dim = getattr(dist_param, 'partition_dim', -1)

            if ref_param.shape == dist_param.shape:
                # Same shard size (same TP layout or both replicated).
                dist_param.data.copy_(ref_param.data.to(dist_param.dtype))
                continue

            assert partition_dim >= 0, (
                f"Param '{name}': shapes differ "
                f"(ref={tuple(ref_param.shape)}, dist={tuple(dist_param.shape)}) "
                f"but partition_dim<0 — cannot reshard a replicated param."
            )

            # Different TP sizes: gather ref shards, then slice for dist.
            shards = [
                torch.empty_like(ref_param.data) for _ in range(ref_tp_size)
            ]
            dist.all_gather(
                shards, ref_param.data.contiguous(), group=ref_tp_group
            )
            full_weight = torch.cat(shards, dim=partition_dim)
            dist_slice = torch.tensor_split(
                full_weight, dist_tp_size, dim=partition_dim
            )[dist_tp_rank]

            assert dist_slice.shape == dist_param.shape, (
                f"Param '{name}': sliced.shape={tuple(dist_slice.shape)} != "
                f"dist.shape={tuple(dist_param.shape)} "
                f"(ref_tp={ref_tp_size}, dist_tp={dist_tp_size}, "
                f"partition_dim={partition_dim})"
            )
            dist_param.data.copy_(dist_slice.to(dist_param.dtype))


def _global_abs_diff_stats(a, b, pg=None):
    """Absolute-diff stats plus reference-tensor magnitude stats, across ``pg``.

    Reports both the abs-diff distribution AND the magnitude of ``b`` (the
    reference tensor) so the caller can judge scale: a max abs-diff of 1.0
    is catastrophic for values of O(1), but fine for values of O(100). The
    relative-diff column (``rel_max = max_diff / ref_max``) gives a quick
    percentage read.

    Useful when the per-rank tensors cover different shards — all-reducing
    MAX/MIN (and MAX of per-rank p95/p99 as a conservative worst-case) lets
    rank 0 print a global view of drift across every shard in ``pg``. Mean
    is SUM/world_size, which is the true global mean when every rank holds
    the same number of elements (true here — shards have the same shape).
    """
    diff = (a.float() - b.float()).abs().flatten()
    ref = b.float().abs().flatten()
    n = diff.numel()

    if n == 0:
        zero = torch.tensor(0.0, device='cuda')
        local_min = local_max = local_mean = local_p50 = local_p95 = local_p99 = zero
        local_ref_max = local_ref_p95 = local_ref_mean = zero
    else:
        local_min = diff.min()
        local_max = diff.max()
        local_mean = diff.mean()
        local_p50 = diff.quantile(0.50)
        local_p95 = diff.quantile(0.95)
        local_p99 = diff.quantile(0.99)
        local_ref_max = ref.max()
        local_ref_p95 = ref.quantile(0.95)
        local_ref_mean = ref.mean()

    world = dist.get_world_size(pg) if dist.is_initialized() else 1
    if world > 1:
        g_min = local_min.clone()
        g_max = local_max.clone()
        g_mean = local_mean.clone()
        g_p50 = local_p50.clone()
        g_p95 = local_p95.clone()
        g_p99 = local_p99.clone()
        g_ref_max = local_ref_max.clone()
        g_ref_p95 = local_ref_p95.clone()
        g_ref_mean = local_ref_mean.clone()
        dist.all_reduce(g_min, op=dist.ReduceOp.MIN, group=pg)
        dist.all_reduce(g_max, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_mean, op=dist.ReduceOp.SUM, group=pg)
        dist.all_reduce(g_p50, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_p95, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_p99, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_ref_max, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_ref_p95, op=dist.ReduceOp.MAX, group=pg)
        dist.all_reduce(g_ref_mean, op=dist.ReduceOp.SUM, group=pg)
        g_mean = g_mean / world
        g_ref_mean = g_ref_mean / world
        return {
            'min': g_min.item(),
            'max': g_max.item(),
            'mean': g_mean.item(),
            'p50_worst': g_p50.item(),
            'p95_worst': g_p95.item(),
            'p99_worst': g_p99.item(),
            'ref_max': g_ref_max.item(),
            'ref_p95': g_ref_p95.item(),
            'ref_mean': g_ref_mean.item(),
            'numel_per_rank': n,
            'ranks': world,
        }
    return {
        'min': local_min.item(),
        'max': local_max.item(),
        'mean': local_mean.item(),
        'p50_worst': local_p50.item(),
        'p95_worst': local_p95.item(),
        'p99_worst': local_p99.item(),
        'ref_max': local_ref_max.item(),
        'ref_p95': local_ref_p95.item(),
        'ref_mean': local_ref_mean.item(),
        'numel_per_rank': n,
        'ranks': 1,
    }


def _fmt_diff_stats(s):
    ref_max = s.get('ref_max', 0.0)
    rel_max = (s['max'] / ref_max) if ref_max > 0 else float('inf')
    return (
        f"min={s['min']:.2e} p50={s['p50_worst']:.2e} mean={s['mean']:.2e} "
        f"p95={s['p95_worst']:.2e} p99={s['p99_worst']:.2e} "
        f"max={s['max']:.2e} | ref_max={ref_max:.2e} ref_p95={s.get('ref_p95', 0.0):.2e} "
        f"ref_mean={s.get('ref_mean', 0.0):.2e} rel_max={rel_max:.1%} "
        f"(n_per_rank={s['numel_per_rank']}, ranks={s['ranks']})"
    )


def _print_from_rank0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def _register_logits_capture(mimo_model):
    """Forward hook on the LLM ``output_layer``; captures per-microbatch logits.

    The hook runs on every microbatch forward. ``output`` from
    ``ColumnParallelLinear`` is ``(logits, bias)`` with logits shape
    ``[s, b, v/tp]`` — this rank's per-DP-slot, per-TP-vocab-shard slice
    of the global logits tensor. Cloning so backward doesn't mutate.

    Returns ``(captures, handle)``; caller must ``handle.remove()`` after
    the schedule completes.
    """
    gpt = unwrap_model(mimo_model.language_model)
    captures = []

    def hook(_module, _inputs, output):
        logits = output[0] if isinstance(output, tuple) else output
        captures.append(logits.detach().clone())

    handle = gpt.output_layer.register_forward_hook(hook)
    return captures, handle


def _register_llm_input_capture(mimo_model):
    """Forward pre-hook on the GPT ``decoder``; captures post-bridge hidden states.

    This is the activation entering the transformer block AFTER embedding
    (skipped when MIMO passes ``decoder_input``) AND after the bridge has
    moved the encoder output into the LLM's TP/DP layout. Shape is
    ``[s, b_local, h_full]`` — hidden dim is NOT TP-sharded at this point.

    Comparing dist vs ref at this capture isolates "does the bridge deliver
    mathematically equivalent inputs to the LLM?" from downstream LLM TP
    forward drift. If this oracle passes but ``llm_logits`` fails, the
    divergence is inside the LLM TP forward; if this fails, the bridge
    (fan_in/fan_out vs equal) is not equivalent.
    """
    gpt = unwrap_model(mimo_model.language_model)
    captures = []

    def pre_hook(_module, args, kwargs):
        hidden = kwargs.get('hidden_states', None)
        if hidden is None and args:
            hidden = args[0]
        if hidden is not None:
            captures.append(hidden.detach().clone())

    handle = gpt.decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return captures, handle


def _gather_bs_dp(local_tensor, llm_dp_pg):
    """All-gather ``[s, b, h]`` across LLM DP along the batch dim."""
    dp_size = dist.get_world_size(llm_dp_pg)
    if dp_size <= 1:
        return local_tensor.contiguous()
    contig = local_tensor.contiguous()
    shards = [torch.empty_like(contig) for _ in range(dp_size)]
    dist.all_gather(shards, contig, group=llm_dp_pg)
    return torch.cat(shards, dim=1)


def _assert_llm_input_match(
    ref_captures,
    dist_captures,
    ref_llm_grid,
    dist_llm_grid,
    rtol=1e-3,
    atol=1e-3,
):
    """Post-bridge oracle: hidden states entering the LLM decoder match.

    Hidden dim is not TP-sharded at the decoder input, so only DP-gather
    across the LLM DP group is needed to reconstruct the full-batch tensor.
    """
    assert len(ref_captures) == len(dist_captures), (
        f"Microbatch count mismatch: ref={len(ref_captures)}, "
        f"dist={len(dist_captures)}"
    )
    ref_dp_pg = ref_llm_grid.get_pg("dp")
    dist_dp_pg = dist_llm_grid.get_pg("dp")

    mismatches = []
    for mbs_idx, (ref_local, dist_local) in enumerate(
        zip(ref_captures, dist_captures)
    ):
        ref_full = _gather_bs_dp(ref_local, ref_dp_pg)
        dist_full = _gather_bs_dp(dist_local, dist_dp_pg)
        assert ref_full.shape == dist_full.shape, (
            f"mbs[{mbs_idx}]: gathered llm-input shape mismatch — "
            f"ref={tuple(ref_full.shape)}, dist={tuple(dist_full.shape)}"
        )
        stats = _global_abs_diff_stats(dist_full, ref_full, pg=dist.group.WORLD)
        _print_from_rank0(
            f"[llm-input-diff] mbs[{mbs_idx}] shape={tuple(ref_full.shape)} "
            f"{_fmt_diff_stats(stats)}"
        )
        try:
            torch.testing.assert_close(
                dist_full, ref_full, rtol=rtol, atol=atol
            )
        except AssertionError as e:
            mismatches.append((mbs_idx, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  mbs[{i}]: {msg}" for i, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: llm-input diverged on {len(mismatches)} microbatch(es):\n"
            f"{details}"
        )


def _gather_logits_full_batch(local_logits, llm_tp_pg, llm_dp_pg):
    """All-gather ``[s, b, v/tp]`` across LLM TP (vocab dim) then DP (batch dim).

    Returns ``[s, b * dp_size, v]`` — the full global-batch logits,
    identical on every rank of the LLM grid. Used to compare dist vs ref
    on the same global slots regardless of how TP/DP slices them.
    """
    tp_size = dist.get_world_size(llm_tp_pg)
    dp_size = dist.get_world_size(llm_dp_pg)

    vocab_full = local_logits.contiguous()
    if tp_size > 1:
        shards = [torch.empty_like(vocab_full) for _ in range(tp_size)]
        dist.all_gather(shards, vocab_full, group=llm_tp_pg)
        vocab_full = torch.cat(shards, dim=-1)

    batch_full = vocab_full.contiguous()
    if dp_size > 1:
        shards = [torch.empty_like(batch_full) for _ in range(dp_size)]
        dist.all_gather(shards, batch_full, group=llm_dp_pg)
        batch_full = torch.cat(shards, dim=1)

    return batch_full


def _assert_llm_logits_match(
    ref_captures,
    dist_captures,
    ref_llm_grid,
    dist_llm_grid,
    rtol=1e-2,
    atol=1e-2,
):
    """Logits oracle: TP+DP-gathered full-batch logits match microbatch-by-microbatch.

    Dist and ref share the same global batch on every rank (broadcast from
    rank 0), and with the HyperCommGrid layout both reconstruct global
    batch rows 0..N in the same order after TP+DP all-gather (see
    ``_slice_global_batch_*`` helpers for how the slicing lines up).
    The only numerical difference between the two gathered logits is
    fp32 accumulation order across a different LLM TP shape — hence the
    loose ``rtol=atol=1e-2`` default.
    """
    assert len(ref_captures) == len(dist_captures), (
        f"Microbatch count mismatch: ref={len(ref_captures)}, "
        f"dist={len(dist_captures)}"
    )
    ref_tp_pg = ref_llm_grid.get_pg("tp")
    ref_dp_pg = ref_llm_grid.get_pg("dp")
    dist_tp_pg = dist_llm_grid.get_pg("tp")
    dist_dp_pg = dist_llm_grid.get_pg("dp")

    mismatches = []
    for mbs_idx, (ref_local, dist_local) in enumerate(
        zip(ref_captures, dist_captures)
    ):
        ref_full = _gather_logits_full_batch(ref_local, ref_tp_pg, ref_dp_pg)
        dist_full = _gather_logits_full_batch(dist_local, dist_tp_pg, dist_dp_pg)
        assert ref_full.shape == dist_full.shape, (
            f"mbs[{mbs_idx}]: gathered logits shape mismatch — "
            f"ref={tuple(ref_full.shape)}, dist={tuple(dist_full.shape)}"
        )
        # Gathered full-batch logits are identical on every LLM-grid rank,
        # so stats at rank 0 represent the tensor globally — no reduction
        # needed across other ranks.
        stats = _global_abs_diff_stats(dist_full, ref_full, pg=dist.group.WORLD)
        _print_from_rank0(
            f"[logits-diff] mbs[{mbs_idx}] shape={tuple(ref_full.shape)} "
            f"{_fmt_diff_stats(stats)}"
        )
        try:
            torch.testing.assert_close(
                dist_full, ref_full, rtol=rtol, atol=atol
            )
        except AssertionError as e:
            mismatches.append((mbs_idx, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  mbs[{i}]: {msg}" for i, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: logits diverged on {len(mismatches)} microbatch(es):\n"
            f"{details}"
        )


def _snapshot_first_layer_encoder_grads(mimo_model, encoder_name):
    """Clone ``param.main_grad`` for every ``.layers.0.`` encoder param.

    ``main_grad`` holds the post-DDP-reduction gradient (reduced across
    encoder DP), populated by the backward pass and consumed by
    ``optimizer.step()``. Snapshot between backward and step so the values
    aren't yet zeroed.
    """
    encoder = mimo_model.modality_submodules[encoder_name].module
    snap = {}
    for name, param in encoder.named_parameters():
        if '.layers.0.' not in name:
            continue
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            continue
        snap[name] = grad.detach().clone()
    return snap


def _assert_first_layer_grads_match(ref_snap, dist_snap, rtol=1e-3, atol=1e-3):
    """First-layer encoder grad oracle: shard-to-shard match between ref and dist.

    Ref and dist use identical encoder TP/DP layout, so for every
    ``layers.0.*`` encoder parameter their local shards line up 1:1.
    Under correct grad scaling both main_grads equal the DP=1 gradient,
    so the per-shard values must match within fp32 precision. Tighter
    tolerances than the logits oracle are possible because the encoder
    forward is identical on both sides — only the LLM TP layout differs,
    and that noise enters via the gradient flowing back into the encoder.
    """
    assert set(ref_snap.keys()) == set(dist_snap.keys()), (
        f"First-layer param name mismatch — "
        f"ref-only: {set(ref_snap) - set(dist_snap)}, "
        f"dist-only: {set(dist_snap) - set(ref_snap)}"
    )
    mismatches = []
    for name in sorted(ref_snap):
        ref_g = ref_snap[name]
        dist_g = dist_snap[name]
        assert ref_g.shape == dist_g.shape, (
            f"Param '{name}': grad shape {tuple(ref_g.shape)} != "
            f"{tuple(dist_g.shape)} — caller must match encoder TP."
        )
        # Every rank holds its own TP shard of this param; all-reduce
        # across the full world so rank 0 prints the worst-case drift
        # across all shards.
        stats = _global_abs_diff_stats(dist_g, ref_g, pg=dist.group.WORLD)
        _print_from_rank0(
            f"[grad-diff] {name} shape={tuple(ref_g.shape)} "
            f"{_fmt_diff_stats(stats)}"
        )
        try:
            torch.testing.assert_close(dist_g, ref_g, rtol=rtol, atol=atol)
        except AssertionError as e:
            mismatches.append((name, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  {n}: {msg}" for n, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: {len(mismatches)} first-layer encoder grad(s) "
            f"diverged between dist and ref:\n{details}"
        )


def _assert_encoder_weights_match(
    ref_module, dist_module, rtol=1e-3, atol=1e-3
):
    """Assert every dist encoder shard matches the ref encoder shard.

    Caller is responsible for ensuring ref and dist have the same encoder TP
    layout (same ``enc_tp`` and ``enc_dp``), so each rank's shards line up
    1:1 and can be compared directly. Under correct grad scaling and
    identical initial state, one Adam step yields shard-wise equal post-step
    weights — modulo fp32 TP accumulation-order drift from the LLM TP
    layout differing between the two models.
    """
    ref_params = dict(ref_module.named_parameters())

    mismatches = []
    for name, dist_param in dist_module.named_parameters():
        ref_param = ref_params[name]
        assert ref_param.shape == dist_param.shape, (
            f"Param '{name}': ref.shape={tuple(ref_param.shape)} != "
            f"dist.shape={tuple(dist_param.shape)} — caller must match encoder TP."
        )
        stats = _global_abs_diff_stats(
            dist_param.data, ref_param.data, pg=dist.group.WORLD
        )
        _print_from_rank0(
            f"[weight-diff] {name} shape={tuple(ref_param.shape)} "
            f"{_fmt_diff_stats(stats)}"
        )
        try:
            torch.testing.assert_close(
                dist_param.data, ref_param.data, rtol=rtol, atol=atol
            )
        except AssertionError as e:
            mismatches.append((name, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  {n}: {msg}" for n, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: {len(mismatches)} encoder param(s) diverged between "
            f"heterogeneous-DP dist model and equal-DP reference:\n{details}"
        )


class _BatchIterator:
    """Minimal iterator over a pre-generated list of batches."""

    def __init__(self, batches):
        self.batches = batches
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.batches):
            raise StopIteration
        b = self.batches[self.idx]
        self.idx += 1
        return b


def _run_forward_backward(
    mimo_model,
    batches,
    enc_grid,
    llm_grid,
    encoder_name,
    language_pg,
    micro_batch_size,
    seq_length,
    num_microbatches,
):
    """One forward/backward pass through the mimo schedule."""
    return schedule.forward_backward_no_pipelining(
        forward_step_func=partial(
            forward_step,
            encoder_grid=enc_grid,
            llm_grid=llm_grid,
            encoder_name=encoder_name,
        ),
        data_iterator=_BatchIterator(batches),
        model=[mimo_model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        pg_collection=language_pg,
    )


class TestColocatedGradientScalingCorrectness:
    """Verify heterogeneous-DP encoder grad scaling against an equal-DP reference.

    The critical invariant: with ``calculate_per_token_loss=True`` on both
    sub-model configs, DDP's ``gradient_scaling_factor`` is pinned to
    1.0 and each side's DDP reduction is a pure SUM. The custom
    ``finalize_grads_func`` then divides both encoder and LLM grads by
    ``1/N_global`` (true global valid-token count), so the aggregate
    gradient on every encoder shard equals the DP=1 per-token-mean
    gradient. The reference uses the same encoder TP/DP as dist but with
    ``enc_tp == llm_tp`` and ``enc_dp == llm_dp`` (identity bridge), so
    after one Adam step the dist model's sharded weights match the ref
    model's sharded weights within fp32 precision.

    If the scaling were wrong (e.g., if either DDP applied its default
    ``1/dp_size`` on top of the per-token mean, or if the custom finalize
    used the encoder DP group's sum-of-local-counts instead of the
    globally lifted ``N_global``), the encoder's reduced grad would be
    skewed and post-step weights would diverge — a single optimizer step
    is sufficient to detect.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def setup_method(self):
        # Track MimoModels built by the test so teardown can release any
        # ColocatedBridgeCommunicator subgroups before destroy_all_grids.
        self._mimo_models = []

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        for model in self._mimo_models:
            model.destroy()
        self._mimo_models.clear()
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"),
        reason="Requires PyTorch 2.3+",
    )
    @pytest.mark.parametrize(
        "enc_tp,enc_dp,llm_tp,llm_dp",
        [(2, 4, 4, 2), (4, 2, 2, 4)],
        ids=["fan_in", "fan_out"],
    )
    @pytest.mark.parametrize(
        "mask_pattern", ["uniform", "asymmetric"], ids=["uniform", "asymmetric"]
    )
    @pytest.mark.parametrize("num_microbatches", [1, 4], ids=["mbs1", "mbs4"])
    def test_dist_matches_dp1_reference_post_step_weights(
        self, enc_tp, enc_dp, llm_tp, llm_dp, mask_pattern, num_microbatches
    ):
        """Heterogeneous-DP dist post-step encoder weights match equal-DP reference.

        Builds two MimoModels on every rank:

        * Dist: the heterogeneous TP/DP config under test, with
          ``calculate_per_token_loss=True`` + custom finalize hook that
          pure-SUMs DDP and externally divides by ``N_global``.
        * Ref: equal-DP uniform with ``enc_tp=dist_enc_tp``,
          ``enc_dp=dist_enc_dp``, ``llm_tp=dist_enc_tp``,
          ``llm_dp=dist_enc_dp`` — bridge is
          ``BridgeDirection.EQUAL`` (identity passthrough), and the
          encoder TP sharding matches dist's exactly so shards line up
          1:1 for comparison.

        Both models run the same finalize wiring; both DDPs pure-SUM
        across their own DP group, then divide uniformly by ``N_global``.
        LLM TP differs between the two models, which introduces fp32 TP
        accumulation-order drift in the gradient flowing back to the
        encoder but does not change the per-token-mean invariant that the
        post-step encoder oracle checks.

        Reference weights are copied into the distributed model so both
        start from identical state. One Adam step later, the dist shards
        should match the ref shards within fp32 precision.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        micro_batch_size = 2

        # Global batch spans the larger DP side; dist pre-slices per rank
        # before forward_step (which further slices encoder/LLM side).
        global_batch_size = micro_batch_size * max(enc_dp, llm_dp)

        # Grids: dist is heterogeneous; ref is equal-DP uniform matching
        # dist's encoder so the bridge is identity and encoder shards
        # align 1:1 for direct comparison.
        dist_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        dist_llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)
        ref_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        ref_llm_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        create_all_embedding_groups(
            [dist_enc_grid, dist_llm_grid, ref_enc_grid, ref_llm_grid]
        )

        # Both sub-model TransformerConfigs set calculate_per_token_loss=True
        # (via per_token_loss=True on get_mimo_model), which pins DDP's
        # gradient_scaling_factor to 1.0 — pure SUM across DP on both sides.
        # Under the 3-tuple loss_func + custom finalize_grads_func in
        # _wire_training_hooks, grads are divided uniformly by N_global,
        # which is the true global per-token mean on every shard.
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            bucket_size=10000,
            use_distributed_optimizer=True,
        )

        # Build dist first (heterogeneous TP/DP).
        torch.manual_seed(12345)
        dist_mimo, _, _, dist_language_pg, dist_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=dist_enc_grid,
            llm_grid=dist_llm_grid,
            hidden_size=hidden_size,
            num_layers=2,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
            bf16=False,
            bias=False,
            dropout=False,
            per_token_loss=True,
        )
        dist_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(dist_mimo)

        # Reference with equal-DP uniform (enc_tp == llm_tp, enc_dp == llm_dp).
        torch.manual_seed(12345)
        ref_mimo, _, _, ref_language_pg, ref_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=ref_enc_grid,
            llm_grid=ref_llm_grid,
            hidden_size=hidden_size,
            num_layers=2,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
            bf16=False,
            bias=False,
            dropout=False,
            per_token_loss=True,
        )
        ref_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(ref_mimo)

        # Force identical initial state: encoder shards already match
        # (same TP layout), so the helper copies shard-to-shard. LLM
        # shards don't match (ref_llm_tp=enc_tp, dist_llm_tp=llm_tp), so
        # the helper all-gathers ref's shards across ref's TP group and
        # re-slices for dist's TP group.
        _copy_ref_params_to_dist(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            ref_enc_grid.get_pg("tp"),
            dist_enc_grid.get_pg("tp"),
        )
        _copy_ref_params_to_dist(
            ref_mimo.language_model.module,
            dist_mimo.language_model.module,
            ref_llm_grid.get_pg("tp"),
            dist_llm_grid.get_pg("tp"),
        )

        _wire_training_hooks(dist_mimo, dist_language_pg, dist_vision_pg)
        _wire_training_hooks(ref_mimo, ref_language_pg, ref_vision_pg)

        # Distributed optimizers snapshot current param.data into fp32 master
        # weights at __init__, so both must be built AFTER the ref-to-dist
        # param copy above.
        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            bf16=False,
            use_distributed_optimizer=True,
        )
        dist_optimizer = get_mimo_optimizer(dist_mimo, opt_config)
        ref_optimizer = get_mimo_optimizer(ref_mimo, opt_config)

        # Data: one deterministic global batch, identical on every rank.
        torch.manual_seed(99999)
        global_batches = _generate_and_broadcast_global_batches(
            global_mbs=global_batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_name=encoder_name,
            num_batches=num_microbatches,
            mask_pattern=mask_pattern,
        )
        dist_batches = [
            _slice_global_batch_for_dist(b, dist_enc_grid, dist_llm_grid)
            for b in global_batches
        ]
        # Ref is uniform (enc_dp == llm_dp), so _slice_global_batch_for_dist
        # returns the full batch; slice explicitly by enc_dp so each rank
        # sees the same per-rank batch size as dist's encoder does.
        ref_batches = [
            _slice_global_batch_by_dp(b, ref_enc_grid.get_pg("dp"))
            for b in global_batches
        ]
        ref_per_rank_batch_size = global_batch_size // enc_dp

        # Logits capture: hook fires on every microbatch forward.
        # Registered before forward/backward, removed right after so the
        # hook doesn't leak across the second model's run.
        dist_logits, dist_logits_hook = _register_logits_capture(dist_mimo)
        ref_logits, ref_logits_hook = _register_logits_capture(ref_mimo)
        dist_llm_input, dist_input_hook = _register_llm_input_capture(dist_mimo)
        ref_llm_input, ref_input_hook = _register_llm_input_capture(ref_mimo)

        try:
            # One optimizer step on dist (heterogeneous forward_step slicing).
            dist_optimizer.zero_grad()
            _run_forward_backward(
                mimo_model=dist_mimo,
                batches=dist_batches,
                enc_grid=dist_enc_grid,
                llm_grid=dist_llm_grid,
                encoder_name=encoder_name,
                language_pg=dist_language_pg,
                micro_batch_size=micro_batch_size,
                seq_length=seq_length,
                num_microbatches=num_microbatches,
            )
            # Snapshot encoder first-layer grads AFTER backward and BEFORE
            # optimizer.step() consumes/zeros the grad buffer.
            dist_first_layer_grads = _snapshot_first_layer_encoder_grads(
                dist_mimo, encoder_name
            )
            dist_success, dist_grad_norm, _ = dist_optimizer.step()
            assert dist_success, "Dist optimizer step failed"
            assert dist_grad_norm is not None and dist_grad_norm > 0, (
                f"Dist grad_norm={dist_grad_norm} — encoder grads may have been "
                "silently zeroed by wrong scaling"
            )

            # One optimizer step on ref (enc_dp == llm_dp → forward_step skips slicing).
            ref_optimizer.zero_grad()
            _run_forward_backward(
                mimo_model=ref_mimo,
                batches=ref_batches,
                enc_grid=ref_enc_grid,
                llm_grid=ref_llm_grid,
                encoder_name=encoder_name,
                language_pg=ref_language_pg,
                micro_batch_size=ref_per_rank_batch_size,
                seq_length=seq_length,
                num_microbatches=num_microbatches,
            )
            ref_first_layer_grads = _snapshot_first_layer_encoder_grads(
                ref_mimo, encoder_name
            )
            ref_success, ref_grad_norm, _ = ref_optimizer.step()
            assert ref_success, "Ref optimizer step failed"
            assert ref_grad_norm is not None and ref_grad_norm > 0, (
                f"Ref grad_norm={ref_grad_norm}"
            )
        finally:
            dist_logits_hook.remove()
            ref_logits_hook.remove()
            dist_input_hook.remove()
            ref_input_hook.remove()

        # Run all three oracles regardless of individual failures so the
        # diff-stats print covers every layer. Order: encoder weights /
        # first-layer grads first (tightest — same encoder TP/DP layout
        # → shards align 1:1), then LLM logits last (loosest — different
        # LLM TP layout drives fp32 accumulation drift). Each oracle
        # printed its own min/mean/p95/p99/max before its assertion ran,
        # so the user sees the full drift distribution for every test.
        failures = []

        try:
            _assert_encoder_weights_match(
                ref_mimo.modality_submodules[encoder_name].module,
                dist_mimo.modality_submodules[encoder_name].module,
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as e:
            failures.append(('encoder_weights', str(e)))

        try:
            _assert_first_layer_grads_match(
                ref_first_layer_grads,
                dist_first_layer_grads,
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as e:
            failures.append(('first_layer_grads', str(e)))

        try:
            _assert_llm_input_match(
                ref_llm_input,
                dist_llm_input,
                ref_llm_grid,
                dist_llm_grid,
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as e:
            failures.append(('llm_input', str(e)))

        try:
            _assert_llm_logits_match(
                ref_logits,
                dist_logits,
                ref_llm_grid,
                dist_llm_grid,
                rtol=1e-2,
                atol=1e-2,
            )
        except AssertionError as e:
            failures.append(('llm_logits', str(e)))

        if failures:
            summary = "\n\n".join(
                f"== {oracle} ==\n{msg}" for oracle, msg in failures
            )
            raise AssertionError(
                f"{len(failures)} oracle(s) failed:\n{summary}"
            )
