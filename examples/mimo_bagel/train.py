# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
This script provides a basic training loop for MIMO models.
"""

import os
import sys
from functools import partial
from typing import Any, Dict, Iterator

import torch

# Add the parent directory to the path to import from megatron
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
# )
# from data.mock import (
#     train_valid_test_datasets_provider as mock_train_valid_test_datasets_provider,
# )
from examples.mimo_bagel.data.hf_dataloader import bagel_dataloader_provider
# from examples.mimo_bagel.data.energon_dataloader_provider import bagel_energon_dataloader_provider
from examples.mimo_bagel.utils.data_helpers import bagel_packed_batch_to_mimo_batch
from examples.mimo_bagel.utils.model_helpers import get_pg_collection
# from model_providers.mock import model_provider_mock_vlm_single_encoder
from model_providers.bagel import model_provider_bagel
from utils.data_helpers import broadcast_nested_data_batch, shard_data_for_cp, get_packed_seq_params

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

_MODEL_PROVIDERS = {
    # "mock": model_provider_mock_vlm_single_encoder,
    "bagel": model_provider_bagel,
    "bagel_mot": partial(model_provider_bagel, decoder_layer_module="Qwen2MoTDecoderLayer"),
}

_DATASET_PROVIDERS = {
    # "mock": mock_train_valid_test_datasets_provider,
    "bagel": bagel_dataloader_provider,
    # "bagel_energon": bagel_energon_dataloader_provider,
}

def add_mimo_args(parser):
    """Add MIMO-specific arguments to the parser."""
    group = parser.add_argument_group('MIMO', 'MIMO specific arguments')

    # MIMO-specific parameters
    group.add_argument('--dataset-provider', type=str, default='bagel_energon', help='Dataset provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm, bagel, bagel_energon]')
    group.add_argument('--model-provider', type=str, default='bagel_mot', help='Model provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm, bagel]')
    group.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint to load')

    # mock dataloader related args
    # can control mock samples with total seq length and image seq length
    group.add_argument('--image-size', type=int, default=224, help='Image size for vision encoder')
    group.add_argument('--total-seq-length', type=int, default=512, help='Total sequence length')
    group.add_argument('--pad-token-id', type=int, default=0, help='Padding token ID')
    group.add_argument('--image-token-id', type=int, default=32000, help='Image token ID')
    group.add_argument(
        '--image-seq-length', type=int, default=197, help='Number of image tokens to pad'
    )
    group.add_argument(
        '--audio-encoder-model', type=str, default=None, help='Audio encoder model name'
    )
    group.add_argument(
        '--hf-assign-unused-tokens', type=str, nargs='+', default=None,
                       help='Assigning unused tokens to special tokens. Example: '
                       '--hf-assign-unused-tokens "<audio>,32002" "<video>,32003"'
    )
    # checkpoint related args
    group.add_argument('--language-model-checkpoint', type=str, default=None, help='Path to language model checkpoint to load')
    # energon dataloader related args
    group.add_argument('--packing-buffer-size', type=int, default=None, help='Packing buffer size when using sequence packing')

    # Bagel-specific args
    group.add_argument('--text-cond-dropout-prob', type=float, default=0.1, help='Text conditional dropout probability')
    group.add_argument('--vit-cond-dropout-prob', type=float, default=0.4, help='VIT conditional dropout probability')
    group.add_argument('--vae-cond-dropout-prob', type=float, default=0.3, help='VAE conditional dropout probability')
    # group.add_argument('--vae-image-downsample', type=int, default=16, help='VAE image downsample factor')
    group.add_argument('--max-latent-size', type=int, default=32, help='Maximum latent grid size (patches per side) for the VAE latent tensor.')
    group.add_argument('--vit-patch-size', type=int, default=14, help='VIT patch size')
    group.add_argument('--max-num-patch-per-side', type=int, default=70, help='Max number of patches per side')
    group.add_argument('--max-num-tokens-per-sample', type=int, default=16384, help='Max number of tokens per sample')
    group.add_argument('--max-num-tokens', type=int, default=36864, help='Max number of tokens')
    group.add_argument('--prefer-buffer-before', type=int, default=16384, help='Prefer buffer before this number of tokens')
    group.add_argument('--interpolate-pos', action='store_true', help='Whether to interpolate position embeddings')
    group.add_argument('--use-flex-attention', action='store_true', help='Whether to use flex attention')
    group.add_argument('--mot-stream-overlap', action='store_true',
                       help='Run the und/gen MLP branches of the MoT layer on two side '
                            'CUDA streams during training. Off by default. Backward '
                            'inherits the side streams via autograd stream tracking. '
                            'Worth enabling when the gen branch is a MoE layer (its '
                            'all-to-all latency can hide under und compute) or when '
                            'TP/CP collectives are running on a separate comm stream.')

    # Bagel loss-balance knobs (match original BAGEL `--ce-weight` / `--mse-weight`).
    # See plan/M4_support_for_bagel.md "Loss-aggregation mismatch" for the
    # detailed analysis of why these are needed.
    group.add_argument('--ce-weight', type=float, default=1.0,
                       help='Weight applied to the cross-entropy loss term '
                            '(matches original BAGEL --ce-weight, default 1.0).')
    group.add_argument('--mse-weight', type=float, default=1.0,
                       help='Weight applied to the diffusion MSE loss term '
                            '(matches original BAGEL --mse-weight, default 1.0).')
    group.add_argument('--ce-loss-reweighting', action='store_true',
                       help='Reweight CE by ce_loss_weights, matching original BAGEL '
                            '--ce-loss-reweighting. Disabled by default.')

    # Bagel-specific args
    group.add_argument('--llm-path', type=str, default=None, help='Path to LLM checkpoint to load')
    group.add_argument('--vit-path', type=str, default=None, help='Path to VIT checkpoint to load')
    group.add_argument(
        '--recompute-vit',
        action='store_true',
        help='Enable full-ViT activation checkpointing: wraps the entire '
             'SigLIP encoder (all transformer layers) as a single checkpoint '
             'block so only the encoder input is retained during forward. '
             'Near-zero ViT forward activation memory at the cost of one '
             'extra full ViT forward during backward. Recommended when '
             'per-sample ViT patch count varies significantly across DP ranks.',
    )

    # Language model backend selection
    group.add_argument('--language-use-mcore', action='store_true',
                       help='Use Megatron Core GPTModel-based language model instead of HuggingFace. '
                            'Default is False (use HuggingFace BagelLLMHuggingFaceModel)')


    #diffusion related args
    group.add_argument('--vae-path', type=str, default=None, help='Path to vae checkpoint')
    group.add_argument('--latent-patch-size', type=int, default=2, help='Spatial size (in VAE pixels) covered by each latent patch.')
    group.add_argument('--timestep-shift', type=float, default=1.0, help='Timestep shift for the diffusion model')
    return parser


def get_batch(data_iterator: Iterator[Dict[str, Any]], pg_collection):
    """Generate a batch for MIMO model training.

    Args:
        data_iterator: Iterator over the dataset
        pg_collection: ProcessGroupCollection providing tp/cp/dp groups.

    Returns:
        tuple: Batch data for model training
    """
    args = get_args()

    # Pipeline parallelism: Phase C wires pre_process/post_process gating in
    # BagelMimoModel and BagelMCoreModel.  PP stages within the same DP coord
    # MUST consume identical data so that packed_seq_params (computed locally
    # on each stage from the data) agrees across stages — otherwise the
    # CP-aware FlexAttention reshape fails on later stages.
    #
    # The PackedDataset internally calls np.random.randn() (see
    # bagel/data/dataset_base.py:429) for diffusion timesteps. With PP=1,
    # let the iterator consume RNG state naturally, matching the HF BAGEL
    # loop. With PP>1, Megatron seeds numpy with `seed + 100*pp_rank`
    # (training/initialize.py:415), so PP siblings would draw different
    # timesteps. In that case only, re-seed numpy/random by DP rank around
    # the iterator step so PP siblings sample identical data.

    tp_group = pg_collection.tp
    cp_group = pg_collection.cp
    tp_rank = torch.distributed.get_rank(tp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    cp_size = torch.distributed.get_world_size(cp_group)
    pp_group = pg_collection.pp
    pp_size = torch.distributed.get_world_size(pp_group)

    # Broadcast data - only get data on tensor parallel rank 0
    # data iterator is None on other tp ranks
    # TP Rank-0 reads next batch.
    if tp_rank == 0 and cp_rank == 0:
        try:
            if pp_size == 1:
                data = next(data_iterator)
            else:
                # PP-determinism: scope numpy / random state to a
                # dp-rank-keyed seed for the duration of this iterator step.
                import numpy as _np
                import random as _random

                _global_step = getattr(args, 'curr_iteration', 0) or 0
                _data_seed = (
                    getattr(args, 'seed', 1234) + 7919 * _global_step
                    + 31 * torch.distributed.get_rank(pg_collection.dp)
                )
                _np_state = _np.random.get_state()
                _py_state = _random.getstate()
                _np.random.seed(_data_seed)
                _random.seed(_data_seed)
                try:
                    data = next(data_iterator)
                finally:
                    _np.random.set_state(_np_state)
                    _random.setstate(_py_state)
            has_data = torch.tensor([1], dtype=torch.uint8, device='cuda')
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device='cuda')
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device='cuda')
        data = None
        diffusion_wrapper = getattr(args, 'diffusion_wrapper', None)
        if diffusion_wrapper is not None:
            diffusion_wrapper.remove_vae()

    tp_src = torch.distributed.get_process_group_ranks(tp_group)[0]
    torch.distributed.broadcast(has_data, tp_src, group=tp_group)

    # Propagate has_data across the CP group so that CP rank 1+ know whether the
    # iterator is exhausted.  With TP=1 every rank is its own TP group, meaning
    # the TP broadcast above only reaches the rank itself; CP rank 1 would otherwise
    # have an uninitialised has_data value and might return None prematurely.
    if cp_size > 1:
        cp_src = torch.distributed.get_process_group_ranks(cp_group)[0]
        torch.distributed.broadcast(has_data, cp_src, group=cp_group)

    if has_data.item() == 0:
        # iterator exhausted on all ranks
        # we need this to avoid race condition when first tp rank hits StopIteration
        return None

    # MiMo forward pass expects
    # input_ids: torch.Tensor,
    # position_ids: Optional[torch.Tensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # loss_mask: Optional[torch.Tensor] = None,
    # labels: Optional[torch.Tensor] = None,
    # modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    # modality_seq_lengths: Optional[Dict[str, torch.Tensor]] = None,

    # For the modality inputs, the keys can be arbitrary
    # so we do a broadcast of the schema followed by a broadcast of the actual data
    # check broadcast_nested_data_batch for more details

    # Check if this is bagel dataset (PackedDataset or energon-packed format)
    if args.dataset_provider in ('bagel', 'bagel_energon') and data is not None:
        # Convert bagel packed batch to MIMO format
        diffusion_wrapper = getattr(args, 'diffusion_wrapper', None)
        if diffusion_wrapper is not None:
            diffusion_wrapper.cuda()
        # Phase E2: VAE re-run on first/last PP stage only.
        # The diffusion modality submodule (consumes ``latents``) only runs
        # on the first PP stage; MSE loss (consumes ``vis_gen_target``)
        # only fires on the last PP stage. Middle stages (PP>=3) discard
        # both, so VAE encode + add_noise are pure waste there. Both calls
        # are deterministic (no per-rank randomness; ``np.random.randn``
        # for noise seeds the SAME draws across PP siblings thanks to the
        # numpy state scoping above), so first and last stage produce
        # bit-identical results — no broadcast needed. At PP<=2 every rank
        # is first or last, so behaviour is unchanged.
        from megatron.core import parallel_state as _mpu
        run_vae = (
            _mpu.is_pipeline_first_stage(ignore_virtual=True)
            or _mpu.is_pipeline_last_stage(ignore_virtual=True)
        )
        data = bagel_packed_batch_to_mimo_batch(
            data, diffusion_wrapper=diffusion_wrapper, run_vae_encode=run_vae,
        )

    if cp_rank == 0:
        data = broadcast_nested_data_batch(data, group='tp')

    if cp_size > 1:
        data = broadcast_nested_data_batch(data, group='cp')

    data = get_packed_seq_params(data)

    diffusion_wrapper = getattr(args, 'diffusion_wrapper', None)
    vae_dim = None
    if diffusion_wrapper is not None:
        vae_dim = diffusion_wrapper.vae_params.z_channels * diffusion_wrapper.latent_patch_size ** 2

    # shard_data_for_cp is required even when cp_size==1: it produces the dense
    # gen_loss_mask/vis_gen_target form the model expects and drops mse_loss_indexes.
    data = shard_data_for_cp(data, cp_group=cp_group, vae_dim=vae_dim)
    return data


def loss_func(ce_loss, loss_mask, mse_loss, mse_loss_mask):
    """BAGEL-style loss aggregation for MIMO model training.

    Mirrors the original BAGEL training loop in
    ``bagel-package/bagel/train/pretrain_unified_navit.py:695-727``: each
    loss term is normalised by its global token / weight count and scaled
    by a CLI weight (``--ce-weight`` / ``--mse-weight``). Returning
    ``num_tokens=1`` bypasses Megatron's auto-divide in
    ``schedules.py:262-266`` because we have already done the BAGEL-style
    normalisation here; Megatron's ``/num_microbatches`` (gradient
    accumulation scaling) is preserved.

    Reported metrics use Megatron's two-value ``[numerator, denominator]``
    convention so training.py all-reduces them across DP/CP ranks and logs
    a global average. This matches the stable 12691662 experiment where
    ``ce`` and ``lm loss`` are the same scalar, ``mse`` and ``mse loss`` are
    the same scalar, and ``loss = ce * ce_weight + mse * mse_weight``.

    Args:
        ce_loss: per-rank-local unweighted CE values, shape [N_local_valid] or None.
        loss_mask: per-rank-local CE loss-mask weights, shape [actual_lund].
        mse_loss: per-rank-local MSE per-token×channel values, shape
            [Lgen_local, vae_dim] or None.
        mse_loss_mask: per-rank-local MSE token mask, shape [Lgen_local].

    Returns:
        tuple: (loss, num_tokens=1, metrics_dict)
    """
    args = get_args()
    ce_weight  = getattr(args, 'ce_weight',  1.0)
    mse_weight = getattr(args, 'mse_weight', 1.0)
    ce_loss_reweighting = getattr(args, 'ce_loss_reweighting', False)

    # Loss aggregation runs across the data-parallel + context-parallel group
    # (TP ranks compute identical losses, so they're not in the reduction).
    # Use mpu directly here since loss_func has no model handle; this is
    # equivalent to ``ProcessGroupCollection.use_mpu_process_groups().dp_cp``.
    from megatron.core import parallel_state as mpu
    dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    dp_cp_world_size = torch.distributed.get_world_size(dp_cp_group)

    loss = torch.tensor(0.0, device='cuda')

    # ── CE term: match HF BAGEL's ce_loss_reweighting switch ──────────────
    ce_sum_local = torch.tensor(0.0, device='cuda')
    ce_denominator_local = torch.tensor(0.0, device='cuda')
    ce_term = None
    if ce_loss is not None:
        ce_loss = ce_loss.float()
        ce_weights = loss_mask.view(-1)[loss_mask.view(-1) > 0].to(torch.float)
        if ce_loss_reweighting:
            ce_sum_local = (ce_loss * ce_weights).sum()
            ce_denominator_local = ce_weights.sum()
        else:
            ce_sum_local = ce_loss.sum()
            ce_denominator_local = torch.tensor(ce_loss.numel(), dtype=torch.float, device='cuda')

        ce_denominator_global = ce_denominator_local.clone()
        torch.distributed.all_reduce(
            ce_denominator_global,
            op=torch.distributed.ReduceOp.SUM,
            group=dp_cp_group,
        )
        # ce_term ≈ global HF-style CE × dp_cp_world_size.
        # FSDP mean-reduce of grads will divide by world_size, leaving the
        # HF-style average as the effective gradient signal.
        ce_term = ce_sum_local * dp_cp_world_size / torch.clamp(ce_denominator_global, min=1.0)
        loss = loss + ce_term * ce_weight

    # ── MSE term: BAGEL-style global per-token avg MSE × W ─────────────────
    mse_sum_local = torch.tensor(0.0, device='cuda')
    mse_tokens_local = torch.tensor(0.0, device='cuda')
    mse_term = None
    if mse_loss is not None:
        mse_sum_local = mse_loss.float().mean(dim=-1).sum()
        mse_tokens_local = mse_loss_mask.sum().to(torch.float)

        mse_tokens_global = mse_tokens_local.clone()
        torch.distributed.all_reduce(
            mse_tokens_global,
            op=torch.distributed.ReduceOp.SUM,
            group=dp_cp_group,
        )
        mse_term = mse_sum_local * dp_cp_world_size / torch.clamp(mse_tokens_global, min=1.0)
        loss = loss + mse_term * mse_weight

    # Reporting: use Megatron's two-value metric path so logs/W&B show global
    # DP/CP averages, matching the stable 12691662 experiment.
    metrics = {}
    if ce_term is not None:
        ce_report = torch.cat([ce_sum_local.detach().view(1), ce_denominator_local.detach().view(1)])
        metrics['ce'] = ce_report

    if mse_term is not None:
        mse_report = torch.cat([mse_sum_local.detach().view(1), mse_tokens_local.detach().view(1)])
        metrics['mse'] = mse_report

    metrics['loss'] = torch.cat(
        [
            loss.detach().view(1),
            torch.ones(1, dtype=loss.dtype, device=loss.device),
        ]
    )

    # num_tokens=1 makes Megatron's `output_tensor /= clamp(num_tokens, min=1)`
    # a no-op (we have already normalised). The /num_microbatches divide that
    # follows is the gradient-accumulation scaling — keep it.
    return (
        loss,
        torch.tensor(1, dtype=torch.int, device='cuda'),
        metrics,
    )

    # # Standard MIMO output format (tensor)
    # losses = output_tensor.float()

    # loss_mask = loss_mask.contiguous().view(-1).float()

    # total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    # total_loss = torch.sum(losses.view(-1) * loss_mask)
    # reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    # return (total_loss, total_tokens, {'lm loss': reporting_loss})


_mem_log_done = False

# --------------------------------------------------------------------------
# Per-iteration memory profiling (ranks 0 and 24, first 3 iterations).
# Detects inter-iteration static growth (e.g. Adam lazy init) and captures
# a full memory snapshot on OOM so we can identify every live tensor.
# --------------------------------------------------------------------------
_fwd_call_count = 0
# Ranks 0,1,2,3 are the 4 CP ranks sharing dp=0 (same sample, different CP chunk).
# Rank 24 is worst-case (cp=0, dp=6) that OOMs on iter 2.
_MEM_TARGET_RANKS = {0, 1, 2, 3, 24}
_MEM_SNAPSHOT_RANKS = {0, 1, 2, 3, 24}
_MEM_MAX_ITERS = 3
# Micro-steps per iteration = global_batch_size / micro_batch_size / DP_size.
# Derived dynamically on first call from args to handle any DP config correctly.
_MICRO_STEPS_PER_ITER = None  # populated lazily; fallback = 1 if detection fails.

_WORKSPACE_PATH = '/workspace/megatron-lm-bagel'
_snapshot_dumped_ranks: set = set()


def _get_micro_steps_per_iter(pg_collection):
    """Compute number of micro-steps per iteration on this DP rank.

    micro_steps_per_iter = global_batch_size / (micro_batch_size × DP_size)
    With 32 GPUs, TP=1, PP=1, CP=4: non-expert DP = 32/4 = 8.
    With --global-batch-size 8 --micro-batch-size 1: 8 / (1 × 8) = 1.
    """
    global _MICRO_STEPS_PER_ITER
    if _MICRO_STEPS_PER_ITER is not None:
        return _MICRO_STEPS_PER_ITER
    try:
        args = get_args()
        gbs = args.global_batch_size
        mbs = args.micro_batch_size
        dp_size = torch.distributed.get_world_size(pg_collection.dp)
        _MICRO_STEPS_PER_ITER = max(1, gbs // (mbs * dp_size))
    except Exception:
        _MICRO_STEPS_PER_ITER = 1
    return _MICRO_STEPS_PER_ITER


def _oom_snapshot_hook(exc_type, exc_val, exc_tb):
    """sys.excepthook: dump a memory snapshot to disk on unhandled OOM."""
    if issubclass(exc_type, torch.cuda.OutOfMemoryError):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        path = f'{_WORKSPACE_PATH}/logs/oom_snapshot_rank{rank}.pkl'
        try:
            torch.cuda.memory._dump_snapshot(path)
            print(f"[MEM_SNAPSHOT rank{rank}] Dumped to {path}", flush=True)
        except Exception as e:
            print(f"[MEM_SNAPSHOT rank{rank}] Dump failed: {e}", flush=True)
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _oom_snapshot_hook


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.

    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance

    Returns:
        tuple: (output_tensor, loss_function)
    """
    global _mem_log_done, _fwd_call_count
    rank = torch.distributed.get_rank()

    pg_collection = get_pg_collection(model)

    _fwd_call_count += 1
    micro_steps = _get_micro_steps_per_iter(pg_collection)
    iter_num = (_fwd_call_count - 1) // micro_steps + 1
    is_first_micro = ((_fwd_call_count - 1) % micro_steps == 0)

    # Enable memory-history recording on all snapshot-target ranks from the
    # very first forward so snapshots (dumped either on OOM or at a scheduled
    # point) contain Python call-stack info for every block.
    if rank in _MEM_SNAPSHOT_RANKS and _fwd_call_count == 1:
        torch.cuda.memory._record_memory_history(
            max_entries=1_000_000,
            context='all',
            stacks='python',
        )

    # At the first micro-step of iter N (N>=2), read peak-since-last-reset
    # *before* we reset it. This captures the previous iteration's overall
    # peak (forward + backward + optimizer step) — the metric Megatron's
    # built-in per-rank log shows only for ranks 0-3.
    if (
        rank in _MEM_TARGET_RANKS
        and is_first_micro
        and 2 <= iter_num <= _MEM_MAX_ITERS + 1
    ):
        prev_iter = iter_num - 1
        prev_peak = torch.cuda.max_memory_allocated() / 1e9
        prev_cur = torch.cuda.memory_allocated() / 1e9
        print(
            f"[MEM2 rank{rank} iter{prev_iter}] Overall peak (fwd+bwd+opt): "
            f"{prev_peak:.2f} GB  (end-of-iter allocated: {prev_cur:.2f} GB)",
            flush=True,
        )

    # Per-iteration memory: log before-forward static allocation and peak.
    if rank in _MEM_TARGET_RANKS and is_first_micro and iter_num <= _MEM_MAX_ITERS:
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(
            f"[MEM2 rank{rank} iter{iter_num}] Before forward: "
            f"allocated={alloc:.2f} GB  reserved={res:.2f} GB",
            flush=True,
        )

        # (Snapshot dump for all ranks happens at iter-1 "after forward peak"
        # below — NOT here. Deferring to iter 2 would mean ranks 0..3 never
        # dump because rank 24 OOMs first and kills torchrun.)

    data_batch = get_batch(data_iterator, pg_collection)

    # One-time memory snapshot: static memory before first forward (rank 0)
    if not _mem_log_done and rank == 0:
        torch.cuda.reset_peak_memory_stats()
        static_alloc = torch.cuda.memory_allocated() / 1e9
        static_reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"[MEM] Before 1st forward: "
            f"allocated={static_alloc:.2f} GB  reserved={static_reserved:.2f} GB",
            flush=True,
        )

    # PP awareness: at PP>1, BagelMimoModel returns the raw hidden-state tensor
    # on non-last stages and a 4-tuple (ce, mse, mse_loss_mask, loss_mask) on
    # the last stage.  Megatron's pipeline schedule only invokes loss_func on
    # the last stage, so middle-stage loss_func is never called — return a
    # placeholder closure to keep the (output_tensor, loss_func) contract.
    output_or_tuple = model(**data_batch)
    if isinstance(output_or_tuple, tuple):
        ce_loss, mse_loss, mse_loss_mask, loss_mask = output_or_tuple
        last_stage = True
    else:
        # Middle / first PP stage: send hidden activation to next stage.
        ce_loss = output_or_tuple
        mse_loss = mse_loss_mask = None
        loss_mask = data_batch.get('loss_mask')
        last_stage = False

    # One-time memory snapshot: peak activation memory after first forward
    if not _mem_log_done and rank == 0:
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(
            f"[MEM] After 1st forward:  "
            f"peak_allocated={peak_alloc:.2f} GB  peak_reserved={peak_reserved:.2f} GB",
            flush=True,
        )
        _mem_log_done = True

    # Per-iteration memory: log peak after forward pass.
    if rank in _MEM_TARGET_RANKS and is_first_micro and iter_num <= _MEM_MAX_ITERS:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[MEM2 rank{rank} iter{iter_num}] After forward peak: {peak:.2f} GB",
            flush=True,
        )

        # Dump snapshot at iter 1's first micro-step forward peak for every
        # rank in _MEM_SNAPSHOT_RANKS. This is a safe point: before backward,
        # before any rank can OOM, so ranks 0..3 will all successfully dump
        # apples-to-apples with rank 24 for forward-activation comparison.
        if (
            iter_num == 1
            and rank in _MEM_SNAPSHOT_RANKS
            and rank not in _snapshot_dumped_ranks
        ):
            path = f'{_WORKSPACE_PATH}/logs/snapshot_rank{rank}_iter1_fwd_peak.pkl'
            try:
                torch.cuda.memory._dump_snapshot(path)
                _snapshot_dumped_ranks.add(rank)
                print(
                    f"[MEM_SNAPSHOT rank{rank}] Scheduled dump at iter1 "
                    f"forward-peak → {path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[MEM_SNAPSHOT rank{rank}] Scheduled dump failed: {e}",
                    flush=True,
                )

    # Per-step MFU tracking (timing, DP all-reduce, and logging all handled
    # inside the tracker — no state kept in this module).
    tracker = getattr(get_args(), 'mfu_tracker', None)
    if tracker is not None:
        tracker.step_and_log(data_batch)

    # Per-component CUDA timer is optional; some benchmark checkouts do not
    # include utils/comp_timer.py.
    try:
        from examples.mimo_bagel.utils.comp_timer import get_comp_timer
    except ModuleNotFoundError:
        get_comp_timer = None
    if get_comp_timer is not None:
        get_comp_timer().log_iter(rank=rank)

    # Return output and loss function
    return ce_loss, partial(loss_func,  loss_mask=loss_mask, mse_loss=mse_loss, mse_loss_mask=mse_loss_mask)


def train_valid_test_datasets_provider(*provider_args, vp_stage=None, **provider_kwargs):
    """Dataset provider for MIMO model training.

    Args:
        *provider_args: Additional arguments for the dataset provider
        vp_stage: virtual pipeline stage (required by megatron when VP is enabled;
            our underlying dataset is not VP-aware so we just discard it).
        **provider_kwargs: Additional keyword arguments for the dataset provider
    """
    runtime_args = get_args()
    try:
        dataset_provider = _DATASET_PROVIDERS[runtime_args.dataset_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset provider '{runtime_args.dataset_provider}'. "
            f"Available providers: {list(_DATASET_PROVIDERS.keys())}"
        ) from e

    return dataset_provider(*provider_args, **provider_kwargs)

def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    config=None,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
    pg_collection=None,
    vp_stage=None,
):
    """Model provider for MIMO model training.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder to the model (not supported yet)(default: True)
        add_decoder: Whether to add a decoder to the model (not supported yet)(default: True)
        image_special_token_id: Special token ID for the image modality (default: 32000)
        audio_special_token_id: Special token ID for the audio modality (default: 32002)
    """
    runtime_args = get_args()

    try:
        builder_fn = _MODEL_PROVIDERS[runtime_args.model_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model provider '{runtime_args.model_provider}'. "
            f"Available providers: {list(_MODEL_PROVIDERS.keys())}"
        ) from e

    if runtime_args.model_provider == "bagel":
        language_use_mcore = getattr(runtime_args, 'language_use_mcore', False)
        print(f"Using {'Megatron Core GPTModel-based language model' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "model_path": runtime_args.model_path,
            "language_use_mcore": language_use_mcore,
            "llm_path": runtime_args.llm_path,
            "vit_path": runtime_args.vit_path,
        }
    elif runtime_args.model_provider == "bagel_mot":
        language_use_mcore = getattr(runtime_args, 'language_use_mcore', False)
        print(f"Using {'Megatron Core GPTModel-based language model' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "model_path": runtime_args.model_path,
            "language_use_mcore": language_use_mcore,
            "decoder_layer_module": "Qwen2MoTDecoderLayer",
            "llm_path": runtime_args.llm_path,
            "vit_path": runtime_args.vit_path,
        }
    # elif runtime_args.model_provider == "mock":
    #     kwargs = {
    #         "special_token_id": image_special_token_id,
    #     }
    else:
        raise ValueError(f"Unknown model provider: {runtime_args.model_provider}. Must be one of ['llava_vlm', 'llava_avlm', 'bagel', 'bagel_mot', 'mock']")

    return builder_fn(
        pre_process,
        post_process,
        add_encoder,
        add_decoder,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
        **kwargs,
    )


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True
    import megatron
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.models.bagel.transformer_mot_layer import MoTTransformerLayer
    from examples.mimo_bagel.vision.hf_bagel_vision_encoder import HFBagelVisionEncoderWrapper
    # from megatron.core.models.vision.multimodal_projector import MultimodalProjector

    # Subclass so we can set default fsdp_unit_modules without using partial().
    # training.py uses isinstance(module, megatron_FSDP), so megatron_FSDP must be a class.
    #
    # HFBagelVisionEncoderWrapper and MultimodalProjector are added as fsdp_unit_modules to
    # prevent FSDP bucket misalignment.  When LLM hidden_size > ViT hidden_size (e.g.
    # Qwen3-30B: 2048 > 1152), params with shape[1:].numel()=2048 sort before params with
    # shape[1:].numel()=1152 inside a shared bucket.  Because 2048-dim param totals are not
    # divisible by 1152, the shard boundary cuts a 1152-stride param mid-row, causing
    # make_fsdp_dtensor's local_tensor.view((-1, 1152)) to fail.  Isolating the ViT encoder
    # and the vision projector into their own FSDP units keeps their [N, 1152] params out of
    # any bucket that also contains [M, 2048] diffusion/LLM non-layer params.
    class BagelFullyShardedDataParallel(FullyShardedDataParallel):
        def __init__(self, config, ddp_config, module, fsdp_unit_modules=None, **kwargs):
            if fsdp_unit_modules is None:
                fsdp_unit_modules = [
                    TransformerLayer,
                    MoTTransformerLayer,
                    HFBagelVisionEncoderWrapper,
                    # MultimodalProjector,
                ]
            super().__init__(
                config=config,
                ddp_config=ddp_config,
                module=module,
                fsdp_unit_modules=fsdp_unit_modules,
                **kwargs,
    )

    megatron.training.training.megatron_FSDP = BagelFullyShardedDataParallel
    args = parse_and_validate_args(args_defaults={}, extra_args_provider=add_mimo_args)
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )
