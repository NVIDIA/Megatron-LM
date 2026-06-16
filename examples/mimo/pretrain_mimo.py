# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero MIMO entry that runs on the *stock* Megatron ``train()`` loop.

This is PR-E5 of the NMFW-516 hetero-MIMO upstreaming effort: the end-to-end
integration that drives the Nemotron6-MoE VLM (20L) through Megatron's
production ``megatron/training/training.py::train()`` instead of the prototype's
bespoke loop. Scope: the NON-COLOCATED path (encoder grid and language grid on
disjoint rank spans), mock data.

Why not stock ``pretrain()``?
=============================
Stock ``pretrain(cfg_container, ...)`` does three things this entry cannot use
verbatim:

  1. It calls ``initialize_megatron`` -> ``_initialize_distributed`` ->
     ``mpu.initialize_model_parallel(...)``. MIMO must NOT initialize the MPU
     globals: each MIMO module owns its own ``HyperCommGrid`` with disjoint rank
     spans, so a single world-spanning MPU layout is wrong. We therefore
     replicate only the *non-MPU* pieces of ``initialize_megatron``
     (``set_global_variables`` minus the num-microbatches calculator wiring,
     random seeds, torch.distributed bring-up via MM1) and skip MPU init.
  2. It calls ``setup_model_and_optimizer`` -> ``get_model`` which re-wraps a
     single top-level DDP and builds a single optimizer over the MPU groups. The
     MIMO model needs PER-SUBMODULE DDP (one per module grid) and the
     ``MimoOptimizer`` (one inner optimizer per active module). ``E4
     build_mimo_runtime`` already assembles that; we feed its model list straight
     to ``train()``.
  3. It builds train/valid/test datasets via a dataset provider. MIMO uses the
     role-aware iterator from ``select_data_iterator`` (mock here).

What this entry replicates from ``initialize_megatron`` (and what it skips)
===========================================================================
Replicated (needed global state ``train()`` / the schedule read):

  * ``set_global_variables(args, build_tokenizer=False)`` MINUS its
    num-microbatches calculator call -- args global, timers, tensorboard/wandb,
    experimental flag. We build the calculator ourselves AFTER fixing
    ``args.data_parallel_size`` (see below), because ``set_global_variables``
    would key it on the stock (world-derived) DP size.
  * ``_set_random_seed`` equivalent: ``random`` / ``numpy`` / ``torch`` seeds
    (matches the prototype's ``loop.py`` so dataset-construction RNG agrees).
  * torch.distributed bring-up via MM1 ``initialize_distributed`` (NCCL +
    global memory buffer; NO MPU groups).

Skipped (and why):

  * ``mpu.initialize_model_parallel`` -- MIMO owns per-module grids; a global MPU
    layout is wrong for disjoint spans. See (1) above.
  * ``_compile_dependencies`` -- only compiles the C++ dataset index builder,
    which the mock iterator does not use.
  * ``_init_autoresume`` / ``_initialize_tp_communicators`` -- autoresume and TP
    comm-overlap user buffers are not exercised by this mock run.
  * tokenizer build -- mock data reads ``args.vocab_size`` directly and the
    provider's ``_vocab_size`` falls back to it, so no tokenizer is needed.

The ``data_parallel_size`` fix
==============================
Stock ``validate_args`` recomputes ``args.data_parallel_size = world_size //
(tp*pp*cp)`` from the *stock* (world-spanning) parallelism flags. For the
disjoint hetero layout that value is wrong: the language grid is only ``llm_dp``
wide. ``train()`` reads ``get_num_microbatches()`` (keyed on
``global_batch_size / (micro_batch_size * data_parallel_size)``) and
``mpu.get_data_parallel_world_size()`` for sample accounting. We therefore:

  * set ``args.data_parallel_size = args.llm_dp`` AFTER ``validate_args``;
  * build the num-microbatches calculator with that DP so it yields the
    script's ``--num-microbatches`` (gbs 8 / (mbs 1 * dp 2) = 4);
  * pin ``parallel_state._MPU_DATA_PARALLEL_WORLD_SIZE = llm_dp`` so train()'s
    four ``mpu.get_data_parallel_world_size()`` reads return the language DP size
    without a full MPU init (a bootstrap/MPU-materialization compatibility
    point, per CLAUDE.md).

The grad-finalization clobber
=============================
``E4 build_mimo_runtime`` installs the MIMO dual grad-finalization hook
(``configure_grad_sync`` sets ``model.config.finalize_model_grads_func`` to the
encoder+language finalizer). But stock ``train()`` unconditionally reassigns
``config.finalize_model_grads_func = finalize_model_grads`` (the module-level
import in ``megatron.training.training``). To keep the MIMO hook without editing
core, we monkeypatch that module-level symbol to the MIMO finalizer BEFORE
calling ``train()``. ``config.grad_scale_func`` is also reassigned by ``train()``
to ``optimizer.scale_loss``; for bf16 with no grad scaler the MimoOptimizer loss
scale is 1.0, so ``scale_loss(loss) == loss`` -- behaviorally identical to the
MIMO ``lambda loss: loss``, so no patch is needed there.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import megatron.training.training as training_module
from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    prepare_model_provider_args,
    validate_model_provider_args,
)
from examples.mimo.training.args import add_hetero_grid_args, validate_hetero_grid_args
from examples.mimo.training.bootstrap import build_mimo_runtime
from examples.mimo.training.data import add_data_args
from examples.mimo.training.distributed import (
    initialize_distributed,
    print_rank_0,
    shutdown_distributed,
)
from examples.mimo.training.step import mimo_forward_step
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables as _stock_set_global_variables

# args ``set_global_variables`` requires but which the script does not pass. We
# skip the tokenizer build (mock data reads args.vocab_size directly), so no
# tokenizer-type default is needed.
_ARGS_DEFAULTS = {
    # No dataset path / tokenizer for the mock run.
}


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Compose the model-provider arg group with the hetero grid arg group.

    Both register their own argparse groups onto the stock parser. We also add
    ``--num-microbatches`` here: stock Megatron derives the microbatch count from
    ``global_batch_size / (micro_batch_size * data_parallel_size)`` and has no
    such flag, but the prototype (and our run script) pass it explicitly.
    """
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_data_args(parser)
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=1,
        help="Explicit microbatch count for the hetero MIMO loop (stock derives this "
        "from gbs/mbs/dp; the hetero layout keys it on --llm-dp instead).",
    )
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Run the stock arg pipeline plus the MIMO preset/validation hooks.

    Sequence (handoff section 2):
      parse_args(extra) -> prepare_model_provider_args (preset, BEFORE validate)
      -> stock validate_args -> validate_model_provider_args
      -> validate_hetero_grid_args -> data_parallel_size fix.
    """
    args = parse_args(extra_args_provider)

    # Apply the Nemotron preset BEFORE stock validation so preset-derived sizes
    # (num_layers, hybrid pattern, seq_length, num_experts, image_seq_length, ...)
    # flow into validate_args.
    prepare_model_provider_args(args)

    # Stock validation. ``defaults`` fills only args the user left at default.
    validate_args(args, _ARGS_DEFAULTS)

    # MIMO-specific validation (runs after stock so padded_vocab_size / num_experts
    # are populated).
    validate_model_provider_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    validate_hetero_grid_args(args, world_size)

    # --- data_parallel_size fix (see module docstring). --------------------
    # Stock validate_args set args.data_parallel_size = world_size//(tp*pp*cp).
    # Re-key it on the language grid's DP so get_num_microbatches() and sample
    # accounting reflect the disjoint hetero layout.
    args.data_parallel_size = args.llm_dp

    # Stock throughput/FLOPs logging does ``1 + args.mtp_num_layers``; the Nemotron
    # preset leaves it None (MTP off), which TypeErrors. Default to 0 (no MTP).
    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0

    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set non-MPU global state, then build the num-microbatches calculator on llm_dp.

    Replicates the non-MPU portion of ``initialize_megatron`` (see module
    docstring). ``set_global_variables`` would build the num-microbatches
    calculator keyed on the (now-fixed) ``args.data_parallel_size``; we call it
    with ``build_tokenizer=False`` so no tokenizer is constructed. Because
    ``data_parallel_size`` is already ``llm_dp`` by this point, the calculator it
    builds is correct, so we do not rebuild it.
    """
    _stock_set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)

    # Defensive: ensure the num-microbatches calculator reflects llm_dp even if a
    # future set_global_variables stops keying on args.data_parallel_size.
    from megatron.core.num_microbatches_calculator import get_num_microbatches as _gnmb

    expected = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
    if _gnmb() != expected:
        # Rebuild to the llm_dp-keyed value.
        init_num_microbatches_calculator(
            rank=args.rank,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=args.data_parallel_size,
            decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
        )


def _seed_everything(args: argparse.Namespace) -> None:
    """Seed python/numpy/torch RNG (matches the prototype loop + stock _set_random_seed)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def _build_optimizer(args: argparse.Namespace, model):
    """Build the MimoOptimizer over the per-submodule-DDP MimoModel.

    Mirrors the prototype's ``hetero/optimizer.py::build_optimizer``: a single
    ``OptimizerConfig`` shared by every active module optimizer, distributed
    optimizer on, bf16 unless --fp32.
    """
    return get_mimo_optimizer(
        model[0],
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
    )


def _build_scheduler(args: argparse.Namespace, optimizer) -> OptimizerParamScheduler:
    """Build the stock OptimizerParamScheduler keyed on the llm_dp global batch.

    Ports ``hetero/optimizer.py::build_optimizer_param_scheduler``: scheduler
    "steps" are consumed samples (incremented by the global batch size per
    optimizer step), so iter-based knobs convert via ``iters * global_batch_size``.
    """
    global_batch_size = args.global_batch_size

    if getattr(args, "lr_warmup_samples", None) is not None:
        lr_warmup_steps = args.lr_warmup_samples
    else:
        lr_warmup_steps = args.lr_warmup_iters * global_batch_size

    if getattr(args, "lr_decay_samples", None) is not None:
        lr_decay_steps = args.lr_decay_samples
    else:
        lr_decay_iters = (
            args.lr_decay_iters if args.lr_decay_iters is not None else args.train_iters
        )
        lr_decay_steps = lr_decay_iters * global_batch_size

    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=args.lr,
        min_lr=args.min_lr if args.min_lr is not None else 0.0,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.weight_decay,
        end_wd=args.weight_decay,
        wd_incr_steps=args.train_iters * global_batch_size,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=getattr(args, "lr_wsd_decay_samples", None),
        lr_wsd_decay_style=getattr(args, "lr_wsd_decay_style", None),
    )


def _install_mimo_grad_finalize(model) -> None:
    """Preserve the MIMO dual grad-finalization hook across stock train()'s clobber.

    stock ``train()`` reassigns ``config.finalize_model_grads_func`` to the
    module-level ``megatron.training.training.finalize_model_grads`` symbol. We
    repoint that symbol to the MIMO finalizer that ``configure_grad_sync`` already
    installed on ``model.config`` (in build_mimo_runtime). After train() runs its
    ``config.finalize_model_grads_func = finalize_model_grads`` line, the config
    therefore carries the MIMO hook.
    """
    mimo_finalize = model[0].config.finalize_model_grads_func
    assert mimo_finalize is not None, (
        "expected build_mimo_runtime -> configure_grad_sync to install a MIMO "
        "finalize_model_grads_func on the language config"
    )
    training_module.finalize_model_grads = mimo_finalize


def _install_safe_flops() -> None:
    """Neutralize stock throughput/FLOPs accounting for the hetero MIMO model.

    ``num_floating_point_operations`` derives a single homogeneous model's FLOPs
    from the global (language) args and runs on every rank. That is ill-defined
    for the disjoint encoder(RADIO)+language(hybrid-MoE) layout: encoder ranks
    IndexError parsing the language MoE/hybrid pattern, and the hybrid path trips
    on MTP. FLOPs/throughput is cosmetic for the smoke milestone, so wrap it to
    return 0 on failure. TODO(NMFW-516): proper per-module hetero FLOPs accounting.
    """
    _orig = training_module.num_floating_point_operations

    def _safe(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except Exception:
            return 0

    training_module.num_floating_point_operations = _safe


def _mimo_branch_name(topology) -> str:
    """Return this rank's MIMO branch name ("language" or the encoder grid name).

    Uses grid membership (the same source of truth bootstrap.py uses) to namespace
    per-branch checkpoint state (e.g. the RNG ShardedObject key).
    """
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    if topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid():
        return "language"
    return next(
        (
            name
            for name, grid in topology.grids.items()
            if name != MIMO_LANGUAGE_MODULE_KEY and grid.is_current_rank_in_grid()
        ),
        "language",
    )


def _install_mimo_checkpointing(topology):
    """Replace stock save_checkpoint / load_checkpoint with a hetero-symmetric path.

    Returns the patched ``load_checkpoint`` callable; the caller must invoke it
    explicitly before train() to drive the resume LOAD (see the note at the end of
    this function for why train() does not call it itself).

    Why stock save_checkpoint hangs in a WORLD collective
    =====================================================
    Stock ``save_checkpoint`` (with the default ``--ckpt-fully-parallel-save``)
    wraps the torch_dist save strategy in ``FullyParallelSaveStrategyWrapper`` keyed
    on ``mpu.get_data_parallel_group(with_context_parallel=True)`` -- which, after
    the entry pins parallel_state to this rank's own grid, is only the rank's own
    2-rank branch DP group. ``apply_saving_parallelization`` then exchanges the save
    distribution over that per-branch DP group while the underlying torch DCP save
    planner runs its coordination collectives (gather_object / all_reduce / broadcast
    in state_dict_saver.py) over the WORLD default group. Layering a per-branch
    distribution under world-level DCP coordination across two structurally-different
    grids (encoder ranks contribute a RADIO encoder sharded dict; language ranks a
    Mamba LM + dist-optimizer sharded dict) desynchronizes the world collective:
    the watchdog fires on a fixed-size ALLGATHER on default_pg that not all 8 ranks
    reach symmetrically.

    The prototype's approach (the fix)
    ==================================
    ``examples/mimo/training/hetero/checkpointing.py`` does NOT use stock
    save_checkpoint. It assembles ONE unified sharded state dict (model + optimizer +
    scheduler + per-branch RNG) and calls ``dist_checkpointing.save`` DIRECTLY with
    the default (plain) ``TorchDistSaveShardedStrategy`` -- no FullyParallel DP-group
    wrapper. Every rank contributes only its own branch's shards, but the GLOBAL set
    of ShardedTensors/ShardedObjects is the symmetric union, and ALL coordination
    (``determine_global_metadata``'s world ``all_gather_object`` and the DCP planner's
    world collectives) runs over the world group symmetrically. Variable per-rank
    metadata is exactly what ``all_gather_object`` is built to handle, so disjoint
    per-branch structure is fine as long as no per-DP-group fully-parallel wrapper is
    interposed.

    This installer ports that approach onto the stock train() loop while reusing
    stock ``generate_state_dict`` (which already produces per-rank-symmetric model /
    optimizer / scheduler sharded dicts, and -- with args.use_distributed_optimizer
    pinned True -- selects the ``dp_reshardable`` optimizer format). It monkeypatches
    the ``save_checkpoint`` / ``load_checkpoint`` symbols imported into
    ``megatron.training.training`` so train() drives the hetero-symmetric path.
    """
    import os

    import megatron.training.checkpointing as ckpt_module
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedObject
    from megatron.core.dist_checkpointing.utils import _clean_metadata_for_serialization
    from megatron.training.checkpointing import (
        _build_sharded_state_dict_metadata,
        generate_state_dict,
        get_rng_state,
    )

    branch_name = _mimo_branch_name(topology)
    _TRACKER = "latest_checkpointed_iteration.txt"

    def _iter_dir(root, iteration):
        return os.path.join(root, f"iter_{iteration:07d}")

    def _tracker_path(root):
        return os.path.join(root, _TRACKER)

    def _branch_keyed_rng_state(ckpt_format):
        """Per-branch RNG ShardedObject to avoid cross-grid key collision.

        Stock ``get_rng_state`` keys a single ``ShardedObject('rng_state', ...,
        (pp,tp), (pp_rank,tp_rank), replica_id=dp_rank)`` on the rank's pinned
        parallel_state groups. The encoder and language grids can share a (pp,tp)
        factorization, so their RNG objects would collide on an identical
        key+offset+replica_id with different payloads. Rewrite the key to
        ``mimo.<branch>.rng_state`` (mirrors the prototype's _collect_rng_state) so
        the two branches round-trip independently.
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        rng = get_rng_state(ckpt_format, tp_group, pp_group)
        if isinstance(rng, ShardedObject):
            key = f"mimo.{branch_name}.rng_state"
            rng = ShardedObject(
                key, rng.data, rng.global_shape, rng.global_offset, replica_id=rng.replica_id
            )
        return rng

    def _mimo_save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context=None,
        non_persistent_ckpt=False,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
        **_unused,
    ):
        from megatron.training.global_vars import get_args

        args = get_args()
        if not args.save:
            return
        assert (
            args.ckpt_format == "torch_dist"
        ), f"hetero MIMO checkpointing supports only --ckpt-format torch_dist, got {args.ckpt_format}"

        target_dir = _iter_dir(args.save, iteration)
        if torch.distributed.get_rank() == 0:
            os.makedirs(target_dir, exist_ok=True)
        torch.distributed.barrier()

        print_rank_0(f"saving hetero MIMO checkpoint at iteration {iteration} to {target_dir}")

        rng_state = None if args.no_save_rng else _branch_keyed_rng_state(args.ckpt_format)
        sharded_sd_metadata = _build_sharded_state_dict_metadata(args)
        state_dict = generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
            model_sd_kwargs=dict(metadata=sharded_sd_metadata),
        )
        state_dict["num_floating_point_operations_so_far"] = num_floating_point_operations_so_far

        # Default strategy => plain TorchDistSaveShardedStrategy: world-symmetric
        # coordination, NO FullyParallel DP-group wrapper. This is the key to
        # coordinating the disjoint encoder/language grids (see installer docstring).
        dist_checkpointing.save(
            state_dict,
            target_dir,
            content_metadata=_clean_metadata_for_serialization(sharded_sd_metadata),
        )

        if torch.distributed.get_rank() == 0:
            tmp = _tracker_path(args.save) + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(iteration))
            os.replace(tmp, _tracker_path(args.save))
        torch.distributed.barrier()
        print_rank_0(f"hetero MIMO checkpoint at iteration {iteration} saved")

    def _read_tracker(load_root):
        tracker = _tracker_path(load_root)
        local_iter = -1
        if os.path.isfile(tracker):
            with open(tracker) as f:
                contents = f.read().strip()
            if contents:
                local_iter = int(contents)
        iters = torch.tensor([local_iter], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(iters, op=torch.distributed.ReduceOp.MAX)
        max_iter = int(iters.item())
        return max_iter if max_iter >= 0 else None

    def _mimo_load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        load_arg="load",
        strict=True,
        checkpointing_context=None,
        skip_load_to_model_and_opt=False,
        **_unused,
    ):
        from megatron.training.global_vars import get_args

        args = get_args()
        load_root = getattr(args, load_arg, None)
        if not load_root:
            return 0, 0

        iteration = _read_tracker(load_root)
        if iteration is None:
            print_rank_0(f"no MIMO checkpoint at {load_root}; starting from iteration 0")
            return 0, 0
        source_dir = _iter_dir(load_root, iteration)
        if not os.path.isdir(source_dir):
            raise RuntimeError(
                f"tracker at {load_root} points to iteration {iteration} but {source_dir} is missing"
            )

        is_finetune = bool(args.finetune)
        include_optim = (not args.no_load_optim) and not is_finetune
        include_sched = (not getattr(args, "no_load_scheduler", False)) and not is_finetune
        include_rng = (not args.no_load_rng) and not is_finetune

        print_rank_0(
            f"loading hetero MIMO checkpoint from {source_dir}"
            f" (optimizer={include_optim}, scheduler={include_sched},"
            f" rng={include_rng}, finetune={is_finetune})"
        )

        rng_state = _branch_keyed_rng_state(args.ckpt_format) if include_rng else None
        sharded_sd_metadata = _build_sharded_state_dict_metadata(args)
        request = generate_state_dict(
            args,
            model,
            optimizer if include_optim else None,
            opt_param_scheduler if include_sched else None,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=dict(metadata=sharded_sd_metadata, is_loading=True),
            model_sd_kwargs=dict(metadata=sharded_sd_metadata),
        )
        # ``args`` is common (rank-0) state; it round-trips via common.pt on disk
        # (returned in ``loaded`` below), so don't request it back as a load target.
        request.pop("args", None)

        loaded = dist_checkpointing.load(request, source_dir)

        # Apply the loaded state to model / optimizer / scheduler (the symmetric
        # union assembled by every rank reconstructs into each rank's own shards).
        if not skip_load_to_model_and_opt:
            model[0].load_state_dict(loaded["model"], strict=strict)
            if (
                include_optim
                and optimizer is not None
                and not optimizer.is_stub_optimizer
                and "optimizer" in loaded
            ):
                optimizer.load_state_dict(loaded["optimizer"])
        if include_sched and opt_param_scheduler is not None and "opt_param_scheduler" in loaded:
            opt_param_scheduler.load_state_dict(loaded["opt_param_scheduler"])

        if include_rng and loaded.get("rng_state") is not None:
            _restore_rng_from_loaded(loaded["rng_state"])

        # Restore sample accounting from the checkpoint's args (consumed by stock
        # train()'s batch-size / microbatch bookkeeping).
        ckpt_args = loaded.get("args")
        if ckpt_args is not None and not is_finetune:
            getter = ckpt_args.get if isinstance(ckpt_args, dict) else lambda k, d: getattr(ckpt_args, k, d)
            args.consumed_train_samples = getter("consumed_train_samples", 0)
            args.skipped_train_samples = getter("skipped_train_samples", 0)
            args.consumed_valid_samples = getter("consumed_valid_samples", 0)

        nfpo = int(loaded.get("num_floating_point_operations_so_far", 0))
        resume_iter = 0 if is_finetune else int(loaded.get("iteration", iteration))
        print_rank_0(f"resuming hetero MIMO training at iteration {resume_iter}")
        return resume_iter, nfpo

    # Patch the symbols train() resolved at import time. This routes the
    # mid-training periodic SAVE (save_checkpoint_and_time -> save_checkpoint)
    # through the hetero-symmetric path.
    training_module.save_checkpoint = _mimo_save_checkpoint
    training_module.load_checkpoint = _mimo_load_checkpoint
    ckpt_module.save_checkpoint = _mimo_save_checkpoint
    ckpt_module.load_checkpoint = _mimo_load_checkpoint

    # The resume LOAD is NOT driven by train(): stock load_checkpoint runs inside
    # setup_model_and_optimizer, which this entry bypasses (it builds the MimoModel /
    # MimoOptimizer itself and calls train() directly). So the caller (main()) must
    # invoke this returned load function explicitly BEFORE train() and seed
    # args.iteration so train() resumes at iteration+1. Mirrors the prototype loop,
    # which calls load_checkpoint() explicitly and starts at start_iteration + 1.
    return _mimo_load_checkpoint


def _restore_rng_from_loaded(rng_obj) -> None:
    """Apply RNG state loaded from a per-branch rng ShardedObject (prototype parity)."""
    from megatron.core import tensor_parallel

    payload = rng_obj
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        rng = payload[0]
    elif isinstance(payload, dict):
        rng = payload
    else:
        return
    random.setstate(rng["random_rng_state"])
    np.random.set_state(rng["np_rng_state"])
    torch.set_rng_state(rng["torch_rng_state"])
    torch.cuda.set_rng_state(rng["cuda_rng_state"])
    if rng.get("rng_tracker_states"):
        tensor_parallel.get_cuda_rng_tracker().set_states(rng["rng_tracker_states"])


def _set_mpu_data_parallel_world_size(args: argparse.Namespace) -> None:
    """Pin the MPU DP world size to llm_dp for train()'s sample accounting.

    train() reads ``mpu.get_data_parallel_world_size()`` (4 sites) for consumed-
    sample / batch-size bookkeeping. That getter returns the module global
    ``_MPU_DATA_PARALLEL_WORLD_SIZE`` when set, before touching any MPU group, so
    pinning it lets train() run without a full MPU init. This is a bootstrap /
    MPU-materialization compatibility point (see CLAUDE.md "Megatron Core Process
    Groups").
    """
    parallel_state._MPU_DATA_PARALLEL_WORLD_SIZE = args.llm_dp


def main() -> None:
    """Program entrypoint: stock-args MIMO run on the stock train() loop."""
    args = _parse_and_validate()

    # Non-MPU global setup, then torch.distributed bring-up (no MPU init).
    _setup_globals(args)
    initialize_distributed()
    _seed_everything(args)
    _set_mpu_data_parallel_world_size(args)

    # Per-rank runtime: per-submodule-DDP MimoModel + topology + communicator +
    # role-aware data iterator + grad-sync hook (E4).
    rt = build_mimo_runtime(args)

    optimizer = _build_optimizer(args, rt.model)
    opt_param_scheduler = _build_scheduler(args, optimizer)

    # Keep the MIMO grad-finalization hook alive past stock train()'s reassign.
    _install_mimo_grad_finalize(rt.model)

    # Neutralize stock FLOPs/throughput accounting (ill-defined for the hetero model).
    _install_safe_flops()

    # Replace stock save/load_checkpoint with a hetero-symmetric path: assemble one
    # unified sharded state dict and call dist_checkpointing.save/load directly with
    # the plain (world-coordinated) torch_dist strategy -- NO FullyParallel DP-group
    # wrapper (which deadlocks a world collective across the disjoint grids). Also
    # namespaces the RNG ShardedObject key per branch. See _install_mimo_checkpointing.
    # Returns the load fn; train() never calls it (resume load normally lives in the
    # bypassed setup_model_and_optimizer), so we invoke it explicitly below.
    mimo_load_checkpoint = _install_mimo_checkpointing(rt.topology)

    # Pin this rank's parallel_state model-parallel group to its module's mp group so
    # stock training_log's cosmetic mp-group reductions (e.g. the LR gather in
    # reduce_max_stat_across_model_parallel_group) work without full mpu init. The
    # MIMO schedule/grad reductions use the threaded pg_collection; this only
    # satisfies stock logging-path reads of mpu.get_model_parallel_group().
    # Pin this rank's parallel_state groups to its module's per-module groups so the
    # stock logging + checkpoint paths (which read mpu.get_*_group()) work without
    # full mpu init. The MIMO schedule/grad reductions use the threaded pg_collection;
    # these pins only satisfy stock-path reads (training_log, report_memory,
    # save_checkpoint shard/RNG bookkeeping).
    _pgc = rt.pg_collection
    if getattr(_pgc, "mp", None) is not None:
        parallel_state._MODEL_PARALLEL_GROUP = _pgc.mp
    if getattr(_pgc, "tp", None) is not None:
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = _pgc.tp
    if getattr(_pgc, "pp", None) is not None:
        parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = _pgc.pp
    if getattr(_pgc, "dp", None) is not None:
        parallel_state._DATA_PARALLEL_GROUP = _pgc.dp
        if getattr(_pgc, "dp_cp", None) is not None:
            parallel_state._DATA_PARALLEL_GROUP_WITH_CP = _pgc.dp_cp

    # Bookkeeping fields stock train() reads that setup_model_and_optimizer /
    # the dataset builder would normally set.
    args.iteration = 0
    args.num_floating_point_operations_so_far = 0
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.do_train = True
    args.do_valid = False
    args.do_test = False

    # Every MIMO submodule optimizer is a DistributedOptimizer (see
    # _build_optimizer -> get_mimo_optimizer, and runtime.py's per-submodule DDP
    # wrap with use_distributed_optimizer=True). The top-level ``args`` flag,
    # however, defaults to False because we never run stock setup_model_and_optimizer.
    # Stock checkpointing.py keys two things off it:
    #   1. save -> _build_sharded_state_dict_metadata: only sets
    #      ``distrib_optim_sharding_type`` when this is True. Without it the
    #      per-submodule DistributedOptimizer.sharded_state_dict falls back to the
    #      deprecated 'fully_sharded_model_space' format, which emits ShardedTensors
    #      with ``flattened_range`` and crashes torch_dist's
    #      validate_metadata_integrity ("flattened_range is not supported").
    #   2. The 'Storing distributed optimizer sharded state of type ...' log line.
    # Pinning it True makes the metadata builder select 'dp_reshardable' (the stock
    # default and the format the hetero prototype used), which round-trips cleanly
    # through torch_dist save/load and is persisted in the checkpoint's
    # content_metadata so resume mirrors it.
    args.use_distributed_optimizer = True

    # Resume LOAD (must run after parallel_state pins + args.use_distributed_optimizer
    # so the load request's model/optimizer/RNG sharded keys match what was saved).
    # train() does not call load_checkpoint (the entry bypasses
    # setup_model_and_optimizer), so we drive it here. All 8 ranks call this with the
    # identical world-symmetric load request, mirroring the world-symmetric save.
    # On success it returns the saved iteration (e.g. 2); seeding args.iteration makes
    # stock train() resume the loop at iteration+1 (e.g. 3). consumed-sample / FLOPs
    # accounting is restored inside the load fn.
    if args.load:
        resume_iter, resume_nfpo = mimo_load_checkpoint(
            rt.model, optimizer, opt_param_scheduler
        )
        args.iteration = resume_iter
        args.num_floating_point_operations_so_far = resume_nfpo

    config = rt.model[0].config

    print_rank_0(
        "Starting hetero MIMO training on stock train(): "
        f"world_size={torch.distributed.get_world_size()}, "
        f"llm_dp={args.llm_dp}, num_microbatches={args.num_microbatches}, "
        f"global_batch_size={args.global_batch_size}, train_iters={args.train_iters}"
    )

    try:
        training_module.train(
            forward_step_func=mimo_forward_step,
            model=rt.model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            train_data_iterator=rt.data_iterator,
            valid_data_iterator=None,
            process_non_loss_data_func=None,
            config=config,
            checkpointing_context={},
            non_loss_data_func=None,
            p2p_communicator=rt.communicator,
            schedule_pg_collection=rt.topology.schedule_pg_collection,
        )
        torch.distributed.barrier()
        print_rank_0("Hetero MIMO training (stock loop) completed")
    finally:
        if hasattr(rt.model[0], "destroy"):
            rt.model[0].destroy()
        rt.topology.destroy()
        shutdown_distributed()


if __name__ == "__main__":
    main()
