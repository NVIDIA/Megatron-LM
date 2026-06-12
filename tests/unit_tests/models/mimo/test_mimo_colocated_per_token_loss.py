# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Adversarial test for the ``calculate_per_token_loss=False`` claim in
``colocated_schedule._loss_func`` (NMFW-19 / PR #4784).

``_loss_func``'s docstring (colocated_schedule.py:381-383) asserts that the
3-tuple return is "also safe for standard per-microbatch-mean configs", i.e.
that the colocated PP>1 / heterogeneous-DP schedule works correctly with the
production-default loss configuration ``calculate_per_token_loss=False``.

This test empirically checks that claim under FAN-OUT DP (``enc_dp != llm_dp``)
using the STOCK production grad-finalize path (``finalize_model_grads`` per
sub-model, no custom uniform ``1/N_global`` divide). It is the same
encoder-weight oracle the existing correctness test uses for
``per_token_loss=True``, but with the production default loss config and the
default scaling story instead of the bespoke one.

Why this surfaces the bug
--------------------------
With ``calculate_per_token_loss=False``:

  * The schedule divides each microbatch loss by the LLM-local
    ``num_tokens`` and by ``num_microbatches`` (schedules.py:295-299). That
    divisor is computed from the LLM rank's local slice — and in fan-out the
    LLM is split into MORE DP slices than the encoder, so each LLM rank's
    local token count is smaller than an encoder rank's.
  * Each sub-model's DDP then applies ``gradient_scaling_factor =
    1/dp_size`` of ITS OWN DP group (distributed_data_parallel.py:203). The
    LLM grads are scaled by ``1/llm_dp``; the encoder grads (flowing from the
    same loss through the detached embedding grad) are reduced over the
    encoder DP group and scaled by ``1/enc_dp``.

Because ``enc_dp != llm_dp`` in fan-out, the encoder:LLM effective scale is
``llm_dp/enc_dp != 1``. The stock per-side finalize has no uniform divide to
re-normalize the two sides, so the encoder update lands off by that factor
relative to an equal-DP reference where the bridge is the identity and
``enc_dp == llm_dp``.

The existing correctness test masks this by running ONLY with
``per_token_loss=True`` (which pins ``gradient_scaling_factor=1.0`` on both
DDPs and applies an external uniform ``1/N_global`` divide to both sides),
asserting ``num_tokens is not None`` at L211.

Config (smallest fan-out that isolates the bug, PP=1, 8 GPUs)
-------------------------------------------------------------
Dist: ``enc_tp=4, enc_dp=2, llm_tp=2, llm_pp=1, llm_dp=4`` (fan-out, scale=2).
Ref:  equal-DP uniform ``enc_tp=4, enc_dp=2, llm_tp=4, llm_pp=1, llm_dp=2``
      (identity bridge). Encoder TP/DP match dist exactly, so encoder shards
      align 1:1 and the encoder oracles compare shard-to-shard with no
      gather-and-slice. The LLM TP differs (ref=4 vs dist=2), so the LLM
      starting state is copied with the same two-phase TP-reshard the
      existing correctness test uses (all-gather ref's TP shards across
      ``ref_tp_group``, slice for dist's TP) — any encoder-side divergence is
      then pure grad-scaling skew, not TP accumulation drift.

Why the ref MUST span all 8 ranks (deadlock fix)
------------------------------------------------
The reference grids are built as ``tp=enc_tp, dp=enc_dp``. With
``enc_tp * enc_dp == 8`` the reference covers every world rank, so the
ref-side collectives in the parameter-copy oracle (the TP all-gather inside
``_copy_ref_params_to_dist``) are issued in lockstep by all 8 ranks. The
earlier ``enc_tp=2, enc_dp=2`` reference spanned only ranks 0-3 while the
dist LLM spanned all 8 — so ranks 4-7 iterated dist params with no ref
all-gather partner and the copy deadlocked (ranks 0-3 stuck in ALLGATHER,
ranks 4-5 elsewhere). Matching the existing correctness test's
``enc_tp * enc_dp == 8`` invariant keeps every collective symmetric.

Observable signal
------------------
After one Adam step, the dist encoder's first-layer ``layers.0.*`` shards are
compared against the equal-DP reference's matching shards. If the finding is
correct the encoder weights diverge well beyond ``rtol=atol=1e-3`` (expected
relative skew ~ ``llm_dp/enc_dp = 2``). If the schedule is actually correct
under ``per_token_loss=False`` the weights match and the test PASSES,
disproving the finding.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/mimo/\\
test_mimo_colocated_per_token_loss.py -v -s
"""

import os
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
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    build_no_sync_func,
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.mimo.test_mimo_colocated_correctness import (
    _assert_encoder_weights_match,
    _assert_first_layer_grads_match,
    _copy_ref_params_to_dist,
    _generate_and_broadcast_global_batches,
    _set_deterministic_env,
    _slice_global_batch_by_dp,
    _slice_global_batch_for_dist,
    _snapshot_first_layer_encoder_grads,
    forward_step,
)
from tests.unit_tests.test_utilities import Utils


def _wire_production_default_hooks(mimo_model, language_pg, vision_pg):
    """Attach the PRODUCTION-DEFAULT finalize: stock ``finalize_model_grads``
    per sub-model, with NO custom uniform ``1/N_global`` divide.

    This is the path the ``_loss_func`` docstring claims is safe for
    ``calculate_per_token_loss=False`` configs. Each DDP applies its own
    ``gradient_scaling_factor = 1/dp_size`` (distributed_data_parallel.py:203);
    the schedule already divided the per-microbatch loss by the LLM-local
    ``num_tokens`` and ``num_microbatches`` (schedules.py:295-299). Nothing
    here re-normalizes encoder vs LLM relative scale across differing DP
    sizes — that is exactly the property under test.
    """
    no_sync_func = build_no_sync_func(mimo_model)

    def finalize_grads_func(model_list, num_tokens, force_all_reduce=False, **kwargs):
        # Production default: forward the schedule's num_tokens straight into
        # the stock finalize per side. For calculate_per_token_loss=False the
        # schedule already applied the per-microbatch mean, so finalize does
        # the usual DP all-reduce + layernorm/embedding sync with the DDP's
        # built-in 1/dp_size scaling and no extra divide.
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

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
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


class TestColocatedPerTokenLossFalseScaling:
    """Disprove-or-confirm: does the colocated fan-out path scale encoder vs
    LLM grads consistently under ``calculate_per_token_loss=False``?
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def setup_method(self):
        self._mimo_models = []

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        for model in self._mimo_models:
            model.destroy()
        self._mimo_models.clear()
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    def test_fan_out_per_token_loss_false_encoder_weights_match_equal_dp_ref(self):
        """Fan-out + ``per_token_loss=False`` + stock finalize: dist encoder
        post-step weights must match an equal-DP reference.

        Smallest isolating config (8 GPUs, PP=1):
          * Dist: enc_tp=4, enc_dp=2, llm_tp=2, llm_pp=1, llm_dp=4 (fan-out).
          * Ref:  enc_tp=4, enc_dp=2, llm_tp=4, llm_pp=1, llm_dp=2 (identity
            bridge, equal DP). Encoder TP/DP layout matches dist exactly, so
            the encoder oracles are pure grad-scaling-skew detectors. The
            reference spans all 8 ranks (enc_tp * enc_dp == 8), so the LLM
            param-copy's TP all-gather is issued in lockstep by every rank.

        If the finding holds, the encoder weights diverge by ~ llm_dp/enc_dp
        and the assertion fails with a printed weight/grad diff. If the path
        is correct, the test passes and the finding is disproved.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rank = dist.get_rank()
        encoder_name = "images"
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        num_layers = 2
        micro_batch_size = 2

        enc_tp, enc_dp, llm_tp, llm_pp, llm_dp = 4, 2, 2, 1, 4
        num_microbatches = 1

        # Global batch spans the larger DP side (llm_dp here).
        global_batch_size = micro_batch_size * max(enc_dp, llm_dp)

        dist_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        dist_llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
        ref_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        ref_llm_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        create_all_embedding_groups([dist_enc_grid, dist_llm_grid, ref_enc_grid, ref_llm_grid])

        # PRODUCTION DEFAULT loss config: calculate_per_token_loss=False on
        # both sub-model TransformerConfigs (per_token_loss=False below). This
        # is the configuration the _loss_func docstring claims is safe.
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=True
        )

        torch.manual_seed(12345)
        dist_mimo, _, _, dist_language_pg, dist_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=dist_enc_grid,
            llm_grid=dist_llm_grid,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
            bf16=False,
            bias=False,
            dropout=False,
            per_token_loss=False,
        )
        dist_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(dist_mimo)

        torch.manual_seed(12345)
        ref_mimo, _, _, ref_language_pg, ref_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=ref_enc_grid,
            llm_grid=ref_llm_grid,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
            bf16=False,
            bias=False,
            dropout=False,
            per_token_loss=False,
        )
        ref_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(ref_mimo)

        # Identical initial state. Encoder shards match exactly (same
        # enc_tp/enc_dp) so the encoder copy is shard-to-shard. The LLM TP
        # differs (ref_llm_tp == enc_tp == 4 vs dist_llm_tp == 2), so
        # _copy_ref_params_to_dist all-gathers ref's TP shards across
        # ref_llm_grid's TP group (which, because enc_tp * enc_dp == 8, spans
        # every world rank in lockstep) and slices for dist's TP. This is the
        # SAME symmetric two-phase copy the existing correctness test uses for
        # its llm_pp == 1 fan-out case — no per-rank-divergent collectives.
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

        # PRODUCTION DEFAULT finalize on both sides (no uniform 1/N_global).
        _wire_production_default_hooks(dist_mimo, dist_language_pg, dist_vision_pg)
        _wire_production_default_hooks(ref_mimo, ref_language_pg, ref_vision_pg)

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

        # Deterministic global batches identical on every rank. Use the
        # "uniform" mask so all valid-token counts are equal across samples —
        # this removes any per-sample asymmetry and leaves DP-split count
        # differences (fan-out) as the only token-count effect.
        torch.manual_seed(99999)
        global_batches = _generate_and_broadcast_global_batches(
            global_mbs=global_batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_name=encoder_name,
            num_batches=num_microbatches,
            mask_pattern="uniform",
        )
        dist_batches = [
            _slice_global_batch_for_dist(b, dist_enc_grid, dist_llm_grid) for b in global_batches
        ]
        ref_batches = [
            _slice_global_batch_by_dp(b, ref_enc_grid.get_pg("dp")) for b in global_batches
        ]
        ref_per_rank_batch_size = global_batch_size // enc_dp

        try:
            # Dist step (PP=1 fan-out, no-pipelining schedule + forward_step).
            dist_optimizer.zero_grad()
            schedule.forward_backward_no_pipelining(
                forward_step_func=partial(
                    forward_step,
                    encoder_grid=dist_enc_grid,
                    llm_grid=dist_llm_grid,
                    encoder_name=encoder_name,
                ),
                data_iterator=_BatchIterator(dist_batches),
                model=[dist_mimo],
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
                pg_collection=dist_language_pg,
            )
            dist_first_layer_grads = _snapshot_first_layer_encoder_grads(dist_mimo, encoder_name)
            dist_success, dist_grad_norm, _ = dist_optimizer.step()
            assert dist_success, "Dist optimizer step failed"
            assert dist_grad_norm is not None and dist_grad_norm > 0, (
                f"Dist grad_norm={dist_grad_norm} — encoder grads may have been " "silently zeroed."
            )

            # Ref step (equal-DP, identity bridge, no-pipelining).
            ref_optimizer.zero_grad()
            schedule.forward_backward_no_pipelining(
                forward_step_func=partial(
                    forward_step,
                    encoder_grid=ref_enc_grid,
                    llm_grid=ref_llm_grid,
                    encoder_name=encoder_name,
                ),
                data_iterator=_BatchIterator(ref_batches),
                model=[ref_mimo],
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=ref_per_rank_batch_size,
                forward_only=False,
                pg_collection=ref_language_pg,
            )
            ref_first_layer_grads = _snapshot_first_layer_encoder_grads(ref_mimo, encoder_name)
            ref_success, ref_grad_norm, _ = ref_optimizer.step()
            assert ref_success, "Ref optimizer step failed"
            assert ref_grad_norm is not None and ref_grad_norm > 0, f"Ref grad_norm={ref_grad_norm}"
        except Exception:
            import traceback as _tb

            print(
                f"\n=== rank {rank} TEST EXCEPTION ===\n"
                f"config: enc_tp={enc_tp} enc_dp={enc_dp} llm_tp={llm_tp} "
                f"llm_pp={llm_pp} llm_dp={llm_dp} mbs={num_microbatches}\n"
                f"{_tb.format_exc()}\n=== end rank {rank} exception ===\n",
                flush=True,
            )
            raise

        # Oracles. First-layer encoder grads come straight off the DP reduce
        # (before optimizer munges them), so they pin the bug to scaling;
        # encoder weights are the end-to-end post-step signal.
        failures = []
        try:
            _assert_first_layer_grads_match(
                ref_first_layer_grads, dist_first_layer_grads, rtol=1e-3, atol=1e-3
            )
        except AssertionError as e:
            failures.append(('first_layer_grads', str(e)))

        try:
            _assert_encoder_weights_match(
                ref_mimo.modality_submodules[encoder_name].module,
                dist_mimo.modality_submodules[encoder_name].module,
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as e:
            failures.append(('encoder_weights', str(e)))

        if failures:
            summary = "\n\n".join(f"== {oracle} ==\n{msg}" for oracle, msg in failures)
            print(
                f"\n=== rank {rank} per_token_loss=False fan-out failures ===\n"
                f"config: enc_tp={enc_tp} enc_dp={enc_dp} llm_tp={llm_tp} "
                f"llm_pp={llm_pp} llm_dp={llm_dp} mbs={num_microbatches}\n"
                f"{summary}\n=== end rank {rank} failures ===\n",
                flush=True,
            )
            raise AssertionError(
                f"{len(failures)} oracle(s) failed under calculate_per_token_loss=False "
                f"fan-out (enc_dp={enc_dp}, llm_dp={llm_dp}); encoder vs LLM grad scaling "
                f"is inconsistent (expected ~llm_dp/enc_dp skew):\n{summary}"
            )
