# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Adversarial: three-phase colocated schedule with num_microbatches=1 and LLM PP>1.

The consolidated correctness test
(:mod:`test_mimo_colocated_correctness`) parametrizes
``num_microbatches in {1, 4}`` but gates out the single-microbatch
PP>1 cell::

    if num_microbatches < llm_pp:
        pytest.skip(...)

So the smallest *pipeline-active* case — a single microbatch driven
through an LLM PP>1 1F1B pipeline (all warmup/cooldown bubble, no
steady state) — is never exercised by the three-phase schedule. The
PP=1 ``mbs=1`` cell takes the separate ``forward_backward_no_pipelining``
path, so it does not cover the three-phase code either.

``colocated_forward_backward_with_pp`` claims to support any
``num_microbatches`` (``total_batch % num_microbatches == 0`` at
``colocated_schedule.py:342`` and ``L77`` eagerly drains exactly
``num_microbatches`` batches). The single-microbatch path is the most
likely to expose an off-by-one in the warmup/cooldown grad accumulation
into ``detached_full.grad`` on PP stage 0, or in the deferred
finalize / encoder-grad-broadcast handoff, because there is exactly one
LLM backward feeding ``detached_full.grad`` before Phase 3.

This test relaxes that skip for a single config and checks that the
``num_microbatches=1`` PP=2 result still matches the equal-DP PP=1
reference (which is computed via the no-pipelining path) after one
Adam step. If the deferred finalize or the encoder-grad broadcast
mishandles the single-microbatch shape, the encoder weights / grads
and the PP-aware LLM weights diverge from the reference beyond
tolerance, or the schedule deadlocks (guarded by the runner-level
NCCL timeout). If the code is correct, this test PASSES — disproving
the finding.

Smallest config that exhibits it: ``fan_in_pp2`` with
``num_microbatches=1``::

    enc_tp=2, enc_dp=4, llm_tp=2, llm_pp=2, llm_dp=2

``dist_llm_tp == enc_tp`` keeps the PP-aware LLM weight oracle active
(shards align 1:1 with the PP=1 ref after layer-index remap), and the
encoder TP/DP layout matches the ref so encoder shards compare directly.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest \\
        tests/unit_tests/models/mimo/test_mimo_colocated_single_microbatch.py \\
        -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.mimo.test_mimo_colocated_correctness import (
    _assert_encoder_weights_match,
    _assert_first_layer_grads_match,
    _assert_llm_weights_match_pp_aware,
    _copy_llm_params_pp_aware,
    _copy_ref_params_to_dist,
    _generate_and_broadcast_global_batches,
    _run_forward_backward,
    _set_deterministic_env,
    _slice_global_batch_by_dp,
    _slice_global_batch_for_dist,
    _snapshot_first_layer_encoder_grads,
    _wire_training_hooks,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
)
class TestThreePhaseScheduleSingleMicrobatch:
    """num_microbatches=1 + LLM PP>1 must match the equal-DP PP=1 reference.

    This is the (num_microbatches=1, llm_pp>1) cell that the consolidated
    correctness test skips. The three-phase schedule must either run it
    correctly (single microbatch through the pipeline) or reject it with a
    clear error — silent divergence from the PP=1 reference is a bug.
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

    @pytest.mark.parametrize(
        "enc_tp,enc_dp,llm_tp,llm_pp,llm_dp",
        [(2, 4, 2, 2, 2)],  # fan-in, PP=2 (dist_llm_tp == enc_tp → LLM weight oracle on)
        ids=["fan_in_pp2"],
    )
    @pytest.mark.parametrize(
        "mask_pattern", ["uniform", "asymmetric"], ids=["uniform", "asymmetric"]
    )
    def test_single_microbatch_pp2_matches_dp1_reference(
        self, enc_tp, enc_dp, llm_tp, llm_pp, llm_dp, mask_pattern
    ):
        """One microbatch through PP=2 three-phase schedule == equal-DP PP=1 ref.

        Deliberately exercises the (num_microbatches=1, llm_pp>1) cell the
        correctness test gates out. With a single microbatch the LLM 1F1B
        pipeline is pure warmup/cooldown and exactly one LLM backward feeds
        ``detached_full.grad`` before the encoder-grad broadcast + deferred
        finalize. If that handoff mishandles the single-microbatch shape,
        the post-step encoder weights / first-layer grads and the PP-aware
        LLM weights diverge from the PP=1 reference beyond tolerance.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        # The finding: this is precisely the cell the correctness test skips.
        # Assert the precondition holds so the test self-documents *why* it
        # exists (and fails loudly if the schedule's PP requirement changes).
        num_microbatches = 1
        assert num_microbatches < llm_pp, (
            "This adversarial test only makes sense for the (mbs < llm_pp) cell "
            "that the correctness suite skips."
        )

        rank = dist.get_rank()
        try:
            self._run_body(
                rank, enc_tp, enc_dp, llm_tp, llm_pp, llm_dp, mask_pattern, num_microbatches
            )
        except Exception:
            import traceback as _tb

            print(
                f"\n=== rank {rank} SINGLE-MBS TEST EXCEPTION ===\n"
                f"config: enc_tp={enc_tp} enc_dp={enc_dp} llm_tp={llm_tp} "
                f"llm_pp={llm_pp} llm_dp={llm_dp} mbs={num_microbatches} "
                f"mask={mask_pattern}\n"
                f"{_tb.format_exc()}\n"
                f"=== end rank {rank} exception ===\n",
                flush=True,
            )
            raise

    def _run_body(
        self, rank, enc_tp, enc_dp, llm_tp, llm_pp, llm_dp, mask_pattern, num_microbatches
    ):
        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        num_layers = 2
        assert num_layers % llm_pp == 0
        micro_batch_size = 2

        # Global batch spans the larger DP side (fan-in: enc_dp). Dist
        # pre-slices per rank; ref consumes the full global batch per DP rank.
        global_batch_size = micro_batch_size * max(enc_dp, llm_dp)

        dist_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        dist_llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
        ref_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        ref_llm_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        create_all_embedding_groups([dist_enc_grid, dist_llm_grid, ref_enc_grid, ref_llm_grid])

        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=True
        )

        # Dist: heterogeneous TP/DP + LLM PP=2.
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
            per_token_loss=True,
        )
        dist_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(dist_mimo)

        # Ref: equal-DP uniform, LLM PP=1 (no-pipelining path).
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
            per_token_loss=True,
        )
        ref_mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(ref_mimo)

        # Identical initial state. Encoder TP/DP matches → shard-to-shard copy.
        _copy_ref_params_to_dist(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            ref_enc_grid.get_pg("tp"),
            dist_enc_grid.get_pg("tp"),
        )
        # PP>1 with matching TP: dist's local layers map to ref's global
        # layers, shards align 1:1 post-remap.
        assert llm_tp == enc_tp, "fan_in_pp2 config must keep dist_llm_tp == enc_tp"
        _copy_llm_params_pp_aware(
            ref_mimo.language_model.module,
            dist_mimo.language_model.module,
            pp_rank=dist_llm_grid.get_pg("pp").rank(),
            pp_size=llm_pp,
            num_layers=num_layers,
        )

        # PP>1 dist needs the broadcast-from-last-PP-stage finalize variant.
        _wire_training_hooks(dist_mimo, dist_language_pg, dist_vision_pg, llm_grid=dist_llm_grid)
        _wire_training_hooks(ref_mimo, ref_language_pg, ref_vision_pg)

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

        # Single deterministic global batch, identical on every rank.
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
            _slice_global_batch_for_dist(b, dist_enc_grid, dist_llm_grid) for b in global_batches
        ]
        ref_batches = [
            _slice_global_batch_by_dp(b, ref_enc_grid.get_pg("dp")) for b in global_batches
        ]
        ref_per_rank_batch_size = global_batch_size // enc_dp

        # Dist step: three-phase schedule with num_microbatches=1.
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
        dist_first_layer_grads = _snapshot_first_layer_encoder_grads(dist_mimo, encoder_name)
        dist_success, dist_grad_norm, _ = dist_optimizer.step()
        assert dist_success, "Dist optimizer step failed"
        assert dist_grad_norm is not None and dist_grad_norm > 0, (
            f"Dist grad_norm={dist_grad_norm} — with a single microbatch the "
            "three-phase schedule may have failed to accumulate encoder grads "
            "into detached_full.grad before the deferred finalize."
        )

        # Ref step: no-pipelining (PP=1), equal-DP.
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
        ref_first_layer_grads = _snapshot_first_layer_encoder_grads(ref_mimo, encoder_name)
        ref_success, ref_grad_norm, _ = ref_optimizer.step()
        assert ref_success, "Ref optimizer step failed"
        assert ref_grad_norm is not None and ref_grad_norm > 0, f"Ref grad_norm={ref_grad_norm}"

        # Oracles: run all so the diff-stats print covers every layer.
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
                ref_first_layer_grads, dist_first_layer_grads, rtol=1e-3, atol=1e-3
            )
        except AssertionError as e:
            failures.append(('first_layer_grads', str(e)))

        # PP>1 with matching TP: PP-aware LLM weight oracle.
        try:
            _assert_llm_weights_match_pp_aware(
                ref_mimo.language_model.module,
                dist_mimo.language_model.module,
                pp_rank=dist_llm_grid.get_pg("pp").rank(),
                pp_size=llm_pp,
                num_layers=num_layers,
                rtol=1e-2,
                atol=1e-2,
            )
        except AssertionError as e:
            failures.append(('llm_weights_pp_aware', str(e)))

        if failures:
            summary = "\n\n".join(f"== {oracle} ==\n{msg}" for oracle, msg in failures)
            print(
                f"\n=== rank {rank} single-mbs PP=2 failures ===\n"
                f"config: enc_tp={enc_tp} enc_dp={enc_dp} llm_tp={llm_tp} "
                f"llm_pp={llm_pp} llm_dp={llm_dp} mbs={num_microbatches} mask={mask_pattern}\n"
                f"{summary}\n"
                f"=== end rank {rank} failures ===\n",
                flush=True,
            )
            raise AssertionError(
                f"{len(failures)} oracle(s) diverged for num_microbatches=1 PP=2 vs PP=1 "
                f"reference — the three-phase schedule mishandles the single-microbatch "
                f"shape:\n{summary}"
            )
