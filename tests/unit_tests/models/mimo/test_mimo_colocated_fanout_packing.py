# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Adversarial test for NMFW-19 fan-out passthrough narrowing in
``_build_lm_microbatches`` / ``_passthrough``.

Finding under test
------------------
Under fan-out (``llm_dp > enc_dp``) the three-phase colocated schedule
narrows the bridge-side encoder embeddings to each LLM-DP rank's slot, and
``_build_lm_microbatches`` narrows the LLM-side passthrough tensors
(``input_ids``, ``labels``, ``loss_mask``, ``position_ids``,
``attention_mask``) to the SAME slot via ``_maybe_narrow`` so they line up.

However ``_passthrough`` (colocated_schedule.py:318-328):

  * hard-codes exactly six fields. Any OTHER per-sample field present in the
    batch dict is silently dropped from the LM microbatch.
  * forwards ``packing_kwargs`` UN-narrowed (L327): ``b.get('packing_kwargs')``
    with no ``_maybe_narrow`` call, even though ``input_ids`` / ``labels`` /
    ``loss_mask`` / ``position_ids`` in the SAME microbatch ARE narrowed.

For packed (THD) sequences the ``cu_seqlens`` / per-sample metadata inside
``packing_kwargs`` describes the FULL encoder-DP batch, while ``input_ids``
afterwards describes only this LLM-DP slot. The two desync: the packing
metadata claims more samples / a longer cumulative sequence length than the
narrowed ``input_ids`` actually contains, which under THD attention indexes
out of range.

This test exercises the bug at the HELPER level (``_build_lm_microbatches``)
with a real fan-out grid (enc_dp=1, llm_dp=2). It asserts the invariant that
should hold under fan-out: every per-sample field the LLM consumes lines up
with the narrowed ``input_ids``. If the code is correct (narrows or rejects
packing_kwargs, and does not drop arbitrary fields) the test PASSES; if the
finding is real the documented assertions FAIL with a clear message.

Smallest config that exhibits it: enc_tp=1, enc_dp=1, llm_tp=1, llm_pp=1,
llm_dp=2 (fan-out scale=2). Runs on 2 GPUs; no model is built — only the
schedule helper is called, so runtime is sub-second.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        -m pytest tests/unit_tests/models/mimo/\\
test_mimo_colocated_fanout_packing.py -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.models.mimo.colocated_schedule import _build_lm_microbatches, _fan_out_slot
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
)
from tests.unit_tests.test_utilities import Utils


def _make_packing_kwargs(total_batch, seq_per_sample):
    """Build THD-style packing metadata sized for the FULL enc-DP batch.

    ``cu_seqlens`` is the cumulative sequence-length prefix-sum over all
    ``total_batch`` samples (length ``total_batch + 1``); ``max_seqlen`` and
    ``num_samples`` describe the same full batch. This is exactly what the
    encoder-DP-sized data iterator would emit for a packed sequence before
    any fan-out narrowing.
    """
    cu_seqlens = torch.arange(
        0, (total_batch + 1) * seq_per_sample, seq_per_sample, device='cuda', dtype=torch.int32
    )
    return {
        'cu_seqlens': cu_seqlens,
        'cu_seqlens_q': cu_seqlens.clone(),
        'cu_seqlens_kv': cu_seqlens.clone(),
        'max_seqlen': seq_per_sample,
        'max_seqlen_q': seq_per_sample,
        'max_seqlen_kv': seq_per_sample,
        'num_samples': total_batch,
    }


class TestFanOutPassthroughNarrowing:
    """Helper-level fan-out narrowing-consistency checks for ``_passthrough``."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def setup_method(self):
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    def test_packing_kwargs_and_arbitrary_fields_narrowed_under_fan_out(self):
        """Fan-out (enc_dp=1, llm_dp=2): every per-sample field must line up
        with the narrowed ``input_ids``.

        Builds a single microbatch whose passthrough tensors and
        ``packing_kwargs`` are sized for the full encoder-DP batch
        (``total_batch`` samples). After ``_build_lm_microbatches`` the
        ``input_ids`` are narrowed to ``total_batch / 2`` samples (fan-out
        scale=2). We then assert:

          1. ``packing_kwargs`` describes the SAME number of samples as the
             narrowed ``input_ids`` (it won't — it's forwarded un-narrowed,
             still claiming ``total_batch`` samples -> THD desync).
          2. an arbitrary extra per-sample field (``token_type_ids``) survives
             into the microbatch and is narrowed (it won't — ``_passthrough``
             drops any field outside the hard-coded six).

        Either failing assertion proves the finding; if the schedule narrows
        packing metadata and preserves/narrows arbitrary fields (or rejects
        the combination), both assertions pass and the finding is disproved.
        """
        if self.world_size < 2:
            pytest.skip(f"Requires >=2 GPUs for fan-out (llm_dp=2), got {self.world_size}")

        rank = dist.get_rank()

        # Fan-out: encoder DP=1, LLM DP=2 — both grids must span BOTH ranks.
        # The encoder is replicated (dp=1) so it uses tp=2 to cover {0,1};
        # the LLM splits {0,1} into 2 DP ranks (tp=1, dp=2). enc_dp=1 < llm_dp=2
        # gives fan-out scale=2. (enc_tp=1,dp=1 would span only rank 0, leaving
        # enc_grid.get_pg('dp') None on rank 1.)
        enc_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=1)
        llm_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=2)
        create_all_embedding_groups([enc_grid, llm_grid])

        # Confirm we really are in the fan-out regime that triggers narrowing.
        scale, slot = _fan_out_slot(enc_grid, llm_grid)
        assert scale == 2, f"Expected fan-out scale=2, got {scale} (config not fan-out?)"

        # Full encoder-DP-sized batch: total_batch samples, seq_per_sample
        # tokens each. The data iterator under fan-out yields this size; the
        # bridge narrows encoder embeddings on the LLM side and the schedule
        # is responsible for narrowing the LLM-side fields to match.
        total_batch = 4  # divisible by scale=2 -> narrowed batch = 2
        seq_per_sample = 8
        narrowed_batch = total_batch // scale

        input_ids = torch.arange(
            total_batch * seq_per_sample, device='cuda', dtype=torch.long
        ).reshape(total_batch, seq_per_sample)
        labels = input_ids.clone()
        loss_mask = torch.ones(total_batch, seq_per_sample, device='cuda', dtype=torch.float32)
        position_ids = (
            torch.arange(seq_per_sample, device='cuda').unsqueeze(0).expand(total_batch, -1).clone()
        )
        # Arbitrary extra per-sample field the LLM might consume (e.g. a
        # multimodal token-type tensor). Per-sample dim-0 == total_batch.
        token_type_ids = torch.zeros(total_batch, seq_per_sample, device='cuda', dtype=torch.long)

        packing_kwargs = _make_packing_kwargs(total_batch, seq_per_sample)

        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'attention_mask': None,
            'token_type_ids': token_type_ids,
            'packing_kwargs': packing_kwargs,
        }

        # Encoder output sized for THIS LLM-DP slot (the bridge already
        # narrowed it to narrowed_batch). 3D [seq, batch, hidden].
        hidden = 16
        detached_full = {
            'images': torch.randn(
                seq_per_sample, narrowed_batch, hidden, device='cuda', dtype=torch.float32
            )
        }

        # --- Contract after fix: fan-out + sequence packing is REJECTED ---
        # cu_seqlens / num_samples describe the full encoder-DP batch and cannot be
        # slot-narrowed without recomputing THD offsets, so _build_lm_microbatches
        # raises NotImplementedError rather than silently desyncing packed attention.
        with pytest.raises(NotImplementedError, match="sequence packing"):
            _build_lm_microbatches(
                detached_full, [batch], num_microbatches=1, encoder_grid=enc_grid, llm_grid=llm_grid
            )

        # --- Without packing, the per-sample fields narrow correctly to the slot ---
        batch_no_pack = {k: v for k, v in batch.items() if k != 'packing_kwargs'}
        mb = _build_lm_microbatches(
            detached_full,
            [batch_no_pack],
            num_microbatches=1,
            encoder_grid=enc_grid,
            llm_grid=llm_grid,
        )[0]
        for field in ('input_ids', 'labels', 'loss_mask', 'position_ids'):
            assert mb[field].shape[0] == narrowed_batch, (
                f"rank {rank}: {field} should be narrowed to {narrowed_batch} samples "
                f"on the batch dim, got {mb[field].shape[0]}"
            )
        print(
            f"\n=== rank {rank} fan-out packing rejection + narrowing OK ===\n"
            f"narrowed_batch={narrowed_batch}, total_batch={total_batch}, "
            f"scale={scale}, slot={slot}\n=== end rank {rank} ===\n",
            flush=True,
        )
