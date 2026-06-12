# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Adversarial test (NMFW-19): fan-out narrows 3D mRoPE position_ids wrongly.

Finding under test
------------------
``_build_lm_microbatches`` in ``megatron/core/models/mimo/colocated_schedule.py``
narrows every LLM-side passthrough field (``input_ids``, ``labels``,
``loss_mask``, ``position_ids``) for fan-out (``llm_dp > enc_dp``) via the
inner helper ``_maybe_narrow``, which unconditionally slices ``tensor`` on
``dim=0``::

    bs = tensor.shape[0]
    ss = bs // fan_out_scale
    return tensor[fan_out_slot * ss : (fan_out_slot + 1) * ss].contiguous()

This is correct for 2D ``[batch, seq]`` fields, where ``dim=0`` is the batch.

For **multimodal RoPE** the language model consumes ``position_ids`` shaped
``[rope_dim, batch, seq]`` — exactly the layout ``MimoModel.get_text_embeddings``
explicitly supports at ``model/base.py:311-314`` (it indexes
``position_ids[0, batch_idx, seq_idx]`` for the 3D case). For that layout
``dim=0`` is ``rope_dim``, **not** the batch dimension. ``_maybe_narrow``
therefore slices away RoPE channels and leaves the batch dimension
un-narrowed, so the LLM-side ``position_ids`` no longer line up with the
bridge-narrowed encoder embeddings or with the (correctly narrowed)
``input_ids`` / ``labels`` / ``loss_mask``.

Note that ``attention_mask`` got a dedicated ``_maybe_narrow_attn`` guard for
non-batch-first layouts; ``position_ids`` got none.

What this test does
-------------------
Rather than spin up a full model, it drives the pure tensor-narrowing function
``_build_lm_microbatches`` directly with real HyperCommGrids in a fan-out
config (``enc_tp=4, enc_dp=2`` / ``llm_tp=1, llm_pp=2, llm_dp=4`` on 8 GPUs,
``fan_out_scale = llm_dp // enc_dp = 2``). It feeds a 3D mRoPE
``position_ids`` of shape ``[rope_dim=3, batch=B, seq]`` alongside 2D
``input_ids`` / ``labels`` / ``loss_mask`` of shape ``[batch=B, seq]``.

Correct behavior: the narrowed ``position_ids`` is ``[3, B//scale, seq]`` —
``rope_dim`` intact (still 3), batch halved, lining up with the narrowed 2D
fields whose batch is also ``B//scale``.

Buggy behavior (current code): ``_maybe_narrow`` slices ``dim=0`` (rope_dim),
yielding ``[3//scale, B, seq] = [1, B, seq]`` — rope channels destroyed and
batch left un-narrowed. The assertion below fails with a clear message.

If the code is fixed (e.g. an ndim/shape-aware narrow that targets the batch
dim), this test PASSES — disproving the finding.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/mimo/test_mimo_colocated_fanout_position_ids.py -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.models.mimo.colocated_schedule import _build_lm_microbatches
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
)
from tests.unit_tests.test_utilities import Utils

ROPE_DIM = 3


def _set_deterministic_env():
    for k, v in {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ[k] = v


def _make_batch(enc_per_rank_batch, seq_length, hidden_size, encoder_name, image_seq_length):
    """Build one encoder-DP-sized per-rank batch with a 3D mRoPE position_ids.

    ``position_ids`` is shaped ``[rope_dim, batch, seq]`` — the multimodal-RoPE
    layout consumed at ``model/base.py:311-314``. The 2D fields
    (``input_ids`` / ``labels`` / ``loss_mask``) keep the conventional
    ``[batch, seq]`` layout so the test can verify that, after narrowing, the
    batch dimension of every field agrees.
    """
    input_ids = torch.arange(
        enc_per_rank_batch * seq_length, device='cuda', dtype=torch.long
    ).reshape(enc_per_rank_batch, seq_length)
    labels = input_ids.clone()
    loss_mask = torch.ones(enc_per_rank_batch, seq_length, device='cuda', dtype=torch.float32)

    # 3D mRoPE position_ids: [rope_dim, batch, seq]. Encode (rope_channel,
    # batch_idx) into the values so a wrong-dim slice is detectable by content
    # as well as by shape.
    rope = torch.arange(ROPE_DIM, device='cuda', dtype=torch.long).view(ROPE_DIM, 1, 1)
    bidx = torch.arange(enc_per_rank_batch, device='cuda', dtype=torch.long).view(1, -1, 1)
    sidx = torch.arange(seq_length, device='cuda', dtype=torch.long).view(1, 1, -1)
    position_ids = rope * 1_000_000 + bidx * 1_000 + sidx
    assert position_ids.shape == (ROPE_DIM, enc_per_rank_batch, seq_length)

    encoder_hidden = torch.randn(
        image_seq_length, enc_per_rank_batch, hidden_size, device='cuda', dtype=torch.float32
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            encoder_name: {
                "clip_encoder": {"hidden_states": encoder_hidden, "attention_mask": None}
            }
        },
    }


class TestFanOutNarrows3DmRoPEPositionIds:
    """Fan-out must narrow 3D ``[rope_dim, batch, seq]`` position_ids on batch."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def setup_method(self):
        _set_deterministic_env()

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    def test_fan_out_narrows_position_ids_on_batch_not_rope_dim(self):
        """3D mRoPE position_ids must be narrowed on batch (dim=1), not rope_dim.

        Fan-out config ``enc_dp=2`` / ``llm_dp=4`` → ``fan_out_scale=2``.

        Expected (correct): narrowed position_ids has shape
        ``[ROPE_DIM, enc_batch // scale, seq]`` — rope_dim intact, batch halved,
        agreeing with the narrowed 2D ``input_ids`` batch.

        Actual (buggy current code): ``_maybe_narrow`` slices dim 0 (rope_dim),
        yielding ``[ROPE_DIM // scale, enc_batch, seq] = [1, enc_batch, seq]`` —
        rope channels dropped, batch un-narrowed. The shape assertion fails.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_name = "images"
        enc_tp, enc_dp = 4, 2
        llm_tp, llm_pp, llm_dp = 1, 2, 4
        scale = llm_dp // enc_dp  # 2
        assert scale >= 2, "fan-out scale must be >=2 to exercise the bug"

        hidden_size, seq_length = 32, 8
        image_seq_length = seq_length // 2
        num_microbatches = 1
        # Encoder-DP per-rank batch must be divisible by fan_out_scale (so the
        # narrowed batch is non-degenerate) and by num_microbatches.
        enc_per_rank_batch = scale * num_microbatches * 2  # 4

        enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
        create_all_embedding_groups([enc_grid, llm_grid])

        rank = dist.get_rank()
        try:
            all_batches = [
                _make_batch(
                    enc_per_rank_batch, seq_length, hidden_size, encoder_name, image_seq_length
                )
                for _ in range(num_microbatches)
            ]

            # Stand in for the bridge-narrowed encoder embeddings: fan-out
            # narrows encoder output to the LLM-DP rank's slot, so the detached
            # embeddings the LLM consumes have batch == enc_per_rank_batch //
            # scale. 3D [seq, batch, hidden] layout (the bridge's 3D output).
            llm_batch = enc_per_rank_batch // scale
            detached_full = {
                "clip_encoder": torch.randn(
                    image_seq_length, llm_batch, hidden_size, device='cuda', dtype=torch.float32
                ).requires_grad_(True)
            }

            lm_data = _build_lm_microbatches(
                detached_full, all_batches, num_microbatches, enc_grid, llm_grid
            )

            assert len(lm_data) == num_microbatches
            mb = lm_data[0]
            pos = mb["position_ids"]
            ids = mb["input_ids"]
            enc_emb = mb["encoder_embeddings"]["clip_encoder"]

            # The 2D fields are narrowed correctly on dim 0 (batch).
            assert ids.shape == (
                llm_batch,
                seq_length,
            ), f"input_ids should narrow batch to {llm_batch}; got {tuple(ids.shape)}"
            # The encoder embeddings carry batch == llm_batch on dim 1.
            assert enc_emb.shape[1] == llm_batch, (
                f"encoder_embeddings batch (dim1) should be {llm_batch}; "
                f"got {tuple(enc_emb.shape)}"
            )

            # CORE ASSERTION: 3D mRoPE position_ids must keep rope_dim intact
            # and narrow the BATCH dim so it lines up with input_ids and the
            # bridge-narrowed encoder embeddings.
            #
            # Buggy code slices dim 0 (rope_dim) → shape [ROPE_DIM//scale,
            # enc_per_rank_batch, seq] = [1, 4, 8], failing this assertion.
            expected = (ROPE_DIM, llm_batch, seq_length)
            assert pos.shape == expected, (
                f"Rank {rank}: 3D mRoPE position_ids narrowed on the WRONG "
                f"dimension. Expected {expected} (rope_dim intact, batch "
                f"halved), but got {tuple(pos.shape)}. _maybe_narrow sliced "
                f"dim=0 (rope_dim) instead of the batch dim. enc_batch="
                f"{enc_per_rank_batch}, fan_out_scale={scale}."
            )

            # Stronger content check: the surviving slice must contain ALL
            # rope channels (0..ROPE_DIM-1) and a CONTIGUOUS batch sub-range of
            # size llm_batch, matching this LLM-DP rank's fan-out slot.
            rope_channels = (pos[:, 0, 0] // 1_000_000).tolist()
            assert rope_channels == list(range(ROPE_DIM)), (
                f"Rank {rank}: rope channels corrupted — expected "
                f"{list(range(ROPE_DIM))}, got {rope_channels}. The narrow "
                f"sliced rope_dim instead of batch."
            )
            batch_ids_present = ((pos[0, :, 0] // 1_000) % 1_000).tolist()
            assert len(batch_ids_present) == llm_batch, (
                f"Rank {rank}: position_ids batch dim should be {llm_batch}; "
                f"got {len(batch_ids_present)} entries: {batch_ids_present}."
            )

            _passed = True
        except Exception:
            import traceback as _tb

            print(
                f"\n=== rank {rank} TEST EXCEPTION ===\n"
                f"config: enc_tp={enc_tp} enc_dp={enc_dp} llm_tp={llm_tp} "
                f"llm_pp={llm_pp} llm_dp={llm_dp}\n{_tb.format_exc()}\n"
                f"=== end rank {rank} exception ===\n",
                flush=True,
            )
            _passed = False

        # Reduce pass/fail across ranks so a single-rank failure fails the test
        # everywhere (avoids asymmetric NCCL teardown hiding the assertion).
        flag = torch.tensor([1 if _passed else 0], device='cuda', dtype=torch.int)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        assert flag.item() == 1, (
            "Fan-out 3D mRoPE position_ids narrowing failed on at least one "
            "rank (see per-rank TEST EXCEPTION above)."
        )
