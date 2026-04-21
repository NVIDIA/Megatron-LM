# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Gradient-scaling correctness for colocated MimoModel under heterogeneous DP.

Verifies that a heterogeneous-DP MimoModel configured with
``gradient_reduce_div_factor=1`` produces the same post-step encoder
weights as an **equal-DP** reference built on the SAME encoder TP/DP
layout as the dist model (so the bridge is the identity passthrough —
``BridgeDirection.EQUAL`` in ``ColocatedBridgeCommunicator``). Under
correct grad scaling, both configs yield the DP=1 gradient on every
encoder shard, so the Adam update lands on identical values and the
sharded post-step weights compare directly.

Why an equal-DP reference is the right oracle:
  * Encoder sharding matches exactly — ref and dist both use
    ``enc_tp=dist_enc_tp, enc_dp=dist_enc_dp``. Shards line up 1:1,
    so there is no gather-and-slice in the weight comparison and no
    TP=1-vs-TP>1 accumulation-order drift to contend with.
  * ``enc_dp == llm_dp`` on the ref side → the bridge is identity and
    every encoder rank feeds its colocated LLM rank with no
    redistribution collective.
  * Loss is the num+den global-mean CE all-reduced on the LLM DP group
    (same as ``test_mimo_colocated_e2e.py``). With
    ``gradient_reduce_div_factor=1``, summing local grads across DP
    recovers the DP=1 gradient on both sides.

LLM TP differs between ref (``llm_tp=dist_enc_tp``) and dist
(``llm_tp=dist_llm_tp``), so ref's LLM weights are copied into dist via
all-gather-across-ref-TP + slice-for-dist-TP. The LLM forward then
diverges numerically by bf16 accumulation order, but the aggregate
gradient that flows back into the encoder remains the DP=1 gradient in
both models, which is what the post-step encoder weight oracle checks.

If the heterogeneous-DP scaling is wrong (e.g. dividing by encoder_dp
when it should be 1), the dist encoder's post-step weights diverge from
the ref encoder's weights — a single Adam step is enough to detect.

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
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.test_mimo_colocated_e2e import forward_step
from tests.unit_tests.test_utilities import Utils


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

    Mirrors the wiring in ``run_colocated_test`` so both dist and ref
    models drive the same DDP/optimizer path through the schedule.
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

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model], num_tokens=None, pg_collection=language_pg
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads([submodule], num_tokens=None, pg_collection=vision_pg)

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
):
    """Generate global batches on rank 0 and broadcast so every rank sees
    identical data. Dist pre-slices per rank; ref consumes the full batch.
    """
    rank = dist.get_rank()
    image_seq_length = seq_length // 2
    batches = []

    for _ in range(num_batches):
        if rank == 0:
            encoder_hidden_states = torch.randn(
                image_seq_length,
                global_mbs,
                hidden_size,
                device='cuda',
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
            )
            input_ids = torch.empty(global_mbs, seq_length, dtype=torch.long, device='cuda')

        dist.broadcast(encoder_hidden_states, src=0)
        dist.broadcast(input_ids, src=0)

        labels = input_ids.clone()
        labels[input_ids == image_token_id] = -100
        loss_mask = torch.ones(global_mbs, seq_length, device='cuda', dtype=torch.float32)
        loss_mask[input_ids == image_token_id] = 0.0
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


def _assert_encoder_weights_match(
    ref_module, dist_module, rtol=1e-3, atol=1e-3
):
    """Assert every dist encoder shard matches the ref encoder shard.

    Caller is responsible for ensuring ref and dist have the same encoder TP
    layout (same ``enc_tp`` and ``enc_dp``), so each rank's shards line up
    1:1 and can be compared directly. Under correct grad scaling and
    identical initial state, one Adam step yields shard-wise equal post-step
    weights — modulo bf16 rounding from the LLM TP layout differing between
    the two models.
    """
    ref_params = dict(ref_module.named_parameters())

    mismatches = []
    for name, dist_param in dist_module.named_parameters():
        ref_param = ref_params[name]
        assert ref_param.shape == dist_param.shape, (
            f"Param '{name}': ref.shape={tuple(ref_param.shape)} != "
            f"dist.shape={tuple(dist_param.shape)} — caller must match encoder TP."
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

    The critical invariant: with ``gradient_reduce_div_factor=1`` and a
    num+den global-mean CE, both encoder and LLM DDP reductions are pure
    SUMs. The aggregate gradient on every encoder shard equals the DP=1
    gradient. The reference uses the same encoder TP/DP as dist but with
    ``enc_tp == llm_tp`` and ``enc_dp == llm_dp`` (identity bridge), so
    after one Adam step the dist model's sharded weights match the ref
    model's sharded weights within bf16 precision.

    If the scaling factor were wrong (e.g., dividing by encoder_dp when
    it should be 1), the encoder's reduced grad would be skewed and
    post-step weights would diverge — a single optimizer step is
    sufficient to detect.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
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
    def test_dist_matches_dp1_reference_post_step_weights(
        self, enc_tp, enc_dp, llm_tp, llm_dp
    ):
        """Heterogeneous-DP dist post-step encoder weights match equal-DP reference.

        Builds two MimoModels on every rank:

        * Dist: the heterogeneous TP/DP config under test, using
          ``gradient_reduce_div_factor=1`` to pure-SUM the DDP reductions.
        * Ref: equal-DP uniform with ``enc_tp=dist_enc_tp``,
          ``enc_dp=dist_enc_dp``, ``llm_tp=dist_enc_tp``,
          ``llm_dp=dist_enc_dp`` — bridge is
          ``BridgeDirection.EQUAL`` (identity passthrough), and the
          encoder TP sharding matches dist's exactly so shards line up
          1:1 for comparison.

        Both models use ``gradient_reduce_div_factor=1``, so the
        num+den mean-CE summed across DP yields the DP=1 gradient on
        every encoder shard. LLM TP differs between the two models,
        which introduces bf16 accumulation-order drift in the gradient
        flowing back to the encoder but does not change the DP=1
        invariant that the post-step encoder oracle checks.

        Reference weights are copied into the distributed model so both
        start from identical state. One Adam step later, the dist shards
        should match the ref shards within bf16 precision.
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
        num_microbatches = 1

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

        # Both configs use gradient_reduce_div_factor=1: under num+den mean CE,
        # DDP must pure-SUM regardless of dp_size for the reduced grad to
        # equal the DP=1 gradient on every rank.
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            bucket_size=10000,
            use_distributed_optimizer=True,
            gradient_reduce_div_factor=1,
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
        )
        dist_mimo.model_type = ModelType.encoder_or_decoder

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
        )
        ref_mimo.model_type = ModelType.encoder_or_decoder

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
            bf16=True,
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
        ref_success, ref_grad_norm, _ = ref_optimizer.step()
        assert ref_success, "Ref optimizer step failed"
        assert ref_grad_norm is not None and ref_grad_norm > 0, (
            f"Ref grad_norm={ref_grad_norm}"
        )

        # Main oracle: post-step encoder shards match between dist and
        # ref. Pre-step loss is NOT compared: ref and dist share encoder
        # TP and per-rank encoder batch but diverge on the LLM side
        # (different llm_tp, different per-rank LLM batch size), so the
        # LLM forward — and therefore the reduced loss — differs by bf16
        # accumulation noise. That noise propagates into each model's
        # encoder gradient, so the post-step comparison uses slightly
        # looser tolerances than bf16 rounding alone would require.
        _assert_encoder_weights_match(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            rtol=1e-3,
            atol=1e-3,
        )
