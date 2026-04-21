# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Gradient-scaling correctness for colocated MimoModel under heterogeneous DP.

Verifies that a heterogeneous-DP MimoModel configured with
``gradient_reduce_div_factor=1`` produces the same post-step encoder
weights as a reference that simulates DP=1 by running the full global
batch redundantly on every rank (TP=1, DP=world_size,
``gradient_reduce_div_factor=1``). Under correct grad scaling, both
configurations yield the DP=1 gradient on every encoder shard, so the
Adam update lands on identical values (TP-sliced for the dist model).

Why the reference is equivalent to DP=1:
  * Loss is the num+den global-mean CE all-reduced on the LLM DP group
    only (same as ``test_mimo_colocated_e2e.py``).
  * Every ref rank sees the full batch, so each rank's ``local_num`` is
    the full-batch CE sum and ``local_den`` is the full-batch token
    count. All-reduce over DP=world_size gives ``reduced_den =
    world_size * local_den``; the per-rank grad scalar is
    ``1/reduced_den = 1/(world_size*local_den)``.
  * DDP all-reduce over DP=world_size with ``gradient_reduce_div_factor=1``
    sums identical local grads across ``world_size`` ranks, recovering
    ``1/local_den * full_batch_grad`` — the DP=1 gradient.

If the heterogeneous-DP scaling is wrong (e.g. dividing by encoder_dp
when it should be 1), the dist encoder's post-step weights diverge from
the TP-sliced ref weights — a single Adam step is enough to detect.

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


def _slice_global_batch_for_dist(global_batch, encoder_grid, llm_grid):
    """Pre-slice a global batch to the per-rank batch that ``forward_step`` expects.

    ``forward_step`` assumes each rank already has its LLM-DP slice
    (fan-in) or encoder-DP slice (fan-out); this helper performs that
    slicing so both models can consume the same underlying global batch.
    """
    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if enc_dp > llm_dp:
        split_dp = llm_dp
        split_rank = llm_grid.get_pg("dp").rank()
    elif llm_dp > enc_dp:
        split_dp = enc_dp
        split_rank = encoder_grid.get_pg("dp").rank()
    else:
        return global_batch

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


def _copy_ref_params_to_dist(ref_module, dist_module, dist_tp_group):
    """Copy TP=1 reference params into a TP-sharded dist module, shard by shard.

    Slices along ``partition_dim`` (0 = ColumnParallel, 1 = RowParallel) and
    copies directly for replicated params (partition_dim == -1). Must be
    called **before** constructing the distributed optimizer, which clones
    current param data into fp32 master weights at __init__.
    """
    tp_rank = dist.get_rank(dist_tp_group)
    tp_size = dist.get_world_size(dist_tp_group)
    ref_params = dict(ref_module.named_parameters())

    with torch.no_grad():
        for name, dist_param in dist_module.named_parameters():
            assert name in ref_params, f"Param '{name}' in dist but not in ref"
            ref_param = ref_params[name]
            partition_dim = getattr(dist_param, 'partition_dim', -1)

            if partition_dim >= 0 and tp_size > 1:
                ref_slice = torch.tensor_split(
                    ref_param.data, tp_size, dim=partition_dim
                )[tp_rank]
            else:
                ref_slice = ref_param.data

            assert ref_slice.shape == dist_param.shape, (
                f"Param '{name}': ref_slice.shape={tuple(ref_slice.shape)} != "
                f"dist.shape={tuple(dist_param.shape)} "
                f"(partition_dim={partition_dim}, tp_rank={tp_rank}, tp_size={tp_size})"
            )
            dist_param.data.copy_(ref_slice.to(dist_param.dtype))


def _assert_encoder_weights_match(
    ref_module, dist_module, dist_tp_group, rtol=1e-4, atol=1e-4
):
    """Assert every dist encoder param matches the TP-sliced ref param.

    Under correct grad scaling and identical initial state, one Adam step
    produces bit-equal post-step weights on each encoder shard (modulo
    numerical noise from reduction order and bf16 rounding).
    """
    tp_rank = dist.get_rank(dist_tp_group)
    tp_size = dist.get_world_size(dist_tp_group)
    ref_params = dict(ref_module.named_parameters())

    mismatches = []
    for name, dist_param in dist_module.named_parameters():
        ref_param = ref_params[name]
        partition_dim = getattr(dist_param, 'partition_dim', -1)

        if partition_dim >= 0 and tp_size > 1:
            ref_slice = torch.tensor_split(
                ref_param.data, tp_size, dim=partition_dim
            )[tp_rank]
        else:
            ref_slice = ref_param.data

        try:
            torch.testing.assert_close(
                dist_param.data,
                ref_slice.to(dist_param.dtype),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as e:
            mismatches.append((name, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  {n}: {msg}" for n, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: {len(mismatches)} encoder param(s) diverged between "
            f"heterogeneous-DP dist model and DP=1 reference:\n{details}"
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
    """Verify heterogeneous-DP encoder grad scaling against a DP=1 reference.

    The critical invariant: with ``gradient_reduce_div_factor=1`` and a
    num+den global-mean CE, both encoder and LLM DDP reductions are pure
    SUMs. The aggregate gradient on every encoder shard equals the DP=1
    gradient, so after one Adam step the dist model's sharded weights
    match the TP-sliced reference weights within bf16 precision.

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
        """Heterogeneous-DP dist post-step encoder weights match DP=1 reference.

        Builds two MimoModels on every rank:

        * Dist: the heterogeneous TP/DP config under test, using
          ``gradient_reduce_div_factor=1`` to pure-SUM the DDP reductions.
        * Ref: TP=1, DP=world_size, ``gradient_reduce_div_factor=1``.
          Every rank receives the full global batch. Under the num+den
          mean CE, ``reduced_den = world_size * local_den`` and the
          post-sum reduced grad recovers the DP=1 gradient.

        Reference weights are copied into the distributed model so both
        start from identical state. One Adam step later, the dist shards
        should match the TP-sliced ref params to within bf16 precision.
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

        # Grids: dist is heterogeneous; ref is TP=1 + DP=world_size.
        dist_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        dist_llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)
        ref_enc_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
        ref_llm_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
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

        # Reference with TP=1 and DP=world_size. Same seed so CPU init is
        # deterministic; param names line up with dist's (TP sharding only
        # changes shape, not name).
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

        # Force identical initial state: copy ref's full params into dist's
        # TP-sharded params. TP init schemes differ subtly from TP=1 init
        # (different partition_stride, seed consumption order), so without
        # this copy dist and ref would diverge from step 0.
        _copy_ref_params_to_dist(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            dist_enc_grid.get_pg("tp"),
        )
        _copy_ref_params_to_dist(
            ref_mimo.language_model.module,
            dist_mimo.language_model.module,
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

        # One optimizer step on dist (heterogeneous forward_step slicing).
        dist_optimizer.zero_grad()
        dist_losses = _run_forward_backward(
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
        ref_losses = _run_forward_backward(
            mimo_model=ref_mimo,
            batches=global_batches,
            enc_grid=ref_enc_grid,
            llm_grid=ref_llm_grid,
            encoder_name=encoder_name,
            language_pg=ref_language_pg,
            micro_batch_size=global_batch_size,
            seq_length=seq_length,
            num_microbatches=num_microbatches,
        )
        ref_success, ref_grad_norm, _ = ref_optimizer.step()
        assert ref_success, "Ref optimizer step failed"
        assert ref_grad_norm is not None and ref_grad_norm > 0, (
            f"Ref grad_norm={ref_grad_norm}"
        )

        # Sanity: dist and ref see the same underlying batch, so the
        # reduced global-mean CE loss matches up to numerical noise.
        for dl, rl in zip(dist_losses, ref_losses):
            dv = dl['loss_reduced']
            rv = rl['loss_reduced']
            if isinstance(dv, torch.Tensor):
                dv = dv.item()
            if isinstance(rv, torch.Tensor):
                rv = rv.item()
            assert abs(dv - rv) < 5e-3, (
                f"dist/ref losses diverge before any step: {dv} vs {rv}"
            )

        # Main oracle: post-step encoder weights match shard-wise.
        _assert_encoder_weights_match(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            dist_enc_grid.get_pg("tp"),
            rtol=1e-4,
            atol=1e-4,
        )
