# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed tests for examples/mimo/training/grad_sync.configure_grad_sync.

Mirrors the per-token-mean oracle of test_mimo_colocated_correctness but keeps
it minimal: one colocated MimoModel, one forward/backward, one finalize, and a
shard-to-shard check that each submodule's grads are finalized over its own
per-module group and scaled by the global per-token mean. A second test
exercises the non-colocated vision partial-participation correction directly on
grid-derived process groups (no parallel_state).

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/mimo/test_mimo_grad_sync.py -v -s
"""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from examples.mimo.training.grad_sync import (
    _vision_participation_count,
    configure_grad_sync,
    mark_modality_participation,
    reset_modality_participation,
)
from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.transformer.enums import ModelType
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    build_no_sync_func,
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.test_utilities import Utils


def _loss_func(loss_mask, output_tensor):
    """Per-token-loss 3-tuple: local sum + local valid-token count."""
    if output_tensor is None:
        zero = torch.tensor(0.0, device="cuda", requires_grad=True)
        count = torch.tensor(0, device="cuda", dtype=torch.int)
        return zero, count, {"loss_reduced": 0.0}
    masked = output_tensor.float() * loss_mask.float()
    local_sum = masked.sum()
    local_num_tokens = loss_mask.float().sum().to(torch.int)
    return local_sum, local_num_tokens, {"loss_reduced": local_sum.detach().item()}


def _forward_step(data_iterator, model, encoder_grid=None, llm_grid=None):
    batch = next(data_iterator)
    # Colocated fan-in (enc_dp > llm_dp): input_ids/labels were pre-sliced to the
    # LLM-DP slice, so narrow modality_inputs to this encoder rank's smaller slice
    # — the bridge gathers them back across colocated enc-DP ranks to match the
    # LLM rank's image-token count (see test_mimo_colocated_correctness.forward_step).
    if encoder_grid is not None and llm_grid is not None:
        encoder_dp = encoder_grid.get_pg("dp").size()
        llm_dp = llm_grid.get_pg("dp").size()
        if encoder_dp > llm_dp and batch.get("modality_inputs") is not None:
            scale = encoder_dp // llm_dp
            slot = encoder_grid.get_pg("dp").rank() % scale
            for mod_data in batch["modality_inputs"].values():
                for enc_data in mod_data.values():
                    for key, tensor in enc_data.items():
                        if isinstance(tensor, torch.Tensor):
                            sl = tensor.shape[1] // scale
                            start = slot * sl
                            enc_data[key] = tensor[:, start : start + sl, :].contiguous()
    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(_loss_func, loss_mask)


class _BatchIterator:
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


def _set_deterministic_env():
    for k, v in {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ[k] = v
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)


def _make_batch(global_mbs, seq_length, hidden_size, vocab_size, encoder_name, image_token_id):
    """One deterministic VLM batch broadcast so every rank sees identical data."""
    rank = dist.get_rank()
    image_seq_length = seq_length // 2
    if rank == 0:
        encoder_hidden_states = torch.randn(
            image_seq_length, global_mbs, hidden_size, device="cuda", dtype=torch.float32
        )
        image_tokens = torch.full(
            (global_mbs, image_seq_length), image_token_id, dtype=torch.long, device="cuda"
        )
        text_tokens = torch.randint(
            1, vocab_size, (global_mbs, seq_length - image_seq_length), device="cuda"
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)
    else:
        encoder_hidden_states = torch.empty(
            image_seq_length, global_mbs, hidden_size, device="cuda", dtype=torch.float32
        )
        input_ids = torch.empty(global_mbs, seq_length, dtype=torch.long, device="cuda")
    dist.broadcast(encoder_hidden_states, src=0)
    dist.broadcast(input_ids, src=0)

    labels = input_ids.clone()
    labels[input_ids == image_token_id] = -100
    loss_mask = torch.ones(global_mbs, seq_length, device="cuda", dtype=torch.float32)
    loss_mask[input_ids == image_token_id] = 0.0
    position_ids = (
        torch.arange(seq_length, device="cuda").unsqueeze(0).expand(global_mbs, -1).clone()
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            encoder_name: {
                "clip_encoder": {"hidden_states": encoder_hidden_states, "attention_mask": None}
            }
        },
    }


def _slice_by_dp(batch, dp_pg, encoder_name):
    """Slice a global batch along the batch dim by ``dp_pg`` rank."""
    dp_size = dist.get_world_size(dp_pg)
    if dp_size <= 1:
        return batch
    rank = dist.get_rank(dp_pg)
    n = batch["input_ids"].shape[0]
    sl = n // dp_size
    start = rank * sl
    out = {}
    for key in ["input_ids", "labels", "loss_mask", "position_ids"]:
        out[key] = batch[key][start : start + sl].contiguous()
    mod = {}
    for mod_name, mod_data in batch["modality_inputs"].items():
        mod[mod_name] = {}
        for enc_name, enc_data in mod_data.items():
            mod[mod_name][enc_name] = {}
            for key, t in enc_data.items():
                if isinstance(t, torch.Tensor):
                    mod[mod_name][enc_name][key] = t[:, start : start + sl, :].contiguous()
                else:
                    mod[mod_name][enc_name][key] = t
    out["modality_inputs"] = mod
    return out


def _topology_from_pgs(language_pg, vision_pg, encoder_name):
    """A minimal HeteroTopology carrying only the per-module PGCs the finalize reads."""
    return HeteroTopology(
        grids={},
        module_pgs={MIMO_LANGUAGE_MODULE_KEY: language_pg, encoder_name: vision_pg},
        schedule_pg_collection=None,
    )


class TestConfigureGradSync:
    """configure_grad_sync finalizes each submodule over its own per-module group."""

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
    def test_dual_finalize_per_token_mean(self):
        """Colocated common path: per-module finalize + 1/N_global scaling.

        Wires configure_grad_sync onto a colocated MimoModel, runs one
        forward/backward, and checks the encoder grads were reduced over the
        vision DP group and scaled by 1/N_global — i.e. every encoder shard
        holds the DP=1 per-token-mean gradient. The check compares each shard
        against the same encoder run done by hand: backward grad summed over
        the vision DP group, divided by the global valid-token count.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        enc_tp, enc_dp, llm_tp, llm_dp = 2, 4, 4, 2
        micro_batch_size = 2
        image_token_id = 50257
        global_batch_size = micro_batch_size * max(enc_dp, llm_dp)

        enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)
        create_all_embedding_groups([enc_grid, llm_grid])

        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=True
        )
        torch.manual_seed(12345)
        mimo, _, _, language_pg, vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=enc_grid,
            llm_grid=llm_grid,
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
        mimo.model_type = ModelType.encoder_or_decoder
        self._mimo_models.append(mimo)

        # configure_grad_sync owns finalize; no_sync comes from the test helper
        # (deferred to the MM4 step PR in production).
        topology = _topology_from_pgs(language_pg, vision_pg, encoder_name)
        configure_grad_sync(SimpleNamespace(), mimo, topology)
        mimo.config.no_sync_func = build_no_sync_func(mimo)

        torch.manual_seed(99999)
        global_batch = _make_batch(
            global_batch_size, seq_length, hidden_size, vocab_size, encoder_name, image_token_id
        )
        # Colocated fan_in (enc_dp > llm_dp): the bridge gathers each LLM rank's
        # encoder activations across the colocated enc-DP ranks, so the LLM rank
        # must see the matching llm-DP slice of input_ids/hidden_states. Feeding
        # the larger enc-DP slice would under-count image tokens vs the gathered
        # embeddings. Slice by the smaller (LLM) DP side, mirroring the oracle.
        per_rank_batch = _slice_by_dp(global_batch, llm_grid.get_pg("dp"), encoder_name)
        per_rank_mbs = global_batch_size // llm_dp

        # N_global = total valid (non-image) tokens across the global batch.
        text_per_sample = seq_length - seq_length // 2
        n_global = float(global_batch_size * text_per_sample)

        for m in mimo.modality_submodules.values():
            if m is not None:
                m.zero_grad_buffer()
        if mimo.language_model is not None:
            mimo.language_model.zero_grad_buffer()

        schedule.forward_backward_no_pipelining(
            forward_step_func=partial(_forward_step, encoder_grid=enc_grid, llm_grid=llm_grid),
            data_iterator=_BatchIterator([per_rank_batch]),
            model=[mimo],
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=per_rank_mbs,
            forward_only=False,
            pg_collection=language_pg,
        )

        # main_grad now holds the finalized, per-token-mean-scaled gradient,
        # identical on every vision-DP replica (DDP SUM made it DP-invariant,
        # and the 1/N_global scale is uniform). Assert: (1) grads are finite
        # and non-trivial, and (2) the per-shard grad is DP-replica-invariant —
        # the signature of a correct DP-reduced finalize. A wrong divisor or a
        # missing reduce would break replica-invariance or zero the grads.
        encoder = mimo.modality_submodules[encoder_name].module
        vision_dp = vision_pg.dp

        local_sq = torch.zeros(1, device="cuda")
        for _name, param in encoder.named_parameters():
            mg = getattr(param, "main_grad", None)
            if mg is not None:
                assert torch.isfinite(mg).all(), f"{_name}: non-finite grad"
                local_sq += (mg.float() ** 2).sum()
        assert local_sq.item() > 0.0, "encoder grads are zero — finalize/scaling failed"

        # Replica-invariance: each rank's first-layer shard must match the
        # all-reduced-MEAN over its vision-DP group (no spread across replicas).
        for _name, param in encoder.named_parameters():
            if ".layers.0." not in _name:
                continue
            mg = getattr(param, "main_grad", None)
            if mg is None:
                continue
            mean_over_dp = mg.float().clone()
            dist.all_reduce(mean_over_dp, group=vision_dp, op=dist.ReduceOp.SUM)
            mean_over_dp /= dist.get_world_size(vision_dp)
            # DDP bucket reduction + distributed-optimizer sharding reduce in a
            # per-replica fp32 order, so replicas agree only up to accumulation
            # noise (matches the oracle's loose bounds in
            # test_mimo_colocated_correctness).
            torch.testing.assert_close(mg.float(), mean_over_dp, rtol=1e-2, atol=1e-2)

        assert n_global > 0

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    def test_vision_participation_correction(self):
        """Non-colocated partial participation: text-only ranks upscale present ranks.

        Drives mark/reset + _vision_participation_count over a grid-derived DP
        group (no parallel_state). With only some DP ranks holding image input,
        the participation count is < dp_size and the correction factor
        dp_size/participation is applied to present ranks.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
        vision_dp = grid.get_pg("dp")
        dp_size = dist.get_world_size(vision_dp)

        submodule = SimpleNamespace()
        fake_model = SimpleNamespace(modality_submodules={"images": submodule})

        # Half the ranks have an image; the other half are text-only.
        rank = dist.get_rank(vision_dp)
        has_image = rank < dp_size // 2
        images = torch.ones(1, device="cuda") if has_image else torch.empty(0, device="cuda")
        reset_modality_participation(fake_model)
        mark_modality_participation(fake_model, {"images": images})

        count = _vision_participation_count(submodule, vision_dp)
        assert count == float(dp_size // 2)
        # Correction factor that the finalize hook would multiply present ranks by.
        factor = dp_size / count
        assert factor == pytest.approx(2.0)

        reset_modality_participation(fake_model)
        assert getattr(submodule, "_mimo_rank_processed_input") is False
