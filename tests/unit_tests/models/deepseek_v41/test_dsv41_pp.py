# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU tests (``torchrun --nproc_per_node=2``): two-stage pipeline handoff of the tiny model.

The single-pass hyper-connection aggregation weights (``H_pre``) of the last layer of stage 0
travel to stage 1 appended to the stream tensor. These tests drive the two stages by hand
(send / recv of the stage output and of its gradient) and check that full activation
recomputation on stage 1 reproduces the eager loss and gradients, including the gradient that
flows back into the incoming handoff tensor.
"""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.deepseek_v41.layer_specs import hybrid_dsv41_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.models.deepseek_v41.test_dsv41_model import SEQ, VOCAB, make_tiny_config
from tests.unit_tests.test_utilities import Utils

# 6 model layers (12 pattern positions); the split keeps KV / index sources with their consumers
# and every Engram layer on the first stage.
PATTERN_PP2 = "WEWEDEDE|DEDE"
BATCH = 2


def _build(config, rank):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv41_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=PATTERN_PP2,
        position_embedding_type="none",
        pre_process=(rank == 0),
        post_process=(rank == 1),
    ).cuda()


def _run_two_stage(model, config, ids, pos, labels):
    """Manual 1F1B step of one microbatch; returns the loss on stage 1 (None on stage 0)."""
    rank = torch.distributed.get_rank()
    n, c = config.num_residual_streams, config.hidden_size
    if rank == 0:
        out = model(ids, pos, attention_mask=None)
        assert out.shape == (SEQ, BATCH, n * c + n), out.shape
        torch.distributed.send(out.detach().contiguous(), dst=1)
        grad = torch.empty_like(out)
        torch.distributed.recv(grad, src=1)
        out.backward(grad)
        return None
    handoff = torch.empty((SEQ, BATCH, n * c + n), dtype=config.params_dtype, device="cuda")
    torch.distributed.recv(handoff, src=0)
    handoff.requires_grad_(True)
    model.set_input_tensor(handoff)
    loss = model(ids, pos, attention_mask=None, labels=labels).mean()
    loss.backward()
    torch.distributed.send(handoff.grad.contiguous(), dst=0)
    return loss.detach()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestPipelineHandoff:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        if Utils.world_size != 2:
            pytest.skip("needs torchrun with exactly 2 processes")
        Utils.initialize_model_parallel(1, 2)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("method,num_layers", [("uniform", 2), ("uniform", 4), ("block", 1)])
    def test_recompute_matches_eager_across_stages(self, method, num_layers):
        rank = torch.distributed.get_rank()
        torch.manual_seed(5)
        model_parallel_cuda_manual_seed(5)
        eager_cfg = make_tiny_config(engram=True)
        eager = _build(eager_cfg, rank)
        torch.manual_seed(5)
        model_parallel_cuda_manual_seed(5)
        ckpt_cfg = make_tiny_config(
            engram=True,
            recompute_granularity="full",
            recompute_method=method,
            recompute_num_layers=num_layers,
        )
        ckpt = _build(ckpt_cfg, rank)
        ckpt.load_state_dict(eager.state_dict())

        gen = torch.Generator(device="cuda").manual_seed(9)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda", generator=gen)
        pos = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, SEQ)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda", generator=gen)

        losses = [_run_two_stage(m, eager_cfg, ids, pos, labels) for m in (eager, ckpt)]
        if rank == 1:
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-3, atol=1e-3)
        n_checked = 0
        for (name, p_e), (_, p_c) in zip(eager.named_parameters(), ckpt.named_parameters()):
            if p_e.grad is None:
                assert p_c.grad is None, name
                continue
            n_checked += 1
            torch.testing.assert_close(
                p_e.grad.float(), p_c.grad.float(), rtol=2e-2, atol=2e-2, msg=name
            )
        assert n_checked > 0

    def test_stage_one_uses_incoming_h_pre(self):
        """Stage 1 must not fall back to the identity mix: perturbing the incoming H_pre
        channels changes its output."""
        rank = torch.distributed.get_rank()
        torch.manual_seed(6)
        model_parallel_cuda_manual_seed(6)
        cfg = make_tiny_config(
            engram=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=2,
        )
        model = _build(cfg, rank)  # train mode so the recompute path runs (dropout is 0)
        n, c = cfg.num_residual_streams, cfg.hidden_size
        gen = torch.Generator(device="cuda").manual_seed(10)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda", generator=gen)
        pos = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, SEQ)
        if rank == 0:
            with torch.no_grad():
                out = model(ids, pos, attention_mask=None)
            torch.distributed.send(out.contiguous(), dst=1)
            return
        handoff = torch.empty((SEQ, BATCH, n * c + n), dtype=cfg.params_dtype, device="cuda")
        torch.distributed.recv(handoff, src=0)
        perturbed = handoff.clone()
        perturbed[..., -n:] = perturbed[..., -n:].flip(-1) + 0.5
        outs = []
        for tensor in (handoff, perturbed):
            model.set_input_tensor(tensor)
            with torch.no_grad():
                outs.append(model(ids, pos, attention_mask=None))
        assert not torch.allclose(outs[0].float(), outs[1].float(), atol=1e-3)
