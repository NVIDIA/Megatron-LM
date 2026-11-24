# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP training trajectory: GTP_remat=4 must train exactly like the unsharded model.

Every rank sees the same batch, so the GTP reduce-scatter (a gtp mean) hands each rank the shard
of the full gradient. Both runs therefore take the same plain SGD step on the same weights; only
BF16 reduction order differs. A step that is off by any factor (e.g. 1/gtp) moves the loss well
outside the tolerance after the first step.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTPShardedParam,
    reset_gtp_state,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

HIDDEN = 256
NUM_LAYERS = 2
SEQ = 32
STEPS = 4
LR = 2.0


def _train(gtp_remat_size, initial=None):
    """Build the stack at ``gtp_remat_size``, load ``initial`` and take STEPS SGD steps.

    Returns (per-step losses, unsharded initial weights).
    """
    ps.destroy_model_parallel()
    reset_gtp_state()
    ps.initialize_model_parallel(gtp_remat_size=gtp_remat_size)
    model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        ffn_hidden_size=4 * HIDDEN,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    layers = torch.nn.ModuleList(
        spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=pg)
        for i in range(NUM_LAYERS)
    ).cuda()
    gtp_rank = pg.gtp_remat.rank()
    with torch.no_grad():
        for name, p in layers.named_parameters():
            if initial is None:
                dist.broadcast(p, src=0)
            elif isinstance(p, GTPShardedParam):
                p.copy_(initial[name].chunk(gtp_remat_size)[gtp_rank])
                p.main_grad = torch.zeros_like(p)
            else:
                p.copy_(initial[name])
    assert any(isinstance(p, GTPShardedParam) for p in layers.parameters()) == (gtp_remat_size > 1)
    weights = {n: p.detach().clone() for n, p in layers.named_parameters()}

    # One fixed batch: the loss then falls step over step, at a rate set by the step size.
    batch = torch.randn(
        SEQ,
        1,
        HIDDEN,
        dtype=torch.bfloat16,
        device='cuda',
        generator=torch.Generator('cuda').manual_seed(0),
    )
    losses = []
    for _ in range(STEPS):
        x = batch
        for layer in layers:
            x, _ = layer(x, attention_mask=None)
        loss = x.float().square().mean()
        loss.backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        losses.append(loss.item())
        with torch.no_grad():
            for p in layers.parameters():
                grad = p.main_grad if isinstance(p, GTPShardedParam) else p.grad
                p.sub_(LR * grad)
                grad.zero_()
    return losses, weights


def test_gtp_loss_trajectory_matches_baseline():
    """GTP_remat=4 per-step losses match the unsharded run to BF16 reduction-order noise."""
    if dist.get_world_size() != 4:
        pytest.skip("Requires 4 ranks")
    baseline, initial = _train(1)
    gtp, _ = _train(4, initial)
    print(f"baseline={baseline}\ngtp_remat4={gtp}", flush=True)
    # The loss must actually move, or a wrong step size would go unnoticed.
    assert baseline[-1] < 0.99 * baseline[0], baseline
    # Same GEMMs on the same weights, and an exact gtp mean of identical wgrads: observed
    # bit-exact, so 1e-4 is all margin. A 1/gtp step ends ~20% off the baseline loss.
    torch.testing.assert_close(torch.tensor(gtp), torch.tensor(baseline), rtol=1e-4, atol=0)
