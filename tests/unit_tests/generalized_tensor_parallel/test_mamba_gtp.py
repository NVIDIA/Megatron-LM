# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP + Mamba gradients match the unsharded layer.

Mamba puts in_proj and out_proj on one GTP chain, and in_proj's output width (2*d_inner +
2*groups*d_state + heads) is not a multiple of the GTP alignment, so its shards carry padding
rows. Each rank sees the same batch, so the GTP reduce-scatter (a gtp mean) must hand every rank
exactly its shard of the full gradient.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import fp8_autocast, fp8_model_init

from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    reset_gtp_state,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.tensor_parallel.gtp_api import (
    classify_gtp_remat_chains,
    gtp_native_fp8_load_context,
    is_gtp_param,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _requires_mxfp8

HIDDEN = 256
SEQ = 32
GTP = 4


def _run(gtp_remat_size, native_fp8, initial=None):
    """One fwd+bwd of a Mamba layer -> (input grad, {name: unsharded grad}, initial weights)."""
    ps.destroy_model_parallel()
    reset_gtp_state()
    ps.initialize_model_parallel(gtp_remat_size=gtp_remat_size)
    model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        # in_proj rows: 2*1024 + 2*2*16 + 32 = 2144 (MXFP8-aligned unsharded), padded to 2176 at GTP=4.
        mamba_num_heads=32,
        mamba_head_dim=32,
        mamba_state_dim=16,
        mamba_num_groups=2,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
    )
    submodules = MambaLayerSubmodules(
        mixer=ModuleSpec(
            module=MambaMixer,
            submodules=MambaMixerSubmodules(
                in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
            ),
        ),
        mamba_bda=get_bias_dropout_add,
    )
    use_fp8_params = native_fp8 and gtp_remat_size > 1
    with fp8_model_init(enabled=use_fp8_params, recipe=MXFP8BlockScaling()):
        layer = MambaLayer(config, submodules, layer_number=1, pg_collection=pg).cuda()
    # Native-FP8 shards are FP8 tensor subclasses, not GTPShardedParam.
    gtp_params = {n: p for n, p in layer.named_parameters() if is_gtp_param(p)}
    if gtp_remat_size > 1:
        assert {n.rsplit('.', 2)[-2] for n in gtp_params} == {'in_proj', 'out_proj'}, gtp_params
        assert any(p.pad_length > 0 for p in gtp_params.values()), "no padded shard to test"
        if use_fp8_params:
            assert all(is_float8tensor(p) for p in gtp_params.values())
        classify_gtp_remat_chains(layer)
        assert len({p.chain_id for p in gtp_params.values()}) == 1, "projections must share a chain"

    gtp_rank = pg.gtp_remat.rank()
    load_context = gtp_native_fp8_load_context(layer) if use_fp8_params else nullcontext()
    with torch.no_grad(), load_context:
        for name, p in layer.named_parameters():
            if initial is None:
                dist.broadcast(p, src=0)
            elif name in gtp_params:
                full = initial[name]
                full = torch.nn.functional.pad(full, (0, 0, 0, p.shape[0] * GTP - full.shape[0]))
                p.copy_(full.chunk(GTP)[gtp_rank])
                p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device='cuda')
            else:
                p.copy_(initial[name])
    weights = {n: p.detach().clone() for n, p in layer.named_parameters()}

    x = torch.randn(SEQ, 1, HIDDEN, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    dist.broadcast(x.data, src=0)
    with fp8_autocast(enabled=native_fp8, fp8_recipe=MXFP8BlockScaling()):
        out = layer(x)
    out.float().square().mean().backward()
    wait_for_gtp_grad_reduction_on_current_stream()

    grads = {}
    for name, p in layer.named_parameters():
        if name in gtp_params:
            shards = [torch.empty_like(p.main_grad) for _ in range(GTP)]
            dist.all_gather(shards, p.main_grad, group=pg.gtp_remat)
            grads[name] = torch.cat(shards)[: initial[name].shape[0]]
        else:
            grads[name] = p.grad.float()
    return x.grad.float(), grads, weights


@pytest.mark.parametrize(
    "native_fp8",
    [
        pytest.param(False, id="bf16"),
        pytest.param(True, marks=pytest.mark.launch_on_gb200, id="native_mxfp8"),
    ],
)
def test_mamba_gtp_grads_match_baseline(native_fp8):
    """GTP_remat=4 Mamba input and parameter grads match the unsharded layer.

    native_mxfp8 keeps the GTP in_proj/out_proj shards as MXFP8 tensors (--fp8-param-gather)
    against an unsharded layer quantizing the same BF16 weights on the fly.
    """
    if dist.get_world_size() != GTP:
        pytest.skip(f"Requires {GTP} ranks")
    if native_fp8:
        _requires_mxfp8()
    ref_dx, ref_grads, initial = _run(1, native_fp8)
    dx, grads, _ = _run(GTP, native_fp8, initial)
    # Max error relative to each tensor's own max|value|. Same kernels and inputs on both sides,
    # and MXFP8 blocks never straddle a shard boundary: both variants observed bit-exact, so the
    # tolerance is all margin; a 1/gtp gradient scale is off by 0.75.
    rel_tol = 1e-3
    grads["input"], ref_grads["input"] = dx, ref_dx
    assert grads.keys() == ref_grads.keys()
    errors = {}
    for name, ref in ref_grads.items():
        assert torch.isfinite(grads[name]).all(), f"{name}: non-finite grad"
        errors[name] = ((grads[name] - ref).abs().max() / ref.abs().max()).item()
    print(f"[mamba {native_fp8=}] rel max grad errors {errors}", flush=True)
    bad = {name: err for name, err in errors.items() if not err < rel_tol}
    assert not bad, f"grads differ from the unsharded layer beyond {rel_tol}: {bad}"
