# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU initialization must produce the same global model at every GTP_remat degree.

CPU init exists so a run's initial weights are a function of the seed alone, not of the parallel
layout -- that is what makes convergence comparisons across configurations meaningful. The TE
modules allocate their weight *pre-sharded* under GTP_remat (see ``_gtp_pre_init``), so the CPU path
has to pad-and-slice the master weight the same way (see ``_initialize_affine_weight_cpu``);
otherwise it either shape-mismatches or hands every GTP peer the same rows.

Everything below is exercised on ONE world_size=4, TP=1 stack holding every layer kind at once --
GDP mixer, attention, MLP:

* **GTP_remat 1 / 2 / 4**, each compared **bitwise** against the GTP_remat=1 reference.
* **Both TE parallel modes** -- column (``in_proj``, ``linear_qkv``, ``fc1``) and row
  (``out_proj``, ``linear_proj``, ``fc2``), which are the two slicing geometries.
* **Alignment padding.** The GDP ``in_proj`` is 3104 wide, not a multiple of
  ``32 * gtp_remat_size`` at any degree here, so the padded branch of ``gtp_remat_slice_rows``
  always runs -- asserted rather than assumed.
* **BF16 and MXFP8 primary weights** (``fp8_param``).
* **The quantized-init snapshot.** TE caches ``_high_precision_init_val`` at construction and the
  distributed optimizer seeds its fp32 master params from it, but CPU init writes the real values
  only afterwards -- a stale snapshot means training starts from uninitialized memory. Checked at
  GTP_remat=1 as well, because that bug needs no GTP.
* **Replicated params stay replicated** -- layernorms, and the GDP conv1d / A_log / dt_bias, which
  are GPU-initialized regardless and must come through the CPU path untouched.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.gtp_api import dequantize_gtp_native_fp8, is_gtp_param
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _requires_multi_gpu,
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

WORLD = 4
# GTP_remat degrees exercised; WORLD must be divisible by each.
GTP_SIZES = [2, 4]
HIDDEN = 256
NUM_HEADS = 8
# 320 is deliberately >= 64-aligned but NOT 128-aligned, so linear_fc1 is UNPADDED at
# GTP_remat=2 and PADDED at GTP_remat=4. That asymmetry is what the real model hits (its GDP
# in_proj is 61760: 61760 % 64 == 0 but 61760 % 128 == 64), and a uniformly-aligned size would
# never exercise it.
FFN_HIDDEN = 320
# One stack holding every layer kind at once: GDP mixer, attention, MLP.
LAYER_PATTERN = ["M", "*", "-"]
NUM_LAYERS = len(LAYER_PATTERN)
SEED = 1234
dtype = torch.bfloat16


def _make_config(fp8_param):
    from megatron.core.transformer.transformer_config import TransformerConfig

    fp8_kwargs = dict(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True) if fp8_param else {}
    return TransformerConfig(
        num_attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN,
        add_bias_linear=False,
        params_dtype=dtype,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        mamba_num_groups=4,
        mamba_num_heads=8,
        mamba_head_dim=32,
        **fp8_kwargs,
    )


def _build_block(config, pg_collection):
    """Build a CPU-initialized GDP + attention + MLP stack and move it to the GPU."""
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec

    # Layers are normally constructed inside this context (transformer_block.py); it is what
    # turns fp8_param into quantized primary weights.
    with get_fp8_context(config, is_init=True):
        block = HybridStack(
            config,
            gdp_stack_spec.submodules,
            layer_type_list=LAYER_PATTERN,
            pg_collection=pg_collection,
        )
    return block.cuda()


def _local_values(param):
    """This rank's parameter values as fp32 on cpu, dequantizing quantized primary weights."""
    if getattr(param, "_gtp_native_fp8", False):
        # GTP reclasses native-FP8 shards into a dynamic subclass that TE's dequantize rejects.
        data = dequantize_gtp_native_fp8(param)
    elif hasattr(param, "dequantize"):
        data = param.dequantize()
    else:
        data = param.data
    return data.detach().float().cpu()


def _full_weights(block, gtp_group):
    """name -> full (un-sharded, un-padded) fp32 values, with the padded tail dropped."""
    full = {}
    for name, param in block.named_parameters():
        values = _local_values(param)
        if is_gtp_param(param):
            local = values.cuda()
            shards = [torch.empty_like(local) for _ in range(gtp_group.size())]
            dist.all_gather(shards, local, group=gtp_group)
            values = torch.cat([s.cpu() for s in shards], dim=0)
            pad_length = getattr(param, "pad_length", 0)
            if pad_length:
                values = values[: values.shape[0] - pad_length]
        full[name] = values
    return full


def _init_mpu(gtp_remat_size):
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.gtp_api import configure_gtp_remat_from_recipe
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Training calls this before model construction. Padding is enabled unconditionally (not just
    # for the fp8 cases) so every build shards identically, and so the padded branch of
    # gtp_remat_slice_rows is exercised: the GDP in_proj is 3104 wide, which is not a multiple of
    # 32 * gtp_remat_size at any degree under test.
    configure_gtp_remat_from_recipe(fp8=True, fp8_recipe="mxfp8")
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_remat_size
    )
    # CPU initialization draws from the CPU generator, which every rank seeds identically; the
    # CUDA tracker is seeded too so a layer that falls back to GPU init is not silently skipped.
    torch.manual_seed(SEED)
    model_parallel_cuda_manual_seed(SEED, force_reset_rng=True)
    return ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'pp', 'gtp_remat', 'expt_gtp_remat']
    )


def _worker_cpu_init_is_gtp_invariant(rank, world_size, port, fp8_param, gtp_remat_size):
    from megatron.core import parallel_state as ps

    config = _make_config(fp8_param)

    # ---------------- Reference: GTP_remat off, every rank holds the full weight ------------
    ref_pgs = _init_mpu(gtp_remat_size=1)
    ref_block = _build_block(config, ref_pgs)
    assert not any(
        is_gtp_param(p) for p in ref_block.parameters()
    ), "reference block must not be GTP-sharded"
    reference = _full_weights(ref_block, ref_pgs.gtp_remat)
    del ref_block

    # ---------------- GTP_remat on, weights allocated pre-sharded --------------------------
    gtp_pgs = _init_mpu(gtp_remat_size=gtp_remat_size)
    assert gtp_pgs.gtp_remat.size() == gtp_remat_size, "GTP_remat inactive"
    gtp_block = _build_block(config, gtp_pgs)
    sharded = [p for p in gtp_block.parameters() if is_gtp_param(p)]
    assert sharded, "GTP is not active: the block has no GTP-sharded parameter"
    assert any(
        getattr(p, "pad_length", 0) > 0 for p in sharded
    ), "no GTP-sharded weight needed alignment padding; the padded slice path is not covered"
    gathered = _full_weights(gtp_block, gtp_pgs.gtp_remat)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()

    assert set(gathered) == set(reference), "parameter sets differ between the two layouts"
    for name, values in gathered.items():
        ref = reference[name]
        assert values.shape == ref.shape, f"{name}: shape {values.shape} != reference {ref.shape}"
        torch.testing.assert_close(
            values,
            ref,
            atol=0.0,
            rtol=0.0,
            msg=lambda m, name=name: (
                f"{name}: CPU init at GTP_remat={gtp_remat_size} differs from GTP_remat=1\n{m}"
            ),
        )


def _worker_high_precision_init_val(rank, world_size, port, gtp_remat_size):
    """The optimizer's master-weight snapshot must hold exactly the CPU-initialized values.

    Compared against a BF16 build rather than against the dequantized FP8 weight: both come from
    the same master weight rounded to BF16, so the match is exact. Dequantizing would only allow an
    assertion to within MXFP8 quantization error, which is loose enough to hide a partly-stale
    snapshot.
    """
    bf16_pgs = _init_mpu(gtp_remat_size=gtp_remat_size)
    bf16_block = _build_block(_make_config(fp8_param=False), bf16_pgs)
    bf16_weights = {n: p.data.detach().float().cpu() for n, p in bf16_block.named_parameters()}
    del bf16_block

    fp8_pgs = _init_mpu(gtp_remat_size=gtp_remat_size)
    block = _build_block(_make_config(fp8_param=True), fp8_pgs)

    checked = 0
    for name, param in block.named_parameters():
        getter = getattr(param, "get_high_precision_init_val", None)
        if getter is None or getter() is None:
            continue
        checked += 1
        torch.testing.assert_close(
            getter().float().cpu(),
            bf16_weights[name],
            atol=0.0,
            rtol=0.0,
            msg=lambda m, name=name: f"{name}: stale high-precision init snapshot\n{m}",
        )
    assert checked > 0, "no quantized primary weights found; fp8_param did not take effect"

    from megatron.core import parallel_state as ps

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()


class TestGTPCpuInitialization:
    @pytest.mark.parametrize("fp8_param", [False, True])
    @pytest.mark.parametrize("gtp_remat_size", GTP_SIZES)
    def test_cpu_init_is_gtp_invariant(self, gtp_remat_size, fp8_param):
        _requires_multi_gpu(WORLD)
        if fp8_param:
            _requires_mxfp8()
        _run_distributed(_worker_cpu_init_is_gtp_invariant, WORLD, fp8_param, gtp_remat_size)

    @pytest.mark.parametrize("gtp_remat_size", [1, *GTP_SIZES])
    def test_high_precision_init_val_refreshed(self, gtp_remat_size):
        """Also covered at GTP=1: the stale snapshot is a plain cpu-init + fp8_param bug, and
        nothing about it needs GTP."""
        _requires_multi_gpu(WORLD)
        _requires_mxfp8()
        _run_distributed(_worker_high_precision_init_val, WORLD, gtp_remat_size)
