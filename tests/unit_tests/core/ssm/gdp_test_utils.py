# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared GDP selective-recompute construction and gradient references."""

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.context_parallel.chunkwise import build_packed_sequence_cp_metadata
from megatron.core.ssm.gated_delta_product import (
    HAVE_CUTEDSL_GDP,
    HAVE_CUTEDSL_GDP_CP,
    HAVE_EINOPS,
    HAVE_FLA,
    HAVE_FLA_GDP_CP,
    HAVE_MAMBA_SSM,
    GatedDeltaProductMixer,
    causal_conv1d_fn,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_causal_conv1d_min_version
from tests.unit_tests.test_utilities import Utils

HAVE_GDP_DEPS = (
    HAVE_MAMBA_SSM
    and HAVE_EINOPS
    and (HAVE_FLA or HAVE_CUTEDSL_GDP)
    and causal_conv1d_fn is not None
)

# The mixer places several divisibility constraints on these: d_inner = num_heads *
# head_dim must divide evenly by the group count, and the in_proj output width
# (d_inner * 4 + num_groups * state_dim * 4 + num_heads * 4 = 2064) must stay a
# multiple of 16 so the FP8 GEMM is legal.
HIDDEN_SIZE = 256
MAMBA_NUM_HEADS = 4
MAMBA_HEAD_DIM = 64
MAMBA_NUM_GROUPS = 2
MAMBA_STATE_DIM = 128


def _make_config(recompute_modules, fp8_recipe=None):
    """Build a GDP config; an empty recompute_modules disables selective recompute."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=MAMBA_NUM_HEADS,
        mamba_num_heads=MAMBA_NUM_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        mamba_num_groups=MAMBA_NUM_GROUPS,
        mamba_state_dim=MAMBA_STATE_DIM,
        params_dtype=torch.bfloat16,
        bf16=True,
        recompute_granularity="selective" if recompute_modules else None,
        recompute_modules=list(recompute_modules),
        fp8="hybrid" if fp8_recipe else None,
        fp8_recipe=fp8_recipe if fp8_recipe else "delayed",
    )


def _build_mixer(config, state_dict=None):
    """Build a GatedDeltaProductMixer, optionally loading a shared state dict."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'dp_cp']
    )
    mixer = GatedDeltaProductMixer(
        config,
        gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=pg_collection,
    ).cuda()
    if state_dict is not None:
        mixer.load_state_dict(state_dict)
    return mixer


def _run(mixer, hidden_states, grads, **forward_kwargs):
    """Run one forward/backward and collect the output and every gradient."""
    x = hidden_states.clone().requires_grad_(True)
    with get_fp8_context(mixer.config):
        out, _ = mixer(x, **forward_kwargs)
    out.backward(grads)
    param_grads = {name: param.grad.clone() for name, param in mixer.named_parameters()}
    return out.detach().clone(), x.grad.clone(), param_grads


def _assert_replay_matches(baseline, recomputed):
    """Compare a (output, input grad, param grads) triple; every tensor bitwise."""
    base_out, base_input_grad, base_param_grads = baseline
    recomp_out, recomp_input_grad, recomp_param_grads = recomputed

    torch.testing.assert_close(recomp_out, base_out, rtol=0, atol=0, msg="output drifted")
    torch.testing.assert_close(
        recomp_input_grad, base_input_grad, rtol=0, atol=0, msg="input gradient drifted"
    )
    assert set(recomp_param_grads) == set(base_param_grads)
    for name, base_grad in base_param_grads.items():
        torch.testing.assert_close(
            recomp_param_grads[name], base_grad, rtol=0, atol=0, msg=f"{name} gradient drifted"
        )


def assert_in_proj_recompute_preserves_norm_gate_view(
    monkeypatch, recompute_qkv, fp8_recipe, cp_size, runtime_cp_size=None
):
    """At MB=1, CP1 and chunkwise CP must discard and restore the norm's saved gate."""
    effective_cp_size = runtime_cp_size or cp_size
    if Utils.world_size < effective_cp_size:
        pytest.skip(f"requires at least {effective_cp_size} distributed ranks")
    if effective_cp_size > 1:
        if not is_causal_conv1d_min_version("1.7.0"):
            pytest.skip("chunkwise CP requires causal-conv1d >= 1.7.0")
        if not (HAVE_FLA_GDP_CP if HAVE_FLA else HAVE_CUTEDSL_GDP_CP):
            pytest.skip("the selected GDP backend does not support chunkwise CP")
    if fp8_recipe == "mxfp8":
        te = pytest.importorskip("transformer_engine.pytorch")
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        dynamic_context_parallel=runtime_cp_size is not None,
    )
    model_parallel_cuda_manual_seed(123)
    try:
        modules = ["gdp_in_proj"] + (["gdp_qkv"] if recompute_qkv else [])
        config = _make_config(modules, fp8_recipe=fp8_recipe)
        config.context_parallel_size = cp_size
        config.dynamic_context_parallel = runtime_cp_size is not None
        config.calculate_per_token_loss = runtime_cp_size is not None
        config.linear_cp_mode = "chunkwise"
        config.linear_cp_layout = "contiguous"
        if fp8_recipe:
            config.fp8 = "e4m3"
        config.gdp_cutedsl_kernel = not HAVE_FLA
        # Eight heads align the projection's ba width to MXFP8's 32-element blocks.
        config.mamba_num_heads = 8
        baseline = _build_mixer(config)
        recomputed = _build_mixer(config, state_dict=baseline.state_dict())

        baseline_norm_forward = baseline.norm.forward

        def contiguous_norm_forward(x, z):
            return baseline_norm_forward(x, z.contiguous())

        monkeypatch.setattr(baseline.norm, "forward", contiguous_norm_forward)

        projection_gate_storages = []
        in_proj_preprocess = recomputed._in_proj_preprocess

        def tracked_in_proj_preprocess(*args, **kwargs):
            outputs = in_proj_preprocess(*args, **kwargs)
            projection_gate_storages.append(outputs[0].untyped_storage())
            return outputs

        monkeypatch.setattr(recomputed, "_in_proj_preprocess", tracked_in_proj_preprocess)
        saved_gates = []
        expected_gates = []

        def capture_saved_gate(module, inputs, output):
            z = inputs[1]
            assert not z.is_contiguous()
            assert z.stride(-1) == 1
            assert z.untyped_storage().data_ptr() == projection_gate_storages[0].data_ptr()
            # Inspect the actual Mamba autograd save, including its internal 2D reshape.
            saved_gate = output.grad_fn.saved_tensors[-1]
            assert saved_gate.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
            saved_gates.append(saved_gate)
            expected_gates.append(saved_gate.detach().clone())

            def check_restored_gate(grad_outputs):
                assert len(projection_gate_storages) == 2
                assert (
                    saved_gate.untyped_storage().data_ptr()
                    == projection_gate_storages[1].data_ptr()
                )
                torch.testing.assert_close(saved_gate, expected_gates[0], rtol=0, atol=0)

            output.grad_fn.register_prehook(check_restored_gate)

        norm_hook = recomputed.norm.register_forward_hook(capture_saved_gate)
        hidden_states = torch.randn((256, 1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        grads = torch.randn_like(hidden_states)
        forward_kwargs = {}
        if runtime_cp_size is not None:
            groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["dynamic_cp"])
            cp_group = groups.dynamic_cp[runtime_cp_size]
            cu_seqlens = torch.tensor([0, 256 * runtime_cp_size], dtype=torch.int32, device="cuda")
            packed = PackedSeqParams(
                qkv_format="thd",
                cp_group=cp_group,
                local_cp_size=runtime_cp_size,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                total_tokens=256 * runtime_cp_size,
            )
            forward_kwargs = dict(
                packed_seq_params=packed,
                packed_sequence_cp_metadata=build_packed_sequence_cp_metadata(
                    packed.seq_idx, cp_group.rank(), runtime_cp_size
                ),
            )
        baseline_result = _run(baseline, hidden_states, grads, **forward_kwargs)

        x = hidden_states.clone().requires_grad_(True)
        with get_fp8_context(recomputed.config):
            output, _ = recomputed(x, **forward_kwargs)
        assert len(saved_gates) == len(projection_gate_storages) == 1
        assert saved_gates[0].untyped_storage().nbytes() == 0
        assert projection_gate_storages[0].nbytes() == 0

        output.backward(grads)
        assert len(projection_gate_storages) == 2
        assert saved_gates[0].untyped_storage().data_ptr() == projection_gate_storages[1].data_ptr()
        torch.testing.assert_close(saved_gates[0], expected_gates[0], rtol=0, atol=0)
        recomputed_result = (
            output.detach().clone(),
            x.grad.clone(),
            {name: param.grad.clone() for name, param in recomputed.named_parameters()},
        )
        _assert_replay_matches(baseline_result, recomputed_result)
        norm_hook.remove()
    finally:
        Utils.destroy_model_parallel()
