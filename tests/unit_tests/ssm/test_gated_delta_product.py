# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for GatedDeltaProduct selective recompute.

`gdp_in_proj` recompute is meant to be an exact replay of the input projection and its
preprocessing: the forward pass is unaffected, and the backward pass rebuilds the same
subgraph from the same saved input. So enabling it must not move the numerics at all.
Any drift would mean the replay observed a different weight, a different RNG state, or
a stale storage after the output-discarding checkpoint shared the recomputed one back.

`conv1d.weight` used to be exempt, because the conv backward reduced it with atomicAdd. The
fixture below turns on the kernel's deterministic reduction instead, so it is held to the same
bitwise bar as everything else.
"""

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import causal_conv1d, gated_delta_product
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

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support

    fp8_available, reason_for_no_fp8 = check_fp8_support()
except ImportError:
    fp8_available, reason_for_no_fp8 = False, "transformer-engine is not installed"

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
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
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


def _run(mixer, hidden_states, grads):
    """Run one forward/backward and collect the output and every gradient."""
    x = hidden_states.clone().requires_grad_(True)
    with get_fp8_context(mixer.config):
        out, _ = mixer(x)
    out.backward(grads)
    param_grads = {name: param.grad.clone() for name, param in mixer.named_parameters()}
    return out.detach().clone(), x.grad.clone(), param_grads


@pytest.fixture(autouse=True)
def deterministic_conv1d(monkeypatch):
    """Hold the conv backward to its deterministic reduction for every test in this module.

    Without it, `causal_conv1d_fn` accumulates its weight gradient across blocks with atomicAdd
    on the channel-last layout the mixer feeds it, so two runs of the *same* mixer disagree on
    `conv1d.weight` and a parity test has to tolerate a drifting gradient. Set per-test rather
    than process-wide because the env var is read on each backward call.
    """
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")


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


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize("recompute_qkv", [False, True])
def test_in_proj_recompute_parity(recompute_qkv):
    """`gdp_in_proj` recompute must be an exact replay, alone and with `gdp_qkv`.

    The two are stacked deliberately: `gdp_qkv` saves VKQ, which `gdp_in_proj` discards,
    so this pins the ordering contract between the two recompute hooks -- the in_proj
    replay on the mixer output has to rematerialize VKQ before the QKV replay on
    core_attn_out consumes it.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        modules = ["gdp_in_proj"] + (["gdp_qkv"] if recompute_qkv else [])
        baseline = _build_mixer(_make_config([]))
        recomputed = _build_mixer(_make_config(modules), state_dict=baseline.state_dict())
        assert recomputed.recompute_in_proj and not baseline.recompute_in_proj
        assert recomputed.recompute_qkv == recompute_qkv

        sequence_length, micro_batch_size = 256, 2
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)

        _assert_replay_matches(
            _run(baseline, hidden_states, grads), _run(recomputed, hidden_states, grads)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_recipe", ["tensorwise"])
def test_fp8_in_proj_recompute_parity(fp8_recipe):
    """The replay must stay exact under FP8, where in_proj is the quantized op.

    CheckpointWithoutOutput records the forward recipe and amax state and replays under
    the recorded fp8_autocast, so the replayed GEMM has to see the same scales as the
    original. Drift here would mean the recompute re-derived them from an amax history
    that had already moved on.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        baseline = _build_mixer(_make_config([], fp8_recipe=fp8_recipe))
        recomputed = _build_mixer(
            _make_config(["gdp_in_proj"], fp8_recipe=fp8_recipe), state_dict=baseline.state_dict()
        )
        assert recomputed.recompute_in_proj and not baseline.recompute_in_proj

        # FP8 GEMMs need the token dimension aligned; keep it a multiple of 128.
        sequence_length, micro_batch_size = 256, 2
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)

        _assert_replay_matches(
            _run(baseline, hidden_states, grads), _run(recomputed, hidden_states, grads)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize("recompute_qkv", [False, True])
@pytest.mark.parametrize("fp8_recipe", [None, "mxfp8"])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_in_proj_recompute_preserves_norm_gate_view(
    monkeypatch, recompute_qkv, fp8_recipe, cp_size
):
    """At MB=1, CP1 and chunkwise CP must discard and restore the norm's saved gate."""
    if Utils.world_size < cp_size:
        pytest.skip(f"requires at least {cp_size} distributed ranks")
    if cp_size > 1:
        if not is_causal_conv1d_min_version("1.7.0"):
            pytest.skip("chunkwise CP requires causal-conv1d >= 1.7.0")
        if not (HAVE_FLA_GDP_CP if HAVE_FLA else HAVE_CUTEDSL_GDP_CP):
            pytest.skip("the selected GDP backend does not support chunkwise CP")
    if fp8_recipe == "mxfp8":
        te = pytest.importorskip("transformer_engine.pytorch")
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(123)
    try:
        modules = ["gdp_in_proj"] + (["gdp_qkv"] if recompute_qkv else [])
        config = _make_config(modules, fp8_recipe=fp8_recipe)
        config.context_parallel_size = cp_size
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
        baseline_result = _run(baseline, hidden_states, grads)

        x = hidden_states.clone().requires_grad_(True)
        with get_fp8_context(recomputed.config):
            output, _ = recomputed(x)
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


def _gdp_config(num_householder, recompute_modules=(), cp_mode=None):
    """A config with the given Householder count; ``cp_mode`` selects CP=2 of that kind.

    Four heads give an in_proj width of 516 * (num_householder + 1), halved by head-wise CP's
    all-to-all: 2064 and 1032 with three reflections, multiples of 8, and 1548 and 774 with
    two, which are not.
    """
    config = _make_config(recompute_modules)
    config.gdp_num_householder = num_householder
    config.gdp_cutedsl_kernel = not HAVE_FLA
    if cp_mode is not None:
        config.context_parallel_size = 2
        config.linear_cp_mode = cp_mode
        if cp_mode == "chunkwise":
            config.linear_cp_layout = "contiguous"
    return config


def _build_dense_conv_reference(monkeypatch, config, state_dict):
    """Build a mixer with the given weights whose conv always reads a dense copy of VKQ."""
    reference = _build_mixer(config, state_dict=state_dict)
    prepare_qkv = reference._prepare_qkv
    monkeypatch.setattr(
        reference,
        "_prepare_qkv",
        lambda x, **kwargs: prepare_qkv(x.clone(memory_format=torch.contiguous_format), **kwargs),
    )
    return reference


def _track_conv_input(monkeypatch, mixer):
    """Record where the projection, VKQ and the conv kernel's input live, per call.

    ``_in_proj_preprocess`` calls record the data pointers of z (a view into the projection)
    and of VKQ, and VKQ's storage; conv calls record their input's data pointer. This patches
    the module-level conv, so build it after running any reference mixer.
    """
    calls = {"projection": [], "vkq": [], "vkq_storage": [], "conv": []}
    in_proj_preprocess = mixer._in_proj_preprocess

    def tracked_in_proj_preprocess(*args, **kwargs):
        z, vkq, ba = in_proj_preprocess(*args, **kwargs)
        calls["projection"].append(z.untyped_storage().data_ptr())
        calls["vkq"].append(vkq.untyped_storage().data_ptr())
        calls["vkq_storage"].append(vkq.untyped_storage())
        return z, vkq, ba

    def tracked_causal_conv1d_fn(x, **kwargs):
        calls["conv"].append(x.untyped_storage().data_ptr())
        return causal_conv1d_fn(x=x, **kwargs)

    monkeypatch.setattr(mixer, "_in_proj_preprocess", tracked_in_proj_preprocess)
    for module in (gated_delta_product, causal_conv1d):
        monkeypatch.setattr(module, "causal_conv1d_fn", tracked_causal_conv1d_fn)
    return calls


def _assert_conv_reads_vkq(calls, copied):
    """In the forward pass, the conv reads VKQ as ``_in_proj_preprocess`` returned it: a view
    into the projection, or a copy of it if ``copied``."""
    assert calls["conv"][0] == calls["vkq"][0]
    assert (calls["vkq"][0] != calls["projection"][0]) == copied


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize("num_householder", [2, 3])
@pytest.mark.parametrize(
    ("cp_mode", "micro_batch_size"),
    [(None, 1), (None, 2), ("chunkwise", 1), ("headwise", 1), ("headwise", 2)],
)
def test_conv_reads_vkq_at_any_projection_width(
    monkeypatch, num_householder, cp_mode, micro_batch_size
):
    """The causal conv must run whatever the in_proj width, and copy VKQ only when it must.

    VKQ is a slice of the fused in_proj output, so its channels-last conv view has the
    projection's row pitch as its token stride, and causal_conv1d requires a multiple of 8.
    At batch size 1 the conv reads the projection itself, so a misaligned VKQ has to be
    copied. A larger batch reads the projection's batch-first copy, whose rows are padded, so
    VKQ needs no second copy. Either way the mixer must match, bit for bit, a mixer whose conv
    always reads a dense copy.
    """
    cp_size = 1 if cp_mode is None else 2
    if Utils.world_size < cp_size:
        pytest.skip(f"requires at least {cp_size} distributed ranks")
    if cp_mode == "chunkwise":
        if not is_causal_conv1d_min_version("1.7.0"):
            pytest.skip("chunkwise CP requires causal-conv1d >= 1.7.0")
        if not (HAVE_FLA_GDP_CP if HAVE_FLA else HAVE_CUTEDSL_GDP_CP):
            pytest.skip("the selected GDP backend does not support chunkwise CP")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(123)
    try:
        config = _gdp_config(num_householder, cp_mode=cp_mode)
        mixer = _build_mixer(config)
        reference = _build_dense_conv_reference(monkeypatch, config, mixer.state_dict())

        hidden_states = torch.randn(
            (256, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)
        expected = _run(reference, hidden_states, grads)
        calls = _track_conv_input(monkeypatch, mixer)
        actual = _run(mixer, hidden_states, grads)

        _assert_conv_reads_vkq(calls, copied=num_householder == 2 and micro_batch_size == 1)
        _assert_replay_matches(expected, actual)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.parametrize(
    "recompute_modules", [["gdp_qkv"], ["gdp_in_proj"], ["gdp_in_proj", "gdp_qkv"]]
)
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_recompute_at_misaligned_projection_width(monkeypatch, recompute_modules, micro_batch_size):
    """Selective recompute stays exact when VKQ is copied for the conv, or read in place
    from a padded copy, and `gdp_in_proj` still frees VKQ until the backward pass.

    The conv reads VKQ exactly as `_in_proj_preprocess` returns it, so `gdp_in_proj` discards
    any copy along with the projection, as it does the projection on aligned widths.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        mixer = _build_mixer(_gdp_config(2, recompute_modules))
        reference = _build_dense_conv_reference(monkeypatch, _gdp_config(2), mixer.state_dict())
        assert mixer.recompute_in_proj == ("gdp_in_proj" in recompute_modules)
        assert mixer.recompute_qkv == ("gdp_qkv" in recompute_modules)

        hidden_states = torch.randn(
            (256, micro_batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        grads = torch.randn_like(hidden_states)
        expected = _run(reference, hidden_states, grads)
        calls = _track_conv_input(monkeypatch, mixer)

        x = hidden_states.clone().requires_grad_(True)
        output, _ = mixer(x)
        _assert_conv_reads_vkq(calls, copied=micro_batch_size == 1)
        if mixer.recompute_in_proj:
            assert calls["vkq_storage"][0].nbytes() == 0
        output.backward(grads)
        actual = (
            output.detach().clone(),
            x.grad.clone(),
            {name: param.grad.clone() for name, param in mixer.named_parameters()},
        )
        _assert_replay_matches(expected, actual)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires mamba-ssm, einops, causal-conv1d and FLA")
@pytest.mark.skipif(not HAVE_FLA, reason="static inference does not run the CuTeDSL GDP kernel")
@pytest.mark.parametrize("num_householder", [2, 3])
@pytest.mark.parametrize(("batch_size", "prompt_length"), [(1, 1), (2, 8)])
def test_static_prefill_at_any_projection_width(
    monkeypatch, num_householder, batch_size, prompt_length
):
    """Static-batching prefill runs the same conv input and seeds the conv state from it.

    A one-token prompt at batch size 1 gives VKQ the one layout that ``contiguous()`` would
    return unchanged, size 1 in both batch and token dimensions. The output and both caches
    must match, bit for bit, a mixer whose conv always reads a dense copy.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        config = _gdp_config(num_householder)
        mixer = _build_mixer(config)
        reference = _build_dense_conv_reference(monkeypatch, config, mixer.state_dict())
        hidden_states = torch.randn(
            (prompt_length, batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )

        def prefill(gdp_mixer):
            context = StaticInferenceContext(max_batch_size=batch_size, max_sequence_length=16)
            # HybridStack sets this from sequence_len_offset before it runs its layers.
            context.seqlen_offset = 0
            with torch.no_grad():
                output, _ = gdp_mixer(hidden_states, inference_context=context)
            conv_state, ssm_state = context.key_value_memory_dict[gdp_mixer.layer_number]
            return output, conv_state, ssm_state

        expected = prefill(reference)
        calls = _track_conv_input(monkeypatch, mixer)
        actual = prefill(mixer)

        _assert_conv_reads_vkq(calls, copied=num_householder == 2 and batch_size == 1)
        for name, actual_tensor, expected_tensor in zip(
            ("output", "conv state", "SSM state"), actual, expected
        ):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, msg=f"{name} drifted"
            )
    finally:
        Utils.destroy_model_parallel()


def test_in_proj_recompute_rejects_qkv_offload():
    """gdp_in_proj leaves nothing for the gdp_qkv offload group, so the pair is rejected."""
    with pytest.raises(ValueError, match="gdp_qkv cannot be set in offload_modules"):
        TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=MAMBA_NUM_HEADS,
            recompute_granularity="selective",
            recompute_modules=["gdp_in_proj"],
            fine_grained_activation_offloading=True,
            offload_modules=["gdp_qkv"],
        )
