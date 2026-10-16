# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_reuse_checkpoint_topk``: the replay of a checkpointed DSA layer reuses the
top-k indices and scores selected by its first call instead of rerunning the indexer. With a
frozen indexer (``dsa_indexer_loss_coeff=0``) the replay reuses the indices alone."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_kernels,
    dsa_masking,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.forward_sharing import (
    forward_sharing_lifetime,
    get_forward_sharing_state,
    mtp_repeated_sharing_lifetime,
    preserve_forward_sharing_for_checkpoint,
)
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    SPARSE_ATTENTION_INDEX,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    _make_config as make_small_dsa_config,
)

_ATTENTION_MASK_ARG = 5


def _skip_if_cudnn_dsa_unavailable(world_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the cuDNN DSA backend")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN fused DSA path requires SM90+")
    if Utils.world_size < world_size:
        pytest.skip(f"requires at least {world_size} distributed ranks")
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"cuDNN fused DSA dependencies are unavailable: {exc}")


def _make_config(*, tp: int, cp: int, deterministic: bool) -> MLATransformerConfig:
    config = MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=64,
        num_query_groups=64,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        kv_channels=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=64,
        dsa_indexer_topk_freq=2,
        dsa_indexer_skip_topk_offset=1,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        context_parallel_size=cp,
        cp_comm_type="allgather",
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        gradient_accumulation_fusion=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        use_cpu_initialization=False,
        perform_initialization=True,
    )
    # Set after validation: deterministic mode otherwise asserts launcher-level settings.
    config.deterministic_mode = deterministic
    return config


def _make_attention(config: MLATransformerConfig, pg_collection: ProcessGroupCollection):
    indexer_spec = ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()
    assert attention.index_share and not attention.skip_topk
    return attention.train()


def _make_inputs(config: MLATransformerConfig, *, tp_rank: int, cp_rank: int, cp: int):
    """THD inputs for one packed sequence; each CP rank holds two zigzag chunks."""
    local_rows = 128
    global_rows = local_rows * cp
    tp = config.tensor_model_parallel_size
    heads = config.num_attention_heads
    local_heads = heads // tp
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026 + cp_rank)

    def randn(*shape, scale=1.0):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * scale

    query = randn(local_rows, 1, heads, absorbed_dim)
    key = randn(local_rows, 1, 1, absorbed_dim)
    # Large indexer inputs keep the selected scores away from top-k ties, which would make
    # the two selections legitimately differ.
    x = randn(local_rows, 1, config.hidden_size, scale=8.0)
    qr = randn(local_rows, 1, config.q_lora_rank, scale=8.0)
    up_v_weight = randn(heads, config.v_head_dim, config.kv_lora_rank)
    head_slice = slice(tp_rank * local_heads, (tp_rank + 1) * local_heads)
    rows_per_tp = local_rows // tp
    row_slice = slice(tp_rank * rows_per_tp, (tp_rank + 1) * rows_per_tp)
    cu_seqlens = torch.tensor([0, global_rows], dtype=torch.int32, device="cuda")
    return {
        "query": query[:, :, head_slice].contiguous(),
        "key": key.contiguous(),
        "x": (x[row_slice] if tp > 1 else x).contiguous(),
        "qr": (qr[row_slice] if tp > 1 else qr).contiguous(),
        "up_v_weight": up_v_weight[head_slice].contiguous(),
        "packed_seq_params": PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens.clone(),
            cu_seqlens_kv_padded=cu_seqlens.clone(),
            max_seqlen_q=global_rows,
            max_seqlen_kv=global_rows,
        ),
    }


def _checkpointed_step(attention: DSAttention, inputs: dict, grad_output_seed: int):
    """One full-recompute forward/backward of the layer; returns output, loss and grads."""
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    attention.zero_grad(set_to_none=True)
    packed_seq_params = inputs["packed_seq_params"]
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("query", "key", "up_v_weight")
    }

    def layer(query, key, x, qr, up_v_weight, attention_mask):
        return attention(
            query=query,
            key=key,
            value=None,
            attention_mask=attention_mask,
            x=x,
            qr=qr,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed_seq_params,
            up_v_weight=up_v_weight,
        )

    with forward_sharing_lifetime(packed_seq_params, None, attention.config):
        wrapped = preserve_forward_sharing_for_checkpoint(
            layer, packed_seq_params, attention.config, attention_mask_arg=_ATTENTION_MASK_ARG
        )
        output = tensor_parallel.checkpoint(
            wrapped,
            False,
            leaves["query"],
            leaves["key"],
            inputs["x"],
            inputs["qr"],
            leaves["up_v_weight"],
            None,
        )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(grad_output_seed)
    grad_output = torch.randn(output.shape, device="cuda", dtype=output.dtype, generator=generator)
    output.backward(grad_output)
    loss = DSAIndexerLossLoggingHelper.tracker["values"][0].detach().clone()
    grads = {name: leaf.grad.detach().clone() for name, leaf in leaves.items()}
    for name, parameter in attention.indexer.named_parameters():
        assert parameter.grad is not None, f"indexer {name} has no gradient"
        grads[f"indexer.{name}"] = parameter.grad.detach().clone()
    return output.detach(), loss, grads


@pytest.fixture
def indexer_call_log(monkeypatch):
    """Record every dense indexer top-k call and every stored-top-k loss call."""
    log = {"dense": [], "from_topk": 0}
    dense = dsa_cudnn_kernels._indexer_topk_bshd
    from_topk = dsa_cudnn_kernels.run_sparse_indexer_loss_from_topk

    def counting_dense(*args, **kwargs):
        # Autograd Functions run their forward without grad, so record whether scores were
        # requested rather than the grad mode.
        log["dense"].append(bool(kwargs.get("return_topk_scores", False)))
        return dense(*args, **kwargs)

    def counting_from_topk(*args, **kwargs):
        log["from_topk"] += 1
        return from_topk(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", counting_dense)
    monkeypatch.setattr(dsa_cudnn_kernels, "run_sparse_indexer_loss_from_topk", counting_from_topk)
    return log


def _assert_similar(actual: torch.Tensor, expected: torch.Tensor, *, label: str, eps: float):
    assert torch.isfinite(actual).all(), f"{label} has non-finite values"
    actual, expected = actual.double().flatten(), expected.double().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    norm_ratio = (actual.norm() / expected.norm().clamp_min(1e-30)).item()
    assert cosine > 1 - eps, f"{label}: cosine={cosine:.8f}"
    assert abs(norm_ratio - 1) < eps, f"{label}: norm_ratio={norm_ratio:.8f}"


# The TP2+CP2 case shares the real-kernel cuDNN/flash_mla setup of the TP/SP parity tests,
# which are disabled in dev (flaky_in_dev) and LTS (flaky) CI because it fails with a CUDA
# error there. Re-enable together with them once the kernel/build root cause is resolved.
@pytest.mark.parametrize(
    ("tp", "cp"),
    [
        pytest.param(1, 1, id="tp1cp1"),
        pytest.param(2, 2, id="tp2cp2", marks=[pytest.mark.flaky, pytest.mark.flaky_in_dev]),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("stash", ["used", "declined"])
def test_checkpoint_replay_reuses_first_call_topk(
    tp, cp, deterministic, stash, indexer_call_log, monkeypatch
):
    _skip_if_cudnn_dsa_unavailable(tp * cp)
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = _make_config(tp=tp, cp=cp, deterministic=deterministic)
        attention = _make_attention(config, pg_collection)
        inputs = _make_inputs(
            config, tp_rank=pg_collection.tp.rank(), cp_rank=pg_collection.cp.rank(), cp=cp
        )

        config.dsa_indexer_reuse_checkpoint_topk = False
        baseline = _checkpointed_step(attention, inputs, grad_output_seed=7)
        assert indexer_call_log["dense"] == [False, True]
        assert indexer_call_log["from_topk"] == 0

        indexer_call_log["dense"].clear()
        warnings = []
        monkeypatch.setattr(
            dsa_module,
            "_warn_dsa_checkpoint_topk_reuse_fallback",
            lambda layer_number, reason: warnings.append((layer_number, reason)),
        )
        if stash == "declined":
            monkeypatch.setattr(
                dsa_cudnn_kernels, "run_fused_qk_topk_with_scores", lambda **_kwargs: None
            )
        config.dsa_indexer_reuse_checkpoint_topk = True
        reused = _checkpointed_step(attention, inputs, grad_output_seed=7)

        if stash == "used":
            # The replay computes the loss from the first call's stored top-k: one dense
            # indexer run (the first call, with scores) and no rerun during backward.
            assert indexer_call_log["dense"] == [True]
            assert indexer_call_log["from_topk"] == 1
            assert warnings == []
        else:
            assert indexer_call_log["dense"] == [False, True]
            assert indexer_call_log["from_topk"] == 0
            assert warnings == [(1, "its first call stored no top-k scores")]

        baseline_output, baseline_loss, baseline_grads = baseline
        output, loss, grads = reused
        # Forward attention consumes the same index-sorted selection either way.
        torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
        if deterministic:
            # Deterministic mode canonicalizes slots by index on both paths and avoids
            # atomics, so reusing the first call's top-k is bitwise identical.
            torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
            for name, grad in grads.items():
                torch.testing.assert_close(grad, baseline_grads[name], rtol=0, atol=0, msg=name)
        else:
            # Atomic accumulation in the sparse backward kernels differs run to run, and
            # reuse also changes the slot order inside the selected set (fp32 reduction
            # order), so values agree to rounding.
            torch.testing.assert_close(loss, baseline_loss, rtol=1e-4, atol=1e-7)
            for name, grad in grads.items():
                _assert_similar(grad, baseline_grads[name], label=name, eps=1e-4)
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


def _frozen_step(attention: DSAttention, inputs: dict, grad_output_seed: int, checkpoint: bool):
    """One forward/backward of a frozen-indexer layer, eager or with full recompute.

    Returns the output and the gradients of the attention inputs; the indexer has none.
    """
    attention.zero_grad(set_to_none=True)
    packed_seq_params = inputs["packed_seq_params"]
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("query", "key", "up_v_weight")
    }

    def layer(query, key, x, qr, up_v_weight, attention_mask):
        return attention(
            query=query,
            key=key,
            value=None,
            attention_mask=attention_mask,
            x=x,
            qr=qr,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed_seq_params,
            up_v_weight=up_v_weight,
        )

    args = (leaves["query"], leaves["key"], inputs["x"], inputs["qr"], leaves["up_v_weight"], None)
    with forward_sharing_lifetime(packed_seq_params, None, attention.config):
        if checkpoint:
            wrapped = preserve_forward_sharing_for_checkpoint(
                layer, packed_seq_params, attention.config, attention_mask_arg=_ATTENTION_MASK_ARG
            )
            output = tensor_parallel.checkpoint(wrapped, False, *args)
        else:
            output = layer(*args)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(grad_output_seed)
    grad_output = torch.randn(output.shape, device="cuda", dtype=output.dtype, generator=generator)
    output.backward(grad_output)
    for name, parameter in attention.indexer.named_parameters():
        assert not parameter.requires_grad, f"frozen indexer {name} requires grad"
        assert parameter.grad is None, f"frozen indexer {name} has a gradient"
    return output.detach(), {name: leaf.grad.detach().clone() for name, leaf in leaves.items()}


# Same TP2+CP2 flaky marks as test_checkpoint_replay_reuses_first_call_topk above.
@pytest.mark.parametrize(
    ("tp", "cp"),
    [
        pytest.param(1, 1, id="tp1cp1"),
        pytest.param(2, 2, id="tp2cp2", marks=[pytest.mark.flaky, pytest.mark.flaky_in_dev]),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_frozen_checkpoint_replay_reuses_first_call_indices(
    tp, cp, deterministic, indexer_call_log, monkeypatch
):
    """Frozen indexer: the replay reuses the first call's indices and skips the indexer."""
    _skip_if_cudnn_dsa_unavailable(tp * cp)
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = _make_config(tp=tp, cp=cp, deterministic=deterministic)
        # Freeze before building: DSAIndexer turns requires_grad off at construction.
        config.dsa_indexer_loss_coeff = 0.0
        attention = _make_attention(config, pg_collection)
        inputs = _make_inputs(
            config, tp_rank=pg_collection.tp.rank(), cp_rank=pg_collection.cp.rank(), cp=cp
        )
        indexer_runs = []
        forward_before_topk = attention.indexer.forward_before_topk

        def counting_forward_before_topk(*args, **kwargs):
            indexer_runs.append(torch.is_grad_enabled())
            return forward_before_topk(*args, **kwargs)

        attention.indexer.forward_before_topk = counting_forward_before_topk
        sparse_calls = []
        run_sparse_attention = dsa_module._run_sparse_attention

        def recording_sparse_attention(**kwargs):
            sparse_calls.append((kwargs["topk_indices"], torch.is_grad_enabled()))
            return run_sparse_attention(**kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", recording_sparse_attention)
        warnings = []
        monkeypatch.setattr(
            dsa_module,
            "_warn_dsa_checkpoint_topk_reuse_fallback",
            lambda layer_number, reason: warnings.append((layer_number, reason)),
        )

        runs = {}
        for name, checkpoint, reuse in (
            ("eager", False, True),
            ("replay-rerun", True, False),
            ("replay-reuse", True, True),
        ):
            config.dsa_indexer_reuse_checkpoint_topk = reuse
            indexer_runs.clear()
            sparse_calls.clear()
            indexer_call_log["dense"].clear()
            output, grads = _frozen_step(
                attention, inputs, grad_output_seed=7, checkpoint=checkpoint
            )
            runs[name] = SimpleNamespace(
                output=output,
                grads=grads,
                indexer_runs=list(indexer_runs),
                dense=list(indexer_call_log["dense"]),
                sparse_calls=list(sparse_calls),
            )

        # No score stash and no loss-from-top-k call in any frozen run.
        for name, run in runs.items():
            assert all(not with_scores for with_scores in run.dense), name
        assert indexer_call_log["from_topk"] == 0
        assert warnings == []
        assert runs["eager"].indexer_runs == [False] and len(runs["eager"].dense) == 1
        # Without reuse the replay reruns the indexer projections, scoring and top-k.
        assert runs["replay-rerun"].indexer_runs == [False, False]
        assert len(runs["replay-rerun"].dense) == 2
        # With reuse only the first call runs them, and the replay's sparse attention gets
        # exactly the indices the first call's output was computed with.
        reuse = runs["replay-reuse"]
        assert reuse.indexer_runs == [False] and len(reuse.dense) == 1
        assert [grad_enabled for _, grad_enabled in reuse.sparse_calls] == [False, True]
        assert reuse.sparse_calls[1][0] is reuse.sparse_calls[0][0]

        for name in ("replay-rerun", "replay-reuse"):
            # The checkpoint returns its first call's output, computed like the eager one.
            torch.testing.assert_close(runs[name].output, runs["eager"].output, rtol=0, atol=0)
            for grad_name, grad in runs[name].grads.items():
                expected = runs["eager"].grads[grad_name]
                if deterministic:
                    torch.testing.assert_close(grad, expected, rtol=0, atol=0, msg=grad_name)
                else:
                    # Atomic accumulation in the sparse-attention backward differs run to run.
                    _assert_similar(grad, expected, label=f"{name} {grad_name}", eps=1e-4)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"dsa_kernel_backend": "none"}, "dsa_kernel_backend='cudnn'", id="backend-none"
        ),
        pytest.param(
            {"attention_backend": AttnBackend.unfused}, "attention_backend != unfused", id="unfused"
        ),
        pytest.param(
            {"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss", id="dense-loss"
        ),
        pytest.param({"dsa_indexer_topk_freq": 1}, "dsa_indexer_topk_freq > 1", id="topk-freq-1"),
        pytest.param(
            {"overlap_moe_expert_parallel_comm": True, "recompute_granularity": "full"},
            "overlap_moe_expert_parallel_comm",
            id="overlap-moe-full-recompute",
        ),
    ],
)
def test_reuse_checkpoint_topk_config_validation(monkeypatch, overrides, message):
    from megatron.core import utils as core_utils
    from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
        _make_config as make_small_dsa_config,
    )

    # Config validation only; neither a CUDA device nor the kernel dependencies are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    fields = dict(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_reuse_checkpoint_topk=True,
    )
    fields.update(overrides)
    with pytest.raises(ValueError, match=message):
        make_small_dsa_config(**fields)


def _small_reuse_config(monkeypatch, **overrides):
    """Validate a small DSA config with checkpoint top-k reuse on, without CUDA or kernels."""
    from megatron.core import utils as core_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    fields = dict(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_reuse_checkpoint_topk=True,
    )
    fields.update(overrides)
    return make_small_dsa_config(**fields)


_FP8_DELAYED = {"fp8": "e4m3", "fp8_recipe": "delayed"}
_FP4_CUSTOM = {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": "pkg.factory"}
_NO_INDEXER_PROJ_FP8 = {
    "dsa_indexer_qk_proj_use_quantization": False,
    "dsa_indexer_weights_proj_use_quantization": False,
}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"dsa_indexer_loss_coeff": 0.0}, id="frozen"),
        pytest.param({"dsa_indexer_loss_coeff": None}, id="frozen-unset-coeff"),
        pytest.param({"dsa_indexer_loss_coeff": -1.0}, id="frozen-negative-coeff"),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.0, "dsa_indexer_use_sparse_loss": True},
            id="frozen-sparse-flag",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.0, "fp8": "e4m3", "fp8_recipe": "mxfp8"},
            id="frozen-fp8-mxfp8",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.0, **_FP8_DELAYED, **_NO_INDEXER_PROJ_FP8},
            id="frozen-fp8-delayed-bf16-indexer",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.0, "fp4": "e2m1", "fp4_recipe": "nvfp4"},
            id="frozen-fp4-nvfp4",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.0, **_FP4_CUSTOM, **_NO_INDEXER_PROJ_FP8},
            id="frozen-fp4-custom-bf16-indexer",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.01, "dsa_indexer_use_sparse_loss": True},
            id="unfrozen-sparse",
        ),
        pytest.param(
            {"dsa_indexer_loss_coeff": 0.01, "dsa_indexer_use_sparse_loss": True, **_FP8_DELAYED},
            id="unfrozen-fp8-delayed",
        ),
    ],
)
def test_reuse_checkpoint_topk_config_accepts(monkeypatch, overrides):
    config = _small_reuse_config(monkeypatch, **overrides)
    assert config.dsa_indexer_reuse_checkpoint_topk


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"dsa_kernel_backend": "none"}, "dsa_kernel_backend='cudnn'", id="backend-none"
        ),
        pytest.param(
            {"attention_backend": AttnBackend.unfused}, "attention_backend != unfused", id="unfused"
        ),
        pytest.param({"dsa_indexer_topk_freq": 1}, "dsa_indexer_topk_freq > 1", id="topk-freq-1"),
        pytest.param(
            {"overlap_moe_expert_parallel_comm": True, "recompute_granularity": "full"},
            "overlap_moe_expert_parallel_comm",
            id="overlap-moe-full-recompute",
        ),
        pytest.param({"dsa_indexer_kpool": 2}, "does not support KPool selection", id="kpool-2"),
        pytest.param(_FP8_DELAYED, "fp8_recipe='delayed'", id="fp8-delayed"),
        pytest.param(
            {"fp8": "e4m3", "fp8_recipe": "custom"}, "fp8_recipe='custom'", id="fp8-custom"
        ),
        pytest.param(
            {**_FP8_DELAYED, "dsa_indexer_weights_proj_use_quantization": False},
            "delayed-scaling recompute stash",
            id="fp8-delayed-qk-proj-only",
        ),
        pytest.param(_FP4_CUSTOM, "fp4_recipe='custom'", id="fp4-custom"),
        pytest.param(
            {**_FP4_CUSTOM, "dsa_indexer_qk_proj_use_quantization": False},
            "delayed-scaling recompute stash",
            id="fp4-custom-weights-proj-only",
        ),
    ],
)
def test_reuse_checkpoint_topk_frozen_config_validation(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _small_reuse_config(monkeypatch, dsa_indexer_loss_coeff=0.0, **overrides)


# ---------------------------------------------------------------------------------------
# Frozen indexer on CPU: the indexer projections are stubbed, the fused dispatch is enabled
# and falls back to the PyTorch top-k (no kernel backend is installed), and checkpointing,
# forward sharing and sparse attention are real.
# ---------------------------------------------------------------------------------------

_CPU_SEQ_LEN = 4
# Small-config schedule (dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=3): layer 3
# selects its own top-k and layer 4 reuses it.
_PRODUCER_LAYER = 3
_SKIP_LAYER = 4


@pytest.fixture
def frozen_cpu(monkeypatch):
    """Build CPU DSA layers and record the indexer, top-k and sparse-attention calls."""
    calls = []
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)

    def record(owner, name):
        original = getattr(owner, name)

        def recorded(*args, **kwargs):
            calls.append((name, torch.is_grad_enabled()))
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, recorded)

    record(dsa_masking, "build_fused_indexer_varlen_bounds")
    record(dsa_kernels, "run_fused_qk_topk")
    record(dsa_kernels, "run_fused_qk_topk_with_scores")
    record(dsa_kernels, "run_fused_qk_topk_with_loss")
    record(dsa_kernels, "run_sparse_indexer_loss_from_topk")
    record(dsa_module, "fused_qk_topk_naive")
    sparse_calls = []
    run_sparse_attention = dsa_module._run_sparse_attention

    def recording_sparse_attention(**kwargs):
        sparse_calls.append(
            SimpleNamespace(topk=kwargs["topk_indices"], grad=torch.is_grad_enabled())
        )
        return run_sparse_attention(**kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", recording_sparse_attention)
    warnings = []
    monkeypatch.setattr(
        dsa_module,
        "_warn_dsa_checkpoint_topk_reuse_fallback",
        lambda layer_number, reason: warnings.append((layer_number, reason)),
    )
    return SimpleNamespace(calls=calls, sparse=sparse_calls, warnings=warnings)


def _make_cpu_reuse_config(**overrides):
    """Small DSA config (frozen indexer unless overridden) with checkpoint top-k reuse on."""
    config = make_small_dsa_config(**overrides)
    # Set after validation, which needs CUDA for the cuDNN backend (see the tests above);
    # frozen_cpu enables the fused dispatch instead.
    config.dsa_indexer_reuse_checkpoint_topk = True
    return config


def _make_frozen_cpu_layer(config, layer_number, frozen_cpu, *, is_mtp_layer=False):
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=None),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
        is_mtp_layer=is_mtp_layer,
    )
    if attention.indexer is not None:

        def forward_before_topk(x, _qr, _packed_seq_params):
            frozen_cpu.calls.append(("indexer", torch.is_grad_enabled()))
            generator = torch.Generator().manual_seed(7)
            rows, batch = x.shape[:2]
            q = torch.randn(rows, batch, 2, 8, generator=generator)
            k = torch.randn(rows, batch, 8, generator=generator)
            weights = torch.rand(rows, batch, 2, generator=generator) + 0.5
            return q, k, weights

        attention.indexer.forward_before_topk = forward_before_topk
    return attention.train()


def _frozen_cpu_inputs(config, seed=11):
    generator = torch.Generator().manual_seed(seed)
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    return {
        "query": torch.randn(_CPU_SEQ_LEN, 1, heads, absorbed_dim, generator=generator),
        "key": torch.randn(_CPU_SEQ_LEN, 1, 1, absorbed_dim, generator=generator),
        "x": torch.randn(_CPU_SEQ_LEN, 1, config.hidden_size, generator=generator),
        "qr": torch.randn(_CPU_SEQ_LEN, 1, config.q_lora_rank, generator=generator),
        "up_v_weight": torch.randn(
            heads, config.v_head_dim, config.kv_lora_rank, generator=generator
        ),
    }


def _causal_mask():
    return torch.ones((1, 1, _CPU_SEQ_LEN, _CPU_SEQ_LEN), dtype=torch.bool).triu(1)


def _frozen_cpu_step(layers, inputs, *, mode, tamper_first_call=None):
    """Forward and backward of decoder DSA layers sharing one forward.

    ``mode`` is ``'eager'``, ``'per-layer'`` (one checkpoint per layer) or ``'one-chunk'``
    (all layers in one checkpoint). ``tamper_first_call(payload)`` edits the index-sharing
    payload at the end of each checkpoint's first call, before its snapshot.
    """
    config = layers[0].config
    mask = _causal_mask()
    leaves = {
        name: inputs[name].clone().requires_grad_(True) for name in ("query", "key", "up_v_weight")
    }

    def run(layer_list):
        def forward(query, key, x, qr, up_v_weight, attention_mask):
            output = 0
            for attention in layer_list:
                output = output + attention(
                    query=query,
                    key=key,
                    value=None,
                    attention_mask=attention_mask,
                    x=x,
                    qr=qr,
                    up_v_weight=up_v_weight,
                    attn_mask_type=AttnMaskType.causal,
                )
            if tamper_first_call is not None and not torch.is_grad_enabled():
                state = get_forward_sharing_state(None, attention_mask, config)
                tamper_first_call(state.get(dsa_module._DSAIndexSharingPayload))
            return output

        return forward

    args = (leaves["query"], leaves["key"], inputs["x"], inputs["qr"], leaves["up_v_weight"], mask)

    def checkpointed(layer_list):
        wrapped = preserve_forward_sharing_for_checkpoint(
            run(layer_list), None, config, attention_mask_arg=_ATTENTION_MASK_ARG
        )
        return tensor_parallel.checkpoint(wrapped, False, *args)

    with forward_sharing_lifetime(None, mask, config) as state:
        if mode == "eager":
            output = run(layers)(*args)
        elif mode == "per-layer":
            output = sum(checkpointed([attention]) for attention in layers)
        else:
            assert mode == "one-chunk", mode
            output = checkpointed(layers)
        payload = state.get(dsa_module._DSAIndexSharingPayload)
        published = SimpleNamespace(
            topk=dict(payload.topk_by_layer),
            scores=dict(payload.topk_scores_by_layer),
            key_len=dict(payload.topk_key_len_by_layer),
            frozen_key_len=dict(payload.frozen_topk_key_len_by_layer),
        )
    grad_output = torch.randn(output.shape, generator=torch.Generator().manual_seed(5))
    output.backward(grad_output)
    grads = {name: leaf.grad.clone() for name, leaf in leaves.items()}
    return SimpleNamespace(output=output.detach(), grads=grads, published=published)


def _assert_equal_steps(actual, expected, *, bitwise=True):
    tolerance = dict(rtol=0, atol=0) if bitwise else {}
    torch.testing.assert_close(actual.output, expected.output, **tolerance)
    for name, grad in actual.grads.items():
        torch.testing.assert_close(grad, expected.grads[name], msg=name, **tolerance)


def _same_tensor(actual, expected):
    """The same tensor, or a checkpoint's detached view of it (shared storage and layout)."""
    return actual is expected or (
        actual.data_ptr() == expected.data_ptr()
        and actual.shape == expected.shape
        and actual.stride() == expected.stride()
    )


def _count(calls, name, grad=None):
    return sum(1 for call, enabled in calls if call == name and (grad is None or enabled == grad))


_TOPK_SELECTION_CALLS = (
    "indexer",
    "build_fused_indexer_varlen_bounds",
    "run_fused_qk_topk",
    "fused_qk_topk_naive",
)


@pytest.mark.parametrize("mode", ["per-layer", "one-chunk"])
def test_frozen_replay_reuses_first_call_indices_without_indexer(frozen_cpu, mode):
    """The producer's replay reuses its first call's indices; its skip layer gets the same."""
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)
    skip = _make_frozen_cpu_layer(config, _SKIP_LAYER, frozen_cpu)
    assert not producer.skip_topk and skip.skip_topk and skip.source_layer == _PRODUCER_LAYER
    inputs = _frozen_cpu_inputs(config)

    eager = _frozen_cpu_step([producer, skip], inputs, mode="eager")
    frozen_cpu.calls.clear()
    frozen_cpu.sparse.clear()
    reused = _frozen_cpu_step([producer, skip], inputs, mode=mode)
    reuse_calls, reuse_sparse = list(frozen_cpu.calls), list(frozen_cpu.sparse)

    # Selection runs once, in the producer's no-grad first call; its replay runs no indexer
    # projection, bounds, scorer or top-k, and no loss path.
    for name in _TOPK_SELECTION_CALLS:
        assert _count(reuse_calls, name) == 1, name
    assert _count(reuse_calls, "build_fused_indexer_varlen_bounds", grad=True) == 0
    for name in (
        "run_fused_qk_topk_with_scores",
        "run_fused_qk_topk_with_loss",
        "run_sparse_indexer_loss_from_topk",
    ):
        assert _count(reuse_calls, name) == 0, name
    # No score stash: the frozen first call publishes indices and their key length only.
    first_call_topk = reused.published.topk[_PRODUCER_LAYER]
    assert reused.published.scores == {} and reused.published.key_len == {}
    assert reused.published.frozen_key_len == {_PRODUCER_LAYER: _CPU_SEQ_LEN}
    # Two first calls and two replays, all on the producer's first-call tensor.
    assert sorted(call.grad for call in reuse_sparse) == [False, False, True, True]
    assert all(call.topk is first_call_topk for call in reuse_sparse)
    assert frozen_cpu.warnings == []

    # Reuse off: the replay reruns the selection and gets a new tensor.
    config.dsa_indexer_reuse_checkpoint_topk = False
    frozen_cpu.calls.clear()
    frozen_cpu.sparse.clear()
    rerun = _frozen_cpu_step([producer, skip], inputs, mode=mode)
    for name in _TOPK_SELECTION_CALLS:
        assert _count(frozen_cpu.calls, name) == 2, name
    first_call_topk = rerun.published.topk[_PRODUCER_LAYER]
    replays = [call for call in frozen_cpu.sparse if call.grad]
    assert any(call.topk is not first_call_topk for call in replays)

    # The reference top-k is deterministic, so both replays select the same keys.
    _assert_equal_steps(reused, rerun)
    # One checkpoint per layer accumulates the input gradients over two backward graphs, in
    # another order than eager; one chunk keeps eager's graph.
    _assert_equal_steps(reused, eager, bitwise=mode == "one-chunk")


def test_frozen_replay_without_published_indices_falls_back(frozen_cpu):
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)
    inputs = _frozen_cpu_inputs(config)
    eager = _frozen_cpu_step([producer], inputs, mode="eager")
    frozen_cpu.calls.clear()

    def drop_indices(payload):
        del payload.topk_by_layer[_PRODUCER_LAYER]

    fallback = _frozen_cpu_step(
        [producer], inputs, mode="per-layer", tamper_first_call=drop_indices
    )
    assert frozen_cpu.warnings == [(_PRODUCER_LAYER, "its first call published no top-k indices")]
    for name in _TOPK_SELECTION_CALLS:
        assert _count(frozen_cpu.calls, name) == 2, name
    _assert_equal_steps(fallback, eager)


def test_frozen_replay_rejects_indices_of_another_geometry(frozen_cpu):
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)

    def truncate_rows(payload):
        payload.topk_by_layer[_PRODUCER_LAYER] = payload.topk_by_layer[_PRODUCER_LAYER][:, :-1]

    with pytest.raises(RuntimeError, match="geometry differs from its first call"):
        _frozen_cpu_step(
            [producer],
            _frozen_cpu_inputs(config),
            mode="per-layer",
            tamper_first_call=truncate_rows,
        )


def test_frozen_replay_rejects_indices_of_another_key_length(frozen_cpu):
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)

    def change_key_length(payload):
        payload.frozen_topk_key_len_by_layer[_PRODUCER_LAYER] += 1

    with pytest.raises(RuntimeError, match="geometry differs from its first call"):
        _frozen_cpu_step(
            [producer],
            _frozen_cpu_inputs(config),
            mode="per-layer",
            tamper_first_call=change_key_length,
        )


def test_frozen_replay_rejects_a_layer_run_twice_in_one_checkpoint(frozen_cpu):
    """One checkpoint keeps one first-call top-k per layer, so a repeated layer must fail."""
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)
    inputs = _frozen_cpu_inputs(config)
    eager = _frozen_cpu_step([producer, producer], inputs, mode="eager")

    # The same layer in two checkpoints is fine: each replay reuses its own first call's.
    frozen_cpu.sparse.clear()
    separate = _frozen_cpu_step([producer, producer], inputs, mode="per-layer")
    first_calls = [call.topk for call in frozen_cpu.sparse if not call.grad]
    replays = [call.topk for call in frozen_cpu.sparse if call.grad]
    assert len(first_calls) == 2 and first_calls[0] is not first_calls[1]
    assert sorted(map(id, replays)) == sorted(map(id, first_calls))
    _assert_equal_steps(separate, eager, bitwise=False)

    with pytest.raises(RuntimeError, match=f"ran layer_number={_PRODUCER_LAYER} twice"):
        _frozen_cpu_step([producer, producer], inputs, mode="one-chunk")

    # Without reuse the replay recomputes each run's selection.
    config.dsa_indexer_reuse_checkpoint_topk = False
    _assert_equal_steps(_frozen_cpu_step([producer, producer], inputs, mode="one-chunk"), eager)


def test_frozen_replay_with_kpool_recomputes_the_selection(frozen_cpu, monkeypatch):
    """KPool is excluded from the reuse at runtime as well as by config validation."""
    config = _make_cpu_reuse_config(num_layers=_SKIP_LAYER)
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)
    pool_size = 2
    indexer = producer.indexer
    indexer.index_kpool = pool_size
    indexer.index_kpool_always_select_tail = True
    indexer.index_kpool_use_quantization = False
    indexer.index_kpool_compress_ape = torch.zeros(pool_size, 8)
    stub_forward_before_topk = indexer.forward_before_topk

    def kpool_forward_before_topk(x, qr, packed_seq_params):
        q, k, weights = stub_forward_before_topk(x, qr, packed_seq_params)
        indexer._kpool_gate_score = torch.zeros_like(k)
        return q, k, weights

    indexer.forward_before_topk = kpool_forward_before_topk
    kpool_topk = dsa_module.fused_qk_topk_kpool

    def recorded_kpool_topk(*args, **kwargs):
        frozen_cpu.calls.append(("fused_qk_topk_kpool", torch.is_grad_enabled()))
        return kpool_topk(*args, **kwargs)

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", recorded_kpool_topk)
    inputs = _frozen_cpu_inputs(config)
    eager = _frozen_cpu_step([producer], inputs, mode="eager")
    frozen_cpu.calls.clear()
    frozen_cpu.sparse.clear()
    replayed = _frozen_cpu_step([producer], inputs, mode="per-layer")

    # The first call and the replay both run the indexer and the KPool selection.
    assert _count(frozen_cpu.calls, "indexer") == 2
    assert _count(frozen_cpu.calls, "fused_qk_topk_kpool") == 2
    first_call, replay = frozen_cpu.sparse
    assert not first_call.grad and replay.grad and replay.topk is not first_call.topk
    assert frozen_cpu.warnings == []
    _assert_equal_steps(replayed, eager)


@pytest.mark.parametrize(
    ("loss_coeff", "stashes"), [(0.0, False), (0.01, True)], ids=["frozen", "unfrozen"]
)
def test_first_call_stashes_scores_only_with_indexer_loss(frozen_cpu, loss_coeff, stashes):
    """O1's no-grad first call still asks for the selected scores when the loss is on."""
    config = _make_cpu_reuse_config(
        num_layers=_SKIP_LAYER, dsa_indexer_loss_coeff=loss_coeff, dsa_indexer_use_sparse_loss=True
    )
    producer = _make_frozen_cpu_layer(config, _PRODUCER_LAYER, frozen_cpu)
    inputs = _frozen_cpu_inputs(config)
    mask = _causal_mask()
    with forward_sharing_lifetime(None, mask, config) as state:
        wrapped = preserve_forward_sharing_for_checkpoint(
            lambda *args: producer(
                query=args[0],
                key=args[1],
                value=None,
                attention_mask=args[5],
                x=args[2],
                qr=args[3],
                up_v_weight=args[4],
                attn_mask_type=AttnMaskType.causal,
            ),
            None,
            config,
            attention_mask_arg=_ATTENTION_MASK_ARG,
        )
        query = inputs["query"].clone().requires_grad_(True)
        with torch.no_grad():
            wrapped(query, inputs["key"], inputs["x"], inputs["qr"], inputs["up_v_weight"], mask)
        payload = state.get(dsa_module._DSAIndexSharingPayload)
        assert _PRODUCER_LAYER in payload.topk_by_layer
        # The frozen replay's key-length record is not written with the indexer loss on.
        frozen_key_len = dict(payload.frozen_topk_key_len_by_layer)
    assert frozen_key_len == ({} if stashes else {_PRODUCER_LAYER: _CPU_SEQ_LEN})
    assert _count(frozen_cpu.calls, "run_fused_qk_topk_with_scores") == int(stashes)
    # No backend is installed, so the stash is declined and the plain top-k runs.
    assert _count(frozen_cpu.calls, "run_fused_qk_topk") == 1


@pytest.mark.parametrize(
    "shared_components",
    [[SPARSE_ATTENTION_INDEX], []],
    ids=["mtp-shares-indices", "each-depth-selects"],
)
def test_frozen_replay_with_repeated_mtp_depths(frozen_cpu, shared_components):
    """Every depth's replay sees the indices its own first call used."""
    config = _make_cpu_reuse_config(
        num_layers=2,
        mtp_num_layers=2,
        mtp_repeated_layer_shared_components=shared_components,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    core = _make_frozen_cpu_layer(config, 1, frozen_cpu, is_mtp_layer=True)
    # The repeated physical MTP layer is global layer 3, a top-k-computing layer.
    assert core.layer_number == _PRODUCER_LAYER and not core.skip_topk
    mask = _causal_mask()
    depth_inputs = [torch.zeros(_CPU_SEQ_LEN, 1, 3, requires_grad=True) for _ in range(2)]
    decoder_inputs = [torch.full((_CPU_SEQ_LEN, 1, 3), float(depth)) for depth in range(2)]
    depth_of_call = []

    def projected_forward(hidden_states, decoder_input, attention_mask=None, **_kwargs):
        depth_of_call.append(int(decoder_input[0, 0, 0]))
        inputs = _frozen_cpu_inputs(config)
        inputs["query"] = inputs["query"] * (1 + (hidden_states + decoder_input).mean())
        return core(
            query=inputs["query"],
            key=inputs["key"],
            value=None,
            attention_mask=attention_mask,
            x=inputs["x"],
            qr=inputs["qr"],
            up_v_weight=inputs["up_v_weight"],
            attn_mask_type=AttnMaskType.causal,
        )

    layer = SimpleNamespace(
        config=config,
        mtp_layer_pattern=None,
        mtp_model_layer=SimpleNamespace(self_attention=SimpleNamespace(core_attention=core)),
        _proj_and_transformer_layer=projected_forward,
    )
    outputs = []
    with forward_sharing_lifetime(None, mask, config) as state:
        sharing = (
            mtp_repeated_sharing_lifetime(state, core.layer_number)
            if shared_components
            else nullcontext({})
        )
        with sharing as flags:
            for depth in range(2):
                flags[core.layer_number] = depth == 0
                outputs.append(
                    MultiTokenPredictionLayer._checkpointed_forward(
                        layer, depth_inputs[depth], decoder_inputs[depth], attention_mask=mask
                    )
                )
    sum(output.float().sum() for output in outputs).backward()

    assert all(depth_input.grad is not None for depth_input in depth_inputs)
    assert frozen_cpu.warnings == []
    calls = list(zip(depth_of_call, frozen_cpu.sparse))
    assert len(calls) == 4
    first = {depth: call.topk for depth, call in calls if not call.grad}
    replays = [(depth, call.topk) for depth, call in calls if call.grad]
    assert sorted(first) == [0, 1] and sorted(depth for depth, _ in replays) == [0, 1]
    for depth, topk in replays:
        # A consumer's replay receives the source's indices as a detached checkpoint input.
        assert _same_tensor(topk, first[depth]), depth
    if shared_components:
        # Depth 0 selects; depth 1 consumes its indices through MTP sharing.
        assert _same_tensor(first[1], first[0])
        assert next(topk for depth, topk in replays if depth == 0) is first[0]
        assert _count(frozen_cpu.calls, "indexer") == 1
    else:
        # The same physical layer selects at each depth; each replay reuses its own depth's.
        assert first[1] is not first[0]
        assert _count(frozen_cpu.calls, "indexer") == 2
    assert _count(frozen_cpu.calls, "run_fused_qk_topk_with_scores") == 0
