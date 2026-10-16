# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_reuse_checkpoint_topk``: the replay of a checkpointed DSA layer reuses the
top-k indices and scores selected by its first call instead of rerunning the indexer."""

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
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
    preserve_forward_sharing_for_checkpoint,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_kernel_backend": "none"}, "dsa_kernel_backend='cudnn'"),
        ({"attention_backend": AttnBackend.unfused}, "attention_backend != unfused"),
        ({"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss"),
        ({"dsa_indexer_loss_coeff": 0.0}, "sparse indexer loss"),
        ({"dsa_indexer_topk_freq": 1}, "dsa_indexer_topk_freq > 1"),
        (
            {"overlap_moe_expert_parallel_comm": True, "recompute_granularity": "full"},
            "overlap_moe_expert_parallel_comm",
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
