# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from datetime import timedelta

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

_SIMILARITY_EPS = 5e-3
_DISTRIBUTED_TEST_TIMEOUT = timedelta(minutes=30)


def _initialize_model_parallel() -> None:
    # cuDNN/TileLang DSA tests can have large first-use compile imbalance across ranks.
    # Keep cleanup barriers from timing out while slower ranks are still executing kernels.
    init_process_group = torch.distributed.init_process_group
    barrier = torch.distributed.barrier

    def init_process_group_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", _DISTRIBUTED_TEST_TIMEOUT)
        return init_process_group(*args, **kwargs)

    def barrier_with_current_device(*args, **kwargs):
        if torch.distributed.is_initialized() and torch.distributed.get_backend() == "nccl":
            kwargs.setdefault("device_ids", [torch.cuda.current_device()])
        return barrier(*args, **kwargs)

    try:
        torch.distributed.init_process_group = init_process_group_with_timeout
        torch.distributed.barrier = barrier_with_current_device
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    finally:
        torch.distributed.init_process_group = init_process_group
        torch.distributed.barrier = barrier


def _skip_if_backend_unavailable(backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused DSA backend parity")
    if Utils.world_size < 4:
        pytest.skip("TP2+CP2 DSA parity requires at least 4 distributed ranks")
    if backend == "cudnn":
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("cuDNN fused DSA path requires SM90+")
        missing = []
        try:
            from cudnn import DSA  # noqa: F401
        except ImportError:
            missing.append("cudnn-frontend DSA")
        try:
            from flash_mla import flash_mla_sparse_fwd  # noqa: F401
        except ImportError:
            missing.append("flash_mla")
        if missing:
            pytest.skip(f"cuDNN fused DSA dependencies are unavailable: {', '.join(missing)}")
    elif backend == "tilelang":
        try:
            from megatron.core.transformer.experimental_attention_variant.ops import tilelang_dsa
        except (ImportError, OSError, AttributeError) as exc:
            pytest.skip(f"TileLang DSA backend code is unavailable: {exc}")

        if tilelang_dsa.lighting_indexer is None or tilelang_dsa.SparseMLA is None:
            pytest.skip("TileLang DSA kernels are unavailable")
    else:
        raise AssertionError(f"Unexpected backend {backend}")


def _make_config(
    backend: str,
    *,
    num_attention_heads: int = 64,
    indexer_topk_freq: int = 1,
    indexer_loss_coeff: float = 0.01,
) -> MLATransformerConfig:
    fused = backend != "none"
    return MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=num_attention_heads,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=64,
        dsa_indexer_topk_freq=indexer_topk_freq,
        dsa_indexer_skip_topk_offset=1,
        dsa_indexer_loss_coeff=indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
        calculate_per_token_loss=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=2,
        cp_comm_type="allgather",
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_percent=1.0,
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=False,
        fp4=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        kv_channels=128,
        num_query_groups=num_attention_heads,
        batch_invariant_mode=False,
        cache_mla_latents=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        softmax_scale=None,
        attention_backend=AttnBackend.auto if fused else AttnBackend.unfused,
        dsa_kernel_backend=backend,
    )


def _make_attention(
    config: MLATransformerConfig, pg_collection: ProcessGroupCollection, *, layer_number: int = 1
):
    indexer_submodules = DSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_wk=ModuleSpec(module=TELinear),
        k_norm=ModuleSpec(module=TENorm),
        linear_weights_proj=ModuleSpec(module=TELinear),
    )
    indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
    sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)
    return DSAttention(
        config=config,
        submodules=sparse_attention_submodules,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()


def _randn(shape, *, generator: torch.Generator, dtype: torch.dtype = torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)


def _make_tp2_sp_packed_cp_inputs(
    config: MLATransformerConfig, *, sequence_local_query: bool = False
):
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    assert tp_size == 2
    assert cp_size == 2

    batch = 1
    local_query_rows = 48
    local_cp_rows = local_query_rows * tp_size
    global_sequence = local_cp_rows * cp_size
    local_heads = config.num_attention_heads // tp_size
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim

    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026 + cp_rank)

    head_start = tp_rank * local_heads
    head_end = head_start + local_heads

    query_base = _randn(
        (local_cp_rows, batch, config.num_attention_heads, absorbed_dim), generator=generator
    )
    key = _randn((local_cp_rows, batch, 1, absorbed_dim), generator=generator)
    # Keep selected indexer scores away from the top-k/ReLU boundary so this parity test checks
    # the TP/SP reductions rather than backend-specific tie behavior in nearly equal logits.
    x_base = _randn((local_cp_rows, batch, config.hidden_size), generator=generator) * 8.0
    qr_base = _randn((local_cp_rows, batch, config.q_lora_rank), generator=generator) * 8.0
    up_v_weight_base = _randn(
        (config.num_attention_heads, config.v_head_dim, config.kv_lora_rank), generator=generator
    )

    cu_seqlens = torch.tensor([0, global_sequence], dtype=torch.int32, device="cuda")
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=global_sequence,
        max_seqlen_kv=global_sequence,
    )

    query = query_base[:, :, head_start:head_end, :]
    if sequence_local_query:
        query = query[tp_rank * local_query_rows : (tp_rank + 1) * local_query_rows]

    return {
        "query": query.contiguous(),
        "key": key.contiguous(),
        "x": x_base[tp_rank * local_query_rows : (tp_rank + 1) * local_query_rows].contiguous(),
        "qr": qr_base[tp_rank * local_query_rows : (tp_rank + 1) * local_query_rows].contiguous(),
        "up_v_weight": up_v_weight_base[head_start:head_end].contiguous(),
        "packed_seq_params": packed_seq_params,
    }


def _clone_inputs(inputs: dict[str, torch.Tensor | PackedSeqParams]):
    cloned = {}
    for name, value in inputs.items():
        if isinstance(value, PackedSeqParams):
            cloned[name] = PackedSeqParams(
                qkv_format=value.qkv_format,
                cu_seqlens_q=value.cu_seqlens_q.clone(),
                cu_seqlens_kv=value.cu_seqlens_kv.clone(),
                cu_seqlens_q_padded=value.cu_seqlens_q_padded.clone(),
                cu_seqlens_kv_padded=value.cu_seqlens_kv_padded.clone(),
                max_seqlen_q=value.max_seqlen_q,
                max_seqlen_kv=value.max_seqlen_kv,
            )
        else:
            cloned_value = value.detach().clone()
            if name in {"query", "key", "up_v_weight"}:
                cloned_value.requires_grad_(True)
            cloned[name] = cloned_value
    return cloned


def _forward_attention(attention: DSAttention, inputs: dict[str, torch.Tensor | PackedSeqParams]):
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    output = attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=None,
        x=inputs["x"],
        qr=inputs["qr"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=inputs["packed_seq_params"],
        up_v_weight=inputs["up_v_weight"],
    )
    loss = DSAIndexerLossLoggingHelper.tracker["values"][0].detach().clone()
    return output, loss


def _forward_attention_output_only(
    attention: DSAttention, inputs: dict[str, torch.Tensor | PackedSeqParams]
):
    return attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=None,
        x=inputs["x"],
        qr=inputs["qr"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=inputs["packed_seq_params"],
        up_v_weight=inputs["up_v_weight"],
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_similarity(
    a: torch.Tensor, b: torch.Tensor, *, label: str, eps: float = _SIMILARITY_EPS
):
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    assert torch.isfinite(b).all(), f"{label} reference has non-finite values"
    cosine = _cosine_sim(a, b)
    tensor = _tensor_sim(a, b)
    norm_ratio = (a.double().norm() / b.double().norm().clamp_min(1e-30)).item()
    assert cosine > 1 - eps, f"{label}: cosine_sim={cosine:.6f}, norm_ratio={norm_ratio:.6f}"
    assert tensor > 1 - eps, f"{label}: tensor_sim={tensor:.6f}, norm_ratio={norm_ratio:.6f}"


def _tp_averaged_grad(param: torch.nn.Parameter) -> torch.Tensor:
    grad = param.grad.detach().clone()
    if (
        getattr(param, "average_gradients_across_tp_domain", False)
        and parallel_state.get_tensor_model_parallel_world_size() > 1
    ):
        torch.distributed.all_reduce(grad, group=parallel_state.get_tensor_model_parallel_group())
        grad /= parallel_state.get_tensor_model_parallel_world_size()
    return grad


@contextlib.contextmanager
def _record_backend_calls(backend: str, monkeypatch: pytest.MonkeyPatch):
    called = {"topk": 0, "indexer_loss": 0, "attention": 0, "full": 0}
    if backend == "cudnn":
        original_full = dsa_cudnn_kernels.run_fused_dsa_attention
        original_topk_only = dsa_cudnn_kernels.run_fused_qk_topk
        original_topk = dsa_cudnn_kernels.run_fused_qk_topk_with_loss
        original_attention = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention

        def wrapped_run_fused_dsa_attention(*args, **kwargs):
            result = original_full(*args, **kwargs)
            if result is not None:
                called["full"] += 1
            return result

        def wrapped_run_fused_qk_topk(*args, **kwargs):
            result = original_topk_only(*args, **kwargs)
            if result is not None:
                called["topk"] += 1
            return result

        def wrapped_run_fused_qk_topk_with_loss(*args, **kwargs):
            result = original_topk(*args, **kwargs)
            if result is not None:
                called["indexer_loss"] += 1
            return result

        def wrapped_run_fused_absorbed_sparse_attention(*args, **kwargs):
            result = original_attention(*args, **kwargs)
            if result is not None:
                called["attention"] += 1
            return result

        monkeypatch.setattr(
            dsa_cudnn_kernels, "run_fused_dsa_attention", wrapped_run_fused_dsa_attention
        )
        monkeypatch.setattr(dsa_cudnn_kernels, "run_fused_qk_topk", wrapped_run_fused_qk_topk)
        monkeypatch.setattr(
            dsa_cudnn_kernels, "run_fused_qk_topk_with_loss", wrapped_run_fused_qk_topk_with_loss
        )
        monkeypatch.setattr(
            dsa_cudnn_kernels,
            "run_fused_absorbed_sparse_attention",
            wrapped_run_fused_absorbed_sparse_attention,
        )
    else:
        try:
            from megatron.core.transformer.experimental_attention_variant import (
                dsa_tilelang_kernels,
            )
        except (ImportError, OSError, AttributeError) as exc:
            pytest.skip(f"TileLang DSA backend code is unavailable: {exc}")

        original_topk_only = dsa_tilelang_kernels.run_fused_qk_topk
        original_topk = dsa_tilelang_kernels.run_fused_qk_topk_with_loss
        original_attention = dsa_tilelang_kernels.run_fused_absorbed_sparse_attention

        def wrapped_run_fused_qk_topk(*args, **kwargs):
            result = original_topk_only(*args, **kwargs)
            if result is not None:
                called["topk"] += 1
            return result

        def wrapped_run_fused_qk_topk_with_loss(*args, **kwargs):
            result = original_topk(*args, **kwargs)
            if result is not None:
                called["indexer_loss"] += 1
            return result

        def wrapped_run_fused_absorbed_sparse_attention(*args, **kwargs):
            result = original_attention(*args, **kwargs)
            if result is not None:
                called["attention"] += 1
            return result

        monkeypatch.setattr(dsa_tilelang_kernels, "run_fused_qk_topk", wrapped_run_fused_qk_topk)
        monkeypatch.setattr(
            dsa_tilelang_kernels, "run_fused_qk_topk_with_loss", wrapped_run_fused_qk_topk_with_loss
        )
        monkeypatch.setattr(
            dsa_tilelang_kernels,
            "run_fused_absorbed_sparse_attention",
            wrapped_run_fused_absorbed_sparse_attention,
        )
    yield called


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
def test_packed_cp_tp2_sequence_local_query_backend_matches_unfused_attention(
    backend: str, monkeypatch: pytest.MonkeyPatch
):
    _skip_if_backend_unavailable(backend)

    _initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        reference_config = _make_config("none", indexer_loss_coeff=0.0)
        fused_config = _make_config(backend, indexer_loss_coeff=0.0)
        reference_attention = _make_attention(reference_config, pg_collection).train()
        fused_attention = _make_attention(fused_config, pg_collection).train()
        fused_attention.indexer.load_state_dict(reference_attention.indexer.state_dict())

        inputs = _make_tp2_sp_packed_cp_inputs(reference_config, sequence_local_query=True)
        reference_inputs = _clone_inputs(inputs)
        fused_inputs = _clone_inputs(inputs)

        with _record_backend_calls(backend, monkeypatch) as called:
            fused_output = _forward_attention_output_only(fused_attention, fused_inputs)
        reference_output = _forward_attention_output_only(reference_attention, reference_inputs)

        if backend == "cudnn":
            assert called["full"] > 0, "cuDNN full fused DSA path was not exercised"
        else:
            assert called["topk"] > 0, "TileLang fused indexer path was not exercised"
            assert called["attention"] > 0, "TileLang fused sparse-attention path was not exercised"

        _assert_similarity(fused_output.detach(), reference_output.detach(), label="output")

        grad_output = torch.randn_like(reference_output)
        fused_output.backward(grad_output)
        reference_output.backward(grad_output)
        for name in ("query", "key", "up_v_weight"):
            _assert_similarity(
                fused_inputs[name].grad, reference_inputs[name].grad, label=f"{name} grad"
            )

        loss_config = _make_config(backend)
        loss_attention = _make_attention(loss_config, pg_collection).train()
        with pytest.raises(RuntimeError, match="same query rows"):
            _forward_attention_output_only(loss_attention, _clone_inputs(inputs))
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
@pytest.mark.parametrize("num_attention_heads", [64, 128])
def test_packed_cp_tp2_sequence_parallel_backend_matches_unfused_reference(
    backend: str, num_attention_heads: int, monkeypatch: pytest.MonkeyPatch
):
    _skip_if_backend_unavailable(backend)

    _initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        reference_config = _make_config("none", num_attention_heads=num_attention_heads)
        fused_config = _make_config(backend, num_attention_heads=num_attention_heads)
        reference_attention = _make_attention(reference_config, pg_collection).train()
        fused_attention = _make_attention(fused_config, pg_collection).train()
        fused_attention.indexer.load_state_dict(reference_attention.indexer.state_dict())

        inputs = _make_tp2_sp_packed_cp_inputs(reference_config)
        reference_inputs = _clone_inputs(inputs)
        fused_inputs = _clone_inputs(inputs)

        with _record_backend_calls(backend, monkeypatch) as called:
            fused_output, fused_loss = _forward_attention(fused_attention, fused_inputs)
        reference_output, reference_loss = _forward_attention(reference_attention, reference_inputs)

        if backend == "cudnn":
            assert called["full"] > 0, "cuDNN full fused DSA path was not exercised"
        else:
            assert called["indexer_loss"] > 0, "TileLang fused indexer-loss path was not exercised"
            assert called["attention"] > 0, "TileLang fused sparse-attention path was not exercised"

        _assert_similarity(fused_output.detach(), reference_output.detach(), label="output")
        torch.testing.assert_close(fused_loss, reference_loss, rtol=5e-2, atol=5e-3)

        grad_output = torch.randn_like(reference_output)
        fused_output.backward(grad_output)
        reference_output.backward(grad_output)

        for name in ("query", "key", "up_v_weight"):
            _assert_similarity(
                fused_inputs[name].grad, reference_inputs[name].grad, label=f"{name} grad"
            )

        fused_params = dict(fused_attention.indexer.named_parameters())
        reference_params = dict(reference_attention.indexer.named_parameters())
        for name, fused_param in fused_params.items():
            reference_param = reference_params[name]
            assert fused_param.grad is not None, f"{name} fused grad missing"
            assert reference_param.grad is not None, f"{name} reference grad missing"
            _assert_similarity(
                _tp_averaged_grad(fused_param),
                _tp_averaged_grad(reference_param),
                label=f"{name} grad",
            )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
def test_packed_cp_tp2_sequence_parallel_shared_indexer_backend_matches_unfused_reference(
    backend: str, monkeypatch: pytest.MonkeyPatch
):
    _skip_if_backend_unavailable(backend)

    _initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        reference_config = _make_config("none", num_attention_heads=64, indexer_topk_freq=2)
        fused_config = _make_config(backend, num_attention_heads=64, indexer_topk_freq=2)
        reference_attention = _make_attention(reference_config, pg_collection).train()
        fused_attention = _make_attention(fused_config, pg_collection).train()
        fused_attention.indexer.load_state_dict(reference_attention.indexer.state_dict())

        inputs = _make_tp2_sp_packed_cp_inputs(reference_config)
        reference_inputs = _clone_inputs(inputs)
        fused_inputs = _clone_inputs(inputs)

        with _record_backend_calls(backend, monkeypatch) as called:
            fused_output, fused_loss = _forward_attention(fused_attention, fused_inputs)
        reference_output, reference_loss = _forward_attention(reference_attention, reference_inputs)

        assert called["full"] == 0, "index sharing should use split fused kernels, not full fusion"
        assert called["indexer_loss"] > 0, "fused split indexer-loss path was not exercised"
        assert called["attention"] > 0, "fused split sparse-attention path was not exercised"

        _assert_similarity(fused_output.detach(), reference_output.detach(), label="output")
        torch.testing.assert_close(fused_loss, reference_loss, rtol=5e-2, atol=5e-3)

        grad_output = torch.randn_like(reference_output)
        fused_output.backward(grad_output)
        reference_output.backward(grad_output)

        for name in ("query", "key", "up_v_weight"):
            _assert_similarity(
                fused_inputs[name].grad, reference_inputs[name].grad, label=f"{name} grad"
            )

        fused_params = dict(fused_attention.indexer.named_parameters())
        reference_params = dict(reference_attention.indexer.named_parameters())
        for name, fused_param in fused_params.items():
            reference_param = reference_params[name]
            assert fused_param.grad is not None, f"{name} fused grad missing"
            assert reference_param.grad is not None, f"{name} reference grad missing"
            _assert_similarity(
                _tp_averaged_grad(fused_param),
                _tp_averaged_grad(reference_param),
                label=f"{name} grad",
            )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


# Disabled in dev (flaky_in_dev) and LTS (flaky) CI: this real-kernel cuDNN/flash_mla
# case fails with a CUDA error in CI (deterministic, not truly flaky). Re-enable once the
# kernel/build root cause is resolved.
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize("backend", ["cudnn", "tilelang"])
def test_packed_cp_tp2_sequence_parallel_shared_skip_backend_matches_unfused_reference(
    backend: str, monkeypatch: pytest.MonkeyPatch
):
    _skip_if_backend_unavailable(backend)

    _initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        reference_config = _make_config("none", num_attention_heads=64, indexer_topk_freq=2)
        fused_config = _make_config(backend, num_attention_heads=64, indexer_topk_freq=2)
        reference_source = _make_attention(reference_config, pg_collection, layer_number=1).train()
        fused_source = _make_attention(fused_config, pg_collection, layer_number=1).train()
        reference_skip = _make_attention(reference_config, pg_collection, layer_number=2).train()
        fused_skip = _make_attention(fused_config, pg_collection, layer_number=2).train()
        fused_source.indexer.load_state_dict(reference_source.indexer.state_dict())

        inputs = _make_tp2_sp_packed_cp_inputs(reference_config)
        reference_source_inputs = _clone_inputs(inputs)
        fused_source_inputs = _clone_inputs(inputs)
        reference_skip_inputs = _clone_inputs(inputs)
        fused_skip_inputs = _clone_inputs(inputs)
        reference_skip_inputs["packed_seq_params"] = reference_source_inputs["packed_seq_params"]
        fused_skip_inputs["packed_seq_params"] = fused_source_inputs["packed_seq_params"]

        with _record_backend_calls(backend, monkeypatch) as called:
            _source_output, fused_loss = _forward_attention(fused_source, fused_source_inputs)
            fused_skip_output = _forward_attention_output_only(fused_skip, fused_skip_inputs)
        _reference_source_output, reference_loss = _forward_attention(
            reference_source, reference_source_inputs
        )
        reference_skip_output = _forward_attention_output_only(
            reference_skip, reference_skip_inputs
        )

        assert called["full"] == 0, "index sharing should use split fused kernels, not full fusion"
        assert called["indexer_loss"] > 0, "source layer did not exercise fused split indexer loss"
        assert called["attention"] > 1, "source+skip layers did not both exercise fused attention"

        torch.testing.assert_close(fused_loss, reference_loss, rtol=5e-2, atol=5e-3)
        _assert_similarity(
            fused_skip_output.detach(), reference_skip_output.detach(), label="skip output"
        )

        grad_output = torch.randn_like(reference_skip_output)
        fused_skip_output.backward(grad_output)
        reference_skip_output.backward(grad_output)

        for name in ("query", "key", "up_v_weight"):
            _assert_similarity(
                fused_skip_inputs[name].grad,
                reference_skip_inputs[name].grad,
                label=f"skip {name} grad",
            )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
