# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Qwen3.5-style partial capture must include GDN, softmax attention and MoE prefixes."""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel_layout.graph import (
    cache_packed_gdn_graph_validation,
    supports_packed_gdn_graph,
    validate_packed_gdn_graph_inputs,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import gdn as gdn_module
from megatron.core.ssm.gated_delta_net.common import HAVE_FLA, build_graph_cp_context
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend, CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.transformer.dcp_graph_test_utils import (
    assert_gradients_close,
    dcp_batch,
    dcp_test_context,
    dcp_train_step,
    requires_te_graph_memory_slots,
    track_graph_replays,
)

pytestmark = pytest.mark.internal
requires_fla_graph = pytest.mark.skipif(
    build_graph_cp_context is None, reason="requires an FLA build with fla.ops.cp.graph"
)
_CU_NAMES = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_SCOPE = ["attn", "moe_router", "moe_preprocess"]


def _config(**overrides):
    values = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=256,
        ffn_hidden_size=512,
        experimental_attention_variant="gdn",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_cp_mode="chunkwise",
        cp_partition_mode="contiguous",
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.fused,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        qk_layernorm=True,
        attention_output_gate=True,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_gate=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=1.25,
        moe_use_grouped_tensor=True,
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=512,
        thd_max_packed_sequences=8,
        sequence_packing_scheduler="default_dynamic_cp",
        pad_packed_seq_alignment="max",
        thd_tail_padding_policy="append_dummy_seq",
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=_SCOPE,
        cuda_graph_dynamic_microbatches=True,
        cuda_graph_warmup_steps=3,
    )
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.parametrize("spelling", ["gdn", "gated_delta_net"])
def test_partial_gdn_config_accepts_canonical_and_legacy_names(spelling):
    config = _config(experimental_attention_variant=spelling)
    assert config.experimental_attention_variant == "gdn"
    assert supports_packed_gdn_graph(config)
    assert config.cuda_graph_modules == [
        CudaGraphModule.attn,
        CudaGraphModule.moe_router,
        CudaGraphModule.moe_preprocess,
    ]


def test_partial_headwise_gdn_accepts_cp16_and_rejects_cp32():
    changes = dict(
        linear_cp_mode="headwise",
        cp_partition_mode="zigzag",
        linear_num_key_heads=16,
        linear_num_value_heads=64,
    )
    config = _config(context_parallel_size=16, **changes)
    assert supports_packed_gdn_graph(config)
    validate_packed_gdn_graph_inputs(config, _params(config, 16))
    with pytest.raises(AssertionError, match="linear_num_key_heads"):
        _config(context_parallel_size=32, **changes)
    with pytest.raises(ValueError, match="evenly divide"):
        validate_packed_gdn_graph_inputs(config, _params(config, 32))


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_headwise_bijective_permutation_matches_index_select(dim, dtype, noncontiguous):
    from megatron.core.ssm.gated_delta_net.common import _BijectiveIndexSelect

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    tensor = torch.randn(2, 17, 11, device="cuda", dtype=dtype).transpose(0, 1)
    if not noncontiguous:
        tensor = tensor.contiguous()
    native = tensor.detach().requires_grad_()
    candidate = tensor.detach().clone().requires_grad_()
    indices = torch.randperm(tensor.shape[dim], device="cuda")
    grad_output = torch.randn_like(tensor)
    expected = native.index_select(dim, indices)
    actual = _BijectiveIndexSelect.apply(candidate, dim, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected, native, grad_output)[0]
    actual_grad = torch.autograd.grad(actual, candidate, grad_output)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_headwise_graph_accepts_non_chunk_aligned_sequences(size):
    config = _config(
        linear_cp_mode="headwise",
        cp_partition_mode="zigzag",
        linear_num_key_heads=16,
        linear_num_value_heads=64,
    )
    params = _params(config, size, physical=[0, 130 * size, 384 * size, 512 * size])
    validate_packed_gdn_graph_inputs(config, params)


@requires_fla_graph
@pytest.mark.parametrize(
    "boundaries", [[0, 130, 384, 512, 512], [0, 2, 6, 510, 512], [0, 2, 4, 6, 8, 10, 12, 14, 512]]
)
def test_headwise_graph_context_preserves_unaligned_sequence_boundaries(boundaries):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    ctx = build_graph_cp_context(cu, None, 512, allow_unaligned_sequences=True)
    torch.testing.assert_close(ctx.cu_seqlens, cu)
    counts = [(end - start + 63) // 64 for start, end in zip(boundaries, boundaries[1:])]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    assert ctx.chunk_offsets.tolist() == offsets
    expected = [[seq, chunk] for seq, count in enumerate(counts) for chunk in range(count)]
    capacity = 512 // 64 + len(boundaries) - 2
    last = len(counts) - 1
    expected.extend([last, counts[last] + extra] for extra in range(capacity - len(expected)))
    assert ctx.chunk_indices.tolist() == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"linear_cp_mode": "headwise"},
        {"cp_partition_mode": "zigzag"},
        {"tensor_model_parallel_size": 2, "sequence_parallel": True},
        {"max_seqlen_per_dp_cp_rank": 510},
        {"cuda_graph_impl": "local"},
    ],
)
def test_partial_gdn_config_rejects_unsupported_geometry(changes):
    with pytest.raises((ValueError, AssertionError)):
        _config(**changes)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"cuda_graph_dynamic_microbatches": False}, "require dynamic microbatch slots"),
        ({"moe_paged_stash": True}, "moe_paged_stash"),
        ({"cuda_graph_modules": ["attn", "mamba"]}, "remove 'mamba'"),
    ],
)
def test_dynamic_cp_graph_config_rules(changes, error):
    with pytest.raises(ValueError, match=error):
        _config(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"recompute_granularity": "selective", "recompute_modules": ["gdn"]},
        {"recompute_granularity": "selective", "recompute_modules": ["gdn_norm_out"]},
        {"gdn_conv_pad_alignment": 64},
        {"is_hybrid_model": True, "enable_hyper_connections": True},
    ],
)
def test_packed_gdn_graph_excludes_paths_slot_memory_cannot_protect(changes):
    config = _config()
    assert supports_packed_gdn_graph(config)
    for name, value in changes.items():
        setattr(config, name, value)
    assert not supports_packed_gdn_graph(config)


def _params(config, size, *, valid=None, physical=None):
    end = config.max_seqlen_per_dp_cp_rank * size
    physical = physical or [0, 128 * size, end]
    valid = valid or physical
    tensors = []
    for boundaries in (valid, valid, physical, physical):
        tensors.append(
            torch.tensor(
                boundaries
                + [boundaries[-1]] * (config.thd_max_packed_sequences + 1 - len(boundaries)),
                dtype=torch.int32,
            )
        )
    return PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode=config.cp_partition_mode,
        local_cp_size=size,
        cp_group=object(),
        **dict(zip(_CU_NAMES, tensors)),
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32])
def test_partial_gdn_metadata_roundtrip_preserves_group_and_graph_backend(size):
    config = _config()
    params = _params(config, size, valid=[0, 100 * size, 480 * size])
    config._cuda_graph_capture_dynamic_cp = (size, params.cp_group)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.self_attention = IdentityOp()
    kwargs = dict(packed_seq_params=params)
    layer._decompose_packed_seq_params_to_kwargs(kwargs)
    assert "packed_seq_params" not in kwargs
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)
    restored = kwargs["packed_seq_params"]
    assert restored.gdn_graph_safe
    assert restored.local_cp_size == size and restored.cp_group is params.cp_group
    for name in _CU_NAMES:
        assert getattr(restored, name) is getattr(params, name)


def test_partial_gdn_revalidates_mutated_packs():
    config = _config()
    params = _params(config, 4)
    validate_packed_gdn_graph_inputs(config, params)
    for name in _CU_NAMES:
        getattr(params, name)[1] += 1
    with pytest.raises(ValueError, match="64/2CP-aligned"):
        validate_packed_gdn_graph_inputs(config, params)
    params = _params(config, 4, physical=[0, 0, 2048])
    with pytest.raises(ValueError, match="empty physical"):
        validate_packed_gdn_graph_inputs(config, params)


@pytest.mark.parametrize("raw_storage_mutation", [False, True])
def test_validation_cache_is_forward_scoped_and_rechecks_mutations(
    monkeypatch, raw_storage_mutation
):
    config = _config()
    params = _params(config, 1)
    cpu_reads = []
    original_cpu = torch.Tensor.cpu

    def tracked_cpu(tensor, *args, **kwargs):
        cpu_reads.append(1)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked_cpu)

    class Model:
        @cache_packed_gdn_graph_validation
        def forward(self, mutate=False):
            validate_packed_gdn_graph_inputs(self.config, params)
            validate_packed_gdn_graph_inputs(self.config, params)
            if mutate:
                for name in _CU_NAMES:
                    getattr(params, name)[1] += 1
                validate_packed_gdn_graph_inputs(self.config, params)

    model = Model()
    model.config = config
    model.forward()
    assert len(cpu_reads) == 1
    validate_packed_gdn_graph_inputs(config, params)
    assert len(cpu_reads) == 2, "standalone replay must remain uncached"
    if raw_storage_mutation:
        # Simulate a producer update without a Tensor version bump between forwards.
        for name in _CU_NAMES:
            getattr(params, name).data[1] += 1
        with pytest.raises(ValueError, match="64/2CP-aligned"):
            model.forward()
        assert len(cpu_reads) == 3
    else:
        with pytest.raises(ValueError, match="64/2CP-aligned"):
            model.forward(mutate=True)
        assert len(cpu_reads) == 4
    for name in _CU_NAMES:
        getattr(params, name)[1] = 128
    validate_packed_gdn_graph_inputs(config, params)
    assert len(cpu_reads) == (
        4 if raw_storage_mutation else 5
    ), "failed forwards must clear the cache"


@pytest.mark.skipif(not HAVE_FLA, reason="requires FLA")
def test_attention_capture_requires_fla_graph_extension(monkeypatch):
    monkeypatch.setattr(gdn_module, "build_graph_cp_context", None)
    group = SimpleNamespace(size=lambda: 1)
    with pytest.raises(ImportError, match="fla.ops.cp.graph"):
        gdn_module.GatedDeltaNet(
            _config(),
            submodules=None,
            layer_number=1,
            pg_collection=SimpleNamespace(tp=group, cp=group),
        )


def test_moe_only_scope_does_not_require_gdn_graph_geometry():
    config = _config(
        cuda_graph_modules=["moe_router"], tensor_model_parallel_size=2, sequence_parallel=True
    )
    assert not supports_packed_gdn_graph(config)
    params = _params(config, 4)
    config._cuda_graph_capture_dynamic_cp = (4, params.cp_group)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.self_attention = IdentityOp()
    kwargs = dict(packed_seq_params=params)
    layer._decompose_packed_seq_params_to_kwargs(kwargs)
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)
    assert not getattr(kwargs["packed_seq_params"], "gdn_graph_safe", False)


@requires_fla_graph
@requires_te_graph_memory_slots
@pytest.mark.skipif(not HAVE_HYBRIDEP, reason="requires the HybridEP dispatcher")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8), reason="requires four or eight ranks"
)
@pytest.mark.parametrize("capture_mode", ["slots", "native", "native_shared_pool"])
@pytest.mark.parametrize("linear_cp_mode", ["chunkwise", "headwise"])
def test_partial_gdn_moe_mtp_dcp_replay_matches_eager(monkeypatch, capture_mode, linear_cp_mode):
    shared_slots = capture_mode == "slots"
    # Earlier tests in this process use up dynamo's recompile budget for the compiled GDN input
    # preparation. New specializations would then run eagerly, with bf16 rounding that differs
    # from the compiled reference, so compile every variant of this test afresh.
    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 64)
    with dcp_test_context(monkeypatch, expert_model_parallel_size=4) as groups:
        # Dynamic CP captures one bank per power-of-two size up to DPxCP (the world here).
        cp_sizes = tuple(2**exponent for exponent in range(groups.dp_cp.size().bit_length()))
        changes = (
            dict(
                linear_cp_mode="headwise",
                cp_partition_mode="zigzag",
                linear_num_key_heads=max(cp_sizes),
                linear_num_value_heads=16,
            )
            if linear_cp_mode == "headwise"
            else {}
        )
        config = _config(expert_model_parallel_size=4, mtp_num_layers=1, **changes)
        spec = get_transformer_block_with_experimental_attention_variant_spec(config)
        mtp_spec = get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=128,
            max_sequence_length=512 * max(cp_sizes),
            position_embedding_type="rope",
            mtp_block_spec=mtp_spec,
            pg_collection=groups,
        ).cuda()
        # Match get_model()'s DDP initialization: persistent AccumulateGrad nodes
        # must not bind to the legacy default stream before graph capture.
        init_stream = torch.cuda.Stream()
        init_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(init_stream):
            _grad_accs = [
                parameter.expand_as(parameter).grad_fn.next_functions[0][0]
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        torch.cuda.current_stream().wait_stream(init_stream)
        model.zero_grad_buffer = lambda: model.zero_grad(set_to_none=True)
        helper = None
        try:
            layouts = (
                ((384,), (128, 256), (130, 254), (2, 2, 2, 2, 2, 2, 372))
                if linear_cp_mode == "headwise"
                else ((384,), (128, 256))
            )
            cases = [(size, lengths) for size in cp_sizes for lengths in layouts]
            batches = [
                dcp_batch(config, groups, size, lengths, index)
                for index, (size, lengths) in enumerate(cases)
            ]
            references = []
            for index, batch in enumerate(batches):
                values, gradients = dcp_train_step(model, batch, index, cpu=True)
                assert torch.isfinite(values).all(), ("eager loss", index)
                for name, gradient in gradients.items():
                    assert torch.isfinite(gradient).all(), ("eager gradient", index, name)
                references.append((values, gradients))
                print(
                    f"GDN_PARTIAL_EAGER_REFERENCE rank={torch.distributed.get_rank()} "
                    f"case={index} local_valid_tokens={batch['loss_mask'].sum().item()}",
                    flush=True,
                )
            for index, batch in enumerate(batches):
                params = batch["packed_seq_params"]
                original_max = params.max_seqlen_q, params.max_seqlen_kv
                params.gdn_graph_safe = True
                params.max_seqlen_q = params.max_seqlen_kv = 512 * params.local_cp_size
                try:
                    values, gradients = dcp_train_step(model, batch, index, cpu=True)
                    torch.testing.assert_close(values, references[index][0], rtol=1e-3, atol=1e-3)
                    assert_gradients_close(gradients, references[index][1], index)
                finally:
                    del params.gdn_graph_safe
                    params.max_seqlen_q, params.max_seqlen_kv = original_max
            helper = TECudaGraphHelper(
                [model],
                config,
                seq_length=512 * max(cp_sizes),
                micro_batch_size=1,
                pg_collection=groups,
                dynamic_cp_group_getter=parallel_state.get_dynamic_data_context_parallel_groups,
            )
            assert len(helper.flattened_callables) == 5, "all four layers and MTP must be captured"
            assert (
                sum(
                    type(layer.self_attention).__name__ == "GatedDeltaNet"
                    for layer in helper.flattened_callables
                )
                == 3
            )
            capture_bank = helper._capture_graph_bank
            if not shared_slots:
                monkeypatch.setattr(helper, "_should_share_dynamic_cp_pool", lambda: False)
            native_pool = (
                torch.cuda.graph_pool_handle() if capture_mode == "native_shared_pool" else None
            )
            import transformer_engine.pytorch.graph as te_graph

            copy_span = te_graph._copy_tensor_storage_span
            compact_alpha_copies = []
            alpha_shapes = {
                (
                    (1, 512 * size, config.linear_num_value_heads // size)
                    if linear_cp_mode == "headwise"
                    else (1, 512, config.linear_num_value_heads)
                )
                for size in cp_sizes
            }

            def checked_copy_span(target, source, size_bytes):
                if source.dtype == torch.bfloat16 and tuple(source.shape) in alpha_shapes:
                    assert source.is_contiguous()
                    assert size_bytes == source.numel() * source.element_size()
                    compact_alpha_copies.append(size_bytes)
                return copy_span(target, source, size_bytes)

            monkeypatch.setattr(te_graph, "_copy_tensor_storage_span", checked_copy_span)

            def checked_capture_bank(callables, sample_args, kwargs):
                expected = 5 * len(cp_sizes) if shared_slots else 5
                assert len(callables) == len(kwargs["sample_kwargs"]) == expected
                for callable_, inputs in zip(callables, kwargs["sample_kwargs"]):
                    cp_size = (
                        callable_._capture_context[0]
                        if shared_slots
                        else config._cuda_graph_capture_dynamic_cp[0]
                    )
                    assert inputs["rotary_pos_emb"].shape[0] == 512 * cp_size
                if native_pool is not None:
                    assert helper.pp_group.size() == helper.num_microbatches == 1
                    kwargs = dict(kwargs, pool=native_pool)
                return capture_bank(callables, sample_args, kwargs)

            monkeypatch.setattr(helper, "_capture_graph_bank", checked_capture_bank)
            helper.create_cudagraphs()
            if shared_slots:
                assert compact_alpha_copies, "captured GDN alpha snapshots must use compact storage"
            else:
                assert not compact_alpha_copies, "native banks must not use slot saved-span copies"
            print(f"GDN_PARTIAL_CAPTURE_COMPLETE rank={torch.distributed.get_rank()}", flush=True)
            assert helper.graphs_created()
            for layer in helper.flattened_callables:
                assert set(layer.cuda_graphs_by_dynamic_cp_size) == set(cp_sizes)
            replays = track_graph_replays(monkeypatch)
            for microbatch, index in enumerate(
                (*range(len(cases)), *reversed(range(len(cases))), 0)
            ):
                before = len(replays)
                values, gradients = dcp_train_step(model, batches[index], microbatch, cpu=True)
                assert len(replays) - before >= 10, "every layer must replay forward AND backward"
                expected_values, expected_gradients = references[index]
                torch.testing.assert_close(values, expected_values, rtol=1e-3, atol=1e-3)
                assert_gradients_close(gradients, expected_gradients, index)
                print(
                    f"GDN_PARTIAL_REPLAY_PASS rank={torch.distributed.get_rank()} replay={microbatch} case={index}",
                    flush=True,
                )
            print(
                "GDN_PARTIAL_PASS layers=4 mtp=1 scope=attn,moe_router,moe_preprocess "
                f"cp={','.join(map(str, cp_sizes))} linear_cp_mode={linear_cp_mode}"
            )
        finally:
            torch.cuda.synchronize()
            if helper is not None:
                helper.delete_cuda_graphs()
