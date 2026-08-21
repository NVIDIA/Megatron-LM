from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm import runtime_metadata as runtime


def _config(**overrides) -> DeepseekV4Config:
    values = dict(
        head_dim=512,
        qk_rope_head_dim=64,
        num_attention_heads=64,
        sliding_window=128,
        compress_ratios=[0, 0, 4, 128],
        max_position_embeddings=256,
    )
    values.update(overrides)
    return DeepseekV4Config(**values)


def test_batch_invariance_initialization_is_mandatory(monkeypatch) -> None:
    init = Mock()

    monkeypatch.setattr(runtime, "_symbol", lambda _module, _name: init)
    runtime.initialize_ds4_vllm_batch_invariance()

    init.assert_called_once_with(force=True)


def test_vllm_forward_context_gathers_tokens_on_ep_group(monkeypatch) -> None:
    captured = {}
    ep_group = object()

    def all_gather(outputs, local, *, group):
        assert group is ep_group
        assert local.tolist() == [5]
        outputs[0].copy_(local)
        outputs[1].fill_(7)

    class Override:
        def __init__(self, context):
            self.context = context

        def __enter__(self):
            captured["active"] = self.context

        def __exit__(self, *_args):
            captured["active"] = None

    def symbol(_module, name):
        if name == "DPMetadata":
            return lambda counts: SimpleNamespace(
                num_tokens_across_dp_cpu=counts
            )
        if name == "create_forward_context":
            return lambda _attn, config, **kwargs: SimpleNamespace(
                config=config, **kwargs
            )
        if name == "override_forward_context":
            return Override
        raise AssertionError(name)

    monkeypatch.setattr(runtime.dist, "all_gather", all_gather)
    monkeypatch.setattr(runtime, "_symbol", symbol)
    batch = SimpleNamespace(
        input_ids=torch.arange(3),
        total_tokens=5,
    )
    parallel_state = SimpleNamespace(ep_group=ep_group, ep_size=2)
    config = object()

    with runtime.ds4_vllm_forward_context(
        batch,
        parallel_state,
        vllm_config=config,
    ):
        context = captured["active"]
        assert context.config is config
        assert context.dp_metadata.num_tokens_across_dp_cpu.tolist() == [5, 7]
    assert captured["active"] is None


def test_layer0_rope_uses_official_vllm_config_loader(monkeypatch) -> None:
    hf_config = SimpleNamespace()
    cos = torch.arange(32 * 64, dtype=torch.bfloat16).reshape(32, 64)
    rotary = SimpleNamespace(cos_sin_cache=cos)
    get_config = Mock(return_value=hf_config)
    build = Mock(return_value=rotary)

    def symbol(module, name):
        assert (module, name) == ("vllm.transformers_utils.config", "get_config")
        return get_config

    monkeypatch.setattr(runtime, "_symbol", symbol)
    monkeypatch.setattr(runtime, "_build_rope", build)
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter.from_hf(
        "/model", _config(), device="cpu"
    )

    get_config.assert_called_once_with("/model", trust_remote_code=True)
    build.assert_called_once()
    assert metadata.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(metadata.cos_sin_cache, cos.float(), rtol=0, atol=0)


def test_rope_custom_op_is_built_in_scoped_vllm_config(monkeypatch) -> None:
    events = []
    rotary = SimpleNamespace(to=lambda **_kwargs: rotary)

    class Context:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    def symbol(_module, name):
        if name == "VllmConfig":
            return lambda: "config"
        if name == "set_current_vllm_config":
            return lambda value: Context() if value == "config" else None
        if name == "build_deepseek_v4_rope":
            return lambda *_args, **_kwargs: events.append("build") or rotary
        raise AssertionError(name)

    monkeypatch.setattr(runtime, "_symbol", symbol)
    actual = runtime._build_rope(
        SimpleNamespace(), _config(), compress_ratio=1, device="cpu"
    )

    assert actual is rotary
    assert events == ["enter", "build", "exit"]


def test_native_cp_compressor_reuses_single_request_batch_metadata() -> None:
    config = _config(num_hidden_layers=4)
    positions = torch.arange(37, dtype=torch.int64)
    packed = object()

    metadata = runtime.build_native_cp_attention_metadata(
        config,
        layer_idx=2,
        cos_sin_cache=torch.zeros(256, 64, dtype=torch.float32),
        local_positions=positions,
        packed_seq_params=packed,
    )

    compressor = metadata.cp_compressor_metadata
    assert compressor is not None
    assert metadata.cp_packed_seq_params is packed
    assert metadata.cp_positions is positions
    assert compressor.state_block_table.shape[0] == 1
    assert compressor.token_to_req_indices.shape == (64,)
    assert compressor.token_to_req_indices.count_nonzero().item() == 0


def test_layer0_prefill_metadata_exact_contract(monkeypatch) -> None:
    gather = Mock()
    monkeypatch.setattr(runtime, "_symbol", lambda _module, _name: gather)
    cos = torch.empty(256, 128, dtype=torch.float32)
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(), device="cpu", cos_sin_cache=cos
    ).build_prefill_batch([130])

    assert torch.equal(metadata.positions, torch.arange(130, dtype=torch.int64))
    assert torch.equal(metadata.slot_mapping, metadata.positions)
    assert metadata.swa_cache.shape == (1, 256, 584)
    assert metadata.swa_cache.dtype == torch.uint8
    assert metadata.indices.shape == (130, 1, 128)
    assert metadata.indices.dtype == torch.int32
    assert metadata.topk_length.dtype == torch.int32
    assert metadata.topk_length.tolist() == list(range(1, 129)) + [128, 128]
    assert metadata.indices[0, 0, 0].item() == 0
    assert metadata.indices[128, 0, 0].item() == 1
    assert metadata.indices[129, 0, -1].item() == 129
    assert metadata.kv_workspace.shape == (130, 1, 512)
    assert metadata.output.shape == (130, 64, 512)
    assert metadata.runtime_layout.block_table.tolist() == [[0]]

    metadata.prepare_flash()
    gather.assert_called_once()
    args = gather.call_args
    assert args.args[0].shape == (1, 130, 512)
    assert args.args[1].shape == (1, 256, 584)
    assert args.kwargs["block_size"] == 256
    assert args.kwargs["offset"] == 0
    assert torch.equal(args.kwargs["seq_lens"], torch.tensor([130], dtype=torch.int32))
    assert torch.equal(args.kwargs["gather_lens"], args.kwargs["seq_lens"])


def test_layer0_packed_prefill_metadata_is_sequence_isolated(monkeypatch) -> None:
    gather = Mock()
    monkeypatch.setattr(runtime, "_symbol", lambda _module, _name: gather)
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(),
        device="cpu",
        cos_sin_cache=torch.empty(256, 128, dtype=torch.float32),
    ).build_prefill_batch([3, 2])

    assert metadata.positions.tolist() == [0, 1, 2, 0, 1]
    assert metadata.slot_mapping.tolist() == [0, 1, 2, 256, 257]
    assert metadata.runtime_layout.block_table.tolist() == [[0], [1]]
    assert metadata.runtime_layout.seq_lens.tolist() == [3, 2]
    assert metadata.runtime_layout.query_start_loc.tolist() == [0, 3, 5]
    assert metadata.indices[:3, 0, 0].tolist() == [0, 0, 0]
    assert metadata.indices[3:, 0, 0].tolist() == [3, 3]
    assert metadata.topk_length.tolist() == [1, 2, 3, 1, 2]
    assert metadata.kv_workspace.shape == (6, 1, 512)
    assert metadata.output.shape == (5, 64, 512)

    metadata.prepare_flash()
    args = gather.call_args
    assert args.args[0].shape == (2, 3, 512)
    assert args.args[1].shape == (2, 256, 584)


def test_unified_builder_uses_selected_layer_ratio() -> None:
    builder = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(num_hidden_layers=2, compress_ratios=[0, 4]),
        layer_idx=1,
        device="cpu",
        cos_sin_cache=torch.empty(16, 128),
    )
    assert builder.compress_ratio == 4


def test_moe_metadata_builder_uses_tp_independent_runtime_gate() -> None:
    metadata = runtime.build_moe_metadata(
        _config(num_hidden_layers=4, num_hash_layers=3), "cpu"
    )
    assert isinstance(metadata.gate_linear, runtime._RuntimeGateLinear)
    assert metadata.gate_linear.weight.shape == (256, 4096)
    assert not hasattr(metadata, "build_grouped_moe")


def test_layer1_extended_builder_reuses_swa_only_contract(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_symbol", lambda _module, _name: Mock())
    builder = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(),
        layer_idx=1,
        device="cpu",
        cos_sin_cache=torch.empty(256, 128),
    )
    metadata = builder.build_prefill_batch([5])
    assert metadata.compressor_operation is None
    assert metadata.indexer_operation is None
    assert metadata.indices.shape == (5, 1, 128)


def test_extended_builder_covers_every_full_model_layer() -> None:
    ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    config = _config(
        num_hidden_layers=len(ratios),
        compress_ratios=ratios,
        max_position_embeddings=512,
    )
    cos_sin_cache = torch.empty(512, 128, dtype=torch.float32)

    builders = [
        runtime.DS4SparseIndexerCompressorMetadataAdapter(
            config,
            layer_idx=layer_idx,
            device="cpu",
            cos_sin_cache=cos_sin_cache,
        )
        for layer_idx in range(config.num_hidden_layers)
    ]

    assert [builder.layer_idx for builder in builders] == list(
        range(config.num_hidden_layers)
    )
    assert {builder.compress_ratio for builder in builders} == {1, 4, 128}


@pytest.mark.parametrize("layer_idx", [-1, 44])
def test_unified_builder_rejects_layers_outside_model(layer_idx) -> None:
    ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    with pytest.raises(ValueError, match="outside the model"):
        runtime.DS4SparseIndexerCompressorMetadataAdapter(
            _config(num_hidden_layers=len(ratios), compress_ratios=ratios),
            layer_idx=layer_idx,
            device="cpu",
            cos_sin_cache=torch.empty(256, 128, dtype=torch.float32),
        )


def test_layer2_packed_prefill_metadata_is_sequence_isolated(monkeypatch) -> None:
    def symbol(_module, name):
        if name == "DeepseekV32IndexerMetadata":
            return lambda **kwargs: SimpleNamespace(**kwargs)
        if name == "DeepseekV32IndexerPrefillMetadata":
            return lambda chunks: SimpleNamespace(chunks=chunks)
        if name == "build_prefill_chunk_metadata":
            return lambda *_args: None
        raise AssertionError(name)

    monkeypatch.setattr(runtime, "_symbol", symbol)
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(),
        layer_idx=2,
        device="cpu",
        cos_sin_cache=torch.empty(256, 128),
    ).build_prefill_batch([5, 9])

    main = metadata.compressor_metadata
    indexer = metadata.indexer_metadata.compressor
    assert metadata.positions.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert metadata.slot_mapping.tolist() == [0, 1, 2, 3, 4, 256, 257, 258, 259, 260, 261, 262, 263, 264]
    assert main.token_to_req_indices.tolist() == [0] * 5 + [1] * 9
    assert main.state_block_table.tolist() == [[0, 1, -1], [2, 3, 4]]
    assert main.state_slot_mapping.tolist() == list(range(5)) + list(range(8, 17))
    assert main.k_slot_mapping.tolist() == [-1, -1, -1, 0, -1] + [-1, -1, -1, 64, -1, -1, -1, 65, -1]
    assert indexer.k_slot_mapping.tolist() == main.k_slot_mapping.tolist()
    assert metadata.indexer_metadata.attention_metadata.seq_lens.tolist() == [5, 9]
    assert metadata.indexer_metadata.attention_metadata.max_seq_len == 9
    # The official SparseAttnIndexer workspace covers every compressed K row
    # in the packed chunk: floor(5 / 4) + floor(9 / 4), not max(1, 2).
    assert metadata.indexer_metadata.max_total_seq_len == 3
    assert metadata.kv_workspace.shape == (22, 1, 512)


def test_prepare_flash_rebuilds_derived_layout_for_activation_recompute(
    monkeypatch,
) -> None:
    combine_inputs = []
    gather = Mock()

    def combine(topk, *_args, out):
        combine_inputs.append(topk)
        indices, lengths = out
        indices.fill_(len(combine_inputs))
        lengths.fill_(topk.shape[-1])
        return indices, lengths

    def symbol(_module, name):
        if name == "DeepseekV32IndexerMetadata":
            return lambda **kwargs: SimpleNamespace(**kwargs)
        if name == "DeepseekV32IndexerPrefillMetadata":
            return lambda chunks: SimpleNamespace(chunks=chunks)
        if name == "build_prefill_chunk_metadata":
            return lambda *_args: None
        if name == "dequantize_and_gather_k_cache":
            return gather
        if name == "combine_topk_swa_indices":
            return combine
        raise AssertionError(name)

    monkeypatch.setattr(runtime, "_symbol", symbol)
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(),
        layer_idx=2,
        device="cpu",
        cos_sin_cache=torch.empty(256, 128),
    ).build_prefill_batch([5, 9])
    source_topk = metadata.indices[:, 0]

    metadata.prepare_flash()
    assert metadata.indices.ndim == 3
    assert metadata.indices[0, 0, 0].item() == 1
    metadata.prepare_flash()
    assert metadata.indices.ndim == 3
    assert metadata.indices[0, 0, 0].item() == 2

    assert gather.call_count == 4
    assert len(combine_inputs) == 2
    assert combine_inputs[0] is combine_inputs[1]
    assert combine_inputs[0].untyped_storage().data_ptr() == source_topk.untyped_storage().data_ptr()
    assert combine_inputs[0].shape == source_topk.shape
    assert combine_inputs[0].ndim == 2


@pytest.mark.gpus(1)
def test_layer3_packed_prefill_metadata_is_sequence_isolated(monkeypatch) -> None:
    metadata = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        _config(max_position_embeddings=512),
        layer_idx=3,
        device="cuda",
        cos_sin_cache=torch.empty(512, 128, device="cuda"),
    ).build_prefill_batch([129, 257])

    main = metadata.compressor_metadata
    assert main.token_to_req_indices.tolist() == [0] * 129 + [1] * 257
    assert main.state_block_table.shape == (2, 33)
    assert main.state_block_table[0, -1].item() == -1
    assert main.state_block_table[1, 0].item() == 17
    assert main.k_slot_mapping[127].item() == 0
    assert main.k_slot_mapping[128].item() == -1
    assert main.k_slot_mapping[129 + 127].item() == 2
    assert main.k_slot_mapping[129 + 255].item() == 3
    # Two request-major rows, each padded to max(compressed)=2 plus the full
    # max prefill source=257.  Keeping only a 128-token tail makes early SWA
    # indices negative in combine_topk_swa_indices.
    assert metadata.kv_workspace.shape == (518, 1, 512)


def _compressor_metadata(device: str = "cpu", tokens: int = 2):
    return runtime.DS4CompressorRuntimeMetadata(
        state_cache=torch.zeros(1, 4, 512, dtype=torch.float32, device=device),
        state_slot_mapping=torch.arange(tokens, dtype=torch.int64, device=device),
        state_block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        state_block_size=4,
        token_to_req_indices=torch.zeros(tokens, dtype=torch.int32, device=device),
        k_cache=torch.zeros(1, 64, 132, dtype=torch.uint8, device=device),
        k_slot_mapping=torch.full((tokens,), -1, dtype=torch.int64, device=device),
        cos_sin_cache=torch.zeros(16, 128, dtype=torch.float32, device=device),
        rms_norm_eps=1e-6,
        rope_head_dim=64,
    )


def test_extended_compressor_cpu_contract_calls_official_operations(
    monkeypatch,
) -> None:
    save = Mock()
    compress = Mock()

    def lookup(module, name):
        return {
            "save_partial_states": save,
            "compress_norm_rope_store_triton": compress,
        }[name]

    monkeypatch.setattr(runtime, "_symbol", lookup)
    metadata = _compressor_metadata()
    kv_score = torch.zeros(2, 512, dtype=torch.float32)
    runtime.DS4SparseIndexerCompressorMetadataAdapter.compressor_operation(
        kv_score=kv_score,
        positions=torch.tensor([0, 1], dtype=torch.int64),
        ape=torch.zeros(4, 256, dtype=torch.float32),
        norm_weight=torch.ones(128, dtype=torch.bfloat16),
        compress_ratio=4,
        head_dim=128,
        metadata=metadata,
    )

    save.assert_called_once()
    assert save.call_args.kwargs["state_cache"] is metadata.state_cache
    assert save.call_args.kwargs["state_width"] == 256
    compress.assert_called_once()
    assert compress.call_args.kwargs["kv_cache"] is metadata.k_cache
    assert compress.call_args.kwargs["head_dim"] == 128
    assert compress.call_args.kwargs["token_stride"] == 128


def test_extended_indexer_cpu_contract_calls_official_short_metadata_kernel(
    monkeypatch,
) -> None:
    launch = Mock()

    class Kernel:
        def __getitem__(self, _grid):
            return launch

    monkeypatch.setattr(runtime, "_symbol", lambda _module, _name: Kernel())
    compressor = _compressor_metadata()
    topk = torch.full((2, 4), -1, dtype=torch.int32)
    metadata = runtime.DS4IndexerRuntimeMetadata(
        compressor=compressor,
        attention_metadata=SimpleNamespace(max_seq_len=8),
        k_cache_prefix="indexer.k_cache",
        topk_indices=topk,
        max_model_len=64,
        max_total_seq_len=2,
    )
    result = runtime.DS4SparseIndexerCompressorMetadataAdapter.indexer_operation(
        qr=torch.zeros(2, 8, dtype=torch.bfloat16),
        index_q=torch.zeros(2, 2, 128, dtype=torch.bfloat16),
        index_weights=torch.zeros(2, 2, dtype=torch.bfloat16),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        compress_ratio=4,
        topk=4,
        metadata=metadata,
    )

    assert result is topk
    launch.assert_called_once()
    assert launch.call_args.args[0] is topk
    assert torch.equal(launch.call_args.args[1], metadata.compressor.state_slot_mapping)
    assert launch.call_args.kwargs["COMPRESS_RATIO"] == 4


def _production_ready() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and importlib.util.find_spec("triton") is not None
    )


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA and the matching vLLM DS4 Triton primitives",
)
def test_prefill_indices_are_bitwise_official_vllm() -> None:
    from vllm.models.deepseek_v4.common.ops import combine_topk_swa_indices

    num_tokens = 193
    config = _config(max_position_embeddings=512)
    builder = runtime.DS4SparseIndexerCompressorMetadataAdapter(
        config,
        device="cuda",
        cos_sin_cache=torch.empty(512, 128, dtype=torch.float32, device="cuda"),
    )
    candidate = builder.build_prefill_batch([num_tokens])
    reference_indices, reference_lens = combine_topk_swa_indices(
        torch.empty(num_tokens, config.index_topk, dtype=torch.int32, device="cuda"),
        torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
        torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
        torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
        config.sliding_window,
        1,
        0,
        num_tokens,
        0,
    )
    torch.testing.assert_close(
        candidate.indices.squeeze(1), reference_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(candidate.topk_length, reference_lens, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA and the matching vLLM DS4 Triton primitives",
)
def test_extended_short_indexer_is_bitwise_official_vllm() -> None:
    from vllm.models.deepseek_v4.attention import (
        _fill_short_context_topk_indices,
    )

    positions = torch.arange(8, dtype=torch.int64, device="cuda")
    expected = torch.full((8, 8), -1, dtype=torch.int32, device="cuda")
    actual = expected.clone()
    _fill_short_context_topk_indices[(8,)](
        expected,
        positions,
        TOP_K=8,
        COMPRESS_RATIO=4,
        PADDED_TOP_K=8,
        num_warps=8,
    )
    compressor = _compressor_metadata("cuda")
    metadata = runtime.DS4IndexerRuntimeMetadata(
        compressor=compressor,
        attention_metadata=SimpleNamespace(max_seq_len=8),
        k_cache_prefix="indexer.k_cache",
        topk_indices=actual,
        max_model_len=64,
        max_total_seq_len=2,
    )
    result = runtime.DS4SparseIndexerCompressorMetadataAdapter.indexer_operation(
        qr=torch.zeros(8, 8, dtype=torch.bfloat16, device="cuda"),
        index_q=torch.zeros(8, 2, 128, dtype=torch.bfloat16, device="cuda"),
        index_weights=torch.zeros(8, 2, dtype=torch.bfloat16, device="cuda"),
        positions=positions,
        compress_ratio=4,
        topk=8,
        metadata=metadata,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA and the matching vLLM DS4 Triton primitives",
)
def test_extended_compressor_is_bitwise_official_vllm() -> None:
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )
    from vllm.models.deepseek_v4.common.ops.save_partial_states import (
        save_partial_states,
    )

    torch.manual_seed(31)
    tokens = 4
    positions = torch.arange(tokens, dtype=torch.int64, device="cuda")
    kv_score = torch.randn(tokens, 512, dtype=torch.float32, device="cuda")
    ape = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    norm = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    expected = _compressor_metadata("cuda", tokens)
    actual = _compressor_metadata("cuda", tokens)
    expected.k_slot_mapping.copy_(
        torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device="cuda")
    )
    actual.k_slot_mapping.copy_(expected.k_slot_mapping)

    kv, score = kv_score.split([256, 256], dim=-1)
    save_partial_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        state_cache=expected.state_cache,
        slot_mapping=expected.state_slot_mapping,
        block_size=4,
        state_width=256,
        compress_ratio=4,
        pdl_kwargs={"launch_pdl": False},
    )
    compress_norm_rope_store_triton(
        state_cache=expected.state_cache,
        num_actual=tokens,
        token_to_req_indices=expected.token_to_req_indices,
        positions=positions,
        slot_mapping=expected.state_slot_mapping,
        block_table=expected.state_block_table,
        block_size=4,
        state_width=256,
        cos_sin_cache=expected.cos_sin_cache,
        kv_cache=expected.k_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=expected.k_slot_mapping),
        pdl_kwargs={"launch_pdl": False},
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=norm,
        rms_norm_eps=1e-6,
        quant_block=128,
        token_stride=128,
        scale_dim=4,
    )
    runtime.DS4SparseIndexerCompressorMetadataAdapter.compressor_operation(
        kv_score=kv_score,
        positions=positions,
        ape=ape,
        norm_weight=norm,
        compress_ratio=4,
        head_dim=128,
        metadata=actual,
    )
    torch.testing.assert_close(actual.state_cache, expected.state_cache, rtol=0, atol=0)
    assert torch.equal(actual.k_cache, expected.k_cache)
