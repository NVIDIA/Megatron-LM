from __future__ import annotations

from types import SimpleNamespace

import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm import checkpoint
from megatron.lite.model.deepseek_v4.vllm.checkpoint import DeepseekV4WeightSpec
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization import deployment_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import (
    dequantize_mxfp4,
    quantize_mxfp4,
)


def test_pp_load_map_translates_local_layer_names_to_global_hf_layers() -> None:
    config = DeepseekV4Config(hidden_size=128, n_routed_experts=4)
    spec = DeepseekV4WeightSpec(config, source_block_fp8=False)
    model = SimpleNamespace(layer_indices=[2, 3])

    weight_map = spec.load_weight_map(
        model,
        ParallelState(pp_size=2, pp_rank=1),
        (
            "layers.0.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
            "layers.1.mlp.experts.w2.0",
        ),
    )

    assert weight_map == {
        "layers.2.input_layernorm.weight": ["layers.2.attn_norm.weight"],
        "layers.3.post_attention_layernorm.weight": ["layers.3.ffn_norm.weight"],
        "layers.3.mlp.experts.w2.0": ["layers.3.ffn.experts.0.w2.weight"],
    }


def test_fused_attention_loads_weight_scale_pairs_with_unequal_rows() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.self_attn.fused_wqa_wkv"
    assert spec._load_names(native) == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    ]
    target = torch.Size((384, 128))
    assert spec.hf_target_shape(native, 0, target) == torch.Size((256, 128))
    assert spec.hf_target_shape(native, 1, target) == torch.Size((2, 1))
    assert spec.hf_target_shape(native, 2, target) == torch.Size((128, 128))
    assert spec.hf_target_shape(native, 3, target) == torch.Size((1, 1))
    assert spec.read_hf_source_raw(native, 0, spec._load_names(native)[0])
    assert not spec.read_hf_source_raw(
        "layers.0.self_attn.q_norm", 0, "layers.0.attn.q_norm.weight"
    )

    q = torch.ones(256, 128, dtype=torch.float8_e4m3fn)
    kv = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
    master = spec.hf_to_native(
        native,
        [
            q,
            torch.full((2, 1), 2.0),
            kv,
            torch.full((1, 1), 4.0),
        ],
    )
    assert master.dtype == torch.bfloat16
    assert master.shape == target
    assert torch.all(master[:256] == 2)
    assert torch.all(master[256:] == 4)


def test_mixed_release_mxfp4_expert_loads_to_bf16_master() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        moe_intermediate_size=64,
        n_routed_experts=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.mlp.experts.w2.0"
    source = torch.linspace(-3, 3, 64 * 128, dtype=torch.bfloat16).reshape(64, 128)
    packed, scale = quantize_mxfp4(source)

    master = spec.hf_to_native(native, [packed, scale])

    assert master.dtype == torch.bfloat16
    torch.testing.assert_close(
        master,
        dequantize_mxfp4(packed, scale).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert native not in spec.source_block_scales


def test_mhc_checkpoint_parameters_preserve_fp32_release_values() -> None:
    spec = DeepseekV4WeightSpec(DeepseekV4Config(hidden_size=128))
    source = torch.tensor([0.6337993144989014], dtype=torch.float32)

    master = spec.hf_to_native("layers.0.attn_hc.hc_fn", [source])
    sink = spec.hf_to_native("layers.0.self_attn.attn_sink", [source])

    assert master.dtype == torch.float32
    torch.testing.assert_close(master, source, rtol=0, atol=0)
    assert sink.dtype == torch.float32
    torch.testing.assert_close(sink, source, rtol=0, atol=0)


def test_mhc_export_preserves_fp32_values() -> None:
    model = torch.nn.Module()
    model.hc_head = torch.nn.Module()
    source = torch.tensor([0.6337993144989014], dtype=torch.float32)
    model.hc_head.hc_fn = torch.nn.Parameter(source.bfloat16())

    exported = dict(
        checkpoint.export_hf_weights(
            model,
            DeepseekV4Config(hidden_size=128),
            ParallelState(),
        )
    )

    assert exported["hc_head_fn"].dtype == torch.float32
    torch.testing.assert_close(
        exported["hc_head_fn"], source.bfloat16().float(), rtol=0, atol=0
    )


def test_pipeline_stage_export_uses_global_layer_indices() -> None:
    class _SecondStage(torch.nn.Module):
        layer_indices = [2, 3]

        def state_dict(self, *args, **kwargs):
            assert kwargs.get("keep_vars") is True
            return {
                "layers.0.self_attn.compressor.ape": torch.ones(
                    4, dtype=torch.float32
                ),
                "layers.1.self_attn.indexer.compressor.ape": torch.full(
                    (4,), 2, dtype=torch.float32
                ),
            }

    exported = dict(
        checkpoint.export_hf_weights(
            _SecondStage(),
            DeepseekV4Config(hidden_size=128),
            ParallelState(),
        )
    )

    assert set(exported) == {
        "layers.2.attn.compressor.ape",
        "layers.3.attn.indexer.compressor.ape",
    }
    assert "layers.0.attn.compressor.ape" not in exported


def test_export_registry_preserves_scales_on_gathered_pipeline_tensor() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    native = "layers.2.self_attn.fused_wqa_wkv"
    source_scales = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32)
    spec = DeepseekV4WeightSpec(config)
    spec.export_source_block_scales[native] = source_scales

    # A PP broadcast produces a fresh tensor without arbitrary Parameter attrs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gathered = torch.ones(384, 128, dtype=torch.float32, device=device)
    exported = dict(spec.native_to_hf(native, gathered))

    torch.testing.assert_close(
        exported["layers.2.attn.wq_a.scale"],
        source_scales[:2].to(device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported["layers.2.attn.wkv.scale"],
        source_scales[2:].to(device),
        rtol=0,
        atol=0,
    )


def test_pipeline_export_reuses_common_streaming_collectives(monkeypatch) -> None:
    config = DeepseekV4Config(hidden_size=128, n_routed_experts=4)
    scales = {"layers.2.self_attn.wq_b": torch.ones(1, 1)}
    ps = SimpleNamespace(tp_size=1, etp_size=1, ep_size=1, pp_size=2)
    seen = []

    monkeypatch.setattr(
        checkpoint,
        "_pipeline_export_source_scales",
        lambda chunks, parallel_state, num_experts: scales,
    )

    from megatron.lite.primitive.ckpt import hf_weights

    def _fake_common(chunks, spec, parallel_state, **kwargs):
        seen.append((chunks, spec, parallel_state, kwargs))
        assert spec.export_source_block_scales is scales
        yield "sentinel", torch.ones(1)

    monkeypatch.setattr(hf_weights, "export_hf_weights", _fake_common)
    model = torch.nn.Module()

    exported = list(
        checkpoint.export_hf_weights(
            model,
            config,
            ps,
            target="block_fp8",
            resync_config={"expert_dtype": "fp8"},
            export_dtype="bfloat16",
        )
    )

    assert exported[0][0] == "sentinel"
    assert seen[0][0] is model
    assert seen[0][2] is ps
    assert seen[0][3]["vocab_size"] == config.vocab_size
    assert seen[0][3]["export_dtype"] == "bfloat16"
    assert "target" not in seen[0][3]
    assert "resync_config" not in seen[0][3]


def test_pipeline_gathered_expert_name_is_not_offset_twice() -> None:
    """Common PP export names already carry their global expert suffix."""
    config = DeepseekV4Config(hidden_size=128, n_routed_experts=256)
    spec = DeepseekV4WeightSpec(config)
    spec.ps = SimpleNamespace(ep_size=4, ep_rank=3)

    assert spec._expert_hf_names("layers.2.mlp.experts.w13.7") == [
        "layers.2.ffn.experts.199.w1.weight",
        "layers.2.ffn.experts.199.w3.weight",
    ]

    spec.export_expert_names_are_global = True
    assert spec._expert_hf_names("layers.2.mlp.experts.w13.71") == [
        "layers.2.ffn.experts.71.w1.weight",
        "layers.2.ffn.experts.71.w3.weight",
    ]


def test_pipeline_export_expert_name_contract_covers_global_ids() -> None:
    """The common EP exporter must not apply its ``weight<N>`` parser to DS4."""
    spec = DeepseekV4WeightSpec(
        DeepseekV4Config(hidden_size=128, n_routed_experts=256)
    )
    source = "layers.2.mlp.experts.w13.31"

    assert spec.export_expert_local_id(source) == 31
    assert spec.export_expert_name(source, 255) == (
        "layers.2.mlp.experts.w13.255"
    )


def test_common_ep4_gather_keeps_all_ds4_vllm_expert_shards(monkeypatch) -> None:
    """Regression: the legacy ``weight<N>`` parser collapsed all four shards."""
    from megatron.lite.primitive.ckpt import hf_weights

    spec = DeepseekV4WeightSpec(
        DeepseekV4Config(hidden_size=128, n_routed_experts=256)
    )
    ps = SimpleNamespace(
        ep_size=4,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
    )

    def _fake_ep_all_gather(outputs, tensor, group):
        assert group is ps.ep_group
        for ep_rank, output in enumerate(outputs):
            output.copy_(tensor + ep_rank)

    monkeypatch.setattr(hf_weights, "_ep_all_gather", _fake_ep_all_gather)
    gathered = {}
    hf_weights._gather_expert(
        "layers.2.mlp.experts.w2.31",
        torch.tensor([10.0]),
        spec,
        ps,
        gathered,
        cpu=False,
    )

    assert list(gathered) == [
        "layers.2.mlp.experts.w2.31",
        "layers.2.mlp.experts.w2.95",
        "layers.2.mlp.experts.w2.159",
        "layers.2.mlp.experts.w2.223",
    ]
    assert [value.item() for value in gathered.values()] == [10.0, 11.0, 12.0, 13.0]


def test_pipeline_source_scales_are_globalized_across_ep(monkeypatch) -> None:
    model = torch.nn.Module()
    model._fp8_source_scales_valid = True
    model._fp8_source_scales_by_name = {
        "layers.0.mlp.experts.w2.0": torch.tensor([[1.0]]),
        "layers.0.self_attn.wq_b": torch.tensor([[4.0]]),
    }
    ps = SimpleNamespace(
        ep_size=2,
        ep_group=object(),
        pp_size=1,
        pp_group=None,
    )

    monkeypatch.setattr(checkpoint.dist, "is_initialized", lambda: True)

    def _fake_all_gather(output, local, *, group):
        assert group is ps.ep_group
        output[:] = [
            local,
            {
                "layers.0.mlp.experts.w2.0": torch.tensor([[2.0]]),
                "layers.0.self_attn.wq_b": torch.tensor([[4.0]]),
            },
        ]

    monkeypatch.setattr(checkpoint.dist, "all_gather_object", _fake_all_gather)
    scales = checkpoint._pipeline_export_source_scales(model, ps, num_experts=4)

    assert set(scales) == {
        "layers.0.mlp.experts.w2.0",
        "layers.0.mlp.experts.w2.2",
        "layers.0.self_attn.wq_b",
    }
    torch.testing.assert_close(
        scales["layers.0.mlp.experts.w2.2"], torch.tensor([[2.0]]), rtol=0, atol=0
    )


def test_export_materializes_fsdp2_dtensor_before_conversion() -> None:
    source = torch.tensor([0.6337993144989014], dtype=torch.bfloat16)
    calls = []

    class _DTensorProxy:
        def full_tensor(self):
            calls.append("full_tensor")
            return source

    class _Model(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            return {"hc_head.hc_fn": _DTensorProxy()}

    exported = dict(
        checkpoint.export_hf_weights(
            _Model(),
            DeepseekV4Config(hidden_size=128),
            ParallelState(),
        )
    )
    assert calls == ["full_tensor"]
    torch.testing.assert_close(exported["hc_head_fn"], source.float(), rtol=0, atol=0)


def test_bf16_master_checkpoint_load_skips_fp8_scale_pairs() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config, source_block_fp8=False)
    native = "layers.0.self_attn.fused_wqa_wkv"
    names = spec._load_names(native)
    assert names == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wkv.weight",
    ]
    target = torch.Size((384, 128))
    assert spec.hf_target_shape(native, 0, target) == torch.Size((256, 128))
    assert spec.hf_target_shape(native, 1, target) == torch.Size((128, 128))
    assert not spec.read_hf_source_raw(native, 0, names[0])

    master = spec.hf_to_native(
        native,
        [
            torch.ones(256, 128, dtype=torch.bfloat16),
            torch.full((128, 128), 2, dtype=torch.bfloat16),
        ],
    )
    assert master.dtype == torch.bfloat16
    assert master.shape == target
    assert torch.all(master[:256] == 1)
    assert torch.all(master[256:] == 2)


def test_layer2_compressor_and_indexer_release_names_and_dtypes() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=128,
        head_dim=128,
        index_head_dim=128,
        index_n_heads=2,
        num_attention_heads=2,
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
        num_hash_layers=3,
    )
    spec = DeepseekV4WeightSpec(config)
    assert spec._load_names(
        "layers.2.self_attn.compressor.fused_wkv_wgate"
    ) == [
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
    ]
    assert spec._load_names("layers.2.self_attn.indexer.weights_proj") == [
        "layers.2.attn.indexer.weights_proj.weight"
    ]
    assert spec._load_names("layers.2.self_attn.indexer.wq_b") == [
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.scale",
    ]
    assert spec._load_names(
        "layers.2.self_attn.indexer.compressor.fused_wkv_wgate"
    ) == [
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.wgate.weight",
    ]
    assert spec._load_names("layers.3.self_attn.compressor.ape") == [
        "layers.3.attn.compressor.ape"
    ]

    compressor = spec.hf_to_native(
        "layers.2.self_attn.compressor.fused_wkv_wgate",
        [
            torch.ones(256, 128, dtype=torch.bfloat16),
            torch.full((256, 128), 2, dtype=torch.bfloat16),
        ],
    )
    assert compressor.dtype == torch.bfloat16
    assert compressor.shape == (512, 128)

    indexer = spec.hf_to_native(
        "layers.2.self_attn.indexer.wq_b",
        [
            torch.ones(256, 128, dtype=torch.float8_e4m3fn),
            torch.full((2, 1), 3.0),
        ],
    )
    assert indexer.dtype == torch.bfloat16
    assert indexer.shape == (256, 128)
    assert torch.all(indexer == 3)


def test_fp8_loads_are_replica_local_to_preserve_source_scales() -> None:
    dense_group = object()
    ps = ParallelState(dp_cp_group=dense_group, ep_dp_group=object())

    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.mlp.experts.w13.0", ps
        )
        is None
    )
    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.self_attn.wq_b", ps
        )
        is None
    )
    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.input_layernorm.weight", ps
        )
        is dense_group
    )
    assert (
        DeepseekV4WeightSpec.expert_local_name(
            "layers.0.mlp.experts.w13.128", 0
        )
        == "layers.0.mlp.experts.w13.0"
    )


def test_fused_fp8_export_preserves_bound_source_scales() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config)
    master = torch.nn.Parameter(torch.ones(384, 128, dtype=torch.bfloat16))
    master._fp8_source_scales = torch.tensor(
        [[0.5], [1.0], [2.0]], dtype=torch.float32
    )
    master._fp8_source_scale_version = master._version

    exported = dict(spec.native_to_hf("layers.0.self_attn.fused_wqa_wkv", master))

    torch.testing.assert_close(
        exported["layers.0.attn.wq_a.scale"],
        master._fp8_source_scales[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported["layers.0.attn.wkv.scale"],
        master._fp8_source_scales[2:],
        rtol=0,
        atol=0,
    )


def test_full_export_preserves_bound_source_scales_across_detach() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    master = torch.nn.Parameter(torch.ones(384, 128, dtype=torch.bfloat16))
    master._fp8_source_scales = torch.tensor(
        [[0.5], [1.0], [2.0]], dtype=torch.float32
    )
    master._fp8_source_scale_version = master._version

    class _Model(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            assert kwargs.get("keep_vars") is True
            return {"layers.0.self_attn.fused_wqa_wkv": master}

    exported = dict(checkpoint.export_hf_weights(_Model(), config, ParallelState()))

    torch.testing.assert_close(
        exported["layers.0.attn.wq_a.scale"],
        master._fp8_source_scales[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported["layers.0.attn.wkv.scale"],
        master._fp8_source_scales[2:],
        rtol=0,
        atol=0,
    )


def test_full_export_prefers_metadata_on_live_parameter_over_state_value() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    master = torch.nn.Parameter(torch.ones(384, 128, dtype=torch.bfloat16))
    master._fp8_source_scales = torch.tensor(
        [[0.5], [1.0], [2.0]], dtype=torch.float32
    )
    master._fp8_source_scale_version = master._version

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("master", master)

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            yield "layers.0.self_attn.fused_wqa_wkv", self.master

        def state_dict(self, *args, **kwargs):
            assert kwargs.get("keep_vars") is True
            # Mirrors an FSDP state-dict hook returning a fresh value without
            # arbitrary attributes attached to the live Parameter.
            return {
                "layers.0.self_attn.fused_wqa_wkv": self.master.detach().clone()
            }

    exported = dict(checkpoint.export_hf_weights(_Model(), config, ParallelState()))

    torch.testing.assert_close(
        exported["layers.0.attn.wq_a.scale"],
        master._fp8_source_scales[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported["layers.0.attn.wkv.scale"],
        master._fp8_source_scales[2:],
        rtol=0,
        atol=0,
    )


def test_model_scale_registry_survives_parameter_metadata_loss_then_invalidates() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    native = "layers.0.self_attn.fused_wqa_wkv"
    source_scales = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.master = torch.nn.Parameter(
                torch.ones(384, 128, dtype=torch.bfloat16)
            )

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            yield native, self.master

        def state_dict(self, *args, **kwargs):
            assert kwargs.get("keep_vars") is True
            return {native: self.master}

    model = _Model()
    model._fp8_source_scales_by_name = {native: source_scales}
    model._fp8_source_scales_valid = True

    exported = dict(checkpoint.export_hf_weights(model, config, ParallelState()))
    torch.testing.assert_close(
        exported["layers.0.attn.wq_a.scale"], source_scales[:2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        exported["layers.0.attn.wkv.scale"], source_scales[2:], rtol=0, atol=0
    )

    checkpoint.invalidate_bound_source_scales(model)
    assert model._fp8_source_scales_valid is False
    assert model._fp8_source_scales_by_name == {}


def test_router_checkpoint_names_follow_hash_prefix_semantics() -> None:
    config = DeepseekV4Config(
        num_hidden_layers=4,
        num_hash_layers=3,
        n_routed_experts=256,
    )
    spec = DeepseekV4WeightSpec(config)

    assert spec._load_names("layers.2.mlp.gate.tid2eid") == [
        "layers.2.ffn.gate.tid2eid"
    ]
    assert spec._load_names("layers.3.mlp.gate.expert_bias") == [
        "layers.3.ffn.gate.bias"
    ]
    assert spec._load_names("layers.3.mlp.gate.gate.weight") == [
        "layers.3.ffn.gate.weight"
    ]
    assert spec.hf_to_native(
        "layers.2.mlp.gate.tid2eid",
        [torch.zeros(8, 6, dtype=torch.int64)],
    ).dtype == torch.int32
    assert spec.hf_to_native(
        "layers.3.mlp.gate.expert_bias",
        [torch.zeros(256, dtype=torch.float32)],
    ).dtype == torch.float32


def test_export_and_forward_share_canonical_bf16_to_fp8_quantizer(
    monkeypatch,
) -> None:
    assert (
        checkpoint.quantize_block_fp8_weight
        is deployment_block_fp8.quantize_block_fp8_weight
    )
    calls: list[torch.Tensor] = []

    def cast(value, block_size, use_ue8m0):
        assert block_size == [128, 128]
        assert use_ue8m0 is False
        calls.append(value)
        return (
            value.float().clamp(-1, 1).to(torch.float8_e4m3fn),
            torch.ones(
                value.shape[0] // 128,
                value.shape[1] // 128,
                dtype=torch.float32,
            ),
        )

    def post_process(**kwargs):
        return kwargs["wq"], kwargs["ws"]

    entries = {
        ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"): cast,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        ): post_process,
    }
    monkeypatch.setattr(
        deployment_block_fp8,
        "_import_attr",
        lambda module, name: entries[(module, name)],
    )

    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.self_attn.fused_wqa_wkv"
    master = torch.randn(384, 128, dtype=torch.bfloat16)
    exported = spec.native_to_hf(native, master)
    assert [name for name, _ in exported] == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    ]
    assert exported[0][1].dtype == torch.float8_e4m3fn
    assert exported[1][1].dtype == torch.float32

    deployment_block_fp8.pack_block_fp8_weight(
        torch.nn.Parameter(master[:256].clone())
    )
    assert len(calls) == 3
    assert torch.equal(calls[0], master[:256])
    assert torch.equal(calls[2], master[:256])
