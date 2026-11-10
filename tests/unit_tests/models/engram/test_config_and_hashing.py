# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.config import (
    EngramConfig,
    _validate_packed_sequence_args,
    allocate_table_sizes,
)
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.hashing import (
    build_ngram_hashes,
    compress_token_ids,
    slice_hashes_for_sequence_parallel,
)
from megatron.core.models.engram.layer_specs import apply_engram_to_hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from megatron.core.transformer.wide_residual_config import WideResidualConfig

from ._test_utils import write_tokenizer_map


class _FakeTokenizer:
    def __init__(self, decoded_tokens, raw_tokens):
        self.decoded_tokens = decoded_tokens
        self.raw_tokens = raw_tokens

    def __len__(self):
        return len(self.decoded_tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        assert not skip_special_tokens
        assert len(token_ids) == 1
        return self.decoded_tokens[token_ids[0]]

    def convert_ids_to_tokens(self, token_id):
        return self.raw_tokens[token_id]


def test_official_tokenizer_compression_uses_first_normalized_occurrence():
    # The tokenizer-map tool imports tokenizers and transformers at module level.
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tools.engram.generate_tokenizer_map import build_remap

    tokenizer = _FakeTokenizer(
        decoded_tokens=["É", "é", " \t\n", " ", "", "�"],
        raw_tokens=["tok0", "tok1", "tok2", "tok3", "tok4", "raw-invalid"],
    )

    remap, compressed_vocab_size = build_remap(tokenizer)

    assert remap == [0, 0, 1, 1, 2, 3]
    assert compressed_vocab_size == 4


def test_official_fixed_hash_vector():
    input_ids = torch.tensor([[1, 4, 2, 7]], dtype=torch.int64)
    hashes = build_ngram_hashes(
        input_ids=input_ids,
        tokenizer_remap=torch.arange(16, dtype=torch.int64),
        multipliers=torch.tensor([13, 17, 19], dtype=torch.int64),
        table_sizes=torch.tensor([11, 13, 17, 19], dtype=torch.int64),
        max_ngram_order=3,
        num_hash_heads=2,
        boundary_token_id=3,
    )
    expected = torch.tensor([[[7, 10, 7, 7], [4, 11, 11, 9], [6, 3, 9, 1], [0, 4, 2, 15]]])
    torch.testing.assert_close(hashes, expected)


def test_compression_preserves_negative_sentinels():
    remap = torch.tensor([0, 4, 1, 3, 2], dtype=torch.int64)
    input_ids = torch.tensor([[1, -100, 4, 2]], dtype=torch.int64)
    torch.testing.assert_close(
        compress_token_ids(input_ids, remap), torch.tensor([[4, -100, 2, 1]])
    )


def test_prime_allocation_is_global_and_distinct():
    sizes = allocate_table_sizes((10, 10), (1, 5), 2)
    assert sizes == {1: (11, 13, 17, 19), 5: (23, 29, 31, 37)}
    assert len(set(sizes[1] + sizes[5])) == 8


def test_full_hash_then_sequence_parallel_slice(monkeypatch):
    hashes = torch.arange(2 * 8 * 3).view(2, 8, 3)
    monkeypatch.setattr("megatron.core.models.engram.hashing.get_pg_size", lambda _: 4)
    monkeypatch.setattr("megatron.core.models.engram.hashing.get_pg_rank", lambda _: 2)
    local = slice_hashes_for_sequence_parallel(hashes, local_sequence_length=2, tp_group=object())
    torch.testing.assert_close(local, hashes[:, 4:6])


def _packed_args(**overrides):
    values = dict(
        sft=False,
        use_varlen_dataset=False,
        varlen_sbhd_validation=False,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        max_seqlen_per_dp_cp_rank=0,
        seq_length=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_packed_sequence_detection_matches_the_dataset_packing_rule():
    assert not _validate_packed_sequence_args(_packed_args())
    assert _validate_packed_sequence_args(_packed_args(sft=True))
    assert _validate_packed_sequence_args(_packed_args(use_varlen_dataset=True))
    # SBHD validation reads the varlen dataset but keeps the dense layout, so it is not packed.
    assert not _validate_packed_sequence_args(
        _packed_args(use_varlen_dataset=True, varlen_sbhd_validation=True)
    )
    # Inter-document masking adds document boundaries to seq_length-token rows without padding
    # them to a packing capacity, so its rows are not packed sequences.
    assert not _validate_packed_sequence_args(_packed_args(dataloader_inter_document_masking=True))
    with pytest.raises(ValueError, match="sequence-packing schedulers"):
        _validate_packed_sequence_args(_packed_args(sequence_packing_scheduler="dp_balanced"))
    # Dynamic CP always selects a scheduler as well; it is reported on its own.
    with pytest.raises(ValueError, match="dynamic context parallelism"):
        _validate_packed_sequence_args(
            _packed_args(
                dynamic_context_parallel=True, sequence_packing_scheduler="default_dynamic_cp"
            )
        )
    with pytest.raises(ValueError, match="max-seqlen-per-dp-cp-rank"):
        _validate_packed_sequence_args(
            _packed_args(
                sft=True,
                pipeline_model_parallel_size=2,
                max_seqlen_per_dp_cp_rank=256,
                seq_length=512,
            )
        )
    assert _validate_packed_sequence_args(
        _packed_args(
            sft=True, pipeline_model_parallel_size=2, max_seqlen_per_dp_cp_rank=512, seq_length=512
        )
    )


def test_inter_document_masking_follows_the_pretrain_hybrid_cp_split():
    # Under CP, pretrain_hybrid splits these rows with a padded per-sequence zigzag, which the
    # per-document partition Engram applies to packed rows does not reproduce.
    with pytest.raises(ValueError, match="inter-document-masking with context parallelism"):
        _validate_packed_sequence_args(
            _packed_args(dataloader_inter_document_masking=True, context_parallel_size=2)
        )
    # With --sft the rows are split per document, so the same layout is accepted under CP.
    assert _validate_packed_sequence_args(
        _packed_args(dataloader_inter_document_masking=True, sft=True, context_parallel_size=2)
    )
    # The rows are not padded to a packing capacity, so the pipeline padding rules of packed
    # sequences do not apply.
    assert not _validate_packed_sequence_args(
        _packed_args(
            dataloader_inter_document_masking=True,
            pipeline_model_parallel_size=2,
            max_seqlen_per_dp_cp_rank=None,
            seq_length=512,
        )
    )


def _startup_transformer_config(**overrides):
    values = dict(
        num_layers=8,
        context_parallel_size=1,
        bf16=True,
        fp16=False,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        recompute_granularity=None,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
        fp8=None,
        fp4=None,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        overlap_moe_expert_parallel_comm=False,
        pipeline_model_parallel_size=1,
        pad_packed_seq_alignment=None,
        dynamic_context_parallel=False,
        attention_cp_layout="zigzag",
        linear_cp_layout="zigzag",
        cp_partition_mode="zigzag",
        wide_residual=None,
        moe_shortcut_connection=False,
        use_mup=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _startup_engram_config(tmp_path, layer_ids=(1,)):
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=16, layer_ids=layer_ids)
    return EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=layer_ids,
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )


@pytest.mark.parametrize("pipeline_model_parallel_size", [2, 4])
def test_packed_sequences_with_pipeline_parallelism_need_capacity_padding(
    tmp_path, pipeline_model_parallel_size
):
    config = _startup_engram_config(tmp_path)
    with pytest.raises(ValueError, match="pad-packed-seq-alignment max"):
        config.validate_startup(
            _startup_transformer_config(pipeline_model_parallel_size=pipeline_model_parallel_size),
            16,
            packed_sequences=True,
        )
    # Capacity padding is all packed rows need, with or without middle pipeline stages.
    config.validate_startup(
        _startup_transformer_config(
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pad_packed_seq_alignment="max",
        ),
        16,
        packed_sequences=True,
    )
    # Rows that are not packed sequences are never padded to a capacity.
    config.validate_startup(
        _startup_transformer_config(pipeline_model_parallel_size=pipeline_model_parallel_size), 16
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(wide_residual=WideResidualConfig(num_streams=2)), "wide_residual"),
        (dict(moe_shortcut_connection=True), "moe_shortcut_connection"),
        (dict(enable_attention_residuals=True), "enable_attention_residuals"),
        (dict(use_mup=True), "use_mup"),
        (dict(dynamic_context_parallel=True), "dynamic context parallelism"),
    ],
)
def test_unsupported_transformer_features_are_rejected(tmp_path, overrides, message):
    config = _startup_engram_config(tmp_path)
    config.validate_startup(_startup_transformer_config(), 16)
    with pytest.raises(ValueError, match=message):
        config.validate_startup(_startup_transformer_config(**overrides), 16)


@pytest.mark.parametrize(
    "layout_field", ["attention_cp_layout", "linear_cp_layout", "cp_partition_mode"]
)
def test_context_parallelism_requires_the_zigzag_layout(tmp_path, layout_field):
    config = _startup_engram_config(tmp_path)
    contiguous = {layout_field: "contiguous"}
    with pytest.raises(ValueError, match=f"{layout_field}='contiguous'"):
        config.validate_startup(
            _startup_transformer_config(context_parallel_size=2, **contiguous), 16
        )
    with pytest.raises(ValueError, match=f"{layout_field}='contiguous'"):
        config.validate_startup(
            _startup_transformer_config(context_parallel_size=2, **contiguous),
            16,
            packed_sequences=True,
        )
    # Without context parallelism every layer sees the whole sequence in order.
    config.validate_startup(_startup_transformer_config(**contiguous), 16)


def test_unset_linear_cp_layout_is_treated_as_zigzag(tmp_path):
    # TransformerConfig resolves an unset linear_cp_layout to zigzag; configs that skip that
    # resolution get the same treatment.
    config = _startup_engram_config(tmp_path)
    config.validate_startup(
        _startup_transformer_config(context_parallel_size=2, linear_cp_layout=None), 16
    )


@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe],
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        ["moe_router", "moe_preprocess"],
    ],
)
def test_transformer_engine_graphs_limited_to_moe_scopes_are_accepted(tmp_path, cuda_graph_modules):
    config = _startup_engram_config(tmp_path)
    config.validate_startup(
        _startup_transformer_config(
            cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
        ),
        16,
    )


@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [],
        None,
        "full",
        [CudaGraphModule.attn],
        [CudaGraphModule.attn, CudaGraphModule.moe_router],
        [CudaGraphModule.mlp],
        [CudaGraphModule.mamba],
    ],
)
def test_transformer_engine_graphs_that_can_capture_engram_are_rejected(
    tmp_path, cuda_graph_modules
):
    # An empty scope (or the deprecated "full") captures whole layers.
    config = _startup_engram_config(tmp_path)
    with pytest.raises(ValueError, match="restricted to the MoE scopes"):
        config.validate_startup(
            _startup_transformer_config(
                cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
            ),
            16,
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "full_iteration"])
def test_other_cuda_graph_implementations_are_rejected(tmp_path, cuda_graph_impl):
    config = _startup_engram_config(tmp_path)
    with pytest.raises(ValueError, match=f"cuda_graph_impl='{cuda_graph_impl}'"):
        config.validate_startup(
            _startup_transformer_config(
                cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=[CudaGraphModule.moe]
            ),
            16,
        )


def test_invalid_configuration_messages(tmp_path):
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=16, layer_ids=(1,))
    with pytest.raises(ValueError, match="exactly one value"):
        EngramConfig(
            global_vocab_sizes=(17,),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=0,
            tokenizer_map_path=str(artifact),
        )

    config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )
    # Context parallelism is supported for unpacked and packed rows (unpadded packs partition by
    # cu_seqlens in the CP batch split).
    config.validate_startup(_startup_transformer_config(context_parallel_size=2), 16)
    config.validate_startup(
        _startup_transformer_config(context_parallel_size=2), 16, packed_sequences=True
    )
    with pytest.raises(ValueError, match="expert_tensor_parallel_size == tensor_model_parallel"):
        config.validate_startup(
            _startup_transformer_config(
                tensor_model_parallel_size=2, expert_tensor_parallel_size=1
            ),
            16,
        )
    with pytest.raises(ValueError, match="FP16 parameters"):
        config.validate_startup(_startup_transformer_config(bf16=False, fp16=True), 16)
    # FP32 parameters are accepted (reference-parity runs); BF16 stays the production dtype.
    config.validate_startup(_startup_transformer_config(bf16=False), 16)
    with pytest.raises(ValueError, match="FP4"):
        config.validate_startup(_startup_transformer_config(fp4="e2m1"), 16)
    # FP8/MXFP8 recipes and MTP layers are permitted (Engram modules stay in BF16;
    # MTP layers never build Engram).
    config.validate_startup(_startup_transformer_config(fp8="mxfp8"), 16)
    config.validate_startup(_startup_transformer_config(mtp_num_layers=1), 16)
    # Packed rows are accepted for every variant: boundaries come from cu_seqlens when the
    # variant has no boundary token of its own.
    config.validate_startup(_startup_transformer_config(), 16, packed_sequences=True)
    # Activation recompute is accepted: the recompute path passes input_ids to the Engram
    # layers it replays.
    config.validate_startup(_startup_transformer_config(recompute_granularity="full"), 16)
    config.validate_startup(_startup_transformer_config(recompute_granularity="selective"), 16)
    # VPP needs no guard: the tokens a layer hashes depend on the microbatch, not the chunk.
    config.validate_startup(_startup_transformer_config(virtual_pipeline_model_parallel_size=2), 16)
    with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
        config.validate_startup(
            _startup_transformer_config(overlap_moe_expert_parallel_comm=True), 16
        )
    # The convolution history must fit in one sequence-parallel slice.
    with pytest.raises(ValueError, match="at least 9 positions"):
        config.validate_startup(
            _startup_transformer_config(
                sequence_parallel=True, tensor_model_parallel_size=4, expert_tensor_parallel_size=4
            ),
            16,
            sequence_length=16,
        )
    config.validate_startup(
        _startup_transformer_config(
            sequence_parallel=True, tensor_model_parallel_size=4, expert_tensor_parallel_size=4
        ),
        16,
        sequence_length=64,
    )
    # It must also fit in one zigzag chunk of an unpacked row under context parallelism; packed
    # rows are restored whole before the convolution.
    with pytest.raises(ValueError, match="each zigzag chunk to own at least 9 positions"):
        config.validate_startup(
            _startup_transformer_config(context_parallel_size=4), 16, sequence_length=64
        )
    config.validate_startup(
        _startup_transformer_config(context_parallel_size=4), 16, sequence_length=72
    )
    config.validate_startup(
        _startup_transformer_config(context_parallel_size=4),
        16,
        packed_sequences=True,
        sequence_length=64,
    )
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        config.validate_startup(_startup_transformer_config(), 17)


class _FakeTransformerLayer(BaseTransformerLayer):
    """A layer type that exposes the Engram composition point."""


class _FakeMambaLayer:
    """A layer type that cannot carry the memory."""


@dataclass
class _FakeLayerSubmodules:
    engram: object = IdentityOp


def _layer_spec():
    return ModuleSpec(module=_FakeTransformerLayer, submodules=_FakeLayerSubmodules())


def _stack_spec(**fields):
    return ModuleSpec(module=object, submodules=HybridStackSubmodules(**fields))


def _compose(tmp_path, spec, pattern, layer_ids, cuda_graph_impl="none"):
    return apply_engram_to_hybrid_stack_spec(
        spec,
        _startup_engram_config(tmp_path, layer_ids),
        pattern,
        SimpleNamespace(cuda_graph_impl=cuda_graph_impl),
    )


def _carries_engram(layer_spec):
    engram = layer_spec.submodules.engram
    return isinstance(engram, ModuleSpec) and engram.module is Engram


def test_engram_attaches_to_the_selected_layer_type_only(tmp_path):
    spec = _stack_spec(
        mamba_layer=ModuleSpec(module=_FakeMambaLayer),
        attention_layer=_layer_spec(),
        mlp_layer=_layer_spec(),
    )
    composed = _compose(tmp_path, spec, "M*M-", (2,))
    assert _carries_engram(composed.submodules.attention_layer)
    assert not _carries_engram(composed.submodules.mlp_layer)
    # Stack specs are shared module-level singletons, so the input spec stays untouched.
    assert not _carries_engram(spec.submodules.attention_layer)
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _compose(tmp_path, spec, "M*M-", (1,))


def test_gdn_layers_attach_to_both_gdn_fields(tmp_path):
    composed = _compose(
        tmp_path, _stack_spec(gdn_layer=_layer_spec(), gdn2_layer=_layer_spec()), "GG", (1,)
    )
    assert _carries_engram(composed.submodules.gdn_layer)
    assert _carries_engram(composed.submodules.gdn2_layer)
    composed = _compose(tmp_path, _stack_spec(gdn_layer=_layer_spec()), "GG", (2,))
    assert _carries_engram(composed.submodules.gdn_layer)
    # A field set to IdentityOp cannot carry the memory, even when the other field could.
    with pytest.raises(
        ValueError, match=r"does not accept Engram in this stack spec \(gdn_layer\)"
    ):
        _compose(tmp_path, _stack_spec(gdn_layer=IdentityOp, gdn2_layer=_layer_spec()), "GG", (1,))
    # A field the stack spec does not configure is not built, so the other one carries it.
    composed = _compose(tmp_path, _stack_spec(gdn2_layer=_layer_spec()), "GG", (1,))
    assert _carries_engram(composed.submodules.gdn2_layer)
    assert composed.submodules.gdn_layer is None


def test_kda_and_mla_layers(tmp_path):
    composed = _compose(tmp_path, _stack_spec(kda_layer=_layer_spec()), "KK", (1,))
    assert _carries_engram(composed.submodules.kda_layer)
    composed = _compose(
        tmp_path,
        _stack_spec(mla_layer=_layer_spec(), mla_fused_down_proj_layer=_layer_spec()),
        "++",
        (2,),
    )
    assert _carries_engram(composed.submodules.mla_layer)
    assert _carries_engram(composed.submodules.mla_fused_down_proj_layer)
    # A stack spec that provides no MLA layer rejects '+' layers.
    with pytest.raises(ValueError, match=r"assigns to a '\+' layer"):
        _compose(tmp_path, _stack_spec(attention_layer=_layer_spec()), "M+", (2,))


def test_csa_family_layers(tmp_path):
    shared = _layer_spec()
    composed = _compose(
        tmp_path, _stack_spec(csa_layer=shared, csa_qk_layernorm_layer=shared), "CC", (1,)
    )
    # A layer spec shared by both C fields stays shared.
    assert composed.submodules.csa_layer is composed.submodules.csa_qk_layernorm_layer
    assert _carries_engram(composed.submodules.csa_layer)
    # H and W layers fall back to the C fields when the stack spec has no dedicated field.
    composed = _compose(
        tmp_path,
        _stack_spec(csa_layer=_layer_spec(), csa_qk_layernorm_layer=_layer_spec()),
        "MHMW",
        (2, 4),
    )
    assert _carries_engram(composed.submodules.csa_layer)
    assert _carries_engram(composed.submodules.csa_qk_layernorm_layer)
    with pytest.raises(ValueError, match="provides no layer spec"):
        _compose(tmp_path, _stack_spec(attention_layer=_layer_spec()), "MW", (2,))


def test_moe_layers_are_rejected_under_transformer_engine_graphs(tmp_path):
    spec = _stack_spec(attention_layer=_layer_spec(), moe_layer=_layer_spec())
    assert _carries_engram(_compose(tmp_path, spec, "*E", (2,)).submodules.moe_layer)
    with pytest.raises(ValueError, match="cuda_graph_impl='transformer_engine'"):
        _compose(tmp_path, spec, "*E", (2,), cuda_graph_impl="transformer_engine")
    composed = _compose(tmp_path, spec, "*E", (1,), cuda_graph_impl="transformer_engine")
    assert _carries_engram(composed.submodules.attention_layer)
    assert not _carries_engram(composed.submodules.moe_layer)


def test_layer_numbering_follows_the_hybrid_pattern(tmp_path):
    spec = _stack_spec(
        mamba_layer=ModuleSpec(module=_FakeMambaLayer), attention_layer=_layer_spec()
    )
    # Pipe symbols mark pipeline-stage boundaries and do not shift the layer numbering.
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _compose(tmp_path, spec, "M*|M*", (3,))
    assert _carries_engram(_compose(tmp_path, spec, "M*|M*", (4,)).submodules.attention_layer)
    with pytest.raises(ValueError, match="describes only 4 layers"):
        _compose(tmp_path, spec, "M*M*", (5,))
    with pytest.raises(ValueError, match="requires --hybrid-layer-pattern"):
        _compose(tmp_path, spec, None, (2,))
