# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Owners build their linear projections with their own process-group collection.

MLA (plain, fused and absorbed), cross-attention, the DSA, simplified DSA and CSA indexers, the CSA
compressor and GDN hold a collection and pass it, with its TP group, to every linear they build. A
Transformer Engine linear reads its GTP replica group and the amax group of a per-module
quantization recipe from that collection, and falls back to the global groups without one.

Without CUDA, the local-spec owners build on CPU in one process with a gloo group; the Transformer
Engine and GDN owners need GPUs.
"""

from functools import partial
from operator import attrgetter

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import (
    Utils,
    build_test_pg_collection,
    destroy_test_pg_collection,
)

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEFusedMLP,
        TELayerNormColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
        split_te_layernorm_column_parallel_linear,
    )
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )

_NOT_PASSED = object()


def _record_given_collections(patch, classes):
    """Record on every instance of `classes` the pg_collection keyword it was built with."""
    for cls in classes:
        original_init = cls.__init__

        def init(self, *args, _original_init=original_init, **kwargs):
            _original_init(self, *args, **kwargs)
            self.given_pg_collection = kwargs.get("pg_collection", _NOT_PASSED)

        patch.setattr(cls, "__init__", init)


class _CoreAttention(torch.nn.Module):
    """Stands in for the core attention, which these tests do not run."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _mla_config(**overrides):
    config = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=16,
        rope_type="rope",
        apply_rope_fusion=False,
        add_bias_linear=False,
        use_cpu_initialization=True,
        transformer_impl="local",
    )
    config.update(overrides)
    return MLATransformerConfig(**config)


_MLA_LINEARS = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
]
_MLA_LINEARS_WITHOUT_Q_LORA = [
    "linear_q_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
]


def _build_mla(
    pg_collection, *, column_linear, row_linear, down_linear, attention_cls, **config_overrides
):
    """Build MLA or absorbed MLA; return it and the names of the linears it builds."""
    config = _mla_config(**config_overrides)
    submodules_cls = (
        AbsorbedMLASelfAttentionSubmodules
        if attention_cls is AbsorbedMLASelfAttention
        else MLASelfAttentionSubmodules
    )
    submodules = submodules_cls(
        linear_q_proj=column_linear,
        linear_q_down_proj=down_linear,
        linear_q_up_proj=column_linear,
        linear_kv_down_proj=down_linear,
        linear_kv_up_proj=column_linear,
        core_attention=_CoreAttention,
        linear_proj=row_linear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    module = attention_cls(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return module, (_MLA_LINEARS if config.q_lora_rank is not None else _MLA_LINEARS_WITHOUT_Q_LORA)


def _build_fused_mla(pg_collection, *, column_linear, row_linear, down_linear, **config_overrides):
    submodules = MLASelfAttentionSubmodules(
        linear_qkv_down_proj=down_linear,
        linear_q_up_proj=column_linear,
        linear_kv_up_proj=column_linear,
        core_attention=_CoreAttention,
        linear_proj=row_linear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    module = FusedMLASelfAttention(
        config=_mla_config(**config_overrides),
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return module, ["linear_qkv_down_proj", "linear_q_up_proj", "linear_kv_up_proj", "linear_proj"]


def _build_cross_attention(pg_collection, *, column_linear, row_linear):
    config = TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
    )
    submodules = CrossAttentionSubmodules(
        linear_q=column_linear,
        linear_kv=column_linear,
        core_attention=_CoreAttention,
        linear_proj=row_linear,
    )
    module = CrossAttention(
        config=config, submodules=submodules, layer_number=1, pg_collection=pg_collection
    )
    return module, ["linear_q", "linear_kv", "linear_proj"]


def _build_dsa_indexer(pg_collection, *, linear, norm):
    config = _mla_config(
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=1.0,
    )
    submodules = DSAIndexerSubmodules(
        linear_wq_b=linear, linear_wk=linear, k_norm=norm, linear_weights_proj=linear
    )
    module = DSAIndexer(config, submodules, pg_collection)
    return module, ["linear_wq_b", "linear_wk", "linear_weights_proj"]


def _build_simplified_dsa_indexer(pg_collection, *, linear, norm):
    del norm
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=8,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        use_cpu_initialization=True,
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=4,
        dsa_kernel_backend="min-memory-torch",
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_use_sparse_loss=True,
    )
    # The indexer reads its RoPE settings as dynamic attributes that megatron/training sets.
    config.rotary_percent = 0.0
    submodules = SimplifiedDSGQAIndexerSubmodules(linear_q=linear, linear_k=linear)
    module = SimplifiedDSGQAIndexer(config, submodules, pg_collection=pg_collection)
    return module, ["linear_q", "linear_k"]


def _build_csa_indexer(pg_collection, *, linear, norm):
    config = _mla_config(
        v_head_dim=32,
        multi_latent_attention=True,
        csa_compress_ratios=[4],
        csa_window_size=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
    )
    compressor_submodules = CompressorSubmodules(linear_wkv=linear, linear_wgate=linear, norm=norm)
    submodules = CSAIndexerSubmodules(
        linear_wq_b=linear,
        linear_weights_proj=linear,
        compressor=partial(Compressor, submodules=compressor_submodules),
    )
    module = CSAIndexer(config, submodules, compress_ratio=4, pg_collection=pg_collection)
    return module, [
        "linear_wq_b",
        "linear_weights_proj",
        "compressor.linear_wkv",
        "compressor.linear_wgate",
    ]


def _local_owners():
    """Owners built from local linears, as the local and ModelOpt specs build them."""
    local = dict(column_linear=ColumnParallelLinear, row_linear=RowParallelLinear)
    mla = partial(_build_mla, attention_cls=MLASelfAttention, **local)
    absorbed_mla = partial(_build_mla, attention_cls=AbsorbedMLASelfAttention, **local)
    return {
        "mla": partial(mla, down_linear=ColumnParallelLinear),
        "mla_without_q_lora": partial(mla, down_linear=ColumnParallelLinear, q_lora_rank=None),
        "mla_modelopt_down_projections": partial(mla, down_linear=Linear),
        "fused_mla": partial(_build_fused_mla, down_linear=ColumnParallelLinear, **local),
        "absorbed_mla": partial(absorbed_mla, down_linear=ColumnParallelLinear),
        "absorbed_mla_modelopt_down_projections": partial(absorbed_mla, down_linear=Linear),
        "cross_attention": partial(_build_cross_attention, **local),
        "dsa_indexer": partial(_build_dsa_indexer, linear=Linear, norm=WrappedTorchNorm),
        "simplified_dsa_indexer": partial(
            _build_simplified_dsa_indexer, linear=Linear, norm=WrappedTorchNorm
        ),
        "csa_indexer": partial(_build_csa_indexer, linear=Linear, norm=WrappedTorchNorm),
    }


def _te_owners():
    """Owners built from Transformer Engine linears, as the default specs build them."""
    te = dict(column_linear=TEColumnParallelLinear, row_linear=TERowParallelLinear)
    mla = partial(
        _build_mla,
        attention_cls=MLASelfAttention,
        down_linear=TELinear,
        transformer_impl="transformer_engine",
        **te,
    )
    return {
        "mla": mla,
        "mla_without_q_lora": partial(mla, q_lora_rank=None),
        "fused_mla": partial(
            _build_fused_mla,
            down_linear=TELayerNormColumnParallelLinear,
            transformer_impl="transformer_engine",
            **te,
        ),
        "absorbed_mla": partial(
            _build_mla,
            attention_cls=AbsorbedMLASelfAttention,
            down_linear=TELinear,
            transformer_impl="transformer_engine",
            **te,
        ),
        "cross_attention": partial(_build_cross_attention, **te),
        "dsa_indexer": partial(_build_dsa_indexer, linear=TELinear, norm=TENorm),
        "simplified_dsa_indexer": partial(
            _build_simplified_dsa_indexer, linear=TELinear, norm=TENorm
        ),
        "csa_indexer": partial(_build_csa_indexer, linear=TELinear, norm=TENorm),
    }


# MLA takes the ModelOpt linear for its down projections only with Transformer Engine installed.
_MODELOPT_MLA_OWNERS = {"mla_modelopt_down_projections", "absorbed_mla_modelopt_down_projections"}
_LOCAL_OWNER_NAMES = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name in _MODELOPT_MLA_OWNERS and not HAVE_TE,
            reason="MLA supports the ModelOpt linear only with Transformer Engine",
        ),
    )
    for name in _local_owners()
]
_TE_OWNER_NAMES = [
    "mla",
    "mla_without_q_lora",
    "fused_mla",
    "absorbed_mla",
    "cross_attention",
    "dsa_indexer",
    "simplified_dsa_indexer",
    "csa_indexer",
]


@pytest.fixture
def owner_pg_collection(monkeypatch):
    """A collection over the standard grid's ranks whose groups are not parallel_state's."""
    if torch.cuda.is_available():
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        pg_collection = build_test_pg_collection()
        yield pg_collection
        destroy_test_pg_collection(pg_collection)
        Utils.destroy_model_parallel()
    else:
        created_default_group = not torch.distributed.is_initialized()
        if created_default_group:
            torch.distributed.init_process_group(
                "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
            )
        # Rotary embeddings and the CSA compressor allocate on the current device.
        monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
        group = torch.distributed.new_group([0])
        # GTP is off, as in build_test_pg_collection, so no linear falls back for its axes.
        yield ProcessGroupCollection(
            tp=group, cp=group, pp=group, gtp_remat=None, expt_gtp_remat=None
        )
        torch.distributed.destroy_process_group(group)
        if created_default_group:
            # Later tests in the session must not find this one-rank default group.
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("owner", _LOCAL_OWNER_NAMES)
def test_local_spec_owner_passes_its_collection(monkeypatch, owner_pg_collection, owner):
    """Each local (non-Transformer Engine) linear an owner builds gets the owner's collection."""
    local_classes = (ColumnParallelLinear, RowParallelLinear, Linear)
    _record_given_collections(monkeypatch, local_classes)

    module, linear_names = _local_owners()[owner](owner_pg_collection)

    for name in linear_names:
        linear = attrgetter(name)(module)
        assert type(linear) in local_classes, name
        assert linear.given_pg_collection is owner_pg_collection, name


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine linears need CUDA")
class TestTransformerEngineOwnersPassTheirCollection:
    """Transformer Engine linears keep the collection that their GTP and amax groups come from."""

    def _te_linear_classes(self):
        return (
            TELinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )

    @pytest.mark.parametrize("owner", _TE_OWNER_NAMES)
    def test_owner_passes_its_collection(self, monkeypatch, owner_pg_collection, owner):
        te_classes = self._te_linear_classes()
        _record_given_collections(monkeypatch, te_classes)

        module, linear_names = _te_owners()[owner](owner_pg_collection)

        for name in linear_names:
            linear = attrgetter(name)(module)
            assert isinstance(linear, te_classes), name
            assert linear.given_pg_collection is owner_pg_collection, name
            assert linear._pg_collection is owner_pg_collection, name

    @pytest.mark.skipif(not HAVE_FLA, reason="GDN needs FLA")
    def test_gdn_passes_its_collection(self, monkeypatch, owner_pg_collection):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            num_query_groups=2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            activation_func=F.silu,
            use_cpu_initialization=True,
            bf16=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        te_classes = self._te_linear_classes()
        _record_given_collections(monkeypatch, te_classes)

        gdn = GatedDeltaNet(
            config,
            submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=owner_pg_collection,
        )

        for linear in (gdn.in_proj, gdn.out_proj):
            assert isinstance(linear, te_classes)
            assert linear.given_pg_collection is owner_pg_collection
            assert linear._pg_collection is owner_pg_collection

    def test_split_layernorm_linear_keeps_the_collection(self, owner_pg_collection):
        """MLA's latent cache splits linear_kv_up_proj; the split linear keeps its collection."""
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, normalization="RMSNorm"
        )
        fused = TELayerNormColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=owner_pg_collection.tp,
            pg_collection=owner_pg_collection,
        )

        _, linear = split_te_layernorm_column_parallel_linear(fused, config)

        assert linear._pg_collection is owner_pg_collection
        assert linear._tp_group is owner_pg_collection.tp

    @pytest.mark.skipif(HAVE_TE and TEFusedMLP is None, reason="TE operation-based MLP unavailable")
    def test_fused_mlp_passes_its_collection(self, owner_pg_collection):
        """A layer builds TEFusedMLP through as_mlp_submodule, which passes on its collection."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        submodules = MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        )

        mlp = TEFusedMLP.as_mlp_submodule(
            submodules=submodules,
            config=config,
            pg_collection=owner_pg_collection,
            is_mtp_layer=False,
        )

        assert mlp.pg_collection is owner_pg_collection
        assert mlp.linear_fc1._pg_collection is owner_pg_collection
        assert mlp.linear_fc2._pg_collection is owner_pg_collection
