# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from argparse import ArgumentParser
from dataclasses import replace
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None

_SEED = 42


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Patch hadamard_transform in dsa/csa modules if the library is not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Config / spec helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=16,
    v_head_dim=64,
    qk_pos_emb_head_dim=32,
    q_lora_rank=64,
    output_projection_groups=8,
    output_projection_lora_rank=64,
    csa_compress_ratios=None,
    csa_window_size=8,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    dsa_indexer_n_heads=8,
    dsa_indexer_head_dim=64,
    dsa_indexer_topk=8,
    dsa_indexer_loss_coeff=0.0,
    **extra_config_kwargs,
):
    """Create an MLATransformerConfig for DSv4 hybrid attention tests."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]
    extra_config_kwargs.setdefault('experimental_attention_variant', 'dsv4_hybrid')
    # Native attention fixtures opt out of the DSv4 fused backend default.
    extra_config_kwargs.setdefault('dsa_kernel_backend', 'none')
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        output_projection_groups=output_projection_groups,
        output_projection_lora_rank=output_projection_lora_rank,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_indexer_n_heads=dsa_indexer_n_heads,
        dsa_indexer_head_dim=dsa_indexer_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        **extra_config_kwargs,
    )


def _make_attention_spec(config):
    """Build the full DSv4HybridSelfAttention ModuleSpec using the canonical spec builder."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )

    return get_experimental_attention_variant_module_spec(config=config, backend=TESpecProvider())


def test_attention_latent_norm_epsilon_defaults_to_layernorm_epsilon():
    """Existing DSv4 configs inherit the model-wide norm epsilon."""
    config = _make_config(layernorm_epsilon=3e-6)

    assert config.attention_latent_norm_epsilon == pytest.approx(3e-6)


def test_attention_latent_norm_epsilon_accepts_override():
    """DSv4 latent norms can use the model recipe's dedicated epsilon."""
    config = _make_config(layernorm_epsilon=3e-6, attention_latent_norm_epsilon=1e-5)

    assert config.attention_latent_norm_epsilon == pytest.approx(1e-5)


def test_module_spec_is_built_from_explicit_backend():
    """The production selector should build DSv4 from its explicitly supplied backend."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
        Compressor,
        CSAIndexer,
    )
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )

    class Linear:
        pass

    class ColumnParallelLinear:
        pass

    class RowParallelLinear:
        pass

    class Norm:
        pass

    class Backend:
        def linear(self):
            return Linear

        def column_parallel_linear(self):
            return ColumnParallelLinear

        def row_parallel_linear(self):
            return RowParallelLinear

        def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
            return Norm

    spec = get_experimental_attention_variant_module_spec(_make_config(), Backend())

    assert spec.module is DSv4HybridSelfAttention
    assert spec.submodules.linear_q_down_proj is Linear
    assert spec.submodules.linear_q_up_proj is ColumnParallelLinear
    assert spec.submodules.linear_kv_proj is ColumnParallelLinear
    assert spec.submodules.linear_proj is RowParallelLinear
    core_attention_builder = spec.submodules.core_attention
    assert isinstance(core_attention_builder, partial)
    assert core_attention_builder.func is CompressedSparseAttention

    core_attention_submodules = core_attention_builder.keywords["submodules"]
    compressor_builder = core_attention_submodules.compressor
    assert isinstance(compressor_builder, partial)
    assert compressor_builder.func is Compressor

    indexer_builder = core_attention_submodules.indexer
    assert isinstance(indexer_builder, partial)
    assert indexer_builder.func is CSAIndexer
    assert indexer_builder.keywords["submodules"].compressor is compressor_builder


def test_grouped_output_projection_respects_cpu_initialization(monkeypatch):
    """The custom grouped projection follows the standard CPU/no-init constructor contract."""
    from megatron.core.transformer import identity_op
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as dsv4_attention,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    class SizeOneGroup:
        def size(self) -> int:
            return 1

    def unexpected_cuda_device() -> None:
        raise AssertionError("CPU initialization must not query the current CUDA device")

    def unexpected_parameter_init(_tensor: torch.Tensor) -> None:
        raise AssertionError("perform_initialization=False must skip parameter initialization")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_cuda_device)
    monkeypatch.setattr(dsv4_attention, "RotaryEmbedding", identity_op.IdentityOp)
    monkeypatch.setattr(dsv4_attention, "TELinear", identity_op.IdentityOp)

    config = _make_config(perform_initialization=False, csa_compress_ratios=[0, 0, 0, 0])
    config.init_method = unexpected_parameter_init
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = SizeOneGroup()
    pg_collection.cp = SizeOneGroup()
    submodules = dsv4_attention.DSv4HybridSelfAttentionSubmodules(
        q_layernorm=identity_op.IdentityOp,
        kv_layernorm=identity_op.IdentityOp,
        linear_q_down_proj=identity_op.IdentityOp,
        linear_q_up_proj=identity_op.IdentityOp,
        linear_kv_proj=identity_op.IdentityOp,
        core_attention=ModuleSpec(module=identity_op.IdentityOp),
        linear_proj=identity_op.IdentityOp,
    )

    attention = dsv4_attention.DSv4HybridSelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
        compress_ratio=0,
    )

    assert attention.linear_o_group_proj.device.type == "cpu"


def test_config_includes_mtp_ratio_and_derives_dimensions():
    """DSv4 config should account for MTP and derive its shared Q/KV content width."""
    config = _make_config(num_layers=2, mtp_num_layers=1, csa_compress_ratios=[0, 4, 128])

    expected_content_dim = config.v_head_dim - config.qk_pos_emb_head_dim
    assert config.qk_head_dim == expected_content_dim
    assert config.kv_lora_rank == expected_content_dim
    assert config.hetereogenous_dist_checkpoint is True


def test_config_accepts_cudnn_backend_for_fused_sbhd():
    """DSv4 may select the CSA cuDNN adapter while ordinary DSA keeps its own router."""
    # The cuDNN branch probes ``cudnn.DSA`` wrapper signatures, so stub a module that
    # advertises the compact wrapper contract the validation requires.
    fake_dsa = SimpleNamespace(
        indexer_forward_top_k_wrapper=lambda *args, deterministic=False, **kwargs: None
    )
    fake_cudnn = ModuleType("cudnn")
    fake_cudnn.DSA = fake_dsa
    with (
        patch.dict(sys.modules, {"cudnn": fake_cudnn}),
        patch(
            'megatron.core.transformer.transformer_config._validate_dsa_kernel_backend_dependencies'
        ),
        patch.object(torch.cuda, 'get_device_capability', return_value=(10, 0)),
    ):
        config = _make_config(dsa_kernel_backend="cudnn")

    assert config.dsa_kernel_backend == "cudnn"


def test_config_rejects_sm90_ratio4_dense_indexer_loss():
    """SM90 must use sparse indexer loss for the fused ratio-4 CSA path."""
    with (
        patch(
            'megatron.core.transformer.transformer_config._validate_dsa_kernel_backend_dependencies'
        ),
        patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)),
        pytest.raises(ValueError, match="dense indexer loss is not supported on SM90"),
    ):
        _make_config(dsa_kernel_backend="cudnn", dsa_indexer_loss_coeff=0.1)


def test_config_rejects_tilelang_backend_for_dsv4():
    """The main-owned TileLang ordinary-DSA backend is not a CSA implementation."""
    with pytest.raises(ValueError, match="does not support.*tilelang"):
        _make_config(dsa_kernel_backend="tilelang")


@pytest.mark.parametrize(
    ("variant", "requested", "expected"),
    [("dsv4_hybrid", None, "cudnn"), ("dsv4_hybrid", "none", "none"), ("dsa", None, "none")],
)
def test_cli_backend_default_preserves_explicit_none(variant, requested, expected):
    """The generated CLI preserves omission for variant-aware config defaults."""
    from megatron.training.arguments import _add_network_size_args

    parser = _add_network_size_args(ArgumentParser())
    argv = [] if requested is None else ['--dsa-kernel-backend', requested]
    args = parser.parse_args(argv)
    assert args.dsa_kernel_backend == requested
    with (
        patch(
            'megatron.core.transformer.transformer_config._validate_dsa_kernel_backend_dependencies'
        ),
        patch.object(torch.cuda, 'get_device_capability', return_value=(10, 0)),
    ):
        config = _make_config(
            experimental_attention_variant=variant, dsa_kernel_backend=args.dsa_kernel_backend
        )
    assert config.dsa_kernel_backend == expected


def test_config_accepts_hybrid_model_ratio_tail():
    """HybridModel may expand each MTP depth into multiple attention layers."""
    config = _make_config(num_layers=2, mtp_num_layers=1, csa_compress_ratios=[0, 4, 128, 4])
    assert config.csa_compress_ratios == [0, 4, 128, 4]


def test_hybrid_stack_spec_uses_static_ratio_agnostic_specs():
    """C/H/W use static norm/no-norm specs; layer configs provide their ratios."""
    from megatron.core.models.hybrid.hybrid_layer_specs import (
        hybrid_dsv4_stack_spec,
        hybrid_stack_spec,
    )
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    assert isinstance(hybrid_dsv4_stack_spec, ModuleSpec)
    assert hybrid_dsv4_stack_spec is hybrid_stack_spec
    attention = hybrid_stack_spec.submodules.csa_layer.submodules.self_attention
    assert attention.module is DSv4HybridSelfAttention
    assert "compress_ratio" not in attention.params
    assert attention.submodules.q_layernorm is IdentityOp
    assert attention.submodules.kv_layernorm is IdentityOp
    normalized = hybrid_stack_spec.submodules.csa_qk_layernorm_layer.submodules.self_attention
    assert normalized.module is DSv4HybridSelfAttention
    assert "compress_ratio" not in normalized.params
    assert normalized.submodules.q_layernorm is not IdentityOp
    assert normalized.submodules.kv_layernorm is not IdentityOp


@pytest.mark.parametrize("variant", [None, "dsa"])
@pytest.mark.parametrize("entrypoint", ["hybrid_stack", "attention"])
def test_dsv4_construction_rejects_incompatible_variant_before_backend_work(variant, entrypoint):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
    from megatron.core.transformer.attention import Attention
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )
    from megatron.core.transformer.spec_utils import build_module

    config = replace(_make_config(), experimental_attention_variant=variant)
    with (
        patch.object(Attention, "__init__", return_value=None) as attention_init,
        pytest.raises(ValueError, match="DSv4 attention requires.*dsv4_hybrid"),
    ):
        if entrypoint == "hybrid_stack":
            build_module(
                hybrid_dsv4_stack_spec.submodules.csa_layer.submodules.self_attention,
                config=config,
                layer_number=1,
                pg_collection=ProcessGroupCollection(),
            )
        else:
            DSv4HybridSelfAttention(
                config=config,
                submodules=None,
                layer_number=1,
                pg_collection=ProcessGroupCollection(),
            )
    attention_init.assert_not_called()


def _build_cpu_attention_for_ratio_resolution(monkeypatch, config, explicit_ratio=None):
    """Build enough of DSv4 attention on CPU to exercise ratio selection."""
    from megatron.core.transformer import identity_op
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as dsv4_attention,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec

    class SizeOneGroup:
        def size(self) -> int:
            return 1

    monkeypatch.setattr(dsv4_attention, "RotaryEmbedding", identity_op.IdentityOp)
    monkeypatch.setattr(dsv4_attention, "YarnRotaryEmbedding", identity_op.IdentityOp)
    monkeypatch.setattr(dsv4_attention, "TELinear", identity_op.IdentityOp)

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = SizeOneGroup()
    pg_collection.cp = SizeOneGroup()
    submodules = dsv4_attention.DSv4HybridSelfAttentionSubmodules(
        q_layernorm=identity_op.IdentityOp,
        kv_layernorm=identity_op.IdentityOp,
        linear_q_down_proj=identity_op.IdentityOp,
        linear_q_up_proj=identity_op.IdentityOp,
        linear_kv_proj=identity_op.IdentityOp,
        core_attention=ModuleSpec(module=identity_op.IdentityOp),
        linear_proj=identity_op.IdentityOp,
    )
    kwargs = {}
    if explicit_ratio is not None:
        kwargs["compress_ratio"] = explicit_ratio
    return dsv4_attention.DSv4HybridSelfAttention(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("config_ratio", "explicit_ratio", "expected_ratio"),
    [
        pytest.param(None, None, 4, id="global-list-fallback"),
        pytest.param(128, None, 128, id="layer-config-precedes-global-list"),
        pytest.param(128, 4, 4, id="explicit-argument-precedes-layer-config"),
    ],
)
def test_compress_ratio_resolution_precedence(
    monkeypatch, config_ratio, explicit_ratio, expected_ratio
):
    """Resolve explicit, per-layer, and legacy-list ratios in that priority order."""
    from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
        CSALayerConfig,
    )

    config = _make_config(perform_initialization=False, csa_compress_ratios=[4, 0, 0, 0])
    if config_ratio is not None:
        config = CSALayerConfig.from_config(config)
        config.compress_ratio = config_ratio

    attention = _build_cpu_attention_for_ratio_resolution(
        monkeypatch, config, explicit_ratio=explicit_ratio
    )

    assert attention._dsv4_compress_ratio == expected_ratio


def test_constructor_requires_explicit_process_groups():
    """Production DSv4 construction must not read process groups from global MPU state."""
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )

    with pytest.raises(ValueError, match="explicit ProcessGroupCollection"):
        DSv4HybridSelfAttention(config=None, submodules=None, layer_number=1)


def _build_attention(config, layer_number, pg_collection, **kwargs):
    """Instantiate DSv4 attention through HybridModel's static spec."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.spec_utils import build_module

    layer_spec = (
        hybrid_stack_spec.submodules.csa_qk_layernorm_layer
        if config.qk_layernorm
        else hybrid_stack_spec.submodules.csa_layer
    )
    spec = layer_spec.submodules.self_attention
    return build_module(
        spec, config=config, layer_number=layer_number, pg_collection=pg_collection, **kwargs
    )


# ===========================================================================
# Constructor tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionConstructor:
    """Test construction of DSv4HybridSelfAttention in the supported TP=1 configuration."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_basic_construction(self):
        """Verify the layer builds and has the expected sub-modules."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.identity_op import IdentityOp

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert isinstance(attn, DSv4HybridSelfAttention)
        assert hasattr(attn, 'linear_q_down_proj')
        assert hasattr(attn, 'linear_q_up_proj')
        assert hasattr(attn, 'linear_kv_proj')
        assert hasattr(attn, 'linear_proj')
        assert hasattr(attn, 'linear_o_group_proj')
        assert hasattr(attn, 'core_attention')
        assert hasattr(attn, 'q_layernorm')
        assert hasattr(attn, 'kv_layernorm')
        assert isinstance(attn.q_layernorm, IdentityOp)
        assert isinstance(attn.kv_layernorm, IdentityOp)
        assert not any(
            key.startswith(("q_layernorm.", "kv_layernorm.")) for key in attn.state_dict()
        )

    def test_latent_norm_epsilon_is_scoped_to_q_and_kv_latents(self):
        """The dedicated epsilon must not change the compressor norm."""
        config = _make_config(
            layernorm_epsilon=1e-5,
            attention_latent_norm_epsilon=1e-6,
            qk_layernorm=True,
            csa_compress_ratios=[4, 4, 128, 4],
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_layernorm.eps == pytest.approx(1e-6)
        assert attn.kv_layernorm.eps == pytest.approx(1e-6)
        assert attn.state_dict()["q_layernorm.weight"].shape == (config.q_lora_rank,)
        assert attn.state_dict()["kv_layernorm.weight"].shape == (config.v_head_dim,)
        assert attn.core_attention.compressor.norm.eps == pytest.approx(1e-5)

    def test_q_head_dim_equals_v_head_dim(self):
        """q_head_dim must equal v_head_dim for DSv4 hybrid."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        assert attn.q_head_dim == config.v_head_dim

    def test_current_main_constructor_kwargs(self):
        """Current TransformerLayer forwards module names and pipeline offsets."""
        config = _make_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(
            config,
            layer_number=1,
            pg_collection=pg,
            pp_layer_offset=0,
            name="decoder.layers.0.self_attention",
        )

        assert attn._pp_layer_offset == 0

    def test_missing_hybrid_mtp_ratio_fails_with_layer_context(self):
        """A multi-layer MTP tail should fail clearly instead of raising a bare IndexError."""
        config = _make_config(num_layers=2, mtp_num_layers=1, csa_compress_ratios=[0, 4, 128])
        config.csa_compress_ratios = config.csa_compress_ratios[:2]
        pg = ProcessGroupCollection.use_mpu_process_groups()

        with pytest.raises(ValueError, match="MTP layer 1.*requires at least 3 entries"):
            _build_attention(config, layer_number=1, pg_collection=pg, is_mtp_layer=True)

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_rope_base_varies_with_compress_ratio(self, layer_number):
        """Layers with compress_ratio > 1 should use csa_compress_rotary_base."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=layer_number, pg_collection=pg)

        ratio = ratios[layer_number - 1]
        if ratio > 1:
            expected_base = config.csa_compress_rotary_base
        else:
            expected_base = config.rotary_base

        # inv_freq is derived from rotary_base; verify the correct base was used
        dim = config.qk_pos_emb_head_dim
        recomputed_inv_freq = 1.0 / (
            expected_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        assert torch.allclose(
            attn.rotary_pos_emb.inv_freq.cpu(), recomputed_inv_freq, rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Forward / backward tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionForwardBackward:
    """Test forward and backward passes of DSv4HybridSelfAttention."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_number", [1, 2, 3, 4])
    def test_forward_output_shape(self, layer_number):
        """Forward should produce [sq, b, hidden_size] output."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_backward_gradient_flow(self, layer_number):
        """Backward should produce gradients for all trainable parameters."""
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = (
            torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        output, bias = attn(hidden_states=hidden, attention_mask=None)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None, "No gradient on hidden_states"
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"

    def test_eval_mode(self):
        """Forward should work in eval mode."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        attn.eval()

        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        with torch.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert not torch.isnan(output).any()

    def test_different_seq_lengths(self):
        """Forward should handle various sequence lengths."""
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=2, pg_collection=self.pg).cuda()

        for seq_len in [64, 128, 256]:
            hidden = torch.randn(
                seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
            ).cuda()
            output, bias = attn(hidden_states=hidden, attention_mask=None)
            assert output.shape == (seq_len, batch_size, self.config.hidden_size)


# ===========================================================================
# get_query_key_value_tensors tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridQKV:
    """Test get_query_key_value_tensors internals."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_qkv_shapes(self):
        """Query, key, value should have correct shapes."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(hidden)

        n_heads = self.config.num_attention_heads
        v_dim = self.config.v_head_dim

        assert q.shape == (seq_len, batch_size, n_heads, v_dim)
        # key and value are single-head (MQA-style) with an extra head dim
        assert k.shape[-1] == v_dim
        assert v.shape[-1] == v_dim

    def test_key_equals_value(self):
        """In the wkv path, key and value should be the same tensor."""
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(self.config, layer_number=1, pg_collection=self.pg).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        q, k, v, _, _ = attn.get_query_key_value_tensors(hidden)
        assert torch.equal(k, v), "key and value should be identical in wkv path"


# ===========================================================================
# Grouped output projection tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridGroupedOutput:
    """Test that grouped output projection (wo_a) parameters are created."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_o_group_proj_shape(self):
        """linear_o_group_proj should have the correct shape."""
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        output_projection_groups = 8
        output_projection_lora_rank = 64
        config = _make_config(
            output_projection_groups=output_projection_groups,
            output_projection_lora_rank=output_projection_lora_rank,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_attention(config, layer_number=1, pg_collection=pg)

        expected_out = output_projection_groups * output_projection_lora_rank
        expected_in = (config.v_head_dim * config.num_attention_heads) // output_projection_groups
        assert attn.linear_o_group_proj.shape == (expected_out, expected_in)
        assert attn.linear_o_group_proj.requires_grad


# ===========================================================================
# DSv4 Hybrid Attention + Hash MoE integration tests
# ===========================================================================


def _make_dsv4_hash_moe_config():
    """Create a compact DSv4 config that combines CSA/HCA with hash MoE."""
    return _make_config(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        q_lora_rank=32,
        output_projection_groups=4,
        output_projection_lora_rank=32,
        csa_compress_ratios=[4, 128],
        csa_window_size=16,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_layer_freq=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_router_score_function="sqrtsoftplus",
        moe_n_hash_layers=1,
        actual_vocab_size=128,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=False,
        moe_grouped_gemm=False,
    )


def _build_dsv4_moe_layer(config, layer_number, pg_collection):
    """Instantiate a TransformerLayer from the DSv4 experimental attention spec."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_layer_with_experimental_attention_variant_spec,
    )
    from megatron.core.transformer.spec_utils import build_module

    layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
        config=config, backend=TESpecProvider()
    )
    return build_module(
        layer_specs[layer_number - 1],
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridHashMoEIntegration:
    """Integration coverage for DSv4 hybrid attention with hash MoE and clamped SwiGLU."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.config = _make_dsv4_hash_moe_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_csa_hash_moe_layer_forward_backward(self):
        """Layer 1 should combine DSv4 CSA, hash routing, and clamped SwiGLU."""
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
            DSv4HybridSelfAttention,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer

        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        layer = _build_dsv4_moe_layer(self.config, layer_number=1, pg_collection=self.pg).cuda()
        layer.train()

        assert isinstance(layer.self_attention, DSv4HybridSelfAttention)
        assert layer.self_attention.core_attention.compress_ratio == 4
        assert isinstance(layer.mlp, MoELayer)
        assert layer.mlp.router.is_hash_layer is True
        assert layer.mlp.router.tid2eid is not None
        assert layer.config.activation_func_clamp_value == 10.0
        assert layer.config.activation_func is F.silu
        assert layer.config.gated_linear_unit is True

        hidden = torch.randn(
            seq_len,
            batch_size,
            self.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        input_ids = torch.randint(
            0, self.config.actual_vocab_size, (batch_size, seq_len), device="cuda"
        )

        output, context = layer(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
        loss = output.float().square().mean()
        loss.backward()

        assert context is None
        assert output.shape == hidden.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.self_attention.parameters()
            if p.requires_grad
        )
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.mlp.parameters()
            if p.requires_grad
        )

    def test_hash_moe_layer_requires_input_ids_but_hca_layer_does_not(self):
        """Hash routing is limited to leading layers while later HCA MoE layers remain runnable."""
        seq_len = 256
        batch_size = 1

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        hash_layer = _build_dsv4_moe_layer(
            self.config, layer_number=1, pg_collection=self.pg
        ).cuda()
        hidden = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with pytest.raises(AssertionError, match="input_ids is required for hash-based routing"):
            hash_layer(hidden_states=hidden, attention_mask=None)

        hca_layer = _build_dsv4_moe_layer(self.config, layer_number=2, pg_collection=self.pg).cuda()
        assert hca_layer.self_attention.core_attention.compress_ratio == 128
        assert hca_layer.mlp.router.is_hash_layer is False

        output, context = hca_layer(hidden_states=hidden, attention_mask=None)

        assert context is None
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()


# ===========================================================================
# apply_rope_fusion tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridRopeFusion:
    """Test that apply_rope_fusion=True works for both yarn and non-yarn layers.

    DSv4 Hybrid uses YarnRotaryEmbedding for layers with compress_ratio > 1
    and standard RotaryEmbedding for layers with compress_ratio <= 1. The
    fused RoPE path must obtain cos/sin from both embedding classes via
    get_cached_cos_sin.

    compress_ratios=[0, 4, 128, 4]: layer 1 has ratio 0 (standard
    RotaryEmbedding), layers 2-4 have ratio > 1 (YarnRotaryEmbedding).
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    def test_rope_fusion_forward_backward_parity(self):
        """Fused RoPE forward/backward succeeds and matches the unfused path."""
        seq_len = 128
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        fused_config = _make_config(apply_rope_fusion=True)
        attn_fused = _build_attention(fused_config, layer_number=4, pg_collection=self.pg).cuda()
        attn_fused.train()

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        unfused_config = _make_config(apply_rope_fusion=False)
        attn_unfused = _build_attention(
            unfused_config, layer_number=4, pg_collection=self.pg
        ).cuda()
        attn_unfused.train()

        hidden = torch.randn(
            seq_len, batch_size, fused_config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        out_fused, _ = attn_fused(hidden_states=hidden, attention_mask=None)
        out_unfused, _ = attn_unfused(hidden_states=hidden, attention_mask=None)

        assert out_fused.shape == (seq_len, batch_size, fused_config.hidden_size)
        assert torch.isfinite(out_fused).all()
        # Production code forces ``mscale=1.0`` (DSv4 contract) in both
        # fused and unfused paths, so the only residual is bf16 noise from
        # the fused Triton kernel's different accumulation order vs the
        # PyTorch eager ops. The residual concentrates at output positions
        # whose values are near zero (sign flips on tiny magnitudes drive
        # the worst-case max-abs-diff).
        torch.testing.assert_close(out_fused, out_unfused, atol=3e-2, rtol=3e-2)

        hidden_fused = hidden.detach().clone().requires_grad_(True)
        hidden_unfused = hidden.detach().clone().requires_grad_(True)

        attn_fused(hidden_states=hidden_fused, attention_mask=None)[0].sum().backward()
        attn_unfused(hidden_states=hidden_unfused, attention_mask=None)[0].sum().backward()

        assert hidden_fused.grad is not None
        for name, param in attn_fused.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"


# ===========================================================================
# THD packed-sequence end-to-end
# ===========================================================================
#
# Closes the highest-level integration gap: even though the CSA THD path
# is independently tested in test_attention_variant_csa.py
# (TestCompressedSparseAttentionThd), the DSv4HybridSelfAttention module
# adds its own THD-aware glue around CSA — packed_seq_params propagation
# through get_query_key_value_tensors, the output reshape at line ~290
# (``core_attn_out.reshape(total, 1, -1)`` to recover the 3-D contract),
# and the inverse-RoPE call with cu_seqlens. These tests verify the full
# DSv4Hybrid forward/backward works end-to-end for each ``compress_ratio``.


from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402


def _make_thd_packed_seq_params(seg_lens, device='cuda'):
    """Build ``PackedSeqParams(qkv_format='thd', ...)`` for self-attention
    (``cu_seqlens_q == cu_seqlens_kv``) from a list of per-segment lengths.
    """
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seg_lens, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_len = int(max(seg_lens)) if seg_lens else 0
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format='thd',
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionThd:
    """End-to-end THD forward/backward of :class:`DSv4HybridSelfAttention`
    across all configured ``compress_ratio`` values (0/4/128).

    Each test runs a multi-segment THD batch through the full
    ``attn(hidden_states, packed_seq_params=...)`` pipeline and verifies:

      * Output shape ``(total_tokens, 1, hidden_size)`` — the layer
        re-adds the dummy ``b=1`` axis at line ~293 of
        ``deepseek_v4_hybrid_attention.py``.
      * No NaN.
      * (Backward test) grads flow on ``hidden_states`` and every
        learnable parameter.

    This is the highest-level integration test for THD; lower-level
    parity vs SBHD is established by ``TestCompressedSparseAttentionThd``
    in ``test_attention_variant_csa.py``.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        # ``csa_compress_ratios=[0, 4, 128, 4]`` → layer_number ∈ {1,2,3,4}
        # cover all three CSA ratios (0 = window-only; 4 = full
        # indexer+compressed; 128 = compressor-only, no indexer).
        cls.config = _make_config(dsa_indexer_loss_coeff=1.0)
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()

        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2, 3, 4],
        ids=[
            "ratio_0_window_only",  # layer 1 → ratio=0
            "ratio_4_with_indexer",  # layer 2 → ratio=4
            "ratio_128_compressor_only",  # layer 3 → ratio=128
            "ratio_4_with_indexer_alt",  # layer 4 → ratio=4
        ],
    )
    def test_thd_forward_output_shape(self, layer_number):
        """THD forward through DSv4HybridSelfAttention produces
        ``(total_tokens, 1, hidden_size)`` output, no NaN, for every
        configured ``compress_ratio``.
        """
        seg_lens = [128, 96, 64]  # multi-segment, varied lengths
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        # THD hidden_states shape is ``(total_tokens, 1, hidden_size)``
        # per the DSv4 hybrid contract.
        hidden = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')
        packed = _make_thd_packed_seq_params(seg_lens)

        with torch.no_grad():
            output, _bias = attn(
                hidden_states=hidden, attention_mask=None, packed_seq_params=packed
            )

        assert output.shape == (total, 1, self.config.hidden_size), (
            f"layer {layer_number}: shape {tuple(output.shape)} != "
            f"expected {(total, 1, self.config.hidden_size)}"
        )
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any(), f"layer {layer_number}: NaN in THD forward output"

    @pytest.mark.parametrize(
        "layer_number",
        [1, 2],  # ratio=0 (window-only) and ratio=4 (full indexer pipeline)
        ids=["ratio_0_window_only", "ratio_4_with_indexer"],
    )
    def test_thd_backward_gradient_flow(self, layer_number):
        """THD backward produces grads on ``hidden_states`` and every
        learnable parameter (covers the full indexer-loss path in Path
        B THD when ``layer_number=2`` triggers ratio=4).
        """
        seg_lens = [128, 96]
        total = sum(seg_lens)

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.train()

        hidden = torch.randn(
            total, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda'
        ).requires_grad_(True)
        packed = _make_thd_packed_seq_params(seg_lens)

        output, _bias = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)
        output.sum().backward()

        assert hidden.grad is not None, "no grad on hidden_states"
        assert not torch.isnan(hidden.grad).any()
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no grad on {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad on {name}"

    def test_thd_single_segment_matches_sbhd_b1(self):
        """B=1 single-segment THD output matches the SBHD-b=1 output on
        identical hidden states (sanity check that the DSv4Hybrid
        THD-vs-SBHD glue doesn't silently change the math).

        Uses ``layer_number=1`` (ratio=0, window-only) for determinism —
        no indexer top-K tie-breaking nondeterminism.
        """
        layer_number = 1  # ratio=0 → window-only path, no cuDNN topk
        sq = 128

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        attn = _build_attention(
            self.config, layer_number=layer_number, pg_collection=self.pg
        ).cuda()
        attn.eval()

        hidden = torch.randn(sq, 1, self.config.hidden_size, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            out_sbhd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=None)
            packed = _make_thd_packed_seq_params([sq])
            out_thd, _ = attn(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)

        assert out_sbhd.shape == out_thd.shape
        # Generous tol: the full DSv4Hybrid forward chains many bf16 ops
        # (QKV down/up proj, RoPE, attn, output proj). We're testing for
        # no plumbing bug, not bit-exactness.
        assert torch.allclose(out_sbhd.float(), out_thd.float(), atol=5e-2, rtol=5e-2), (
            f"B=1 SBHD/THD parity failed: max abs diff = "
            f"{(out_sbhd.float() - out_thd.float()).abs().max().item():.4e}"
        )
