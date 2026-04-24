# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def get_mamba_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_mamba_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        """Test GPU forward pass."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern)
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def test_layer_types(self):
        """
        Make sure that the layer types specified with layer_pattern
        were honored.
        """
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern)
        layers = block.layers
        # Note that this matches the order specified by layer_pattern above
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].mlp, MLP)

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = '+'
        assert invalid_symbol not in Symbols.VALID_LAYERS  # sanity check.
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # validate_segment_layers() in hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_mamba_block(layer_pattern)

    def test_gdn_layer_types(self):
        """
        Make sure that G creates a TransformerLayer wrapping GatedDeltaNet,
        while * creates a TransformerLayer wrapping SelfAttention.
        """
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        block = self.get_mamba_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], TransformerLayer)
        assert isinstance(layers[0].self_attention, GatedDeltaNet)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_gdn_gpu_forward(self):
        """Test GPU forward pass with GDN, attention, and Mamba layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
        )
        modules = hybrid_stack_spec.submodules
        block = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def test_dsa_layer_types(self):
        """D symbol creates a TransformerLayer with MLASelfAttention."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_mamba_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and D in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DS_ATTENTION + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_dsa_mamba_block(layer_pattern)


@pytest.mark.internal
class TestFusedLayerValidation:
    """Unit tests for the construction-time validation of fused layers.

    These tests exercise the failure paths in `build_fused_layer` directly
    so they don't need a process group or CUDA set up. The happy path is
    covered by higher-level forward tests in `TestHybridBlock`.
    """

    def _call(self, fused_symbols: str):
        from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
        from megatron.core.models.hybrid.hybrid_layer_fusion import build_fused_layer

        # The mixer attributes are only read for valid fusion groups; the
        # defaults (IdentityOp) are never touched for the error paths here.
        return build_fused_layer(
            fused_symbols,
            submodules=HybridStackSubmodules(),
            config=None,
            layer_number=1,
            pg_collection=None,
            pp_layer_offset=0,
            is_mtp_layer=False,
            add_layer_offset=False,
        )

    def test_single_layer_fusion_rejected(self):
        with pytest.raises(ValueError, match="exactly two fused layers"):
            self._call(Symbols.ATTENTION)

    def test_three_layer_fusion_rejected(self):
        with pytest.raises(ValueError, match="exactly two fused layers"):
            self._call(Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP)

    def test_channel_mixer_first_rejected(self):
        # MLP followed by attention – wrong order.
        with pytest.raises(ValueError, match="first fused layer.*sequence mixer"):
            self._call(Symbols.MLP + Symbols.ATTENTION)

    def test_sequence_mixer_second_rejected(self):
        # Attention followed by Mamba – second slot must be a channel mixer.
        with pytest.raises(ValueError, match="second fused layer.*channel mixer"):
            self._call(Symbols.ATTENTION + Symbols.MAMBA)

    def test_two_channel_mixers_rejected(self):
        with pytest.raises(ValueError, match="first fused layer.*sequence mixer"):
            self._call(Symbols.MLP + Symbols.MOE)


@pytest.mark.internal
class TestMambaMixerForTransformerLayer:
    """Unit tests for the `MambaMixerForTransformerLayer` adapter subclass.

    These probe only the adapter behaviour (kwarg defaulting in `__init__`,
    kwarg filtering in `forward`), so they don't need CUDA, process groups,
    or even a real `MambaMixer` instance – we poke at the class via stubbed-out
    base methods so the tests also work when `mamba-ssm` isn't installed.
    """

    def test_init_defaults_d_model_to_hidden_size(self):
        from types import SimpleNamespace

        from megatron.core.models.hybrid.hybrid_layer_fusion import MambaMixerForTransformerLayer
        from megatron.core.ssm import mamba_mixer as mm

        captured = {}

        def fake_init(self, config, submodules, **kwargs):
            captured["config"] = config
            captured["submodules"] = submodules
            captured["kwargs"] = kwargs

        original = mm.MambaMixer.__init__
        mm.MambaMixer.__init__ = fake_init
        try:
            instance = object.__new__(MambaMixerForTransformerLayer)
            # Simulate TransformerLayer's all-keyword call without `d_model`.
            MambaMixerForTransformerLayer.__init__(
                instance,
                config=SimpleNamespace(hidden_size=128),
                submodules="sub-sentinel",
                layer_number=3,
                pg_collection="pg-sentinel",
                pp_layer_offset=0,
            )
            assert captured["kwargs"]["d_model"] == 128
            assert captured["kwargs"]["layer_number"] == 3
            assert captured["kwargs"]["pg_collection"] == "pg-sentinel"
            assert captured["kwargs"]["pp_layer_offset"] == 0
        finally:
            mm.MambaMixer.__init__ = original

    def test_init_preserves_explicit_d_model(self):
        from types import SimpleNamespace

        from megatron.core.models.hybrid.hybrid_layer_fusion import MambaMixerForTransformerLayer
        from megatron.core.ssm import mamba_mixer as mm

        captured = {}

        def fake_init(self, config, submodules, **kwargs):
            captured["kwargs"] = kwargs

        original = mm.MambaMixer.__init__
        mm.MambaMixer.__init__ = fake_init
        try:
            instance = object.__new__(MambaMixerForTransformerLayer)
            MambaMixerForTransformerLayer.__init__(
                instance,
                config=SimpleNamespace(hidden_size=128),
                submodules="sub-sentinel",
                d_model=999,  # caller's explicit value must win
            )
            assert captured["kwargs"]["d_model"] == 999
        finally:
            mm.MambaMixer.__init__ = original

    def test_forward_filters_transformer_layer_kwargs(self):
        from megatron.core.models.hybrid.hybrid_layer_fusion import MambaMixerForTransformerLayer
        from megatron.core.ssm import mamba_mixer as mm

        captured = {}

        def fake_forward(
            self,
            hidden_states,
            inference_context=None,
            *,
            inference_params=None,
            packed_seq_params=None,
        ):
            captured["hidden_states"] = hidden_states
            captured["inference_context"] = inference_context
            captured["inference_params"] = inference_params
            captured["packed_seq_params"] = packed_seq_params
            return ("out-sentinel", "bias-sentinel")

        original = mm.MambaMixer.forward
        mm.MambaMixer.forward = fake_forward
        try:
            instance = object.__new__(MambaMixerForTransformerLayer)
            # Simulate TransformerLayer.forward's call – passes a bunch of
            # kwargs the base MambaMixer doesn't accept. The adapter must
            # swallow them and forward only the three it uses.
            out, bias = MambaMixerForTransformerLayer.forward(
                instance,
                "hidden-sentinel",
                attention_mask="mask-sentinel",
                inference_context="ctx-sentinel",
                rotary_pos_emb="rot-sentinel",
                rotary_pos_cos="cos-sentinel",
                rotary_pos_sin="sin-sentinel",
                rotary_pos_cos_sin="cos-sin-sentinel",
                attention_bias="bias-sentinel",
                packed_seq_params="packed-sentinel",
                sequence_len_offset="offset-sentinel",
            )
            assert out == "out-sentinel"
            assert bias == "bias-sentinel"
            assert captured["hidden_states"] == "hidden-sentinel"
            assert captured["inference_context"] == "ctx-sentinel"
            assert captured["packed_seq_params"] == "packed-sentinel"
            assert captured["inference_params"] is None
        finally:
            mm.MambaMixer.forward = original

    def test_forward_accepts_unknown_future_kwargs(self):
        """If TransformerLayer.forward grows a new kwarg, the adapter should
        silently absorb it rather than breaking the fused build at runtime.
        """
        from megatron.core.models.hybrid.hybrid_layer_fusion import MambaMixerForTransformerLayer
        from megatron.core.ssm import mamba_mixer as mm

        def fake_forward(
            self,
            hidden_states,
            inference_context=None,
            *,
            inference_params=None,
            packed_seq_params=None,
        ):
            return ("out", None)

        original = mm.MambaMixer.forward
        mm.MambaMixer.forward = fake_forward
        try:
            instance = object.__new__(MambaMixerForTransformerLayer)
            # A brand-new kwarg we've never heard of must not raise.
            out, _ = MambaMixerForTransformerLayer.forward(
                instance, "hidden-sentinel", some_future_kwarg_added_in_2027=42
            )
            assert out == "out"
        finally:
            mm.MambaMixer.forward = original


@pytest.mark.internal
class TestMambaStateShapesWithFusion:
    """Regression test: `HybridStack.mamba_state_shapes_per_request` must
    recognise fused blocks whose sequence mixer is Mamba. Before the fix the
    method only matched the bare `"M"` layer type, so a stack where every
    Mamba was inside a `[M-]` / `[ME]` group returned `None` and
    inference cache allocation silently broke.
    """

    def _make_stub_stack(self, layer_type_list, layers):
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        stub = object.__new__(HybridStack)
        stub.layer_type_list = layer_type_list
        stub.layers = layers
        return stub

    def test_standalone_mamba_is_found(self):
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        class FakeMambaLayer:
            def mamba_state_shapes_per_request(self):
                return ("conv-shape", "ssm-shape")

        stub = self._make_stub_stack(["*", "M", "-"], ["attn", FakeMambaLayer(), "mlp"])
        assert HybridStack.mamba_state_shapes_per_request(stub) == ("conv-shape", "ssm-shape")

    def test_fused_mamba_is_found_via_self_attention(self):
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        class FakeMambaMixer:
            def mamba_state_shapes_per_request(self):
                return ("conv-shape", "ssm-shape")

        class FakeTransformerLayer:
            self_attention = FakeMambaMixer()

        stub = self._make_stub_stack(["*", "M-"], ["attn", FakeTransformerLayer()])
        assert HybridStack.mamba_state_shapes_per_request(stub) == ("conv-shape", "ssm-shape")

    def test_standalone_wins_over_fused(self):
        # When both forms are present, the loop returns the first match.
        # The stand-alone "M" is earlier in this list, so its shapes win.
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        class FakeStandaloneMamba:
            def mamba_state_shapes_per_request(self):
                return "standalone"

        class FakeFusedMixer:
            def mamba_state_shapes_per_request(self):
                return "fused"

        class FakeTransformerLayer:
            self_attention = FakeFusedMixer()

        stub = self._make_stub_stack(["M", "M-"], [FakeStandaloneMamba(), FakeTransformerLayer()])
        assert HybridStack.mamba_state_shapes_per_request(stub) == "standalone"

    def test_no_mamba_at_all_returns_none(self):
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        stub = self._make_stub_stack(["*", "-", "*-"], ["attn", "mlp", "fused-*-"])
        assert HybridStack.mamba_state_shapes_per_request(stub) is None

    def test_fused_me_is_found(self):
        # "[ME]" – Mamba sequence mixer + MoE channel mixer, fused.
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        class FakeMambaMixer:
            def mamba_state_shapes_per_request(self):
                return "me-fused"

        class FakeTransformerLayer:
            self_attention = FakeMambaMixer()

        stub = self._make_stub_stack(["ME"], [FakeTransformerLayer()])
        assert HybridStack.mamba_state_shapes_per_request(stub) == "me-fused"
