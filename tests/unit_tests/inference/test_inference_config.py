# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import subprocess
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import (
    AsyncScheduleMode,
    InferenceConfig,
    MambaInferenceStateConfig,
)
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.quantization.utils import resolve_mxfp8_backend
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE as GDP_CHUNK_SIZE
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_inference_args
from megatron.training.config.inference_config import InferenceSetupConfig
from tests.unit_tests.test_utilities import Utils


class TestInferenceConfig:
    @pytest.mark.parametrize(
        "imports",
        [
            (
                "from megatron.core.inference.config import MambaInferenceStateConfig; "
                "from megatron.core.ssm.mamba_layer import MambaLayer; "
                "from megatron.core.ssm.gated_delta_net import GatedDeltaNet; "
                "from megatron.core.transformer.attention import Attention"
            ),
            (
                "from megatron.core.ssm.mamba_layer import MambaLayer; "
                "from megatron.core.inference.config import MambaInferenceStateConfig"
            ),
            (
                "from megatron.core.ssm.gated_delta_net import GatedDeltaNet; "
                "from megatron.core.inference.config import MambaInferenceStateConfig"
            ),
            (
                "from megatron.core.transformer.attention import Attention; "
                "from megatron.core.inference.config import MambaInferenceStateConfig"
            ),
        ],
    )
    def test_layer_config_modules_do_not_create_inference_import_cycles(self, imports):
        """Inference config and layer implementations import cleanly in either order."""
        if Utils.rank != 0:
            return
        subprocess.run([sys.executable, "-c", imports], check=True)

    @pytest.mark.parametrize(
        ("grouped_gemm_backend", "expected_backend"),
        [
            ("torch", "triton"),
            (InferenceGroupedGemmBackend.TORCH, "triton"),
            ("flashinfer", "flashinfer"),
            (InferenceGroupedGemmBackend.FLASHINFER, "flashinfer"),
        ],
    )
    def test_resolve_mxfp8_backend(self, grouped_gemm_backend, expected_backend):
        assert resolve_mxfp8_backend(grouped_gemm_backend) == expected_backend

    @pytest.mark.parametrize("grouped_gemm_backend", ["vllm", InferenceGroupedGemmBackend.VLLM])
    def test_resolve_mxfp8_backend_rejects_unsupported_backend(self, grouped_gemm_backend):
        with pytest.raises(ValueError, match="does not support inference_grouped_gemm_backend"):
            resolve_mxfp8_backend(grouped_gemm_backend)

    @staticmethod
    def _hybrid_model(layer_config_types, experimental_attention_variant="gdn"):
        return SimpleNamespace(
            config=SimpleNamespace(
                params_dtype=torch.bfloat16,
                batch_invariant_mode=False,
                experimental_attention_variant=experimental_attention_variant,
            ),
            decoder=SimpleNamespace(
                layer_config_list=[
                    object.__new__(layer_config_type) for layer_config_type in layer_config_types
                ],
                layers=[],
            ),
        )

    def test_mamba_inference_state_config_rejects_mixed_recurrent_layers(self):
        """Mamba and GDN cannot share one state shape and prefill chunk size."""
        model = self._hybrid_model([MambaLayerConfig, GDNLayerConfig])

        with pytest.raises(ValueError, match="mixing Mamba and GDN"):
            MambaInferenceStateConfig.from_model(model)

    def test_mamba_inference_state_config_rejects_gdn2(self):
        """GDN2 should fail explicitly instead of missing the GDN inference hooks."""
        model = self._hybrid_model([GDNLayerConfig], experimental_attention_variant="gdn2")

        with pytest.raises(NotImplementedError, match="GDN2"):
            MambaInferenceStateConfig.from_model(model)

    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    @pytest.mark.parametrize(
        "async_sched_mode, expected",
        [
            (None, AsyncScheduleMode.LEGACY),
            ("legacy", AsyncScheduleMode.LEGACY),
            (AsyncScheduleMode.LEGACY, AsyncScheduleMode.LEGACY),
            ("async", AsyncScheduleMode.ASYNC),
            (AsyncScheduleMode.ASYNC, AsyncScheduleMode.ASYNC),
        ],
    )
    def test_async_sched_mode_default_and_coercion(self, async_sched_mode, expected):
        """Ensure async scheduling mode defaults to legacy and accepts strings."""
        kwargs = {} if async_sched_mode is None else {"async_sched_mode": async_sched_mode}
        assert InferenceConfig(**kwargs).async_sched_mode == expected

    @pytest.mark.parametrize("invalid_mode", ["serial", "overlap", "invalid"])
    def test_async_sched_mode_rejects_invalid_value(self, invalid_mode):
        """Ensure invalid async scheduling modes fail during config construction."""
        with pytest.raises(ValueError):
            InferenceConfig(async_sched_mode=invalid_mode)

    def test_async_sched_argparse_plumbing(self):
        """Ensure the CLI exposes async scheduling mode."""
        parser = _add_inference_args(ArgumentParser())
        args = parser.parse_args(["--inference-dynamic-batching-async-sched-mode", "async"])
        assert args.inference_dynamic_batching_async_sched_mode == "async"

    @pytest.mark.parametrize("invalid_mode", ["serial", "overlap"])
    def test_async_sched_argparse_rejects_removed_modes(self, invalid_mode):
        """Ensure the CLI rejects removed async scheduling modes."""
        parser = _add_inference_args(ArgumentParser())
        with pytest.raises(SystemExit):
            parser.parse_args(["--inference-dynamic-batching-async-sched-mode", invalid_mode])

    def test_inference_setup_config_maps_async_sched_mode(self):
        """Ensure declarative inference config maps async scheduling mode to runtime config."""
        model = SimpleNamespace(
            position_embedding_type="rope",
            max_sequence_length=4096,
            pg_collection="pg",
            decoder=SimpleNamespace(layer_config_list=None),
        )
        setup_config = InferenceSetupConfig(inference_dynamic_batching_async_sched_mode="async")

        inference_config = setup_config.to_inference_config(
            model=model,
            kv_cache_management_mode="persist",
            static_kv_memory_pointers=False,
            enable_cuda_graphs=False,
            verbose=False,
        )

        assert inference_config.async_sched_mode == AsyncScheduleMode.ASYNC

    def test_offset_sampling_seed_argparse_plumbing(self):
        """Ensure the CLI can select a shared sampling seed across DP ranks."""
        parser = _add_inference_args(ArgumentParser())
        default_args = parser.parse_args([])
        assert default_args.offset_sampling_seed_by_dp_rank is True

        disabled_args = parser.parse_args(["--use-same-sampling-seed-across-dp-ranks"])
        assert disabled_args.offset_sampling_seed_by_dp_rank is False

    def test_inference_setup_config_maps_offset_sampling_seed_by_dp_rank(self):
        """Ensure declarative inference config maps DP seed offset to runtime config."""
        model = SimpleNamespace(
            position_embedding_type="rope",
            max_sequence_length=4096,
            pg_collection="pg",
            decoder=SimpleNamespace(layer_config_list=None),
        )
        setup_config = InferenceSetupConfig(offset_sampling_seed_by_dp_rank=False)

        inference_config = setup_config.to_inference_config(
            model=model,
            kv_cache_management_mode="persist",
            static_kv_memory_pointers=False,
            enable_cuda_graphs=False,
            verbose=False,
        )

        assert inference_config.offset_sampling_seed_by_dp_rank is False

    def test_mamba_state_config_accepts_layer_config_subclasses(self, monkeypatch):
        """Model-derived Mamba metadata recognizes layer config subclasses."""

        class CustomMambaLayerConfig(MambaLayerConfig):
            pass

        attention_config = object.__new__(AttentionLayerConfig)
        mamba_layer_config = object.__new__(CustomMambaLayerConfig)
        decoder = SimpleNamespace(
            layer_config_list=[attention_config, mamba_layer_config],
            layers=[
                SimpleNamespace(mixer=SimpleNamespace(chunk_size=16)),
                SimpleNamespace(mixer=SimpleNamespace(chunk_size=64)),
            ],
            mamba_state_shapes_per_request=lambda: ((4, 8), (8, 32, 16)),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(batch_invariant_mode=False, params_dtype=torch.bfloat16)
        )
        monkeypatch.setattr(
            "megatron.core.inference.config.get_attr_wrapped_model",
            lambda *_args, **_kwargs: decoder,
        )

        mamba_state_config = MambaInferenceStateConfig.from_model(model)

        assert mamba_state_config is not None
        assert mamba_state_config.layer_type_list == [Symbols.ATTENTION, Symbols.MAMBA]
        assert mamba_state_config.layer_config_list[0] is attention_config
        assert mamba_state_config.layer_config_list[1] is mamba_layer_config
        assert mamba_state_config.conv_states_shape == (4, 8)
        assert mamba_state_config.ssm_states_shape == (8, 32, 16)
        assert mamba_state_config.conv_states_dtype is torch.bfloat16
        assert mamba_state_config.ssm_states_dtype is torch.bfloat16
        assert mamba_state_config.mamba_chunk_size == 64

        decoder.layer_config_list = [attention_config]
        decoder.layers = decoder.layers[:1]
        assert MambaInferenceStateConfig.from_model(model) is None

    def test_mamba_state_config_accepts_legacy_layer_type_list(self):
        """Direct callers can continue constructing inference state from layer symbols."""
        config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA, Symbols.ATTENTION],
            conv_states_shape=(4, 8),
            ssm_states_shape=(8, 32, 16),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )

        assert config.layer_type_list == [Symbols.MAMBA, Symbols.ATTENTION]
        assert config.layer_config_list is None


def _ssm_model(mixers):
    """A stand-in model exposing only what `MambaInferenceStateConfig.from_model` reads."""
    decoder = SimpleNamespace(
        layer_config_list=[object.__new__(MambaLayerConfig) for _ in mixers],
        layers=[SimpleNamespace(mixer=mixer) for mixer in mixers],
        mamba_state_shapes_per_request=lambda: ((16, 4), (2, 8, 16)),
    )
    return SimpleNamespace(
        decoder=decoder,
        config=SimpleNamespace(params_dtype=torch.bfloat16, batch_invariant_mode=False),
    )


def _mamba_mixer(chunk_size=128):
    return SimpleNamespace(chunk_size=chunk_size, ssm_inference_chunk_size=chunk_size)


def _gdp_mixer(chunk_size=128, num_householder=2):
    return SimpleNamespace(
        chunk_size=chunk_size,
        ssm_inference_chunk_size=GDP_CHUNK_SIZE,
        num_householder=num_householder,
    )


class TestSSMChunkAlignment:
    """The chunk-alignment quantum threaded through `MambaInferenceStateConfig`.

    A mixer's `chunk_size` is not always the chunk length its inference kernels
    run at: the forked Gated Delta Product prefill kernels chunk at a fixed 64
    whatever `chunk_size` says. Scheduling decisions that must land on a chunk
    boundary read `ssm_chunk_alignment`, so this is the seam where a wrong answer
    turns into silently unrecordable state boundaries.
    """

    @pytest.mark.internal
    def test_mixer_classes_expose_the_inference_chunk_size(self):
        """Both mixers answer the same question, so `from_model` can ask uniformly."""
        assert isinstance(MambaMixer.ssm_inference_chunk_size, property)
        assert isinstance(GatedDeltaProductMixer.ssm_inference_chunk_size, property)
        # GDP's answer is a constant, so it can be read without an instance.
        assert GatedDeltaProductMixer.ssm_inference_chunk_size.fget(None) == GDP_CHUNK_SIZE
        assert GDP_CHUNK_SIZE == 64

    @pytest.mark.internal
    def test_mamba_only_model_aligns_to_the_mamba_chunk_size(self):
        config = MambaInferenceStateConfig.from_model(_ssm_model([_mamba_mixer(128)] * 3))
        assert config.mamba_chunk_size == 128
        assert config.ssm_chunk_alignment == 128
        assert config.gdp_num_householder == 0

    @pytest.mark.internal
    def test_gdp_only_model_aligns_to_the_gdp_kernel_chunk_size(self):
        """The training-path `chunk_size` of 128 must not be mistaken for the real one."""
        config = MambaInferenceStateConfig.from_model(_ssm_model([_gdp_mixer(chunk_size=128)] * 3))
        assert config.ssm_chunk_alignment == GDP_CHUNK_SIZE
        assert config.gdp_num_householder == 2
        # mamba_chunk_size keeps its old meaning -- it sizes the Mamba2 chunk
        # metadata buffers, which a GDP-only model never reads.
        assert config.mamba_chunk_size == 128

    @pytest.mark.internal
    def test_stack_without_a_recurrent_layer_reports_no_chunking(self):
        """A pipeline stage of pure attention/MLP layers has nothing to report."""
        from megatron.core.ssm.ssm_inference import ssm_chunking

        layer_configs = [object.__new__(AttentionLayerConfig), object.__new__(MLPLayerConfig)]
        assert ssm_chunking(layer_configs, [SimpleNamespace(), SimpleNamespace()]) is None

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "mixers",
        [
            pytest.param([_mamba_mixer(128), _gdp_mixer()], id="mamba-then-gdp"),
            pytest.param([_gdp_mixer(), _mamba_mixer(128)], id="gdp-then-mamba"),
            pytest.param([_mamba_mixer(128), _mamba_mixer(64)], id="differing-chunk-size"),
            pytest.param(
                [_gdp_mixer(num_householder=2), _gdp_mixer(num_householder=3)],
                id="differing-householder",
            ),
        ],
    )
    def test_heterogeneous_stack_is_rejected(self, mixers):
        """The SSM stack is assumed homogeneous; a mixed one must fail loudly.

        Nothing downstream models a per-mixer alignment quantum or per-mixer
        chunk descriptors, so silently applying the first layer's answer to every
        other layer would mis-align boundaries rather than error.
        """
        with pytest.raises(AssertionError, match="every SSM layer must share one chunking"):
            MambaInferenceStateConfig.from_model(_ssm_model(mixers))

    @pytest.mark.internal
    def test_heterogeneous_stack_names_the_offending_layers(self):
        """The message must point at a layer, not just at two tuples of numbers."""
        mixers = [_mamba_mixer(128), _mamba_mixer(128), _gdp_mixer()]
        with pytest.raises(AssertionError, match=r"layer 0 has .* but layer 2 has"):
            MambaInferenceStateConfig.from_model(_ssm_model(mixers))

    @pytest.mark.internal
    def test_alignment_defaults_to_the_mamba_chunk_size(self):
        """Hand-built configs that predate the field keep their old behaviour."""
        config = MambaInferenceStateConfig(
            layer_type_list=None,
            layer_config_list=[object.__new__(MambaLayerConfig)],
            conv_states_shape=(16, 4),
            ssm_states_shape=(2, 8, 16),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
            mamba_chunk_size=64,
        )
        assert config.ssm_chunk_alignment == 64
