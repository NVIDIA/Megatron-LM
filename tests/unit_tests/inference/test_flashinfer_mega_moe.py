# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Config validation and a smoke forward for the flashinfer_mega inference MoE backend."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.moe.mega._deps import _HAVE_FLASHINFER_MOE_EP
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _config(**overrides):
    base = dict(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        moe_router_score_function="softmax",
        # inference_optimized rejects anything else.
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        # The megakernel stacks gate+up into w13 and applies SwiGLU.
        activation_func=F.silu,
        gated_linear_unit=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        # Must be the string, as argparse supplies it: __post_init__ converts to
        # the enum only after the gated-linear-unit backend check, which compares
        # against strings.
        inference_grouped_gemm_backend="flashinfer_mega",
        inference_mega_max_tokens_per_rank=64,
        attention_backend=AttnBackend.local,
        use_cpu_initialization=True,
    )
    base.update(overrides)
    return TransformerConfig(**base)


@pytest.mark.internal
class TestFlashinferMegaConfig:
    def test_enum_value(self):
        assert InferenceGroupedGemmBackend.FLASHINFER_MEGA.value == "flashinfer_mega"

    def test_config_accepts_mega_backend(self):
        cfg = _config(expert_model_parallel_size=1)
        assert cfg.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.FLASHINFER_MEGA

    def test_rejects_non_gated_activation(self):
        from megatron.core.activations import squared_relu

        with pytest.raises(ValueError, match="only implements SwiGLU"):
            _config(activation_func=squared_relu, gated_linear_unit=False)

    def test_rejects_tanh_clamp(self):
        with pytest.raises(ValueError, match="activation_func_tanh_clamp_scale"):
            _config(activation_func_tanh_clamp_scale=7.0)

    def test_rejects_unaligned_moe_ffn_hidden_size(self):
        with pytest.raises(ValueError, match="moe_ffn_hidden_size divisible by 64"):
            _config(moe_ffn_hidden_size=120)

    @pytest.mark.parametrize(
        "precision,hidden,moe_ffn",
        [
            ("bf16", 128, 128),
            ("mxfp8", 128, 160),  # mxfp8 needs moe_ffn % 32 only
            ("nvfp4", 128, 144),  # nvfp4 needs moe_ffn % 16 only
            ("fp8_fp4", 128, 128),
        ],
    )
    def test_accepts_per_precision_alignment(self, precision, hidden, moe_ffn):
        cfg = _config(
            inference_mega_precision=precision, hidden_size=hidden, moe_ffn_hidden_size=moe_ffn
        )
        assert cfg.inference_mega_precision == precision

    @pytest.mark.parametrize(
        "precision,hidden,divisor",
        [("bf16", 112, 32), ("mxfp8", 96, 64), ("nvfp4", 96, 64), ("fp8_fp4", 192, 128)],
    )
    def test_rejects_unaligned_hidden_size(self, precision, hidden, divisor):
        # Each kernel's activation staging quantizer sets its own hidden bound.
        with pytest.raises(ValueError, match=f"hidden_size divisible by {divisor}"):
            _config(inference_mega_precision=precision, hidden_size=hidden)

    def test_rejects_unknown_precision(self):
        with pytest.raises(ValueError, match="inference_mega_precision must be one of"):
            _config(inference_mega_precision="fp6")

    def test_rejects_experts_not_divisible_by_ep(self):
        with pytest.raises(ValueError, match="divisible by"):
            _config(num_moe_experts=6, expert_model_parallel_size=4)


def _training_config(**overrides):
    """A config for ``moe_inference_training_forward``, which requires MoE recompute."""
    return _config(
        **{"recompute_granularity": "selective", "recompute_modules": ["moe"], **overrides}
    )


@pytest.mark.internal
class TestMegaTrainingForwardConfig:
    """What the mega backend adds to moe_inference_training_forward's validation.

    The backend-independent rules (recompute, local CUDA graphs, EP-comm overlap,
    routing that breaks parity) are covered with the generic mode in
    test_inference_training_forward_config.py and are not repeated here.
    """

    def test_accepts_valid_config(self):
        config = _training_config(moe_inference_training_forward=True)
        assert config.moe_inference_training_forward

    def test_defaults_off(self):
        assert not _training_config().moe_inference_training_forward

    def test_mega_backend_is_in_the_allow_list(self):
        """flashinfer_mega is accepted by the generic backend allow-list."""
        config = _training_config(moe_inference_training_forward=True)
        assert config.inference_grouped_gemm_backend.value == "flashinfer_mega"

    def test_mxfp8_is_accepted_without_a_separate_opt_in(self):
        """Opting into the mode is the opt-in to the straight-through gradient."""
        config = _training_config(
            moe_inference_training_forward=True, inference_mega_precision="mxfp8"
        )
        assert config.inference_mega_precision == "mxfp8"

    @pytest.mark.parametrize("precision", ["nvfp4", "fp8_fp4"])
    def test_rejects_a_precision_without_a_packer(self, precision):
        """nvfp4 / fp8_fp4 cannot rebuild their kernel weights from the parameters."""
        with pytest.raises(ValueError, match="only bf16 and mxfp8 can be rebuilt"):
            _training_config(
                moe_inference_training_forward=True, inference_mega_precision=precision
            )

    def test_rejects_ep_comm_overlap(self):
        """The shared kernel-weight scratch assumes one MoE layer runs at a time."""
        with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
            _training_config(
                moe_inference_training_forward=True, overlap_moe_expert_parallel_comm=True
            )

    def test_still_requires_moe_recompute(self):
        """The generic recompute rule applies to the mega backend too."""
        with pytest.raises(ValueError, match="backward must come from a recompute pass"):
            _training_config(
                moe_inference_training_forward=True,
                recompute_granularity=None,
                recompute_modules=None,
            )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAVE_FLASHINFER_MOE_EP, reason="FlashInfer moe_ep mega not installed")
class TestFlashinferMegaForward:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not _is_blackwell(), reason="sm100 mega kernels require Blackwell")
    def test_mega_layer_forward_smoke(self):
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            MegaLocalPassthroughDispatcher,
        )

        MegaLocalPassthroughDispatcher.allocate_buffers()
        config = _config(expert_model_parallel_size=Utils.world_size)
        layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
        local_tokens = 8
        hidden = torch.randn(
            local_tokens, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad(), InferenceMode.active():
            out, _ = layer(hidden)
        assert out.shape == hidden.shape
        assert out.dtype == torch.bfloat16
