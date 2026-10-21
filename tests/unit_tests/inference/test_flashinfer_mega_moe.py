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
            ("bf16_nvfp4", 128, 128),  # hidden % 32, moe_ffn % 64
        ],
    )
    def test_accepts_per_precision_alignment(self, precision, hidden, moe_ffn):
        cfg = _config(
            inference_mega_precision=precision, hidden_size=hidden, moe_ffn_hidden_size=moe_ffn
        )
        assert cfg.inference_mega_precision == precision

    @pytest.mark.parametrize(
        "precision,hidden,divisor",
        [
            ("bf16", 112, 32),
            ("mxfp8", 96, 64),
            ("bf16_nvfp4", 112, 32),
        ],
    )
    def test_rejects_unaligned_hidden_size(self, precision, hidden, divisor):
        # Each kernel's activation staging quantizer sets its own hidden bound.
        with pytest.raises(ValueError, match=f"hidden_size divisible by {divisor}"):
            _config(inference_mega_precision=precision, hidden_size=hidden)

    def test_rejects_bf16_nvfp4_ffn_not_multiple_of_64(self):
        # 160 is a legal mxfp8 width (divisible by 32) and an illegal W4A16 width.
        with pytest.raises(ValueError, match="moe_ffn_hidden_size divisible by 64"):
            _config(inference_mega_precision="bf16_nvfp4", moe_ffn_hidden_size=160)

    def test_rejects_bf16_nvfp4_topk_above_32(self):
        with pytest.raises(ValueError, match="moe_router_topk <= 32"):
            _config(
                inference_mega_precision="bf16_nvfp4",
                moe_router_topk=33,
                num_moe_experts=64,
            )

    def test_rejects_unknown_precision(self):
        with pytest.raises(ValueError, match="inference_mega_precision must be one of"):
            _config(inference_mega_precision="fp6")

    def test_rejects_experts_not_divisible_by_ep(self):
        with pytest.raises(ValueError, match="divisible by"):
            _config(num_moe_experts=6, expert_model_parallel_size=4)


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
