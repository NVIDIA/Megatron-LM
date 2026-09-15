# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the local (non-TE) tensor-parallel layers and the apex CUDA extensions.

* ``VocabParallelEmbedding``: the deterministic branch (``weight[ids]``) and the default
  ``F.embedding`` path, with heavily duplicated ids so the backward accumulates thousands of
  rows into the same embedding row.
* ``vocab_parallel_cross_entropy``: the unfused path deterministic mode relies on.
* ``ColumnParallelLinear`` / ``RowParallelLinear``: cuBLAS GEMMs under the pinned workspace,
  with and without apex ``fused_weight_gradient_mlp_cuda`` gradient-accumulation fusion.
* apex ``FusedLayerNorm`` (persistent and non-persistent) and ``FusedScaleMaskSoftmax``
  (causal and padding CUDA kernels).
"""

import importlib.util

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions import fused_layer_norm
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import init_method_normal
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HAVE_WGRAD_ACCUM_EXT = importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None
HAVE_SCALED_SOFTMAX_EXT = importlib.util.find_spec("scaled_masked_softmax_cuda") is not None and (
    importlib.util.find_spec("scaled_upper_triang_masked_softmax_cuda") is not None
)


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestTensorParallelLayers:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "deterministic_branch", [True, False], ids=["weight_index", "F.embedding"]
    )
    def test_vocab_parallel_embedding_backward_replays(self, deterministic_branch):
        """64 distinct ids over 32k positions: ~500 duplicate rows accumulate per embedding row."""
        seeded()
        config = _config(deterministic_mode=deterministic_branch)
        module = VocabParallelEmbedding(
            8192, 1024, init_method=init_method_normal(0.02), config=config
        ).cuda()
        ids = torch.randint(0, 64, (8, 4096), device="cuda") * 100
        with deterministic_algorithms(deterministic_branch):
            assert_module_replays_bit_exact(
                module,
                (ids,),
                replays=4,
                what=f"VocabParallelEmbedding[det={deterministic_branch}]",
            )

    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_vocab_parallel_cross_entropy_replays(self, label_smoothing):
        seeded()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        logits = torch.randn(4096, 32768, device="cuda", dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 32768 * tp_group.size(), (4096,), device="cuda")
        assert_replays_bit_exact(
            lambda l, t: vocab_parallel_cross_entropy(l, t, label_smoothing, tp_group),
            (logits, target),
            replays=3,
            what="vocab_parallel_cross_entropy",
        )

    @pytest.mark.parametrize(
        "grad_accum_fusion",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not HAVE_WGRAD_ACCUM_EXT, reason="apex fused_weight_gradient_mlp_cuda missing"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("layer", ["column", "row"])
    def test_local_linear_layers_replay(self, layer, grad_accum_fusion):
        seeded()
        config = _config(gradient_accumulation_fusion=grad_accum_fusion)
        if layer == "column":
            module = ColumnParallelLinear(
                4096, 16384, config=config, init_method=init_method_normal(0.02), bias=True
            ).cuda()
            x = torch.randn(4096, 2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        else:
            module = RowParallelLinear(
                16384,
                4096,
                config=config,
                init_method=init_method_normal(0.02),
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
            ).cuda()
            x = torch.randn(4096, 2, 16384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if grad_accum_fusion:
            # The fused wgrad kernel accumulates into a caller-owned fp32 buffer.
            module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
        assert_module_replays_bit_exact(
            module,
            (x,),
            replays=3,
            contention=True,
            what=f"{layer} linear[fusion={grad_accum_fusion}]",
        )


# --- apex fused layer norm ------------------------------------------------------------------


@pytest.mark.skipif(
    not (fused_layer_norm.HAVE_PERSIST_LAYER_NORM or fused_layer_norm.HAVE_FUSED_LAYER_NORM),
    reason="apex fused layer norm unavailable",
)
@pytest.mark.parametrize("hidden", [4096, 4000], ids=["persistent", "non-persistent"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fused_layer_norm_replays(hidden, zero_centered_gamma):
    """dgamma/dbeta reduce over 8192 rows -- the cross-row reduction apex has to order."""
    seeded()
    config = _config(
        hidden_size=hidden,
        normalization="LayerNorm",
        layernorm_zero_centered_gamma=zero_centered_gamma,
    )
    module = (
        fused_layer_norm.FusedLayerNorm(
            config,
            hidden,
            eps=1e-5,
            persist_layer_norm=True,
            zero_centered_gamma=zero_centered_gamma,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(8192, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert_module_replays_bit_exact(
        module, (x,), replays=3, contention=True, what=f"FusedLayerNorm[{hidden}]"
    )


# --- apex fused scale-mask softmax -------------------------------------------------------


@pytest.mark.skipif(not HAVE_SCALED_SOFTMAX_EXT, reason="apex scaled softmax extensions missing")
@pytest.mark.parametrize("mask_type", [AttnMaskType.causal, AttnMaskType.padding])
def test_fused_scale_mask_softmax_replays(mask_type):
    seeded()
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=mask_type,
        scaled_masked_softmax_fusion=True,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
    )
    b, np_, sq, sk = 4, 16, 2048, 2048
    scores = torch.randn(b, np_, sq, sk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if mask_type == AttnMaskType.causal:
        mask = None
    else:
        mask = torch.rand(b, 1, sq, sk, device="cuda") < 0.2
    assert module.is_kernel_available(mask, b, np_, sq, sk), "fused softmax kernel not selected"
    assert_replays_bit_exact(
        lambda s: module(s, mask),
        (scores,),
        replays=3,
        what=f"FusedScaleMaskSoftmax[{mask_type.name}]",
    )
