# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi Delta Attention against the HuggingFace ``glm5_next`` reference implementation.

GLM-5.3-Flash is the checkpoint family KDA was brought up against, so its HF module
(``Glm5NextTextLinearAttention``) is the reference here. The test packs two sequences of
different lengths into one ``thd`` batch and compares against the per-sequence HF forward.
"""

import pytest
import torch

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_kda_module_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kda import HAVE_FLA_KDA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, HEADS, HEAD_DIM, KERNEL = 128, 4, 32, 4


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="flash-linear-attention with fla.ops.kda required.")
@pytest.mark.internal
class TestKimiDeltaAttention:
    """Single-GPU parity of KDA against the HF glm5_next linear-attention layer."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _hf_reference(dtype):
        transformers = pytest.importorskip(
            "transformers", reason="transformers with glm5_next required."
        )
        modeling = pytest.importorskip(
            "transformers.models.glm5_next.modeling_glm5_next",
            reason="transformers with glm5_next required.",
        )

        hf_config = transformers.Glm5NextTextConfig(
            hidden_size=HIDDEN,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            layer_types=["linear_attention"],
            mlp_layer_types=["dense"],
            indexer_types=["full"],
            linear_attn_config={
                "num_heads": HEADS,
                "head_dim": HEAD_DIM,
                "short_conv_kernel_size": KERNEL,
                "gate_lower_bound": -5.0,
                "kda_layers": [0],
                "full_attn_layers": [],
            },
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        hf = modeling.Glm5NextTextLinearAttention(hf_config, layer_idx=0).cuda()
        with torch.no_grad():
            for proj in (
                hf.q_proj,
                hf.k_proj,
                hf.v_proj,
                hf.o_proj,
                hf.b_proj,
                hf.g_a_proj,
                hf.g_b_proj,
                hf.forget_gate.f_a_proj,
                hf.forget_gate.f_b_proj,
            ):
                proj.weight.normal_(0, 0.05)
            hf.conv1d.weight.normal_(0, 0.3)
            hf.forget_gate.A_log.copy_(torch.empty(HEADS).uniform_(1, 16).log())
            hf.forget_gate.dt_bias.uniform_(-3.0, -1.0)
            hf.o_norm.weight.uniform_(0.8, 1.2)
        # The real model keeps projections/conv/norm in bf16 and A_log/dt_bias in fp32.
        hf = hf.to(dtype)
        hf.forget_gate.A_log.data = hf.forget_gate.A_log.data.float()
        hf.forget_gate.dt_bias.data = hf.forget_gate.dt_bias.data.float()
        return hf

    @staticmethod
    def _megatron_module(dtype):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            num_query_groups=4,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            linear_num_value_heads=HEADS,
            linear_num_key_heads=HEADS,
            linear_key_head_dim=HEAD_DIM,
            linear_value_head_dim=HEAD_DIM,
            linear_conv_kernel_dim=KERNEL,
            kda_gate_lower_bound=-5.0,
            layernorm_epsilon=1e-5,
            add_bias_linear=False,
            bf16=True,
            params_dtype=dtype,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=False,
            sequence_parallel=False,
        )
        return build_module(
            get_kda_module_spec(config, TESpecProvider()),
            config=config,
            layer_number=1,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
        )

    @staticmethod
    def _copy_weights(megatron_module, hf):
        qkv_dim = HEADS * HEAD_DIM
        with torch.no_grad():
            for megatron_name, hf_param in (
                ("q_proj", hf.q_proj.weight),
                ("k_proj", hf.k_proj.weight),
                ("v_proj", hf.v_proj.weight),
                ("f_a_proj", hf.forget_gate.f_a_proj.weight),
                ("f_b_proj", hf.forget_gate.f_b_proj.weight),
                ("g_a_proj", hf.g_a_proj.weight),
                ("g_b_proj", hf.g_b_proj.weight),
                ("b_proj", hf.b_proj.weight),
                ("o_proj", hf.o_proj.weight),
            ):
                getattr(megatron_module, megatron_name).weight.copy_(hf_param)
            # HF fuses the q/k/v convolutions into one depthwise conv over [q | k | v] channels.
            q_w, k_w, v_w = torch.split(hf.conv1d.weight, [qkv_dim] * 3, dim=0)
            megatron_module.q_conv1d.weight.copy_(q_w)
            megatron_module.k_conv1d.weight.copy_(k_w)
            megatron_module.v_conv1d.weight.copy_(v_w)
            megatron_module.A_log.copy_(hf.forget_gate.A_log)
            megatron_module.dt_bias.copy_(hf.forget_gate.dt_bias)
            megatron_module.o_norm.weight.copy_(hf.o_norm.weight)

    def test_packed_forward_matches_hf(self):
        dtype = torch.bfloat16
        torch.manual_seed(0)
        hf = self._hf_reference(dtype)
        megatron_module = self._megatron_module(dtype)
        self._copy_weights(megatron_module, hf)

        lengths = [37, 50]
        sequences = [
            torch.randn(1, length, HIDDEN, device="cuda", dtype=dtype) for length in lengths
        ]
        with torch.no_grad():
            hf_out = torch.cat([hf(x) for x in sequences], dim=1)[0].float()  # [t, C]
            packed = torch.cat(sequences, dim=1).transpose(0, 1).contiguous()  # [t, 1, C]
            cu_seqlens = torch.tensor(
                [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
            )
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(lengths),
                max_seqlen_kv=max(lengths),
            )
            megatron_out, bias = megatron_module(packed, packed_seq_params=packed_seq_params)

        assert bias is None
        megatron_out = megatron_out[:, 0].float()
        diff = (megatron_out - hf_out).abs()
        reference_scale = max(hf_out.abs().mean().item(), 1e-3)
        assert diff.mean().item() < 0.02 * reference_scale, diff.mean().item()
        assert diff.max().item() < 0.2 * reference_scale + 1e-2, diff.max().item()
