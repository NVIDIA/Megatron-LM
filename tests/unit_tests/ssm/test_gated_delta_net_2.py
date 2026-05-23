# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net_2 import (
    HAVE_FLA,
    HAVE_GDN2_KERNEL,
    GatedDeltaNet2,
    GatedDeltaNet2Submodules,
    chunk_gdn2,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params

HAVE_GDN2 = HAVE_FLA and HAVE_GDN2_KERNEL


def _make_gdn2(tp_size=1, cp_size=1, sp=False, allow_neg_eigval=False):
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

    config = TransformerConfig(
        hidden_size=256,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=False,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sp,
        context_parallel_size=cp_size,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    backend = LocalSpecProvider()
    submodules = GatedDeltaNet2Submodules(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )

    module = GatedDeltaNet2(
        config,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
        allow_neg_eigval=allow_neg_eigval,
    )
    return module.cuda().bfloat16()


@pytest.mark.skipif(not HAVE_GDN2, reason="A GDN2 chunk_gdn2 kernel is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet2:

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def setup_model_parallel(self, tp_size=1, cp_size=1):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)

    def test_gpu_forward_shape(self):
        self.setup_model_parallel()
        gdn2 = _make_gdn2()

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length, micro_batch_size, gdn2.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        output, bias = gdn2(hidden_states, None)

        assert bias is None
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype

    def test_helper_shapes(self):
        self.setup_model_parallel()
        gdn2 = _make_gdn2()

        batch = 2
        seq_len = 16
        num_v_heads_local = gdn2.num_value_heads // gdn2.tp_size // gdn2.cp_size
        num_k_heads_local = gdn2.num_key_heads // gdn2.tp_size // gdn2.cp_size

        qkv = torch.randn(
            batch,
            seq_len,
            (2 * gdn2.qk_dim_local_tp + gdn2.v_dim_local_tp) // gdn2.cp_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        out_gate = torch.randn(
            batch,
            seq_len,
            gdn2.v_dim_local_tp // gdn2.cp_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        b = torch.randn(
            batch,
            seq_len,
            gdn2.qk_dim_local_tp // gdn2.cp_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        w = torch.randn_like(out_gate)
        alpha = torch.randn_like(b)

        query, key, value, out_gate, b, w, alpha = gdn2._prepare_qkv_for_gated_delta_rule_2(
            qkv, out_gate, b, w, alpha, batch, seq_len
        )

        assert query.shape == (batch, seq_len, num_v_heads_local, gdn2.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn2.key_head_dim)
        assert alpha.shape == (batch, seq_len, num_v_heads_local, gdn2.key_head_dim)
        assert b.shape == (batch, seq_len, num_v_heads_local, gdn2.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn2.value_head_dim)
        assert w.shape == (batch, seq_len, num_v_heads_local, gdn2.value_head_dim)
        assert out_gate.shape == (batch, seq_len, num_v_heads_local, gdn2.value_head_dim)
        assert num_v_heads_local == num_k_heads_local * (
            gdn2.num_value_heads // gdn2.num_key_heads
        )
        assert query.is_contiguous()
        assert key.is_contiguous()
        assert value.is_contiguous()

    @pytest.mark.parametrize("allow_neg_eigval", [False, True])
    def test_compute_g_b_w_bounds(self, allow_neg_eigval):
        self.setup_model_parallel()
        gdn2 = _make_gdn2(allow_neg_eigval=allow_neg_eigval)

        batch = 2
        seq_len = 8
        num_v_heads_local = gdn2.num_value_heads // gdn2.tp_size // gdn2.cp_size
        num_k_heads_local = gdn2.num_key_heads // gdn2.tp_size // gdn2.cp_size
        alpha = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn2.key_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        b = torch.randn_like(alpha)
        w = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn2.value_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        A_log = torch.randn(num_k_heads_local, device=torch.cuda.current_device(), dtype=torch.bfloat16)
        dt_bias = torch.randn(
            num_k_heads_local * gdn2.key_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        g, b, w = gdn2._compute_g_b_w(A_log, dt_bias, alpha, b, w)

        assert g.dtype == torch.float32
        assert g.shape == alpha.shape
        assert b.shape == alpha.shape
        assert w.shape[-1] == gdn2.value_head_dim
        assert torch.all(b >= 0)
        assert torch.all(b <= (2 if allow_neg_eigval else 1))
        assert torch.all(w >= 0)
        assert torch.all(w <= 1)

    def test_gpu_forward_thd_correctness(self):
        self.setup_model_parallel()
        gdn2 = _make_gdn2()
        atol, rtol = 3e-4, 3e-4
        sequence_length = 32
        micro_batch_size = 4
        cu_seqlens = [0, 32, 64, 96, 128]

        hidden_states_sbhd = torch.rand(
            (sequence_length, micro_batch_size, gdn2.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, gdn2.config.hidden_size)
        packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

        output_thd, _ = gdn2(hidden_states_thd, None, packed_seq_params=packed_seq_params)
        output_sbhd, _ = gdn2(hidden_states_sbhd, None)
        output_sbhd_t = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)

        torch.testing.assert_close(output_sbhd_t, output_thd, atol=atol, rtol=rtol)


@pytest.mark.skipif(not HAVE_GDN2, reason="A GDN2 chunk_gdn2 kernel is not installed.")
def test_chunk_gdn2_matches_tiny_reference():
    torch.manual_seed(123)
    device = torch.cuda.current_device()
    q = torch.randn(1, 4, 2, 4, device=device, dtype=torch.float32)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 4, 2, 4, device=device, dtype=torch.float32)
    g = -torch.rand_like(q)
    b = torch.rand_like(q)
    w = torch.rand_like(v)

    actual, _ = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=1.0,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    expected = torch.zeros_like(actual)
    state = torch.zeros(2, 4, 4, device=device, dtype=torch.float32)
    for t in range(q.shape[1]):
        for h in range(q.shape[2]):
            state_bar = torch.exp(g[0, t, h]).unsqueeze(-1) * state[h]
            erase = b[0, t, h] * k[0, t, h]
            write = w[0, t, h] * v[0, t, h]
            state[h] = state_bar + torch.outer(k[0, t, h], write - state_bar.transpose(0, 1) @ erase)
            expected[0, t, h] = state[h].transpose(0, 1) @ q[0, t, h]

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
