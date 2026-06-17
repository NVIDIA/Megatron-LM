# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    HAVE_FLA,
    HAVE_FLA_GDN2,
    GatedDeltaNet,
    GatedDeltaNet2,
    chunk_gdn2,
    torch_chunk_gated_delta_rule,
    torch_chunk_gdn2,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params


try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)


def _make_gdn_config(**overrides):
    config_kwargs = {
        "hidden_size": 128,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "num_layers": 1,
        "normalization": "RMSNorm",
        "use_cpu_initialization": True,
        "layernorm_zero_centered_gamma": True,
        "num_attention_heads": 8,
        "activation_func": F.silu,
        "bf16": True,
        "experimental_attention_variant": "gated_delta_net",
        "linear_attention_freq": [1],
        "transformer_impl": "transformer_engine",
    }
    config_kwargs.update(overrides)
    return TransformerConfig(**config_kwargs)


def test_gdn_pre_gated_delta_rule_fusion_defaults_to_disabled():
    config = _make_gdn_config()
    assert not config.gdn_pre_gated_delta_rule_fusion


def test_gdn_pre_gated_delta_rule_fusion_accepts_gdn_variant():
    config = _make_gdn_config(gdn_pre_gated_delta_rule_fusion=True)
    assert config.gdn_pre_gated_delta_rule_fusion


def test_gdn_pre_gated_delta_rule_fusion_requires_gdn_variant():
    with pytest.raises(ValueError, match="experimental_attention_variant='gated_delta_net'"):
        _make_gdn_config(
            experimental_attention_variant=None,
            linear_attention_freq=None,
            gdn_pre_gated_delta_rule_fusion=True,
        )


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetTestBase):

    def test_gpu_forward(self):
        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        attention_mask = None

        output, bias = gdn(hidden_states, attention_mask)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert output.shape[0] == seq_length // self.sp_size // self.cp_size, (
            f"Output shape {output.shape[0]=} mismatch with "
            f" {seq_length=} // {self.sp_size=} // {self.cp_size=}."
        )
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == gdn.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {gdn.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"

    def test_gpu_forward_thd_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        if self.use_gdn2:
            # FLA uses different kernels for SBHD and THD:
            # https://github.com/fla-org/flash-linear-attention/blob/ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0/fla/ops/gdn2/chunk_intra.py#L40-L53
            # so we relax the error bound here
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 3e-4, 3e-4

        # Input shape
        sequence_length = 32
        micro_batch_size = 4
        cu_seqlens = [0, 32, 64, 96, 128]
        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size)
        )
        attention_mask_sbhd = None
        hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        attention_mask_thd = None
        packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

        # THD format
        output_thd, _ = self.gdn(
            hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
        )
        # SBHD format
        output_sbhd, _ = self.gdn(hidden_states_sbhd, attention_mask_sbhd)
        output_sbhd_T = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)

        rank = torch.distributed.get_rank()
        assert output_thd.shape[0] == sub_sequence_length * micro_batch_size
        assert output_thd.shape[1] == 1
        assert output_thd.shape[2] == self.gdn.config.hidden_size
        torch.testing.assert_close(
            output_sbhd_T,
            output_thd,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Output mismatch ({rank=}): {msg}",
        )

    def test_deterministic_mode(self):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        det_config = copy.deepcopy(self.transformer_config)
        det_config.deterministic_mode = True

        gdn_spec = get_experimental_attention_variant_module_spec(config=det_config)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        gdn = (
            gdn_spec.module(
                det_config,
                submodules=gdn_spec.submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # deterministic_mode must select the variant's torch-native kernel, not FLA.
        if self.use_gdn2:
            assert isinstance(gdn, GatedDeltaNet2)
            assert gdn.gated_delta_rule is torch_chunk_gdn2
        else:
            assert isinstance(gdn, GatedDeltaNet)
            assert gdn.gated_delta_rule is torch_chunk_gated_delta_rule

        micro_batch_size = 2
        seq_length = 64
        torch.manual_seed(0)
        base_input = torch.randn(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        def run(module):
            hidden_states = base_input.clone().requires_grad_(True)
            output, _ = module(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach().clone()
                for name, param in module.named_parameters()
                if param.grad is not None
            }
            module.zero_grad(set_to_none=True)
            return output.detach().clone(), grads, hidden_states.grad.detach().clone()

        out1, grads1, input_grad1 = run(gdn)
        out2, grads2, input_grad2 = run(gdn)

        rank = torch.distributed.get_rank()
        assert torch.equal(out1, out2), f"Output not reproducible ({rank=})"
        assert torch.equal(input_grad1, input_grad2), f"Input grad not reproducible ({rank=})"
        assert set(grads1.keys()) == set(grads2.keys())
        for name in grads1:
            assert torch.equal(
                grads1[name], grads2[name]
            ), f"Grad not reproducible for {name} ({rank=})"

    def test_module_construction(self):
        gdn = self.gdn
        if self.use_gdn2:
            assert isinstance(gdn, GatedDeltaNet2)
            assert gdn.gated_delta_rule is chunk_gdn2
            assert gdn.in_proj_dim == 4 * gdn.qk_dim + 3 * gdn.v_dim
            assert gdn.A_log.shape == (gdn.num_key_heads // self.tp_size,)
            assert gdn.dt_bias.shape == (gdn.qk_dim // self.tp_size,)
        else:
            assert isinstance(gdn, GatedDeltaNet)
            assert gdn.in_proj_dim == 2 * gdn.qk_dim + 2 * gdn.v_dim + 2 * gdn.num_value_heads
            assert gdn.A_log.shape == (gdn.num_value_heads // self.tp_size,)
            assert gdn.dt_bias.shape == (gdn.num_value_heads // self.tp_size,)

    def test_inference_state_shapes(self):
        if self.use_gdn2:
            pytest.skip("GDN2 inference is not supported.")
        assert self.gdn.mamba_state_shapes_per_request() == (
            (self.gdn.conv_dim_local_tp, self.gdn.conv_kernel_dim),
            (self.gdn.num_v_heads_local_tp, self.gdn.key_head_dim, self.gdn.value_head_dim),
        )

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        device = torch.cuda.current_device()
        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // gdn.cp_size
        num_k_heads_local = gdn.num_key_heads // gdn.tp_size // gdn.cp_size
        qk_dim_local = gdn.qk_dim_local_tp // gdn.cp_size
        v_dim_local = gdn.v_dim_local_tp // gdn.cp_size

        qkv = torch.randn(
            batch, seq_len, 2 * qk_dim_local + v_dim_local, device=device, dtype=torch.bfloat16
        )
        gate = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn.value_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        if self.use_gdn2:
            gate_feats = (
                torch.randn(batch, seq_len, qk_dim_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, qk_dim_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, v_dim_local, device=device, dtype=torch.bfloat16),
            )  # f, b, w
            A_log_mock = torch.randn(num_k_heads_local, device=device, dtype=torch.bfloat16)
            dt_bias_mock = torch.randn(qk_dim_local, device=device, dtype=torch.bfloat16)
            expected_keys = {"q", "k", "v", "g", "b", "w"}
        else:
            gate_feats = (
                torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
            )  # beta, alpha
            A_log_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)
            dt_bias_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)
            expected_keys = {"q", "k", "v", "g", "beta"}

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        with torch._dynamo.config.patch(disable=True):
            kernel_inputs = gdn._prepare_input_for_gated_delta_rule(
                qkv, gate, A_log_mock, dt_bias_mock, batch, seq_len, *gate_feats
            )

        # The output gate (z) rides along under "gate" and is popped by forward before
        # the kernel call; everything else is passed straight through as kernel kwargs.
        gate_out = kernel_inputs.pop("gate")
        assert set(kernel_inputs) == expected_keys

        query, key, value, g = (kernel_inputs[k] for k in ("q", "k", "v", "g"))
        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        assert gate_out.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        for t in (query, key, value, gate_out, *kernel_inputs.values()):
            assert t.is_contiguous()

        if self.use_gdn2:
            # Per-channel decay and erase/write gates squashed to [0, 1]
            b, w = kernel_inputs["b"], kernel_inputs["w"]
            assert g.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
            assert b.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
            assert w.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
            assert (g <= 0).all()
            assert (b >= 0).all() and (b <= 1).all()
            assert (w >= 0).all() and (w <= 1).all()
        else:
            # Per-head decay and write strength beta
            beta = kernel_inputs["beta"]
            assert g.shape == (batch, seq_len, num_v_heads_local)
            assert beta.shape == (batch, seq_len, num_v_heads_local)
            assert (g <= 0).all()
            assert (beta >= 0).all() and (beta <= 1).all()


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not HAVE_FUSED_PRE_GDR, reason="causal-conv1d fused backward is not installed.")
@pytest.mark.internal
class TestFusedPreGatedDeltaRule:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        self.pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        self.fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False
        )
        self.fused_gdn.load_state_dict(self.unfused_gdn.state_dict())

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _build_gdn(
        self,
        gdn_pre_gated_delta_rule_fusion: bool,
        *,
        deterministic_mode: bool = True,
        conv_kernel_dim: int = 2,
    ):
        transformer_config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=conv_kernel_dim,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
            deterministic_mode=deterministic_mode,
            gdn_pre_gated_delta_rule_fusion=gdn_pre_gated_delta_rule_fusion,
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=transformer_config
        ).submodules
        gdn = GatedDeltaNet(
            transformer_config,
            submodules=gdn_submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=self.pg_collection,
        )
        return gdn.cuda().bfloat16()

    def _packed_pre_gated_delta_rule_reference(self, gdn, qkvzba, cu_seqlens):
        """Run the unfused pre-GDR path independently on each packed sequence."""

        segment_outputs = [[] for _ in range(6)]
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            outputs = gdn.pre_gated_delta_rule(
                qkvzba[start:end],
                batch=1,
                seq_len=end - start,
                cp_size=gdn.cp_size,
                cp_group=gdn.pg_collection.cp,
            )
            for output_list, output in zip(segment_outputs, outputs):
                output_list.append(output)
        return tuple(torch.cat(outputs, dim=1) for outputs in segment_outputs)

    def _assert_pre_gated_delta_rule_outputs_close(
        self, fused_outputs, unfused_outputs, *, atol: float, rtol: float, output_tolerances=None
    ):
        """Compare named pre-GDR outputs with optional per-output tolerances."""

        output_names = ("query", "key", "value", "gate", "beta", "g")
        output_tolerances = output_tolerances or {}
        for name, fused, unfused in zip(output_names, fused_outputs, unfused_outputs):
            output_atol, output_rtol = output_tolerances.get(name, (atol, rtol))
            torch.testing.assert_close(
                fused,
                unfused,
                atol=output_atol,
                rtol=output_rtol,
                msg=lambda msg, output_name=name: f"{output_name} mismatch: {msg}",
            )

    def test_fused_and_unfused_forward_match(self):
        hidden_states = torch.randn(
            (32, 2, self.unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            unfused_output, unfused_bias = self.unfused_gdn(hidden_states, None)
            fused_output, fused_bias = self.fused_gdn(hidden_states, None)

        torch.testing.assert_close(fused_output, unfused_output, atol=1e-3, rtol=1e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_forward_thd_match(self):
        unfused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(unfused_gdn.state_dict())

        hidden_states = torch.randn(
            (32, 1, unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        cu_seqlens = torch.tensor(
            [0, 1, 4, 11, 32], device=torch.cuda.current_device(), dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=21,
            max_seqlen_kv=21,
            total_tokens=hidden_states.shape[0],
        )
        assert packed_seq_params.seq_idx is not None

        with torch.no_grad():
            unfused_output, unfused_bias = unfused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )
            fused_output, fused_bias = fused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )

        torch.testing.assert_close(fused_output, unfused_output, atol=2e-3, rtol=2e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_forward_thd_padding_match(self):
        unfused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(unfused_gdn.state_dict())

        hidden_states = torch.randn(
            (12, 1, unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        cu_seqlens = torch.tensor(
            [0, 1, 4, 9], device=torch.cuda.current_device(), dtype=torch.int32
        )
        cu_seqlens_padded = torch.tensor(
            [0, 2, 6, 12], device=torch.cuda.current_device(), dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=6,
            max_seqlen_kv=6,
            total_tokens=hidden_states.shape[0],
        )
        assert packed_seq_params.seq_idx is not None

        with torch.no_grad():
            unfused_output, unfused_bias = unfused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )
            fused_output, fused_bias = fused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )

        torch.testing.assert_close(fused_output, unfused_output, atol=2e-3, rtol=2e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_pre_gated_delta_rule_match(self):
        batch = 2
        seq_len = 32
        hidden_states = torch.randn(
            (seq_len, batch, self.unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            qkvzba, _ = self.unfused_gdn.in_proj(hidden_states)
            unfused_outputs = self.unfused_gdn.pre_gated_delta_rule(
                qkvzba, batch, seq_len, self.unfused_gdn.cp_size, self.unfused_gdn.pg_collection.cp
            )
            fused_outputs = self.fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba)

        self._assert_pre_gated_delta_rule_outputs_close(
            fused_outputs,
            unfused_outputs,
            atol=1e-3,
            rtol=1e-3,
            output_tolerances={"g": (1e-3, 3e-3)},
        )

    def test_fused_and_unfused_pre_gated_delta_rule_backward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 2
        seq_len = 32
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        qkvzba_unfused = qkvzba.detach().clone().requires_grad_(True)
        qkvzba_fused = qkvzba.detach().clone().requires_grad_(True)

        reference_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)

        unfused_outputs = reference_gdn.pre_gated_delta_rule(
            qkvzba_unfused, batch, seq_len, reference_gdn.cp_size, reference_gdn.pg_collection.cp
        )
        fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba_fused)
        grad_outputs = [torch.randn_like(output.float()) for output in unfused_outputs]

        unfused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(unfused_outputs, grad_outputs)
        )
        fused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(fused_outputs, grad_outputs)
        )
        unfused_loss.backward()
        fused_loss.backward()

        torch.testing.assert_close(qkvzba_fused.grad, qkvzba_unfused.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            fused_gdn.conv1d.weight.grad, reference_gdn.conv1d.weight.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.A_log.grad, reference_gdn.A_log.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.dt_bias.grad, reference_gdn.dt_bias.grad, atol=3e-2, rtol=3e-2
        )

    def test_fused_and_unfused_packed_pre_gated_delta_rule_forward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 1
        cu_seqlens = torch.tensor(
            [0, 1, 4, 6, 11], device=torch.cuda.current_device(), dtype=torch.int32
        )
        seq_len = cu_seqlens[-1].item()
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            unfused_outputs = self._packed_pre_gated_delta_rule_reference(
                reference_gdn, qkvzba, cu_seqlens
            )
            fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(
                qkvzba, cu_seqlens_q=cu_seqlens
            )

        self._assert_pre_gated_delta_rule_outputs_close(
            fused_outputs, unfused_outputs, atol=2e-3, rtol=2e-3
        )

    def test_fused_and_unfused_packed_pre_gated_delta_rule_backward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 1
        cu_seqlens = torch.tensor(
            [0, 1, 4, 6, 11], device=torch.cuda.current_device(), dtype=torch.int32
        )
        seq_len = cu_seqlens[-1].item()
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        qkvzba_unfused = qkvzba.detach().clone().requires_grad_(True)
        qkvzba_fused = qkvzba.detach().clone().requires_grad_(True)

        reference_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)

        unfused_outputs = self._packed_pre_gated_delta_rule_reference(
            reference_gdn, qkvzba_unfused, cu_seqlens
        )
        fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(
            qkvzba_fused, cu_seqlens_q=cu_seqlens
        )
        grad_outputs = [torch.randn_like(output.float()) for output in unfused_outputs]

        unfused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(unfused_outputs, grad_outputs)
        )
        fused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(fused_outputs, grad_outputs)
        )
        unfused_loss.backward()
        fused_loss.backward()

        torch.testing.assert_close(qkvzba_fused.grad, qkvzba_unfused.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            fused_gdn.conv1d.weight.grad, reference_gdn.conv1d.weight.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.A_log.grad, reference_gdn.A_log.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.dt_bias.grad, reference_gdn.dt_bias.grad, atol=3e-2, rtol=3e-2
        )

    def test_fused_packed_conv_forward_boundary_isolation(self):
        from megatron.core.fusions.fused_pre_gated_delta_rule import (
            fused_streamed_pre_gated_delta_rule,
        )

        seq_len = 5
        boundary = 3
        num_key_heads = 1
        num_value_heads = 4
        key_head_dim = 32
        value_head_dim = 32
        conv_width = 4
        qk_channels = num_key_heads * key_head_dim
        v_channels = num_value_heads * value_head_dim
        k_offset = qk_channels
        v_offset = 2 * qk_channels
        total_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
        device = torch.cuda.current_device()

        qkvzba = torch.zeros((seq_len, 1, total_channels), device=device, dtype=torch.bfloat16)
        qkvzba[boundary - 1, 0, :qk_channels] = 10.0
        qkvzba[boundary - 1, 0, k_offset : k_offset + qk_channels] = 10.0
        qkvzba[boundary - 1, 0, v_offset : v_offset + v_channels] = 10.0
        conv_weight = torch.zeros((2 * qk_channels + v_channels, 1, conv_width), device=device)
        conv_weight[:qk_channels, 0, conv_width - 2] = 1.0
        conv_weight[k_offset : k_offset + qk_channels, 0, conv_width - 2] = 1.0
        conv_weight[v_offset : v_offset + v_channels, 0, conv_width - 2] = 1.0
        A_log = torch.zeros((num_value_heads,), device=device, dtype=torch.bfloat16)
        dt_bias = torch.zeros((num_value_heads,), device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, boundary, seq_len], device=device, dtype=torch.int32)

        query, key, value, _, _, _ = fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv_weight.to(torch.bfloat16),
            None,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            cu_seqlens=cu_seqlens,
        )

        torch.testing.assert_close(
            query[0, boundary], torch.zeros_like(query[0, boundary]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            key[0, boundary], torch.zeros_like(key[0, boundary]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            value[0, boundary], torch.zeros_like(value[0, boundary]), atol=0.0, rtol=0.0
        )

    def test_fused_packed_conv_backward_boundary_isolation(self):
        from megatron.core.fusions.fused_pre_gated_delta_rule import (
            fused_streamed_pre_gated_delta_rule,
        )

        seq_len = 5
        boundary = 3
        num_key_heads = 1
        num_value_heads = 4
        key_head_dim = 32
        value_head_dim = 32
        conv_width = 4
        qk_channels = num_key_heads * key_head_dim
        v_channels = num_value_heads * value_head_dim
        k_offset = qk_channels
        v_offset = 2 * qk_channels
        total_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
        device = torch.cuda.current_device()

        qkvzba = torch.zeros(
            (seq_len, 1, total_channels), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        conv_weight = torch.zeros(
            (2 * qk_channels + v_channels, 1, conv_width),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with torch.no_grad():
            qkvzba[boundary - 1, 0, :qk_channels] = 10.0
            qkvzba[boundary - 1, 0, k_offset : k_offset + qk_channels] = 10.0
            qkvzba[boundary - 1, 0, v_offset : v_offset + v_channels] = 10.0
            conv_weight[:qk_channels, 0, conv_width - 2] = 1.0
            conv_weight[k_offset : k_offset + qk_channels, 0, conv_width - 2] = 1.0
            conv_weight[v_offset : v_offset + v_channels, 0, conv_width - 2] = 1.0
        A_log = torch.zeros(
            (num_value_heads,), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        dt_bias = torch.zeros(
            (num_value_heads,), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        cu_seqlens = torch.tensor([0, boundary, seq_len], device=device, dtype=torch.int32)

        query, key, value, gate, beta, g = fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv_weight,
            None,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            cu_seqlens=cu_seqlens,
        )

        loss = (
            query[0, boundary].float().sum()
            + key[0, boundary].float().sum()
            + value[0, boundary].float().sum()
        )
        loss = loss + 0.0 * (gate.float().sum() + beta.float().sum() + g.float().sum())
        loss.backward()

        leaked_q_grad = qkvzba.grad[boundary - 1, 0, :qk_channels]
        leaked_k_grad = qkvzba.grad[boundary - 1, 0, k_offset : k_offset + qk_channels]
        leaked_v_grad = qkvzba.grad[boundary - 1, 0, v_offset : v_offset + v_channels]
        torch.testing.assert_close(
            leaked_q_grad, torch.zeros_like(leaked_q_grad), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            leaked_k_grad, torch.zeros_like(leaked_k_grad), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            leaked_v_grad, torch.zeros_like(leaked_v_grad), atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGDNCuSeqlensResolve:

    @pytest.fixture
    def mock_gdn(self):
        class MockGDN:
            _resolve_cu_seqlens = GatedDeltaNet._resolve_cu_seqlens

        return MockGDN()

    def test_padded_preferred_when_available(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q", cp_size=2)
        assert torch.equal(result, padded)

    def test_actual_used_when_no_padding(self, mock_gdn):
        actual = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)
        assert torch.equal(result, actual)

    def test_raises_when_padding_mismatch(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_raises_when_padded_mismatches_total(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1004], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_raises_when_not_divisible_by_cp_size(self, mock_gdn):
        actual = torch.tensor([0, 505, 1008], dtype=torch.int32)
        with pytest.raises(ValueError, match="must be divisible by cp_size"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_cp1_still_validates_total(self, mock_gdn):
        mock_gdn.cp_size = 1
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=1)
