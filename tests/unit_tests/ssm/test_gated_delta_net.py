# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
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
from megatron.core.ssm.gated_delta_net.common import l2norm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
def test_torch_gdn_l2norm_flag_matches_explicit_normalization(use_gdn2):
    """The torch fallback flag must match explicit q/k normalization."""
    torch.manual_seed(123)
    torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())
    device = torch.cuda.current_device()
    batch_size, sequence_length, num_heads, head_dim = 1, 8, 2, 16
    shape = (batch_size, sequence_length, num_heads, head_dim)
    q = torch.randn(shape, device=device)
    k = torch.randn(shape, device=device)
    v = torch.randn(shape, device=device)

    if use_gdn2:
        kernel = torch_chunk_gdn2
        kernel_kwargs = {
            "g": torch.full(shape, -0.01, device=device),
            "b": torch.full(shape, 0.1, device=device),
            "w": torch.full(shape, 0.1, device=device),
        }
    else:
        kernel = torch_chunk_gated_delta_rule
        gate_shape = (batch_size, sequence_length, num_heads)
        kernel_kwargs = {
            "g": torch.full(gate_shape, -0.01, device=device),
            "beta": torch.full(gate_shape, 0.1, device=device),
        }

    actual_output, actual_state = kernel(
        q=q,
        k=k,
        v=v,
        chunk_size=8,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        **kernel_kwargs,
    )
    expected_output, expected_state = kernel(
        q=l2norm(q),
        k=l2norm(k),
        v=v,
        chunk_size=8,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        **kernel_kwargs,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
def test_gdn_in_kernel_l2norm_matches_caller_end_to_end():
    """GDN's BF16 in-kernel route must match the former caller-normalized route."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)
        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=4,
            num_query_groups=2,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            context_parallel_size=1,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_spec = get_experimental_attention_variant_module_spec(config=config)

        def build_gdn():
            return (
                gdn_spec.module(
                    config,
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

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        kernel_norm_gdn = build_gdn()
        caller_norm_gdn = build_gdn()
        caller_norm_gdn.load_state_dict(kernel_norm_gdn.state_dict())

        def install_route_probe(module, caller_normalizes):
            original_prepare = module._prepare_input_for_gated_delta_rule
            original_kernel = module.gated_delta_rule
            observed = {}

            def wrapped_prepare(
                qkv,
                gate,
                A_log_local_cp,
                dt_bias_local_cp,
                batch,
                seq_len,
                *gate_feats,
                use_qk_l2norm_in_kernel=False,
            ):
                observed["prepare_flag"] = use_qk_l2norm_in_kernel
                effective_flag = False if caller_normalizes else use_qk_l2norm_in_kernel
                kernel_inputs = original_prepare(
                    qkv,
                    gate,
                    A_log_local_cp,
                    dt_bias_local_cp,
                    batch,
                    seq_len,
                    *gate_feats,
                    use_qk_l2norm_in_kernel=effective_flag,
                )

                # The new route must pass the raw post-convolution q/k to FLA. This
                # explicit check catches accidental caller normalization (and hence
                # double normalization) even when BF16 rounding hides the difference.
                if not caller_normalizes:
                    query_key, _ = torch.split(
                        qkv.detach(),
                        [
                            2 * module.qk_dim_local_tp // module.cp_size,
                            module.v_dim_local_tp // module.cp_size,
                        ],
                        dim=-1,
                    )
                    query_key = query_key.reshape(batch, seq_len, -1, module.key_head_dim)
                    split_size = module.qk_dim_local_tp // module.key_head_dim // module.cp_size
                    raw_query, raw_key = torch.split(query_key, [split_size, split_size], dim=2)
                    repeat_factor = module.num_value_heads // module.num_key_heads
                    if repeat_factor > 1:
                        raw_query = raw_query.repeat_interleave(repeat_factor, dim=2)
                        raw_key = raw_key.repeat_interleave(repeat_factor, dim=2)
                    observed["prepare_kept_qk_raw"] = torch.equal(
                        kernel_inputs["q"].detach(), raw_query.contiguous()
                    ) and torch.equal(kernel_inputs["k"].detach(), raw_key.contiguous())
                return kernel_inputs

            def wrapped_kernel(*args, **kwargs):
                observed["kernel_flag"] = kwargs.get("use_qk_l2norm_in_kernel", False)
                if caller_normalizes:
                    kwargs = {**kwargs, "use_qk_l2norm_in_kernel": False}
                return original_kernel(*args, **kwargs)

            module._prepare_input_for_gated_delta_rule = wrapped_prepare
            module.gated_delta_rule = wrapped_kernel
            return observed

        kernel_route = install_route_probe(kernel_norm_gdn, caller_normalizes=False)
        caller_route = install_route_probe(caller_norm_gdn, caller_normalizes=True)

        torch.manual_seed(123)
        hidden_states = torch.randn(
            (16, 2, config.hidden_size), device=torch.cuda.current_device(), dtype=torch.bfloat16
        )

        def run(module):
            module.zero_grad(set_to_none=True)
            module_input = hidden_states.detach().clone().requires_grad_(True)
            output, _ = module(module_input, None)
            output.float().square().mean().backward()
            parameter_grads = {
                name: parameter.grad.detach().clone()
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            }
            return output.detach(), module_input.grad.detach().clone(), parameter_grads

        kernel_output, kernel_input_grad, kernel_parameter_grads = run(kernel_norm_gdn)
        caller_output, caller_input_grad, caller_parameter_grads = run(caller_norm_gdn)

        assert kernel_route["prepare_flag"] is True
        assert kernel_route["kernel_flag"] is True
        assert kernel_route["prepare_kept_qk_raw"] is True
        # The compatibility route still goes through GatedDeltaNet.forward, then
        # overrides only where normalization is performed to model the former code.
        assert caller_route["prepare_flag"] is True
        assert caller_route["kernel_flag"] is True

        atol = rtol = 2e-2
        torch.testing.assert_close(
            kernel_output.float(), caller_output.float(), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            kernel_input_grad.float(), caller_input_grad.float(), atol=atol, rtol=rtol
        )
        assert set(kernel_parameter_grads) == set(caller_parameter_grads)
        for name in kernel_parameter_grads:
            torch.testing.assert_close(
                kernel_parameter_grads[name].float(),
                caller_parameter_grads[name].float(),
                atol=atol,
                rtol=rtol,
                msg=lambda msg, name=name: f"Parameter grad mismatch for {name}: {msg}",
            )
    finally:
        Utils.destroy_model_parallel()


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
