# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy

import einops
import pytest
import torch
from packaging import version
from torch.nn import functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention_context_parallel import (
    AttentionFuncionWithContextParallel,
    to_zz_mask_attn_bias,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize(
    ("transformer_impl", "fallback_to_eager_attn"),
    [("transformer_engine", False), ("transformer_engine", True), ("native", False)],
)
class TestParallelAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate, transformer_impl, fallback_to_eager_attn):
        if output_gate:
            if transformer_impl == "native":
                pytest.skip("Native implementation does not support output gate.")
            if fallback_to_eager_attn:
                pytest.skip("No need to test output gate for fallback_to_eager_attn = True.")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_output_gate=output_gate,
            transformer_impl=transformer_impl,
            fallback_to_eager_attn=fallback_to_eager_attn,
        )
        if transformer_impl == "transformer_engine":
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                fallback_to_eager_attn=fallback_to_eager_attn
            )
        else:
            layer_spec = get_gpt_layer_local_spec()
        attn_layer_spec = layer_spec.submodules.self_attention.submodules
        self.parallel_attention = SelfAttention(
            self.transformer_config, attn_layer_spec, layer_number=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])

        hidden_size = self.transformer_config.hidden_size
        standard_num_weights = (
            hidden_size * hidden_size * 4 + hidden_size * 4  # QKVO weight  # QKVO bias
        )
        if self.transformer_config.attention_output_gate:
            standard_num_weights += hidden_size * hidden_size + hidden_size  # Gate weight and bias
        if self.transformer_config.transformer_impl == "transformer_engine":
            standard_num_weights += hidden_size * 2  # fused pre layernorm weight and bias

        assert (
            num_weights == standard_num_weights
        ), f"{num_weights=} does not match {standard_num_weights=}."

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = self.parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not is_te_min_version("1.4.0"), reason="Fused RoPE requires TE >= 1.4.0")
    @pytest.mark.parametrize("rotary_interleaved", [True, False])
    @pytest.mark.parametrize("fused_qkv_rope", [True, False])
    def test_fused_rope_gpu_forward(self, rotary_interleaved, fused_qkv_rope):
        if self.transformer_config.fallback_to_eager_attn:
            pytest.skip("No need to test fused RoPE for fallback_to_eager_attn = True.")
        self.parallel_attention.config.apply_rope_fusion = True
        if rotary_interleaved and not is_te_min_version("2.3.0"):
            pytest.skip("Only TE >= 2.3.0 supports interleaved fused RoPE.")
        if fused_qkv_rope and self.parallel_attention.config.attention_output_gate:
            pytest.skip("Fused QKV RoPE does not support gated attention for now.")
        if fused_qkv_rope and not HAVE_FUSED_QKV_ROPE:
            pytest.skip("Fused QKV RoPE not available.")
        self.parallel_attention.config.rotary_interleaved = rotary_interleaved
        self.parallel_attention.config.fused_single_qkv_rope = fused_qkv_rope
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False
        self.parallel_attention.config.rotary_interleaved = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert "core_attn" in config.recompute_modules
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "native"])
class TestSelfAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate, transformer_impl):
        if transformer_impl == "native":
            if output_gate:
                pytest.skip("Native implementation does not support output gate.")
        self.transformer_impl = transformer_impl
        self.output_gate = output_gate
        Utils.destroy_model_parallel()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def run_self_attention(self, pg_collection):
        tensor_model_parallel_size = torch.distributed.get_world_size(pg_collection.tp)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            attention_output_gate=self.output_gate,
            tensor_model_parallel_size=tensor_model_parallel_size,
            use_cpu_initialization=False,
            transformer_impl=self.transformer_impl,
        )
        if self.transformer_impl == "transformer_engine":
            get_gpt_layer_spec_fn = get_gpt_layer_with_transformer_engine_spec
        else:
            get_gpt_layer_spec_fn = get_gpt_layer_local_spec
        self.self_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_spec_fn().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )

        config = self.self_attention.config
        sequence_length = 127
        micro_batch_size = 2

        self.self_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.self_attention.config.hidden_size),
            device='cuda',
        )
        hidden_states_ref = copy.deepcopy(hidden_states)

        output, bias = self.self_attention(hidden_states, None)
        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    def test_self_attention_mpu(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.internal
    def test_self_attention_independent_pg_smoke(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimensions cp, tp (reversed from device mesh order)
        grid = HyperCommGrid([cp_size, tp_size], ["cp", "tp"])

        # Get TP and CP process groups from HyperCommGrid
        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)


def _torch_native_attention(query, key, value, attention_mask, sinks, scaling: float):
    """Torch native attention implementation
    This was not in the original implementation and slightly affect results;
    it prevents overflow in BF16/FP16 when training with batch size > 1 we clamp max values.
    """
    # Rearrange query, key, value to (b, h, s, d)
    query = einops.rearrange(query, 's b h d -> b h s d')
    key = einops.rearrange(key, 's b h d -> b h s d')
    value = einops.rearrange(value, 's b h d -> b h s d')

    # Compute attention weights
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        nheads = query.shape[1]
        nheads_k = key.shape[1]
        heads_k_stride = 1
        mask_bias = to_zz_mask_attn_bias(
            attention_mask, 1, nheads, nheads_k, heads_k_stride, query.device, query.dtype
        )
        attn_weights = attn_weights + mask_bias

    # Add sinks to attention weights
    if sinks is None:
        combined_logits = attn_weights
    else:
        sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # Compute attention scores
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    if sinks is None:
        scores = probs
    else:
        scores = probs[..., :-1]

    # Compute attention output
    attn_output = torch.matmul(scores, value)
    attn_output = einops.rearrange(attn_output, 'b h s d -> s b h d')
    attn_output = attn_output.contiguous()
    return attn_output


def test_eager_attention_function():
    # Configuration
    batch_size = 4
    num_heads = 2
    head_dim = 256
    seq_len_q = 512
    seq_len_k = 2048
    scale = 1 / (head_dim**2)

    # Initialize inputs
    q = torch.rand(
        (seq_len_q, batch_size, num_heads, head_dim),
        device='cuda',
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.rand(
        (seq_len_k, batch_size, num_heads, head_dim),
        device='cuda',
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.rand(
        (seq_len_k, batch_size, num_heads, head_dim),
        device='cuda',
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def randbool(shape, **kwargs):
        return torch.randn(shape, **kwargs) > 0

    attn_bias = randbool((batch_size, 1, seq_len_q, seq_len_k), device='cuda')
    sinks = None

    # Torch native attention forward and backward pass
    out_torch = _torch_native_attention(
        query=q, key=k, value=v, attention_mask=attn_bias, sinks=sinks, scaling=scale
    )
    loss_torch = out_torch.sum()
    loss_torch.backward()
    torch_q_grad = q.grad.clone()
    torch_k_grad = k.grad.clone()
    torch_v_grad = v.grad.clone()
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    if sinks is not None:
        torch_sinks_grad = sinks.grad.clone()
        sinks.grad.zero_()
    else:
        torch_sinks_grad = None

    # Custom attention forward and backward pass
    out_custom = AttentionFuncionWithContextParallel.apply(
        q, k, v, attn_bias, 0.0, scale, None  # dropout
    )
    loss_custom = out_custom.sum()
    loss_custom.backward()
    custom_q_grad = q.grad.clone()
    custom_k_grad = k.grad.clone()
    custom_v_grad = v.grad.clone()
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    if sinks is not None:
        custom_sinks_grad = sinks.grad.clone()
        sinks.grad.zero_()
    else:
        custom_sinks_grad = None

    # Check attention output and gradients
    assert torch.equal(out_custom, out_torch), "Mismatch in attention output"
    tol = {"atol": 1e-4, "rtol": 1e-4}
    for tensor_name, tensor_torch, tensor_custom in [
        ("q_grad", torch_q_grad, custom_q_grad),
        ("k_grad", torch_k_grad, custom_k_grad),
        ("v_grad", torch_v_grad, custom_v_grad),
        ("sinks_grad", torch_sinks_grad, custom_sinks_grad),
    ]:
        if (tensor_torch is not None) and (tensor_custom is not None):
            torch.testing.assert_close(
                out_custom, out_torch, **tol, msg=lambda msg: f"Mismatch in {tensor_name}: {msg}"
            )
