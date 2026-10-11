# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Default absorbed DSA against a PyTorch-native attention and output-gate reference.

The gate placement and projection follow Tencent's designated Transformers implementation:
https://github.com/huggingface/transformers/blob/fa40619176b63effb726ab0d0000925327cad7db/src/transformers/models/hy_v4/modeling_hy_v4.py
Tencent's checkpoint/finetuning reference is pinned at:
https://huggingface.co/tencent/Hy4-preview/tree/705d81ee51566a186d645b74c974d642ef2828fe

The existing NativeDSA implements the independent PyTorch absorbed attention and
indexer reference. This test adds a bias-free gate from the attention input and
applies it after V up projection, before the output projection. The independent
reference computes sigmoid in FP32 and casts its result to the attention dtype,
following MCore's default MLA precision policy. The attention parity cases
disable indexer auxiliary loss to isolate the attention/output-gate gradients.
Deferred-gradient cases exercise only the new real TE gate projection: main's
complete absorbed DSA deferred-gradient path is outside this feature's scope.
Activation rotation is disabled through its supported config, with no
production-path mocks.
"""

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAttention,
)
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention, MLASelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import (
    _missing_cudnn_dsa_kernel_dependencies,
    init_method_normal,
    scaled_init_method_normal,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    NativeDSA,
    assert_similarity,
)


class NativeGatedOutputProjection(nn.Module):
    """PyTorch reference for HY4's value-space output gate and output projection."""

    def __init__(self, config: MLATransformerConfig, output_proj: nn.Module):
        super().__init__()
        self.output_proj = output_proj
        self.gate_input = None
        self.num_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim
        self.granularity = config.gated_attention_proj_granularity
        if config.attention_output_gate:
            gate_rows = (
                self.num_heads * self.v_head_dim
                if self.granularity == "elementwise"
                else self.num_heads
            )
            self.gate_proj = nn.Linear(config.hidden_size, gate_rows, bias=False)
        else:
            self.gate_proj = None

    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            gate_states = self.gate_proj(self.gate_input)
            scale = torch.sigmoid(gate_states.float()).to(attn_output.dtype)
            if self.granularity == "headwise":
                scale = scale.repeat_interleave(self.v_head_dim, dim=-1)
            attn_output = attn_output * scale
        return self.output_proj(attn_output)


class NativeGatedDSA(NativeDSA):
    """Independent native DSA with the HY4 gate at its pre-output projection boundary."""

    def __init__(self, config: MLATransformerConfig):
        # Main's native reference exposes one epsilon for all three norms.
        assert (
            config.layernorm_epsilon
            == config.attention_latent_norm_epsilon
            == config.dsa_indexer_k_norm_epsilon
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_head_dim=config.qk_head_dim,
            qk_pos_emb_head_dim=config.qk_pos_emb_head_dim,
            v_head_dim=config.v_head_dim,
            dsa_indexer_n_heads=config.dsa_indexer_n_heads,
            dsa_indexer_head_dim=config.dsa_indexer_head_dim,
            dsa_indexer_topk=config.dsa_indexer_topk,
            dsa_indexer_use_sparse_loss=config.dsa_indexer_use_sparse_loss,
            layernorm_epsilon=config.attention_latent_norm_epsilon,
            rotary_base=config.rotary_base,
            rotary_scaling_factor=config.rotary_scaling_factor,
            original_max_position_embeddings=config.original_max_position_embeddings,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            mscale=config.mscale,
            rope_factor=config.rotary_scaling_factor,
            calculate_per_token_loss=config.calculate_per_token_loss,
        )
        self.linear_proj = NativeGatedOutputProjection(config, self.linear_proj)

    def forward(self, hidden_states, attention_mask):
        self.linear_proj.gate_input = hidden_states.transpose(0, 1).contiguous()
        return super().forward(hidden_states, attention_mask)


def _make_config(
    *,
    gate_granularity,
    tp_size,
    cp_size,
    recompute_module,
    production_shape=False,
    delayed=False,
    kernel_backend="none",
    indexer_loss_coeff=0.0,
    indexer_topk=8,
):
    # One full HY4 attention shape guards the real gate projection width. The
    # distributed matrix uses a reduced proxy, retaining unequal QK/V dimensions,
    # compressed KV, and genuine top-k selection while keeping CI cost bounded.
    dimensions = (
        dict(
            hidden_size=6144,
            num_attention_heads=64,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_head_dim=192,
            qk_pos_emb_head_dim=64,
            v_head_dim=256,
            dsa_indexer_n_heads=32,
            dsa_indexer_head_dim=128,
        )
        if production_shape
        else dict(
            hidden_size=512,
            num_attention_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=96,
            dsa_indexer_n_heads=4,
            dsa_indexer_head_dim=64,
        )
    )
    return MLATransformerConfig(
        **dimensions,
        num_layers=1,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        attention_output_gate=gate_granularity is not None,
        gated_attention_proj_granularity=gate_granularity or "elementwise",
        dsa_indexer_topk=indexer_topk,  # HY4's production top-k is 2048.
        dsa_indexer_loss_coeff=indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_epsilon=1e-6,
        attention_latent_norm_epsilon=1e-6,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=tp_size > 1,
        cp_comm_type="allgather",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rope_type="rope",
        rotary_base=10000,
        rotary_scaling_factor=1,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_position_embeddings=4096,
        apply_rope_fusion=False,
        recompute_granularity="selective" if recompute_module else None,
        recompute_modules=[recompute_module] if recompute_module else [],
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1),
        gradient_accumulation_fusion=False,
        delay_wgrad_compute=delayed,
        # The isolated gate test constructs a valid EP-overlap configuration
        # for TE's delayed wgrad instead of mutating flags after validation.
        overlap_moe_expert_parallel_comm=delayed,
        expert_model_parallel_size=2 if delayed else 1,
        num_moe_experts=2 if delayed else None,
        moe_token_dispatcher_type='alltoall',
        attention_backend=AttnBackend.unfused if kernel_backend == "none" else AttnBackend.auto,
        dsa_kernel_backend=kernel_backend,
        mla_down_proj_fusion=False,
    )


def _mapped_name(name):
    if name.startswith("indexer."):
        return "core_attention." + name
    if name == "linear_proj.output_proj.weight":
        return "linear_proj.weight"
    if name == "linear_proj.gate_proj.weight":
        return "linear_gate.weight"
    return name


def _parameter_partition(name, tensor, tp_rank, tp_size):
    if name in {"linear_q_up_proj.weight", "linear_kv_up_proj.weight", "linear_gate.weight"}:
        return tensor.chunk(tp_size, dim=0)[tp_rank]
    if name == "linear_proj.weight":
        return tensor.chunk(tp_size, dim=1)[tp_rank]
    return tensor


def _cp_positions(sequence_lengths, cp_rank, cp_size, device):
    # Megatron's zigzag CP ordering: the rank's leading and mirrored trailing
    # chunks of each individual packed sequence, retaining packed sequence order.
    positions = []
    offset = 0
    for length in sequence_lengths:
        chunks = torch.arange(offset, offset + length, device=device).chunk(2 * cp_size)
        positions.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
        offset += length
    return torch.cat(positions)


def _reduced_parameter_grad(param, *, tp_size, cp_size):
    gradient = param.grad.detach().clone()
    # The outer training gradient finalizer normally sums SP-replicated weights.
    # This module-only comparison performs those same reductions explicitly.
    if tp_size > 1 and getattr(param, "sequence_parallel", False):
        dist.all_reduce(gradient, group=parallel_state.get_tensor_model_parallel_group())
    if cp_size > 1:
        dist.all_reduce(gradient, group=parallel_state.get_context_parallel_group())
    return gradient


def _autograd_node_names(tensor):
    """Observe the executed real backend without replacing a kernel/dispatcher."""
    names = set()
    visited = set()
    pending = [tensor.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        names.add(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return names


def _run_native_parity(
    gate_granularity,
    *,
    tp_size=1,
    cp_size=1,
    qkv_format="sbhd",
    recompute_module=None,
    production_shape=False,
    saturated_gate=False,
    kernel_backend="none",
    indexer_loss_coeff=0.0,
    indexer_topk=8,
):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    previous_loss_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    if previous_loss_scale is not None:
        previous_loss_scale = previous_loss_scale.detach().clone()
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity=gate_granularity,
            tp_size=tp_size,
            cp_size=cp_size,
            recompute_module=recompute_module,
            production_shape=production_shape,
            kernel_backend=kernel_backend,
            indexer_loss_coeff=indexer_loss_coeff,
            indexer_topk=indexer_topk,
        )
        spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
        actual = build_module(spec, config=config, layer_number=1, cp_comm_type="allgather").cuda()
        assert isinstance(actual, AbsorbedMLASelfAttention)
        assert isinstance(actual.core_attention, DSAttention)
        assert (actual.linear_gate is not None) == (gate_granularity is not None)

        # Independent full-head/full-sequence reference on each rank. Slice only
        # weights and input/output positions at the distributed module boundary.
        torch.manual_seed(5678)
        reference = NativeGatedDSA(config).cuda().bfloat16()
        if saturated_gate:
            with torch.no_grad():
                reference.linear_proj.gate_proj.weight.zero_()
                reference.linear_proj.gate_proj.weight[:, 0] = 8.0
        actual_parameters = dict(actual.named_parameters())
        mapped = {_mapped_name(name) for name, _ in reference.named_parameters()}
        assert mapped == actual_parameters.keys()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                actual_name = _mapped_name(name)
                shard = _parameter_partition(actual_name, parameter, tp_rank, tp_size)
                assert actual_parameters[actual_name].shape == shard.shape, actual_name
                actual_parameters[actual_name].copy_(shard)

        sequence_lengths = [32, 16] if qkv_format == "thd" else [32]
        sequence_length = sum(sequence_lengths)
        generator = torch.Generator(device="cuda").manual_seed(91011)
        hidden_states = torch.randn(
            (sequence_length, 1, config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        output_gradient = torch.randn(
            hidden_states.shape, dtype=hidden_states.dtype, device="cuda", generator=generator
        )
        if saturated_gate:
            hidden_states[..., 0] = 1.0
            with torch.no_grad():
                gate_logits = reference.linear_proj.gate_proj(hidden_states)
                scale_fp32 = torch.sigmoid(gate_logits.float())
                assert torch.all(gate_logits == 8)
                assert torch.all(scale_fp32 < 1)
                assert torch.all(scale_fp32.to(hidden_states.dtype) == 1)
        cp_positions = _cp_positions(sequence_lengths, cp_rank, cp_size, hidden_states.device)
        local_positions = cp_positions.chunk(tp_size)[tp_rank]
        actual_input = hidden_states.index_select(0, local_positions).detach().requires_grad_(True)
        reference_input = hidden_states.detach().clone().requires_grad_(True)

        packed_seq_params = None
        if qkv_format == "thd":
            cu_seqlens = torch.tensor([0, 32, 48], dtype=torch.int32, device="cuda")
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens,
                max_seqlen_q=max(sequence_lengths),
                max_seqlen_kv=max(sequence_lengths),
                qkv_format="thd",
            )

        # Running the independent reference separately per packed sequence gives
        # causal boundaries and RoPE resets without using MCore's mask/RoPE helpers.
        native_outputs = []
        native_losses = []
        start = 0
        for length in sequence_lengths:
            causal_mask = torch.triu(
                torch.full((1, 1, length, length), -torch.inf, device="cuda"), diagonal=1
            )
            native_output, native_loss = reference(
                reference_input[start : start + length], causal_mask
            )
            native_outputs.append(native_output)
            native_losses.append(native_loss * length)
            start += length
        expected = torch.cat(native_outputs)
        if indexer_loss_coeff > 0:
            DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        output, _ = actual(actual_input, attention_mask=None, packed_seq_params=packed_seq_params)
        if kernel_backend == "cudnn":
            nodes = _autograd_node_names(output)
            assert nodes.intersection(
                {"FusedIndexerSparseAttnFuncBackward", "FusedSparseAttentionFuncBackward"}
            ), f"cuDNN/FlashMLA DSA fell back to unfused attention: {sorted(nodes)}"
        assert_similarity(
            output, expected.index_select(0, local_positions), label="attention output"
        )
        output.backward(output_gradient.index_select(0, local_positions))
        expected.backward(output_gradient)
        if indexer_loss_coeff > 0:
            # NativeDSA detaches the attention teacher and indexer inputs. Its
            # auxiliary backward independently trains only the indexer weights.
            native_loss = torch.stack(native_losses).sum() / sequence_length
            (native_loss * indexer_loss_coeff).backward()
        assert_similarity(
            actual_input.grad,
            reference_input.grad.index_select(0, local_positions),
            label="attention input gradient",
        )
        for name, parameter in reference.named_parameters():
            actual_name = _mapped_name(name)
            actual_parameter = actual_parameters[actual_name]
            if name.startswith("indexer.") and indexer_loss_coeff == 0:
                assert parameter.grad is None and actual_parameter.grad is None
                continue
            assert parameter.grad is not None, name
            assert actual_parameter.grad is not None, actual_name
            actual_gradient = _reduced_parameter_grad(
                actual_parameter, tp_size=tp_size, cp_size=cp_size
            )
            reference_gradient = _parameter_partition(actual_name, parameter.grad, tp_rank, tp_size)
            if name.startswith("indexer."):
                assert torch.count_nonzero(actual_gradient) > 0, actual_name
                assert torch.count_nonzero(reference_gradient) > 0, name
            if saturated_gate and actual_name == "linear_gate.weight":
                # The forward scale rounds to BF16 one, but FP32 sigmoid saves
                # its unrounded output and must preserve a nonzero gate VJP.
                assert torch.count_nonzero(actual_gradient) > 0
                assert torch.count_nonzero(reference_gradient) > 0
            assert_similarity(actual_gradient, reference_gradient, label=actual_name)
    finally:
        DSAIndexerLossAutoScaler.main_loss_backward_scale = previous_loss_scale
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("gate_granularity", [None, "elementwise", "headwise"])
@pytest.mark.parametrize(
    ("tp_size", "cp_size", "qkv_format", "recompute_module"),
    [
        (1, 1, "sbhd", None),
        (1, 1, "thd", "mla_up_proj"),
        (2, 1, "sbhd", "mla_up_proj"),
        (1, 2, "sbhd", None),
        (2, 2, "thd", "core_attn"),
        (2, 2, "thd", "mla_up_proj"),
    ],
)
def test_gated_absorbed_dsa_matches_native(
    gate_granularity, tp_size, cp_size, qkv_format, recompute_module
):
    _run_native_parity(
        gate_granularity,
        tp_size=tp_size,
        cp_size=cp_size,
        qkv_format=qkv_format,
        recompute_module=recompute_module,
    )


def test_hy4_attention_shape_gated_dsa_matches_native():
    _run_native_parity("elementwise", production_shape=True)


def test_hy4_attention_shape_cudnn_gated_dsa_matches_native():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused DSA parity")
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        torch.cuda.set_device(int(local_rank))
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN fused DSA path requires SM90+")
    missing = _missing_cudnn_dsa_kernel_dependencies()
    if missing:
        pytest.skip(f"cuDNN fused DSA dependencies are unavailable: {', '.join(missing)}")
    _run_native_parity("elementwise", production_shape=True, kernel_backend="cudnn")


def test_gated_dsa_fp32_sigmoid_saturated_gate_gradient():
    # The controlled saturation input can move a near-tied Top-K8 boundary:
    # observed row 20 had an actual 8th/9th score gap of 1.6e-6 and one changed key.
    # Select the full causal history to isolate the FP32 sigmoid VJP here while
    # retaining all output/input/parameter-gradient checks. Sparse parity cases
    # use the default Top-K8 and exercise genuine key selection.
    _run_native_parity("elementwise", saturated_gate=True, indexer_topk=32)


@pytest.mark.parametrize("gate_granularity", [None, "elementwise", "headwise"])
@pytest.mark.parametrize(("tp_size", "cp_size"), [(1, 1), (2, 2)])
def test_gated_dsa_checkpoint_preserves_gate_row_sharding(gate_granularity, tp_size, cp_size):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity=gate_granularity,
            tp_size=tp_size,
            cp_size=cp_size,
            recompute_module=None,
        )
        actual = build_module(
            get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider()),
            config=config,
            layer_number=1,
            cp_comm_type="allgather",
        ).cuda()
        state = actual.sharded_state_dict(prefix="attention.", sharded_offsets=((0, 1, 3),))
        key = "attention.linear_gate.weight"
        if gate_granularity is None:
            assert key not in state
            return

        gate = state[key]
        assert isinstance(gate, ShardedTensor)
        rows = (
            config.num_attention_heads * config.v_head_dim
            if gate_granularity == "elementwise"
            else config.num_attention_heads
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        assert gate.key == key
        assert gate.local_shape == (rows // tp_size, config.hidden_size)
        assert gate.global_shape == (3, rows, config.hidden_size)
        assert gate.global_offset == (1, tp_rank * rows // tp_size, 0)
        assert gate.axis_fragmentations == (3, tp_size, 1)
        assert gate.prepend_axis_num == 1
        assert gate.dtype == config.params_dtype
        assert gate.replica_id == (
            0,
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )
        assert gate.data.data_ptr() == actual.linear_gate.weight.data_ptr()
        assert "attention.linear_gate.bias" not in state
        gate.validate_metadata_integrity()
    finally:
        Utils.destroy_model_parallel()


def test_gated_dsa_missing_gate_module_spec_is_rejected():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity="elementwise", tp_size=1, cp_size=1, recompute_module=None
        )
        spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
        spec.submodules.linear_gate = None
        with pytest.raises(
            ValueError, match="MLA output gating requires a linear_gate module spec"
        ):
            build_module(spec, config=config, layer_number=1, cp_comm_type="allgather")
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("fused_down_projection", [False, True], ids=["ordinary", "fused-down"])
def test_ordinary_mla_module_rejects_output_gating(fused_down_projection):
    """A valid gated DSA config cannot silently enable gates in ordinary MLA."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity="elementwise", tp_size=1, cp_size=1, recompute_module=None
        )
        spec = get_gpt_layer_with_transformer_engine_submodules(
            multi_latent_attention=True,
            qk_layernorm=True,
            mla_down_proj_fusion=fused_down_projection,
        ).self_attention
        assert spec.module is (FusedMLASelfAttention if fused_down_projection else MLASelfAttention)
        with pytest.raises(
            NotImplementedError, match="Output gating is only supported for absorbed MLA"
        ):
            build_module(spec, config=config, layer_number=1)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("gate_granularity", ["elementwise", "headwise"])
def test_ordinary_mla_config_rejects_output_gating(gate_granularity):
    config = _make_config(
        gate_granularity=gate_granularity, tp_size=1, cp_size=1, recompute_module=None
    )
    with pytest.raises(
        NotImplementedError, match="Output gating is only supported for absorbed MLA"
    ):
        replace(config, experimental_attention_variant=None)


@pytest.mark.parametrize("gate_granularity", ["elementwise", "headwise"])
def test_gated_absorbed_mla_rejects_fused_down_projection(gate_granularity):
    config = _make_config(
        gate_granularity=gate_granularity, tp_size=1, cp_size=1, recompute_module=None
    )
    with pytest.raises(
        ValueError, match="Absorbed MLA output gating requires unfused down projections"
    ):
        replace(config, mla_down_proj_fusion=True)


@pytest.mark.parametrize("gate_granularity", ["elementwise", "headwise"])
def test_hybrid_inference_dsa_rejects_output_gating(gate_granularity):
    # Main's inference DSA spec uses ordinary MLA, which must explicitly reject
    # the enabled flag rather than construct an ungated inference module.
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity=gate_granularity, tp_size=1, cp_size=1, recompute_module=None
        )
        spec = hybrid_inference_stack_spec.submodules.dsa_layer.submodules.self_attention
        assert spec.module is MLASelfAttention
        with pytest.raises(
            NotImplementedError, match="Output gating is only supported for absorbed MLA"
        ):
            build_module(spec, config=config, layer_number=1, cp_comm_type="allgather")
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("gate_granularity", ["elementwise", "headwise"])
def test_gated_dsa_projection_delayed_weight_gradient_matches_native(gate_granularity):
    """Check the new TE gate's real deferred queue, not full DSA deferred support."""
    assert torch.cuda.is_available(), "Deferred TE gate gradients require real CUDA."
    torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(
            gate_granularity=gate_granularity,
            tp_size=1,
            cp_size=1,
            recompute_module=None,
            delayed=True,
        )
        spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
        delayed = build_module(spec, config=config, layer_number=1, cp_comm_type="allgather").cuda()
        immediate = build_module(
            spec,
            config=replace(config, delay_wgrad_compute=False),
            layer_number=1,
            cp_comm_type="allgather",
        ).cuda()
        assert isinstance(delayed.linear_gate, TEColumnParallelLinear)
        assert isinstance(immediate.linear_gate, TEColumnParallelLinear)
        native = NativeGatedOutputProjection(config, nn.Identity()).cuda().bfloat16()
        with torch.no_grad():
            immediate.linear_gate.weight.copy_(delayed.linear_gate.weight)
            native.gate_proj.weight.copy_(delayed.linear_gate.weight)
        store = delayed.linear_gate.wgrad_store
        assert store.delay_wgrad_compute()
        assert store.context.empty()
        width = config.num_attention_heads * config.v_head_dim
        generator = torch.Generator(device="cuda").manual_seed(5678)

        # Two microbatches catch both a missing flush and stale TE queue entries.
        # QKV, indexer, and output projections never execute in this isolated
        # scheduling test; full attention math is covered by native parity above.
        for _ in range(2):
            delayed.zero_grad(set_to_none=True)
            immediate.zero_grad(set_to_none=True)
            native.zero_grad(set_to_none=True)
            gate_input = torch.randn(
                (32, 1, config.hidden_size),
                dtype=config.params_dtype,
                device="cuda",
                generator=generator,
            )
            values = torch.randn(
                (32, 1, width), dtype=config.params_dtype, device="cuda", generator=generator
            )
            upstream_gradient = torch.randn(
                values.shape, dtype=values.dtype, device="cuda", generator=generator
            )
            delayed_input = gate_input.detach().clone().requires_grad_(True)
            immediate_input = gate_input.detach().clone().requires_grad_(True)
            native_input = gate_input.detach().clone().requires_grad_(True)
            delayed_values = values.detach().clone().requires_grad_(True)
            immediate_values = values.detach().clone().requires_grad_(True)
            native_values = values.detach().clone().requires_grad_(True)
            native.gate_input = native_input

            output = delayed._project_and_apply_mla_output_gate(delayed_values, delayed_input)
            immediate_output = immediate._project_and_apply_mla_output_gate(
                immediate_values, immediate_input
            )
            native_output = native(native_values)
            torch.testing.assert_close(output, immediate_output, atol=0, rtol=0)
            assert_similarity(output, native_output, label="deferred gate output")
            output.backward(upstream_gradient)
            immediate_output.backward(upstream_gradient)
            native_output.backward(upstream_gradient)
            assert delayed.linear_gate.weight.grad is None
            assert store.context.qsize() == 1
            torch.testing.assert_close(delayed_input.grad, immediate_input.grad, atol=0, rtol=0)
            torch.testing.assert_close(delayed_values.grad, immediate_values.grad, atol=0, rtol=0)
            assert_similarity(delayed_input.grad, native_input.grad, label="gate input gradient")
            assert_similarity(delayed_values.grad, native_values.grad, label="gate value gradient")

            # Use the real TE projection's scheduling API. Calling the enclosing
            # DSA backward_dw would exercise unrelated, unsupported main paths.
            delayed.linear_gate.backward_dw()
            assert store.context.empty()
            assert delayed.linear_gate.weight.grad is not None
            assert torch.count_nonzero(delayed.linear_gate.weight.grad) > 0
            torch.testing.assert_close(
                delayed.linear_gate.weight.grad, immediate.linear_gate.weight.grad, atol=0, rtol=0
            )
            assert_similarity(
                delayed.linear_gate.weight.grad,
                native.gate_proj.weight.grad,
                label="deferred gate weight gradient",
            )
            for name, parameter in delayed.named_parameters():
                if name != "linear_gate.weight":
                    assert parameter.grad is None, name
    finally:
        Utils.destroy_model_parallel()
