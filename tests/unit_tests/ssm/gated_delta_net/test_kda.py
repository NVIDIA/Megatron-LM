# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import os
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)

os.environ.update({"NCCL_NVLS_ENABLE": "0"})

_KDA_INPUT_GRAD_ATOL = 3.2e-2
_KDA_INPUT_GRAD_RELATIVE_L2 = 5e-3


def _assert_bias_disabled(module):
    bias = getattr(module, "bias", None)
    assert bias is None or bias.numel() == 0


def _make_config(
    tp_size: int = 1,
    cp_size: int = 1,
    sequence_parallel: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    linear_cp_mode: str = "headwise",
    f_lora_rank: int | None = None,
    gate_lora_rank: int | None = None,
) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=128,
        num_layers=1,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        kda_f_lora_rank=f_lora_rank,
        kda_gate_lora_rank=gate_lora_rank,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        normalization="RMSNorm",
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=params_dtype == torch.bfloat16,
        params_dtype=params_dtype,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sequence_parallel,
        linear_cp_mode=linear_cp_mode,
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
    )


def test_kda_norm_out_recompute_config_accepts_hybrid_model():
    config = replace(
        _make_config(),
        is_hybrid_model=True,
        recompute_granularity="selective",
        recompute_modules=["gdn_norm_out"],
    )

    assert config.recompute_modules == ["gdn_norm_out"]


def test_kda_full_recompute_config_accepts_hybrid_model():
    config = replace(
        _make_config(),
        is_hybrid_model=True,
        recompute_granularity="selective",
        recompute_modules=["gdn"],
    )

    assert config.recompute_modules == ["gdn"]


def _build_kda(config: TransformerConfig) -> KimiDeltaAttention:
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    return KimiDeltaAttention(
        config,
        submodules=spec.submodules,
        layer_number=1,
        conv_init=1.0,
        pg_collection=ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        ),
    ).cuda()


def test_kda_config_rejects_unsupported_head_layouts():
    with pytest.raises(ValueError, match="equal key and value head counts"):
        replace(_make_config(), linear_num_value_heads=8)

    with pytest.raises(ValueError, match="equal key and value head dimensions"):
        replace(_make_config(), linear_value_head_dim=64)

    with pytest.raises(ValueError, match="kda_f_lora_rank"):
        replace(_make_config(), kda_f_lora_rank=0)

    with pytest.raises(ValueError, match="kda_gate_lora_rank"):
        replace(_make_config(), kda_gate_lora_rank=-1)

    with pytest.raises(ValueError, match="must be either 'headwise' or 'chunkwise'"):
        _make_config(linear_cp_mode="tokenwise")

    with pytest.raises(AssertionError, match="linear_head_parallel_size"):
        _make_config(cp_size=3, linear_cp_mode="headwise")

    chunkwise_config = _make_config(cp_size=3, linear_cp_mode="chunkwise")
    assert chunkwise_config.linear_cp_mode == "chunkwise"


@pytest.mark.parametrize("lower_bound", [None, -5.1, 0.0, 1.0, float("nan")])
def test_kda_rejects_invalid_safe_gate_lower_bound(lower_bound):
    with pytest.raises(ValueError, match="kda_lower_bound"):
        replace(_make_config(), kda_safe_gate=True, kda_lower_bound=lower_bound)


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -1e-6])
def test_kda_accepts_valid_safe_gate_lower_bound(lower_bound):
    config = replace(_make_config(), kda_safe_gate=True, kda_lower_bound=lower_bound)

    assert config.kda_lower_bound == lower_bound


def test_kda_ignores_lower_bound_when_safe_gate_is_disabled():
    config = replace(_make_config(), kda_safe_gate=False, kda_lower_bound=None)

    assert config.kda_lower_bound is None


def test_kda_rejects_invalid_packed_boundaries():
    q = torch.tensor([0, 8, 16], dtype=torch.int32)
    KimiDeltaAttention._validate_packed_cu_seqlens(q, q.clone())

    with pytest.raises(ValueError, match="cu_seqlens_q to equal cu_seqlens_kv"):
        KimiDeltaAttention._validate_packed_cu_seqlens(
            q, torch.tensor([0, 7, 16], dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="at least one sequence in both Q and KV"):
        empty = torch.tensor([0], dtype=torch.int32)
        KimiDeltaAttention._validate_packed_cu_seqlens(empty, empty.clone())


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.parametrize(
    ("f_lora_rank", "gate_lora_rank"),
    [(None, None), (16, None), (8, 12), (None, 12)],
    ids=[
        "legacy-fused",
        "low-rank-f-full-rank-gate",
        "low-rank-f-low-rank-gate",
        "full-rank-f-low-rank-gate",
    ],
)
def test_kda_forward_backward(f_lora_rank, gate_lora_rank):
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = _make_config(f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank)
        kda = _build_kda(config)
        legacy_fused = f_lora_rank is None and gate_lora_rank is None
        assert kda.use_legacy_fused_projections == legacy_fused
        if legacy_fused:
            assert kda.in_proj_split_names == ["query", "key", "value", "g", "gate"]
            assert kda.in_proj_dim == 3 * kda.qk_dim + 2 * kda.v_dim
        else:
            assert kda.in_proj_split_names == ["query", "key", "value"]
            assert kda.in_proj_dim == 2 * kda.qk_dim + kda.v_dim
        assert kda.A_log.dtype == torch.float32
        assert kda.dt_bias.dtype == torch.float32
        assert getattr(kda.A_log, "keep_in_fp32", False)
        assert getattr(kda.dt_bias, "keep_in_fp32", False)

        hidden_states = torch.randn(
            (32, 2, config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        state_keys = set(kda.state_dict())
        separate_projection_prefixes = (
            "f_proj.",
            "f_a_proj.",
            "f_b_proj.",
            "g_proj.",
            "g_a_proj.",
            "g_b_proj.",
        )
        separate_projection_keys = {
            key for key in state_keys if key.startswith(separate_projection_prefixes)
        }
        if legacy_fused:
            assert not separate_projection_keys
            assert kda.in_proj.weight.shape == (3 * kda.qk_dim + 2 * kda.v_dim, config.hidden_size)
        else:
            projection_input = hidden_states.detach()
            if f_lora_rank is None:
                assert "f_proj.weight" in state_keys
                _assert_bias_disabled(kda.f_proj)
                f_actual, _ = kda.f_proj(projection_input)
                f_expected = F.linear(projection_input, kda.f_proj.weight)
            else:
                assert kda.f_a_proj.weight.shape == (f_lora_rank, config.hidden_size)
                assert kda.f_b_proj.weight.shape == (kda.qk_dim, f_lora_rank)
                _assert_bias_disabled(kda.f_a_proj)
                _assert_bias_disabled(kda.f_b_proj)
                f_latent, _ = kda.f_a_proj(projection_input)
                f_actual, _ = kda.f_b_proj(f_latent)
                f_expected = F.linear(
                    F.linear(projection_input, kda.f_a_proj.weight), kda.f_b_proj.weight
                )
            torch.testing.assert_close(f_actual, f_expected, atol=1e-2, rtol=1e-2)

        gate_weight_keys = {
            key
            for key in state_keys
            if key.startswith(("g_proj.", "g_a_proj.", "g_b_proj.")) and key.endswith(".weight")
        }
        if legacy_fused:
            assert not gate_weight_keys
        elif gate_lora_rank is None:
            assert gate_weight_keys == {"g_proj.weight"}
            _assert_bias_disabled(kda.g_proj)
            gate_actual, _ = kda.g_proj(projection_input)
            gate_expected = F.linear(projection_input, kda.g_proj.weight)
            torch.testing.assert_close(gate_actual, gate_expected, atol=1e-2, rtol=1e-2)
        else:
            assert gate_weight_keys == {"g_a_proj.weight", "g_b_proj.weight"}
            _assert_bias_disabled(kda.g_a_proj)
            _assert_bias_disabled(kda.g_b_proj)
            gate_latent, _ = kda.g_a_proj(projection_input)
            gate_actual, _ = kda.g_b_proj(gate_latent)
            gate_expected = F.linear(
                F.linear(projection_input, kda.g_a_proj.weight), kda.g_b_proj.weight
            )
            torch.testing.assert_close(gate_actual, gate_expected, atol=1e-2, rtol=1e-2)

        output, bias = kda(hidden_states, None)
        assert bias is None
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
        output.float().square().mean().backward()
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        for name, parameter in kda.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
def test_kda_gate_parameters_remain_fp32_after_model_cast():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _make_config(params_dtype=torch.float32, f_lora_rank=16)
        kda = _build_kda(config)
        config.bf16 = True
        wrapped = Float16Module(config=config, module=kda)
        assert wrapped.module.in_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.beta_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.A_log.dtype == torch.float32
        assert wrapped.module.dt_bias.dtype == torch.float32
        assert wrapped.module.f_a_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.f_b_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.g_proj.weight.dtype == torch.bfloat16
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
def test_kda_packed_matches_unpacked_cp1():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(321)
        kda = _build_kda(_make_config())
        unpacked = torch.randn(
            (16, 2, kda.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        packed = unpacked.transpose(0, 1).contiguous().reshape(-1, 1, kda.config.hidden_size)
        packed_seq_params = make_test_packed_seq_params(cu_seqlens=[0, 16, 32])

        unpacked_output, _ = kda(unpacked, None)
        packed_output, _ = kda(packed, None, packed_seq_params=packed_seq_params)
        torch.testing.assert_close(
            packed_output[:, 0],
            unpacked_output.transpose(0, 1).reshape(-1, kda.config.hidden_size),
            atol=5e-3,
            rtol=5e-3,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.parametrize("recompute_module", ["gdn_norm_out", "gdn"])
def test_kda_selective_recompute(recompute_module):
    Utils.initialize_model_parallel(1, 1)
    try:

        def run(module, hidden_states):
            output, _ = module(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: parameter.grad.detach().clone()
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            }
            return output.detach(), hidden_states.grad.detach().clone(), grads

        base_config = _make_config()
        recompute_config = copy.deepcopy(base_config)
        recompute_config.recompute_granularity = "selective"
        recompute_config.recompute_modules = [recompute_module]

        model_parallel_cuda_manual_seed(42)
        base_kda = _build_kda(base_config)
        model_parallel_cuda_manual_seed(42)
        recompute_kda = _build_kda(recompute_config)
        recompute_kda.load_state_dict(base_kda.state_dict())

        torch.manual_seed(42)
        hidden_states = torch.randn(
            (64, 2, base_config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        base_input = hidden_states.detach().clone().requires_grad_(True)
        recompute_input = hidden_states.detach().clone().requires_grad_(True)

        base_output, base_input_grad, base_grads = run(base_kda, base_input)
        recompute_output, recompute_input_grad, recompute_grads = run(
            recompute_kda, recompute_input
        )

        assert base_kda.recompute_norm_out is False
        assert base_kda.recompute_gdn is False
        assert base_kda.norm_out_checkpoint is None
        assert recompute_kda.recompute_norm_out is (recompute_module == "gdn_norm_out")
        assert recompute_kda.recompute_gdn is (recompute_module == "gdn")
        assert (recompute_kda.norm_out_checkpoint is not None) is (
            recompute_module == "gdn_norm_out"
        )
        torch.testing.assert_close(recompute_output, base_output, atol=0, rtol=0)
        torch.testing.assert_close(recompute_input_grad, base_input_grad, atol=0, rtol=0)
        assert recompute_grads.keys() == base_grads.keys()
        for name in base_grads:
            # Recomputed BF16 kernels can choose a different reduction order. Keep output and
            # input-gradient checks bitwise while allowing only BF16-scale parameter-gradient
            # rounding.
            torch.testing.assert_close(
                recompute_grads[name],
                base_grads[name],
                atol=2e-4,
                rtol=5e-3,
                msg=f"Selective recompute gradient mismatch for {name}",
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
def test_kda_cp2_packed_with_physical_padding_matches_cp1():
    """Match CP2 packed KDA against CP1 when physical and actual boundaries differ."""

    sequence_length = 32
    micro_batch_size = 4
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        packed_seq_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 128]
        )
        torch.manual_seed(2026)
        hidden_states_sbhd = torch.rand(
            (sequence_length, micro_batch_size, 128),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        model_parallel_cuda_manual_seed(2026)
        reference_kda = _build_kda(_make_config(cp_size=1))
        reference_state = {
            name: tensor.detach().clone() for name, tensor in reference_kda.state_dict().items()
        }
        reference_input = (
            hidden_states_sbhd.transpose(0, 1)
            .contiguous()
            .view(-1, 1, hidden_states_sbhd.shape[-1])
            .requires_grad_(True)
        )
        reference_output, reference_bias = reference_kda(
            reference_input, None, packed_seq_params=packed_seq_params
        )
        assert reference_bias is None
        reference_output.float().sum().backward()
        reference_output_sbhd = (
            reference_output.detach()
            .view(micro_batch_size, sequence_length, -1)
            .transpose(0, 1)
            .contiguous()
        )
        reference_input_grad_sbhd = (
            reference_input.grad.detach()
            .view(micro_batch_size, sequence_length, -1)
            .transpose(0, 1)
            .contiguous()
        )
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        model_parallel_cuda_manual_seed(2026)
        parallel_kda = _build_kda(_make_config(cp_size=2))
        parallel_kda.load_state_dict(reference_state)
        cp_group = parallel_state.get_context_parallel_group()

        def get_packed_tensor_on_this_cp_rank(tensor):
            local_tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            return local_tensor.transpose(0, 1).contiguous().view(-1, 1, tensor.shape[-1])

        parallel_input = get_packed_tensor_on_this_cp_rank(hidden_states_sbhd)
        parallel_input = parallel_input.detach().requires_grad_(True)
        parallel_output, parallel_bias = parallel_kda(
            parallel_input, None, packed_seq_params=packed_seq_params
        )
        assert parallel_bias is None
        parallel_output.float().sum().backward()

        expected_output = get_packed_tensor_on_this_cp_rank(reference_output_sbhd)
        expected_input_grad = get_packed_tensor_on_this_cp_rank(reference_input_grad_sbhd)
        torch.testing.assert_close(parallel_output, expected_output, atol=1e-2, rtol=1e-2)
        grad_diff = (parallel_input.grad - expected_input_grad).float()
        relative_l2 = torch.linalg.vector_norm(grad_diff) / torch.linalg.vector_norm(
            expected_input_grad.float()
        ).clamp_min(1e-12)
        assert relative_l2 < _KDA_INPUT_GRAD_RELATIVE_L2
        # The fixed-seed CP comparison differs by at most one observed BF16 bin
        # (0.03125) while aggregate relative L2 remains below 0.5%.
        torch.testing.assert_close(
            parallel_input.grad, expected_input_grad, atol=_KDA_INPUT_GRAD_ATOL, rtol=1e-2
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ("sequence_packing", "tp", "sp", "cp", "linear_cp_mode", "f_lora_rank", "gate_lora_rank"),
    [
        (False, 2, True, 1, "headwise", None, None),
        (True, 1, False, 2, "headwise", 8, 12),
        (True, 2, True, 2, "headwise", 16, None),
        (True, 1, False, 2, "chunkwise", None, 12),
        (True, 2, True, 2, "chunkwise", 8, 12),
    ],
)
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
def test_parallel_kda_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, linear_cp_mode, f_lora_rank, gate_lora_rank
):
    config = _make_config(
        linear_cp_mode=linear_cp_mode, f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank
    )
    _test_parallel_attention_correctness(
        transformer_config=config,
        transformer_layer_spec=hybrid_stack_spec.submodules.kda_layer,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=1e-2,
        rtol=1e-2,
        # Parallel projection/reduction order can move individual BF16 values by
        # one observed bin; the dedicated padded-CP test above also bounds relative L2.
        input_grad_atol=_KDA_INPUT_GRAD_ATOL,
        input_grad_rtol=1e-2,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=42,
        sequence_length=128,
        micro_batch_size=2,
        sequence_packing=sequence_packing,
    )
