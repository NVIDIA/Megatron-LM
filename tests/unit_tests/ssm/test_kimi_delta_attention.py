# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, GatedDeltaNet, KimiDeltaAttention
from megatron.core.ssm.gated_delta_net.common import _GDNBase
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


def test_kda_and_gdn_are_sibling_gdn_family_variants():
    assert issubclass(GatedDeltaNet, _GDNBase)
    assert issubclass(KimiDeltaAttention, _GDNBase)
    assert not issubclass(KimiDeltaAttention, GatedDeltaNet)


def _make_config(
    tp_size: int = 1,
    cp_size: int = 1,
    sequence_parallel: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
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
        transformer_impl="transformer_engine",
    )


def test_kda_norm_out_recompute_config_accepts_gdn_family_selector():
    config = replace(
        _make_config(), recompute_granularity="selective", recompute_modules=["gdn_norm_out"]
    )

    assert config.recompute_modules == ["gdn_norm_out"]


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


def test_kda_rejects_unsupported_head_layouts():
    config = _make_config()
    config.linear_num_value_heads = 8
    with pytest.raises(ValueError, match="equal key and value head counts"):
        KimiDeltaAttention._validate_config(config)

    config = _make_config()
    config.linear_value_head_dim = 64
    with pytest.raises(ValueError, match="equal key and value head dimensions"):
        KimiDeltaAttention._validate_config(config)

    config = _make_config()
    config.context_parallel_size = 3
    with pytest.raises(ValueError, match="tensor parallel size times context parallel size"):
        KimiDeltaAttention._validate_config(config)


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
def test_kda_forward_backward():
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = _make_config()
        kda = _build_kda(config)
        assert kda.in_proj_split_names == ["query", "key", "value", "g", "gate"]
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
        config = _make_config(params_dtype=torch.float32)
        kda = _build_kda(config)
        config.bf16 = True
        wrapped = Float16Module(config=config, module=kda)
        assert wrapped.module.in_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.beta_proj.weight.dtype == torch.bfloat16
        assert wrapped.module.A_log.dtype == torch.float32
        assert wrapped.module.dt_bias.dtype == torch.float32
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
def test_kda_selective_recompute_norm_out():
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
        recompute_config.recompute_modules = ["gdn_norm_out"]

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
        assert base_kda.norm_out_checkpoint is None
        assert recompute_kda.recompute_norm_out is True
        assert recompute_kda.norm_out_checkpoint is not None
        torch.testing.assert_close(recompute_output, base_output, atol=0, rtol=0)
        torch.testing.assert_close(recompute_input_grad, base_input_grad, atol=0, rtol=0)
        assert recompute_grads.keys() == base_grads.keys()
        for name in base_grads:
            # The checkpointed gate/norm backward can choose a different BF16 reduction order.
            # Keep output and input-gradient checks bitwise, while allowing only BF16-scale
            # rounding in parameter gradients.
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
    ("sequence_packing", "tp", "sp", "cp"),
    [(False, 2, True, 1), (True, 1, False, 2), (True, 2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
def test_parallel_kda_correctness(tmp_path_dist_ckpt, sequence_packing, tp, sp, cp):
    config = _make_config()
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
