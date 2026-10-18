# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_FLA_GDN2
from megatron.core.ssm.gated_delta_net.common import (
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_utils import _unpack_sequence
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, run_config
):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 5e-4, 5e-4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=4,
        sequence_packing=sequence_packing,
    )


@pytest.mark.parametrize("variant", ["gated_delta_net", "gdn2"])
@pytest.mark.parametrize(("tp", "sp"), [(2, False), (4, False), (2, True), (4, True)])
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_gdn_out_norm_grad_is_summed_across_tp(variant, tp, sp):
    """The replicated out_norm weight must get the full-head gradient on every TP rank.

    Each TP rank owns a slice of the value heads that out_norm scales, so its backward only
    accumulates a partial gradient. After finalize_model_grads, the gradient must be the sum of
    the partials over the TP group, identical on every TP rank, with and without sequence
    parallelism.
    """
    if variant == "gdn2" and not HAVE_FLA_GDN2:
        pytest.skip("FLA with GDN2 support is not installed.")
    if Utils.world_size % tp != 0:
        pytest.skip(f"world size {Utils.world_size} is not divisible by tp={tp}")

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        dp_group = parallel_state.get_data_parallel_group()
        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp,
            sequence_parallel=sp,
            experimental_attention_variant=variant,
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_spec = get_experimental_attention_variant_module_spec(config=config)
        gdn = gdn_spec.module(
            config,
            submodules=gdn_spec.submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(
                tp=tp_group, cp=parallel_state.get_context_parallel_group()
            ),
        ).cuda()
        model = DistributedDataParallel(
            config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), gdn
        )

        # The same full-sequence input and loss weights on every rank.
        seq_length, micro_batch_size = 64, 2
        torch.manual_seed(0)
        shape = (seq_length, micro_batch_size, config.hidden_size)
        hidden_states = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        loss_weights = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if sp:
            local_seq_length = seq_length // tp
            local_slice = slice(
                tp_group.rank() * local_seq_length, (tp_group.rank() + 1) * local_seq_length
            )
            hidden_states = hidden_states[local_slice]
            loss_weights = loss_weights[local_slice]

        model.zero_grad_buffer()
        output, _ = model(hidden_states.requires_grad_(), attention_mask=None)
        (output.float() * loss_weights.float()).sum().backward()

        params = list(gdn.out_norm.parameters())
        assert params, "out_norm has no parameters"
        # Before finalize_model_grads, main_grad holds this rank's partial (own-heads) gradient.
        expected = []
        for param in params:
            grad = param.main_grad.detach().clone()
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG, group=dp_group)
            expected.append(grad)

        finalize_model_grads([model])

        for param, expected_grad in zip(params, expected):
            grad = param.main_grad.detach()
            gathered = [torch.empty_like(grad) for _ in range(tp)]
            torch.distributed.all_gather(gathered, grad.contiguous(), group=tp_group)
            for rank, other in enumerate(gathered):
                assert torch.equal(
                    other, gathered[0]
                ), f"out_norm main_grad differs between TP rank 0 and TP rank {rank}"
            torch.testing.assert_close(grad, expected_grad)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", [2, 4], scope="class")
@pytest.mark.internal
@pytest.mark.skip(
    "Used to verify the correctness of the fused THD AllToAll implementation, locally validated thus no need to run on CI."
)
class TestFusedThdAllToAll:
    """Verify fused 1 AllToAll + permute matches the per-sequence, per-channel loop in GDN."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_method(self, request, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        # Attach on the class so every test method can read self.cp_*.
        request.cls.cp_size = cp_size
        request.cls.cp_group = parallel_state.get_context_parallel_group()
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _per_seq_a2a_cp2hp(local_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        unpacked = _unpack_sequence(local_t, cu_seqlens // cp_size, dim=0)
        outputs = []
        for x in unpacked:
            outputs.append(
                tensor_a2a_cp2hp(
                    x,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_group,
                    split_sections=split_sections,
                    undo_attention_load_balancing=True,
                )
            )
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _per_seq_a2a_hp2cp(global_t, cu_seqlens, cp_group, split_sections=None):
        unpacked = _unpack_sequence(global_t, cu_seqlens, dim=0)
        outputs = []
        for x in unpacked:
            outputs.append(
                tensor_a2a_hp2cp(
                    x,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_group,
                    split_sections=split_sections,
                    redo_attention_load_balancing=True,
                )
            )
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _batched_a2a_cp2hp(local_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        t_global = int(cu_seqlens[-1].item())
        if split_sections is not None and cp_size > 1:
            head_perm = _build_head_perm_for_split_sections(split_sections, cp_size, local_t.device)
            local_t = local_t.index_select(-1, head_perm)
        naive = tensor_a2a_cp2hp(
            local_t,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            split_sections=None,
            undo_attention_load_balancing=False,
        )
        idx, _ = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, t_global)
        return naive.index_select(0, idx)

    @staticmethod
    def _batched_a2a_hp2cp(global_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        t_global = int(cu_seqlens[-1].item())
        _, inv = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, t_global)
        permuted = global_t.index_select(0, inv)
        return tensor_a2a_hp2cp(
            permuted,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            split_sections=split_sections,
            redo_attention_load_balancing=False,
        )

    @pytest.mark.parametrize(
        "cu_seqlens",
        [
            (0, 32, 64),  # 2 equal sequences
            (0, 32, 64, 96, 128),  # 4 equal sequences (matches existing THD test)
            (0, 16, 48, 80),  # 3 unequal sequences
        ],
    )
    @pytest.mark.parametrize("split_sections", [(8, 8, 4, 16, 32, 4)])
    def test_cp2hp_batched_matches_per_seq(self, cu_seqlens, split_sections):
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if (torch.diff(cu) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        T_local = T_global // self.cp_size
        hidden = sum(split_sections)
        torch.manual_seed(42)
        local_t = (
            torch.rand(T_local, 1, hidden, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        out_ref = self._per_seq_a2a_cp2hp(local_t, cu, self.cp_group, split_sections=split_sections)
        out_fused = self._batched_a2a_cp2hp(
            local_t, cu, self.cp_group, split_sections=split_sections
        )

        rank = torch.distributed.get_rank()
        assert torch.equal(out_fused, out_ref), (
            f"Batched CP->HP mismatch on rank={rank} " f"(split_sections={split_sections})"
        )

    @pytest.mark.parametrize("cu_seqlens", [(0, 32, 64), (0, 32, 64, 96, 128), (0, 16, 48, 80)])
    def test_hp2cp_batched_matches_per_seq(self, cu_seqlens):
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if ((cu[1:] - cu[:-1]) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        hidden = 32
        # Hidden must be divisible by cp_size for the HP-sharded input layout.
        assert hidden % self.cp_size == 0
        h_local = hidden // self.cp_size
        torch.manual_seed(42)
        global_t = (
            torch.rand(T_global, 1, h_local, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        out_ref = self._per_seq_a2a_hp2cp(global_t, cu, self.cp_group)
        out_fused = self._batched_a2a_hp2cp(global_t, cu, self.cp_group)

        rank = torch.distributed.get_rank()
        assert torch.equal(out_fused, out_ref), f"Batched HP->CP mismatch on rank={rank}"

    @pytest.mark.parametrize("cu_seqlens", [(0, 32, 64, 96, 128)])
    def test_cp2hp_hp2cp_round_trip(self, cu_seqlens):
        """cp2hp followed by hp2cp on the batched path should be the identity."""
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if ((cu[1:] - cu[:-1]) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        T_local = T_global // self.cp_size
        hidden = 32
        torch.manual_seed(7)
        local_t = (
            torch.rand(T_local, 1, hidden, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        mid = self._batched_a2a_cp2hp(local_t, cu, self.cp_group)
        back = self._batched_a2a_hp2cp(mid, cu, self.cp_group)

        assert torch.equal(back, local_t), "Batched cp2hp -> hp2cp not identity"
