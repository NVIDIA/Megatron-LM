# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net.common import (
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# NVLS doesn't support one single GPU to be shared by multiple ranks, so disable this in test
os.environ.update({"NCCL_NVLS_ENABLE": "0"})


def _unpack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor, dim=1) -> list[torch.Tensor]:
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp", "linear_cp_mode"),
    [
        (4, False, 1, None),  # TP w/o SP
        (4, True, 1, None),  # TP w/ SP
        (1, False, 2, "headwise"),  # Headwise CP
        (2, False, 2, "headwise"),  # TP w/o SP + Headwise CP
        (2, True, 2, "headwise"),  # TP w/ SP + Headwise CP
        (1, False, 2, "chunkwise"),  # Chunkwise CP
        (2, False, 2, "chunkwise"),  # TP w/o SP + chunkwise CP
        (2, True, 2, "chunkwise"),  # TP w/ SP + chunkwise CP
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, linear_cp_mode
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
        linear_cp_mode=linear_cp_mode,
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    cosine_similarity_threshold = None
    if cp > 1:
        atol, rtol = 2e-3, 1e-2
        cosine_similarity_threshold = 0.9999
    else:
        atol, rtol = 2e-4, 2e-3
        cosine_similarity_threshold = 0.99999

    is_chunkwise_cp = linear_cp_mode == "chunkwise" and cp > 1
    micro_batch_size = 1 if is_chunkwise_cp and not sequence_packing else 4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        cosine_similarity_threshold=cosine_similarity_threshold,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=micro_batch_size,
        sequence_packing=sequence_packing,
        cp_partition_mode="contiguous" if is_chunkwise_cp else "zigzag",
        compare_param_grads=is_chunkwise_cp and tp == 1 and not sequence_packing,
    )


@pytest.mark.parametrize("cp_size", [2, 4], scope="class")
@pytest.mark.internal
class TestFusedThdAllToAll:
    """Verify fused 1 AllToAll + permute matches the per-sequence, per-channel loop in GDN."""

    @pytest.fixture(scope='class', autouse=True)
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

    # ---- Optimized: single a2a + production permutation helper ----

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
            split_sections=None,  # always single fused a2a
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
