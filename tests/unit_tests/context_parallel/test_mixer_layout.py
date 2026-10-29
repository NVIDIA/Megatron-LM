# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scheduler mixer adapters preserve recurrence order and document state boundaries."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel import finalize_packed_seq_params
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _indices(layout, docs, cp_rank, tp_rank, tp_size):
    if layout == "contiguous":
        result = torch.arange(cp_rank * 256, (cp_rank + 1) * 256, device="cuda")
    else:
        chunks = []
        for start, end in zip(docs[:-1], docs[1:]):
            width = (end - start) // 4
            for chunk in (cp_rank, 3 - cp_rank):
                chunks.append(
                    torch.arange(start + chunk * width, start + (chunk + 1) * width, device="cuda")
                )
        result = torch.cat(chunks)
    return result.chunk(tp_size)[tp_rank]


def _gather_sequence(value, indices, group):
    values = [torch.empty_like(value) for _ in range(group.size())]
    rows = [torch.empty_like(indices) for _ in range(group.size())]
    dist.all_gather(values, value.contiguous(), group=group)
    dist.all_gather(rows, indices, group=group)
    result = value.new_empty((512, *value.shape[1:]))
    for tensor, positions in zip(values, rows):
        result.index_copy_(0, positions, tensor)
    return result


@pytest.mark.parametrize("variant", ["mamba", "gdp"])
@pytest.mark.parametrize("linear_layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_scheduler_mixer_converts_actual_boundary_layout(variant, linear_layout, packed, tp_size):
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    if variant == "gdp":
        pytest.importorskip("fla")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=2)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "tp_cp"])
        boundary_layout = "zigzag" if linear_layout == "contiguous" else "contiguous"
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            mamba_num_heads=8,
            mamba_head_dim=64,
            mamba_num_groups=4,
            mamba_state_dim=128,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=2,
            sequence_parallel=tp_size > 1,
            gradient_accumulation_fusion=False,
            linear_cp_mode="headwise",
            linear_cp_layout=linear_layout,
            cp_partition_mode=boundary_layout,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=512,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=True,
        )
        mixer_type = MambaMixer if variant == "mamba" else GatedDeltaProductMixer
        spec = hybrid_stack_spec if variant == "mamba" else gdp_stack_spec
        model = mixer_type(
            config,
            spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=groups,
        ).cuda()
        docs = [0, 128, 512] if packed else [0, 512]
        cu = torch.tensor(docs, device="cuda", dtype=torch.int32)
        torch.manual_seed(456)
        full = torch.randn(512, 1, 256, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(full)

        def params(layout):
            if not packed:
                return None
            metadata = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=384,
                max_seqlen_kv=384,
                total_tokens=512,
                cp_partition_mode=layout,
            )
            finalize_packed_seq_params(metadata, groups.cp)
            return metadata

        def run(layout, staged):
            model.zero_grad(set_to_none=True)
            rows = _indices(layout, docs, groups.cp.rank(), groups.tp.rank(), tp_size)
            x = full[rows].detach().clone().requires_grad_()
            metadata = params(layout)
            model._cp_input_partition_mode = layout
            if staged:
                core = model.forward_pre_attn_and_core_attn(x, packed_seq_params=metadata)
                if metadata is not None:
                    assert metadata.cp_partition_mode == layout
                # Interleave a no-conversion call before projecting the converted call.
                with torch.no_grad():
                    model._cp_input_partition_mode = linear_layout
                    target_rows = _indices(
                        linear_layout, docs, groups.cp.rank(), groups.tp.rank(), tp_size
                    )
                    other = model.forward_pre_attn_and_core_attn(
                        full[target_rows], packed_seq_params=params(linear_layout)
                    )
                    model.forward_post_core_attn(other)
                y, _ = model.forward_post_core_attn(core)
            else:
                y, _ = model(x, packed_seq_params=metadata)
            (y.float() * upstream[rows].float()).sum().backward()
            output = _gather_sequence(y.detach(), rows, groups.tp_cp)
            input_grad = _gather_sequence(x.grad, rows, groups.tp_cp)
            grads = {
                name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
            }
            return output, input_grad, grads

        reference = run(linear_layout, False)
        for staged in (False, True):
            candidate = run(boundary_layout, staged)
            for actual, expected in zip(candidate[:2], reference[:2]):
                torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            assert candidate[2].keys() == reference[2].keys()
            for name in reference[2]:
                torch.testing.assert_close(
                    candidate[2][name], reference[2][name], atol=5e-3, rtol=5e-3, msg=name
                )
    finally:
        Utils.destroy_model_parallel()
