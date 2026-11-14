# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise actual Megatron HybridStack shortcut/MoE modules on two GPUs."""

import logging
import os

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.torch_mimo_fsdp import TorchMimoFSDP, TorchMimoOptimizer
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig

logging.basicConfig(level=logging.INFO)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
parallel_state.initialize_model_parallel(
    expert_model_parallel_size=2, expert_tensor_parallel_size=1
)
model_parallel_cuda_manual_seed(123)
pg = ProcessGroupCollection.use_mpu_process_groups()
wide = os.environ.get("MIMO_TEST_WIDE_REPLAY") == "1"
extra = (
    dict(
        wide_residual=WideResidualConfig(num_streams=4, learned_retention=True),
        recompute_granularity="selective",
        recompute_modules=["residual_stream", "moe_act"],
        residual_stream_recompute_num_layers=1,
    )
    if wide
    else {}
)
mtp = os.environ.get("MIMO_TEST_MTP") == "1"
if mtp:
    extra.update(mtp_num_layers=2, mtp_use_repeated_layer=True, mtp_loss_scaling_factor=0.3)
config = TransformerConfig(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=256,
    num_attention_heads=4,
    num_moe_experts=4,
    moe_router_topk=2,
    moe_ffn_hidden_size=128,
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    gradient_accumulation_fusion=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type='alltoall',
    expert_model_parallel_size=2,
    expert_tensor_parallel_size=1,
    moe_shortcut_connection=True,
    normalization='RMSNorm',
    attention_backend=AttnBackend.unfused,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    use_cpu_initialization=True,
    **extra,
)
model_kwargs = dict(
    config=config,
    hybrid_stack_spec=wide_residual_hybrid_stack_spec if wide else hybrid_stack_spec,
    vocab_size=256,
    max_sequence_length=32,
    hybrid_layer_pattern='*E*E/*E/*E' if mtp else '*E*E',
    pg_collection=pg,
)
if os.environ.get("MIMO_TEST_MIMO") == "1":
    from megatron.core.models.mimo import MimoModel, MimoModelConfig
    from megatron.core.transformer.spec_utils import ModuleSpec

    model = MimoModel(
        MimoModelConfig(language_model_spec=ModuleSpec(module=HybridModel, params=model_kwargs)),
        cp_group=pg.cp,
        tp_group=pg.tp,
        tp_cp_group=pg.tp_cp,
    )
else:
    model = HybridModel(**model_kwargs)
# Config initializes model weights in BF16; residual controllers initialize in FP32.
# MIMO pretrained-before-wrap preserves constructor/checkpoint dtypes.
model = model.cuda()
original_dtypes = {name: p.dtype for name, p in model.named_parameters()}
if wide:
    assert set(original_dtypes.values()) == {torch.bfloat16, torch.float32}, original_dtypes
model = TorchMimoFSDP(config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model, pg)
optimizer = TorchMimoOptimizer(
    model, OptimizerConfig(optimizer='adam', bf16=True, lr=1e-4), dist.group.WORLD, dist.group.WORLD
)
for step in range(3):
    optimizer.zero_grad()
    for micro in range(4):
        tokens = torch.randint(0, 256, (1, 32), device='cuda')
        positions = torch.arange(32, device='cuda').unsqueeze(0)
        result = model(tokens, positions, None, labels=tokens)
        loss = (result[0] if isinstance(result, tuple) else result).mean()
        assert torch.isfinite(loss)
        loss.backward()
    success, norm, _ = optimizer.step()
    assert success and torch.isfinite(norm) and norm > 0
    assert {name: p.dtype for name, p in model.module.named_parameters()} == original_dtypes
    if dist.get_rank() == 0:
        logging.info(
            "PASS actual HybridStack shortcut EP2 step %d loss %.8f norm %.8f",
            step,
            float(loss.detach()),
            float(norm),
        )
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
