# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise semantic Muon through the real DDP and layer-wise optimizer factory."""

import os
import sys

import pytest
import torch
from torch import nn

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.muon_layout import MuonProjectionLayout
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
    pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="Requires emerging_optimizers"),
    pytest.mark.skipif(int(os.getenv("WORLD_SIZE", "1")) < 2, reason="Requires multiple ranks"),
]


@pytest.fixture(autouse=True)
def setup_model_parallel():
    """Use real NCCL data parallelism; keep semantic matrices local to each TP rank."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def test_cli_selects_layerwise_for_per_head_muon(monkeypatch):
    """The documented CLI selects whole-parameter ownership, not flattened shards."""
    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_muon_layerwise_integration.py",
            "--num-layers",
            "1",
            "--hidden-size",
            "32",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--train-iters",
            "1",
            "--lr",
            "0.003",
            "--tokenizer-type",
            "NullTokenizer",
            "--vocab-size",
            "32",
            "--bf16",
            "--optimizer",
            "muon",
            "--muon-split-qkv-per-head",
            "--use-distributed-optimizer",
        ],
    )
    args = validate_args(parse_args())
    assert args.muon_split_qkv_per_head
    assert args.use_layer_wise_distributed_optimizer
    assert not args.use_distributed_optimizer


@pytest.mark.parametrize("use_param_layout", [False, True])
def test_layerwise_preserves_semantic_masters_and_updates(use_param_layout):
    """Controlled reduced gradients match unsharded updates and real DP parameter sync."""
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(391)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    model = nn.Module()
    model.config = config
    model.projections = nn.ModuleList(
        nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16) for _ in range(8)
    )
    # Exercise separate Adam/DistributedOptimizer routing in the full-layout case.
    model.bias = nn.Parameter(torch.zeros(8, device="cuda", dtype=torch.bfloat16))
    for projection in model.projections:
        projection.weight.muon_layout = MuonProjectionLayout((4, 4), (False, True))
    references = {}
    for name, param in model.named_parameters():
        reference = nn.Parameter(param.detach().float().clone())
        if hasattr(param, "muon_layout"):
            reference.muon_layout = param.muon_layout
        references[name] = reference

    opt_config = OptimizerConfig(
        optimizer="muon",
        lr=0.003,
        weight_decay=0.1,
        bf16=True,
        clip_grad=0.0,
        use_layer_wise_distributed_optimizer=True,
        muon_split_qkv_per_head=True,
        muon_momentum=0.93,
        muon_nesterov=True,
        muon_fp32_matmul_prec="highest",
        adam_beta1=0.8,
        adam_beta2=0.97,
        adam_eps=1e-7,
    )
    ddp = wrap_model_chunks_with_ddp(
        [model],
        config,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True),
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=use_param_layout,
        pg_collection=pg,
    )[0]
    optimizer = get_megatron_optimizer(
        opt_config, [ddp], use_gloo_process_groups=False, pg_collection=pg
    )
    children = [optimizer, *optimizer.chained_optimizers]
    layerwise = next(
        child for child in children if isinstance(child, LayerWiseDistributedOptimizer)
    )
    assert layerwise.use_buffer_param_sync == use_param_layout
    assert any(isinstance(child, DistributedOptimizer) for child in children) == use_param_layout
    raw_muon = next(
        child.optimizer
        for child in layerwise.chained_optimizers
        if isinstance(child.optimizer, TensorParallelMuon)
    )
    masters = [p for group in raw_muon.param_groups for p in group["params"]]
    assert all(p.shape == (8, 8) and p.dtype == torch.float32 for p in masters)
    owned_count = torch.tensor(len(masters), device="cuda")
    torch.distributed.all_reduce(owned_count, group=pg.dp_cp)
    assert owned_count.item() == 8

    reference_muon = TensorParallelMuon(
        [p for p in references.values() if p.ndim == 2],
        lr=0.003,
        weight_decay=0.1,
        momentum=0.93,
        nesterov=True,
        split_qkv=True,
        split_qkv_per_head=True,
        fp32_matmul_prec="highest",
        adamw_betas=(0.8, 0.97),
        adamw_eps=1e-7,
    )
    reference_adam = torch.optim.AdamW(
        [references["bias"]], lr=0.003, betas=(0.8, 0.97), eps=1e-7, weight_decay=0.0
    )
    for step in range(3):
        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        # Supply identical already-reduced gradients to each DP replica. This
        # isolates optimizer ownership/update/gather from model/backward numerics.
        for index, (name, param) in enumerate(model.named_parameters()):
            grad = torch.arange(param.numel(), device="cuda", dtype=torch.float32)
            grad = torch.sin(grad + index + step).reshape(param.shape)
            param.main_grad.copy_(grad)
            references[name].grad = grad.clone()
        reference_muon.step()
        reference_adam.step()
        success, _, _ = optimizer.step()
        assert success
        for name, param in model.named_parameters():
            reference = references[name]
            torch.testing.assert_close(param, reference.to(param.dtype), atol=0, rtol=0)
            master = getattr(param, "main_param", None)
            if master is None or master not in raw_muon.state:
                continue
            assert master.shape == param.shape
            assert master.muon_layout == param.muon_layout
            state, expected = raw_muon.state[master], reference_muon.state[reference]
            assert state["step"] == step + 1
            for key in ("momentum_buffer", "gate_exp_avg", "gate_exp_avg_sq"):
                assert state[key].shape == master.shape
                torch.testing.assert_close(state[key], expected[key], atol=1e-6, rtol=1e-6)
