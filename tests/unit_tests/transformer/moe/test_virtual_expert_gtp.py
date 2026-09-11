# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Four-rank virtual-expert/GTP persistent wgrad integration (also runs on GB200)."""

import gc
import os

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core import parallel_state as ps
from megatron.core.activations import squared_relu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    _VirtualExperts,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != 4 or not fused_a2a.HAVE_HYBRIDEP,
        reason="requires four ranks and HybridEP",
    ),
]


class _ExpertStack(MegatronModule):
    def __init__(self, config, pg_collection, repeat_last=False):
        super().__init__(config)
        self.repeat_last = repeat_last
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=True
        )
        spec.submodules.input_layernorm = IdentityOp
        spec.submodules.self_attention = IdentityOp
        spec.submodules.self_attn_bda = IdentityFuncOp
        self.layers = torch.nn.ModuleList(
            [
                MoETransformerLayer(
                    config,
                    spec.submodules,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                    name=f"layers.{i}",
                )
                for i in range(config.num_layers)
            ]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask=None)
        if self.repeat_last:
            hidden_states, _ = self.layers[-1](hidden_states, attention_mask=None)
        return hidden_states


@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_virtual_expert_gtp_persistent_wgrads_match_recycled_scratch(monkeypatch, mxfp8):
    """Real DDP/GTP gradients and outputs agree with recycled scratch across changed inputs."""
    if mxfp8 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    Utils.initialize_distributed()
    ps.destroy_model_parallel()
    gtp._GTP_PARAMS.clear()
    ps.initialize_model_parallel(
        expert_model_parallel_size=2, expert_gtp_remat_size=2, gtp_remat_size=2 if mxfp8 else 1
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=3,
        hidden_size=512,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        expert_model_parallel_size=2,
        expert_gtp_weight_remat_size=2,
        gtp_weight_remat_size=2 if mxfp8 else 1,
        gtp_remat_opt_in_modules=["moe_latent_proj"] if mxfp8 else [],
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        gated_linear_unit=not mxfp8,
        activation_func=squared_relu if mxfp8 else F.silu,
        use_fused_weighted_squared_relu=mxfp8,
        moe_latent_size=512 if mxfp8 else None,
        moe_shared_expert_intermediate_size=512 if mxfp8 else None,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
        fp8="e4m3" if mxfp8 else None,
        fp8_recipe="mxfp8",
        fp8_param=mxfp8,
        moe_router_padding_for_quantization=mxfp8,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups()
    with get_fp8_context(config, is_init=True):
        module = _ExpertStack(config, pg, repeat_last=mxfp8).cuda()
    model = DDP(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            check_for_nan_in_grad=False,
            average_in_collective=False,
        ),
        module=module,
        pg_collection=pg,
    )
    gtp.classify_gtp_remat_chains(
        model, cuda_graph_modules=config.cuda_graph_modules, cuda_graph_impl="none"
    )
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", mxfp8)
    monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
    params = list(model.parameters())
    experts = [p for p in params if getattr(p, "is_routed_expert", False)]
    assert experts and all(p.gtp_remat_size == 2 for p in experts)
    assert all("ungraphed" in p.chain_id for p in experts)
    assert all(is_mxfp8tensor(p) == mxfp8 and p.main_grad.dtype == torch.bfloat16 for p in experts)

    def run(seed):
        model.zero_grad_buffer()
        rng = torch.Generator(device="cuda").manual_seed(seed + ps.get_expert_model_parallel_rank())
        x = torch.randn(
            32,
            1,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
            generator=rng,
        )
        dy = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)
        model.set_is_first_microbatch()
        with get_fp8_context(config):
            y = model(x)
        saved_output = y.detach().clone()
        saved_input = x.detach().clone()
        # Distinct upstream gradients exercise every output component and router gradient.
        (y.float() * dy).sum().backward()
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        model.finish_grad_sync()
        torch.cuda.synchronize()
        return saved_output, x.grad.clone(), [p.main_grad.clone() for p in params], saved_input

    get_weight_table = _VirtualExperts.get_weight_table
    reference_layers = set()

    def recycled_reference_table(owner, fc_layer, key, sources):
        # The legacy control uses moving scratch; explicitly rebuild only its gradient tables.
        reference_layers.add(owner)
        if key == "grad":
            owner._tables[fc_layer].pop(key, None)
        return get_weight_table(owner, fc_layer, key, sources)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_VirtualExperts, "get_weight_table", recycled_reference_table)
            for param_type in {type(p) for p in experts}:
                get_wgrad_tensor = param_type.get_wgrad_tensor
                patch.setattr(
                    param_type,
                    "get_wgrad_tensor",
                    lambda p, method=get_wgrad_tensor, **kwargs: method(p),
                )
            refs = [run(seed) for seed in (21, 22)]
            assert not any(hasattr(p, "_gtp_wgrad_ring_slot") for p in experts)
        while reference_layers:
            for tables in reference_layers.pop()._tables:
                tables.pop("grad", None)
        for result in refs:
            for tensor in (result[0], result[1], *result[2]):
                assert torch.isfinite(tensor).all() and tensor.float().norm() > 0

        slots = None
        for seed in (21, 22, 21, 22):
            result = run(seed)
            reference = refs[seed - 21]
            torch.testing.assert_close(result[3], reference[3], atol=0, rtol=0)
            for name, actual, expected in zip(
                ("output", "input gradient", *(name for name, _ in model.named_parameters())),
                (result[0], result[1], *result[2]),
                (reference[0], reference[1], *reference[2]),
            ):
                rel_error = (actual.float() - expected.float()).norm() / expected.float().norm()
                assert rel_error < 0.02, f"{name}: relative L2 error {rel_error}"
                torch.testing.assert_close(
                    actual, expected, atol=0.01, rtol=0.02, msg=lambda msg: f"{name}: {msg}"
                )
            pointers = [p._gtp_wgrad_ring_slot.tensor.data_ptr() for p in experts]
            if slots is None:
                slots = pointers
                # Three layers share two slots per FC role/expert/shape domain.
                assert len(set(slots)) < len(slots)

                def check_static_table(owner, fc_layer, key, sources):
                    tables = tuple(owner._tables[fc_layer].values())
                    result = get_weight_table(owner, fc_layer, key, sources)
                    assert any(result is tensor for tensor, _ in tables)
                    return result

                monkeypatch.setattr(_VirtualExperts, "get_weight_table", check_static_table)
            else:
                assert pointers == slots
    finally:
        torch.cuda.synchronize()
        del model, module, params, experts
        reference_layers.clear()
        FP8GlobalStateManager.reset()
        gc.collect()
        VirtualExpertLoadBalancer.finalize()
        gtp.reset_gtp_state()
        gtp._GTP_PARAMS.clear()
        ps.destroy_model_parallel()
        fused_a2a.reset_hybrid_ep_buffer()
