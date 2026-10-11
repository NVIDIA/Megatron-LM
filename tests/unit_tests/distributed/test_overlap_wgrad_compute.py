# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.module._common import WeightGradStore

    HAVE_WGRAD_STREAM = hasattr(WeightGradStore, "set_stream")
except ImportError:
    HAVE_WGRAD_STREAM = False

SEQ_LEN = 64
VOCAB_SIZE = 256


def _build_ddp_model(overlap_wgrad_compute: bool, num_moe_experts, output_layer_chunks=None):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=num_moe_experts,
        moe_router_topk=2,
        moe_grouped_gemm=num_moe_experts is not None,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        add_bias_linear=False,
        bf16=True,
        gradient_accumulation_fusion=True,
        overlap_wgrad_compute=overlap_wgrad_compute,
        overlap_wgrad_compute_output_layer_chunks=output_layer_chunks,
        use_cpu_initialization=True,
    )
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=num_moe_experts is not None
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        parallel_output=True,
    ).cuda()
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, use_distributed_optimizer=True, bucket_size=10000
    )
    return DistributedDataParallel(config, ddp_config=ddp_config, module=model)


def _grads_after_one_step(model: DistributedDataParallel, data):
    model.zero_grad_buffer()
    loss = model(**data).sum()
    loss.backward()
    # Every deferred weight gradient was released and registered with its bucket group.
    for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
        assert set(bucket_group.per_param_grad_ready_counts) == bucket_group.params
    model.finish_grad_sync()
    torch.cuda.synchronize()
    return loss.detach(), {name: p.main_grad.clone() for name, p in model.named_parameters()}


@pytest.mark.skipif(not HAVE_WGRAD_STREAM, reason="needs TE WeightGradStore.set_stream")
@pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="needs TE >= 2.3.0")
class TestOverlapWgradCompute:
    """overlap_wgrad_compute: the weight gradients computed on the side stream (and reduced
    from it) equal the inline ones, for dense and MoE layers, the router and the chunked
    output layer."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("num_moe_experts", [None, 4])
    @pytest.mark.parametrize("output_layer_chunks", [None, 3])
    def test_gradients_match_inline_wgrad(self, num_moe_experts, output_layer_chunks):
        torch.manual_seed(0)
        data = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device="cuda"),
            "position_ids": torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(2, -1),
            "attention_mask": torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.bool, device="cuda"),
        }
        data["labels"] = data["input_ids"].clone()

        ref_model = _build_ddp_model(False, num_moe_experts)
        model = _build_ddp_model(True, num_moe_experts, output_layer_chunks)
        with torch.no_grad():
            for p_ref, p in zip(ref_model.parameters(), model.parameters()):
                p.copy_(p_ref)
        assert model.wgrad_stream is not None
        assert model.module.output_layer.wgrad_store is not None

        ref_loss, ref_grads = _grads_after_one_step(ref_model, data)
        loss, grads = _grads_after_one_step(model, data)

        torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
        for name, ref_grad in ref_grads.items():
            if name.endswith("output_layer.weight"):
                # Chunked torch GEMMs instead of the fused apex kernel: same fp32 accumulation,
                # the summation order may differ.
                torch.testing.assert_close(grads[name], ref_grad, rtol=1e-3, atol=1e-3)
            else:
                torch.testing.assert_close(grads[name], ref_grad, rtol=0, atol=0, msg=name)

    def test_second_iteration_reuses_the_stream(self):
        data = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device="cuda"),
            "position_ids": torch.arange(SEQ_LEN, device="cuda").unsqueeze(0),
            "attention_mask": None,
        }
        data["labels"] = data["input_ids"].clone()
        model = _build_ddp_model(True, None)
        _, first = _grads_after_one_step(model, data)
        _, second = _grads_after_one_step(model, data)
        for name, grad in first.items():
            torch.testing.assert_close(second[name], grad, rtol=0, atol=0, msg=name)
