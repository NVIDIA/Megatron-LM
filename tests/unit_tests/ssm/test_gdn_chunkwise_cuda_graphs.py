# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Capture and replay complete chunkwise GDN forward and backward passes."""

from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from tests.unit_tests.test_utilities import Utils

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE_GDN
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.context_parallel import gdn_layout
from megatron.core.ssm.gated_delta_net import chunkwise
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


@pytest.mark.skipif(not HAVE_TE_GDN, reason="Transformer Engine GDN is unavailable")
@pytest.mark.parametrize("cp_size,packed", [(2, False), (4, True)])
def test_chunkwise_gdn_cuda_graph(cp_size: int, packed: bool) -> None:
    """Replay new activations and loss gradients against eager full-module results."""
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    graph = None
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            normalization="RMSNorm",
            activation_func=F.silu,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            context_parallel_size=cp_size,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
            gdn_kernel_backend="transformer_engine",
            gdn_pre_gated_delta_rule_fusion=True,
            gdn_gated_output_norm_fusion=True,
            gdn_cp_mode="chunkwise",
            gradient_accumulation_fusion=False,
            cuda_graph_impl="local",
        )
        groups = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        module = (
            spec.module(config, submodules=spec.submodules, layer_number=1, pg_collection=groups)
            .cuda()
            .bfloat16()
        )
        parameters = tuple(module.parameters())
        hidden = torch.randn(1024 // cp_size, 1, 256, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_()
        output_grad = torch.randn_like(hidden)
        metadata = None
        if packed:
            cu = torch.tensor([0, 192, 512, 1024], dtype=torch.int32, device="cuda")
            metadata = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module.zero_grad(set_to_none=True)
                hidden.grad = None
                module(hidden, None, packed_seq_params=metadata)[0].backward(output_grad)
        torch.cuda.current_stream().wait_stream(stream)
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        if packed:
            cu = cu.clone()
            metadata = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output, _ = module(hidden, None, packed_seq_params=metadata)
            output.backward(output_grad)
        captured_grads = (hidden.grad, *(parameter.grad for parameter in parameters))
        # Captured metadata construction must survive eager cache eviction.
        with gdn_layout._lock:
            gdn_layout._entries.clear()
        for boundaries in ([0, 8, 520, 1024], [0, 256, 384, 1024]):
            if packed:
                cu.copy_(torch.tensor(boundaries, dtype=torch.int32, device="cuda"))
            values, incoming = torch.randn_like(hidden), torch.randn_like(output_grad)
            eager_hidden = values.detach().requires_grad_()
            # Compare the same document ownership; eager training normally uses
            # globally contiguous shards with different local parameter gradients.
            with patch.object(
                chunkwise,
                "get_chunkwise_layout",
                partial(gdn_layout.get_chunkwise_layout, use_cuda_graphs=True),
            ):
                eager_output, _ = module(eager_hidden, None, packed_seq_params=metadata)
                eager_grads = torch.autograd.grad(
                    eager_output, (eager_hidden, *parameters), grad_outputs=incoming
                )
            with torch.no_grad():
                hidden.copy_(values)
                output_grad.copy_(incoming)
                for gradient in captured_grads:
                    gradient.zero_()
            graph.replay()
            torch.testing.assert_close(output, eager_output, atol=2e-3, rtol=2e-3)
            for actual, expected in zip(captured_grads, eager_grads):
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    finally:
        if graph is not None:
            graph.reset()
        Utils.destroy_model_parallel()
