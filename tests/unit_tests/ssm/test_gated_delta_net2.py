# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN2 correctness in a separate CI bucket from GDN."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_FLA_GDN2
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_cases import GatedDeltaNetModuleTestCases
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("use_gdn2", [True], ids=["gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetModuleTestCases):
    """GDN2 module correctness across the original parallel layouts."""


@pytest.mark.skipif(not HAVE_FLA_GDN2, reason="FLA with GDN2 support is not installed.")
@pytest.mark.internal
def test_gdn2_single_token_sequences_in_cuda_graph():
    """In inference, packed sequences of one token each, as an MTP draft step without a context
    packs its requests, run without host syncs, so a CUDA graph captures them, and give every
    token the output it has as a sequence of its own."""
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=512,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            num_attention_heads=4,
            activation_func=F.silu,
            bf16=True,
            experimental_attention_variant="gdn2",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        gdn2 = spec.module(
            config,
            submodules=spec.submodules,
            layer_number=1,
            conv_init=1.0,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        )
        gdn2 = gdn2.cuda().bfloat16()
        num_tokens = 6
        hidden_states = torch.randn(
            (num_tokens, 1, config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        packed_seq_params = PackedSeqParams.single_token_sequences(
            num_tokens, device=hidden_states.device
        )

        with torch.inference_mode(), InferenceMode.active():
            alone_output = torch.cat(
                [gdn2(hidden_states[i : i + 1], None)[0] for i in range(num_tokens)]
            )
            # Warm up, then capture the packed forward.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                gdn2(hidden_states, None, packed_seq_params=packed_seq_params)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packed_output, _ = gdn2(hidden_states, None, packed_seq_params=packed_seq_params)
            graph.replay()

        # The recurrent and chunked kernels round their bf16 intermediates differently.
        torch.testing.assert_close(
            packed_output, alone_output, rtol=0, atol=2e-2 * alone_output.abs().max().item()
        )
    finally:
        Utils.destroy_model_parallel()
