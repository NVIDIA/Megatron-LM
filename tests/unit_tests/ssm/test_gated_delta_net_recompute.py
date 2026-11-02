# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetTestBase):
    def test_selective_recompute_norm_out(self):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        def build_gdn(config):
            gdn_spec = get_experimental_attention_variant_module_spec(config=config)
            gdn = gdn_spec.module(
                config,
                submodules=gdn_spec.submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            return gdn.cuda().bfloat16()

        def run(gdn, hidden_states):
            output, _ = gdn(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach()
                for name, param in gdn.named_parameters()
                if param.grad is not None
            }
            input_grad = hidden_states.grad.detach().clone()
            return output.detach(), grads, input_grad

        micro_batch_size = 2
        seq_length = 64
        base_config = copy.deepcopy(self.transformer_config)
        rec_config = copy.deepcopy(self.transformer_config)
        rec_config.recompute_granularity = "selective"
        rec_config.recompute_modules = ["gdn_norm_out"]

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = torch.randn(
            (
                seq_length // self.sp_size // self.cp_size,
                micro_batch_size,
                self.gdn.config.hidden_size,
            ),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # --- Baseline (no recompute) ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        base_gdn = build_gdn(base_config)
        assert base_gdn.recompute_norm_out is False
        base_output, base_grads, base_input_grad = run(base_gdn, hidden_states)
        hidden_states.grad = None
        assert base_gdn.norm_out_checkpoint is None
        del base_gdn
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        rec_gdn = build_gdn(rec_config)
        assert rec_gdn.recompute_norm_out is True
        rec_output, rec_grads, rec_input_grad = run(rec_gdn, hidden_states)
        assert rec_gdn.norm_out_checkpoint is not None

        rank = torch.distributed.get_rank()
        assert torch.equal(rec_output, base_output), f"Output not identical ({rank=})"
        assert torch.equal(rec_input_grad, base_input_grad), f"Input grad not identical ({rank=})"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            assert torch.equal(
                rec_grads[name], base_grads[name]
            ), f"Grad not identical for {name} ({rank=})"
