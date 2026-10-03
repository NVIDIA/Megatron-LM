# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetTestBase):
    @pytest.mark.parametrize(
        "recompute_modules",
        [
            ["gdn_norm_out"],
            ["gdn_in_proj"],
            ["gdn_qkv"],
            ["gdn_in_proj", "gdn_qkv"],
            ["gdn_in_proj", "gdn_qkv", "gdn_norm_out"],
        ],
        ids=lambda modules: "+".join(modules),
    )
    def test_selective_recompute(self, recompute_modules):
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
        rec_config.recompute_modules = recompute_modules

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
        assert base_gdn.recompute_in_proj is False
        assert base_gdn.recompute_qkv is False
        base_output, base_grads, base_input_grad = run(base_gdn, hidden_states)
        hidden_states.grad = None
        assert base_gdn.norm_out_checkpoint is None
        del base_gdn
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        rec_gdn = build_gdn(rec_config)
        assert rec_gdn.recompute_norm_out == ("gdn_norm_out" in recompute_modules)
        assert rec_gdn.recompute_in_proj == ("gdn_in_proj" in recompute_modules)
        assert rec_gdn.recompute_qkv == ("gdn_qkv" in recompute_modules)

        # Every requested checkpoint must release its outputs during the forward pass.
        discarded = []
        original_discard = CheckpointWithoutOutput._discard_outputs

        def recording_discard(ckpt):
            original_discard(ckpt)
            discarded.append([out.untyped_storage().nbytes() for out in ckpt.outputs])

        with mock.patch.object(CheckpointWithoutOutput, "_discard_outputs", recording_discard):
            rec_output, rec_grads, rec_input_grad = run(rec_gdn, hidden_states)
        assert len(discarded) == len(recompute_modules)
        assert all(nbytes == 0 for outputs in discarded for nbytes in outputs)
        if "gdn_norm_out" in recompute_modules:
            assert rec_gdn.norm_out_checkpoint is not None

        rank = torch.distributed.get_rank()
        assert torch.equal(rec_output, base_output), f"Output not identical ({rank=})"
        assert torch.equal(rec_input_grad, base_input_grad), f"Input grad not identical ({rank=})"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            assert torch.equal(
                rec_grads[name], base_grads[name]
            ), f"Grad not identical for {name} ({rank=})"


def _gdn_config(**overrides):
    """A minimal GDN TransformerConfig; constructing it runs the recompute/offload checks."""
    kwargs = dict(
        hidden_size=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_layers=1,
        linear_attention_freq=1,
        num_attention_heads=4,
        normalization="RMSNorm",
        activation_func=F.silu,
        experimental_attention_variant="gdn",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("module", ["gdn_in_proj", "gdn_qkv"])
def test_gdn_recompute_modules_are_accepted(module):
    config = _gdn_config(recompute_granularity="selective", recompute_modules=[module])
    assert module in config.recompute_modules


@pytest.mark.parametrize("module", ["gdn_in_proj", "gdn_qkv"])
def test_gdn_recompute_modules_require_gdn_variant(module):
    with pytest.raises(ValueError, match=f"{module} in recompute_modules is only supported"):
        TransformerConfig(
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            recompute_granularity="selective",
            recompute_modules=[module],
        )


def test_gdn_qkv_recompute_rejects_pre_gated_delta_rule_fusion():
    with pytest.raises(ValueError, match="gdn_qkv in recompute_modules is not supported"):
        _gdn_config(
            recompute_granularity="selective",
            recompute_modules=["gdn_qkv"],
            gdn_pre_gated_delta_rule_fusion=True,
        )


def test_gdn_in_proj_recompute_allows_pre_gated_delta_rule_fusion():
    config = _gdn_config(
        recompute_granularity="selective",
        recompute_modules=["gdn_in_proj"],
        gdn_pre_gated_delta_rule_fusion=True,
    )
    assert config.recompute_modules == ["gdn_in_proj"]


def test_gdn_qkv_offload_is_accepted():
    config = _gdn_config(fine_grained_activation_offloading=True, offload_modules=["gdn_qkv"])
    assert config.offload_modules == ["gdn_qkv"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (dict(experimental_attention_variant=None), "gdn_qkv in offload_modules is only supported"),
        (
            dict(gdn_pre_gated_delta_rule_fusion=True),
            "gdn_qkv in offload_modules is not supported with gdn_pre_gated_delta_rule_fusion",
        ),
        (
            dict(recompute_granularity="selective", recompute_modules=["gdn_in_proj"]),
            "gdn_qkv cannot be set in offload_modules together with gdn_in_proj",
        ),
    ],
    ids=["non-gdn", "pre-gdr-fusion", "with-gdn-in-proj"],
)
def test_gdn_qkv_offload_rejections(overrides, match):
    with pytest.raises(ValueError, match=match):
        _gdn_config(
            fine_grained_activation_offloading=True, offload_modules=["gdn_qkv"], **overrides
        )
