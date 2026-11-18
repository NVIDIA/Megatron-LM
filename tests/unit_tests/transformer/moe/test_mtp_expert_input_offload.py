# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from https://github.com/NVIDIA-NeMo/nv-mistralai-megatron/pull/797
# Source revision: aeab73be76294f8930377aae1e853b4e930e8dd0
# License: BSD-3-Clause; original NVIDIA notices retained (see LICENSE).
"""Actual TE expert/offload/recompute forward and gradient parity on CUDA."""

import unittest
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _build(offload, recompute, selector=None, *, precision_case="override"):
    if selector is None:
        selector = (
            ["fused_group_mlp"] if precision_case == "override" else ["expert_fc1", "moe_act"]
        )
    recipe = RecipeConfig.from_config_dict(
        {
            'configs': {
                'high_precision': {
                    'transformer_engine_config_type': 'TEQuantizationParams',
                    'training_recipe': {},
                }
            },
            'matchers': {
                'mtp_bf16': {
                    'type': 'glob',
                    'enabled': True,
                    'pattern': '*mtp.layers.*',
                    'config': 'high_precision',
                }
            },
        }
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8=None if precision_case == "nvfp4_mtp" else 'e4m3',
        fp8_recipe='mxfp8',
        fp4='e2m1' if precision_case == "nvfp4_mtp" else None,
        fp4_recipe='nvfp4',
        first_last_layers_bf16=precision_case == "first_last_bf16",
        moe_router_load_balancing_type='sinkhorn',
        moe_router_topk=1,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=False,
        quant_recipe=recipe if precision_case == "override" else None,
        fine_grained_activation_offloading=offload,
        offload_modules=selector,
        recompute_granularity='selective' if recompute else None,
        recompute_modules=['moe_act'] if recompute else [],
        gradient_accumulation_fusion=False,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(2, moe_grouped_gemm=True).mlp
    )
    experts = (
        MoELayer(config, spec, name='language.mtp.layers.0.mtp_model_layer.mlp', is_mtp_layer=True)
        .experts.cuda()
        .train()
    )
    assert not experts._with_fused_impl
    assert experts.offload_expert_fc1 == (
        offload and ("fused_group_mlp" in selector or "expert_fc1" in selector)
    )
    assert experts.offload_moe_act == (
        offload and ("fused_group_mlp" in selector or "moe_act" in selector)
    )
    return experts


def _execute(model, original, offload, buffer_gib=0.0, counts=None):
    x = original.clone().requires_grad_()
    probs = torch.full((x.shape[0],), 0.5, device='cuda', dtype=torch.float32, requires_grad=True)
    model.zero_grad(set_to_none=True)
    if offload:
        off_interface.init_chunk_handler(0, None, None, 40000, 0, 1.0, buffer_size_gib=buffer_gib)
    out = x
    counts = torch.tensor(
        counts if counts is not None else [x.shape[0] // 2] * 2, device='cpu', dtype=torch.long
    )
    for _ in range(4):
        value, bias = model(out.clone(), counts, probs)
        assert bias is None
        out = out + value
    transfers = 0
    if offload:
        chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
        transfers = sum(
            isinstance(t, tuple) for g in chunk.offload_groups for t in g._tensors.values()
        )
        assert transfers > 0, 'The case must exercise actual D2H offloading'
        invalid = [
            list(t.shape)
            for g in chunk.offload_groups
            for t in g._tensors.values()
            if isinstance(t, torch.Tensor) and t.numel() > 0 and t.untyped_storage().nbytes() == 0
        ]
        assert not invalid, f'Nonempty saved tensors have freed storage: {invalid}'
        retained = {
            g._name: g.retained_storage_bytes
            for g in chunk.offload_groups
            if g.retained_storage_bytes
        }
        assert not retained, f'The alias guard retained GPU storage: {retained}'
    result = out.detach().clone()
    (out.float().square().mean()).backward()
    torch.cuda.synchronize()
    grads = {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }
    assert grads
    answer = result, x.grad.detach().clone(), probs.grad.detach().clone(), grads, transfers
    if offload:
        assert not chunk._groups_to_reload
        assert all(not g._tensors for g in chunk.offload_groups)
        off_interface.reset()
    return answer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for activation offloading")
class TestMTPExpertInputOffload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_te_min_version("2.14.0"):
            raise unittest.SkipTest("Expert offload tests require TE >= 2.14")
        Utils.initialize_model_parallel(1, 1)

    @classmethod
    def tearDownClass(cls):
        off_interface.reset_instance()
        Utils.destroy_model_parallel()

    def test_bf16_override_offload_preserves_outputs_and_gradients(self):
        """Per-expert slices below the copy threshold must survive parent release."""
        self._check_bf16_override_offload(fp8_autocast=False)

    def test_bf16_override_inside_mxfp8_autocast_preserves_outputs_and_gradients(self):
        """A per-module BF16 override takes precedence over active global MXFP8."""
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP8 requires SM100+ (Blackwell)")
        self._check_bf16_override_offload(fp8_autocast=True)

    def _check_bf16_override_offload(self, *, fp8_autocast: bool):
        for selector in (
            ["fused_group_mlp"],
            ["expert_fc1", "moe_act"],
            ["expert_fc1"],
            ["moe_act"],
        ):
            for recompute in (False, True):
                for buffer_gib in (0.0, 0.0625):
                    with self.subTest(
                        selector=selector, recompute=recompute, buffer_gib=buffer_gib
                    ):
                        off_interface.reset_instance()
                        reference = _build(False, recompute, selector)
                        tested = _build(True, recompute, selector)
                        tested.load_state_dict(reference.state_dict())
                        torch.manual_seed(987)
                        # Warmup, a local expert with no tokens, and a changed routed extent.
                        for rows, counts in ((512, [256, 256]), (512, [0, 512]), (768, [384, 384])):
                            original = (
                                torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) * 0.1
                            )
                            with (
                                get_fp8_context(reference.config) if fp8_autocast else nullcontext()
                            ):
                                expected = _execute(reference, original, False, counts=counts)
                            with get_fp8_context(tested.config) if fp8_autocast else nullcontext():
                                actual = _execute(
                                    tested, original, True, buffer_gib=buffer_gib, counts=counts
                                )
                            for a, b in zip(actual[:3], expected[:3]):
                                torch.testing.assert_close(a, b, atol=0, rtol=0)
                            self.assertEqual(actual[3].keys(), expected[3].keys())
                            for name in actual[3]:
                                torch.testing.assert_close(
                                    actual[3][name], expected[3][name], atol=0, rtol=0
                                )
                        del reference, tested, original, expected, actual
                        off_interface.reset_instance()
                        torch.cuda.empty_cache()

    def test_runtime_bf16_modes_preserve_outputs_and_gradients(self):
        """Global quantization does not imply a quantized expert invocation."""
        for precision_case in ("nvfp4_mtp", "first_last_bf16"):
            for recompute in (False, True):
                for buffer_gib in (0.0, 0.0625):
                    with self.subTest(
                        precision_case=precision_case, recompute=recompute, buffer_gib=buffer_gib
                    ):
                        if precision_case == "nvfp4_mtp" and not is_te_min_version("2.7.0"):
                            self.skipTest("NVFP4 requires Transformer Engine >= 2.7")
                        off_interface.reset_instance()
                        reference = _build(False, recompute, precision_case=precision_case)
                        tested = _build(True, recompute, precision_case=precision_case)
                        tested.load_state_dict(reference.state_dict())

                        def context(model):
                            # Match the callers: MTP currently omits FP4 autocast,
                            # while first/last BF16 layers use the layer-aware context.
                            if precision_case == "first_last_bf16":
                                return get_fp8_context(model.config, layer_no=0)
                            return nullcontext()

                        torch.manual_seed(987)
                        for rows, counts in ((512, [256, 256]), (512, [0, 512]), (768, [384, 384])):
                            original = (
                                torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) * 0.1
                            )
                            with context(reference):
                                expected = _execute(reference, original, False, counts=counts)
                            with context(tested):
                                actual = _execute(
                                    tested, original, True, buffer_gib=buffer_gib, counts=counts
                                )
                            for a, b in zip(actual[:3], expected[:3]):
                                torch.testing.assert_close(a, b, atol=0, rtol=0)
                            self.assertEqual(actual[3].keys(), expected[3].keys())
                            for name in actual[3]:
                                torch.testing.assert_close(
                                    actual[3][name], expected[3][name], atol=0, rtol=0
                                )
                        del reference, tested, original, expected, actual
                        off_interface.reset_instance()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
