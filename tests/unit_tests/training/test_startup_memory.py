# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""CPU scheduling/opt-in checks plus a GPU test of original state preservation."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'startup_memory_under_test', ROOT / 'megatron/training/startup_memory.py'
)
assert SPEC is not None and SPEC.loader is not None
startup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(startup)


class StartupMemoryTests(unittest.TestCase):

    def test_tensor_and_te_generator_tracker_states_are_cloned_without_aliasing(self):
        class TensorState:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return TensorState(self.value)

        class GeneratorState:
            def __init__(self, value):
                self.value = value

            def clone_state(self):
                return GeneratorState(self.value)

            def get_state(self):
                return TensorState(self.value)

        tensor, generator = TensorState(17), GeneratorState(29)
        original = {'native': tensor, 'te': generator}
        torch = SimpleNamespace(Tensor=TensorState, Generator=GeneratorState)
        with patch.dict(sys.modules, {'torch': torch}):
            cloned = startup._clone_tracker_states(original)
            self.assertIsInstance(cloned['native'], TensorState)
            self.assertIsInstance(cloned['te'], GeneratorState)
            self.assertIsNot(cloned['native'], tensor)
            self.assertIsNot(cloned['te'], generator)
            cloned['native'].value = 101
            cloned['te'].value = 202
            self.assertEqual(startup._tracker_state_bytes(tensor).value, 17)
            self.assertEqual(startup._tracker_state_bytes(generator).value, 29)
            self.assertIs(original['native'], tensor)
            self.assertIs(original['te'], generator)
            with self.assertRaises(TypeError):
                startup._clone_tracker_states({'unknown': object()})

    def test_prewarm_preserves_original_tensors_and_rng_on_gpu(self):
        self._check_gpu_prewarm(te_rng_tracker=False)

    def test_prewarm_preserves_original_te_generator_objects_and_rng_on_gpu(self):
        self._check_gpu_prewarm(te_rng_tracker=True)

    def _check_gpu_prewarm(self, te_rng_tracker):
        try:
            import torch
        except ImportError:
            self.skipTest('Native Torch/CUDA required for state-preservation execution')
        if not torch.cuda.is_available():
            self.skipTest('CUDA required; CPU scheduling checks remain independent')
        import random

        import numpy as np

        from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.module import Float16Module
        from megatron.core.transformer.transformer_config import TransformerConfig
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        model_parallel_cuda_manual_seed(123, te_rng_tracker=te_rng_tracker, force_reset_rng=True)
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                mamba_num_heads=4,
                mamba_head_dim=32,
                mamba_num_groups=4,
                mamba_state_dim=128,
                bf16=True,
                params_dtype=torch.bfloat16,
                gradient_accumulation_fusion=True,
                recompute_granularity='selective',
                recompute_modules=['gdp_in_proj', 'gdp_qkv'],
            )
            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'dp_cp'])
            model = GatedDeltaProductMixer(
                config,
                gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
                128,
                layer_number=1,
                pg_collection=pg,
            ).cuda()
            model = Float16Module(config, model).module
            before = []
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.5)
                parameter.main_grad = torch.full(parameter.shape, 0.25, device='cuda')
                before.append(
                    (
                        parameter.detach().clone(),
                        parameter.grad.clone(),
                        parameter.main_grad.clone(),
                    )
                )
            tracker = get_cuda_rng_tracker()
            original_states = tracker.get_states()
            tracked_bytes = {
                key: (value.get_state() if isinstance(value, torch.Generator) else value).clone()
                for key, value in original_states.items()
            }
            if te_rng_tracker:
                self.assertTrue(
                    any(isinstance(value, torch.Generator) for value in original_states.values())
                )
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            precision = torch.get_float32_matmul_precision()
            self.assertTrue(startup.prewarm_gdp_kernels([model], 64, 1))
            self.assertTrue(config.gradient_accumulation_fusion)
            restored = tracker.get_states()
            self.assertEqual(set(restored), set(original_states))
            for key, original in original_states.items():
                self.assertIs(restored[key], original)
                actual = (
                    restored[key].get_state()
                    if isinstance(restored[key], torch.Generator)
                    else restored[key]
                )
                self.assertTrue(torch.equal(actual, tracked_bytes[key]))
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_rng))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
            self.assertEqual(torch.get_float32_matmul_precision(), precision)
            for parameter, saved in zip(model.parameters(), before):
                self.assertTrue(torch.equal(parameter, saved[0]))
                self.assertTrue(torch.equal(parameter.grad, saved[1]))
                self.assertTrue(torch.equal(parameter.main_grad, saved[2]))
        finally:
            Utils.destroy_model_parallel()


if __name__ == '__main__':
    unittest.main()
