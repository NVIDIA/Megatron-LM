# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical regressions for fused Muon/AdamW projections; also runnable on CPU.

Run without the GPU-only repository fixtures:
    python -m unittest tests.unit_tests.test_muon_semantics
"""

import copy
import os
import unittest
from types import SimpleNamespace

import torch

from megatron.core.muon_layout import MuonProjectionLayout
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
)


@unittest.skipUnless(HAVE_EMERGING_OPTIMIZERS, 'emerging_optimizers is not installed')
class TestMuonSemantics(unittest.TestCase):
    def setUp(self):
        self.old_device = torch.get_default_device()
        if os.environ.get('MUON_TEST_DEVICE') == 'cuda':
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
            torch.set_default_device('cuda')
        torch.manual_seed(193)

    def tearDown(self):
        torch.set_default_device(self.old_device)

    def optimizer(self, params, **kwargs):
        return TensorParallelMuon(
            params,
            lr=0.003,
            momentum=0.93,
            nesterov=True,
            weight_decay=0.1,
            split_qkv=True,
            split_qkv_per_head=True,
            fp32_matmul_prec='highest',
            adamw_betas=(0.8, 0.97),
            adamw_eps=1e-7,
            **kwargs,
        )

    def check_reference(self, layout, steps=5, width=8, **kwargs):
        """Compare every row with independently constructed Muon or torch AdamW."""
        p = torch.nn.Parameter(torch.randn(sum(layout.splits), width))
        p.muon_layout = layout
        opt = self.optimizer([p], **kwargs)
        refs = [torch.nn.Parameter(t.clone()) for t in p.detach().split(layout.splits)]
        optimizers = []
        for q, adam in zip(refs, layout.adamw):
            if adam:
                o = torch.optim.AdamW(
                    [q], lr=0.003, betas=(0.8, 0.97), eps=1e-7, weight_decay=0.1, foreach=False
                )
            else:
                o = TensorParallelMuon(
                    [q],
                    lr=0.003,
                    momentum=0.93,
                    nesterov=True,
                    weight_decay=0.1,
                    fp32_matmul_prec='highest',
                    **kwargs,
                )
            optimizers.append(o)
        seen = []
        real_ns = opt.scaled_orthogonalize_fn

        def observe(g, *args, **kwargs):
            seen.append(tuple(g.shape))
            self.assertGreater(g.shape[-2], 1, 'Scalar control row reached NS')
            return real_ns(g, *args, **kwargs)

        opt.scaled_orthogonalize_fn = observe
        for step in range(steps):
            # Include zeros, a nonstationary LR and nonzero weight decay.
            grad = torch.randn_like(p) if step != 2 else torch.zeros_like(p)
            lr = 0.003 * (step + 1) / steps
            opt.param_groups[0]['lr'] = lr
            p.grad = grad.clone()
            opt.step()
            for q, o, g in zip(refs, optimizers, grad.split(layout.splits)):
                o.param_groups[0]['lr'] = lr
                q.grad = g.clone()
                o.step()
            torch.testing.assert_close(p, torch.cat(refs), atol=2e-6, rtol=2e-6)
        self.assertEqual(opt.state[p]['step'], steps)
        for key, v in opt.state[p].items():
            if key != 'step':
                self.assertEqual(v.shape, p.shape)
        self.assertTrue(seen or all(layout.adamw))
        return p, opt

    def test_config_rejects_incompatible_per_head_modes(self):
        with self.assertRaisesRegex(ValueError, "requires muon_split_qkv"):
            OptimizerConfig(muon_split_qkv_per_head=True, muon_split_qkv=False)
        with self.assertRaisesRegex(ValueError, "layer_sharded"):
            OptimizerConfig(
                optimizer="muon",
                muon_split_qkv_per_head=True,
                muon_tp_mode="layer_sharded",
                use_layer_wise_distributed_optimizer=True,
            )

    def test_layer_sharded_registry_omits_per_head_only_kwargs(self):
        import inspect

        from megatron.core.optimizer.emerging_optimizers import _muon_registry_config_to_kwargs
        from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

        config = OptimizerConfig(
            optimizer="muon",
            muon_tp_mode="layer_sharded",
            muon_split_qkv=False,
            use_layer_wise_distributed_optimizer=True,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=4)
        )
        kwargs = _muon_registry_config_to_kwargs(config, [model], SimpleNamespace())
        inspect.signature(LayerShardedMuon).bind([], **kwargs)
        self.assertNotIn("split_qkv_per_head", kwargs)
        self.assertNotIn("adamw_betas", kwargs)

    def test_opt_in_does_not_change_default_updates(self):
        p = torch.nn.Parameter(torch.randn(12, 8))
        q = torch.nn.Parameter(p.detach().clone())
        p.muon_layout = MuonProjectionLayout((4, 8), (False, True))
        opts = [TensorParallelMuon([t], fp32_matmul_prec="highest") for t in (p, q)]
        for _ in range(3):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
            for opt in opts:
                opt.step()
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        self.assertNotIn("gate_exp_avg", opts[0].state[p])

    def test_gdn1_controls_are_adamw(self):
        layout = MuonProjectionLayout.gdn(
            ['query', 'key', 'value', 'z', 'beta', 'alpha'], (8, 8, 12, 12, 3, 3), 4, 4
        )
        self.assertEqual(layout.adamw_ranges(0, 46), ((28, 46),))
        self.check_reference(layout)

    def test_gdn2_uses_variant_sections(self):
        layout = MuonProjectionLayout.gdn(
            ['query', 'key', 'value', 'z', 'f', 'b', 'w'], (8, 8, 12, 12, 8, 8, 12), 4, 4
        )
        self.assertEqual(sum(layout.splits), 68)
        self.assertEqual(layout.adamw_ranges(0, 68), ((28, 68),))
        self.check_reference(layout)

    def test_attention_output_gates_are_adamw(self):
        config = SimpleNamespace(
            num_attention_heads=4, num_query_groups=2, kv_channels=4, attention_output_gate=True
        )
        self.check_reference(MuonProjectionLayout.attention(config))

    def test_swiglu_gate_and_up_are_separate_matrices(self):
        self.check_reference(
            MuonProjectionLayout.matrices((12, 12), tp_local=True, tp_partitioned=True)
        )

    def test_mla_latent_and_rope_are_separate(self):
        self.check_reference(MuonProjectionLayout.matrices((12, 4)))
        self.check_reference(MuonProjectionLayout.matrices((8, 12, 4)))

    def test_mla_without_rope(self):
        layout = MuonProjectionLayout.matrices((12, 0))
        self.assertEqual(layout.splits, (12,))
        self.check_reference(layout)

    def test_unequal_mla_splits_batch_without_changing_real_scale(self):
        self.check_reference(MuonProjectionLayout.matrices((12, 4)), width=32)
        self.check_reference(
            MuonProjectionLayout.matrices((12, 4)), width=32, scale_mode='unit_rms_norm'
        )

    def test_all_adamw_shard(self):
        self.check_reference(MuonProjectionLayout((3, 1), (True, True)))

    def test_adaptive_muon_retains_second_moment_initialization(self):
        for per_head in (False, True):
            p = torch.nn.Parameter(torch.randn(8, 8))
            p.muon_layout = MuonProjectionLayout.matrices((4, 4))
            opt = TensorParallelAdaptiveMuon(
                [p], split_qkv=per_head, split_qkv_per_head=per_head, fp32_matmul_prec='highest'
            )
            p.grad = torch.randn_like(p)
            opt.step()
            self.assertTrue(torch.isfinite(p).all())
            self.assertGreater(torch.count_nonzero(opt.state[p]['moment2_buffer']), 0)

    def test_bad_layout_fails_instead_of_whole_matrix_fallback(self):
        p = torch.nn.Parameter(torch.randn(10, 8))
        p.muon_layout = MuonProjectionLayout.matrices((4, 4))
        with self.assertRaisesRegex(ValueError, 'Refusing whole-matrix fallback'):
            self.optimizer([p])

    def test_state_dict_resume_matches_uninterrupted(self):
        layout = MuonProjectionLayout((4, 4, 1), (False, True, True))
        p, opt = self.check_reference(layout, steps=3)
        q = torch.nn.Parameter(p.detach().clone())
        q.muon_layout = layout
        resumed = self.optimizer([q])
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(3):
            grad = torch.randn_like(p)
            p.grad, q.grad = grad.clone(), grad.clone()
            opt.step()
            resumed.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_missing_grad_uses_parameter_adam_step(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        q = torch.nn.Parameter(torch.randn(8, 8))
        p.muon_layout = q.muon_layout = MuonProjectionLayout((4, 4), (False, True))
        opt = self.optimizer([p, q])
        ref = torch.nn.Parameter(q.detach()[4:].clone())
        adam = torch.optim.AdamW([ref], lr=0.003, betas=(0.8, 0.97), eps=1e-7, weight_decay=0.1)
        for i in range(6):
            p.grad = torch.randn_like(p)
            q.grad = None if i in (0, 3) else torch.randn_like(q)
            ref.grad = None if q.grad is None else q.grad[4:].clone()
            opt.step()
            adam.step()
            torch.testing.assert_close(q[4:], ref, atol=2e-6, rtol=2e-6)
        resumed = self.optimizer([p, q])
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        self.assertEqual(resumed.state[q]['step'], 4)

    def test_old_checkpoint_requires_explicit_reset(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        p.muon_layout = MuonProjectionLayout((4, 4), (False, True))
        opt = self.optimizer([p])
        state = opt.state_dict()
        del state['param_groups'][0]['muon_semantic_version']
        with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
            opt.load_state_dict(state)

    def test_gate_rows_never_enter_ns(self):
        p = torch.nn.Parameter(torch.randn(12, 8))
        p.muon_layout = MuonProjectionLayout((4, 4, 4), (False, True, False))
        opt = self.optimizer([p])
        calls = []

        def identity(g, *args, **kwargs):
            calls.append(g.clone())
            return g

        opt.scaled_orthogonalize_fn = identity
        p.grad = torch.ones_like(p)
        p.grad[4:8] = 1e6
        opt.step()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (2, 4, 8))
        self.assertLess(float(calls[0].max()), 10)

    def test_metadata_copied_to_master_parameter(self):
        from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes

        p = torch.nn.Parameter(torch.randn(8, 8))
        p.muon_layout = MuonProjectionLayout((4, 4), (False, True))
        q = torch.nn.Parameter(p.detach().clone())
        copy_tensor_model_parallel_attributes(q, p)
        self.assertEqual(q.muon_layout, p.muon_layout)

    def test_flattened_semantic_parameter_is_rejected(self):
        """Flattened optimizer shards cannot be interpreted as projection rows."""
        p = torch.nn.Parameter(torch.randn(64))
        p.muon_layout = MuonProjectionLayout((4, 4), (False, True))
        with self.assertRaisesRegex(ValueError, "2-D.*LayerWiseDistributedOptimizer"):
            self.optimizer([p])

    def test_checkpoint_tensors_follow_parameter_sharding(self):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
        from megatron.core.dist_checkpointing.optimizer import optim_state_to_sharding_state
        from megatron.core.optimizer.optimizer import MegatronOptimizer

        p, opt = self.check_reference(MuonProjectionLayout((4, 4), (False, True)))
        state = copy.deepcopy(opt.state_dict())
        model_shard = ShardedTensor.from_rank_offsets('projection', p.detach())
        self.assertEqual(MegatronOptimizer._extract_common_per_param_step(state), 5)
        optim_state_to_sharding_state(state, {0: model_shard}, exclude_keys=('step',))
        self.assertEqual(
            set(state['state'][0]), {'momentum_buffer', 'gate_exp_avg', 'gate_exp_avg_sq'}
        )
        for tensor in state['state'][0].values():
            self.assertEqual(tensor.global_shape, tuple(p.shape))
            self.assertEqual(tensor.local_shape, tuple(p.shape))

    def test_batched_and_unbatched_ns_match(self):
        from unittest.mock import patch

        layout = MuonProjectionLayout((4, 4, 6, 1), (False, False, False, True))
        with patch(
            'megatron.core.optimizer.emerging_optimizers._supports_batched_newton_schulz',
            return_value=False,
        ):
            self.check_reference(layout)


if __name__ == '__main__':
    unittest.main()
