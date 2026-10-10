# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay per-head NS dispatch and mixed AdamW updates, including optimizer state.

Also runnable on CPU with ``python -m unittest``. GPU CI exercises the same
updates under side-stream contention; module construction is covered separately
in test_muon_per_head.py.
"""

import contextlib
import unittest
from types import SimpleNamespace

import torch

from megatron.core.muon_layout import MuonProjectionLayout
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.determinism.utils import RacingStreams


@unittest.skipUnless(HAVE_EMERGING_OPTIMIZERS, "emerging_optimizers is not installed")
class TestPerHeadMuonReplay(unittest.TestCase):
    def test_projection_updates_and_moments_replay(self):
        """Identical gradients reproduce every parameter and moment byte for byte."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = SimpleNamespace(
            num_attention_heads=4, num_query_groups=2, kv_channels=32, attention_output_gate=True
        )
        layouts = [
            MuonProjectionLayout.attention(config),
            MuonProjectionLayout.gdn(
                ("query", "key", "value", "z", "beta", "alpha"), (64, 64, 128, 128, 4, 4), 32, 32
            ),
            MuonProjectionLayout.gdn(
                ("query", "key", "value", "z", "f", "b", "w"),
                (64, 64, 128, 128, 64, 64, 128),
                32,
                32,
            ),
            MuonProjectionLayout.matrices((128, 128), tp_local=True, tp_partitioned=True),
            MuonProjectionLayout.matrices((64, 32) * 4),
            MuonProjectionLayout.matrices((128, 32)),
            MuonProjectionLayout.matrices((64, 128, 32), tp_reorder_splits=(64, 160)),
        ]
        generator = torch.Generator(device=device).manual_seed(713)
        for layout in layouts:
            with self.subTest(layout=layout):
                initial = torch.randn(sum(layout.splits), 256, generator=generator, device=device)
                grads = [
                    torch.randn(initial.shape, generator=generator, device=device) for _ in range(3)
                ]

                def run():
                    parameter = torch.nn.Parameter(initial.clone())
                    # The mixed-precision wrapper copies the model's layout to the master.
                    model_parameter = torch.nn.Parameter(initial.clone())
                    model_parameter.muon_layout = layout
                    copy_tensor_model_parallel_attributes(parameter, model_parameter)
                    optimizer = TensorParallelMuon(
                        [parameter],
                        split_qkv=True,
                        split_qkv_per_head=True,
                        fp32_matmul_prec="highest",
                        lr=0.003,
                        weight_decay=0.1,
                    )
                    context = RacingStreams() if device == "cuda" else contextlib.nullcontext()
                    with context:
                        for grad in grads:
                            parameter.grad = grad.clone()
                            optimizer.step()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    state = optimizer.state[parameter]
                    tensors = [parameter.detach()] + [
                        state[key] for key in sorted(state) if torch.is_tensor(state[key])
                    ]
                    return tensors, state["step"]

                reference, reference_step = run()
                for _ in range(3):
                    actual, step = run()
                    self.assertEqual(step, reference_step)
                    self.assertEqual(len(actual), len(reference))
                    for expected, observed in zip(reference, actual):
                        self.assertTrue(bytes_equal(expected, observed))


if __name__ == "__main__":
    unittest.main()
