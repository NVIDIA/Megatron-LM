# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Intended-image GPU gate for real duplicated-mode expert Muon batching."""

import pytest
import torch

import megatron.core.optimizer.emerging_optimizers as muon
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not muon.HAVE_EMERGING_OPTIMIZERS or not torch.cuda.is_available(),
    reason="requires CUDA and emerging_optimizers",
)


def assert_bounded(actual, expected):
    difference = (actual.double() - expected.double()).abs()
    assert difference.max().item() <= 1e-5
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    assert (torch.linalg.vector_norm(difference) / denominator).item() <= 5e-3


@pytest.mark.parametrize("shape", [(1856, 672), (672, 1856)])
def test_three_positive_lr_updates_native_vs_batch32(shape):
    matrices = 33
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        assert groups.expt_tp.size() == groups.expt_gtp_remat.size() == 1
        torch.manual_seed(2137)
        initial = [
            torch.randn(shape, device="cuda", dtype=torch.float32) * 0.0173
            for _ in range(matrices + 1)
        ]
        dense_initial = torch.randn((17, 11), device="cuda", dtype=torch.float32) * 0.0173
        owners, optimizers = [], []
        for batch in (1, 32):
            params = [torch.nn.Parameter(value.clone()) for value in initial]
            for p in params:
                p.expert_tp = True
                p.tensor_model_parallel = True
                p.partition_dim = 0
                p.partition_stride = 1
            dense = torch.nn.Parameter(dense_initial.clone())
            dense.expert_tp = False
            dense.tensor_model_parallel = False
            dense.partition_dim = -1
            owners.append((params, dense))
            optimizers.append(
                muon.TensorParallelMuon(
                    [
                        {"params": params, "is_expert_parallel": True},
                        {"params": [dense], "is_expert_parallel": False},
                    ],
                    lr=2.16e-4,
                    momentum=0.9,
                    nesterov=False,
                    weight_decay=0.1,
                    coefficient_type="polar_express",
                    num_ns_steps=16,
                    scale_mode="spectral",
                    extra_scale_factor=0.2,
                    fp32_matmul_prec="medium",
                    tp_mode="duplicated",
                    pg_collection=groups,
                    expert_ns_batch_size=batch,
                )
            )
        baseline, fast = optimizers
        replay = fast._expert_ns_replay
        replay_shapes = []

        def tracked_replay(value, **kwargs):
            replay_shapes.append(tuple(value.shape))
            return replay(value, **kwargs)

        fast._expert_ns_replay = tracked_replay
        update_outputs = [{}, {}]
        original_methods = [optimizer.orthogonalize for optimizer in optimizers]
        # Observe the actual orthogonalized buffers without substituting computation.
        for index, optimizer in enumerate(optimizers):
            original = original_methods[index]
            positions = {id(p): i for i, p in enumerate(owners[index][0] + [owners[index][1]])}

            def observe(
                p, grad, *, _index=index, _original=original, _positions=positions, **kwargs
            ):
                value = _original(p, grad, **kwargs)
                update_outputs[_index][_positions[id(p)]] = value.detach().clone()
                return value

            optimizer.orthogonalize = observe
        try:
            for step, lr in enumerate((2.16e-4, 1.08e-4, 3.24e-4)):
                grads = [torch.randn_like(value) * 0.01 for value in initial[:matrices]]
                dense_grad = torch.randn_like(dense_initial) * 0.01
                for index, optimizer in enumerate(optimizers):
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    for p, grad in zip(owners[index][0][:-1], grads):
                        p.grad = grad.clone()
                    owners[index][0][-1].grad = None  # permanent gradNone state/parameter witness
                    owners[index][1].grad = dense_grad.clone()
                    update_outputs[index].clear()
                cpu_rng, cuda_rng = (
                    torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state().clone(),
                )
                before_calls = len(replay_shapes)
                baseline.step()
                fast.step()
                assert torch.equal(cpu_rng, torch.get_rng_state())
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
                assert baseline._expert_ns_batch is fast._expert_ns_batch is None
                if step == 0:
                    assert not replay_shapes  # native empty-state initialization, not batching
                else:
                    current = replay_shapes[before_calls:]
                    assert [row[0] for row in current] == (
                        [32, 1] if matrices == 33 else [matrices]
                    )
                for i, (old, new) in enumerate(zip(owners[0][0], owners[1][0])):
                    assert old.dtype == new.dtype == torch.float32
                    assert_bounded(new, old)
                    if i == matrices:
                        assert old.grad is new.grad is None
                        assert not baseline.state.get(old) and not fast.state.get(new)
                        assert torch.equal(old, initial[i]) and torch.equal(new, initial[i])
                        continue
                    assert torch.equal(old.grad, grads[i]) and torch.equal(new.grad, grads[i])
                    assert set(baseline.state[old]) == set(fast.state[new]) == {"momentum_buffer"}
                    assert (
                        baseline.state[old]["momentum_buffer"].dtype
                        == fast.state[new]["momentum_buffer"].dtype
                        == torch.float32
                    )
                    assert torch.equal(
                        baseline.state[old]["momentum_buffer"], fast.state[new]["momentum_buffer"]
                    )
                    assert_bounded(update_outputs[1][i], update_outputs[0][i])
                assert torch.equal(owners[0][1], owners[1][1])
                assert torch.equal(owners[0][1].grad, dense_grad) and torch.equal(
                    owners[1][1].grad, dense_grad
                )
                assert torch.equal(update_outputs[0][matrices + 1], update_outputs[1][matrices + 1])
        finally:
            for optimizer in optimizers:
                del optimizer.orthogonalize
    finally:
        Utils.destroy_model_parallel()
