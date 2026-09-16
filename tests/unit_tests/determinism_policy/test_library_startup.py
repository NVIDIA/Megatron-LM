# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fresh-process test of public MCore/training startup imports on the GPU stack."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the MCore GPU dependency stack")
@pytest.mark.parametrize("entrypoint", ["core", "training"])
def test_policy_precedes_first_library_kernel(entrypoint, tmp_path):
    script = textwrap.dedent('''
        import json
        import os
        import sys
        from types import SimpleNamespace
        import torch
        from megatron.core.determinism import configure_determinism

        options = dict(
            deterministic_mode=True, cross_entropy_loss_fusion=False, tp_comm_overlap=False
        )
        if sys.argv[1] == 'training':
            from megatron.training.determinism import apply_determinism_to_args
            policy = apply_determinism_to_args(SimpleNamespace(**options))
        else:
            policy = configure_determinism(options)
        assert not torch.cuda.is_initialized()
        assert not torch.distributed.is_initialized()

        from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        torch.distributed.init_process_group('nccl', init_method=sys.argv[2], rank=0, world_size=1)
        try:
            torch.manual_seed(123)
            config = ModelParallelConfig(
                **options, use_cpu_initialization=True, gradient_accumulation_fusion=False
            )
            world = torch.distributed.group.WORLD
            groups = ProcessGroupCollection(tp=world, gtp_remat=world)
            layer = ColumnParallelLinear(
                64, 64, config=config, init_method=torch.nn.init.normal_, bias=True,
                gather_output=False, tp_group=world, pg_collection=groups
            ).cuda()
            x = torch.randn(32, 2, 64, device='cuda', requires_grad=True)
            def replay():
                layer.zero_grad(set_to_none=True)
                x.grad = None
                output, _ = layer(x)
                output.square().mean().backward()
                tensors = (output, x.grad, layer.weight.grad, layer.bias.grad)
                return [t.detach().clone() for t in tensors]
            first, second = replay(), replay()
            assert all(t.numel() for t in first)
            for a, b in zip(first, second, strict=True):
                assert torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8),
                    b.contiguous().reshape(-1).view(torch.uint8),
                )
            assert configure_determinism(config) == policy
            current = os.environ['CUBLAS_WORKSPACE_CONFIG']
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' if current == ':4096:8' else ':4096:8'
            try:
                configure_determinism(config)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Accepted a late workspace-policy change')
            print(json.dumps({'policy': policy, 'compared_tensors': len(first)}))
        finally:
            torch.distributed.destroy_process_group()
    ''')
    # Each pytest rank owns a distinct subprocess/store and retains its local GPU.
    store = (tmp_path / f"store-{os.getpid()}").as_uri()
    result = subprocess.run(
        [sys.executable, "-c", script, entrypoint, store],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"compared_tensors": 4' in result.stdout
