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
        from megatron.determinism import configure_determinism, is_determinism_configured

        options = dict(
            deterministic_mode=True, cross_entropy_loss_fusion=False, tp_comm_overlap=False
        )
        policy = configure_determinism(options)
        assert not torch.cuda.is_initialized()
        assert not torch.distributed.is_initialized()
        assert 'megatron.core' not in sys.modules
        original_lazy_init = torch.cuda._lazy_init
        def checked_lazy_init(*args, **kwargs):
            assert is_determinism_configured()
            assert torch.are_deterministic_algorithms_enabled()
            return original_lazy_init(*args, **kwargs)
        torch.cuda._lazy_init = checked_lazy_init

        from megatron.core.determinism import configure_determinism as legacy_configure
        assert legacy_configure is configure_determinism
        if sys.argv[1] == 'training':
            from megatron.training.determinism import apply_determinism_to_args
            assert apply_determinism_to_args(SimpleNamespace(**options)) == policy
        else:
            assert legacy_configure(options) == policy

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


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the MCore GPU dependency stack")
def test_late_import_rejected_and_checkpoint_safeguards_preserved():
    script = textwrap.dedent('''
        import io
        import pickle
        import torch
        import megatron.core
        from megatron.core.determinism import configure_determinism
        from megatron.core.safe_globals import SAFE_GLOBALS, safe_load_from_bytes

        torch.cuda.init()
        try:
            configure_determinism({'deterministic_mode': True})
        except RuntimeError as error:
            assert 'before importing Core/Bridge' in str(error)
        else:
            raise AssertionError('Accepted a first call after CUDA initialization')
        assert megatron.core.mpu is megatron.core.parallel_state
        assert torch.storage._load_from_bytes is safe_load_from_bytes
        assert all(cls in torch.serialization.get_safe_globals() for cls in SAFE_GLOBALS)
        buffer = io.BytesIO()
        torch.save(torch.arange(7), buffer)
        assert torch.equal(torch.storage._load_from_bytes(buffer.getvalue()), torch.arange(7))
        class Unregistered:
            pass
        buffer = io.BytesIO()
        torch.save(Unregistered(), buffer)
        try:
            torch.storage._load_from_bytes(buffer.getvalue())
        except pickle.UnpicklingError:
            pass
        else:
            raise AssertionError('Core import did not preserve weights-only loading')
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the MCore GPU dependency stack")
@pytest.mark.parametrize("entrypoint", ["pretrain_gpt.py", "pretrain_hybrid.py"])
@pytest.mark.parametrize("mode", ["cli", "yaml"])
def test_training_entrypoint_configures_before_import_time_cuda(entrypoint, mode, tmp_path):
    # Exercise real imports and the real parser; --help bounds this test before
    # data/model setup. The observer still executes the actual CUDA initializer.
    script = textwrap.dedent('''
        import runpy
        import sys
        import torch
        from megatron.determinism import is_determinism_configured
        original_lazy_init = torch.cuda._lazy_init
        def checked_lazy_init(*args, **kwargs):
            assert is_determinism_configured(), 'CUDA initialized before early policy'
            assert torch.are_deterministic_algorithms_enabled()
            return original_lazy_init(*args, **kwargs)
        torch.cuda._lazy_init = checked_lazy_init
        sys.argv = sys.argv[1:]
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('Training parser did not exit after --help')
        assert is_determinism_configured()
        assert 'megatron.core' in sys.modules
    ''')
    options = ["--deterministic-mode"]
    if mode == "yaml":
        path = tmp_path / "config.yaml"
        path.write_text("model_parallel:\n  deterministic_mode: true\n")
        options = ["--yaml-cfg", str(path)]
    result = subprocess.run(
        [sys.executable, "-c", script, entrypoint, *options, "--help"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the MCore GPU dependency stack")
@pytest.mark.parametrize("mode", ["cli", "yaml"])
def test_resolved_training_options_cannot_disable_early_policy(mode, tmp_path):
    script = textwrap.dedent('''
        import sys
        from megatron.determinism import configure_determinism
        configure_determinism({'deterministic_mode': True})
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.yaml_arguments import load_yaml, validate_yaml
        mode, path = sys.argv[1:]
        if mode == 'cli':
            sys.argv = ['test', '--num-layers', '1', '--hidden-size', '128',
                        '--num-attention-heads', '4', '--seq-length', '32',
                        '--max-position-embeddings', '32', '--micro-batch-size', '1',
                        '--train-iters', '2']
            args, validate = parse_args(), validate_args
        else:
            args, validate = load_yaml(path), validate_yaml
        try:
            validate(args)
        except AssertionError as error:
            assert 'deterministic_mode=True' in str(error), str(error)
        else:
            raise AssertionError('Resolved config disabled the early policy')
    ''')
    path = tmp_path / "disabled.yaml"
    path.write_text(textwrap.dedent('''
        data_path: null
        world_size: 1
        rank: 0
        account_for_embedding_in_pipeline_split: false
        deterministic_mode: false
        model_parallel:
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          context_parallel_size: 1
          tp_comm_overlap: false
        language_model: {}
    '''))
    result = subprocess.run(
        [sys.executable, "-c", script, mode, str(path)],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
