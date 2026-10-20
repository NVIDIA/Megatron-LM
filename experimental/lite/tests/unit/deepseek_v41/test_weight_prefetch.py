# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Actual constructor/HF path: native bytes unchanged, host table never resident on GPU."""
import gc
import json
from dataclasses import asdict

import pytest
import torch
import torch.distributed as dist
from _ds41_fixture import tiny_config
from test_training import _build, _initialize


@pytest.mark.gpus(1)
def test_prefetch_weights_matches_synchronous_load_and_gpu_memory(tmp_path):
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.cuda.set_device(0)
    dist.init_process_group('nccl', f'file://{tmp_path}/store', rank=0, world_size=1)
    try:
        (tmp_path / 'config.json').write_text(json.dumps(asdict(tiny_config())))
        rt, handle, model, ps = _build(str(tmp_path))
        _initialize(model, ps, tmp_path)
        expected = {
            name: value.cpu().clone() if isinstance(value, torch.Tensor) else value
            for name, value in model.state_dict().items()
        }
        cfg, impl = model.release, model.impl
        del model, handle, rt
        gc.collect()
        torch.cuda.empty_cache()
        peaks = []
        for enabled in (False, True):
            from dataclasses import replace

            options = replace(
                impl,
                initialize_weights=False,
                prefetch_weights=enabled,
                hf_path=str(tmp_path / 'release-rank0'),
            )
            torch.cuda.reset_peak_memory_stats()
            bundle = protocol.build_model(cfg, impl_cfg=options)
            model = bundle.chunks[0]
            assert (model._load_reader is not None) is enabled
            protocol.load_hf_weights(model, options.hf_path, cfg, bundle.parallel_state)
            assert model._load_reader is None
            actual = model.state_dict()
            assert actual.keys() == expected.keys()
            for name, value in actual.items():
                want = expected[name]
                if not isinstance(value, torch.Tensor):
                    assert value is want is None, name  # BF16 TE has no serialized FP8 extra state
                    continue
                assert value.dtype == want.dtype and value.shape == want.shape
                assert torch.equal(
                    value.detach().cpu().reshape(-1).view(torch.uint8),
                    want.reshape(-1).view(torch.uint8),
                ), name
            for memory in model.engrams.values():
                for value in memory.embed.embedding.buffers():
                    assert value.device.type == 'cpu' and value.is_pinned()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
            del model, bundle, actual, value, memory
            gc.collect()
            torch.cuda.empty_cache()
        assert peaks[1] <= peaks[0] + (16 << 20)
        metrics = dict(zip(("synchronous_peak_bytes", "prefetch_peak_bytes"), peaks))
        (tmp_path / "prefetch-memory.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()
