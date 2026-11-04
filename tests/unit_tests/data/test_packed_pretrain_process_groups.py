# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import pretrain_gpt
import pretrain_hybrid
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


class _Wrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@pytest.fixture(params=[pretrain_gpt, pretrain_hybrid])
def packed_entrypoint(request, monkeypatch):
    module = request.param
    pgs = ProcessGroupCollection(
        tp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        pp=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
        cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=3,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=1,
    )
    args = SimpleNamespace(
        sequence_packing_scheduler="default_dynamic_cp", dynamic_context_parallel=True
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "core_transformer_config_from_args", lambda args: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail(*args, **kwargs):
        pytest.fail("Explicit process-group paths must not read parallel_state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith("get_") and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    return module, pgs, config


@pytest.mark.parametrize("vp_stage,expected_mtp", [(0, False), (2, True)])
def test_packed_batch_uses_supplied_groups_and_mtp_pipeline_stage(
    packed_entrypoint, monkeypatch, vp_stage, expected_mtp
):
    module, pgs, config = packed_entrypoint
    bootstrap = Mock(side_effect=AssertionError("Explicit groups must not bootstrap globals"))
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", bootstrap)
    result, iterator = object(), object()
    fetch = Mock(return_value=result)
    monkeypatch.setattr(module, "get_batch_on_this_rank_for_sequence_packing", fetch)

    assert module.get_batch(iterator, vp_stage=vp_stage, pg_collection=pgs) is result
    assert fetch.call_args.args == (iterator,)
    kwargs = fetch.call_args.kwargs
    assert kwargs["pg_collection"] is pgs
    assert kwargs["config"] is config
    assert kwargs["vp_stage"] == vp_stage
    assert kwargs["mtp_on_this_rank"] is expected_mtp
    assert kwargs["dynamic_cp"] is True
    assert "dynamic_cp_group_func" not in kwargs
    assert "dynamic_tp_cp_group_func" not in kwargs
    assert kwargs.get("return_context_parallel_batch", False) is (module is pretrain_hybrid)
    bootstrap.assert_not_called()


def test_direct_packed_entrypoint_bootstraps_collection(packed_entrypoint, monkeypatch):
    module, pgs, _ = packed_entrypoint
    bootstrap = Mock(return_value=pgs)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", bootstrap)
    result, iterator = object(), object()
    fetch = Mock(return_value=result)
    monkeypatch.setattr(module, "get_batch_on_this_rank_for_sequence_packing", fetch)

    assert module.get_batch(iterator, vp_stage=2) is result
    assert fetch.call_args.kwargs["pg_collection"] is pgs
    assert set(bootstrap.call_args.kwargs["required_pgs"]) == {
        "tp",
        "pp",
        "cp",
        "tp_cp",
        "dynamic_cp",
        "dynamic_tp_cp",
    }


def test_forward_step_passes_groups_from_wrapped_model(packed_entrypoint, monkeypatch):
    module, pgs, _ = packed_entrypoint
    output, iterator = object(), object()
    model = _Wrapper(_Wrapper(Mock(spec=[], pg_collection=pgs, vp_stage=2, return_value=output)))
    if module is pretrain_gpt:
        batch = (None,) * 7
    else:
        batch = SimpleNamespace(get_batch=lambda: {}, get_packed_seq_params=lambda: None)
    fetch = Mock(return_value=batch)
    monkeypatch.setattr(module, "get_batch", fetch)
    monkeypatch.setattr(module, "get_timers", lambda: Mock())
    stimer = MagicMock()
    stimer.return_value = stimer
    monkeypatch.setattr(module, "stimer", stimer)

    forward_output, loss = module.forward_step(iterator, model)
    assert forward_output is output
    assert callable(loss)
    assert fetch.call_args.args == (iterator, 2)
    assert fetch.call_args.kwargs["pg_collection"] is pgs
