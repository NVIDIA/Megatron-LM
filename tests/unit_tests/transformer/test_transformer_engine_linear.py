# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions import transformer_engine as te_ext


@pytest.mark.skipif(not te_ext.HAVE_TE, reason="Transformer Engine is required")
def test_duplicated_linear_retains_checkpoint_group_without_enabling_te_tp(monkeypatch):
    """An explicit TP group is metadata only for a duplicated TE linear."""
    checkpoint_tp_group = object()
    checkpoint_dp_cp_group = object()
    checkpoint_wrap_kwargs = {}
    te_init_kwargs = {}

    def fake_te_linear_init(module, **kwargs):
        torch.nn.Module.__init__(module)
        module.parallel_mode = kwargs["parallel_mode"]
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(1, 1)))
        te_init_kwargs.update(kwargs)

    config = SimpleNamespace(
        delay_wgrad_compute=False,
        disable_parameter_transpose_cache=False,
        expert_gtp_weight_remat_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        gradient_accumulation_fusion=False,
        gtp_weight_remat_size=1,
        perform_initialization=False,
        quant_recipe=None,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        tp_comm_overlap=False,
    )
    rng_tracker = SimpleNamespace(is_initialized=lambda: False)

    monkeypatch.setattr(te_ext.te.pytorch.Linear, "__init__", fake_te_linear_init)
    monkeypatch.setattr(te_ext, "_get_extra_te_kwargs", lambda config: {})
    monkeypatch.setattr(te_ext, "get_cuda_rng_tracker", lambda: rng_tracker)
    monkeypatch.setattr(te_ext, "get_quant_config_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(te_ext, "is_te_min_version", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        te_ext,
        "make_sharded_tensors_for_checkpoint",
        lambda *args, **kwargs: checkpoint_wrap_kwargs.update(kwargs) or {},
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "is_initialized", lambda: False)

    module = te_ext.TELinear(
        2,
        3,
        parallel_mode="duplicated",
        config=config,
        init_method=lambda weight: weight,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=checkpoint_tp_group,
    )

    assert te_init_kwargs["tp_group"] is None
    assert te_init_kwargs["tp_size"] == 1
    assert te_init_kwargs["parallel_mode"] is None
    assert module._tp_group is checkpoint_tp_group

    monkeypatch.setattr(module, "state_dict", lambda *args, **kwargs: {"weight": module.weight})
    module.sharded_state_dict(metadata={"dp_cp_group": checkpoint_dp_cp_group})

    assert checkpoint_wrap_kwargs["tp_group"] is checkpoint_tp_group
    assert checkpoint_wrap_kwargs["dp_cp_group"] is checkpoint_dp_cp_group
