# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for the attributes Megatron-FSDP copies onto its distributed parameters.

finish_grad_sync swaps the module's parameters for the distributed ones built by
_init_optimizer_named_parameters before finalize_model_grads runs, so any tag the finalizer
reads must survive that copy. These tests call the method on a stub buffer and need neither
CUDA nor a process group.
"""

from types import SimpleNamespace

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    ParamAndGradBuffer,
)


def test_dist_params_keep_tp_gradient_sum_tag():
    """A TP-replicated parameter with TP-partial gradients (e.g. GatedDeltaNet's out_norm)
    must stay tagged, or its gradient is never summed across TP ranks."""
    tagged = torch.nn.Parameter(torch.ones(4))
    setattr(tagged, "sum_gradients_across_tp_domain", True)
    untagged = torch.nn.Parameter(torch.ones(4))
    names = {tagged: "out_norm.weight", untagged: "linear.weight"}
    buffer = SimpleNamespace(
        parameter_groups=[SimpleNamespace(params=[tagged, untagged], model_weight_buffer=None)],
        param_to_name=names,
        dist_main_weight={name: param.detach().clone() for param, name in names.items()},
        dist_index=None,
    )

    dist_params = dict(ParamAndGradBuffer._init_optimizer_named_parameters(buffer))

    assert dist_params["out_norm.weight"] is not tagged
    assert getattr(dist_params["out_norm.weight"], "sum_gradients_across_tp_domain", False)
    assert not hasattr(dist_params["linear.weight"], "sum_gradients_across_tp_domain")
