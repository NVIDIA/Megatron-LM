# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

GATED_ACTIVATION = ["swiglu", "geglu", "fast-swiglu", "fast-geglu"]


def is_gated_activation(helper):
    """Check whether the model is gated activation"""
    return helper.activation in GATED_ACTIVATION or helper.transformer_config.gated_linear_unit
