# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch miles with the Megatron Lite backend patch."""

from __future__ import annotations

import asyncio

from miles.utils import arguments as miles_arguments
from miles.utils.tracking_utils import finish_tracking
from miles_mlite.arguments import add_mlite_arguments, validate_mlite_args


def _patch_epsilon_alias_for_miles_validate() -> None:
    original_validate = miles_arguments.hf_validate_args
    if getattr(original_validate, "_mlite_epsilon_alias", False):
        return

    def validate_with_epsilon_alias(args, hf_config):
        if hasattr(args, "norm_epsilon") and not hasattr(args, "layernorm_epsilon"):
            args.layernorm_epsilon = args.norm_epsilon
        if hasattr(args, "layernorm_epsilon") and not hasattr(args, "norm_epsilon"):
            args.norm_epsilon = args.layernorm_epsilon
        return original_validate(args, hf_config)

    validate_with_epsilon_alias._mlite_epsilon_alias = True
    miles_arguments.hf_validate_args = validate_with_epsilon_alias


def _select_train_loop(args):
    if getattr(args, "colocate", False):
        from train import train
    else:
        from train_async import train
    return train


def main() -> None:
    _patch_epsilon_alias_for_miles_validate()
    args = miles_arguments.parse_args(add_custom_arguments=add_mlite_arguments)
    if getattr(args, "mlite_backend_patch", False):
        from miles_mlite.backend_patch import patch_miles_backend

        patch_miles_backend()
        validate_mlite_args(args)
    train = _select_train_loop(args)
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()


if __name__ == "__main__":
    main()
