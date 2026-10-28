# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Register PEFT transformations before distributed model wrapping."""

from torch import nn

from megatron.training.models.base import ModelConfig


def register_peft_pre_wrap_hook(model_config: ModelConfig, peft: object | None) -> None:
    """Replace the setup-owned PEFT hook without disturbing user-owned hooks.

    Args:
        model_config: Model configuration used by a distributed model builder.
        peft: Callable accepting model chunks and ``training=True``, or ``None``.

    Raises:
        TypeError: If ``peft`` is not callable or returns invalid model chunks.
    """
    if peft is not None and not callable(peft):
        raise TypeError("peft must be callable or None")

    previous_hook = getattr(model_config, "_peft_pre_wrap_hook", None)
    insert_index = len(model_config.pre_wrap_hooks)
    if previous_hook is not None:
        insert_index = next(
            (
                index
                for index, hook in enumerate(model_config.pre_wrap_hooks)
                if hook is previous_hook
            ),
            insert_index,
        )
        model_config.pre_wrap_hooks[:] = [
            hook for hook in model_config.pre_wrap_hooks if hook is not previous_hook
        ]
        delattr(model_config, "_peft_pre_wrap_hook")

    if peft is None:
        return

    def apply_peft(model_chunks: list[nn.Module]) -> list[nn.Module]:
        transformed = peft(model_chunks, training=True)
        if (
            not isinstance(transformed, list)
            or len(transformed) != len(model_chunks)
            or not all(isinstance(chunk, nn.Module) for chunk in transformed)
        ):
            raise TypeError("peft must return a list of modules")
        return transformed

    model_config.pre_wrap_hooks.insert(insert_index, apply_peft)
    setattr(model_config, "_peft_pre_wrap_hook", apply_peft)
