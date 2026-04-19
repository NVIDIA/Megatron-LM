# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

import logging

from megatron.core.models.hybrid.hybrid_model import *  # noqa: F401,F403 # pylint: disable=unused-import
from megatron.core.parameterization import SCALING_RECIPE_DEPTH_MUP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


class MambaModel(HybridModel):
    """Backward-compatible wrapper that accepts the deprecated mamba_stack_spec kwarg."""

    def __init__(self, *args, mamba_stack_spec: ModuleSpec = None, **kwargs):
        config = kwargs.get('config') if 'config' in kwargs else (args[0] if args else None)
        if getattr(config, 'scaling_recipe', None) == SCALING_RECIPE_DEPTH_MUP:
            raise NotImplementedError(
                "scaling_recipe='depth_mup' currently supports dense GPT-style residual "
                "Transformer blocks only. MambaModel is out of scope for v1."
            )
        log_single_rank(
            logger, logging.WARNING, "MambaModel has been deprecated. Use HybridModel instead."
        )
        if mamba_stack_spec is not None:
            if 'hybrid_stack_spec' in kwargs or (args and len(args) >= 2):
                raise ValueError(
                    "Cannot specify both hybrid_stack_spec and mamba_stack_spec. "
                    "mamba_stack_spec has been deprecated; use hybrid_stack_spec instead."
                )
            kwargs['hybrid_stack_spec'] = mamba_stack_spec
        super().__init__(*args, **kwargs)
