# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composition contract for layers consuming explicit microbatch token context."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from torch import Tensor

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class TokenContextProvider(Protocol):
    """Compose layer-owned modules with explicit per-call token context.

    Providers own no parameters or mutable microbatch state. Their replacement
    specs build ordinary registered children of the selected layers. Consumers
    declare ``accepts_token_context = True`` and accept a ``token_context`` Tensor
    keyword in both normal execution and activation recomputation.
    """

    def layer_spec_overrides(
        self, submodules: object, layer_config_list: Sequence[TransformerConfig], layer_offset: int
    ) -> Mapping[int, ModuleSpec]:
        """Return replacements for this segment, indexed by global layer position."""
        ...

    def prepare(
        self,
        input_ids: Tensor | None,
        *,
        inference_context: BaseInferenceContext | None,
        packed_seq_params: PackedSeqParams | None,
        cp_batch: ContextParallelBatch | None,
    ) -> Tensor:
        """Return microbatch context without retaining mutable per-forward state."""
        ...
