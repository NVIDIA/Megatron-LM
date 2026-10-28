# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The seams between the run config and the legacy arguments inside the training loop.

The helpers are pure functions of the container and the argument namespace, called at the
application boundary with ``get_run_config()`` and ``get_args()``; the config classes read no
global.

``data_sequence_length``: the sample length the loop runs at. ``cfg.data.dataset.sequence_length``
when the section has a dataset; ``args.seq_length``, read only in that case, for the entry points
whose datasets come from their own callbacks (their remaining ``args`` dependency until their
dataset configs exist). The namespace is not touched when the section has a dataset, so a stale or
missing ``args.seq_length`` cannot leak into a GPT run.

``model_topology`` and ``model_layout``: the model-parallel values the data path reads. From
``cfg.model.transformer`` when the container has a model section; from the namespace for a
container built without ``model_cfg`` (``pretrain_hybrid_flex.py`` today). That fallback is a
compatibility condition of the container, not of one script, and goes when every container has a
model section.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelTopology:
    """The model-parallel values the loader, ``get_batch`` and ``dummy_train_step`` read."""

    context_parallel_size: int
    pipeline_model_parallel_size: int
    dynamic_context_parallel: bool
    sequence_packing_scheduler: Optional[str]
    source: str
    """``"model"`` (read from ``cfg.model.transformer``) or ``"args"`` (no model section)."""


def _transformer(cfg: Any) -> Any:
    return getattr(getattr(cfg, "model", None), "transformer", None)


def model_topology(cfg: Any, args: Any) -> ModelTopology:
    """The model-parallel values the data path reads, from the model section when there is one."""
    transformer = _transformer(cfg)
    if transformer is not None:
        return ModelTopology(
            context_parallel_size=transformer.context_parallel_size,
            pipeline_model_parallel_size=transformer.pipeline_model_parallel_size,
            dynamic_context_parallel=bool(transformer.dynamic_context_parallel),
            sequence_packing_scheduler=transformer.sequence_packing_scheduler,
            source="model",
        )
    return ModelTopology(
        context_parallel_size=args.context_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        dynamic_context_parallel=bool(args.dynamic_context_parallel),
        sequence_packing_scheduler=args.sequence_packing_scheduler,
        source="args",
    )


def model_layout(cfg: Any, args: Any) -> tuple[Any, Optional[int]]:
    """``(pipeline_model_parallel_layout, mtp_num_layers)`` for the rank predicate: from the model
    section, else from the transformer config the legacy arguments describe (what the predicate
    built from the arguments before the container carried a model section)."""
    transformer = _transformer(cfg)
    if transformer is None:
        from megatron.training.arguments import core_transformer_config_from_args

        transformer = core_transformer_config_from_args(args)
    return transformer.pipeline_model_parallel_layout, transformer.mtp_num_layers


def data_sequence_length(cfg: Any, args: Any) -> int:
    """The sample length the loop runs at: the dataset's when the section has one, else the
    namespace's (read only then)."""
    dataset = getattr(getattr(cfg, "data", None), "dataset", None)
    if dataset is not None:
        return dataset.sequence_length
    return args.seq_length
