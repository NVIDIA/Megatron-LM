# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The seam between the run config and the legacy arguments inside the training loop.

``data_sequence_length`` is a pure function of the container and the argument namespace, called at
the application boundary with ``get_run_config()`` and ``get_args()``; the config classes read no
global. It is the sample length the loop runs at: ``cfg.data.dataset.sequence_length`` when the
section has a dataset; ``args.seq_length``, read only in that case, for the entry points whose
datasets come from their own callbacks (their remaining ``args`` dependency until their dataset
configs exist). The namespace is not touched when the section has a dataset, so a stale or missing
``args.seq_length`` cannot leak into a GPT run.

The model-parallel values the data path reads come from ``cfg.model.transformer``: every training
container has a model section (``pretrain_cfg_container_from_args`` derives one from the arguments
when the entry point passes none).
"""

from typing import Any


def data_sequence_length(cfg: Any, args: Any) -> int:
    """The sample length the loop runs at: the dataset's when the section has one, else the
    namespace's (read only then)."""
    dataset = getattr(getattr(cfg, "data", None), "dataset", None)
    if dataset is not None:
        return dataset.sequence_length
    return args.seq_length
