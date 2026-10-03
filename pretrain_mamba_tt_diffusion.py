# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain a Mamba-hybrid model with two-tower block-wise diffusion.

Thin wrapper around the standard ``pretrain_mamba.py`` entry point.
Reuses ``get_batch``, ``loss_func``, and ``train_valid_test_datasets_provider``
from ``pretrain_mamba`` and overrides:

* **Model builder** — uses :func:`two_tower_mamba_builder` instead of the
  default single-tower builder.
* **Forward step** — omits ``packed_seq_params`` and ``loss_mask`` from the
  model call because :class:`TwoTowerMambaModel` handles corruption and
  per-token loss internally.
* **Extra args** — registers ``--tt-diffusion-*`` flags via
  :func:`add_two_tower_diffusion_args`.
"""

import time

_PROGRAM_START_TIME = time.time()

from functools import partial

from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.diffusion.two_tower.arguments import add_two_tower_diffusion_args
from megatron.diffusion.two_tower.builder import two_tower_mamba_builder
from megatron.training import get_timers, inprocess_restart, pretrain, set_startup_timestamps
from model_provider import model_provider
from pretrain_mamba import get_batch, loss_func, train_valid_test_datasets_provider

try:
    from megatron.post_training.arguments import add_modelopt_args

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def forward_step(data_iterator, model):
    """Forward training step for two-tower diffusion.

    Fetches a micro-batch via ``get_batch`` (shared with ``pretrain_mamba``),
    then calls :meth:`TwoTowerMambaModel.forward` with ``(tokens,
    position_ids, attention_mask, labels)``.  ``packed_seq_params`` and
    ``loss_mask`` are intentionally omitted from the model call because
    corruption and per-token loss computation are handled inside the model.

    Args:
        data_iterator: Megatron data iterator yielding micro-batches.
        model: A :class:`TwoTowerMambaModel` instance (possibly wrapped by
            pipeline-parallel or DDP wrappers).

    Returns:
        Tuple[Tensor, Callable]: ``(output_tensor, partial(loss_func, ...))``.
    """
    timers = get_timers()

    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen) = (
            get_batch(data_iterator, vp_stage)
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model=model)


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()

    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    train_valid_test_datasets_provider.is_distributed = True

    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    def extra_args_fn(parser):
        parser = add_two_tower_diffusion_args(parser)
        if has_nvidia_modelopt:
            parser = add_modelopt_args(parser)
        return parser

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, two_tower_mamba_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        store=store,
        extra_args_provider=extra_args_fn,
    )
