# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Pretrain Gemma 4 (E4B).

Thin entrypoint mirroring ``pretrain_gpt.py``: it reuses the GPT data pipeline,
batch generation, loss, and forward step verbatim and only swaps in a Gemma4
model builder. The builder constructs a :class:`Gemma4TransformerConfig` (which
carries the heterogeneous per-layer specs, softcap, sqrt(H) embedding scaling
and PLE knobs) and a :class:`Gemma4Model`, which applies the final-logit softcap
and sqrt(H) embedding scaling internally so they reach the training loss path.
"""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _apply_nvrx_version_shim():
    """Work around a container<->MLM incompatibility BEFORE megatron is imported.

    The container's ``nvidia_resiliency_ext`` ships the async-ckpt modules but
    exposes no ``__version__``, so MLM's ``dist_checkpointing/strategies/nvrx.py``
    ``is_nvrx_min_version()`` crashes at import. Populate ``__version__`` first.
    We never use async checkpointing here (single device, DDP=1), so the value
    only needs to satisfy the ``>=0.6.0`` guard. Mirrors code_dev/scripts/mlm_env.py.
    """
    try:
        import nvidia_resiliency_ext as _nvrx

        if not hasattr(_nvrx, "__version__"):
            v = None
            try:
                from importlib.metadata import version

                v = version("nvidia_resiliency_ext")
            except Exception:
                v = None
            try:
                from packaging.version import Version as _V

                if v is None or _V(v) < _V("0.6.0"):
                    v = "0.6.0"
            except Exception:
                v = v or "0.6.0"
            _nvrx.__version__ = v
    except Exception:
        # nvrx absent entirely -> MLM's HAVE_NVRX=False path handles it fine.
        pass


_apply_nvrx_version_shim()

from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.core.models.gemma4.gemma4_layer_specs import (
    get_gemma4_layer_local_spec,
    get_gemma4_layer_with_transformer_engine_spec,
)
from megatron.core.models.gemma4.gemma4_model import Gemma4Model
from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig
from megatron.training import pretrain, print_rank_0, set_startup_timestamps
from megatron.training.argument_utils import (
    core_transformer_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider

# Reuse the GPT data pipeline, batch logic, loss, and forward step unchanged.
from pretrain_gpt import forward_step, get_embedding_ranks, train_valid_test_datasets_provider


def gemma4_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a :class:`Gemma4Model` (mirrors ``gpt_builder`` for the GPT model)."""
    print_rank_0('building Gemma4 model ...')
    if config is None:
        # Build the Gemma4 subclass so its heterogeneous per-layer specs, softcap,
        # sqrt(H) embedding scaling, and PLE dims are populated from the defaults.
        config = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)

    # Gemma4 MLP is a GeGLU with tanh-approx GELU. There is no CLI flag for this exact
    # combination (--swiglu is SiLU, --quick-geglu is quick_gelu), so set it here to
    # match the HF model regardless of the activation flags passed on the command line.
    config.gated_linear_unit = True
    config.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    config.bias_activation_fusion = False

    # The PLE per-layer token table is indexed by the SAME vocab as embed_tokens
    # (HF: vocab_size_per_layer_input == vocab_size). Keep them consistent with the
    # model vocab so a smaller --vocab-size (e.g. for a smoke run) shrinks the PLE
    # table too instead of leaving it pinned at the 262144 default.
    config.vocab_size_per_layer_input = args.padded_vocab_size

    use_te = args.transformer_impl == "transformer_engine"
    if use_te:
        transformer_layer_spec = get_gemma4_layer_with_transformer_engine_spec(config)
    else:
        transformer_layer_spec = get_gemma4_layer_local_spec(config)

    model = Gemma4Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        # Gemma4 ties input/output embeddings; default share unless explicitly untied.
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    return model


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup).
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets.
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    args = parse_and_validate_args(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # Use the Gemma4 transformer config in the pretrain config container too so the
    # container is consistent with the model the builder constructs.
    transformer_cfg = core_transformer_config_from_args(args, config_class=Gemma4TransformerConfig)
    model_cfg = gpt_config_from_args(args, config=transformer_cfg)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gemma4_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )
