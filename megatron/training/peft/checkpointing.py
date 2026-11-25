# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""PEFT checkpoint selection, using the native checkpoint loader and saver."""

from megatron.core.utils import unwrap_model


def checkpoint_kwargs(model, config):
    """Select trainable state without retaining a second configuration object.

    The checkpoint layer also retains mutable router expert-bias buffers. Base keys
    are unchanged by Core's adapter wrappers, so remove only their runtime wrapper
    components when selecting any additional trainable base parameters.
    """
    if config.peft is None:
        return {}
    chunks = unwrap_model(model)
    if not isinstance(chunks, list):
        chunks = [chunks]
    names = {
        name.replace("to_wrap.", "")
        for chunk in chunks
        for name, parameter in chunk.named_parameters()
        if parameter.requires_grad
    }

    def select(key):
        return key in names or ".adapter." in key or key.startswith("adapter.")

    return {"adapter_key_filter": select}


def load_base_weights(model, config, pg_collection):
    """Load frozen base weights before adapters exist, without changing run policy."""
    from megatron.training.checkpointing import checkpoint_exists, load_checkpoint

    path = config.checkpoint.pretrained_checkpoint
    if not checkpoint_exists(path):
        raise FileNotFoundError(f"PEFT base checkpoint was not found: {path}")
    load_checkpoint(
        model,
        None,
        None,
        load_dir=path,
        pretrained_checkpoint=None,
        finetune=True,
        load_optim=False,
        load_rng=False,
        restore_training_state=False,
        record_finetune=False,
        ckpt_step=None,
        non_persistent_ckpt_type=None,
        tp_group=pg_collection.tp,
        pp_group=pg_collection.pp,
        dp_cp_group=pg_collection.dp_cp,
        dp_group=pg_collection.dp,
        expt_dp_group=pg_collection.expt_dp,
    )
