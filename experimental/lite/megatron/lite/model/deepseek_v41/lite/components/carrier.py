# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model source schema on the generic differentiable pipeline codec."""
from megatron.lite.primitive.parallel.source_carrier import pack_values, unpack_values


def pack_sources(hidden, pre_mix, state, prefix, group):
    sources = {
        k: v for k, v in state.values.items() if k[0] not in {"attention_inputs", "main_latent"}
    }
    return pack_values(
        (hidden, pre_mix, sources, prefix.data_ptr(), id(group)), device=hidden.device
    )


def unpack_sources(carrier, prefix, group):
    hidden, pre_mix, sources, old_pointer, old_group = unpack_values(carrier)

    def rebase(key):
        if isinstance(key, tuple):
            return tuple(rebase(k) for k in key)
        if isinstance(key, int) and not isinstance(key, bool):
            if key == old_pointer:
                return prefix.data_ptr()
            if key == old_group:
                return id(group)
        return key

    return hidden, pre_mix, {rebase(k): v for k, v in sources.items()}
