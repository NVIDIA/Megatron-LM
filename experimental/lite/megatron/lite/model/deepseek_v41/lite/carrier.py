# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DS4.1 residual/pre-mix pair on the existing BF16 pipeline carrier."""
import torch


def pack_sources(hidden, pre_mix, state, prefix, group):
    """Pack model source tensors into Core's ordinary dynamic-shape PP tensor.

    FP32 source values use additive BF16 components, preserving their values
    and differentiable connection. Integer/layout metadata uses byte values;
    no process group, Python object or native layout object is transmitted.
    """
    import json

    body = []

    def encode(value):
        if isinstance(value, torch.Tensor):
            value = value.contiguous()
            dtype, shape = str(value.dtype).removeprefix("torch."), list(value.shape)
            if value.dtype == torch.float32:
                parts, remainder = [], value
                for _ in range(4):
                    piece = remainder.to(torch.bfloat16)
                    parts.append(piece.flatten())
                    remainder = remainder - piece.float()
            elif value.dtype == torch.bfloat16:
                parts = [value.flatten()]
            elif not value.is_floating_point():
                parts = [value.view(torch.uint8).flatten().to(torch.bfloat16)]
            else:
                raise TypeError(f"Unsupported DS4.1 source carrier dtype {value.dtype}")
            body.extend(parts)
            return ["tensor", dtype, shape, [p.numel() for p in parts]]
        if isinstance(value, dict):
            return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
        if isinstance(value, (tuple, list)):
            return ["tuple" if isinstance(value, tuple) else "list", [encode(v) for v in value]]
        if isinstance(value, torch.dtype):
            return ["dtype", str(value).removeprefix("torch.")]
        if isinstance(value, torch.device):
            return ["device"]
        return ["literal", value]

    sources = {
        k: v for k, v in state.values.items() if k[0] not in {"attention_inputs", "main_latent"}
    }
    metadata = encode((hidden, pre_mix, sources, prefix.data_ptr(), id(group)))
    header = json.dumps(metadata, separators=(",", ":")).encode()
    lead = torch.tensor(
        list(len(header).to_bytes(4, "little") + header), dtype=torch.bfloat16, device=hidden.device
    )
    return torch.cat((lead, *body)).reshape(1, 1, -1)


def unpack_sources(carrier, prefix, group):
    """Restore a local state scope, rebasing process-local layout identities."""
    import json

    flat = carrier.reshape(-1)
    size = int.from_bytes(bytes(flat[:4].to(torch.uint8).cpu().tolist()), "little")
    metadata = json.loads(bytes(flat[4 : 4 + size].to(torch.uint8).cpu().tolist()))
    cursor = 4 + size

    def decode(record):
        nonlocal cursor
        kind = record[0]
        if kind == "tensor":
            _, dtype, shape, counts = record
            values = []
            for count in counts:
                values.append(flat[cursor : cursor + count])
                cursor += count
            if dtype == "float32":
                result = values[0].float()
                for value in values[1:]:
                    result = result + value.float()
            elif dtype == "bfloat16":
                result = values[0]
            else:
                result = values[0].to(torch.uint8).view(getattr(torch, dtype))
            return result.reshape(shape)
        if kind == "dict":
            return {decode(k): decode(v) for k, v in record[1]}
        if kind in ("list", "tuple"):
            values = [decode(v) for v in record[1]]
            return values if kind == "list" else tuple(values)
        if kind == "dtype":
            return getattr(torch, record[1])
        if kind == "device":
            return carrier.device
        return record[1]

    hidden, pre_mix, sources, old_pointer, old_group = decode(metadata)
    if cursor != flat.numel():
        raise ValueError("DS4.1 source carrier has trailing data")

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
