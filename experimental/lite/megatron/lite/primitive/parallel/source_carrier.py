# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A differentiable BF16 tensor carrier for nested tensor and metadata values."""
import torch


def pack_values(value, *, device):
    """Pack model source tensors into Core's ordinary dynamic-shape PP tensor."""
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
                raise TypeError(f"Unsupported Tensor carrier dtype {value.dtype}")
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

    metadata = encode(value)
    header = json.dumps(metadata, separators=(",", ":")).encode()
    lead = torch.tensor(
        list(len(header).to_bytes(4, "little") + header), dtype=torch.bfloat16, device=device
    )
    return torch.cat(
        [torch.nn.functional.pad(part, (0, -part.numel() % 8)) for part in (lead, *body)]
    ).reshape(1, 1, -1)


def unpack_values(carrier):
    """Restore a local state scope, rebasing process-local layout identities."""
    import json

    flat = carrier.reshape(-1)
    size = int.from_bytes(bytes(flat[:4].to(torch.uint8).cpu().tolist()), "little")
    metadata = json.loads(bytes(flat[4 : 4 + size].to(torch.uint8).cpu().tolist()))
    cursor = (4 + size + 7) // 8 * 8

    def decode(record):
        nonlocal cursor
        kind = record[0]
        if kind == "tensor":
            _, dtype, shape, counts = record
            values = []
            for count in counts:
                values.append(flat[cursor : cursor + count])
                cursor += (count + 7) // 8 * 8
            if dtype == "float32":
                result = values[0].float()
                for value in values[1:]:
                    result = result + value.float()
                result = torch.where(result == 0, values[0].float(), result)
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

    value = decode(metadata)
    if cursor != flat.numel():
        raise ValueError("Tensor carrier has trailing data")

    return value
