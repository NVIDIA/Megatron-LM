# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Byte fingerprints of tensors and of a vLLM model's full state (C8 sync evidence)."""
import hashlib

import torch


def checksum(t):
    """(dtype, shape, numel-bytes, two position-weighted wraparound int64 sums), chunked."""
    if not t.is_contiguous():
        t = t.contiguous()
    b = t.detach().reshape(-1).view(torch.uint8)
    n = b.numel()
    s1 = s2 = 0
    chunk = 1 << 28
    for off in range(0, n, chunk):
        c = b[off : off + chunk]
        pad = (-c.numel()) % 8
        if pad:
            c = torch.cat([c, c.new_zeros(pad)])
        w = c.view(torch.int64)
        idx = torch.arange(w.numel(), device=w.device, dtype=torch.int64) + off // 8
        s1 += int(w.sum())
        s2 += int((w * (idx * 2 + 1)).sum())
    return [str(t.dtype), list(t.shape), n, s1 % (1 << 64), s2 % (1 << 64)]


def _host_table(t):
    b = t.detach().reshape(-1).view(torch.uint8)
    head = b[: 1 << 26].numpy().tobytes() + b[-(1 << 26) :].numpy().tobytes()
    return [
        str(t.dtype),
        list(t.shape),
        b.numel(),
        "headtail",
        hashlib.sha256(head).hexdigest()[:16],
    ]


def fingerprint_model(model):
    """Every parameter, buffer and tensor attribute reachable from the modules (KV cache excluded)."""
    seen, out = set(), {}

    def visit(prefix, obj, depth=0):
        if isinstance(obj, torch.Tensor):
            if obj.is_meta:
                out[prefix] = ["meta", list(obj.shape)]
                return
            key = (obj.data_ptr(), obj.dtype, tuple(obj.shape)) if obj.numel() else None
            if key is not None and key in seen:
                out[prefix] = ["alias"]
                return
            if key is not None:
                seen.add(key)
            nbytes = obj.numel() * obj.element_size()
            if obj.device.type == "cpu" and nbytes > (1 << 32):
                out[prefix] = _host_table(obj)
            elif not obj.is_contiguous() and nbytes > (8 << 30):
                out[prefix] = [str(obj.dtype), list(obj.shape), "noncontig-skipped"]
            else:
                out[prefix] = checksum(obj)
        elif depth < 2 and isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(f"{prefix}[{i}]", v, depth + 1)
        elif depth < 2 and isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    visit(f"{prefix}[{k}]", v, depth + 1)

    for mname, mod in model.named_modules():
        for n, p in list(mod._parameters.items()) + list(mod._buffers.items()):
            if p is not None:
                visit(f"{mname}.{n}", p)
        for n, v in vars(mod).items():
            if n.startswith("_") or isinstance(v, torch.nn.Module) or "kv_cache" in n:
                continue
            visit(f"{mname}.@{n}", v)
    return out
