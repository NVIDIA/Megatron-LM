# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in tensor summaries for BAGEL forward-alignment debugging.

The audit is intentionally inert unless ``BAGEL_LAYER_ALIGNMENT_AUDIT=1`` is
present before Python imports this module.  When enabled, only global rank zero
and the first MoT layer emit summaries, and each stage is printed once.  The
one-shot behavior keeps activation recomputation from duplicating the step-0
trace.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import Tensor


_AUDIT_ENABLED = os.environ.get("BAGEL_LAYER_ALIGNMENT_AUDIT", "0") == "1"
_AUDIT_DUMP_DIR = os.environ.get("BAGEL_LAYER_ALIGNMENT_DUMP_DIR")
_AUDITED_KEYS: set[tuple[int, str, str]] = set()


def layer_alignment_audit_enabled(layer_number: int) -> bool:
    """Return whether this process should emit the layer-0 alignment trace."""

    if not _AUDIT_ENABLED or layer_number != 1:
        return False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _first_tensor(value) -> Optional[Tensor]:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def audit_branch_tensor(
    stage: str,
    branch: str,
    tensor: Tensor,
    *,
    layer_number: int,
) -> None:
    """Print one detached FP32 summary without changing the forward tensor."""

    if not layer_alignment_audit_enabled(layer_number):
        return
    key = (layer_number, stage, branch)
    if key in _AUDITED_KEYS:
        return
    _AUDITED_KEYS.add(key)

    detached = tensor.detach()
    fp32 = detached.float()
    first10 = fp32.reshape(-1)[:10].cpu().tolist()
    fp32_sum = fp32.sum().item()
    if _AUDIT_DUMP_DIR:
        dump_dir = Path(_AUDIT_DUMP_DIR)
        dump_dir.mkdir(parents=True, exist_ok=True)
        filename = f"layer{layer_number - 1}.{stage}.{branch}.pt"
        torch.save(detached.cpu(), dump_dir / filename)
    print(
        "[BAGEL_LAYER_ALIGNMENT_AUDIT] "
        f"layer={layer_number - 1} stage={stage} branch={branch} "
        f"shape={list(detached.shape)} dtype={detached.dtype} "
        f"fp32_sum={fp32_sum:.9g} first10={first10}",
        flush=True,
    )


def audit_compact_tensor(
    stage: str,
    tensor: Tensor,
    und_length: int,
    *,
    layer_number: int,
) -> None:
    """Summarize the understanding and generation slices of a compact tensor."""

    if not layer_alignment_audit_enabled(layer_number):
        return
    audit_branch_tensor(
        stage,
        "und",
        tensor[:und_length],
        layer_number=layer_number,
    )
    audit_branch_tensor(
        stage,
        "gen",
        tensor[und_length:],
        layer_number=layer_number,
    )


@contextmanager
def audit_mlp_linears(
    mlp: torch.nn.Module,
    *,
    branch: str,
    layer_number: int,
) -> Iterator[None]:
    """Capture FC1, post-activation, and FC2 tensors through temporary hooks.

    MCore's MLP exposes the post-SwiGLU activation as the input to ``linear_fc2``.
    Hooks are installed only for the env-gated layer-0 trace and are removed
    immediately after the branch forward.
    """

    if not layer_alignment_audit_enabled(layer_number):
        yield
        return

    handles = []
    linear_fc1 = getattr(mlp, "linear_fc1", None)
    linear_fc2 = getattr(mlp, "linear_fc2", None)

    if linear_fc1 is not None:

        def _fc1_output_hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                audit_branch_tensor(
                    "mlp.fc1_output",
                    branch,
                    tensor,
                    layer_number=layer_number,
                )

        handles.append(linear_fc1.register_forward_hook(_fc1_output_hook))

    if linear_fc2 is not None:

        def _fc2_input_hook(_module, inputs):
            tensor = _first_tensor(inputs)
            if tensor is not None:
                audit_branch_tensor(
                    "mlp.fc1_activation",
                    branch,
                    tensor,
                    layer_number=layer_number,
                )

        def _fc2_output_hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                audit_branch_tensor(
                    "mlp.fc2_output",
                    branch,
                    tensor,
                    layer_number=layer_number,
                )

        handles.append(linear_fc2.register_forward_pre_hook(_fc2_input_hook))
        handles.append(linear_fc2.register_forward_hook(_fc2_output_hook))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
