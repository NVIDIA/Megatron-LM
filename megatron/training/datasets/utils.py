# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Shared utilities for training-side dataset helpers."""

import json
import os
from typing import Any, Optional


def load_json_arg(spec: Optional[str]) -> Optional[Any]:
    """Parse a CLI JSON argument that may be either a JSON literal or a path
    to a JSON file.

    The argument is interpreted as a file path when ``spec`` points to an
    existing regular file on the local filesystem; otherwise it is parsed as
    a JSON literal string. Returns ``None`` when ``spec`` itself is ``None``,
    so callers can use it transparently for optional CLI flags.

    Used by the ``--sft-mock-dataset-config-json`` and
    ``--varlen-mock-dataset-config-json`` flags, which both accept either an
    inline JSON snippet or the path to a file containing the same JSON
    document.
    """
    if spec is None:
        return None
    if os.path.isfile(spec):
        with open(spec, "r") as f:
            return json.load(f)
    return json.loads(spec)
