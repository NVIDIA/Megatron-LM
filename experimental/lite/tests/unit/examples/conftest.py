# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Skip the verl integration tests when verl is not installed."""

import importlib.util

_HAVE_VERL = all(importlib.util.find_spec(m) for m in ("verl", "verl_mlite"))


def pytest_ignore_collect(collection_path, config):
    """Ignore this directory's tests when verl is absent.

    They import verl at module scope, so without it they fail collection and
    read as broken rather than absent; verl is not a dependency of lite itself.
    ``collect_ignore_glob`` is not enough here: it is bypassed when the files are
    named explicitly on the command line, which is how the repository's own
    runner invokes them.
    """
    del config
    return not _HAVE_VERL and collection_path.name.startswith("test_")
