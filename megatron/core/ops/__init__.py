# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Where the implementation of every operation lives, organized by operation.

This package is an implementation home, not a selection layer. It holds no registry and no
resolver, and nothing here decides which backend a run gets -- that is the job of the
``BackendSpecProvider`` implementations in :mod:`megatron.core.models.backends` and
:mod:`megatron.core.extensions.transformer_engine_spec_provider`, which import their targets
from here.

Two files per family::

    <family>/__init__.py    the contract every backend for this family must meet
    <family>/backends.py    the backends themselves, side by side

Every optional-package import lives *inside* the method that returns its target, so importing
this package pulls in no optional dependency and no backend that nobody selected. That is what
lets a provider name a backend without the call site having to guard on whether it is
installed, which is what the scattered ``HAVE_*`` flags used to do.

See ``megatron/core/ops/README.md``.
"""

from megatron.core.ops._availability import installed_version, is_installed, require

__all__ = ["installed_version", "is_installed", "require"]
