# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Author-specific exceptions to required CODEOWNERS team approvals."""

_AUTHOR_EXEMPT_OWNERS = {"tdene": frozenset({"@NVIDIA/reinforcement-learning"})}


def get_exempt_owners(author_login: str) -> frozenset[str]:
    """Return fully qualified owners exempted for this pull request's author."""
    return _AUTHOR_EXEMPT_OWNERS.get(author_login.casefold(), frozenset())
