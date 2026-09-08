# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Skip the verl integration tests when verl is not installed."""

import importlib.util

# These import verl at module scope, so without it they fail collection and read
# as broken rather than absent. verl is not a dependency of lite itself.
collect_ignore_glob = [] if importlib.util.find_spec("verl") else ["test_*.py"]
