# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI

# Some of this code was adopted from https://github.com/triton-lang/triton
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid
from pathlib import Path

try:
    from triton import __version__ as triton_version
    from triton.runtime.cache import FileCacheManager
except ImportError:
    raise ImportError("triton is required by the Mamba model but cannot be imported")


def _version_no_greater_than(version, version_limit):
    major, minor, _ = map(int, version.split('.'))
    limit_major, limit_minor = map(int, version_limit.split('.'))
    return major < limit_major or (major == limit_major and minor <= limit_minor)


def default_cache_dir():
    """Provides a default path for the Triton cache directory."""
    return os.path.join(Path.home(), ".triton", "cache")


class ParallelFileCacheManager(FileCacheManager):
    """
    This patched version of ParallelFileCacheManager prevents errors related
    to the builing of the Triton compiler cache when the number of model
    parallel ranks is greater than one, including when certain types of file
    system are used (such as Lustre).

    Usage:
    export TRITON_CACHE_DIR=<chosen-cache-location>
    export TRITON_CACHE_MANAGER=megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager

    This patch implements the changes in the following two Triton project pull
    requests:
    1. https://github.com/triton-lang/triton/pull/3544
    2. https://github.com/triton-lang/triton/pull/4295

    The above changes will probably be included in Triton release version 3.2,
    making this patch no longer necessary.
    """

    def put(self, data, filename, binary=True) -> str:
        """A patched version of put, implementing PR 3544 and PR 4295."""
        patch_limit = '3.1'
        assert _version_no_greater_than(triton_version, patch_limit), (
            "Assertion failed: ParallelFileCacheManager patch should not be "
            f"used beyond Triton version {patch_limit}."
        )
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath
