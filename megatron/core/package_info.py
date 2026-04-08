# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


MAJOR = 0
MINOR = 18
PATCH = 0
PRE_RELEASE = ''

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

import os as _os  # noqa: I001
import subprocess as _subprocess


if not int(_os.getenv('NO_VCS_VERSION', '0')):
    try:
        _git = _subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            universal_newlines=True,
        )
    except (_subprocess.CalledProcessError, OSError):
        pass
    else:
        __version__ += f'+{_git.stdout.strip()}'

__package_name__ = 'megatron_core'
__contact_names__ = 'NVIDIA'
__contact_emails__ = 'nemo-toolkit@nvidia.com'  # use NeMo Email
__homepage__ = 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html'
__repository_url__ = 'https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = (
    'Megatron Core - a library for efficient and scalable training of transformer based models'
)
__license__ = 'BSD-3'
__keywords__ = (
    'deep learning, machine learning, gpu, NLP, NLU, language, transformer, nvidia, pytorch, torch'
)
