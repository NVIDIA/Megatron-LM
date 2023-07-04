# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


MAJOR = 0
MINOR = 2
PATCH = 0
PRE_RELEASE = ''

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'megatron_core'
__contact_names__ = 'NVIDIA'
__contact_emails__ = 'nemo-toolkit@nvidia.com' # use NeMo Email
__homepage__ = 'https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/' # use NeMo homepage
__repository_url__ = 'https://github.com/NVIDIA/Megatron-LM/megatron/core'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = 'Megatron Core - a library for efficient and scalable training of transformer based models'
__license__ = 'BSD-3'
__keywords__ = 'deep learning, machine learning, gpu, NLP, NLU, language, transformer, nvidia, pytorch, torch'
