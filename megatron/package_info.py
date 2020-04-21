# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MAJOR = 0
MINOR = 0
PATCH = 1

# Use the following formatting: (major, minor, patch)
VERSION = (MAJOR, MINOR, PATCH)

__version__ = '.'.join(map(str, VERSION[:3]))
__package_name__ = 'megatron_lm'
__contact_names__ = 'NVIDIA'
__contact_emails__ = 'ekmb.new@gmail.com'
__url__ = 'https://github.com/NVIDIA/Megatron-LM'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.'
__license__ = 'Apache2'
__keywords__ = 'deep learning, Megatron, gpu, NLP, nvidia, pytorch, torch, language'

