# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

MAJOR = 1
MINOR = 1.2

# Use the following formatting: (major, minor)
VERSION = (MAJOR, MINOR)

__version__ = '.'.join(map(str, VERSION))
__package_name__ = 'megatron-lm'
__contact_names__ = 'NVIDIA INC'
__url__ = 'https://github.com/NVIDIA/Megatron-LM'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.'
__license__ = 'See https://github.com/NVIDIA/Megatron-LM/blob/master/LICENSE'
__keywords__ = 'deep learning, Megatron, gpu, NLP, nvidia, pytorch, torch, language'

