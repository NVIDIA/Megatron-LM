# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys


def _redirect_nonzero_ranks():
    rank = int(os.getenv("RANK", "0"))
    redirect_nonzero_ranks = os.getenv("MFSDP_UT_REDIRECT_NONZERO_RANKS", "1").lower() == "1"
    if rank != 0 and redirect_nonzero_ranks:
        os.makedirs("pytest_ranks", exist_ok=True)
        f = open(f"pytest_ranks/rank{rank}.out", "w", buffering=1)
        sys.stdout = f
        sys.stderr = f


_redirect_nonzero_ranks()
