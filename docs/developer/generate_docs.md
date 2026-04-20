<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Generating Docs Locally

To generate docs locally, use the following commands:

```
cd docs
uv run --only-group docs sphinx-autobuild . _build/html --port 8080 --host 127.0.0.1
```

Docs will be generated at <http://localhost:8080/>.

**Recommended:** set the environment variable `SKIP_AUTODOC=true` when generating docs 
to skip the generation of `apidocs`.