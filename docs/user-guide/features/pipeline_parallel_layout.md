<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Custom Pipeline Model Parallel Layout

*This is an experimental feature and may be changed.*

`--pipeline-model-parallel-layout` takes a string that defines pipeline parallel partitioning. Use it to balance partitioning when for an imbalanced model. For example, to partition a DeepSeek-V3-style stack (61 decoder layers and one MTP layer) with PP16 and VPP2, pass arguments similar to the following:

```bash
--pipeline-model-parallel-size 16
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

The table below shows one possible rank map for that layout:

| PP \ VPP rank |            0            |       1       |
|---------------|-------------------------|---------------|
|       0       | embedding + 3 × decoder |  2 × decoder  |
|      1~13     |        2 × decoder      |  2 × decoder  |
|       14      |        2 × decoder      |      mtp      |
|       15      |        2 × decoder      |      loss     |

In the layout string, stages are split by `|`. Replicated stages or layers use multiplication (for example, `t*3`). Commas are optional for readability. Symbols:

* `E`: embedding layer
* `t`: transformer decoder layer
* `m`: MTP layer
* `L`: loss calculation layer

**Note:** Empty stages are allowed, for example `E||t|L` (the second stage is empty).
