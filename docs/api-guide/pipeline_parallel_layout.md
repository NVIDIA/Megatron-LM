# Custom Pipeline Model Parallel Layout

*This is an experimental feature and may be changed.*

`--pipeline-model-parallel-layout` is a flexible API for defining the pipeline parallel partitioning, which is essential for balanced partitioning for an imbalanced model. For example, to partition DeepSeek-V3 (61 decoder layers + 1 mtp layer) with PP16VPP2, we can include the arguments as follows:

```bash
--pipeline-model-parallel-size 16
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

| PP \ VPP rank |            0            |       1       |
|---------------|-------------------------|---------------|
|       0       | embedding + 3 × decoder |  2 × decoder  |
|      1~13     |        2 × decoder      |  2 × decoder  |
|       14      |        2 × decoder      |      mtp      |
|       15      |        2 × decoder      |      loss     |

In the layout string, stages are split by '|'. Replicated stages or layers can be described with multiplication. Commas can be used cosmetically. Symbol choices:

* `E` = embedding layer
* `t` = transformer decoder layer
* `m` = MTP layer
* `L` = loss calculation layer

Note that it is legal to have empty stages, e.g., `E||t|L` (the second stage is empty).
