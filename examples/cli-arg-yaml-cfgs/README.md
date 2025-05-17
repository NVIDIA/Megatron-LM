Researchers often need to compare the performance of different models or evaluate the same model under varying hyperparameters and datasets.

While Megatron-LM provides a command-line-based approach for configuration, which is quite straightforward from an engineering perspective, it is not user-friendly. Even AI infrastructure engineers often spend considerable time to find the correct command-line arguments.

On the other hand, when dealing with complex models that require numerous command-line arguments, writing multiple comparative experiments can easily lead to errors, making scripts difficult to maintain.

Previously, the `--yaml-cfg` option had a drawback: it was incompatible with command-line arguments and did not support multiple YAML files. If I modified the model definition, I would also need to rewrite the trainer definition.

To address this, I submitted a pull request introducing `--cli-arg-yaml-cfgs`, which supports multiple configuration files and appends CLI arguments instead of overwriting them.

This makes switching between multiple models much more convenient, and certain trainer arguments no longer need to be rewritten.

```bash
MODEL_YAML=examples/cli-arg-yaml-cfgs/gpt.yaml # maybe other models

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    --cli-arg-yaml-cfgs $TRAINING_YAML $MODEL_YAML \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
```
