# Native MLM recipe pretraining

A recipe is a Python function that returns a config container
(`PretrainConfigContainer`) describing a training configuration tested for
convergence and reasonably good performance. Early or unoptimized recipes are
clearly marked.
Think of a recipe as a YAML configuration file written in Python: you can build
nested configs, reuse settings, and modify them with ordinary Python code.

These recipes originated in
[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and now live in
Megatron-LM, so you don't need the separate Bridge repository to run them.

To get started, choose a recipe and adjust it for your model, data, and hardware.
You're welcome to modify, override, or extend it to suit your training run.
Your recipe function should take no arguments. Set your dataset and dataloader through `cfg.data`;
the framework finalizes the config for you when training starts.

From the repository root, launch your recipe with:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/run_recipe.py \
    --recipe my_recipes:pretrain_config train.train_iters=50
```

Replace `my_recipes:pretrain_config` with your recipe's importable module and
function, and choose a GPU count matching its parallelism settings. You can
override existing config fields by adding `KEY=VALUE` arguments, as shown above.
Add `--dry-run` to validate and inspect your config without starting training.

The runner selects the text forward step for GPT/Hybrid configs and the VLM step
for other model configs. If you need a custom step, pass
`--forward-step module:function`. For VLM recipes, you'll also need a compatible
data pipeline in `cfg.data`. Only load recipe and forward-step modules you trust,
since they execute Python code.

If you find a configuration that improves performance while preserving the
existing recipe's convergence behavior, we'd love for you to contribute it back
to Megatron-LM!
