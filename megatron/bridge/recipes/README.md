# Native MLM recipe pretraining

Recipes are Python functions that take no arguments and return an MLM
`PretrainConfigContainer`. Configure the model, tokenizer, dataset, and dataloader
in the recipe, with data settings under `cfg.data`. The framework finalizes the
config when training starts; recipes should not call `finalize()` themselves.
The separate Megatron-Bridge repository is not required.

From the repository root, launch your importable recipe with:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/run_recipe.py \
    --recipe my_recipes:pretrain_config train.train_iters=50
```

Replace `my_recipes:pretrain_config` with your recipe's module and function, and
choose a GPU count matching its parallelism settings. Trailing `KEY=VALUE`
arguments override existing config fields. Use `--dry-run` to validate and
display the config without starting training.

GPT/Hybrid configs default to the text forward step; other model configs default
to the VLM step. Use `--forward-step module:function` to select a custom step.
VLM recipes also need a compatible data pipeline in `cfg.data`.
Only load trusted recipe and forward-step modules: they execute Python code.
