# Native MLM recipe pretraining

Recipes construct an MLM `PretrainConfigContainer`. `scripts/run_recipe.py` calls
`megatron.training.api.pretrain(config, forward_step_func, callbacks=None)`.
There is no dependency on the separate Megatron-Bridge repository, no config
adapter, and no fabricated CLI namespace. Forward steps and callbacks use MLM
interfaces, not Bridge's `GlobalState` interface.

**`get_args()` can return `None` after this change.** Config-only runs do not
register arguments. Migrated consumers should read their config owner; optional
legacy features must tolerate absent arguments.

The wrapper supplies MLM's config-driven dataset provider and GPT/Hybrid model
builder to the existing training loop. Framework startup finalizes the config;
recipes should not call `finalize()` themselves. Recipes must specify their
dataset and dataloader through `cfg.data`; there is no dataset-provider override.
Custom model providers still use the lower-level training entrypoint.

## Recipe runner

The recipe is a trusted, importable zero-argument Python factory returning an
MLM `PretrainConfigContainer`. Trailing `KEY=VALUE` overrides edit existing
public dataclass fields (numbers, booleans, `null`, lists, or strings); repeated
keys use the last value. Unknown fields are rejected. Define optional config
sections, dataset classes, and Python callables in the recipe itself.
Both `--recipe` and `--forward-step` import and execute Python code. Only use
trusted modules; this runner is not a sandbox for untrusted recipe names or YAML.

The runner uses `megatron.training.forward_steps.llm_step.forward_step` for exact
`GPTModelConfig` and `HybridModelConfig` types; all other types default to `vlm_step`.
This temporary selection rule must expand as more recipes and step functions are added.
This adapts Bridge's text step to MLM runtime services and its TP-rank-zero data
loaders, without a repository-root script dependency, ModelOpt step, or `GlobalState`.
Use `--forward-step module:function` for an explicit MLM-compatible step.
VLM recipes still require a compatible dataset in `cfg.data`; selecting a step
does not supply synthetic image inputs or a VLM data pipeline.
`--dry-run` validates and displays the unfinalized config without setting global
config, initializing distributed services, building datasets, or training.
It is not a substitute for runtime validation against the actual GPU topology.

`megatron.training.forward_steps.vlm_step.forward_step` adapts Bridge's shared VLM step.
The model owns the vision encoder (HF or Megatron), visual embedding insertion,
multimodal positions, and context-parallel slicing. The step passes visual model
kwargs and collator-provided packed metadata through, and uses the model's
returned local loss mask when present. It shares the masked loss with `llm_step`;
there is no Qwen-specific step or encoder-backend switch.

The VLM step retains Bridge's loader contract: matching, full-sequence batches
on every model-parallel rank, with `visual_inputs` as normalized tensor kwargs
or an object exposing `normalized_for_model()`. Padding/packing belongs in the
collator. **The current recipe API's GPT dataset provider does not yet supply
that VLM data pipeline.** This step alone does not enable end-to-end VLM recipes;
their dataset/loader integration through `cfg.data` remains separate.

Dataset selection stays in `cfg.data`. The runner neither synchronizes sequence
lengths nor finalizes the config: both remain under the existing framework
ownership rules. It calls `megatron.training.api.pretrain`, which delegates to
the existing training loop.

For checkpointing, set `cfg.checkpoint.save` and `save_interval`. For resumption,
also set `cfg.checkpoint.load`, keeping the intended scheduler horizon unchanged
across launches. Config-only save/resume currently supports `torch_dist`. Saved
configuration and runtime topology replace optional CLI metadata; checkpoints
from native runs do not contain fabricated `args`. Older tools that require
checkpoint `args` are not yet supported. Config-only MoE upcycling is also deferred.

Offline cached-logit distillation is owned by `cfg.offline_distillation`, but
the packaged forward steps do not yet select its student loss. Use the existing
offline-distillation entrypoints for that workflow. Ordinary pretraining needs
no legacy argument namespace.

The runner uses the dataset/config implementation already on `main`. It does
not introduce data-schema changes. Phase boundaries are owned by
`cfg.train.phase_transition_iterations`, and optimizer mapping diagnostics by
`cfg.logger.dump_param_to_param_group_map`. Other optional CLI-only features,
including full-dataset training and upcycling, are not migrated here.

## Remaining legacy runtime reads

The following `getattr(args, ...)` reads remain in shared training/startup and
checkpoint code. Their defaults let ordinary config-only pretraining run, but
do not make these optional features config-owned. This is a follow-up inventory,
not a proposal to migrate them all in this PR.

| Location under `megatron/training/` | Fields | Purpose |
| --- | --- | --- |
| `training.py`: `pretrain`, `_get_train_full_dataset_sample_count` | `train_full_dataset`, `deduplicate_dataloader_across_context_parallel` | Determine the full-dataset training horizon. |
| `training.py`: `setup_model_and_optimizer` | `moe_use_upcycling` (two reads) | Select the legacy upcycling path, which also mutates model args. |
| `training.py`: `train` | `log_packed_sequence_stats` | Enable packed-sequence statistics logging. |
| `global_vars.py`: `initialize_runtime_services` | `enable_experimental` | Enable experimental Core features. |
| `global_vars.py`: `_build_tokenizer` | `moe_num_hash_layers`, `hash_moe_vocab_size`, `yaml_cfg` | Backfill hash-MoE vocabulary into legacy arguments. |
| `initialize.py`: `finish_mpu_init` | `mimo_llm_offset` | Include the first non-colocated MIMO language-model rank in logging. |
| `checkpointing.py`: `save_checkpoint` | `dataloader_save` | Save external dataloader state. |
| `checkpointing.py`: `load_checkpoint` | `wide_residual_num_streams`, `allow_llm_only_checkpoint` | Legacy wide-residual fallback and language-only pretrained checkpoint loading. |

Separate compatibility/boundary reads should not be confused with missing
training config ownership:

- `checkpointing.py`: `inference_shards` is an inference/legacy-format guard;
  `_gtp_pad_for_alignment_from_args` reads `fp4`, `fp8_recipe`, and `fp8` for
  legacy checkpoint metadata and argument-based loaders. `load_args_from_checkpoint`
  also reads the selected load argument, `use_megatron_fsdp`, and dynamic field names.
- `global_vars.py`: inference telemetry still reads `data_parallel_size` and
  `seq_length`. Training telemetry already uses config owners.
- `initialize.py`: `write_args_to_tensorboard` enumerates CLI metadata only when
  args exists; changing logging formats is separate from ownership migration.
- CLI adapters/validation (`arguments.py`, `argument_utils.py`, `yaml_arguments.py`,
  `determinism.py`, and the DeepSeek V4 argument normalizer) intentionally read args.
- `utils/common_utils.py`: `is_gtp_remat_active(args)` has a misleading parameter
  name, but its training and conversion callers pass a `TransformerConfig`, not
  the legacy namespace.

`freeze_all_layers` is already owned by `cfg.train`; its runtime reads no longer
use args. Saved checkpoint argument readers remain necessary for backward
compatibility and are not part of this live-argument cleanup.
