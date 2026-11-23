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
recipes should not call `finalize()` themselves. An optional `dataset_provider`
uses MLM's existing dataset callback interface with a loader-only `cfg.data`.
Custom model providers still use the lower-level training entrypoint.

## Llama 3 example

Use the generic recipe runner from the repository root:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
    --nproc-per-node=4 scripts/run_recipe.py \
    --recipe megatron.bridge.recipes.llama:llama3_8b_pretrain_config \
    model.transformer.context_parallel_size=1 \
    model.transformer.tensor_model_parallel_size=2 \
    model.transformer.max_seq_length_per_sample=512 \
    data.dataset.sequence_length=512 \
    train.global_batch_size=4 train.train_iters=20 \
    scheduler.lr_warmup_iters=1 scheduler.lr_decay_iters=20 \
    validation.eval_iters=1 validation.eval_interval=5 \
    data.dataloader.num_workers=0 logger.log_interval=1
```

The recipe is a trusted, importable zero-argument Python factory returning an
MLM `PretrainConfigContainer`. Trailing `KEY=VALUE` overrides edit existing
public dataclass fields (numbers, booleans, `null`, lists, or strings); repeated
keys use the last value. Unknown fields are rejected. Define optional config
sections, dataset classes, and Python callables in the recipe itself.

The runner selects the GPT/Hybrid or supported VLM forward step from the model config.
Use `--forward-step module:function` for an explicit MLM-compatible step.
`--dry-run` validates and displays the unfinalized config without setting global
config, initializing distributed services, building datasets, or training.
It is not a substitute for runtime validation against the actual GPU topology.

Dataset selection stays in `cfg.data`. The runner neither synchronizes sequence
lengths nor finalizes the config: both remain under the existing framework
ownership rules. It calls `megatron.training.api.pretrain`, which delegates to
the existing training loop.

The independently expressed settings follow the
[Bridge Llama 3 8B BF16 recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/eea53e431a8b0b4e9c4f47a76784c6f3200c9a1f/src/megatron/bridge/recipes/llama/h100/llama3.py).
It uses mock data and a NullTokenizer, not pretrained weights or the original
Llama tokenizer. Unlike the reference's 32K mock vocabulary, this recipe retains
Llama 3's 128256-token architecture vocabulary. This is scratch training, not a
model-quality benchmark. Replace the dataset and tokenizer together for real data.

Import the factory into a Python launch script and set explicit run overrides:

```python
from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.training.api import pretrain
from pretrain_gpt import forward_step

cfg = llama3_8b_pretrain_config()
cfg.model.transformer.context_parallel_size = 1
cfg.model.transformer.tensor_model_parallel_size = 2
cfg.model.transformer.max_seq_length_per_sample = 512
cfg.data.dataset.sequence_length = 512
cfg.train.global_batch_size = 4
cfg.train.train_iters = 20
cfg.scheduler.lr_warmup_iters = 1
cfg.scheduler.lr_decay_iters = 20
cfg.validation.eval_iters = 1
cfg.validation.eval_interval = 5
cfg.data.dataloader.num_workers = 0
cfg.logger.log_interval = 1
pretrain(cfg, forward_step)
```

Run that script from the repository root in an MLM training environment:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
    --nproc-per-node=4 your_recipe.py
```

The command uses TP=2 and DP=2. Select topology and batch size for your hardware.
Reduce model dimensions explicitly for a small smoke test. Dataset length and model execution length remain separate config
owners; this fixed-length pretraining example sets both to the same value.

For checkpointing, set `cfg.checkpoint.save` and `save_interval`. For resumption,
also set `cfg.checkpoint.load`, keeping the intended scheduler horizon unchanged
across launches. Config-only save/resume currently supports `torch_dist`. Saved
configuration and runtime topology replace optional CLI metadata; checkpoints
from native runs do not contain fabricated `args`. Older tools that require
checkpoint `args` are not yet supported. Config-only MoE upcycling is also deferred.

## Vision-language inputs

For Qwen3.5 VL and Nemotron3.5 Super VL model configs, the runner selects
`megatron.bridge.training.vlm.mock_vlm_datasets_provider` and its forward step.
Use a loader-only `cfg.data` for this callback. Each sample contains one synthetic
image followed by random text, with shifted causal labels and no loss on image
placeholders. Vision, projection, and
language parameters are trainable. This tests image-conditioned training, not
real-data quality, conversation processing, video, or SFT packing.

The supplied VL path currently requires PP=CP=1 and unpacked inputs. TP/EP/DP are
supported. For a real dataset, supply `--dataset-provider module:function` and
an MLM-compatible `--forward-step` as needed. Both are callbacks to the same
training loop. These restrictions apply to the supplied mock VLM path, not to
custom pairs of callbacks. Dataset providers must honor MLM's distributed-rank
and sampler interfaces and supply matching media metadata and shifted labels.

Report steady-state throughput separately from compilation and startup;
MLM's current Hybrid FLOPs estimate covers the
language model, not the vision encoder/projector, so a VL TFLOPS result using
that estimate is a language-only estimate over the whole VL iteration time.

Offline cached-logit distillation remains on its CLI path until
[#9635](https://github.com/NVIDIA/Megatron-LM/pull/9635) supplies
`cfg.offline_distillation`. This entrypoint does not add a competing cached-logit
config or fabricate args to enable that feature. Ordinary pretraining needs no
legacy argument namespace.

This example uses the dataset/config implementation already on `main`. It does
not introduce data-schema changes. Phase boundaries are owned by
`cfg.train.phase_transition_iterations`, and optimizer mapping diagnostics by
`cfg.optimizer.dump_param_to_param_group_map`. Other optional CLI-only features,
including full-dataset training and upcycling, are not migrated here.
