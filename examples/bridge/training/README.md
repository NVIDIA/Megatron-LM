# Native MLM recipe pretraining

`llama3_pretrain.py` constructs an MLM `PretrainConfigContainer` and calls
`megatron.training.api.pretrain(config, forward_step_func, callbacks=None)`.
It does not import Megatron Bridge, adapt Bridge configs, parse training CLI
arguments, or create a legacy argument namespace. The forward step and callbacks
use MLM interfaces, not Bridge's `GlobalState` interface.

The wrapper supplies MLM's config-driven dataset provider and GPT/Hybrid model
builder to the existing training loop. Framework startup finalizes the config;
recipes should not call `finalize()` themselves. Custom callback-provided models
and datasets can continue to use the lower-level training entrypoint.

## Llama 3 example

Use the generic recipe runner from the repository root:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run \
    --nproc-per-node=2 scripts/run_recipe.py \
    --recipe examples.bridge.training.llama3_pretrain:llama3_pretrain_config \
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

The runner selects the standard GPT/Hybrid forward step from the model config.
Use `--forward-step module:function` for an explicit MLM-compatible step.
`--dry-run` validates and displays the unfinalized config without setting global
config, initializing distributed services, building datasets, or training.
It is not a substitute for runtime validation against the actual GPU topology.

Dataset selection stays in `cfg.data`. The runner neither synchronizes sequence
lengths nor finalizes the config: both remain under the existing framework
ownership rules. It calls the current `megatron.training.api.pretrain` wrapper;
neither that wrapper nor the underlying training API is changed by the runner.

The independently expressed settings follow the
[Bridge Llama 3 8B BF16 recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/eea53e431a8b0b4e9c4f47a76784c6f3200c9a1f/src/megatron/bridge/recipes/llama/h100/llama3.py).
Like that scratch-training recipe, this uses mock data and a 32K NullTokenizer,
not pretrained weights or the original Llama tokenizer. It is a training-path
example, not a model-quality benchmark.

Import the factory into a Python launch script and set explicit run overrides:

```python
from examples.bridge.training.llama3_pretrain import llama3_pretrain_config
from megatron.training.api import pretrain
from pretrain_gpt import forward_step

cfg = llama3_pretrain_config()
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
    --nproc-per-node=2 your_recipe.py
```

This short configuration fits two 80GB GPUs using TP=2. The factory's original
CP=2 topology replicates model weights and may need more memory; select topology
and batch size for your hardware. Reduce model dimensions explicitly for a small
smoke test. Dataset length and model execution length remain separate config
owners; this fixed-length pretraining example sets both to the same value.

For checkpointing, set `cfg.checkpoint.save` and `save_interval`. For resumption,
also set `cfg.checkpoint.load`, keeping the intended scheduler horizon unchanged
across launches. Config-only save/resume currently supports `torch_dist`. Saved
configuration and runtime topology replace optional CLI metadata; checkpoints
from native runs do not contain fabricated `args`. Older tools that require
checkpoint `args` are not yet supported. Config-only MoE upcycling is also deferred.

Offline cached-logit distillation remains on its CLI path until
[#9635](https://github.com/NVIDIA/Megatron-LM/pull/9635) supplies
`cfg.offline_distillation`. This entrypoint does not add a competing cached-logit
config or fabricate args to enable that feature. Ordinary pretraining needs no
legacy argument namespace.

This example uses the dataset/config implementation already on `main`. It does
not introduce data-schema changes or migrate optional CLI-only features such as
full-dataset training, dataset phase transitions, frozen-teacher mode, or upcycling.
