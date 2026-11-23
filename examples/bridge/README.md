# Hugging Face model conversion

This directory provides Hugging Face import/export examples backed by
`megatron.bridge` inside Megatron-LM. It does not include a second training loop.

## Supported models and scope

The registered families are Nemotron-H, Nemotron 3.5 Super VL, and the Qwen 3.5
family's dense/MoE text and vision-language models, including compatible Qwen 3.8
checkpoints. Other Bridge model families are not registered by this package.
Super VL can also import/export its text-only projection with `--text-only`;
Qwen VL does not support that projection.

Nemotron text and Super VL implementations live together under
`megatron.bridge.models.nemotron_3`. The text conversion class is `Nemotron3Bridge`;
HF architecture/config identifiers retain their original names for compatibility.

Native I/O supports builder-backed `torch_dist` model weights and MLM
`run_config.yaml` metadata. It does not convert optimizer state, training progress,
legacy args/provider checkpoints, PEFT adapters, or ModelOpt checkpoint transforms.
Import/export is not a training-resume operation. Low-memory destructive saves are
not exposed by these examples.

The conversion package shares Bridge's Python namespace. **Do not install the
separate `megatron-bridge` distribution into the same environment.** Use a clean
environment with the MLM checkout or wheel as the package owner.

## Upstream reference

The initial port uses the per-file Bridge revisions recorded in the source headers.
The conversion fixes called out in review were cross-checked against Bridge main
[`b57951ad292e43209f1cc7d4b5500768365a20ac`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/b57951ad292e43209f1cc7d4b5500768365a20ac):
dense Nemotron config export (#6817), FP8 import scales (#6882), RoPE config export
(#6883), reused-tensor caching (#6837), grouped-expert CPU staging (#6948), mixed
wildcards (#6884), and later-pipeline-stage padding masks (#6881).
MLM already contains #9333, which replaces Bridge #6868's temporary CUDA-graph
RoPE workaround. This remains a selected-model port, not a copy of the whole
Bridge repository.

## Environment

Use MLM's GPU development environment with Transformer Engine and the SSM/Gated
DeltaNet dependencies required by these Hybrid models. Add conversion dependencies:

```bash
uv sync --extra training --extra dev --extra ssm --extra bridge --group test
```

The optional `bridge` extra declares Transformers, safetensors, Accelerate,
OmegaConf, PyYAML, Pydantic, einops and Rich. Transformers follows Bridge's
`>=5.21.1,<5.24` range, with 5.23.0 in the lockfile. The NeMo Run launcher additionally
uses `nemo-run`, included in MLM's test dependency group; direct worker invocation
does not require that launcher.

Use a pinned HF revision for remote inputs. `--trust-remote-code` explicitly
permits executing the checkpoint's custom Python code; use it only for trusted
sources. The supported Transformers range includes native Super VL configurations.

## GPU import/export

From the repository root, inside an existing GPU allocation:

```bash
PYTHONPATH=. uv run --no-sync python -m torch.distributed.run \
  --nproc-per-node=8 examples/bridge/conversion/run_conversion.py import --device gpu \
  --hf-model /models/super-vl --megatron-path /checkpoints/super-vl \
  --tp 2 --ep 8 --etp 1 --trust-remote-code

PYTHONPATH=. uv run --no-sync python -m torch.distributed.run \
  --nproc-per-node=8 examples/bridge/conversion/run_conversion.py export --device gpu \
  --hf-model /models/super-vl --megatron-path /checkpoints/super-vl \
  --hf-path /exports/super-vl --tp 2 --ep 8 --etp 1 --trust-remote-code \
  --export-weight-dtype bfloat16
```

GPU counts are illustrative; size the allocation for the actual model.
Use an empty output directory. Existing outputs are protected unless the
conversion CLI's explicit overwrite option is supplied.

`convert.sh` wraps the workers with local/Slurm NeMo Run execution.
Run `bash examples/bridge/conversion/convert.sh --help` for launcher options.
The CPU conversion path is retained from Bridge, and Gloo loading requests CPU
initialization. It is not yet fully CPU-only for Qwen3.5: MCore's Gated DeltaNet
constructs some parameters directly on CUDA even with CPU initialization enabled.
Use GPU conversion for these models; CPU-only conversion needs a separate Core fix.

Native loading preserves saved topology unless the caller supplies explicit
parallelism overrides. GTP checkpoints reject incompatible native topology
changes; export to HF first, then import with the desired topology.

## Python API

For a single-GPU HF → Megatron model → HF round-trip example:

```bash
PYTHONPATH=. uv run --no-sync python examples/bridge/conversion/hf_megatron_roundtrip.py \
  --hf-model-id /models/qwen-checkpoint --output-dir /exports
```

Choose a supported model that fits on one GPU. The example prints per-weight
comparisons (mismatches are displayed, not asserted) and saves to a new directory;
it does not save an intermediate native checkpoint or run training.

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("/models/checkpoint")
model_config = bridge.get_model_config()
# Edit model_config before construction.
models = bridge.get_model(model_config, load_weights=True, wrap_with_ddp=False)
```

These configs use MLM model builders and can be inspected and serialized without
constructing a model. Model construction uses only the config/builder APIs above;
the older model-provider classes and APIs are not included.
This port does not provide Bridge recipes, data pipelines,
PEFT training, or a unified end-to-end training entrypoint.

## Tests

The migrated Bridge unit tests live under `tests/unit_tests/bridge`,
inside MLM's existing models CI bucket:

```bash
uv run --no-sync python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m pytest -q tests/unit_tests/bridge
```

They cover config serialization and copying, dispatch, tensor mappings, Hybrid
builders, Nemotron-H and Qwen config/mapping conversion, and native checkpoint I/O.
Tests for unported model families, Bridge's training CLI, and PEFT are excluded.

## Source

Model/conversion code is adapted from Apache-2.0 Megatron Bridge at
[`53c20f81`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/53c20f81cb575f19af34ff0ae1749fb13b07999b),
Qwen Hybrid components at
[`0f062059`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/0f06205924d4e19b0723aa8c33e34a1959ca17fe),
and conversion launchers at
[`08035723`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/08035723e7a86f94fc54a7f4473bd0519f2ac941).
Copied files retain their original notices and per-file source links.
The shared HF wrappers retain some general-purpose interfaces, but only the
registered model families above are supported here.
