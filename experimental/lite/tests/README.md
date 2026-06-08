# Megatron Lite Validation

`experimental/lite/tests` separates MLite validation into two layers:

- Unit tests: CPU/single-process contract tests for primitive, model, runtime, checkpoint sentinels, config, and helper behavior. Pure helper tests stub optional imports when no Transformer Engine runtime path is exercised; tests that need the real package explicitly skip when unavailable.
- Smoke tests: real `torch.distributed` tests for TP/EP/PP/CP/FSDP2/offload/checkpoint/distopt and tiny Qwen lite forward/backward behavior. Smoke runs are capped at one node and at most 8 GPUs.

Run unit coverage:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" pytest experimental/lite/tests/unit
```

Run smoke coverage on one node:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" MLITE_RUN_SMOKE=1 MLITE_SMOKE_NPROC=8 \
  experimental/lite/tests/run_primitive_validation.sh
```

The smoke suite is skipped by default in regular `pytest` runs. Enable it with `--mlite-smoke` or `MLITE_RUN_SMOKE=1`.

Current matrix:

| Surface | Unit | Smoke |
| --- | --- | --- |
| TP/EP/PP/CP/SP topology | `unit/primitive/test_parallel_unit.py`, `unit/primitive/test_parallel_dimensions_independent_unit.py` | `smoke/primitive/test_parallel_topologies_smoke.py` |
| TP linear/vocab primitives | `unit/primitive/test_parallel_dimensions_independent_unit.py` | Qwen model smoke exercises TP linear surfaces |
| EP token dispatch | `unit/primitive/test_parallel_dimensions_independent_unit.py` | Qwen model smoke exercises router, dispatcher, and experts |
| THD packing helpers | `unit/primitive/test_parallel_unit.py` | CP topology smoke exercises distributed CP groups |
| GQA/attention split contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises attention forward/backward |
| MoE router/aux-loss contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises router, dispatcher, and experts |
| LoRA adapter primitives | `unit/primitive/test_module_primitives_independent_unit.py` | Qwen model smoke can enable adapters in follow-up coverage |
| MTP/MRoPE/Gated Delta helper contracts | `unit/primitive/test_module_primitives_independent_unit.py`, `unit/primitive/test_ops_data_trainstep_unit.py` | Qwen3.5 MoE model smoke exercises MRoPE/Gated DeltaNet paths |
| Loss/logprob/math ops | `unit/primitive/test_ops_data_trainstep_unit.py` | Qwen model smoke exercises loss plumbing |
| Data/recompute/train-step primitives | `unit/primitive/test_ops_data_trainstep_unit.py` | model/runtime smoke exercises training loop integration |
| DDP + distributed optimizer | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | `smoke/primitive/test_distopt_checkpoint_smoke.py` |
| FSDP2 config/wrap/offload | `unit/primitive/test_fsdp2_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| FSDP2 save/load resume | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Checkpoint restore vs direct training | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | FSDP2 and distopt checkpoint smokes cover distributed restore paths |
| Runtime backend registry/config | `unit/primitive/test_runtime_config_unit.py`, `unit/runtime/test_runtime_backend_unit.py` | covered through checkpoint/model handles |
| Runtime env/offload controls | `unit/runtime/test_runtime_backend_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Optimizer update-state offload fraction | `unit/primitive/test_runtime_config_unit.py` and single-process CUDA coverage in `unit/primitive/test_fsdp2_offload_gpu.py` | multi-rank FSDP2 grad clipping is xfail until the follow-up bugfix PR |
| Qwen3 MoE lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |
| Qwen3.5 MoE lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |

Classic FSDP is not a separate MLite primitive in the current source tree; MLite's native sharded optimizer coverage is FSDP2 plus Megatron DDP/distopt.
