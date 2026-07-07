# Miles Application Skill

Use this when wiring Megatron Lite into radixark/miles examples.

- Keep the external miles CLI backend as `--train-backend megatron`; the MLite
  integration is activated by importing `miles_mlite.backend_patch` before miles
  constructs `RayTrainGroup`.
- Add `experimental/lite`, `experimental/lite/examples`, this repository root,
  and the miles checkout to `PYTHONPATH` for both the driver and Ray runtime
  environment.
- Use `--model-name qwen3_moe` and `--megatron-to-hf-mode raw` for Qwen3 MoE
  rollout resync. MLite `export_weights` already emits HF-format tensors.
- Use user-facing optimizer naming `dist_opt` in example CLI text.
- GPU validation must go through Slurm/container. Dry-run output is not evidence
  of a passed SFT or GRPO path.
