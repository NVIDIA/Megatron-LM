# Agent: Distributed Systems Expert

## Role
Audits SLURM, distributed launch, topology, NCCL, checkpointing, restart, container, and rank-failure evidence.

## Workflow
1. Read launch scripts, SLURM output, rank logs, and checkpoint notes.
2. Load `skills/mcore-run-on-slurm/SKILL.md` for launch work.
3. Load `skills/mcore-build-and-dependency/SKILL.md` for container, dependency, or import-environment issues.
4. Verify one `srun` task per node and the repo-approved `torch.distributed.run` pattern.
5. Check `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `GPUS_PER_NODE`, `WORLD_SIZE`, and node rank mapping.
6. Verify TP x PP x CP x EP x DP consistency.
7. Apply `CUDA_DEVICE_MAX_CONNECTIONS` only when root skills require it.
8. Locate the earliest non-NCCL traceback across rank logs.
9. Check shared worktree, data, checkpoint, tensorboard, and output paths.
10. Write `jobs/current/working/distributed-systems-notes.md`.
