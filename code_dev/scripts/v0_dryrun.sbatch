#!/bin/bash
# V0 baseline-green (TP=1 vs HF ref) + TP=2 init dry-run (no-SP and +SP smoke).
# 2 GPUs, single node. Dumps logits for cross-TP parity comparison.
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=400G
#SBATCH --time=00:40:00
#SBATCH --job-name=IMPL-gemma4-tpsp_v0_dryrun
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/logs/%j_%x.out

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION
IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
SCR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/scripts
OUTDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/implementations/IMPL-gemma4-tpsp

srun --container-image=$IMAGE \
     --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
     --no-container-mount-home \
     bash -c "
set -x
cd $SCR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1

echo '########## V0: TP=1 forward vs HF ref (baseline green) ##########'
torchrun --nproc_per_node=1 --master_port=12399 $SCR/tp_fwd.py --tp 1 --out $OUTDIR/logits_tp1.pt --capture-layers
echo EXIT_V0=\$?

echo '########## DRY-RUN: TP=2 (no SP) forward ##########'
torchrun --nproc_per_node=2 --master_port=12401 $SCR/tp_fwd.py --tp 2 --out $OUTDIR/logits_tp2.pt --capture-layers
echo EXIT_TP2=\$?

echo '########## SMOKE: TP=2 + SequenceParallel forward (SP currently unwired) ##########'
torchrun --nproc_per_node=2 --master_port=12403 $SCR/tp_fwd.py --tp 2 --sequence-parallel --out $OUTDIR/logits_tp2sp.pt --capture-layers
echo EXIT_TP2SP=\$?

echo '########## DONE ##########'
"
