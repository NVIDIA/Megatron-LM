#!/bin/bash
# V7: checkpoint reshard round-trip. TP=1 ckpt -> save as TP=2 -> load at TP=1 ->
# reproduce TP=1 oracle logits (lossless dist-ckpt => expect BITWISE). 2 GPUs.
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=400G
#SBATCH --time=00:40:00
#SBATCH --job-name=IMPL-gemma4-tpsp_v7_reshard
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/logs/%j_%x.out

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION
IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
SCR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/scripts
IMPL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm
OUTDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/implementations/IMPL-gemma4-tpsp
SRC=$IMPL/mlm_ckpt
TP2=$OUTDIR/mlm_ckpt_tp2reshard

srun --container-image=$IMAGE \
     --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
     --no-container-mount-home \
     bash -c "
set -x
cd $SCR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1

echo '########## V7a: load TP=1 src, re-save at TP=2 ##########'
torchrun --nproc_per_node=2 --master_port=12451 $SCR/tp_reshard.py --mode save --tp 2 --src $SRC --save-to $TP2
echo EXIT_SAVE=\$?

echo '########## V7b: load the TP=2 reshard back at TP=1, verify vs oracle ##########'
torchrun --nproc_per_node=1 --master_port=12452 $SCR/tp_reshard.py --mode verify --tp 1 --src $TP2 --ref-logits $OUTDIR/logits_tp1.pt
echo EXIT_VERIFY=\$?
echo '########## DONE ##########'
"
