#!/bin/bash
# V1/V2 @ TP=2+SP: TP=1 oracle dump + TP=2+SP DUT dump + parity compare. 2 GPUs.
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=400G
#SBATCH --time=00:40:00
#SBATCH --job-name=IMPL-gemma4-tpsp_v_sp2
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

echo '########## oracle: TP=1 ##########'
torchrun --nproc_per_node=1 --master_port=12399 $SCR/tp_fwd.py --tp 1 --out $OUTDIR/logits_tp1.pt --capture-layers
echo EXIT_TP1=\$?

echo '########## DUT: TP=2 + SequenceParallel ##########'
torchrun --nproc_per_node=2 --master_port=12401 $SCR/tp_fwd.py --tp 2 --sequence-parallel --out $OUTDIR/logits_tp2sp.pt --capture-layers
echo EXIT_TP2SP=\$?

echo '########## COMPARE TP=2+SP vs TP=1 ##########'
python3 $SCR/compare_logits.py --ref $OUTDIR/logits_tp1.pt --dut $OUTDIR/logits_tp2sp.pt
echo EXIT_CMP=\$?
echo '########## DONE ##########'
"
