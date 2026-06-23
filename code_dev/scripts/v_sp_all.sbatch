#!/bin/bash
# V1-V5 @ TP=2/4/8 +SP vs TP=1 oracle, padded seq=16 (divisible by 8). One 8-GPU job.
#SBATCH --account=coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=1600G
#SBATCH --time=00:50:00
#SBATCH --job-name=IMPL-gemma4-tpsp_v_sp_all
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/logs/%j_%x.out

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SLURM_CPU_BIND SLURM_DISTRIBUTION
IMAGE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/nemo-25.11.nemotron_3_nano-mamba_ssm_2.3.0.sqsh
SCR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/scripts
OUTDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/Megatron-LM-tpsp/code_dev/shared-state/implementations/IMPL-gemma4-tpsp
PAD=16

srun --container-image=$IMAGE \
     --container-mounts=/lustre:/lustre,/home/ataghibakhsh:/home/ataghibakhsh \
     --no-container-mount-home \
     bash -c "
set -x
cd $SCR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1

echo '########## oracle: TP=1 (padded seq=$PAD) ##########'
torchrun --nproc_per_node=1 --master_port=12399 $SCR/tp_fwd.py --tp 1 --pad-seq-to $PAD --out $OUTDIR/logits_tp1_pad.pt --capture-layers
echo EXIT_TP1=\$?

for N in 2 4 8; do
  echo \"########## DUT: TP=\$N + SP (padded seq=$PAD) ##########\"
  torchrun --nproc_per_node=\$N --master_port=124\$N\$N $SCR/tp_fwd.py --tp \$N --sequence-parallel --pad-seq-to $PAD --out $OUTDIR/logits_tp\${N}sp_pad.pt --capture-layers
  echo EXIT_TP\${N}SP=\$?
  echo \"########## COMPARE TP=\$N+SP vs TP=1 ##########\"
  python3 $SCR/compare_logits.py --ref $OUTDIR/logits_tp1_pad.pt --dut $OUTDIR/logits_tp\${N}sp_pad.pt
done
echo '########## DONE ##########'
"
