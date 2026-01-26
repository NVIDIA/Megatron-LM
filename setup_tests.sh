export DATASET_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci
export TGT_IMAGE=gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main
export PARTITION=interactive
export ACCOUNT=llmservice_fm_text
source /lustre/fsw/portfolios/adlr/users/wdykas/code/mcore-tests-env/bin/activate
mkdir -p $HOME/.local/bin
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O $HOME/.local/bin/yq &&\
chmod +x $HOME/.local/bin/yq
export PATH=$HOME/.local/bin/:$PATH

bash tests/functional_tests/shell_test_utils/start_interactive_job.sh \
    --partition $PARTITION \
    --slurm-account $ACCOUNT \
    --image $TGT_IMAGE \
    --dataset-dir $DATASET_DIR



# srun -p interactive -A llmservice_fm_text -N 1 --pty \
#      --container-mounts "/lustre,/lustre/fsw/portfolios/llmservice/users/jbarker/workspace/mrl_internal/megatron-rl:/opt/megatron-lm,/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci:/mnt/artifacts" \
#      --container-image gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main \
#      --gpus 8 \
#      --exclusive \
#      --job-name "llmservice_nlp_fm-megatron-dev:interactive" \
#      -t 2:00:00 \
#      bash -l
