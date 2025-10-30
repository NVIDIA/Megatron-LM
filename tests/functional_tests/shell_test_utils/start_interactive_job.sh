#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Required options:"
    echo "  --partition PARTITION    Slurm partition"
    echo "  --slurm-account ACCOUNT  Slurm account/PPP"
    echo "  --image IMAGE           Container image"
    echo "  --dataset-dir DIR       Dataset root directory"
    echo
    echo "Optional options:"
    echo "  --time TIME             Job time limit (default: 1:00:00)"
    echo
    echo "Example:"
    echo "  $0 --partition dgx --slurm-account nvidia --image nvcr.io/nvidia/pytorch:23.10-py3 --dataset-dir /datasets"
    echo "  $0 --partition dgx --slurm-account nvidia --image nvcr.io/nvidia/pytorch:23.10-py3 --dataset-dir /datasets --time 2:00:00"
}

# Initialize variables
PARTITION=""
SLURM_ACCOUNT=""
IMAGE=""
DATASET_DIR=""
TIME="1:00:00"
RECIPES_DIR="tests/test_utils/recipes"
CONTAINER_MOUNTS=""
NO_GPUS_PER_TASK="FALSE"

# Declare associative array for tracking unique mounts
declare -A seen_mounts

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --partition)
        PARTITION="$2"
        shift 2
        ;;
    --slurm-account)
        SLURM_ACCOUNT="$2"
        shift 2
        ;;
    --image)
        IMAGE="$2"
        shift 2
        ;;
    --dataset-dir)
        DATASET_DIR="$2"
        shift 2
        ;;
    --time)
        TIME="$2"
        shift 2
        ;;
    --no-gpus-per-task)
        NO_GPUS_PER_TASK="TRUE"
        shift 1
        ;;
    --help)
        print_usage
        exit 0
        ;;
    *)
        echo "Error: Unknown option '$1'"
        print_usage
        exit 1
        ;;
    esac
done

# Check if yq is installed
if ! command -v yq &>/dev/null; then
    echo "Error: yq is not installed. Please install it first."
    exit 1
fi

# Validate required arguments
if [ -z "$PARTITION" ] || [ -z "$SLURM_ACCOUNT" ] || [ -z "$IMAGE" ] || [ -z "$DATASET_DIR" ]; then
    echo "Error: Missing required arguments"
    print_usage
    exit 1
fi

# Add current directory to container mounts
CONTAINER_MOUNTS="$DATASET_DIR:/mnt/artifacts,$(pwd):/opt/megatron-lm"

# Build the final srun command
SRUN_CMD="srun \
    --partition=$PARTITION \
    --account=$SLURM_ACCOUNT \
    --container-image=$IMAGE \
    --container-workdir=/opt/megatron-lm \
    --container-mounts=$CONTAINER_MOUNTS \
    --nodes=1 \
    $(if [ "$NO_GPUS_PER_TASK" = "FALSE" ]; then echo "--gpus-per-task=8"; fi) \
    --time=$TIME \
    --pty bash"

printf "Generated srun command with all container mounts:\n\n"
echo "$SRUN_CMD"
echo
read -p "Execute this command? (y/n): " response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Executing command..."
    eval "$SRUN_CMD"
else
    echo "Command not executed."
fi
