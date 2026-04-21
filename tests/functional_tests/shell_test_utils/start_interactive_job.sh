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

# Check if recipes directory exists
if [ ! -d "$RECIPES_DIR" ]; then
    echo "Error: Recipes directory '$RECIPES_DIR' does not exist"
    exit 1
fi

# Create copy of recipes with interpolated artifacts
python -m tests.test_utils.python_scripts.common --recipes-dir $RECIPES_DIR --output-dir $RECIPES_DIR/interpolated

# Add current directory to container mounts
CONTAINER_MOUNTS="$(pwd):/opt/megatron-lm"

# Process each YAML file in the recipes directory
if [ ! -f "$YAML_FILE" ]; then
    continue
fi

echo "Processing $(basename "$YAML_FILE")..."
YAML_FILE=workflows.yaml
# Extract artifacts from YAML file
while IFS=: read -r value key; do
    # Skip empty or malformed entries
    if [ -z "$value" ] || [ -z "$key" ] || [ "$value" = "/data/" ] || [ "$key" = "/data/" ]; then
        continue
    fi

    # Skip entries that don't start with a forward slash
    if [[ ! "$key" =~ ^/ ]]; then
        continue
    fi

    # Create the mount string
    mount="${DATASET_DIR}/${value}:${key}"

    # Skip if we've seen this mount before
    if [ "${seen_mounts[$mount]}" = "1" ]; then
        echo "Skipping duplicate mount: $mount"
        continue
    fi

    # Mark this mount as seen
    seen_mounts[$mount]=1

    if [ -z "$CONTAINER_MOUNTS" ]; then
        CONTAINER_MOUNTS="$mount"
    else
        CONTAINER_MOUNTS="${CONTAINER_MOUNTS},$mount"
    fi
done < <(yq eval '.[].spec.artifacts | to_entries | .[] | "\(.value):\(.key)"' "$YAML_FILE")
rm $YAML_FILE

# Build the final srun command
SRUN_CMD="srun \
    --partition=$PARTITION \
    --account=$SLURM_ACCOUNT \
    --container-image=$IMAGE \
    --container-workdir=/opt/megatron-lm \
    --container-mounts=$CONTAINER_MOUNTS \
    --nodes=1 \
    --gpus-per-task=8 \
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
