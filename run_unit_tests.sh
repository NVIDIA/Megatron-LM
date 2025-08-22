#!/bin/bash

set -e
set -x

# Install mock for unit tests
pip install mock

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "Number of GPUs: $NUM_GPUS"

OUT_DIR=output
mkdir -p $OUT_DIR

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

# Find all test files recursively
TEST_FILES=$(find tests/unit_tests -type f -name "test_*.py")

for file in $TEST_FILES; do
    echo "Running test file: $file"
    torchrun --standalone --nproc_per_node=$NUM_GPUS -m pytest \
        --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" \
        --csv $OUT_DIR/test_report_$(basename $file .py).csv \
        $file --dist=loadscope

    if [[ $? -ne 0 ]]; then
        echo "Test failed in $file. Stopping execution."
        exit 1
    fi
done

echo "All test files passed successfully."

# Merge all individual CSVs into one unified report
python - <<EOF
import os
import pandas as pd

output_dir = "output"
csv_files = [f for f in os.listdir(output_dir) if f.startswith("test_report_") and f.endswith(".csv")]

dfs = []
for file in csv_files:
    path = os.path.join(output_dir, file)
    df = pd.read_csv(path)
    df["source_file"] = file  # Optional: track which file the results came from
    dfs.append(df)

if dfs:
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df.to_csv(os.path.join(output_dir, "unified_test_report.csv"), index=False)
    print("Unified test report saved to output/unified_test_report.csv")
else:
    print("No test report CSV files found to merge.")
EOF