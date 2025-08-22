#!/bin/bash

set -x
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and not experimental"

if [[ "$HIP_ARCHITECTURES" == "gfx90a" ]]; then
    PYTEST_MARKERS="$PYTEST_MARKERS and not failing_on_rocm_mi250"
fi

echo "=============================================================================="
echo "Starting main unit tests with markers: $PYTEST_MARKERS"
echo "=============================================================================="

torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_all.csv tests/unit_tests/ --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_a2a_overlap_report.csv tests/unit_tests/a2a_overlap --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_data_report.csv tests/unit_tests/data --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_dist_checkpointing_report.csv tests/unit_tests/dist_checkpointing --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_distributed_report.csv tests/unit_tests/distributed --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_export_report.csv tests/unit_tests/export --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_extensions_report.csv tests/unit_tests/extensions --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_fusions_report.csv tests/unit_tests/fusions --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_inference_report.csv tests/unit_tests/inference --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_models_report.csv tests/unit_tests/models --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_pipeline_parallel_report.csv tests/unit_tests/pipeline_parallel --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_post_training_report.csv tests/unit_tests/post_training --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_ssm_report.csv tests/unit_tests/ssm --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_tensor_parallel_report.csv tests/unit_tests/tensor_parallel --dist=loadscope
torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 --showlocals --tb=long -v -s -m "$PYTEST_MARKERS" --csv output/test_transformer_report.csv tests/unit_tests/transformer --dist=loadscope
echo "Main unit tests completed. Report saved to output/distributed_test_report.csv"

echo ""
echo "=============================================================================="
echo "Starting experimental unit tests."

PYTEST_MARKERS="(not flaky and not flaky_in_dev and not internal and not failing_on_rocm and not failing_on_upstream or test_on_rocm) and experimental"
echo "Using markers: $PYTEST_MARKERS"
echo "=============================================================================="

torchrun --master_port=29502 --nproc_per_node=8 -m pytest --maxfail=0 -v -s -m "$PYTEST_MARKERS" --csv output/experimental_test_report.csv tests/unit_tests/ --dist=loadscope --experimental

echo "Experimental unit tests completed. Report saved to output/experimental_test_report.csv"

echo ""
echo "All test runs finished."