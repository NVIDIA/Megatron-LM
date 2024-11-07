#!/bin/bash

set -x
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 -m pytest --color=yes -m "not flaky and not internal and not failing_on_rocm_mi250 and not failing_on_rocm" --csv output/test_report.csv tests/unit_tests/