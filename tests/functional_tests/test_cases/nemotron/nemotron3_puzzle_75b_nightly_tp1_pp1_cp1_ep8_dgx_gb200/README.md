# Full-size Puzzle checkpoint-resume test

The importable [Puzzle architecture](../../../../../examples/hybrid/puzzle.py) defines
the published 75B-A9B model using layer-config objects, including all heterogeneous
MoE dimensions and one inferred MTP head. This test initializes that architecture
from scratch; it does not download or convert the Hugging Face checkpoint.

The [Nemotron GB200 recipe](../../../../test_utils/recipes/gb200/nemotron.yaml) selects
`nemotron3_puzzle_75b_nightly_tp1_pp1_cp1_ep8_dgx_gb200` in `dev`, nightly/L2, with
two nodes of four GB200 GPUs and five repetitions. It launches
`examples/hybrid/puzzle.py` with TP1/PP1/CP1/EP8/ETP1. Each repetition trains 20
steps, saves at step 10, and resumes that checkpoint through step 20 with optimizer
and RNG restoration. The harness checks the resumed losses against the first run.

## Target-hardware acceptance

Full-size memory fit and numerical acceptance require an actual eight-GB200 run.
Do not substitute a scaled model or copy another model's golden values. The
expected baseline file is `golden_values_dev_dgx_gb200.json`; it must be generated
from Puzzle's target-hardware TensorBoard results and committed after review.

Until that file exists, the first baseline comparison fails even if both training
phases finish. The harness extracts first-run results into the output directory
before comparison, but compares goldens before checking resumed losses. After
obtaining the baseline, rerun the test to exercise both checks. A recipe alone is
not evidence that the full-size model fits or that checkpoint resume has passed.
