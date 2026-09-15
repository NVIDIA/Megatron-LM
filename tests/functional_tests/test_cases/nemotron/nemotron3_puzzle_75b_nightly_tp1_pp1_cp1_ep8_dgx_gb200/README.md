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
Do not substitute a scaled model or copy another model's golden values.
`golden_values_dev_dgx_gb200.json` comes directly from the first 20-step training
phase of [GB200 job 439930823](https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/439930823),
at commit `964c8bdb7893f1fe85d92f05407af647054f2d64`. All six metrics retain the
TensorBoard collector's original values; iteration timing starts at step 2.

The baseline establishes reference metrics, not checkpoint-resume acceptance.
A clean five-repeat run must pass both golden and resumed-loss comparisons.
Allow a 9,000-second allocation for the full-size checkpoint I/O and all five
repetitions (`--functional-test-time-limit 9000` with the internal CI helper).
